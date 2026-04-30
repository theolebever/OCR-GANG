#include "training.h"
#include "../common.h"
#include "../network/tools.h"
#include "../network/network.h"
#include "../network/cnn.h"
#include "augmentation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>


enum
{
    OCR_CLASS_COUNT = 52,
    TRAIN_SPLIT_NUMERATOR = 4,
    TRAIN_SPLIT_DENOMINATOR = 5,
    TRAIN_AUGMENT_MULTIPLIER = 50,
    MAX_EPOCHS = 200,
    EARLY_STOPPING_PATIENCE = 30,
    LR_DECAY_PERIOD = 50
};

static TrainingDataSet *allocate_dataset_shell(int capacity)
{
    TrainingDataSet *dataset = malloc(sizeof(TrainingDataSet));
    if (dataset == NULL)
        errx(1, "Failed to allocate dataset");

    dataset->inputs = malloc(sizeof(double *) * capacity);
    dataset->labels = malloc(sizeof(char) * capacity);
    dataset->count = 0;
    dataset->capacity = capacity;

    if (dataset->inputs == NULL || dataset->labels == NULL)
        errx(1, "Failed to allocate dataset samples");

    return dataset;
}

static void free_dataset_shell(TrainingDataSet *dataset)
{
    if (dataset == NULL) return;
    free(dataset->inputs);
    free(dataset->labels);
    free(dataset);
}

static void move_sample_reference(TrainingDataSet *dst, TrainingDataSet *src, int index)
{
    dst->inputs[dst->count] = src->inputs[index];
    dst->labels[dst->count] = src->labels[index];
    dst->count++;
}

static int class_train_target(int total)
{
    int target = (total * TRAIN_SPLIT_NUMERATOR) / TRAIN_SPLIT_DENOMINATOR;
    if (target <= 0 && total > 1)
        target = total - 1;
    return target;
}

static void split_dataset_stratified(TrainingDataSet *dataset,
                                     TrainingDataSet **train_out,
                                     TrainingDataSet **val_out)
{
    TrainingDataSet *train_set = allocate_dataset_shell(dataset->count);
    TrainingDataSet *val_set = allocate_dataset_shell(dataset->count);

    int class_totals[OCR_CLASS_COUNT] = {0};
    int class_seen[OCR_CLASS_COUNT] = {0};
    int *indices = malloc(sizeof(int) * dataset->count);
    if (indices == NULL)
        errx(1, "Failed to allocate split indices");

    for (int i = 0; i < dataset->count; i++)
    {
        int label = LabelIndex(dataset->labels[i]);
        if (label >= 0) class_totals[label]++;
        indices[i] = i;
    }

    shuffle(indices, dataset->count);
    for (int i = 0; i < dataset->count; i++)
    {
        int source_index = indices[i];
        int label = LabelIndex(dataset->labels[source_index]);
        if (label < 0)
        {
            move_sample_reference(train_set, dataset, source_index);
            continue;
        }

        if (class_seen[label] < class_train_target(class_totals[label]))
            move_sample_reference(train_set, dataset, source_index);
        else
            move_sample_reference(val_set, dataset, source_index);

        class_seen[label]++;
    }

    free(indices);
    *train_out = train_set;
    *val_out = val_set;
}

static void set_goal(struct network *net, int label_index)
{
    for (int g = 0; g < net->number_of_outputs; g++)
        net->goal[g] = 0.0;
    net->goal[label_index] = 1.0;
}

static int argmax_output(struct network *net)
{
    int max_out = 0;
    for (int o = 1; o < net->number_of_outputs; o++)
        if (net->output_layer[o] > net->output_layer[max_out])
            max_out = o;
    return max_out;
}

static int predict_label(CNN *cnn, struct network *net, double *input)
{
    cnn_forward(cnn, input, net->input_layer);
    forward_pass(net);
    return argmax_output(net);
}

static float validation_accuracy(CNN *cnn, struct network *net, TrainingDataSet *val_set)
{
    int correct = 0;
    set_training_mode(net, 0);

    for (int i = 0; i < val_set->count; i++)
    {
        int label_index = LabelIndex(val_set->labels[i]);
        if (label_index == -1) continue;
        if (predict_label(cnn, net, val_set->inputs[i]) == label_index)
            correct++;
    }

    set_training_mode(net, 1);
    return val_set->count > 0 ? (float)correct / val_set->count * 100.0f : 0.0f;
}

void TrainNetwork(void)
{
    printf("Loading Dataset...\n");
    TrainingDataSet *dataset = loadDataSet();

    if (dataset == NULL)
        errx(1, "Failed to load dataset!");

    TrainingDataSet *train_set = NULL;
    TrainingDataSet *val_set = NULL;
    split_dataset_stratified(dataset, &train_set, &val_set);

    printf("\n=== DATASET ANALYSIS ===\n");
    printf("Original samples: %d (Train: %d, Val: %d)\n",
           dataset->count, train_set->count, val_set->count);

    free_dataset_shell(dataset);

    // Augment ONLY the training set
    printf("Augmenting training set by %dx...\n", TRAIN_AUGMENT_MULTIPLIER);
    augment_dataset(train_set, TRAIN_AUGMENT_MULTIPLIER);
    printf("Augmentation complete. Training set size: %d\n", train_set->count);

    // Initialize CNN (load saved weights if they exist, fall back to fresh init on failure)
    printf("\nInitializing CNN (Conv 3x3 -> Pool 2x2)...\n");
    CNN *cnn = init_cnn();
    if (!cnn) errx(1, "Failed to init CNN");
    if (!fileempty(OCR_CNN_WEIGHTS))
    {
        if (load_cnn(OCR_CNN_WEIGHTS, cnn))
            printf("Loaded CNN weights from %s\n", OCR_CNN_WEIGHTS);
        else
            cnn_reset(cnn);
    }

    // Initialize MLP with CNN output size (1352 inputs)
    // 1352 = 8 filters * 13 * 13
    int hidden_nodes = OCR_HIDDEN_NODES;
    printf("\n=== NETWORK CONFIGURATION ===\n");
    printf("Architecture: CNN -> %d-%d-52\n", FLATTEN_SIZE, hidden_nodes);

    struct network *net = InitializeNetwork(FLATTEN_SIZE, hidden_nodes, 52, OCR_MLP_WEIGHTS);
    if (net == NULL) errx(1, "Failed to initialize network!");

    int epochs = MAX_EPOCHS;
    int *indices = malloc(sizeof(int) * train_set->count);
    for(int i = 0; i < train_set->count; i++) indices[i] = i;

    // Training Hyperparameters (Adam optimizer)
    net->eta = 0.001;  // Adam default learning rate

    printf("Learning rate: %.5f (Adam)\n", net->eta);

    float best_val_accuracy = -1.0f;
    int epochs_without_improvement = 0;

    printf("Starting Training...\n");
    printf("================================================================================\n");

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        shuffle(indices, train_set->count);
        int epoch_correct = 0;
        double total_error = 0.0;
        int trained_samples = 0;

        // Training phase
        for (int i = 0; i < train_set->count; i++)
        {
            int idx = indices[i];

            int label_index = LabelIndex(train_set->labels[idx]);
            if (label_index == -1) continue;

            cnn_forward(cnn, train_set->inputs[idx], net->input_layer);
            set_goal(net, label_index);
            forward_pass(net);

            // Cross-entropy loss on the correct class
            double p = net->output_layer[label_index];
            total_error += -my_log(p + 1e-12);
            trained_samples++;

            if (argmax_output(net) == label_index)
                epoch_correct++;

            back_propagation(net);
            cnn_backward(cnn, net->delta_input, net->eta * 0.1);
        }

        int denom = trained_samples > 0 ? trained_samples : 1;
        float train_accuracy = (float)epoch_correct / denom * 100.0f;
        double avg_loss = total_error / denom;

        float val_accuracy = validation_accuracy(cnn, net, val_set);

        printf("Epoch %3d/%d | Train: %6.2f%% | Val: %6.2f%% | Loss: %.5f",
               epoch + 1, epochs, train_accuracy, val_accuracy, avg_loss);

        if (val_accuracy > best_val_accuracy)
        {
            best_val_accuracy = val_accuracy;
            epochs_without_improvement = 0;
            printf(" * NEW BEST");
            save_network(OCR_MLP_WEIGHTS, net);
            save_cnn(OCR_CNN_WEIGHTS, cnn);
        }
        else
        {
            epochs_without_improvement++;
        }
        printf("\n");

        if ((epoch + 1) % LR_DECAY_PERIOD == 0 && net->eta > 1e-5) {
            net->eta *= 0.8;
            printf("    -> Learning rate adjusted to: %.6f\n", net->eta);
        }

        if (epochs_without_improvement >= EARLY_STOPPING_PATIENCE)
        {
            printf("\nEarly stopping.\n");
            break;
        }
    }

    printf("\nTraining complete. Best validation model kept on disk.\n");

    free(indices);
    freeDataSet(train_set);
    freeDataSet(val_set);
    freeNetwork(net);
    free_cnn(cnn);
}

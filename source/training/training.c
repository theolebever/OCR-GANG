#include "training.h"
#include "../network/tools.h"
#include "../network/network.h"
#include "../network/cnn.h"
#include "augmentation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>



void TrainNetwork(void)
{
    printf("Loading Dataset...\n");
    TrainingDataSet *dataset = loadDataSet();
    
    if (dataset == NULL)
        errx(1, "Failed to load dataset!");

    // Augment dataset (50x size)
    augment_dataset(dataset, 50);
    
    printf("\n=== DATASET ANALYSIS ===\n");
    printf("Total samples: %d\n", dataset->count);

    // Initialize CNN (load saved weights if they exist)
    printf("\nInitializing CNN (Conv 3x3 -> Pool 2x2)...\n");
    CNN *cnn = init_cnn();
    if (!cnn) errx(1, "Failed to init CNN");
    if (!fileempty("source/OCR-data/cnnwb.txt"))
    {
        load_cnn("source/OCR-data/cnnwb.txt", cnn);
        printf("Loaded CNN weights from source/OCR-data/cnnwb.txt\n");
    }

    // Initialize MLP with CNN output size (1352 inputs)
    // 1352 = 8 filters * 13 * 13
    int hidden_nodes = OCR_HIDDEN_NODES; 
    printf("\n=== NETWORK CONFIGURATION ===\n");
    printf("Architecture: CNN -> %d-%d-52\n", FLATTEN_SIZE, hidden_nodes);
    
    struct network *net = InitializeNetwork(FLATTEN_SIZE, hidden_nodes, 52, "source/OCR-data/ocrwb.txt");
    if (net == NULL) errx(1, "Failed to initialize network!");

    int epochs = 200;
    int *indices = malloc(sizeof(int) * dataset->count);
    for(int i = 0; i < dataset->count; i++) indices[i] = i;

    // Training Hyperparameters (Adam optimizer)
    net->eta = 0.001;  // Adam default learning rate

    printf("Learning rate: %.5f (Adam)\n", net->eta);

    float best_accuracy = 0.0f;
    int epochs_without_improvement = 0;
    
    printf("Starting Training...\n");
    printf("================================================================================\n");
    
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        shuffle(indices, dataset->count);
        int epoch_correct = 0;
        double total_error = 0.0;
        
        for (int i = 0; i < dataset->count; i++)
        {
            int idx = indices[i];
            
            // 1. CNN Forward — writes directly into MLP input layer, no malloc
            cnn_forward(cnn, dataset->inputs[idx], net->input_layer);
            
            // 3. Set Goal
            memset(net->goal, 0, sizeof(double) * net->number_of_outputs);
            int labelIndex = -1;
            char label = dataset->labels[idx];
             if (label >= 'A' && label <= 'Z') labelIndex = label - 'A';
             else if (label >= 'a' && label <= 'z') labelIndex = label - 'a' + 26;

             if (labelIndex != -1) net->goal[labelIndex] = 1.0;

            // 4. MLP Forward
            forward_pass(net);

            // Display loss: 1 - p_correct (goes to 0 as network improves)
            double p = net->output_layer[labelIndex > -1 ? labelIndex : 0];
            total_error += 1.0 - p;

            int max_out = 0;
            for (int o = 1; o < net->number_of_outputs; o++)
                if (net->output_layer[o] > net->output_layer[max_out]) max_out = o;
            if (max_out == labelIndex) epoch_correct++;

            // 5. MLP Backward
            back_propagation(net);

            // 6. CNN Backward — use a smaller LR than the MLP to keep gradients stable
            cnn_backward(cnn, net->delta_input, net->eta * 0.1);
        }
        
        float epoch_accuracy = (float)epoch_correct / dataset->count * 100.0f;
        double avg_loss = total_error / dataset->count; // avg (1 - p_correct), range [0,1]

        printf("Epoch %3d/%d | Accuracy: %6.2f%% | Loss: %.5f",
               epoch + 1, epochs, epoch_accuracy, avg_loss);
        
        if (epoch_accuracy > best_accuracy)
        {
            best_accuracy = epoch_accuracy;
            epochs_without_improvement = 0;
            printf(" * NEW BEST");
            save_network("source/OCR-data/ocrwb.txt", net);
            save_cnn("source/OCR-data/cnnwb.txt", cnn);
        }
        else
        {
            epochs_without_improvement++;
        }
        printf("\n");
        
         if ((epoch + 1) % 50 == 0 && net->eta > 0.001) {
            net->eta *= 0.8; 
            printf("    -> Learning rate adjusted to: %.6f\n", net->eta);
        }

        if (epochs_without_improvement >= 30) // Stricter checking
        {
            printf("\nEarly stopping.\n");
            break;
        }
    }

    printf("\nSaving final model...\n");
    save_network("source/OCR-data/ocrwb.txt", net);
    save_cnn("source/OCR-data/cnnwb.txt", cnn);
    
    free(indices);
    freeDataSet(dataset);
    freeNetwork(net);
    free_cnn(cnn);
}

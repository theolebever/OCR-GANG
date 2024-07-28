#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <stdbool.h>

#include "XOR.h"
#include "tools.h"

#define KGRN "\x1B[32m"
#define KRED "\x1B[31m"
#define KWHT "\x1B[37m"

#define INITIAL_ETA 0.1f
#define INITIAL_ALPHA 0.7f
#define BATCH_SIZE 32
#define LEARNING_RATE_DECAY 0.01
#define EARLY_STOPPING_WINDOW 10
#define EARLY_STOPPING_THRESHOLD 0.0001

static const char *NETWORK_FILE = "source/Xor/xorwb.txt";
static const char *RESULT_FILE = "source/Xor/xordata.txt";

static const int INPUT_NEURONS = 2;
static const int HIDDEN_NEURONS = 2;
static const int OUTPUT_NEURONS = 1;
static const int TRAINING_SETS = 4;

static const double TRAINING_INPUTS[] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
static const double TRAINING_OUTPUTS[] = {0.0, 1.0, 1.0, 0.0};

void run_xor_demo()
{
    struct fnn *network = init_xor(NETWORK_FILE);

    char answer[10];
    printf("Do you want to train the neural network or use it?\n1 = Train it\n2 = Use it\n");
    if (fgets(answer, sizeof(answer), stdin) == NULL)
    {
        fprintf(stderr, "Error reading input\n");
        exit(1);
    }

    switch (answer[0])
    {
    case '1':
        train_network(network);
        break;
    case '2':
        use_network(network);
        break;
    default:
        printf("%sInvalid choice!%s\n", KRED, KWHT);
    }

    free_network(network);
}

void train_network(struct fnn *network)
{
    FILE *result_file = fopen(RESULT_FILE, "w");
    if (result_file == NULL)
    {
        fprintf(stderr, "Error opening result file\n");
        exit(1);
    }

    int training_order[TRAINING_SETS];
    for (int i = 0; i < TRAINING_SETS; i++)
    {
        training_order[i] = i;
    }

    double errors[EARLY_STOPPING_WINDOW] = {0};
    int error_index = 0;

    printf("Started training...\n");
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++)
    {
        double epoch_error = 0.0;
        shuffle(training_order, TRAINING_SETS);

        for (int i = 0; i < TRAINING_SETS; i++)
        {
            int index = training_order[i];
            network->input_layer[0] = TRAINING_INPUTS[2 * index];
            network->input_layer[1] = TRAINING_INPUTS[2 * index + 1];
            network->goal[0] = TRAINING_OUTPUTS[index];

            forward_pass_xor(network);
            back_propagation(network);
            update_weights_and_biases(network);

            epoch_error += pow(network->goal[0] - network->output_layer[0], 2);
            save_training_results(network, result_file, index);
        }

        adaptive_learning_rate(network);
        epoch_error /= TRAINING_SETS;
        errors[error_index] = epoch_error;
        error_index = (error_index + 1) % EARLY_STOPPING_WINDOW;

        if (early_stopping(errors, EARLY_STOPPING_WINDOW, EARLY_STOPPING_THRESHOLD))
        {
            printf("Early stopping at epoch %d\n", epoch);
            break;
        }

        if (epoch % 100 == 0)
        {
            printf("Epoch %d, Error: %f\n", epoch, epoch_error);
        }
        fprintf(result_file, "\n");
    }

    fclose(result_file);
    save_network(NETWORK_FILE, network);
    printf("%sTraining completed!%s\n", KGRN, KWHT);
}

void use_network(struct fnn *network)
{
    printf("Using the trained network...\n");
    for (int i = 0; i < INPUT_NEURONS; i++)
    {
        printf("Please input number %d (0 or 1): ", i + 1);
        if (scanf("%lf", &network->input_layer[i]) != 1)
        {
            fprintf(stderr, "Invalid input\n");
            exit(1);
        }
        while (getchar() != '\n')
            ; // Clear input buffer
    }

    forward_pass_xor(network);
    printf("The neural network output: %f\n", network->output_layer[0]);
    printf("Interpreted result: %s\n", network->output_layer[0] > 0.5 ? "1" : "0");
}

void save_training_results(struct fnn *network, FILE *result_file, int index)
{
    fprintf(result_file, "input: %f ^ %f => output = %f, expected: %f\n",
            network->input_layer[0], network->input_layer[1],
            network->output_layer[0], TRAINING_OUTPUTS[index]);
}

// Standard matrix multiplication
void matrix_multiply(double *A, double *B, double *C, int m, int n, int k)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i * n + j] = 0;
            for (int p = 0; p < k; p++)
            {
                C[i * n + j] += A[i * k + p] * B[p * n + j];
            }
        }
    }
}

// Adaptive learning rate
void adaptive_learning_rate(struct fnn *net)
{
    static int epoch = 0;
    net->eta = INITIAL_ETA / (1 + LEARNING_RATE_DECAY * epoch);
    epoch++;
}

// Early stopping
bool early_stopping(double *errors, int window_size, double threshold)
{
    if (window_size < 2)
        return false;

    double sum = 0;
    for (int i = 1; i < window_size; i++)
    {
        sum += fabs(errors[i] - errors[i - 1]);
    }

    return (sum / (window_size - 1)) < threshold;
}

// Modified forward pass using ReLU
void forward_pass_xor(struct fnn *net)
{
    // Hidden layer
    matrix_multiply(net->input_layer, net->hidden_weights, net->hidden_layer,
                    1, net->number_of_hidden_nodes, net->number_of_inputs);
    for (size_t j = 0; j < net->number_of_hidden_nodes; j++)
    {
        net->hidden_layer[j] = relu(net->hidden_layer[j] + net->hidden_layer_bias[j]);
    }

    // Output layer
    matrix_multiply(net->hidden_layer, net->output_weights, net->output_layer,
                    1, net->number_of_outputs, net->number_of_hidden_nodes);
    for (size_t j = 0; j < net->number_of_outputs; j++)
    {
        net->output_layer[j] = relu(net->output_layer[j] + net->output_layer_bias[j]);
    }
}

// Modified back propagation using ReLU derivative
void back_propagation(struct fnn *net)
{
    for (size_t o = 0; o < net->number_of_outputs; o++)
    {
        net->delta_output[o] = (net->goal[o] - net->output_layer[o]) * drelu(net->output_layer[o]);
    }
    for (size_t h = 0; h < net->number_of_hidden_nodes; h++)
    {
        double sum = 0.0;
        for (size_t o = 0; o < net->number_of_outputs; o++)
        {
            sum += net->output_weights[h * net->number_of_outputs + o] * net->delta_output[o];
        }
        net->delta_hidden[h] = sum * drelu(net->hidden_layer[h]);
    }
}

struct fnn *init_xor(const char *filepath)
{
    struct fnn *network = malloc(sizeof(struct fnn));
    if (network == NULL)
    {
        errx(1, "Not enough memory!");
    }
    network->number_of_inputs = INPUT_NEURONS;
    network->number_of_hidden_nodes = HIDDEN_NEURONS;
    network->number_of_outputs = OUTPUT_NEURONS;
    network->input_layer = calloc(network->number_of_inputs, sizeof(double));

    network->hidden_layer =
        calloc(network->number_of_hidden_nodes, sizeof(double));
    network->delta_hidden =
        calloc(network->number_of_hidden_nodes, sizeof(double));
    network->hidden_layer_bias =
        calloc(network->number_of_hidden_nodes, sizeof(double));
    network->hidden_weights =
        calloc(network->number_of_inputs * network->number_of_hidden_nodes,
               sizeof(double));
    network->delta_hidden_weights =
        calloc(network->number_of_inputs * network->number_of_hidden_nodes,
               sizeof(double));

    network->output_layer = calloc(network->number_of_outputs, sizeof(double));
    network->delta_output = calloc(network->number_of_outputs, sizeof(double));
    network->output_layer_bias =
        calloc(network->number_of_outputs, sizeof(double));
    network->output_weights =
        calloc(network->number_of_hidden_nodes * network->number_of_outputs,
               sizeof(double));
    network->delta_output_weights =
        calloc(network->number_of_hidden_nodes * network->number_of_outputs,
               sizeof(double));

    network->goal = calloc(network->number_of_outputs, sizeof(double));
    network->eta = INITIAL_ETA;
    network->alpha = INITIAL_ALPHA;

    if (!file_empty(filepath))
    {
        load_network(filepath, network);
    }
    else
    {
        initialization(network);
    }
    return network;
}

void initialization(struct fnn *net)
{
    for (size_t i = 0; i < net->number_of_inputs; i++)
    {
        for (size_t j = 0; j < net->number_of_hidden_nodes; j++)
        {
            net->hidden_layer_bias[j] = init_weight();
            net->hidden_weights[i * net->number_of_hidden_nodes + j] =
                init_weight();
        }
    }
    for (size_t k = 0; k < net->number_of_hidden_nodes; k++)
    {
        for (size_t l = 0; l < net->number_of_outputs; l++)
        {
            net->output_layer_bias[l] = init_weight();
            net->output_weights[k * net->number_of_outputs + l] = init_weight();
        }
    }
}

void update_weights_and_biases(struct fnn *net)
{
    // Weights and biases between Input and Hidden layers
    for (size_t i = 0; i < net->number_of_inputs; i++)
    {
        for (size_t j = 0; j < net->number_of_hidden_nodes; j++)
        {
            net->hidden_weights[i * net->number_of_hidden_nodes + j] +=
                net->eta * net->input_layer[i] * net->delta_hidden[j];
            net->hidden_layer_bias[j] += net->eta * net->delta_hidden[j];
        }
    }

    // Weights between Hidden and Ouput layers
    for (size_t o = 0; o < net->number_of_outputs; o++)
    {
        for (size_t h = 0; h < net->number_of_hidden_nodes; h++)
        {
            net->output_weights[h * net->number_of_outputs + o] +=
                net->eta * net->delta_output[o] * net->hidden_layer[h] + net->alpha * net->delta_output_weights[h * net->number_of_outputs + o];

            net->delta_output_weights[h * net->number_of_outputs + o] =
                net->eta * net->delta_output[o] * net->hidden_layer[h];
        }
        net->output_layer_bias[o] += net->eta * net->delta_output[o];
    }
}

void free_network(struct fnn *net)
{
    free(net->input_layer);
    free(net->hidden_layer);
    free(net->delta_hidden);
    free(net->hidden_layer_bias);
    free(net->hidden_weights);
    free(net->delta_hidden_weights);
    free(net->output_layer);
    free(net->delta_output);
    free(net->output_layer_bias);
    free(net->output_weights);
    free(net->delta_output_weights);
    free(net->goal);
    free(net);
}
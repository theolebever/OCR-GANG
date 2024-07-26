// File for Sigmoid functions in particular and other functions needed for
// operating the neural network Used for tweaking the weigth of each node in the
// neural network
#include "network.h"
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <stdbool.h>
#include "tools.h"

#define INITIAL_ETA 0.1f
#define INITIAL_ALPHA 0.7f
#define BATCH_SIZE 32
#define LEARNING_RATE_DECAY 0.01
#define EARLY_STOPPING_WINDOW 10
#define EARLY_STOPPING_THRESHOLD 0.0001


// Standard matrix multiplication
void matrix_multiply(double* A, double* B, double* C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int p = 0; p < k; p++) {
                C[i * n + j] += A[i * k + p] * B[p * n + j];
            }
        }
    }
}

// ReLU activation function
double relu(double x) {
    return x > 0 ? x : 0;
}

double drelu(double x) {
    return x > 0 ? 1 : 0;
}

// Adaptive learning rate
void adaptive_learning_rate(struct network *net) {
    static int epoch = 0;
    net->eta = INITIAL_ETA / (1 + LEARNING_RATE_DECAY * epoch);
    epoch++;
}

// Early stopping
bool early_stopping(double* errors, int window_size, double threshold) {
    if (window_size < 2) return false;
    
    double sum = 0;
    for (int i = 1; i < window_size; i++) {
        sum += fabs(errors[i] - errors[i-1]);
    }
    
    return (sum / (window_size - 1)) < threshold;
}

// Modified forward pass using ReLU
void forward_pass(struct network *net) {
    // Hidden layer
    matrix_multiply(net->input_layer, net->hidden_weights, net->hidden_layer, 
                    1, net->number_of_hidden_nodes, net->number_of_inputs);
    for (size_t j = 0; j < net->number_of_hidden_nodes; j++) {
        net->hidden_layer[j] = relu(net->hidden_layer[j] + net->hidden_layer_bias[j]);
    }

    // Output layer
    matrix_multiply(net->hidden_layer, net->output_weights, net->output_layer, 
                    1, net->number_of_outputs, net->number_of_hidden_nodes);
    for (size_t j = 0; j < net->number_of_outputs; j++) {
        net->output_layer[j] = relu(net->output_layer[j] + net->output_layer_bias[j]);
    }
}

// Modified back propagation using ReLU derivative
void back_propagation(struct network *net) {
    for (size_t o = 0; o < net->number_of_outputs; o++) {
        net->delta_output[o] = (net->goal[o] - net->output_layer[o]) * drelu(net->output_layer[o]);
    }
    for (size_t h = 0; h < net->number_of_hidden_nodes; h++) {
        double sum = 0.0;
        for (size_t o = 0; o < net->number_of_outputs; o++) {
            sum += net->output_weights[h * net->number_of_outputs + o] * net->delta_output[o];
        }
        net->delta_hidden[h] = sum * drelu(net->hidden_layer[h]);
    }
}

// Mini-batch training
void mini_batch_training(struct network *net, int ***chars_matrix, int total_samples) {
    double errors[EARLY_STOPPING_WINDOW] = {0};
    int error_index = 0;

    for (int epoch = 0; epoch < 1000; epoch++) {  // Arbitrary number of epochs
        double epoch_error = 0;

        for (int i = 0; i < total_samples; i += BATCH_SIZE) {
            // Process batch
            for (int j = 0; j < BATCH_SIZE && (i + j) < total_samples; j++) {
                input_image(net, i + j, chars_matrix);
                forward_pass(net);
                back_propagation(net);
                
                // Calculate error
                for (size_t o = 0; o < net->number_of_outputs; o++) {
                    epoch_error += pow(net->goal[o] - net->output_layer[o], 2);
                }
            }
            
            // Update weights after processing the batch
            update_weights_and_biases(net);
        }
        
        // Adaptive learning rate
        adaptive_learning_rate(net);
        
        // Early stopping
        errors[error_index] = epoch_error / total_samples;
        error_index = (error_index + 1) % EARLY_STOPPING_WINDOW;
        
        if (early_stopping(errors, EARLY_STOPPING_WINDOW, EARLY_STOPPING_THRESHOLD)) {
            printf("Early stopping at epoch %d\n", epoch);
            break;
        }
    }
}

struct network *initialize_network(double i, double h, double o, const char *filepath)
{
    struct network *network = malloc(sizeof(struct network));
    if (network == NULL)
    {
        errx(1, "Not enough memory!");
    }
    network->number_of_inputs = i;
    network->number_of_hidden_nodes = h;
    network->number_of_outputs = o;
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

    if (!fileempty(filepath))
    {
        load_network(filepath, network);
    }
    else
    {
        initialization(network);
    }
    return network;
}

void initialization(struct network *net)
{
    for (int i = 0; i < net->number_of_inputs; i++)
    {
        for (int j = 0; j < net->number_of_hidden_nodes; j++)
        {
            net->hidden_layer_bias[j] = init_weight();
            net->hidden_weights[i * net->number_of_hidden_nodes + j] =
                init_weight();
        }
    }
    for (int k = 0; k < net->number_of_hidden_nodes; k++)
    {
        for (int l = 0; l < net->number_of_outputs; l++)
        {
            net->output_layer_bias[l] = init_weight();
            net->output_weights[k * net->number_of_outputs + l] = init_weight();
        }
    }
}

void update_weights_and_biases(struct network *net)
{
    // Weights and biases between Input and Hidden layers
    for (int i = 0; i < net->number_of_inputs; i++)
    {
        for (int j = 0; j < net->number_of_hidden_nodes; j++)
        {
            net->hidden_weights[i * net->number_of_hidden_nodes + j] +=
                net->eta * net->input_layer[i] * net->delta_hidden[j];
            net->hidden_layer_bias[j] += net->eta * net->delta_hidden[j];
        }
    }

    // Weights between Hidden and Ouput layers
    for (int o = 0; o < net->number_of_outputs; o++)
    {
        for (int h = 0; h < net->number_of_hidden_nodes; h++)
        {
            net->output_weights[h * net->number_of_outputs + o] +=
                net->eta * net->delta_output[o] * net->hidden_layer[h]
                + net->alpha
                    * net->delta_output_weights[h * net->number_of_outputs + o];

            net->delta_output_weights[h * net->number_of_outputs + o] =
                net->eta * net->delta_output[o] * net->hidden_layer[h];
        }
        net->output_layer_bias[o] += net->eta * net->delta_output[o];
    }
}


void free_network(struct network *net)
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

int input_image(struct network *net, size_t index, int ***chars_matrix)
{
    int is_espace = 1;
    for (size_t i = 0; i < 784; i++)
    {
        net->input_layer[i] = (*chars_matrix)[index][i];
        if (net->input_layer[i] == 1)
        {
            is_espace = 0;
        }
    }
    return is_espace;
}

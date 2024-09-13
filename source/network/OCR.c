#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "OCR.h"
#include "fc.h"
#include "conv.h"
#include "pool.h"
#include "tools.h"
#include "early_stop.h"

Network *create_ocr_network()
{
    Network *net = create_network(6); // 6 layers total

    // Assuming 28x28 binary input
    add_conv_layer(net, 0, INPUT_HEIGHT, INPUT_WIDTH, 1, 3, 3, 32);
    add_pool_layer(net, 1, 18, 18, 32, 2, 2, 2);
    add_conv_layer(net, 2, 9, 9, 32, 3, 3, 64);
    add_pool_layer(net, 3, 7, 7, 64, 2, 2, 2);
    add_fc_layer(net, 4, 3 * 3 * 64, 128, true);
    add_fc_layer(net, 5, 128, 52, false);

    return net;
}

// Function to create a new network
// Function to create a new network
Network *create_network(int num_layers)
{
    Network *net = (Network *)malloc(sizeof(Network));
    if (!net)
    {
        perror("Failed to allocate memory for Network");
        exit(EXIT_FAILURE);
    }
    net->num_layers = num_layers;

    net->layers = (Layer **)malloc(num_layers * sizeof(Layer *));
    if (!net->layers)
    {
        perror("Failed to allocate memory for layers");
        free(net);
        exit(EXIT_FAILURE);
    }

    net->optimizers = (AdamOptimizer **)malloc(num_layers * sizeof(AdamOptimizer *));
    if (!net->optimizers)
    {
        perror("Failed to allocate memory for optimizers");
        free(net->layers);
        free(net);
        exit(EXIT_FAILURE);
    }

    // Initialize optimizers and layers to NULL
    for (int i = 0; i < num_layers; i++)
    {
        net->layers[i] = NULL;
        net->optimizers[i] = NULL;
    }

    return net;
}

void free_network_cnn(Network *net)
{
    if (net == NULL)
    {
        return;
    }

    // Free layer-specific resources
    for (int i = 0; i < net->num_layers; i++)
    {
        if (net->layers[i] == NULL)
        {
            continue;
        }

        Layer *layer = net->layers[i];

        switch (layer->type)
        {
        case LAYER_CONV:
        {
            ConvLayer *conv = (ConvLayer *)layer;
            free(conv->weights);
            free(conv->biases);
            free(conv->weight_gradients);
            free(conv->bias_gradients);
            break;
        }
        case LAYER_POOL:
        {
            PoolLayer *pool = (PoolLayer *)layer;
            free(pool->max_indices);
            break;
        }
        case LAYER_FC:
        {
            FCLayer *fc = (FCLayer *)layer;
            free(fc->weights);
            free(fc->biases);
            free(fc->weight_gradients);
            free(fc->bias_gradients);
            break;
        }
        default:
            fprintf(stderr, "Unknown layer type in free_network_cnn\n");
            break;
        }

        // Free input and output volumes if not already freed
        if (layer->input)
        {
            if (layer->input->data)
            {
                free(layer->input->data);
            }
            free(layer->input);
            layer->input = NULL;
        }
        if (layer->output)
        {
            if (layer->output->data)
            {
                free(layer->output->data);
            }
            free(layer->output);
            layer->output = NULL;
        }

        free(layer);
        net->layers[i] = NULL;
    }

    // Free optimizers
    for (int i = 0; i < net->num_layers; i++)
    {
        if (net->optimizers[i])
        {
            free_adam(net->optimizers[i]);
            net->optimizers[i] = NULL;
        }
    }
    free(net->optimizers);
    net->optimizers = NULL;

    // Free layers array
    free(net->layers);
    net->layers = NULL;

    // Free the network structure
    free(net);
}

void softmax(float *input, int size)
{
    float max = input[0];
    for (int i = 1; i < size; i++)
    {
        if (input[i] > max)
        {
            max = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }

    for (int i = 0; i < size; i++)
    {
        input[i] /= sum;
    }
}

void forward_pass_ocr(Network *net, int *input_data)
{
    // Copy input data to the first layer's input
    Volume *first_layer_input = net->layers[0]->input;
    int input_size = first_layer_input->width * first_layer_input->height * first_layer_input->depth;

    for (int i = 0; i < input_size; i++)
    {
        first_layer_input->data[i] = (float)input_data[i];
    }

    for (int i = 0; i < net->num_layers; i++)
    {
        Layer *layer = net->layers[i];
        Volume *layer_input = (i == 0) ? first_layer_input : net->layers[i - 1]->output;

        switch (layer->type)
        {
        case LAYER_CONV:
            conv_forward((ConvLayer *)layer, layer_input);
            break;
        case LAYER_POOL:
            pool_forward((PoolLayer *)layer, layer_input);
            break;
        case LAYER_FC:
            fc_forward((FCLayer *)layer, layer_input);
            break;
        default:
            break;
        }
    }
}

void backward_pass(Network *net, float *target)
{
    // Determine the maximum size needed across all layers
    size_t max_size = 0;
    for (int i = 0; i < net->num_layers; i++)
    {
        size_t layer_size = net->layers[i]->output->width *
                            net->layers[i]->output->height *
                            net->layers[i]->output->depth;
        if (layer_size > max_size)
        {
            max_size = layer_size;
        }
    }

    // Allocate error with the maximum size
    float *error = malloc(max_size * sizeof(float));
    if (!error)
    {
        fprintf(stderr, "Failed to allocate memory for error in backward_pass\n");
        exit(EXIT_FAILURE);
    }

    // Initialize error for the output layer
    FCLayer *output_layer = (FCLayer *)net->layers[net->num_layers - 1];
    int output_size = output_layer->output_size;
    float *output = output_layer->base.output->data;
    for (int i = 0; i < output_size; i++)
    {
        error[i] = output[i] - target[i];
    }

    // Backward pass through each layer
    for (int i = net->num_layers - 1; i >= 0; i--)
    {
        Layer *layer = net->layers[i];
        switch (layer->type)
        {
        case LAYER_CONV:
            conv_backward(layer, error);
            break;
        case LAYER_POOL:
            pool_backward(layer, error);
            break;
        case LAYER_FC:
            fc_backward(layer, error);
            break;
        default:
            fprintf(stderr, "Unknown layer type %d in backward pass\n", layer->type);
            free(error);
            exit(EXIT_FAILURE);
        }

        // Update error for next layer
        if (i > 0)
        {
            size_t input_size = net->layers[i]->input->width *
                                net->layers[i]->input->height *
                                net->layers[i]->input->depth;
            memcpy(error, net->layers[i]->input->data, input_size * sizeof(float));
        }
    }

    free(error);
}

void update_parameters(Network *net, float learning_rate)
{
    for (int i = 0; i < net->num_layers; i++)
    {
        switch (net->layers[i]->type)
        {
        case LAYER_CONV:
        {
            ConvLayer *layer = (ConvLayer *)net->layers[i];
            int weight_size = layer->filter_width * layer->filter_height * layer->base.input->depth * layer->num_filters;
            // Update weights
            adam_update(net->optimizers[i], layer->weights, layer->weight_gradients, weight_size, learning_rate);

            // Update biases
            adam_update(net->optimizers[i], layer->biases, layer->bias_gradients, layer->num_filters, learning_rate);

            // Reset gradients
            memset(layer->weight_gradients, 0, weight_size * sizeof(float));
            memset(layer->bias_gradients, 0, layer->num_filters * sizeof(float));
            break;
        }
        case LAYER_FC:
        {
            FCLayer *layer = (FCLayer *)net->layers[i];
            int weight_size = layer->input_size * layer->output_size;
            // Update weights
            adam_update(net->optimizers[i], layer->weights, layer->weight_gradients, weight_size, learning_rate);

            // Update biases
            adam_update(net->optimizers[i], layer->biases, layer->bias_gradients, layer->output_size, learning_rate);

            // Reset gradients
            memset(layer->weight_gradients, 0, weight_size * sizeof(float));
            memset(layer->bias_gradients, 0, layer->output_size * sizeof(float));
            break;
        }
        case LAYER_POOL:
            // Pooling layers have no parameters to update
            break;
        default:
            fprintf(stderr, "Unknown layer type in parameter update\n");
            exit(1);
        }
    }
}

float calculate_loss(Network *net, float *target)
{
    FCLayer *output_layer = (FCLayer *)net->layers[net->num_layers - 1];
    float loss = 0.0f;
    float epsilon = 1e-10f; // Small value to avoid log(0)

    // Apply softmax to the output
    softmax(output_layer->base.output->data, output_layer->output_size);

    // Calculate cross-entropy loss
    for (int i = 0; i < output_layer->output_size; i++)
    {
        float y = target[i];
        float y_pred = output_layer->base.output->data[i];
        loss -= y * logf(y_pred + epsilon);
    }

    return loss;
}

void train(Network *net, int ****training_matrix, int num_samples_per_char, int epochs, float initial_lr)
{
    float *target = (float *)calloc(52, sizeof(float)); // 52 output classes (A-Z, a-z)
    if (!target)
    {
        perror("Memory allocation failed for target");
        exit(EXIT_FAILURE);
    }

    char expected_result[52] = {'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E',
                                'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                                'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N',
                                'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                                'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W',
                                'w', 'X', 'x', 'Y', 'y', 'Z', 'z'};

    // Initialize early stopping
    EarlyStopping *es = init_early_stopping(net, 10); // patience of 10 epochs
    if (!es)
    {
        perror("Failed to initialize early stopping");
        free(target);
        exit(EXIT_FAILURE);
    }

    float learning_rate = initial_lr;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float total_loss = 0;
        int total_samples = 0;

        shuffle_char(expected_result, 52);

        for (size_t i = 0; i < 52; i++)
        {
            // Determine the target index for the character
            char letter = expected_result[i];
            int target_index = (letter >= 'a') ? (letter - 'a' + 26) : (letter - 'A');

            // Reset all targets to 0.0
            memset(target, 0, 52 * sizeof(float));
            target[target_index] = 1.0f; // Set the target for current character

            for (int index = 0; index < num_samples_per_char; index++)
            {
                forward_pass_ocr(net, training_matrix[i][index][0]);
                float loss = calculate_loss(net, target);
                total_loss += loss;
                total_samples++;

                // Backward pass and parameter update
                backward_pass(net, target);
                update_parameters(net, learning_rate);
            }
        }
        // Calculate average loss for this epoch
        float avg_loss = total_loss / total_samples;

        // // Check for early stopping
        // if (should_stop(es, avg_loss, net, epoch))
        // {
        //     printf("Early stopping at epoch %d\n", epoch);
        //     break;
        // }

        if (epoch % 10 == 0)
        {
            printf("Epoch %d, Average Loss: %f\n", epoch, avg_loss);
        }
    }

    // Restore best parameters
    restore_best_params(net, es);

    // Free resources
    free_early_stopping(es);
    free(target);
}

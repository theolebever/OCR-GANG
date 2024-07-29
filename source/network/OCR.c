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

#define DEBUG 0

Network *create_ocr_network()
{
    Network *net = create_network(6); // 6 layers total

    // Assuming 28x28 binary input
    add_conv_layer(net, 0, INPUT_HEIGHT, INPUT_WIDTH, 1, 3, 3, 32);
    add_pool_layer(net, 1, 18, 18, 32, 2, 2, 2);
    add_conv_layer(net, 2, 9, 9, 32, 3, 3, 64);
    add_pool_layer(net, 3, 7, 7, 64, 2, 2, 2);
    add_fc_layer(net, 4, 3 * 3 * 64, 128);
    add_fc_layer(net, 5, 128, 52);

    return net;
}

// Function to create a new network
Network *create_network(int num_layers)
{
    Network *net = (Network *)malloc(sizeof(Network));
    net->num_layers = num_layers;
    net->layers = (Layer **)malloc(num_layers * sizeof(Layer *));
    net->optimizers = (AdamOptimizer **)malloc(num_layers * sizeof(AdamOptimizer *));
    return net;
}

void free_network_cnn(Network *net)
{
    if (net == NULL)
    {
        return;
    }

    // First, free all layer-specific resources
    for (int i = 0; i < net->num_layers; i++)
    {
        if (net->layers[i] == NULL)
        {
            continue;
        }

        switch (net->layers[i]->type)
        {
        case LAYER_CONV:
        {
            ConvLayer *conv = (ConvLayer *)net->layers[i];
            free(conv->weights);
            free(conv->biases);
            free(conv->weight_gradients);
            free(conv->bias_gradients);
            free_adam(net->optimizers[i]);
            break;
        }
        case LAYER_POOL:
        {
            PoolLayer *pool = (PoolLayer *)net->layers[i];
            free(pool->max_indices);
            break;
        }
        case LAYER_FC:
        {
            FCLayer *fc = (FCLayer *)net->layers[i];
            free(fc->weights);
            free(fc->biases);
            free(fc->weight_gradients);
            free(fc->bias_gradients);
            free_adam(net->optimizers[i]);
            break;
        }
        default:
            break;
        }
    }

    // Now, free input, output volumes and optimizers
    for (int i = 0; i < net->num_layers; i++)
    {
        if (net->layers[i]->input && (i == 0 || net->layers[i]->input != net->layers[i - 1]->output))
        {
            free(net->layers[i]->input->data);
            free(net->layers[i]->input);
        }
        if (net->layers[i]->output)
        {
            free(net->layers[i]->output->data);
            free(net->layers[i]->output);
        }
    }

    // Finally, free the layers themselves and the network structure
    for (int i = 0; i < net->num_layers; i++)
    {
        free(net->layers[i]);
    }
    free(net->optimizers);
    free(net->layers);
    free(net);
}

void forward_pass_ocr(Network *net, float *input)
{
    // Assuming the first layer is always convolutional or FC
    memcpy(net->layers[0]->input->data, input, net->layers[0]->input->width * net->layers[0]->input->height * net->layers[0]->input->depth * sizeof(float));

    for (int i = 0; i < net->num_layers; i++)
    {
        switch (net->layers[i]->type)
        {
        case LAYER_CONV:
            net->layers[i]->forward(net->layers[i], net->layers[i]->input);
            break;
        case LAYER_POOL:
            pool_forward((PoolLayer *)net->layers[i], net->layers[i]->input);
            break;
        case LAYER_FC:
            net->layers[i]->forward(net->layers[i], net->layers[i]->input);
            break;
        default:
            fprintf(stderr, "Unknown layer type\n");
            exit(EXIT_FAILURE);
        }

        // Set input of next layer to output of current layer, if not last layer
        if (i < net->num_layers - 1)
        {
            net->layers[i + 1]->input = net->layers[i]->output;
        }
    }
}

void backward_pass(Network *net, float *target)
{
    // Determine the maximum size needed across all layers
    size_t max_size = 0;
    for (int i = 0; i < net->num_layers; i++)
    {
        size_t layer_size = net->layers[i]->input->width *
                            net->layers[i]->input->height *
                            net->layers[i]->input->depth;
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
        return;
    }

    // Initialize error for the output layer
    int output_size = ((FCLayer *)net->layers[net->num_layers - 1])->output_size;
    float *output = net->layers[net->num_layers - 1]->output->data;
    for (int i = 0; i < output_size; i++)
    {
        error[i] = output[i] - target[i];
    }

    // Backward pass through each layer
    for (int i = net->num_layers - 1; i >= 0; i--)
    {
        switch (net->layers[i]->type)
        {
        case LAYER_CONV:
            net->layers[i]->backward(net->layers[i], error);
            break;
        case LAYER_POOL:
            pool_backward((PoolLayer *)net->layers[i], error);
            break;
        case LAYER_FC:
            net->layers[i]->backward(net->layers[i], error);
            break;
        default:
            fprintf(stderr, "Unknown layer type in backward pass\n");
            free(error);
            return;
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
            for (int j = 0; j < weight_size; j++)
            {
                layer->weights[j] -= learning_rate * layer->weight_gradients[j];
                layer->weight_gradients[j] = 0; // Reset gradients
            }
            for (int j = 0; j < layer->num_filters; j++)
            {
                layer->biases[j] -= learning_rate * layer->bias_gradients[j];
                layer->bias_gradients[j] = 0; // Reset gradients
            }

            // TODO: Fix Adam implementation
            // Update weights
            // adam_update(net->optimizers[i], layer->weights, layer->weight_gradients, weight_size, learning_rate);

            // // Update biases
            // adam_update(net->optimizers[i], layer->biases, layer->bias_gradients, layer->num_filters, learning_rate);

            // // Reset gradients
            // memset(layer->weight_gradients, 0, weight_size * sizeof(float));
            // memset(layer->bias_gradients, 0, layer->num_filters * sizeof(float));
            break;
        }
        case LAYER_FC:
        {
            FCLayer *layer = (FCLayer *)net->layers[i];
            int weight_size = layer->input_size * layer->output_size;
            for (int j = 0; j < weight_size; j++)
            {
                layer->weights[j] -= learning_rate * layer->weight_gradients[j];
                layer->weight_gradients[j] = 0; // Reset gradients
            }
            for (int j = 0; j < layer->output_size; j++)
            {
                layer->biases[j] -= learning_rate * layer->bias_gradients[j];
                layer->bias_gradients[j] = 0; // Reset gradients
            }

            // Update weights
            // adam_update(net->optimizers[i], layer->weights, layer->weight_gradients, weight_size, learning_rate);

            // // Update biases
            // adam_update(net->optimizers[i], layer->biases, layer->bias_gradients, layer->output_size, learning_rate);

            // // Reset gradients
            // memset(layer->weight_gradients, 0, weight_size * sizeof(float));
            // memset(layer->bias_gradients, 0, layer->output_size * sizeof(float));
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
    float loss = 0;
    float epsilon = 1e-10;

    for (int i = 0; i < output_layer->output_size; i++)
    {
        float y = target[i];
        float y_pred = output_layer->base.output->data[i];
        loss -= y * log(y_pred + epsilon);
    }

    return loss / output_layer->output_size;
}

void train(Network *net, const char *filematrix, char *expected_result, int num_samples_per_char, int epochs, float learning_rate)
{
    double *input = (double *)calloc(INPUT_HEIGHT * INPUT_WIDTH, sizeof(double));
    float *target = (float *)calloc(52, sizeof(float)); // 52 output classes (A-Z, a-z)

    // Initialize early stopping
    EarlyStopping *es = init_early_stopping(net, 10); // patience of 10 epochs

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float total_loss = 0;
        int total_samples = 0;

        for (size_t i = 0; i < 52; i++)
        {
            // Loop over all characters (A-Z, a-z)
            char letter = expected_result[i];
            int target_index = (letter >= 'a') ? (letter - 'a' + 26) : (letter - 'A');

            // Reset all targets to 0.0
            memset(target, 0, 52 * sizeof(float));
            target[target_index] = 1.0; // Set the target for current character

            for (int j = 0; j < num_samples_per_char; j++)
            {
                // Loop over samples for each character
                char *newpath = update_path(filematrix, strlen(filematrix), letter, j);
                read_binary_image(newpath, input);

                forward_pass_ocr(net, (float *)input);
                float loss = calculate_loss(net, target);
                total_loss += loss;
                total_samples++;

                backward_pass(net, target);
                update_parameters(net, learning_rate);

                // Optional debug display
                if (DEBUG)
                {
                    char result = retrieve_answer(net);
                    printf("Gave path : %s / Result expected was : %c / Result provided is : %c\n", newpath, letter, result);
                }

                free(newpath);
            }
        }

        // Calculate average loss for this epoch
        float avg_loss = total_loss / total_samples;

        // Check for early stopping
        // if (should_stop(es, avg_loss, net, epoch))
        // {
        //     break;
        // }

        if (epoch % 10 == 0)
        {
            printf("Epoch %d, Average Loss: %f\n", epoch, avg_loss);
        }
    }

    // Restore best parameters
    restore_best_params(net, es);

    printf("Best model found at epoch %d with validation loss %f\n", es->best_epoch, es->best_val_loss);

    free(input);
    free(target);
    free_early_stopping(es);
}

// Prediction Function
int predict(Network *net, float *input)
{
    forward_pass_ocr(net, input);

    // Assume the last layer is fully connected with outputs for each class
    FCLayer *output_layer = (FCLayer *)net->layers[net->num_layers - 1];
    float max_val = output_layer->base.output->data[0];
    int max_idx = 0;

    for (int i = 1; i < output_layer->output_size; i++)
    {
        if (output_layer->base.output->data[i] > max_val)
        {
            max_val = output_layer->base.output->data[i];
            max_idx = i;
        }
    }

    return max_idx;
}
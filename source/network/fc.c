#include "fc.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "tools.h"

void add_fc_layer(Network *net, int layer_index, int input_size, int output_size)
{
    FCLayer *fc = (FCLayer *)malloc(sizeof(FCLayer));
    fc->base.type = LAYER_FC;

    // Initialize input volume
    fc->base.input = (Volume *)malloc(sizeof(Volume));
    fc->base.input->width = 1;
    fc->base.input->height = 1;
    fc->base.input->depth = input_size;
    fc->base.input->data = (float *)calloc(input_size, sizeof(float));

    // Initialize output volume
    fc->base.output = (Volume *)malloc(sizeof(Volume));
    fc->base.output->width = 1;
    fc->base.output->height = 1;
    fc->base.output->depth = output_size;
    fc->base.output->data = (float *)calloc(output_size, sizeof(float));

    // Set layer properties
    fc->input_size = input_size;
    fc->output_size = output_size;

    // Allocate and initialize weights
    int weights_size = input_size * output_size;
    fc->weights = (float *)malloc(weights_size * sizeof(float));
    xavier_init(fc->weights, input_size, output_size);

    // Allocate and initialize biases
    fc->biases = (float *)calloc(output_size, sizeof(float));

    // Allocate memory for gradients
    fc->weight_gradients = (float *)calloc(weights_size, sizeof(float));
    fc->bias_gradients = (float *)calloc(output_size, sizeof(float));

    // Set forward and backward functions
    fc->base.forward = fc_forward;
    fc->base.backward = fc_backward;

    // Setup Adam optimize
    int param_count = input_size * output_size + output_size;
    net->optimizers[layer_index] = init_adam(param_count, 0.85, 0.995, 1e-7);

    // Add layer to network
    net->layers[layer_index] = (Layer *)fc;
}

void fc_forward(Layer *layer, Volume *input)
{
    FCLayer *fc = (FCLayer *)layer;
    for (int i = 0; i < fc->output_size; i++)
    {
        float sum = 0;
        for (int j = 0; j < fc->input_size; j++)
        {
            sum += input->data[j] * fc->weights[i * fc->input_size + j];
        }
        sum += fc->biases[i];
        layer->output->data[i] = relu(sum);
    }
}

void fc_backward(Layer *layer, float *upstream_gradient)
{
    FCLayer *fc = (FCLayer *)layer;

    // Ensure we have valid input and output
    if (layer->input == NULL || layer->output == NULL || upstream_gradient == NULL)
    {
        fprintf(stderr, "Error: NULL input, output, or upstream_gradient in fc_backward\n");
        return;
    }

    // Ensure input and output sizes are valid
    if (fc->input_size <= 0 || fc->output_size <= 0)
    {
        fprintf(stderr, "Error: Invalid input or output size in fc_backward\n");
        return;
    }

    // Compute gradients for weights and biases
    for (int i = 0; i < fc->output_size; i++)
    {
        float output = layer->output->data[i];
        float error_derivative = output * (1 - output) * upstream_gradient[i];

        for (int j = 0; j < fc->input_size; j++)
        {
            fc->weight_gradients[i * fc->input_size + j] += error_derivative * layer->input->data[j];
        }
        fc->bias_gradients[i] += error_derivative;
    }

    // Compute gradients for inputs (to be passed to previous layer)
    float *input_gradients = calloc(fc->input_size, sizeof(float));
    if (input_gradients == NULL)
    {
        fprintf(stderr, "Memory allocation failed in fc_backward\n");
        return;
    }

    for (int i = 0; i < fc->input_size; i++)
    {
        float sum = 0;
        for (int j = 0; j < fc->output_size; j++)
        {
            float output = layer->output->data[j];
            float error_derivative = output * (1 - output) * upstream_gradient[j];
            sum += error_derivative * fc->weights[j * fc->input_size + i];
        }
        input_gradients[i] = sum;
    }

    memcpy(layer->input->data, input_gradients, fc->input_size * sizeof(float));
    free(input_gradients);
}
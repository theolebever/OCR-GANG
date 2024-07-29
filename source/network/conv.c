#include "conv.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "tools.h"

// Function to add a convolutional layer
void add_conv_layer(Network *net, int layer_index, int in_w, int in_h, int in_d,
                    int filter_w, int filter_h, int num_filters)
{
    ConvLayer *conv = (ConvLayer *)malloc(sizeof(ConvLayer));
    conv->base.type = LAYER_CONV;

    // Initialize input volume
    conv->base.input = (Volume *)malloc(sizeof(Volume));
    conv->base.input->width = in_w;
    conv->base.input->height = in_h;
    conv->base.input->depth = in_d;
    conv->base.input->data = (float *)calloc(in_w * in_h * in_d, sizeof(float));
    if (!conv->base.input->data)
    {
        perror("Memory allocation failed for input volume");
        exit(EXIT_FAILURE);
    }

    // Initialize output volume
    int out_w = in_w - filter_w + 1;
    int out_h = in_h - filter_h + 1;
    conv->base.output = (Volume *)malloc(sizeof(Volume));
    conv->base.output->width = out_w;
    conv->base.output->height = out_h;
    conv->base.output->depth = num_filters;
    conv->base.output->data = (float *)calloc(out_w * out_h * num_filters, sizeof(float));
    if (!conv->base.output->data)
    {
        perror("Memory allocation failed for output volume");
        exit(EXIT_FAILURE);
    }

    // Set layer properties
    conv->filter_width = filter_w;
    conv->filter_height = filter_h;
    conv->num_filters = num_filters;

    // Allocate and initialize weights
    int weights_size = filter_w * filter_h * in_d * num_filters;
    conv->weights = (float *)malloc(weights_size * sizeof(float));
    if (!conv->weights)
    {
        perror("Memory allocation failed for weights");
        exit(EXIT_FAILURE);
    }
    xavier_init(conv->weights, filter_w * filter_h, in_d * num_filters);

    // Allocate and initialize biases
    conv->biases = (float *)calloc(num_filters, sizeof(float));
    if (!conv->biases)
    {
        perror("Memory allocation failed for biases");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for gradients
    conv->weight_gradients = (float *)calloc(weights_size, sizeof(float));
    if (!conv->weight_gradients)
    {
        perror("Memory allocation failed for weight_gradients");
        exit(EXIT_FAILURE);
    }
    conv->bias_gradients = (float *)calloc(num_filters, sizeof(float));
    if (!conv->bias_gradients)
    {
        perror("Memory allocation failed for bias_gradients");
        exit(EXIT_FAILURE);
    }

    // Set forward and backward functions
    conv->base.forward = conv_forward;
    conv->base.backward = conv_backward;

    // Setup Adam optimizer
    int param_count = filter_w * filter_h * in_d * num_filters + num_filters;
    net->optimizers[layer_index] = init_adam(param_count, 0.85, 0.995, 1e-7);

    // Add layer to network
    net->layers[layer_index] = (Layer *)conv;
}

void conv_forward(Layer *layer, Volume *input)
{
    ConvLayer *conv = (ConvLayer *)layer;
    int out_w = input->width - conv->filter_width + 1;
    int out_h = input->height - conv->filter_height + 1;

    for (int f = 0; f < conv->num_filters; f++)
    {
        for (int y = 0; y < out_h; y++)
        {
            for (int x = 0; x < out_w; x++)
            {
                float sum = 0;
                for (int fy = 0; fy < conv->filter_height; fy++)
                {
                    for (int fx = 0; fx < conv->filter_width; fx++)
                    {
                        for (int d = 0; d < input->depth; d++)
                        {
                            int ix = x + fx;
                            int iy = y + fy;
                            float input_val = input->data[(iy * input->width + ix) * input->depth + d];
                            float weight = conv->weights[(f * conv->filter_height * conv->filter_width + fy * conv->filter_width + fx) * input->depth + d];
                            sum += input_val * weight;
                        }
                    }
                }
                sum += conv->biases[f];
                layer->output->data[(y * out_w + x) * conv->num_filters + f] = (sum > 0) ? sum : 0; // ReLU activation
            }
        }
    }
}

void conv_backward(Layer *layer, float *upstream_gradient)
{
    ConvLayer *conv = (ConvLayer *)layer;
    int in_w = layer->input->width;
    int in_h = layer->input->height;
    int in_d = layer->input->depth;
    int out_w = layer->output->width;
    int out_h = layer->output->height;
    int f_w = conv->filter_width;
    int f_h = conv->filter_height;
    int num_f = conv->num_filters;

    // Initialize gradients
    float *input_gradients = calloc(in_w * in_h * in_d, sizeof(float));

    // Compute gradients
    for (int f = 0; f < num_f; f++)
    {
        for (int y = 0; y < out_h; y++)
        {
            for (int x = 0; x < out_w; x++)
            {
                int out_idx = (y * out_w + x) * num_f + f;
                float gradient = upstream_gradient[out_idx];

                if (layer->output->data[out_idx] > 0)
                { // ReLU derivative
                    for (int fy = 0; fy < f_h; fy++)
                    {
                        for (int fx = 0; fx < f_w; fx++)
                        {
                            for (int d = 0; d < in_d; d++)
                            {
                                int ix = x + fx;
                                int iy = y + fy;
                                int in_idx = (iy * in_w + ix) * in_d + d;
                                int w_idx = ((f * f_h + fy) * f_w + fx) * in_d + d;

                                input_gradients[in_idx] += gradient * conv->weights[w_idx];
                                conv->weight_gradients[w_idx] += gradient * layer->input->data[in_idx];
                            }
                        }
                    }
                    conv->bias_gradients[f] += gradient;
                }
            }
        }
    }

    memcpy(layer->input->data, input_gradients, in_w * in_h * in_d * sizeof(float));
    free(input_gradients);
}
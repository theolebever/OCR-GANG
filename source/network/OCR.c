#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "OCR.h"
#include "tools.h"

Network *create_ocr_network()
{
    Network *net = create_network(7); // 7 layers total

    // Assuming 28x28 binary input
    add_conv_layer(net, 0, INPUT_HEIGHT, INPUT_WIDTH, 1, 3, 3, 32); // 32 3x3 filters
    add_pool_layer(net, 1, 18, 18, 32, 2, 2, 2);                    // 2x2 max pooling
    add_conv_layer(net, 2, 9, 9, 32, 3, 3, 64);                     // 64 3x3 filters
    add_pool_layer(net, 3, 7, 7, 64, 2, 2, 2);                      // 2x2 max pooling
    add_fc_layer(net, 4, 3 * 3 * 64, 128);                          // Fully connected layer
    add_fc_layer(net, 5, 128, 26);                                  // Output layer (assuming 26 characters)

    return net;
}

// Function to create a new network
Network *create_network(int num_layers)
{
    Network *net = (Network *)malloc(sizeof(Network));
    net->num_layers = num_layers;
    net->layers = (Layer **)malloc(num_layers * sizeof(Layer *));
    return net;
}

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

    // Initialize output volume
    int out_w = in_w - filter_w + 1;
    int out_h = in_h - filter_h + 1;
    conv->base.output = (Volume *)malloc(sizeof(Volume));
    conv->base.output->width = out_w;
    conv->base.output->height = out_h;
    conv->base.output->depth = num_filters;
    conv->base.output->data = (float *)calloc(out_w * out_h * num_filters, sizeof(float));

    // Set layer properties
    conv->filter_width = filter_w;
    conv->filter_height = filter_h;
    conv->num_filters = num_filters;

    // Allocate and initialize weights
    int weights_size = filter_w * filter_h * in_d * num_filters;
    conv->weights = (float *)malloc(weights_size * sizeof(float));
    xavier_init(conv->weights, filter_w * filter_h * in_d, out_w * out_h);

    // Allocate and initialize biases
    conv->biases = (float *)calloc(num_filters, sizeof(float));

    // Allocate memory for gradients
    conv->weight_gradients = (float *)calloc(weights_size, sizeof(float));
    conv->bias_gradients = (float *)calloc(num_filters, sizeof(float));

    // Set forward and backward functions
    conv->base.forward = conv_forward;
    conv->base.backward = conv_backward;

    // Add layer to network
    net->layers[layer_index] = (Layer *)conv;
}

// Function to add a pooling layer
void add_pool_layer(Network *net, int layer_index, int in_w, int in_h, int in_d,
                    int pool_w, int pool_h, int stride)
{
    PoolLayer *pool = (PoolLayer *)malloc(sizeof(PoolLayer));
    pool->base.type = LAYER_POOL;
    pool->base.input = (Volume *)malloc(sizeof(Volume));
    pool->base.output = (Volume *)malloc(sizeof(Volume));
    pool->pool_width = pool_w;
    pool->pool_height = pool_h;
    pool->stride = stride;

    int output_w = (in_w - pool_w) / stride + 1;
    int output_h = (in_h - pool_h) / stride + 1;
    pool->max_indices = (int *)malloc(output_w * output_h * in_d * sizeof(int));

    net->layers[layer_index] = (Layer *)pool;
}

// Function to add a fully connected layer

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

    // Add layer to network
    net->layers[layer_index] = (Layer *)fc;
}

// Function to free network memory
void free_network_cnn(Network *net)
{
    for (int i = 0; i < net->num_layers; i++)
    {
        // Free layer-specific memory
        switch (net->layers[i]->type)
        {
        case LAYER_CONV:
            // Free ConvLayer specific memory
            break;
        case LAYER_POOL:
            // Free PoolLayer specific memory
            break;
        case LAYER_FC:
            // Free FCLayer specific memory
            break;
        default:
            break;
        }
        free(net->layers[i]->input);
        free(net->layers[i]->output);
        free(net->layers[i]);
    }
    free(net->layers);
    free(net);
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

void pool_forward(PoolLayer *layer, Volume *input)
{
    int out_w = (input->width - layer->pool_width) / layer->stride + 1;
    int out_h = (input->height - layer->pool_height) / layer->stride + 1;

    for (int d = 0; d < input->depth; d++)
    {
        for (int y = 0; y < out_h; y++)
        {
            for (int x = 0; x < out_w; x++)
            {
                float max_val = -INFINITY;
                int max_idx = -1;
                for (int py = 0; py < layer->pool_height; py++)
                {
                    for (int px = 0; px < layer->pool_width; px++)
                    {
                        int ix = x * layer->stride + px;
                        int iy = y * layer->stride + py;
                        float val = input->data[(iy * input->width + ix) * input->depth + d];
                        if (val > max_val)
                        {
                            max_val = val;
                            max_idx = (iy * input->width + ix) * input->depth + d;
                        }
                    }
                }
                layer->base.output->data[(y * out_w + x) * input->depth + d] = max_val;
                layer->max_indices[(y * out_w + x) * input->depth + d] = max_idx;
            }
        }
    }
}

void pool_backward(PoolLayer *layer, float *upstream_gradient)
{
    int out_w = layer->base.output->width;
    int out_h = layer->base.output->height;
    int in_w = layer->base.input->width;
    int in_h = layer->base.input->height;
    int depth = layer->base.input->depth;

    // Initialize input gradients to zero
    float *input_gradients = calloc(in_w * in_h * depth, sizeof(float));

    for (int d = 0; d < depth; d++)
    {
        for (int y = 0; y < out_h; y++)
        {
            for (int x = 0; x < out_w; x++)
            {
                int out_idx = (y * out_w + x) * depth + d;
                int max_idx = layer->max_indices[out_idx];
                input_gradients[max_idx] += upstream_gradient[out_idx];
            }
        }
    }

    // Store input gradients for the previous layer
    memcpy(layer->base.input->data, input_gradients, in_w * in_h * depth * sizeof(float));
    free(input_gradients);
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
        layer->output->data[i] = sigmoid(sum);
    }
}

void fc_backward(Layer *layer, float *upstream_gradient)
{
    FCLayer *fc = (FCLayer *)layer;

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
            exit(1);
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
    int output_size = ((FCLayer *)net->layers[net->num_layers - 1])->output_size;
    float *output = net->layers[net->num_layers - 1]->output->data;

    // Compute initial error (assuming cross-entropy loss for classification)
    float *error = malloc(output_size * sizeof(float));
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
            exit(1);
        }

        // Update error for next layer
        if (i > 0)
        {
            memcpy(error, net->layers[i]->input->data, net->layers[i]->input->width * net->layers[i]->input->height * net->layers[i]->input->depth * sizeof(float));
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

void train(Network *net, float **training_data, float **labels, int num_samples, int epochs, float learning_rate)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int i = 0; i < num_samples; i++)
        {
            forward_pass_ocr(net, training_data[i]);
            backward_pass(net, labels[i]);
            update_parameters(net, learning_rate);
        }
    }
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
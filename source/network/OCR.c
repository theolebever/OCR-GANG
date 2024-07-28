#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "OCR.h"
#include "tools.h"
#include "early_stop.h"

#define DEBUG 0

Network *create_ocr_network()
{
    Network *net = create_network(6); // 6 layers total

    // Assuming 28x28 binary input
    add_conv_layer(net, 0, INPUT_HEIGHT, INPUT_WIDTH, 1, 3, 3, 32); // 32 3x3 filters
    add_pool_layer(net, 1, 18, 18, 32, 2, 2, 2);                    // 2x2 max pooling
    add_conv_layer(net, 2, 9, 9, 32, 3, 3, 64);                     // 64 3x3 filters
    add_pool_layer(net, 3, 7, 7, 64, 2, 2, 2);                      // 2x2 max pooling
    add_fc_layer(net, 4, 3 * 3 * 64, 128);                          // Fully connected layer
    add_fc_layer(net, 5, 128, 52);                                  // Output layer (52 characters)

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

    pool->base.input->width = in_w;
    pool->base.input->height = in_h;
    pool->base.input->depth = in_d;
    pool->base.input->data = (float *)calloc(in_w * in_h * in_d, sizeof(float));

    int output_w = (in_w - pool_w) / stride + 1;
    int output_h = (in_h - pool_h) / stride + 1;
    pool->base.output->width = output_w;
    pool->base.output->height = output_h;
    pool->base.output->depth = in_d;
    pool->base.output->data = (float *)calloc(output_w * output_h * in_d, sizeof(float));

    pool->max_indices = (int *)malloc(output_w * output_h * in_d * sizeof(int));

    if (!pool->base.input->data || !pool->base.output->data || !pool->max_indices)
    {
        perror("Memory allocation failed in add_pool_layer");
        exit(EXIT_FAILURE);
    }

    pool->base.forward = (void (*)(Layer *, Volume *))pool_forward;
    pool->base.backward = (void (*)(Layer *, float *))pool_backward;

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

    // Setup Adam optimize
    int param_count = input_size * output_size + output_size;
    net->optimizers[layer_index] = init_adam(param_count, 0.85, 0.995, 1e-7);

    // Add layer to network
    net->layers[layer_index] = (Layer *)fc;
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
    int out_w = layer->base.output->width;
    int out_h = layer->base.output->height;

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
                        if (ix < input->width && iy < input->height)
                        { // Bounds check
                            float val = input->data[(iy * input->width + ix) * input->depth + d];
                            if (val > max_val)
                            {
                                max_val = val;
                                max_idx = (iy * input->width + ix) * input->depth + d;
                            }
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
            // // Update weights
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

            // // Update weights
            // adam_update(net->optimizers[i], layer->weights, layer->weight_gradients, weight_size, learning_rate);

            // // Update biases
            // adam_update(net->optimizers[i], layer->biases, layer->bias_gradients, layer->output_size, learning_rate);

            // // Reset gradients
            // memset(layer->weight_gradients, 0, weight_size * sizeof(float));
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
    // Using the Cross-Entropy Loss
    FCLayer *output_layer = (FCLayer *)net->layers[net->num_layers - 1];
    float loss = 0;
    for (int i = 0; i < output_layer->output_size; i++)
    {
        float y = target[i];
        float y_pred = output_layer->base.output->data[i];
        // Add small epsilon to avoid log(0)
        loss -= y * log(y_pred + 1e-10) + (1 - y) * log(1 - y_pred + 1e-10);
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
                backward_pass(net, target);
                update_parameters(net, learning_rate);

                // Calculate loss
                float loss = calculate_loss(net, target);
                total_loss += loss;
                total_samples++;

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
        if (should_stop(es, avg_loss, net, epoch))
        {
            break;
        }

        if (epoch % 100 == 0)
        {
            printf("Epoch %d, Average Loss: %f\n", epoch, avg_loss);
        }
    }

    // Restore best parameters
    int idx = 0;
    for (int i = 0; i < net->num_layers; i++)
    {
        if (net->layers[i]->type == LAYER_CONV)
        {
            ConvLayer *conv = (ConvLayer *)net->layers[i];
            int weights_size = conv->filter_width * conv->filter_height * conv->base.input->depth * conv->num_filters;
            memcpy(conv->weights, es->best_params + idx, weights_size * sizeof(float));
            idx += weights_size;
            memcpy(conv->biases, es->best_params + idx, conv->num_filters * sizeof(float));
            idx += conv->num_filters;
        }
        else if (net->layers[i]->type == LAYER_FC)
        {
            FCLayer *fc = (FCLayer *)net->layers[i];
            int weights_size = fc->input_size * fc->output_size;
            memcpy(fc->weights, es->best_params + idx, weights_size * sizeof(float));
            idx += weights_size;
            memcpy(fc->biases, es->best_params + idx, fc->output_size * sizeof(float));
            idx += fc->output_size;
        }
    }

    printf("Best model found at epoch %d with validation loss %f\n", es->best_epoch, es->best_val_loss);

    free(input);
    free(target);
    free_early_stopping(es);
}

char retrieve_answer(Network *net)
{
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

    char result;
    if (max_idx < 26)
    {
        result = 'A' + max_idx;
    }
    else
    {
        result = 'a' + (max_idx - 26);
    }

    return result;
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
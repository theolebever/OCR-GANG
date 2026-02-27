// File for Sigmoid functions in particular and other functions needed for
// operating the neural network Used for tweaking the weigth of each node in the
// neural network
#include "network.h"

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tools.h"

void freeNetwork(struct network *net)
{
    if (!net)
        return;

    free(net->input_layer);
    free(net->delta_input);

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

    //free(net->hidden_pre_activation);

    free(net->goal);

    free(net);
}


struct network *InitializeNetwork(double i, double h, double o, char *filepath)
{
    struct network *network = malloc(sizeof(struct network));
    if (network == NULL)
    {
        errx(1, "Not enough memory!");
    }
    network->number_of_inputs = i;
    network->number_of_hidden_nodes = h;
    network->number_of_outputs = o;

    // Initialize all pointers to NULL to ensure safe cleanup on error
    network->input_layer = NULL;
    network->hidden_layer = NULL;
    network->delta_hidden = NULL;
    network->hidden_layer_bias = NULL;
    network->hidden_weights = NULL;
    network->delta_hidden_weights = NULL;
    network->output_layer = NULL;
    network->delta_output = NULL;
    network->output_layer_bias = NULL;
    network->output_weights = NULL;
    network->delta_output_weights = NULL;
    network->goal = NULL;

    network->input_layer = calloc(network->number_of_inputs, sizeof(double));
    network->delta_input = calloc(network->number_of_inputs, sizeof(double));
    if (network->input_layer == NULL || network->delta_input == NULL)
    {
        freeNetwork(network);
        errx(1, "Not enough memory!");
    }

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

    if (network->hidden_layer == NULL || network->delta_hidden == NULL ||
        network->hidden_layer_bias == NULL || network->hidden_weights == NULL ||
        network->delta_hidden_weights == NULL)
    {
        freeNetwork(network);
        errx(1, "Not enough memory!");
    }

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

    if (network->output_layer == NULL || network->delta_output == NULL ||
        network->output_layer_bias == NULL || network->output_weights == NULL ||
        network->delta_output_weights == NULL || network->goal == NULL)
    {
        freeNetwork(network);
        errx(1, "Not enough memory!");
    }

    // OPTIMIZED for tiny datasets: Much lower learning rate with lower momentum
    network->eta = 0.01f;   // Very conservative learning rate
    network->alpha = 0.9f;   // Moderate momentum

    if (filepath != NULL && !fileempty(filepath))
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
    // He initialization for hidden layer (ReLU)
    for (int i = 0; i < net->number_of_inputs; i++)
    {
        for (int j = 0; j < net->number_of_hidden_nodes; j++)
        {
            net->hidden_weights[i * net->number_of_hidden_nodes + j] =
                init_weight_he(net->number_of_inputs);
        }
    }

    // Small positive bias to avoid dead ReLUs
    for (int j = 0; j < net->number_of_hidden_nodes; j++)
        net->hidden_layer_bias[j] = 0.01;

    // Xavier initialization for output layer
    for (int k = 0; k < net->number_of_hidden_nodes; k++)
    {
        for (int l = 0; l < net->number_of_outputs; l++)
        {
            net->output_weights[k * net->number_of_outputs + l] =
                init_weight_xavier(net->number_of_hidden_nodes,
                                   net->number_of_outputs);
        }
    }

    for (int l = 0; l < net->number_of_outputs; l++)
        net->output_layer_bias[l] = 0.0;

    // IMPORTANT: reset momentum buffers
    memset(net->delta_hidden_weights, 0,
           sizeof(double) * net->number_of_inputs * net->number_of_hidden_nodes);

    memset(net->delta_output_weights, 0,
           sizeof(double) * net->number_of_hidden_nodes * net->number_of_outputs);
}


void forward_pass(struct network *net)
{
    int H = net->number_of_hidden_nodes;
    int O = net->number_of_outputs;

    // Hidden layer — initialize with biases
    for (int j = 0; j < H; j++)
        net->hidden_layer[j] = net->hidden_layer_bias[j];

    // Accumulate: i outer, j inner -> sequential access to hidden_weights row i
    for (int i = 0; i < net->number_of_inputs; i++)
    {
        double in_i = net->input_layer[i];
        double *w_row = net->hidden_weights + i * H;
        for (int j = 0; j < H; j++)
            net->hidden_layer[j] += in_i * w_row[j];
    }

    for (int j = 0; j < H; j++)
        net->hidden_layer[j] = relu(net->hidden_layer[j]);

    // Output layer — initialize with biases
    for (int o = 0; o < O; o++)
        net->output_layer[o] = net->output_layer_bias[o];

    // Accumulate: h outer, o inner -> sequential access to output_weights row h
    for (int h = 0; h < H; h++)
    {
        double hid_h = net->hidden_layer[h];
        double *w_row = net->output_weights + h * O;
        for (int o = 0; o < O; o++)
            net->output_layer[o] += hid_h * w_row[o];
    }

    softmax(net->output_layer, O);
}


void back_propagation(struct network *net)
{
    int H = net->number_of_hidden_nodes;
    int O = net->number_of_outputs;
    double eta = net->eta;
    double alpha = net->alpha;

    // Output layer delta (Softmax + Cross Entropy)
    for (int o = 0; o < O; o++)
        net->delta_output[o] = net->output_layer[o] - net->goal[o];

    // Hidden layer delta — h outer, o inner -> sequential access to output_weights row h
    for (int h = 0; h < H; h++)
    {
        double sum = 0.0;
        double *w_row = net->output_weights + h * O;
        for (int o = 0; o < O; o++)
            sum += w_row[o] * net->delta_output[o];
        net->delta_hidden[h] = sum * dRelu(net->hidden_layer[h]);
    }

    // Update output weights — h outer, o inner -> sequential row access
    for (int h = 0; h < H; h++)
    {
        double hid_h = net->hidden_layer[h];
        double *w_row  = net->output_weights       + h * O;
        double *dw_row = net->delta_output_weights + h * O;
        for (int o = 0; o < O; o++)
        {
            double grad = net->delta_output[o] * hid_h;
            double delta = -eta * grad + alpha * dw_row[o];
            // Clip the final weight update, not the raw gradient
            if (delta >  0.5) delta =  0.5;
            if (delta < -0.5) delta = -0.5;
            w_row[o]  += delta;
            dw_row[o]  = delta;
        }
    }

    // Update output biases
    for (int o = 0; o < O; o++)
        net->output_layer_bias[o] -= eta * net->delta_output[o];

    // Update hidden weights — i outer, h inner -> sequential row access
    for (int i = 0; i < net->number_of_inputs; i++)
    {
        double in_i  = net->input_layer[i];
        double *w_row  = net->hidden_weights       + i * H;
        double *dw_row = net->delta_hidden_weights + i * H;
        for (int h = 0; h < H; h++)
        {
            double grad = net->delta_hidden[h] * in_i;
            double delta = -eta * grad + alpha * dw_row[h];
            // Clip the final weight update, not the raw gradient
            if (delta >  0.5) delta =  0.5;
            if (delta < -0.5) delta = -0.5;
            w_row[h]  += delta;
            dw_row[h]  = delta;
        }
    }

    // Update hidden biases
    for (int h = 0; h < H; h++)
        net->hidden_layer_bias[h] -= eta * net->delta_hidden[h];

    // Compute input gradients for CNN — i outer, h inner -> sequential row access
    for (int i = 0; i < net->number_of_inputs; i++)
    {
        double sum = 0.0;
        double *w_row = net->hidden_weights + i * H;
        for (int h = 0; h < H; h++)
            sum += w_row[h] * net->delta_hidden[h];
        net->delta_input[i] = sum;
    }
}


int InputImage(struct network *net, size_t index, int ***chars_matrix)
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
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
    if (network->input_layer == NULL)
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
    // Hidden layer
    for (int j = 0; j < net->number_of_hidden_nodes; j++)
    {
        double z = net->hidden_layer_bias[j];

        for (int i = 0; i < net->number_of_inputs; i++)
        {
            z += net->input_layer[i] *
                 net->hidden_weights[i * net->number_of_hidden_nodes + j];
        }

        net->hidden_layer[j] = relu(z);
    }

    // Output layer (pre-softmax)
    for (int o = 0; o < net->number_of_outputs; o++)
    {
        double z = net->output_layer_bias[o];

        for (int h = 0; h < net->number_of_hidden_nodes; h++)
        {
            z += net->hidden_layer[h] *
                 net->output_weights[h * net->number_of_outputs + o];
        }

        net->output_layer[o] = z;
    }

    softmax(net->output_layer, net->number_of_outputs);
}


void back_propagation(struct network *net)
{
    // Output layer delta (Softmax + Cross Entropy)
    for (int o = 0; o < net->number_of_outputs; o++)
    {
        net->delta_output[o] = net->output_layer[o] - net->goal[o];
    }

    // Hidden layer delta
    for (int h = 0; h < net->number_of_hidden_nodes; h++)
    {
        double sum = 0.0;
        for (int o = 0; o < net->number_of_outputs; o++)
        {
            sum += net->output_weights[h * net->number_of_outputs + o] *
                   net->delta_output[o];
        }

        net->delta_hidden[h] = sum * dRelu(net->hidden_layer[h]);
    }

    // Update output weights
    for (int h = 0; h < net->number_of_hidden_nodes; h++)
    {
        for (int o = 0; o < net->number_of_outputs; o++)
        {
            int idx = h * net->number_of_outputs + o;

            double grad = net->delta_output[o] * net->hidden_layer[h];
            double delta = -net->eta * grad +
                           net->alpha * net->delta_output_weights[idx];
            
            // Gradient Clipping
            if (delta > 1.0) delta = 1.0;
            if (delta < -1.0) delta = -1.0;

            net->output_weights[idx] += delta;
            net->delta_output_weights[idx] = delta;
        }
    }

    // Update output biases
    for (int o = 0; o < net->number_of_outputs; o++)
        net->output_layer_bias[o] -= net->eta * net->delta_output[o];

    // Update hidden weights
    for (int i = 0; i < net->number_of_inputs; i++)
    {
        for (int h = 0; h < net->number_of_hidden_nodes; h++)
        {
            int idx = i * net->number_of_hidden_nodes + h;

            double grad = net->delta_hidden[h] * net->input_layer[i];
            double delta = -net->eta * grad +
                           net->alpha * net->delta_hidden_weights[idx];

            // Gradient Clipping
            if (delta > 1.0) delta = 1.0;
            if (delta < -1.0) delta = -1.0;

            net->hidden_weights[idx] += delta;
            net->delta_hidden_weights[idx] = delta;
        }
    }

    // Update hidden biases
    for (int h = 0; h < net->number_of_hidden_nodes; h++)
        net->hidden_layer_bias[h] -= net->eta * net->delta_hidden[h];
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
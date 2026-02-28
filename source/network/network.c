// File for Sigmoid functions in particular and other functions needed for
// operating the neural network Used for tweaking the weigth of each node in the
// neural network
#include "network.h"

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tools.h"

// Adam hyperparameters
#define ADAM_BETA1  0.9
#define ADAM_BETA2  0.999
#define ADAM_EPS    1e-8

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
    free(net->m_hidden_weights);
    free(net->v_hidden_weights);
    free(net->m_hidden_bias);
    free(net->v_hidden_bias);

    free(net->output_layer);
    free(net->delta_output);
    free(net->output_layer_bias);
    free(net->output_weights);
    free(net->m_output_weights);
    free(net->v_output_weights);
    free(net->m_output_bias);
    free(net->v_output_bias);

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
    network->delta_input = NULL;
    network->hidden_layer = NULL;
    network->delta_hidden = NULL;
    network->hidden_layer_bias = NULL;
    network->hidden_weights = NULL;
    network->m_hidden_weights = NULL;
    network->v_hidden_weights = NULL;
    network->m_hidden_bias = NULL;
    network->v_hidden_bias = NULL;
    network->output_layer = NULL;
    network->delta_output = NULL;
    network->output_layer_bias = NULL;
    network->output_weights = NULL;
    network->m_output_weights = NULL;
    network->v_output_weights = NULL;
    network->m_output_bias = NULL;
    network->v_output_bias = NULL;
    network->goal = NULL;
    network->hidden_pre_activation = NULL;

    network->input_layer = calloc(network->number_of_inputs, sizeof(double));
    network->delta_input = calloc(network->number_of_inputs, sizeof(double));
    if (network->input_layer == NULL || network->delta_input == NULL)
    {
        freeNetwork(network);
        errx(1, "Not enough memory!");
    }

    int I = network->number_of_inputs;
    int H = network->number_of_hidden_nodes;
    int O = network->number_of_outputs;

    network->hidden_layer       = calloc(H, sizeof(double));
    network->delta_hidden       = calloc(H, sizeof(double));
    network->hidden_layer_bias  = calloc(H, sizeof(double));
    network->hidden_weights     = calloc(I * H, sizeof(double));
    network->m_hidden_weights   = calloc(I * H, sizeof(double));
    network->v_hidden_weights   = calloc(I * H, sizeof(double));
    network->m_hidden_bias      = calloc(H, sizeof(double));
    network->v_hidden_bias      = calloc(H, sizeof(double));

    if (network->hidden_layer == NULL || network->delta_hidden == NULL ||
        network->hidden_layer_bias == NULL || network->hidden_weights == NULL ||
        network->m_hidden_weights == NULL || network->v_hidden_weights == NULL ||
        network->m_hidden_bias == NULL || network->v_hidden_bias == NULL)
    {
        freeNetwork(network);
        errx(1, "Not enough memory!");
    }

    network->output_layer       = calloc(O, sizeof(double));
    network->delta_output       = calloc(O, sizeof(double));
    network->output_layer_bias  = calloc(O, sizeof(double));
    network->output_weights     = calloc(H * O, sizeof(double));
    network->m_output_weights   = calloc(H * O, sizeof(double));
    network->v_output_weights   = calloc(H * O, sizeof(double));
    network->m_output_bias      = calloc(O, sizeof(double));
    network->v_output_bias      = calloc(O, sizeof(double));
    network->goal               = calloc(O, sizeof(double));

    if (network->output_layer == NULL || network->delta_output == NULL ||
        network->output_layer_bias == NULL || network->output_weights == NULL ||
        network->m_output_weights == NULL || network->v_output_weights == NULL ||
        network->m_output_bias == NULL || network->v_output_bias == NULL ||
        network->goal == NULL)
    {
        freeNetwork(network);
        errx(1, "Not enough memory!");
    }

    network->eta          = 0.001;  // Adam default learning rate
    network->adam_t       = 0;
    network->adam_beta1_t = 1.0;
    network->adam_beta2_t = 1.0;

    if (filepath != NULL && !fileempty(filepath))
    {
        load_network(filepath, network);
        // Adam state is not persisted; moment buffers stay zeroed, timestep stays 0
    }
    else
    {
        initialization(network);
    }
    return network;
}

void initialization(struct network *net)
{
    int I = net->number_of_inputs;
    int H = net->number_of_hidden_nodes;
    int O = net->number_of_outputs;

    // He initialization for hidden layer (ReLU)
    for (int i = 0; i < I; i++)
    {
        for (int j = 0; j < H; j++)
        {
            net->hidden_weights[i * H + j] =
                init_weight_he(net->number_of_inputs);
        }
    }

    // Small positive bias to avoid dead ReLUs
    for (int j = 0; j < H; j++)
        net->hidden_layer_bias[j] = 0.01;

    // Xavier initialization for output layer
    for (int k = 0; k < H; k++)
    {
        for (int l = 0; l < O; l++)
        {
            net->output_weights[k * O + l] =
                init_weight_xavier(H, O);
        }
    }

    for (int l = 0; l < O; l++)
        net->output_layer_bias[l] = 0.0;

    // Reset Adam moment buffers and timestep
    memset(net->m_hidden_weights, 0, sizeof(double) * I * H);
    memset(net->v_hidden_weights, 0, sizeof(double) * I * H);
    memset(net->m_hidden_bias,    0, sizeof(double) * H);
    memset(net->v_hidden_bias,    0, sizeof(double) * H);

    memset(net->m_output_weights, 0, sizeof(double) * H * O);
    memset(net->v_output_weights, 0, sizeof(double) * H * O);
    memset(net->m_output_bias,    0, sizeof(double) * O);
    memset(net->v_output_bias,    0, sizeof(double) * O);

    net->adam_t      = 0;
    net->adam_beta1_t = 1.0;
    net->adam_beta2_t = 1.0;
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

    if (O == 1)
    {
        // Binary classification (XOR): use sigmoid
        net->output_layer[0] = sigmoid(net->output_layer[0]);
    }
    else
    {
        // Multi-class (OCR): use softmax
        softmax(net->output_layer, O);
    }
}


void back_propagation(struct network *net)
{
    int H = net->number_of_hidden_nodes;
    int O = net->number_of_outputs;
    double eta = net->eta;

    // Advance Adam timestep; update running beta^t products
    net->adam_t += 1;
    net->adam_beta1_t *= ADAM_BETA1;
    net->adam_beta2_t *= ADAM_BETA2;
    double bc1 = 1.0 - net->adam_beta1_t;
    double bc2 = 1.0 - net->adam_beta2_t;

    // Output layer delta (Softmax + Cross Entropy combined gradient)
    for (int o = 0; o < O; o++)
        net->delta_output[o] = net->output_layer[o] - net->goal[o];

    // Hidden layer delta
    for (int h = 0; h < H; h++)
    {
        double sum = 0.0;
        double *w_row = net->output_weights + h * O;
        for (int o = 0; o < O; o++)
            sum += w_row[o] * net->delta_output[o];
        net->delta_hidden[h] = sum * dRelu(net->hidden_layer[h]);
    }

    // Update output weights with Adam
    // Skip rows where hidden activation is zero (dead ReLU neurons)
    for (int h = 0; h < H; h++)
    {
        double hid_h = net->hidden_layer[h];
        if (hid_h == 0.0) continue;
        double *w_row  = net->output_weights    + h * O;
        double *m_row  = net->m_output_weights  + h * O;
        double *v_row  = net->v_output_weights  + h * O;
        for (int o = 0; o < O; o++)
        {
            double grad = net->delta_output[o] * hid_h;
            m_row[o] = ADAM_BETA1 * m_row[o] + (1.0 - ADAM_BETA1) * grad;
            v_row[o] = ADAM_BETA2 * v_row[o] + (1.0 - ADAM_BETA2) * grad * grad;
            double m_hat = m_row[o] / bc1;
            double v_hat = v_row[o] / bc2;
            w_row[o] -= eta * m_hat / (my_sqrt(v_hat) + ADAM_EPS);
        }
    }

    // Update output biases with Adam
    for (int o = 0; o < O; o++)
    {
        double grad = net->delta_output[o];
        net->m_output_bias[o] = ADAM_BETA1 * net->m_output_bias[o] + (1.0 - ADAM_BETA1) * grad;
        net->v_output_bias[o] = ADAM_BETA2 * net->v_output_bias[o] + (1.0 - ADAM_BETA2) * grad * grad;
        double m_hat = net->m_output_bias[o] / bc1;
        double v_hat = net->v_output_bias[o] / bc2;
        net->output_layer_bias[o] -= eta * m_hat / (my_sqrt(v_hat) + ADAM_EPS);
    }

    // Update hidden weights with Adam
    // Skip rows where input is zero — no gradient flows, moments unchanged
    for (int i = 0; i < net->number_of_inputs; i++)
    {
        double in_i = net->input_layer[i];
        if (in_i == 0.0) continue;
        double *w_row  = net->hidden_weights    + i * H;
        double *m_row  = net->m_hidden_weights  + i * H;
        double *v_row  = net->v_hidden_weights  + i * H;
        for (int h = 0; h < H; h++)
        {
            double grad = net->delta_hidden[h] * in_i;
            m_row[h] = ADAM_BETA1 * m_row[h] + (1.0 - ADAM_BETA1) * grad;
            v_row[h] = ADAM_BETA2 * v_row[h] + (1.0 - ADAM_BETA2) * grad * grad;
            double m_hat = m_row[h] / bc1;
            double v_hat = v_row[h] / bc2;
            w_row[h] -= eta * m_hat / (my_sqrt(v_hat) + ADAM_EPS);
        }
    }

    // Update hidden biases with Adam
    for (int h = 0; h < H; h++)
    {
        double grad = net->delta_hidden[h];
        net->m_hidden_bias[h] = ADAM_BETA1 * net->m_hidden_bias[h] + (1.0 - ADAM_BETA1) * grad;
        net->v_hidden_bias[h] = ADAM_BETA2 * net->v_hidden_bias[h] + (1.0 - ADAM_BETA2) * grad * grad;
        double m_hat = net->m_hidden_bias[h] / bc1;
        double v_hat = net->v_hidden_bias[h] / bc2;
        net->hidden_layer_bias[h] -= eta * m_hat / (my_sqrt(v_hat) + ADAM_EPS);
    }

    // Compute input gradients for CNN
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

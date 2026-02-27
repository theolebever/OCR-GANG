#ifndef NN_H_
#define NN_H_

#include <stddef.h>
struct network
{
    int number_of_inputs;
    int number_of_hidden_nodes;
    int number_of_outputs;
    double *input_layer;
    double *delta_input; // Gradients for the input layer (needed for CNN backprop)

    double *hidden_layer;
    double *delta_hidden;
    double *hidden_layer_bias;
    double *hidden_weights;

    // Adam moment buffers for hidden weights
    double *m_hidden_weights; // 1st moment
    double *v_hidden_weights; // 2nd moment

    // Adam moment buffers for hidden biases
    double *m_hidden_bias;
    double *v_hidden_bias;

    double *output_layer;
    double *delta_output;
    double *output_layer_bias;
    double *output_weights;

    // Adam moment buffers for output weights
    double *m_output_weights; // 1st moment
    double *v_output_weights; // 2nd moment

    // Adam moment buffers for output biases
    double *m_output_bias;
    double *v_output_bias;

    double *hidden_pre_activation;

    double eta;         // Learning rate
    long adam_t;        // Adam timestep counter
    double adam_beta1_t; // ADAM_BETA1^t (running product for bias correction)
    double adam_beta2_t; // ADAM_BETA2^t (running product for bias correction)

    double *goal;
};

struct network *InitializeNetwork(double i, double h, double o, char *filepath);

void initialization(struct network *net);

void forward_pass(struct network *net);

void back_propagation(struct network *net);

void updateweightsetbiases(struct network *net);

int InputImage(struct network *net, size_t index, int ***chars_matrix);

void freeNetwork(struct network *net);

#define OCR_HIDDEN_NODES 128

#endif

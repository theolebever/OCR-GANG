#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdbool.h>

#define BATCH_SIZE 32
#define MAX_EPOCHS 2500
#define EARLY_STOPPING_WINDOW 10
#define EARLY_STOPPING_THRESHOLD 0.0001

struct network
{
    size_t number_of_inputs;
    size_t number_of_hidden_nodes;
    size_t number_of_outputs;
    double *input_layer;

    double *hidden_layer;
    double *delta_hidden;
    double *hidden_layer_bias;
    double *hidden_weights;
    double *delta_hidden_weights;

    double *output_layer;
    double *delta_output;
    double *output_layer_bias;
    double *output_weights;
    double *delta_output_weights;

    double eta;
    double alpha;
    double *goal;
};

struct network *initialize_network(double i, double h, double o, const char *filepath);

void initialization(struct network *net);

void forward_pass(struct network *net);

void back_propagation(struct network *net);

void update_weights_and_biases(struct network *net);

int input_image(struct network *net, size_t index, int ***chars_matrix);

void free_network(struct network *net);

void adaptive_learning_rate(struct network *net);

bool early_stopping(double *errors, int window_size, double threshold);

void mini_batch_training(struct network *net, int ***chars_matrix, int total_samples);

#endif

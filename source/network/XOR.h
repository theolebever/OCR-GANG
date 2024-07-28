#ifndef XOR_H_
#define XOR_H_

#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

#define BATCH_SIZE 32
#define MAX_EPOCHS 2500
#define EARLY_STOPPING_WINDOW 10
#define EARLY_STOPPING_THRESHOLD 0.0001

#define KRED "\x1B[31m"
#define KWHT "\x1B[37m"
#define KGRN "\x1B[32m"

struct fnn
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

void run_xor_demo();
struct fnn *init_xor(const char *filepath);
void initialization(struct fnn *net);
void forward_pass_xor(struct fnn *net);
void back_propagation(struct fnn *net);
void update_weights_and_biases(struct fnn *net);
void free_network(struct fnn *net);
void adaptive_learning_rate(struct fnn *net);
bool early_stopping(double *errors, int window_size, double threshold);
void train_network(struct fnn *network);
void use_network(struct fnn *network);
void save_training_results(struct fnn *network, FILE *result_file, int index);

#endif
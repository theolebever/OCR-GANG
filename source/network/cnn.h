#ifndef CNN_H
#define CNN_H

#include "tools.h"
#include <stdlib.h>

#define CONV_SIZE 3
#define POOL_SIZE 2
#define NUM_FILTERS 8
#define INPUT_W 28
#define INPUT_H 28

// Derived dimensions
#define CONV_W (INPUT_W - CONV_SIZE + 1) // 26
#define CONV_H (INPUT_H - CONV_SIZE + 1) // 26
#define POOL_W (CONV_W / POOL_SIZE)      // 13
#define POOL_H (CONV_H / POOL_SIZE)      // 13
#define FLATTEN_SIZE (NUM_FILTERS * POOL_W * POOL_H) // 8 * 169 = 1352

typedef struct {
    // Weights: [NUM_FILTERS][3][3]
    double filters[NUM_FILTERS][CONV_SIZE][CONV_SIZE];
    double filter_grads[NUM_FILTERS][CONV_SIZE][CONV_SIZE];

    // Adam moment buffers for filters
    double m_filters[NUM_FILTERS][CONV_SIZE][CONV_SIZE];
    double v_filters[NUM_FILTERS][CONV_SIZE][CONV_SIZE];

    // Biases: [NUM_FILTERS]
    double biases[NUM_FILTERS];
    double bias_grads[NUM_FILTERS];

    // Adam moment buffers for biases
    double m_biases[NUM_FILTERS];
    double v_biases[NUM_FILTERS];

    // Adam timestep and running beta^t products
    long adam_t;
    double adam_beta1_t;
    double adam_beta2_t;

    // Intermediate states for backprop
    double input[INPUT_H][INPUT_W];
    double conv_output[NUM_FILTERS][CONV_H][CONV_W]; // 26x26
    double pool_output[NUM_FILTERS][POOL_H][POOL_W];
    int    pool_mask[NUM_FILTERS][POOL_H][POOL_W];

} CNN;

CNN* init_cnn();
void free_cnn(CNN* cnn);

// Forward pass: writes 1352 doubles into out[]. No allocation.
void cnn_forward(CNN* cnn, double image[784], double *out);

// Backward pass: Takes gradients coming FROM the dense layer (1352 doubles)
// Updates CNN weights internally.
void cnn_backward(CNN* cnn, double* output_gradients, double eta);

#endif

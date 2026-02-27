#include "cnn.h"
#include <stdio.h>
#include <math.h>

CNN* init_cnn() {
    CNN* cnn = calloc(1, sizeof(CNN));
    if (!cnn) return NULL;

    // Xavier/He Initialization for filters
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < CONV_SIZE; i++) {
            for (int j = 0; j < CONV_SIZE; j++) {
                // He initialization for Relu
                cnn->filters[f][i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * sqrt(2.0 / (CONV_SIZE * CONV_SIZE));
            }
        }
        cnn->biases[f] = 0.0;
    }
    return cnn;
}

void free_cnn(CNN* cnn) {
    if (cnn) free(cnn);
}

// 2D access helper
static inline double get_img(double* img, int x, int y) {
    return img[y * INPUT_W + x];
}

void cnn_forward(CNN* cnn, double image[784], double *out) {
    // 1. Copy input to internal storage
    for (int y = 0; y < INPUT_H; y++) {
        for (int x = 0; x < INPUT_W; x++) {
            cnn->input[y][x] = image[y * INPUT_W + x];
        }
    }

    // 2. Convolution (valid padding) + ReLU
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int y = 0; y < CONV_H; y++) {
            for (int x = 0; x < CONV_W; x++) {
                double sum = cnn->biases[f];
                for (int i = 0; i < CONV_SIZE; i++) {
                    for (int j = 0; j < CONV_SIZE; j++) {
                        sum += cnn->input[y + i][x + j] * cnn->filters[f][i][j];
                    }
                }
                cnn->conv_output[f][y][x] = (sum > 0) ? sum : 0.0;
            }
        }
    }

    // 3. Max Pooling (2x2)
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int y = 0; y < POOL_H; y++) {
            for (int x = 0; x < POOL_W; x++) {
                double max_val = -1e9;
                int max_idx = 0;
                int start_y = y * POOL_SIZE;
                int start_x = x * POOL_SIZE;

                for (int i = 0; i < POOL_SIZE; i++) {
                    for (int j = 0; j < POOL_SIZE; j++) {
                        double val = cnn->conv_output[f][start_y + i][start_x + j];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = i * POOL_SIZE + j;
                        }
                    }
                }
                cnn->pool_output[f][y][x] = max_val;
                cnn->pool_mask[f][y][x] = max_idx;
            }
        }
    }

    // 4. Flatten directly into caller-supplied buffer (no malloc)
    int idx = 0;
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int y = 0; y < POOL_H; y++) {
            for (int x = 0; x < POOL_W; x++) {
                out[idx++] = cnn->pool_output[f][y][x];
            }
        }
    }
}

void cnn_backward(CNN* cnn, double* output_gradients, double eta) {
    // output_gradients size = FLATTEN_SIZE (1352)
    // 1. Un-flatten gradients into pooling layer gradients
    double pool_grads[NUM_FILTERS][POOL_H][POOL_W];
    int idx = 0;
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int y = 0; y < POOL_H; y++) {
            for (int x = 0; x < POOL_W; x++) {
                pool_grads[f][y][x] = output_gradients[idx++];
            }
        }
    }

    // 2. Backprop through Max Pooling (Upsample)
    double conv_grads[NUM_FILTERS][CONV_H][CONV_W] = {0}; // Initialize to 0

    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int y = 0; y < POOL_H; y++) {
            for (int x = 0; x < POOL_W; x++) {
                int max_idx = cnn->pool_mask[f][y][x];
                int dy = max_idx / POOL_SIZE;
                int dx = max_idx % POOL_SIZE;
                
                int target_y = y * POOL_SIZE + dy;
                int target_x = x * POOL_SIZE + dx;
                
                conv_grads[f][target_y][target_x] = pool_grads[f][y][x];
            }
        }
    }

    // 3. Backprop through ReLU
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int y = 0; y < CONV_H; y++) {
            for (int x = 0; x < CONV_W; x++) {
                if (cnn->conv_output[f][y][x] <= 0) {
                    conv_grads[f][y][x] = 0; // Derivative of ReLU is 0 for x<=0
                }
                // else it stays as is (derivative is 1)
            }
        }
    }

    // 4. Backprop through Convolution (Gradients for Filters and Biases)
    // Clear gradients
    for(int f=0; f<NUM_FILTERS; f++) {
        cnn->bias_grads[f] = 0;
        for(int i=0; i<CONV_SIZE; i++)
            for(int j=0; j<CONV_SIZE; j++)
                cnn->filter_grads[f][i][j] = 0;
    }

    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int y = 0; y < CONV_H; y++) {
            for (int x = 0; x < CONV_W; x++) {
                double grad = conv_grads[f][y][x];
                
                // Accumulate bias gradient
                cnn->bias_grads[f] += grad;

                // Accumulate filter gradients
                for (int i = 0; i < CONV_SIZE; i++) {
                    for (int j = 0; j < CONV_SIZE; j++) {
                        // Input causing this output was at (y+i, x+j)
                        cnn->filter_grads[f][i][j] += cnn->input[y + i][x + j] * grad;
                    }
                }
            }
        }
    }

    // 5. Update Weights (Gradient Descent)
    for (int f = 0; f < NUM_FILTERS; f++) {
        cnn->biases[f] -= eta * cnn->bias_grads[f];
        
        for (int i = 0; i < CONV_SIZE; i++) {
            for (int j = 0; j < CONV_SIZE; j++) {
                 // Gradient Clipping for CNN weights too
                double delta = -eta * cnn->filter_grads[f][i][j];
                if (delta > 0.1) delta = 0.1;
                if (delta < -0.1) delta = -0.1;

                cnn->filters[f][i][j] += delta;
            }
        }
    }
}

#include "cnn.h"
#include "../common.h"
#include <stdio.h>
#include <string.h>

CNN* init_cnn() {
    CNN* cnn = calloc(1, sizeof(CNN));
    if (!cnn) return NULL;

    // He initialization for filters
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < CONV_SIZE; i++) {
            for (int j = 0; j < CONV_SIZE; j++) {
                cnn->filters[f][i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0)
                                        * my_sqrt(2.0 / (CONV_SIZE * CONV_SIZE));
            }
        }
        cnn->biases[f] = 0.0;
    }

    cnn->adam_t      = 0;
    cnn->adam_beta1_t = 1.0;
    cnn->adam_beta2_t = 1.0;
    cnn->image_ptr   = NULL;

    return cnn;
}

void free_cnn(CNN* cnn) {
    if (cnn) free(cnn);
}

// Access flat image as 2D: image[y][x] = image_ptr[y * INPUT_W + x]
#define IMG(y, x) cnn->image_ptr[(y) * INPUT_W + (x)]

void cnn_forward(CNN* cnn, double image[IMAGE_PIXELS], double *out) {
    // Store pointer to input (no copy)
    cnn->image_ptr = image;

    // Convolution (valid padding) + ReLU — 3x3 kernel fully unrolled
    for (int f = 0; f < NUM_FILTERS; f++) {
        double f00 = cnn->filters[f][0][0], f01 = cnn->filters[f][0][1], f02 = cnn->filters[f][0][2];
        double f10 = cnn->filters[f][1][0], f11 = cnn->filters[f][1][1], f12 = cnn->filters[f][1][2];
        double f20 = cnn->filters[f][2][0], f21 = cnn->filters[f][2][1], f22 = cnn->filters[f][2][2];
        double bias = cnn->biases[f];

        for (int y = 0; y < CONV_H; y++) {
            for (int x = 0; x < CONV_W; x++) {
                double sum = bias
                    + IMG(y,   x) * f00 + IMG(y,   x+1) * f01 + IMG(y,   x+2) * f02
                    + IMG(y+1, x) * f10 + IMG(y+1, x+1) * f11 + IMG(y+1, x+2) * f12
                    + IMG(y+2, x) * f20 + IMG(y+2, x+1) * f21 + IMG(y+2, x+2) * f22;
                cnn->conv_output[f][y][x] = (sum > 0) ? sum : 0.0;
            }
        }
    }

    // Max Pooling (2x2)
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int y = 0; y < POOL_H; y++) {
            for (int x = 0; x < POOL_W; x++) {
                int sy = y * POOL_SIZE;
                int sx = x * POOL_SIZE;
                double v00 = cnn->conv_output[f][sy][sx];
                double v01 = cnn->conv_output[f][sy][sx+1];
                double v10 = cnn->conv_output[f][sy+1][sx];
                double v11 = cnn->conv_output[f][sy+1][sx+1];

                double max_val = v00;
                int max_idx = 0;
                if (v01 > max_val) { max_val = v01; max_idx = 1; }
                if (v10 > max_val) { max_val = v10; max_idx = 2; }
                if (v11 > max_val) { max_val = v11; max_idx = 3; }

                cnn->pool_output[f][y][x] = max_val;
                cnn->pool_mask[f][y][x] = max_idx;
            }
        }
    }

    // Flatten directly into caller-supplied buffer
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
    // Advance Adam timestep
    cnn->adam_t += 1;
    cnn->adam_beta1_t *= ADAM_BETA1;
    cnn->adam_beta2_t *= ADAM_BETA2;
    double inv_bc1 = 1.0 / (1.0 - cnn->adam_beta1_t);
    double inv_bc2 = 1.0 / (1.0 - cnn->adam_beta2_t);

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

    // 2. Backprop through Max Pooling
    double conv_grads[NUM_FILTERS][CONV_H][CONV_W];
    memset(conv_grads, 0, sizeof(conv_grads));

    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int y = 0; y < POOL_H; y++) {
            for (int x = 0; x < POOL_W; x++) {
                int max_idx = cnn->pool_mask[f][y][x];
                int dy = max_idx / POOL_SIZE;
                int dx = max_idx % POOL_SIZE;
                conv_grads[f][y * POOL_SIZE + dy][x * POOL_SIZE + dx] = pool_grads[f][y][x];
            }
        }
    }

    // 3. Backprop through ReLU
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int y = 0; y < CONV_H; y++) {
            for (int x = 0; x < CONV_W; x++) {
                if (cnn->conv_output[f][y][x] <= 0)
                    conv_grads[f][y][x] = 0;
            }
        }
    }

    // 4. Accumulate filter and bias gradients (3x3 unrolled)
    for (int f = 0; f < NUM_FILTERS; f++) {
        cnn->bias_grads[f] = 0;
        for (int i = 0; i < CONV_SIZE; i++)
            for (int j = 0; j < CONV_SIZE; j++)
                cnn->filter_grads[f][i][j] = 0;
    }

    for (int f = 0; f < NUM_FILTERS; f++) {
        double *fg = &cnn->filter_grads[f][0][0];
        for (int y = 0; y < CONV_H; y++) {
            for (int x = 0; x < CONV_W; x++) {
                double grad = conv_grads[f][y][x];
                if (grad == 0.0) continue;
                cnn->bias_grads[f] += grad;
                fg[0] += IMG(y,   x)   * grad;
                fg[1] += IMG(y,   x+1) * grad;
                fg[2] += IMG(y,   x+2) * grad;
                fg[3] += IMG(y+1, x)   * grad;
                fg[4] += IMG(y+1, x+1) * grad;
                fg[5] += IMG(y+1, x+2) * grad;
                fg[6] += IMG(y+2, x)   * grad;
                fg[7] += IMG(y+2, x+1) * grad;
                fg[8] += IMG(y+2, x+2) * grad;
            }
        }
    }

    // 5. Update weights with Adam (precomputed inverse bias corrections)
    for (int f = 0; f < NUM_FILTERS; f++) {
        // Bias update
        double bg = cnn->bias_grads[f];
        cnn->m_biases[f] = ADAM_BETA1 * cnn->m_biases[f] + (1.0 - ADAM_BETA1) * bg;
        cnn->v_biases[f] = ADAM_BETA2 * cnn->v_biases[f] + (1.0 - ADAM_BETA2) * bg * bg;
        double m_hat_b = cnn->m_biases[f] * inv_bc1;
        double v_hat_b = cnn->v_biases[f] * inv_bc2;
        cnn->biases[f] -= eta * m_hat_b / (my_sqrt(v_hat_b) + ADAM_EPS);

        // Filter weight update
        for (int i = 0; i < CONV_SIZE; i++) {
            for (int j = 0; j < CONV_SIZE; j++) {
                double fg = cnn->filter_grads[f][i][j];
                cnn->m_filters[f][i][j] = ADAM_BETA1 * cnn->m_filters[f][i][j] + (1.0 - ADAM_BETA1) * fg;
                cnn->v_filters[f][i][j] = ADAM_BETA2 * cnn->v_filters[f][i][j] + (1.0 - ADAM_BETA2) * fg * fg;
                double m_hat = cnn->m_filters[f][i][j] * inv_bc1;
                double v_hat = cnn->v_filters[f][i][j] * inv_bc2;
                cnn->filters[f][i][j] -= eta * m_hat / (my_sqrt(v_hat) + ADAM_EPS);
            }
        }
    }
}

#undef IMG

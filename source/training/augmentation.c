#include "augmentation.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#define PI 3.14159265359

// Helper: safe access to pixel in 28x28 double array
// (Inline implementation or macro could be used if needed, but currently unused)

// Rotate a 28x28 image by `angle` degrees into caller-supplied `output`
void rotate_matrix(double *input, double angle, double *output) {
    double rads = angle * PI / 180.0;
    double cx = 13.5;
    double cy = 13.5;
    double cos_a = cos(rads);
    double sin_a = sin(rads);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            double src_x = (x - cx) * cos_a + (y - cy) * sin_a + cx;
            double src_y = -(x - cx) * sin_a + (y - cy) * cos_a + cy;
            int nx = (int)round(src_x);
            int ny = (int)round(src_y);
            output[y * 28 + x] = (nx >= 0 && nx < 28 && ny >= 0 && ny < 28)
                                  ? input[ny * 28 + nx] : 0.0;
        }
    }
}

// Shift image by dx, dy into caller-supplied `output`
void shift_matrix(double *input, int dx, int dy, double *output) {
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int src_x = x - dx;
            int src_y = y - dy;
            output[y * 28 + x] = (src_x >= 0 && src_x < 28 && src_y >= 0 && src_y < 28)
                                  ? input[src_y * 28 + src_x] : 0.0;
        }
    }
}

// Add random noise into caller-supplied `output`
void add_noise(double *input, double intensity, double *output) {
    memcpy(output, input, 784 * sizeof(double));
    for (int i = 0; i < 784; i++) {
        if (((double)rand() / RAND_MAX) < intensity)
            output[i] = (output[i] > 0.5) ? 0.0 : 1.0;
    }
}

// Scale image into caller-supplied `output`
void scale_matrix(double *input, double scale, double *output) {
    double cx = 13.5;
    double cy = 13.5;

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            double src_x = (x - cx) / scale + cx;
            double src_y = (y - cy) / scale + cy;
            int nx = (int)round(src_x);
            int ny = (int)round(src_y);
            output[y * 28 + x] = (nx >= 0 && nx < 28 && ny >= 0 && ny < 28)
                                  ? input[ny * 28 + nx] : 0.0;
        }
    }
}

// Main augmentation function
int augment_dataset(TrainingDataSet *dataset, int multiplier) {
    if (!dataset || multiplier <= 1) return 0;

    printf("Augmenting dataset by %dx...\n", multiplier);

    int original_count = dataset->count;
    int target_count = original_count * multiplier;

    double **new_inputs = realloc(dataset->inputs, target_count * sizeof(double *));
    char    *new_labels = realloc(dataset->labels, target_count * sizeof(char));

    if (!new_inputs || !new_labels) {
        printf("Error: Memory allocation failed during augmentation.\n");
        return 0;
    }

    dataset->inputs   = new_inputs;
    dataset->labels   = new_labels;
    dataset->capacity = target_count;

    // Single scratch buffer reused for every transform — no per-sample malloc
    double scratch[784];

    int current_idx = original_count;

    for (int i = 0; i < original_count; i++) {
        double *original_img = dataset->inputs[i];
        char label = dataset->labels[i];

        for (int m = 1; m < multiplier; m++) {
            int op = rand() % 4;

            if (op == 0) {
                double angle = (rand() % 31) - 15;
                rotate_matrix(original_img, angle, scratch);
            } else if (op == 1) {
                int dx = (rand() % 5) - 2;
                int dy = (rand() % 5) - 2;
                shift_matrix(original_img, dx, dy, scratch);
            } else if (op == 2) {
                double noise_level = 0.01 + ((double)rand() / RAND_MAX) * 0.04;
                add_noise(original_img, noise_level, scratch);
            } else {
                double scale = 0.8 + ((double)rand() / RAND_MAX) * 0.4;
                scale_matrix(original_img, scale, scratch);
            }

            double *new_img = malloc(784 * sizeof(double));
            if (!new_img) break;
            memcpy(new_img, scratch, 784 * sizeof(double));
            dataset->inputs[current_idx] = new_img;
            dataset->labels[current_idx] = label;
            current_idx++;
        }
    }

    dataset->count = current_idx;
    printf("Augmentation complete. New dataset size: %d\n", dataset->count);
    return 1;
}

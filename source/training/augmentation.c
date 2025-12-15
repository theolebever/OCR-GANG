#include "augmentation.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#define PI 3.14159265359

// Helper: safe access to pixel in 28x28 double array
// (Inline implementation or macro could be used if needed, but currently unused)

// Rotate a 28x28 image by `angle` degrees (around center)
double *rotate_matrix(double *input, double angle) {
    double *output = calloc(784, sizeof(double));
    if (!output) return NULL;

    double rads = angle * PI / 180.0;
    double cx = 13.5; // Center of 28x28 image
    double cy = 13.5;

    double cos_a = cos(rads);
    double sin_a = sin(rads);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            // Target coordinate (x, y)
            // We want to find which source pixel maps to here
            // Reverse rotation
            double src_x = (x - cx) * cos_a + (y - cy) * sin_a + cx;
            double src_y = -(x - cx) * sin_a + (y - cy) * cos_a + cy;

            // Bilinear interpolation or simple nearest neighbor
            // Let's do nearest neighbor to keep it sharp for OCR binary images
            int nx = (int)round(src_x);
            int ny = (int)round(src_y);

            if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) {
                output[y * 28 + x] = input[ny * 28 + nx];
            } else {
                output[y * 28 + x] = 0.0;
            }
        }
    }
    return output;
}

// Shift image by dx, dy
double *shift_matrix(double *input, int dx, int dy) {
    double *output = calloc(784, sizeof(double));
    if (!output) return NULL;

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int src_x = x - dx;
            int src_y = y - dy;
            
            if (src_x >= 0 && src_x < 28 && src_y >= 0 && src_y < 28) {
                output[y * 28 + x] = input[src_y * 28 + src_x];
            } else {
                output[y * 28 + x] = 0.0;
            }
        }
    }
    return output;
}

// Add random noise (flip bits with probability `intensity`)
double *add_noise(double *input, double intensity) {
    double *output = malloc(784 * sizeof(double));
    if (!output) return NULL;
    memcpy(output, input, 784 * sizeof(double));

    for (int i = 0; i < 784; i++) {
        if (((double)rand() / RAND_MAX) < intensity) {
            // Flip 0->1 or 1->0 (approximately)
            output[i] = (output[i] > 0.5) ? 0.0 : 1.0;
        }
    }
    return output;
}

// Main augmentation function
int augment_dataset(TrainingDataSet *dataset, int multiplier) {
    if (!dataset || multiplier <= 1) return 0;

    printf("Augmenting dataset by %dx...\n", multiplier);

    int original_count = dataset->count;
    int target_count = original_count * multiplier;

    // Reallocate arrays
    double **new_inputs = realloc(dataset->inputs, target_count * sizeof(double *));
    char *new_labels = realloc(dataset->labels, target_count * sizeof(char));

    if (!new_inputs || !new_labels) {
        printf("Error: Memory allocation failed during augmentation.\n");
        return 0; // Partial failure, potentially dangerous if realloc returns NULL but orig ptrs valid? 
                  // Standard realloc: if fail, returns NULL, original block untouched. 
                  // But here we're updating struct members. Let's just return 0.
    }

    dataset->inputs = new_inputs;
    dataset->labels = new_labels;

    int current_idx = original_count;

    for (int i = 0; i < original_count; i++) {
        double *original_img = dataset->inputs[i];
        char label = dataset->labels[i];

        // Generate (multiplier - 1) new images for each original
        for (int m = 1; m < multiplier; m++) {
            double *new_img = NULL;
            int op = rand() % 3; // Choose operation

            if (op == 0) {
                // Rotation (-15 to +15 degrees)
                double angle = (rand() % 31) - 15; 
                new_img = rotate_matrix(original_img, angle);
            } else if (op == 1) {
                // Shift (-2 to +2 pixels)
                int dx = (rand() % 5) - 2;
                int dy = (rand() % 5) - 2;
                new_img = shift_matrix(original_img, dx, dy);
            } else {
                // Noise (1% to 5% noise)
                double noise_level = 0.01 + ((double)rand() / RAND_MAX) * 0.04;
                new_img = add_noise(original_img, noise_level);
            }

            if (new_img) {
                dataset->inputs[current_idx] = new_img;
                dataset->labels[current_idx] = label;
                current_idx++;
            }
        }
    }
    
    dataset->count = current_idx;
    printf("Augmentation complete. New dataset size: %d\n", dataset->count);
    return 1;
}

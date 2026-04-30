#include "augmentation.h"
#include "../common.h"
#include <stdlib.h>
#include "../network/tools.h"
#include <string.h>
#include <stdio.h>



static int rotation_maps_ready = 0;
static int rotation_maps[41][IMAGE_PIXELS];

static void init_rotation_maps(void)
{
    if (rotation_maps_ready)
        return;

    const double cx = 13.5;
    const double cy = 13.5;

    for (int angle = -20; angle <= 20; angle++)
    {
        double rads = angle * MY_PI / 180.0;
        double cos_a = my_cos(rads);
        double sin_a = my_sin(rads);
        int *map = rotation_maps[angle + 20];

        for (int y = 0; y < IMAGE_SIZE; y++)
        {
            for (int x = 0; x < IMAGE_SIZE; x++)
            {
                double src_x = (x - cx) * cos_a + (y - cy) * sin_a + cx;
                double src_y = -(x - cx) * sin_a + (y - cy) * cos_a + cy;
                int nx = (int)(0.5 + src_x);
                int ny = (int)(0.5 + src_y);
                map[y * IMAGE_SIZE + x] =
                    (nx >= 0 && nx < IMAGE_SIZE && ny >= 0 && ny < IMAGE_SIZE)
                        ? ny * IMAGE_SIZE + nx
                        : -1;
            }
        }
    }

    rotation_maps_ready = 1;
}

// Rotate a 28x28 image by `angle` degrees into caller-supplied `output`
void rotate_matrix(double *input, double angle, double *output) {
    int angle_i = (int)angle;
    if ((double)angle_i == angle && angle_i >= -20 && angle_i <= 20)
    {
        init_rotation_maps();
        int *map = rotation_maps[angle_i + 20];
        for (int i = 0; i < IMAGE_PIXELS; i++)
            output[i] = map[i] >= 0 ? input[map[i]] : 0.0;
        return;
    }

    double rads = angle * MY_PI / 180.0;
    double cx = 13.5;
    double cy = 13.5;
    double cos_a = my_cos(rads);
    double sin_a = my_sin(rads);

    for (int y = 0; y < IMAGE_SIZE; y++) {
        for (int x = 0; x < IMAGE_SIZE; x++) {
            double src_x = (x - cx) * cos_a + (y - cy) * sin_a + cx;
            double src_y = -(x - cx) * sin_a + (y - cy) * cos_a + cy;
            int nx = (int)(0.5 + src_x);
            int ny = (int)(0.5 + src_y);
            output[y * IMAGE_SIZE + x] = (nx >= 0 && nx < IMAGE_SIZE && ny >= 0 && ny < IMAGE_SIZE)
                                  ? input[ny * IMAGE_SIZE + nx] : 0.0;
        }
    }
}

// Shift image by dx, dy into caller-supplied `output`
void shift_matrix(double *input, int dx, int dy, double *output) {
    memset(output, 0, IMAGE_PIXELS * sizeof(double));

    int src_x0 = dx > 0 ? 0 : -dx;
    int dst_x0 = dx > 0 ? dx : 0;
    int copy_w = IMAGE_SIZE - (dx > 0 ? dx : -dx);

    int src_y0 = dy > 0 ? 0 : -dy;
    int dst_y0 = dy > 0 ? dy : 0;
    int copy_h = IMAGE_SIZE - (dy > 0 ? dy : -dy);

    if (copy_w <= 0 || copy_h <= 0)
        return;

    for (int row = 0; row < copy_h; row++) {
        memcpy(output + (dst_y0 + row) * IMAGE_SIZE + dst_x0,
               input + (src_y0 + row) * IMAGE_SIZE + src_x0,
               (size_t)copy_w * sizeof(double));
    }
}

// Add random noise into caller-supplied `output`
void add_noise(double *input, double intensity, double *output) {
    memcpy(output, input, IMAGE_PIXELS * sizeof(double));
    for (int i = 0; i < IMAGE_PIXELS; i++) {
        if (((double)rand() / RAND_MAX) < intensity)
            output[i] = (output[i] > 0.5) ? 0.0 : 1.0;
    }
}

// Scale image into caller-supplied `output`
void scale_matrix(double *input, double scale, double *output) {
    double cx = 13.5;
    double cy = 13.5;

    for (int y = 0; y < IMAGE_SIZE; y++) {
        for (int x = 0; x < IMAGE_SIZE; x++) {
            double src_x = (x - cx) / scale + cx;
            double src_y = (y - cy) / scale + cy;
            int nx = (int)(0.5 + src_x);
            int ny = (int)(0.5 + src_y);
            output[y * IMAGE_SIZE + x] = (nx >= 0 && nx < IMAGE_SIZE && ny >= 0 && ny < IMAGE_SIZE)
                                  ? input[ny * IMAGE_SIZE + nx] : 0.0;
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
    double scratch[IMAGE_PIXELS];

    int current_idx = original_count;

    for (int i = 0; i < original_count; i++) {
        double *original_img = dataset->inputs[i];
        char label = dataset->labels[i];

        for (int m = 1; m < multiplier; m++) {
            int op = rand() % 4;

            if (op == 0) {
                double angle = (rand() % 41) - 20;  // -20 to +20 degrees
                rotate_matrix(original_img, angle, scratch);
            } else if (op == 1) {
                int dx = (rand() % 7) - 3;  // -3 to +3 pixels
                int dy = (rand() % 7) - 3;
                shift_matrix(original_img, dx, dy, scratch);
            } else if (op == 2) {
                double noise_level = 0.02 + ((double)rand() / RAND_MAX) * 0.08;  // 2-10%
                add_noise(original_img, noise_level, scratch);
            } else {
                double scale = 0.75 + ((double)rand() / RAND_MAX) * 0.5;  // 0.75-1.25
                scale_matrix(original_img, scale, scratch);
            }

            double *new_img = malloc(IMAGE_PIXELS * sizeof(double));
            if (!new_img) break;
            memcpy(new_img, scratch, IMAGE_PIXELS * sizeof(double));
            dataset->inputs[current_idx] = new_img;
            dataset->labels[current_idx] = label;
            current_idx++;
        }
    }

    dataset->count = current_idx;
    printf("Augmentation complete. New dataset size: %d\n", dataset->count);
    return 1;
}

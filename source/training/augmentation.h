#ifndef AUGMENTATION_H
#define AUGMENTATION_H

#include "../network/tools.h"
#include "training.h"
#include "../common.h"

// Augment the dataset in-memory by a multiplier factor
// E.g. multiplier=10 means the dataset size increases by 10x
// Returns 1 on success, 0 on failure
int augment_dataset(TrainingDataSet *dataset, int multiplier);

// Individual transformation functions — write into caller-supplied output[IMAGE_PIXELS]
void rotate_matrix(double *input, double angle, double *output);
void shift_matrix(double *input, int dx, int dy, double *output);
void scale_matrix(double *input, double scale_factor, double *output);
void add_noise(double *input, double intensity, double *output);

#endif

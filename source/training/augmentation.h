#ifndef AUGMENTATION_H
#define AUGMENTATION_H

#include "../network/tools.h"
#include "training.h"

// Augment the dataset in-memory by a multiplier factor
// E.g. multiplier=10 means the dataset size increases by 10x
// Returns 1 on success, 0 on failure
int augment_dataset(TrainingDataSet *dataset, int multiplier);

// Individual transformation functions (exposed for testing if needed)
double *rotate_matrix(double *input, double angle);
double *shift_matrix(double *input, int dx, int dy);
double *add_noise(double *input, double intensity);

#endif

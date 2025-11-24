#ifndef TRAINING_H
#define TRAINING_H

#include "../network/network.h"

// Trains the neural network
void TrainNetwork(void);

// Helper to print training statistics
void PrintTrainingStats(char expected, char recognized, int *correct_count, int total_count);

#endif

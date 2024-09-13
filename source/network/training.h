#ifndef TRAINING_H
#define TRAINING_H

typedef struct
{
    int ****data;       // The training matrix
    int counts[52][50]; // Counts for each character and index
} TrainingData;

TrainingData *prepare_training();
void free_training_data(TrainingData *training_data);

#endif // !TRAINING_H

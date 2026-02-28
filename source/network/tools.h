#ifndef TOOLS_H_
#define TOOLS_H_

#include <stddef.h>
#include "../network/network.h"


// Training dataset structure for efficient data loading
typedef struct
{
    double **inputs;  // Array of input vectors (each IMAGE_PIXELS values for 28x28 images)
    char *labels;     // Array of expected characters
    int count;        // Total number of training samples
    int capacity;     // Allocated capacity (for pre-allocation, internal use)
} TrainingDataSet;

void progressBar(int step, int nb);
double expo(double x);
double my_sqrt(double x);
double my_sin(double x);
double my_cos(double x);
double sigmoid(double x);
double dSigmoid(double x);
double relu(double x);
double dRelu(double x);
void softmax(double *input, int n);
double init_weight();
double init_weight_he(int fan_in);
double init_weight_xavier(int fan_in, int fan_out);

int cfileexists(const char *filename);
int fileempty(const char *filename);
void save_network(const char *filename, struct network *network);
void load_network(const char *filename, struct network *network);
// CNN save/load — uses void* to avoid circular include with cnn.h
void save_cnn(const char *filename, void *cnn);
void load_cnn(const char *filename, void *cnn);
void shuffle(int *array, size_t n);
size_t IndexAnswer(struct network *net);
char RetrieveChar(size_t val);
size_t ExpectedPos(char c);
void ExpectedOutput(struct network *network, char c);
char *updatepath(char *filepath, size_t len, char c, size_t index);
void PrintState(char expected, char obtained);
void InputFromTXT(char *filepath, struct network *net);
void PrepareTraining(void);

// Load all training data into memory
TrainingDataSet *loadDataSet(void);

// Free the training dataset
void freeDataSet(TrainingDataSet *dataset);

#endif
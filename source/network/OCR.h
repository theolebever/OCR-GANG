#ifndef OCR_H_
#define OCR_H_

#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28
#define INPUT_DEPTH 1

// Define layer types
typedef enum
{
    LAYER_INPUT,
    LAYER_CONV,
    LAYER_POOL,
    LAYER_FC,
    LAYER_OUTPUT
} LayerType;

// Structure to hold 3D volume of data
typedef struct
{
    int width, height, depth;
    float *data;
} Volume;

// Generic layer structure
typedef struct Layer
{
    LayerType type;
    Volume *input;
    Volume *output;
} Layer;

// Convolutional layer
typedef struct
{
    Layer base;
    int filter_width, filter_height, num_filters;
    float *weights;
    float *biases;
    float *weight_gradients;
    float *bias_gradients;
} ConvLayer;

// Pooling layer
typedef struct
{
    Layer base;
    int pool_width, pool_height, stride;
    int *max_indices; // To store indices for max pooling
} PoolLayer;

// Fully connected layer
typedef struct
{
    Layer base;
    int input_size, output_size;
    float *weights;
    float *biases;
    float *weight_gradients;
    float *bias_gradients;
} FCLayer;

// Network structure
typedef struct
{
    int num_layers;
    Layer **layers;
} Network;

Network *create_ocr_network();
Network *create_network(int num_layers);
void free_network_cnn(Network *net);
void forward_pass_ocr(Network *net, float *input);
void backward_pass(Network *net, float *target);
void update_parameters(Network *net, float learning_rate);
Volume *create_volume(int width, int height, int depth);
void free_volume(Volume *vol);
void train(Network *net, const char *filematrix, char *expected_result, int num_samples_per_char, int epochs, float learning_rate);
int predict(Network *net, float *input);
char retrieve_answer(Network *net);

#endif
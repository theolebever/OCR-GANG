#ifndef OCR_H_
#define OCR_H_

#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28

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
    void (*forward)(struct Layer *layer, Volume *input);
    void (*backward)(struct Layer *layer, float *upstream_gradient);
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
void add_conv_layer(Network *net, int layer_index, int in_w, int in_h, int in_d,
                    int filter_w, int filter_h, int num_filters);
void add_pool_layer(Network *net, int layer_index, int in_w, int in_h, int in_d,
                    int pool_w, int pool_h, int stride);
void add_fc_layer(Network *net, int layer_index, int input_size, int output_size);
void free_network_cnn(Network *net);
void conv_forward(Layer *layer, Volume *input);
void conv_backward(Layer *layer, float *upstream_gradient);
void pool_forward(PoolLayer *layer, Volume *input);
void pool_backward(PoolLayer *layer, float *upstream_gradient);
void fc_forward(Layer *layer, Volume *input);
void fc_backward(Layer *layer, float *upstream_gradient);
void forward_pass_ocr(Network *net, float *input);
void backward_pass(Network *net, float *target);
void update_parameters(Network *net, float learning_rate);
void train(Network *net, float **training_data, float **labels, int num_samples, int epochs, float learning_rate);
int predict(Network *net, float *input);

#endif
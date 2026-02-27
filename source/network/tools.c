#include "../network/tools.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>

#include "../network/network.h"
#include "../network/cnn.h"
#include "../process/process.h"
#include "../sdl/our_sdl.h"
#include "../segmentation/segmentation.h"
#include "SDL/SDL.h"
#include "SDL/SDL_image.h"

#define KRED "\x1B[31m"
#define KWHT "\x1B[37m"
#define KGRN "\x1B[32m"

void progressBar(int step, int nb)
{
    printf("\e[?25l");
    int percent = (step * 100) / nb;
    const int pwidth = 72;
    int pos = (step * pwidth) / nb;
    printf("[");
    for (int i = 0; i < pos; i++)
    {
        printf("%c", '=');
    }
    printf("%*c ", pwidth - pos + 1, ']');
    printf(" %3d%%\r", percent);
    fflush(stdout);
}

double expo(double x) 
{
    // Handle special cases
    if (x == 0) return 1.0;
    
    // For negative x, use exp(x) = 1/exp(-x)
    if (x < 0) return 1.0 / expo(-x);

    // Scaling and Squaring method
    // Scale x down to [0, 1)
    int n = 0;
    // Fix: Prevent infinite loop if x is Infinity or extremely large
    // 2^1024 is larger than DBL_MAX, so n > 1024 is unreasonable
    while (x > 1.0 && n < 2000) {
        x /= 2.0;
        n++;
    }

    // Prevent infinite loop if x is extremely large (prevent Inf return)
    if (n >= 2000) return (x > 0) ? 1.79769e+308 : 0.0;

    // Taylor series for small x
    double sum = 1.0;
    double term = 1.0;
    for (int i = 1; i < 20; ++i) 
    {
        term *= x / i;
        sum += term;
    }

    // Square n times
    for (int i = 0; i < n; i++) {
        sum *= sum;
    }
    
    return sum;
}

double my_sqrt(double x)
{
    if (x < 0) return -1.0;
    if (x == 0) return 0.0;
    double guess = x / 2.0;
    for (int i = 0; i < 20; i++)
    {
        guess = (guess + x / guess) / 2.0;
    }
    return guess;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + expo(-x));
}

double dSigmoid(double x)
{
    return x * (1.0 - x);
}

double relu(double x)
{
    // Leaky ReLU to avoid dead neurons
    return x > 0.0 ? x : 0.01 * x;
}

double dRelu(double x)
{
    return x > 0.0 ? 1.0 : 0.01;
}

void softmax(double *input, int n)
{
    double max = input[0];
    for (int i = 1; i < n; i++)
    {
        if (input[i] > max) max = input[i];
    }

    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        input[i] = expo(input[i] - max); // Subtract max for numerical stability
        sum += input[i];
    }

    for (int i = 0; i < n; i++)
    {
        input[i] /= sum;
    }
}

// Uniform random number between min and max
double random_uniform(double min, double max)
{
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// He Initialization for ReLU (Uniform)
// Range: [-sqrt(6/n_in), sqrt(6/n_in)]
double init_weight_he(int fan_in)
{
    double limit = my_sqrt(6.0 / fan_in);
    return random_uniform(-limit, limit);
}

// Xavier/Glorot Initialization for Sigmoid/Softmax (Uniform)
// Range: [-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out))]
double init_weight_xavier(int fan_in, int fan_out)
{
    double limit = my_sqrt(6.0 / (fan_in + fan_out));
    return random_uniform(-limit, limit);
}

double init_weight()
{
    return ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
}

int cfileexists(const char *filename)
{
    if (filename == NULL) return 0;
    FILE *file = fopen(filename, "r");
    if (file == NULL) return 0;
    fclose(file);
    return 1;
}

int fileempty(const char *filename)
{
    if (filename == NULL) return 1;
    FILE *fptr = fopen(filename, "r");
    if (fptr == NULL) return 1;

    fseek(fptr, 0, SEEK_END);
    unsigned long len = (unsigned long)ftell(fptr);
    fclose(fptr);
    return (len > 0) ? 0 : 1;
}

void save_network(const char *filename, struct network *network)
{
    if (filename == NULL || network == NULL) return;
    FILE *output = fopen(filename, "w");
    if (output == NULL) return;

    // Save hidden biases (once per hidden node)
    for (int j = 0; j < network->number_of_hidden_nodes; j++)
    {
        fprintf(output, "%lf\n", network->hidden_layer_bias[j]);
    }

    // Save hidden weights
    for (int i = 0; i < network->number_of_inputs * network->number_of_hidden_nodes; i++)
    {
        fprintf(output, "%lf\n", network->hidden_weights[i]);
    }

    // Save output biases (once per output node)
    for (int j = 0; j < network->number_of_outputs; j++)
    {
        fprintf(output, "%lf\n", network->output_layer_bias[j]);
    }

    // Save output weights
    for (int i = 0; i < network->number_of_hidden_nodes * network->number_of_outputs; i++)
    {
        fprintf(output, "%lf\n", network->output_weights[i]);
    }

    fclose(output);
}

void load_network(const char *filename, struct network *network)
{
    if (filename == NULL || network == NULL) return;
    FILE *input = fopen(filename, "r");
    if (input == NULL) return;

    // Load hidden biases (once per hidden node)
    for (int j = 0; j < network->number_of_hidden_nodes; j++)
    {
        fscanf(input, "%lf\n", &network->hidden_layer_bias[j]);
    }

    // Load hidden weights
    for (int i = 0; i < network->number_of_inputs * network->number_of_hidden_nodes; i++)
    {
        fscanf(input, "%lf\n", &network->hidden_weights[i]);
    }

    // Load output biases (once per output node)
    for (int j = 0; j < network->number_of_outputs; j++)
    {
        fscanf(input, "%lf\n", &network->output_layer_bias[j]);
    }

    // Load output weights
    for (int i = 0; i < network->number_of_hidden_nodes * network->number_of_outputs; i++)
    {
        fscanf(input, "%lf\n", &network->output_weights[i]);
    }

    fclose(input);
}

void shuffle(int *array, size_t n)
{
    if (array == NULL || n <= 1) return;
    for (size_t i = 0; i < n - 1; i++)
    {
        size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

size_t IndexAnswer(struct network *net)
{
    if (net == NULL || net->output_layer == NULL) return 0;
    size_t index = 0;
    for (size_t i = 1; i < (size_t)net->number_of_outputs; i++)
    {
        if (net->output_layer[i] > net->output_layer[index])
        {
            index = i;
        }
    }
    return index;
}

char RetrieveChar(size_t val)
{
    char c;
    if (val <= 25) c = val + 65;
    else if (val > 25 && val <= 51) c = (val + 97 - 26);
    else c = '?';
    return c;
}

size_t ExpectedPos(char c)
{
    size_t index = 0;
    if (c >= 'A' && c <= 'Z') index = (size_t)(c - 65);
    else if (c >= 'a' && c <= 'z') index = (size_t)(c - 97 + 26);
    return index;
}

void ExpectedOutput(struct network *network, char c)
{
    if (network == NULL || network->goal == NULL) return;
    for (int i = 0; i < network->number_of_outputs; i++) network->goal[i] = 0;

    if (c >= 'A' && c <= 'Z') network->goal[(int)(c)-65] = 1;
    else if (c >= 'a' && c <= 'z') network->goal[((int)(c)-97) + 26] = 1;
}

char *updatepath(char *filepath, size_t len, char c, size_t index)
{
    if (filepath == NULL) return NULL;
    char *newpath = malloc(len + 20);
    if (newpath == NULL) return NULL;
    
    // Simple implementation for now
    sprintf(newpath, "%s_%c_%lu.bmp", filepath, c, index);
    return newpath;
}

void PrintState(char expected, char obtained)
{
    printf("Expected: %c, Obtained: %c\n", expected, obtained);
}

void InputFromTXT(char *filepath, struct network *net)
{
    // Suppress unused parameter warnings
    (void)filepath;
    (void)net;
    
    // This function appears to be incomplete or a stub
    // If you need to implement it, add the proper logic here
}

void freeDataSet(TrainingDataSet *dataset)
{
    if (dataset == NULL) return;
    
    for (int i = 0; i < dataset->count; i++)
    {
        free(dataset->inputs[i]);
    }
    free(dataset->inputs);
    free(dataset->labels);
    free(dataset);
}

// Helper to properly resize an image to 28x28
// Mirrors the inference pipeline: square-pad on white, binarize r<128, then resize.
double *resize_image_to_28x28(SDL_Surface *img)
{
    double *input = calloc(784, sizeof(double));
    if (input == NULL) return NULL;

    // Square-pad: place the image centered on a white square canvas
    int size = img->w > img->h ? img->w : img->h;
    int off_x = size / 2 - img->w / 2;
    int off_y = size / 2 - img->h / 2;

    int *temp_matrix = calloc(size * size, sizeof(int)); // white (0) by default
    if (temp_matrix == NULL)
    {
        free(input);
        return NULL;
    }

    // Binarize using the same threshold as ImageToMatrix in segmentation.c
    for (int y = 0; y < img->h; y++)
    {
        for (int x = 0; x < img->w; x++)
        {
            Uint32 pixel = get_pixel(img, x, y);
            Uint8 r, g, b;
            SDL_GetRGB(pixel, img->format, &r, &g, &b);
            (void)g; (void)b;
            // Match ImageToMatrix inference: r < 128 -> black (1), else -> white (0)
            temp_matrix[(y + off_y) * size + (x + off_x)] = (r < 128) ? 1 : 0;
        }
    }

    // Resize using the Resize1 function from segmentation
    int *resized = Resize1(temp_matrix, 28, 28, size, size);
    free(temp_matrix);
    
    if (resized == NULL)
    {
        free(input);
        return NULL;
    }
    
    // Convert to double array
    for (int i = 0; i < 784; i++)
    {
        input[i] = (double)resized[i];
    }
    
    free(resized);
    return input;
}

// Helper to load a single directory of images
void load_directory(const char *path, TrainingDataSet *dataset, int is_uppercase)
{
    DIR *d;
    struct dirent *dir;
    d = opendir(path);
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            if (strstr(dir->d_name, ".png") || strstr(dir->d_name, ".jpg") || strstr(dir->d_name, ".bmp"))
            {
                char fullpath[512];
                snprintf(fullpath, sizeof(fullpath), "%s/%s", path, dir->d_name);

                SDL_Surface *img = load__image(fullpath);
                if (img)
                {
                    double *input = resize_image_to_28x28(img);
                    SDL_FreeSurface(img);

                    if (input == NULL)
                    {
                        printf("Warning: Failed to resize image %s\n", fullpath);
                        continue;
                    }

                    // Grow arrays with capacity doubling (avoids realloc per element)
                    if (dataset->count == dataset->capacity)
                    {
                        int new_cap = dataset->capacity == 0 ? 64 : dataset->capacity * 2;
                        double **new_inputs = realloc(dataset->inputs, sizeof(double *) * new_cap);
                        char   *new_labels  = realloc(dataset->labels, sizeof(char)     * new_cap);
                        if (!new_inputs || !new_labels)
                        {
                            free(input);
                            printf("Error: Memory allocation failed\n");
                            break;
                        }
                        dataset->inputs   = new_inputs;
                        dataset->labels   = new_labels;
                        dataset->capacity = new_cap;
                    }

                    char label = dir->d_name[0];
                    if (is_uppercase && label >= 'a' && label <= 'z') label -= 32;
                    if (!is_uppercase && label >= 'A' && label <= 'Z') label += 32;

                    dataset->inputs[dataset->count] = input;
                    dataset->labels[dataset->count] = label;
                    dataset->count++;
                }
            }
        }
        closedir(d);
    }
    else
    {
        printf("Failed to open directory: %s\n", path);
    }
}

TrainingDataSet *loadDataSet(void)
{
    TrainingDataSet *dataset = malloc(sizeof(TrainingDataSet));
    if (dataset == NULL) return NULL;

    dataset->inputs   = NULL;
    dataset->labels   = NULL;
    dataset->count    = 0;
    dataset->capacity = 0;

    load_directory("img/training/maj", dataset, 1);
    load_directory("img/training/min", dataset, 0);

    if (dataset->count == 0)
    {
        printf("ERROR: No training images found in directories!\n");
        printf("       Expected: img/training/maj/ and img/training/min/\n");
        freeDataSet(dataset);
        return NULL;
    }

    return dataset;
}

void save_cnn(const char *filename, void *cnn_ptr)
{
    if (filename == NULL || cnn_ptr == NULL) return;
    CNN *cnn = (CNN *)cnn_ptr;
    FILE *f = fopen(filename, "w");
    if (f == NULL) return;

    for (int fi = 0; fi < NUM_FILTERS; fi++)
    {
        fprintf(f, "%lf\n", cnn->biases[fi]);
        for (int i = 0; i < CONV_SIZE; i++)
            for (int j = 0; j < CONV_SIZE; j++)
                fprintf(f, "%lf\n", cnn->filters[fi][i][j]);
    }

    fclose(f);
}

void load_cnn(const char *filename, void *cnn_ptr)
{
    if (filename == NULL || cnn_ptr == NULL) return;
    CNN *cnn = (CNN *)cnn_ptr;
    FILE *f = fopen(filename, "r");
    if (f == NULL) return;

    for (int fi = 0; fi < NUM_FILTERS; fi++)
    {
        fscanf(f, "%lf\n", &cnn->biases[fi]);
        for (int i = 0; i < CONV_SIZE; i++)
            for (int j = 0; j < CONV_SIZE; j++)
                fscanf(f, "%lf\n", &cnn->filters[fi][i][j]);
    }

    fclose(f);
}
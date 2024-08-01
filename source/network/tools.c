#include <stdio.h>
#include <math.h>

#include "../network/tools.h"
#include "../process/process.h"
#include "../sdl/our_sdl.h"
#include "../segmentation/segmentation.h"
#include "SDL/SDL.h"
#include "SDL/SDL_image.h"
#include "XOR.h"
#include "OCR.h"

#define KRED "\x1B[31m"
#define KWHT "\x1B[37m"
#define KGRN "\x1B[32m"

/* ##################################################
                ACTIVATION FUNCTIONS
   ################################################## */

float random_float()
{
    return ((float)rand() / (float)RAND_MAX) * 2 - 1;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x)
{
    float s = sigmoid(x);
    return s * (1 - s);
}

// ReLU activation function
double relu(double x)
{
    return x > 0 ? x : 0;
}

double drelu(double x)
{
    return x > 0 ? 1 : 0;
}

/* ##################################################
                INIT FUNCTIONS
   ################################################## */

void xavier_init(float *weights, int fan_in, int fan_out)
{
    float scale = sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < fan_in * fan_out; i++)
    {
        weights[i] = random_float() * scale;
    }
}

// Init all weights and biases between 0.0 and 1.0
double init_weight()
{
    return ((float)rand() / (float)RAND_MAX) * 2 - 1;
}

/* ##################################################
                FILE HANDLING FUNCTIONS
   ################################################## */

int file_exists(const char *filename)
{
    /* try to open file to read */
    FILE *file;
    file = fopen(filename, "r");
    if (!file)
    {
        fclose(file);
        return 0;
    }
    fclose(file);
    return 1;
}

int file_empty(const char *filename)
{

    FILE *fptr;
    fptr = fopen(filename, "r");
    if (fptr == NULL)
    {
        return 1;
    }
    fseek(fptr, 0, SEEK_END);
    unsigned long len = (unsigned long)ftell(fptr);
    return !(len > 0);
}

/* ##################################################
                XOR NETWORK FUNCTIONS
   ################################################## */

void save_network(const char *filename, struct fnn *network)
{
    if (!file_exists(filename))
    {
        printf("Cannot save network!\n");
        return;
    }

    FILE *output = fopen(filename, "w");
    for (size_t k = 0; k < network->number_of_inputs; k++)
    {
        for (size_t o = 0; o < network->number_of_hidden_nodes; o++)
        {
            fprintf(output, "%lf %lf\n", network->hidden_layer_bias[o],
                    network->hidden_weights[k * network->number_of_hidden_nodes + o]);
        }
    }
    for (size_t i = 0; i < network->number_of_hidden_nodes; i++)
    {
        for (size_t a = 0; a < network->number_of_outputs; a++)
        {
            fprintf(
                output, "%lf %lf\n", network->output_layer_bias[a],
                network->output_weights[i * network->number_of_outputs + a]);
        }
    }
    fclose(output);
}

void load_network(const char *filename, struct fnn *network)
{
    FILE *input = fopen(filename, "r");
    for (size_t k = 0; k < network->number_of_inputs; k++)
    {
        for (size_t o = 0; o < network->number_of_hidden_nodes; o++)
        {
            int nb_char = fscanf(input, "%lf %lf\n", &network->hidden_layer_bias[o],
                                 &network->hidden_weights[k * network->number_of_hidden_nodes + o]);
            (void)nb_char;
        }
    }
    for (size_t i = 0; i < network->number_of_hidden_nodes; i++)
    {
        for (size_t a = 0; a < network->number_of_outputs; a++)
        {
            int nb_char = fscanf(
                input, "%lf %lf\n", &network->output_layer_bias[a],
                &network->output_weights[i * network->number_of_outputs + a]);
            (void)nb_char;
        }
    }
    fclose(input);
}

/* ##################################################
                TRAINING NETWORK FUNCTIONS
   ################################################## */

void shuffle_char(char *array, size_t n)
{
    if (n > 1)
    {
        for (size_t i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            char t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        for (size_t i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

char *update_path(const char *filepath, size_t len, char c, size_t index)
{
    // Allocate enough space for the new path with the updated index and ".png"
    size_t new_len = len + 4 + 3;                         // original length + ".png" + max 3 digits for index
    char *newpath = malloc((new_len + 1) * sizeof(char)); // +1 for the null terminator
    if (newpath == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Copy the original path
    strncpy(newpath, filepath, len);
    newpath[len] = '\0'; // Ensure null-termination

    // Update specific positions
    if (len > 17)
        newpath[17] = c;
    if (len > 15)
    {
        if (c <= 'Z')
        {
            newpath[14] = 'a';
            newpath[15] = 'j';
        }
        else
        {
            newpath[14] = 'i';
            newpath[15] = 'n';
        }
    }

    // Add the index at the end, before ".png", making sure to handle multiple digits
    snprintf(newpath + 18, new_len - 21, "%zu", index); // -21 to leave space for ".png"

    // Append ".png"
    strcat(newpath, ".png");

    return newpath;
}

void input_from_txt(char *filepath, struct fnn *net)
{
    FILE *file = fopen(filepath, "r");
    if (file == NULL)
    {
        exit(1);
    }
    size_t size = 28;
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            int nb = fscanf(file, "%lf", &net->input_layer[i * size + j]);
            (void)nb;
        }
        int nb = fscanf(file, "\n");
        (void)nb;
    }
    fclose(file);
}

void read_binary_image(const char *filepath, double *arr)
{
    FILE *file = fopen(filepath, "r");
    if (file == NULL)
    {
        exit(1);
    }
    size_t size = 28;
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            int nb = fscanf(file, "%lf ", &arr[i * size + j]);
            (void)nb;
        }
        int nb = fscanf(file, "\n");
        (void)nb;
    }
    fclose(file);
}

int ****prepare_training()
{
    init_sdl();
    const char *base_filepath = "img/training/maj/A0.png";
    char expected_result[52] = {'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E',
                                'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                                'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N',
                                'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                                'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W',
                                'w', 'X', 'x', 'Y', 'y', 'Z', 'z'};

    int ****all_chars_matrices = malloc(52 * sizeof(int ***)); // 52 for each expected result character
    for (size_t i = 0; i < 52; i++)
    {
        all_chars_matrices[i] = malloc(50 * sizeof(int **)); // Allocate space for 4 fonts for each character
        for (size_t index = 0; index < 50; index++)
        {
            all_chars_matrices[i][index] = NULL;
            char *filepath = update_path(base_filepath, strlen(base_filepath), expected_result[i], index);
            SDL_Surface *image = load__image(filepath);
            image = black_and_white(image);
            DrawRedLines(image);
            int BlocCount = CountBlocs(image);
            SDL_Surface ***chars = malloc(sizeof(SDL_Surface **) * BlocCount);
            SDL_Surface **blocs = malloc(sizeof(SDL_Surface *) * BlocCount);
            int *charslen = DivideIntoBlocs(image, blocs, chars, BlocCount);

            int len = ImageToMatrix(chars, &all_chars_matrices[i][index], charslen, BlocCount);
            (void)len;
            // Free allocated memory
            free(filepath);
            for (int j = 0; j < BlocCount; ++j)
            {
                SDL_FreeSurface(blocs[j]);
                for (int k = 0; k < charslen[j]; ++k)
                {
                    SDL_FreeSurface(chars[j][k]);
                }
                free(chars[j]);
            }
            free(chars);
            free(blocs);
            free(charslen);
            SDL_FreeSurface(image);
        }
    }

    return all_chars_matrices;
}

void restore_best_params(Network *net, EarlyStopping *es)
{
    int idx = 0;
    for (int i = 0; i < net->num_layers; i++)
    {
        if (net->layers[i]->type == LAYER_CONV)
        {
            ConvLayer *conv = (ConvLayer *)net->layers[i];
            int weights_size = conv->filter_width * conv->filter_height * conv->base.input->depth * conv->num_filters;
            memcpy(conv->weights, es->best_params + idx, weights_size * sizeof(float));
            idx += weights_size;
            memcpy(conv->biases, es->best_params + idx, conv->num_filters * sizeof(float));
            idx += conv->num_filters;
        }
        else if (net->layers[i]->type == LAYER_FC)
        {
            FCLayer *fc = (FCLayer *)net->layers[i];
            int weights_size = fc->input_size * fc->output_size;
            memcpy(fc->weights, es->best_params + idx, weights_size * sizeof(float));
            idx += weights_size;
            memcpy(fc->biases, es->best_params + idx, fc->output_size * sizeof(float));
            idx += fc->output_size;
        }
    }
}

char retrieve_answer(Network *net)
{
    // Assume the last layer is fully connected with outputs for each class
    FCLayer *output_layer = (FCLayer *)net->layers[net->num_layers - 1];
    float max_val = output_layer->base.output->data[0];
    int max_idx = 0;

    for (int i = 1; i < output_layer->output_size; i++)
    {
        if (output_layer->base.output->data[i] > max_val)
        {
            max_val = output_layer->base.output->data[i];
            max_idx = i;
        }
    }

    char result;
    if (max_idx < 26)
    {
        result = 'A' + max_idx;
    }
    else
    {
        result = 'a' + (max_idx - 26);
    }

    return result;
}
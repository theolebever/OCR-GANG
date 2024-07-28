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
    // Allocate one extra byte for the null terminator
    char *newpath = malloc((len + 1) * sizeof(char));
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
    if (len > 18)
        newpath[18] = (char)(index + 48);
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

void prepare_training()
{
    init_sdl();
    const char *base_filepath = "img/training/maj/A0.png";
    const char *base_filematrix = "img/training/maj/A0.txt";
    char expected_result[52] = {'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E',
                                'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                                'J', 'j', 'K', 'k', 'L', 'I', 'M', 'm', 'N',
                                'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                                'S', 's', 'I', 't', 'U', 'u', 'V', 'v', 'W',
                                'w', 'X', 'x', 'Y', 'y', 'Z', 'z'};
    int **chars_matrix = NULL;
    int nb = 52;

    for (size_t i = 0; i < (size_t)nb; i++)
    {
        for (size_t index = 0; index < 4; index++)
        {
            char *filepath = update_path(base_filepath, strlen(base_filepath),
                                         expected_result[i], index);
            char *filematrix = update_path(base_filematrix, strlen(base_filematrix),
                                           expected_result[i], index);
            SDL_Surface *image = load__image(filepath);
            image = black_and_white(image);
            DrawRedLines(image);
            int BlocCount = CountBlocs(image);
            SDL_Surface ***chars = malloc(sizeof(SDL_Surface **) * BlocCount);
            SDL_Surface **blocs = malloc(sizeof(SDL_Surface *) * BlocCount);
            int *charslen = DivideIntoBlocs(image, blocs, chars, BlocCount);

            for (int j = 0; j < BlocCount; ++j)
            {
                SDL_FreeSurface(blocs[j]);
            }

            int chars_count = ImageToMatrix(chars, &chars_matrix, charslen, BlocCount);
            SaveMatrix(chars_matrix, filematrix);

            // Free allocated memory
            free(filepath);
            free(filematrix);

            // Free chars
            for (int j = 0; j < BlocCount; ++j)
            {
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

            // Free chars_matrix
            for (int j = 0; j < chars_count; ++j)
            {
                free(chars_matrix[j]);
            }
            free(chars_matrix);
            chars_matrix = NULL;
        }
    }
}
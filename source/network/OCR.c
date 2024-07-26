#include "SDL/SDL.h"
#include "SDL/SDL_image.h"
#include "err.h"
#include "../GUI/gui.h"
#include "../network/network.h"
#include "../network/tools.h"
#include "../process/process.h"
#include "../sdl/our_sdl.h"
#include "../segmentation/segmentation.h"
#include <math.h>

#define HIDDEN_LAYER 200
#define INPUT_LAYER 28
#define OUTPUT_LAYER 52

void free_surfaces(SDL_Surface **surfaces, int count)
{
    for (int i = 0; i < count; ++i)
    {
        SDL_FreeSurface(surfaces[i]);
    }
    free(surfaces);
}

void free_chars(SDL_Surface ***chars, int *charslen, int BlocCount)
{
    for (int j = 0; j < BlocCount; ++j)
    {
        free_surfaces(chars[j], charslen[j]);
    }
    free(chars);
    free(charslen);
}

void free_chars_matrix(int **chars_matrix, int chars_count)
{
    for (int i = 0; i < chars_count; ++i)
    {
        free(chars_matrix[i]);
    }
    free(chars_matrix);
}

void perform_ocr(char *filepath)
{
    struct network *network = initialize_network(INPUT_LAYER * INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER, "source/OCR/ocrwb.txt");
    init_sdl();
    SDL_Surface *image = load__image(filepath);
    image = black_and_white(image);
    DrawRedLines(image);
    int BlocCount = CountBlocs(image);
    SDL_Surface ***chars = malloc(sizeof(SDL_Surface **) * BlocCount);
    SDL_Surface **blocs = malloc(sizeof(SDL_Surface *) * BlocCount);
    int *charslen = DivideIntoBlocs(image, blocs, chars, BlocCount);
    SDL_SaveBMP(image, "segmentation.bmp");
    
    free_surfaces(blocs, BlocCount);

    int **chars_matrix = NULL;
    int chars_count = ImageToMatrix(chars, &chars_matrix, charslen, BlocCount);
    
    free_chars(chars, charslen, BlocCount);

    char *result = calloc(chars_count + 1, sizeof(char));
    if (result == NULL) exit(1);

    for (size_t index = 0; index < (size_t)chars_count; index++)
    {
        int is_espace = input_image(network, index, &chars_matrix);
        if (!is_espace)
        {
            forward_pass(network);
            size_t idx = index_answer(network);
            result[index] = retrieve_char(idx);
        }
        else
        {
            result[index] = ' ';
        }
    }

    free_chars_matrix(chars_matrix, chars_count);

    SDL_FreeSurface(image);
    SDL_Quit();
    free_network(network);
    
    printf("%s\n", result);
    free(result);
}

void train_neural_network()
{
    struct network *network = initialize_network(INPUT_LAYER * INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER, "source/OCR/ocrwb.txt");
    const char *base_filepath = "img/training/maj/A0.txt";
    char expected_result[52] = { 'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E',
                                 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                                 'J', 'j', 'K', 'k', 'L', 'I', 'M', 'm', 'N',
                                 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                                 'S', 's', 'I', 't', 'U', 'u', 'V', 'v', 'W',
                                 'w', 'X', 'x', 'Y', 'y', 'Z', 'z' };
    int total_samples = 52;
    double errors[EARLY_STOPPING_WINDOW] = {0};
    int error_index = 0;

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++)
    {
        double epoch_error = 0;
       
        for (int batch_start = 0; batch_start < total_samples; batch_start += BATCH_SIZE)
        {
            double batch_error = 0;
           
            for (int i = batch_start; i < batch_start + BATCH_SIZE && i < total_samples; i++)
            {
                expected_output(network, expected_result[i]);
                
                // Update filepath for each sample
                char *filepath = update_path(base_filepath, strlen(base_filepath), expected_result[i], i % 4);
                
                input_from_txt(filepath, network);
                forward_pass(network);                
                back_propagation(network);
               
                for (size_t o = 0; o < network->number_of_outputs; o++)
                {
                    batch_error += pow(network->goal[o] - network->output_layer[o], 2);
                }
                
                // Free the dynamically allocated filepath
                free(filepath);
            }
           
            update_weights_and_biases(network);
            epoch_error += batch_error;
        }
       
        adaptive_learning_rate(network);
       
        errors[error_index] = epoch_error / total_samples;
        error_index = (error_index + 1) % EARLY_STOPPING_WINDOW;
       
        if (early_stopping(errors, EARLY_STOPPING_WINDOW, EARLY_STOPPING_THRESHOLD))
        {
            printf("Early stopping at epoch %d\n", epoch);
            break;
        }
       
        printf("Epoch %d, Error: %f\n", epoch, epoch_error / total_samples);
    }
    save_network("source/OCR/ocrwb.txt", network);
    free_network(network);
}
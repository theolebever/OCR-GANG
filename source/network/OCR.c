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



void perform_ocr(char *filepath)
{
    struct network *network =
        initialize_network(28 * 28, 20, 52, "../OCR/ocrwb.txt");
    init_sdl();
    SDL_Surface *image = load__image(filepath);
    image = black_and_white(image);
    DrawRedLines(image);
    int BlocCount = CountBlocs(image);
    SDL_Surface ***chars = malloc(sizeof(SDL_Surface **) * BlocCount);
    SDL_Surface **blocs = malloc(sizeof(SDL_Surface *) * BlocCount);
    int *charslen = DivideIntoBlocs(image, blocs, chars, BlocCount);
    SDL_SaveBMP(image, "segmentation.bmp");
    for (int j = 0; j < BlocCount; ++j)
    {
        SDL_FreeSurface(blocs[j]);
    }
    int **chars_matrix = NULL;
    int chars_count = ImageToMatrix(chars, &chars_matrix, charslen, BlocCount);

    char *result = calloc(chars_count, sizeof(char));

    for (size_t index = 0; index < (size_t)chars_count; index++)
    {
        int is_espace = input_image(network, index, &chars_matrix);
        if (!is_espace)
        {
            forward_pass(network);
            size_t index_answer = IndexAnswer(network);
            result[index] = RetrieveChar(index_answer);
        }
        else
        {
            result[index] = ' ';
        }
    }
    SDL_Quit();
    free_network(network);
    printf("%s\n", result);
}

void TNeuralNetwork()
{
    struct network *network =
        initialize_network(28 * 28, 20, 52, "../OCR/ocrwb.txt");
    char *filepath = "img/training/maj/A0.txt\0";
    char expected_result[52] = { 'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E',
                                 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                                 'J', 'j', 'K', 'k', 'L', 'I', 'M', 'm', 'N',
                                 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                                 'S', 's', 'I', 't', 'U', 'u', 'V', 'v', 'W',
                                 'w', 'X', 'x', 'Y', 'y', 'Z', 'z' };
    int total_samples = 52;  // Number of characters in expected_result
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
                ExpectedOutput(network, expected_result[i]);
                InputFromTXT(filepath, network);
                forward_pass(network);
                
                char recognized = RetrieveChar(IndexAnswer(network));
                PrintState(expected_result[i], recognized);
                
                back_propagation(network);
                
                // Calculate error
                for (int o = 0; o < network->number_of_outputs; o++)
                {
                    batch_error += pow(network->goal[o] - network->output_layer[o], 2);
                }
            }
            
            update_weights_and_biases(network);
            epoch_error += batch_error;
        }
        
        // Adaptive learning rate
        adaptive_learning_rate(network);
        
        // Early stopping
        errors[error_index] = epoch_error / total_samples;
        error_index = (error_index + 1) % EARLY_STOPPING_WINDOW;
        
        if (early_stopping(errors, EARLY_STOPPING_WINDOW, EARLY_STOPPING_THRESHOLD))
        {
            printf("Early stopping at epoch %d\n", epoch);
            break;
        }
        
        printf("Epoch %d, Error: %f\n", epoch, epoch_error / total_samples);
    }

    save_network("../OCR/ocrwb.txt", network);
    free_network(network);
}
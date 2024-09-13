#include "training.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../sdl/our_sdl.h"
#include "../segmentation/segmentation.h"
#include "SDL/SDL.h"
#include "SDL/SDL_image.h"
#include "tools.h"
#include "../process/process.h"

TrainingData *prepare_training()
{
    init_sdl();
    const char *base_filepath = "img/training/maj/A0.png";
    char expected_result[52] = {'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E',
                                'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                                'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N',
                                'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                                'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W',
                                'w', 'X', 'x', 'Y', 'y', 'Z', 'z'};

    TrainingData *training_data = malloc(sizeof(TrainingData));
    training_data->data = malloc(52 * sizeof(int ***));

    for (size_t i = 0; i < 52; i++)
    {
        training_data->data[i] = malloc(50 * sizeof(int **));
        for (size_t index = 0; index < 50; index++)
        {
            training_data->data[i][index] = NULL;
            char *filepath = update_path(base_filepath, strlen(base_filepath), expected_result[i], index);
            SDL_Surface *image = load__image(filepath);
            image = black_and_white(image);
            DrawRedLines(image);
            int BlocCount = CountBlocs(image);
            SDL_Surface ***chars = malloc(sizeof(SDL_Surface **) * BlocCount);
            SDL_Surface **blocs = malloc(sizeof(SDL_Surface *) * BlocCount);
            int *charslen = DivideIntoBlocs(image, blocs, chars, BlocCount);

            int len = ImageToMatrix(chars, &training_data->data[i][index], charslen, BlocCount);
            training_data->counts[i][index] = len; // Store the count

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

    return training_data;
}

void free_training_data(TrainingData *training_data)
{
    for (size_t i = 0; i < 52; i++)
    {
        for (size_t index = 0; index < 50; index++)
        {
            int len = training_data->counts[i][index];
            if (training_data->data[i][index] != NULL)
            {
                for (int count = 0; count < len; count++)
                {
                    free(training_data->data[i][index][count]);
                }
                free(training_data->data[i][index]);
            }
        }
        free(training_data->data[i]);
    }
    free(training_data->data);
    free(training_data);
}

#include "ocr.h"
#include "../common.h"
#include "../common.h"
#include "../network/tools.h"
#include "../network/network.h"
#include "../network/cnn.h"
#include "../sdl/our_sdl.h"
#include "../segmentation/segmentation.h"
#include "../process/process.h"
#include "../GUI/gui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>

// Helper to avoid duplication
char* PerformOCR(char *filepath)
{
    if (filepath == NULL) return NULL;

    // Initialize CNN and load its saved weights
    CNN *cnn = init_cnn();
    if (cnn == NULL) return NULL;
    load_cnn(OCR_CNN_WEIGHTS, cnn);

    // Initialize MLP with CNN output size (must match training: FLATTEN_SIZE inputs)
    struct network *network = InitializeNetwork(FLATTEN_SIZE, OCR_HIDDEN_NODES, 52, OCR_MLP_WEIGHTS);
    if (network == NULL)
    {
        free_cnn(cnn);
        return NULL;
    }

    // Initialize SDL and load image
    init_sdl();
    SDL_Surface *image = load_image(filepath);
    if (image == NULL)
    {
        freeNetwork(network);
        free_cnn(cnn);
        SDL_Quit();
        return NULL;
    }

    // Process image
    image = black_and_white(image);
    DrawRedLines(image);

    int BlocCount = CountBlocs(image);
    SDL_Surface ***chars = malloc(sizeof(SDL_Surface **) * BlocCount);
    SDL_Surface **blocs = malloc(sizeof(SDL_Surface *) * BlocCount);

    if (chars == NULL || blocs == NULL)
    {
        SDL_FreeSurface(image);
        free(chars);
        free(blocs);
        freeNetwork(network);
        free_cnn(cnn);
        SDL_Quit();
        return NULL;
    }

    int *charslen = DivideIntoBlocs(image, blocs, chars, BlocCount);
    if (charslen == NULL)
    {
        SDL_FreeSurface(image);
        free(chars);
        free(blocs);
        freeNetwork(network);
        free_cnn(cnn);
        SDL_Quit();
        return NULL;
    }

    // Save segmented image
    SDL_SaveBMP(image, "segmentation.bmp");

    // Free bloc surfaces
    for (int j = 0; j < BlocCount; ++j)
    {
        if (blocs[j] != NULL)
            SDL_FreeSurface(blocs[j]);
    }

    // Convert image to matrix for neural network processing
    int **chars_matrix = NULL;
    int chars_count = ImageToMatrix(chars, &chars_matrix, charslen, BlocCount);

    char *result = calloc(chars_count + 1, sizeof(char));
    if (result == NULL)
    {
        SDL_FreeSurface(image);
        free(chars);
        free(blocs);
        free(charslen);
        freeNetwork(network);
        free_cnn(cnn);
        SDL_Quit();
        return NULL;
    }

    // Process each character
    for (int index = 0; index < chars_count; index++)
    {
        // Check for space (all-zero pixel matrix)
        int is_space = 1;
        for (int i = 0; i < IMAGE_PIXELS; i++)
        {
            if (chars_matrix[index][i] == 1) { is_space = 0; break; }
        }

        if (!is_space)
        {
            // Build double array from int array
            double img_double[IMAGE_PIXELS];
            for (int i = 0; i < IMAGE_PIXELS; i++)
                img_double[i] = (double)chars_matrix[index][i];

            // Run CNN forward pass — writes directly into MLP input layer, no malloc
            cnn_forward(cnn, img_double, network->input_layer);

            // Run MLP forward pass
            forward_pass(network);
            size_t index_answer = IndexAnswer(network);
            result[index] = RetrieveChar(index_answer);
        }
        else
        {
            result[index] = ' ';
        }
    }
    result[chars_count] = '\0';

    freeNetwork(network);
    free_cnn(cnn);

    for (int i = 0; i < chars_count; i++)
        free(chars_matrix[i]);
    free(chars_matrix);

    for (int b = 0; b < BlocCount; b++)
    {
        if (chars && chars[b])
        {
            for (int c = 0; c < charslen[b]; c++)
            {
                if (chars[b][c])
                    SDL_FreeSurface(chars[b][c]);
            }
            free(chars[b]);
        }
    }

    free(chars);
    free(blocs);
    free(charslen);

    SDL_FreeSurface(image);
    SDL_Quit();

    return result;
}

void StartOCR(char *filepath)
{
    if (filepath == NULL)
    {
        errx(1, "Invalid file path");
    }

    char *result = PerformOCR(filepath);
    if (result == NULL)
    {
        errx(1, "OCR Failed");
    }

    printf("OCR Result: %s\n", result);
    free(result);
}

int OCR(GtkButton *button, GtkTextBuffer *buffer)
{
    UNUSED(button);
    // Access global filename from gui.c or pass it. 
    // For now, assuming extern or we need to fix this dependency.
    // Let's assume filename is available via gui.h or we need to include it.
    // Actually, gui.c has `gchar *filename = "";` as global.
    // We should probably move `filename` to a shared state or access it.
    
    // For this refactor, I will assume `filename` is accessible if I include `gui.h` 
    // BUT `gui.h` might not declare it as extern.
    // I will check `gui.h` later. For now, I'll declare it extern here.
    extern gchar *filename;
    extern char *text;

    if (filename == NULL || strlen(filename) == 0)
    {
        g_print("No file selected!\n");
        return EXIT_FAILURE;
    }

    char *result = PerformOCR((char*)filename);
    if (result == NULL)
    {
        g_print("OCR Failed!\n");
        return EXIT_FAILURE;
    }

    g_print("OCR Done !\n");
    g_print("Result: %s\n", result);

    // Free previous OCR result if there was one
    if (text != NULL)
        free(text);

    // text takes ownership of result; do not free result separately
    text = result;
    gtk_text_buffer_set_text(buffer, text, strlen(text));

    return EXIT_SUCCESS;
}

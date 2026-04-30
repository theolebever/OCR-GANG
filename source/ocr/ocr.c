#include "ocr.h"
#include "../common.h"
#include "../network/tools.h"
#include "../network/network.h"
#include "../network/cnn.h"
#include "../sdl/our_sdl.h"
#include "../segmentation/segmentation.h"
#include "../process/process.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <err.h>

typedef struct
{
    CNN *cnn;
    struct network *network;
    SDL_Surface *image;
    SDL_Surface ***chars;
    SDL_Surface **blocs;
    int *charslen;
    int **chars_matrix;
    int bloc_count;
    int chars_count;
} OcrContext;

static void free_segmented_chars(SDL_Surface ***chars, int *charslen, int bloc_count)
{
    if (chars == NULL) return;

    for (int b = 0; b < bloc_count; b++)
    {
        if (chars[b] == NULL) continue;

        for (int c = 0; charslen != NULL && c < charslen[b]; c++)
        {
            if (chars[b][c] != NULL)
                SDL_FreeSurface(chars[b][c]);
        }
        free(chars[b]);
    }
    free(chars);
}

static void free_ocr_context(OcrContext *ctx)
{
    if (ctx == NULL) return;

    if (ctx->chars_matrix != NULL)
    {
        for (int i = 0; i < ctx->chars_count; i++)
            free(ctx->chars_matrix[i]);
        free(ctx->chars_matrix);
    }

    free_segmented_chars(ctx->chars, ctx->charslen, ctx->bloc_count);

    if (ctx->blocs != NULL)
    {
        for (int b = 0; b < ctx->bloc_count; b++)
            if (ctx->blocs[b] != NULL)
                SDL_FreeSurface(ctx->blocs[b]);
        free(ctx->blocs);
    }

    free(ctx->charslen);

    if (ctx->image != NULL)
        SDL_FreeSurface(ctx->image);
    freeNetwork(ctx->network);
    free_cnn(ctx->cnn);
    SDL_Quit();
}

static char recognize_matrix(CNN *cnn, struct network *network, int *matrix)
{
    double input[IMAGE_PIXELS];
    for (int i = 0; i < IMAGE_PIXELS; i++)
        input[i] = (double)matrix[i];

    cnn_forward(cnn, input, network->input_layer);
    forward_pass(network);
    return RetrieveChar(IndexAnswer(network));
}

static char *build_ocr_result(OcrContext *ctx)
{
    int newline_count = ctx->bloc_count > 0 ? ctx->bloc_count - 1 : 0;
    char *result = calloc(ctx->chars_count + newline_count + 1, sizeof(char));
    if (result == NULL)
        return NULL;

    int matrix_idx = 0;
    int out_idx = 0;
    for (int b = 0; b < ctx->bloc_count; b++)
    {
        for (int c = 0; c < ctx->charslen[b]; c++)
        {
            int *matrix = ctx->chars_matrix[matrix_idx++];
            result[out_idx++] = matrix == NULL
                ? ' '
                : recognize_matrix(ctx->cnn, ctx->network, matrix);
        }
        if (b < ctx->bloc_count - 1)
            result[out_idx++] = '\n';
    }
    result[out_idx] = '\0';
    return result;
}

char *PerformOCR(const char *filepath)
{
    if (filepath == NULL) return NULL;

    OcrContext ctx;
    memset(&ctx, 0, sizeof(ctx));

    // Initialize CNN and load its saved weights. Fall back to fresh init on failure.
    ctx.cnn = init_cnn();
    if (ctx.cnn == NULL) return NULL;
    if (!load_cnn(OCR_CNN_WEIGHTS, ctx.cnn))
        cnn_reset(ctx.cnn);

    // Initialize MLP with CNN output size (must match training: FLATTEN_SIZE inputs)
    ctx.network = InitializeNetwork(FLATTEN_SIZE, OCR_HIDDEN_NODES, 52, OCR_MLP_WEIGHTS);
    if (ctx.network == NULL)
    {
        free_ocr_context(&ctx);
        return NULL;
    }
    set_training_mode(ctx.network, 0);

    // Initialize SDL and load image
    init_sdl();
    ctx.image = load_image(filepath);
    if (ctx.image == NULL)
    {
        free_ocr_context(&ctx);
        return NULL;
    }

    // Process image
    ctx.image = black_and_white(ctx.image);
    DrawRedLines(ctx.image);

    ctx.bloc_count = CountBlocs(ctx.image);
    ctx.chars = calloc(ctx.bloc_count, sizeof(SDL_Surface **));
    ctx.blocs = calloc(ctx.bloc_count, sizeof(SDL_Surface *));

    if (ctx.chars == NULL || ctx.blocs == NULL)
    {
        free_ocr_context(&ctx);
        return NULL;
    }

    ctx.charslen = DivideIntoBlocs(ctx.image, ctx.blocs, ctx.chars, ctx.bloc_count);
    if (ctx.charslen == NULL)
    {
        free_ocr_context(&ctx);
        return NULL;
    }

    // Save segmented image
    SDL_SaveBMP(ctx.image, "segmentation.bmp");

    // Convert image to matrix for neural network processing.
    // chars_matrix entries are NULL for spaces (preserved positionally).
    ctx.chars_count = ImageToMatrix(ctx.chars, &ctx.chars_matrix, ctx.charslen, ctx.bloc_count);
    char *result = build_ocr_result(&ctx);

    free_ocr_context(&ctx);

    return result;
}

void StartOCR(const char *filepath)
{
    if (filepath == NULL)
        errx(1, "Invalid file path");

    char *result = PerformOCR(filepath);
    if (result == NULL)
        errx(1, "OCR Failed");

    printf("OCR Result: %s\n", result);
    free(result);
}

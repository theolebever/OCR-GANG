#include "segmentation.h"
#include "../common.h"

#include <stdio.h>

#include "../sdl/our_sdl.h"
#include "err.h"

void DrawRedLines(SDL_Surface *image)
{
    Uint32 pixel;
    Uint8 red;
    int is_empty; // is_empty is boolean
    for (int i = 0; i < image->h; i++)
    {
        is_empty = 1;
        for (int j = 0; j < image->w; j++)
        {
            pixel = get_pixel(image, j, i);
            red = getRed(pixel, image->format);

            if (red == 0)
                is_empty = 0;
        }
        if (is_empty)
        {
            for (int j = 0; j < image->w; j++)
            {
                put_pixel(image, j, i, SDL_MapRGB(image->format, 128, 0, 0));
            }
        }
    }
}

int CountBlocs(SDL_Surface *image)
{
    Uint32 pixel;
    Uint8 red;
    int Count = 0; // Count each bloc in image
    int is_empty; // is_empty is boolean
    int y_max;
    for (int i = 0; i < image->h; i++)
    {
        pixel = get_pixel(image, 0, i);
        red = getRed(pixel, image->format);
        if (red == 0 || red == 255)
        {
            is_empty = 1;
            y_max = i;
            while (is_empty && y_max < image->h)
            {
                y_max++;
                pixel = get_pixel(image, 0, y_max);
                red = getRed(pixel, image->format);
                if (red == 128)
                    is_empty = 0;
            }
            Count++;
            i = y_max;
        }
    }
    return Count;
}

int SizeOfChar(SDL_Surface *bloc)
{
    Uint32 pixel;
    Uint8 red;
    int charSize = 20;
    int is_empty;
    int charx_max;
    int charXmin;
    for (int i = 0; i < bloc->w; i++)
    {
        pixel = get_pixel(bloc, i, 0);
        red = getRed(pixel, bloc->format);
        if (red == 0 || red == 255)
        {
            is_empty = 1;
            charXmin = i;
            charx_max = i;
            while (is_empty && charx_max < bloc->w)
            {
                charx_max++;
                pixel = get_pixel(bloc, charx_max, 0);
                red = getRed(pixel, bloc->format);
                if (red == 128)
                    is_empty = 0;
            }
            charSize = (charSize + charx_max - charXmin) / 2;
            i = charx_max;
        }
    }
    return charSize;
}

int *DivideIntoBlocs(SDL_Surface *image,
                     SDL_Surface **blocs,
                     SDL_Surface ***chars,
                     int Len)
{
    int *CharsCount = calloc(Len, sizeof(int));
    if (!CharsCount)
        errx(1, "OOM DivideIntoBlocs");

    int Count = 0;

    for (int y = 0; y < image->h && Count < Len; y++)
    {
        Uint32 pixel = get_pixel(image, 0, y);
        Uint8 red = getRed(pixel, image->format);

        if (red == 0 || red == 255)
        {
            int y_start = y;
            while (y < image->h)
            {
                pixel = get_pixel(image, 0, y);
                red = getRed(pixel, image->format);
                if (red == 128)
                    break;
                y++;
            }

            SDL_Rect bloc = {0, y_start, image->w, y - y_start};
            blocs[Count] = SDL_CreateRGBSurface(0, bloc.w, bloc.h, 32, 0,0,0,0);
            if (!blocs[Count])
                errx(1, "SDL_CreateRGBSurface failed");

            SDL_BlitSurface(image, &bloc, blocs[Count], NULL);
            DrawLinesUp(blocs[Count]);

            int nb_chars = CountChars(blocs[Count]);
            CharsCount[Count] = nb_chars;
            chars[Count] = calloc(nb_chars, sizeof(SDL_Surface *));
            if (!chars[Count])
                errx(1, "OOM chars");

            int c = 0;
            for (int x = 0; x < blocs[Count]->w && c < nb_chars; x++)
            {
                pixel = get_pixel(blocs[Count], x, 0);
                red = getRed(pixel, blocs[Count]->format);

                if (red == 0 || red == 255)
                {
                    int x_start = x;
                    while (x < blocs[Count]->w)
                    {
                        pixel = get_pixel(blocs[Count], x, 0);
                        red = getRed(pixel, blocs[Count]->format);
                        if (red == 128)
                            break;
                        x++;
                    }

                    SDL_Rect chr = {x_start, 0, x - x_start, blocs[Count]->h};
                    int size = chr.w > chr.h ? chr.w : chr.h;
                    SDL_Rect center = {
                        size/2 - chr.w/2,
                        size/2 - chr.h/2,
                        chr.w,
                        chr.h
                    };

                    chars[Count][c] =
                        SDL_CreateRGBSurface(0, size, size, 32,0,0,0,0);
                    SDL_FillRect(chars[Count][c], NULL, SDL_MapRGB(chars[Count][c]->format,255,255,255));
                    SDL_BlitSurface(blocs[Count], &chr, chars[Count][c], &center);
                    c++;
                }
            }
            Count++;
        }
    }
    return CharsCount;
}


void DrawLinesUp(SDL_Surface *image)
{
    Uint32 pixel;
    Uint8 red;
    int is_empty;
    for (int i = 0; i < image->w; i++)
    {
        is_empty = 1;
        for (int j = 0; j < image->h; j++)
        {
            pixel = get_pixel(image, i, j);
            red = getRed(pixel, image->format);
            if (red == 0)
                is_empty = 0;
        }
        if (is_empty)
        {
            for (int j = 0; j < image->h; j++)
            {
                put_pixel(image, i, j, SDL_MapRGB(image->format, 128, 0, 0));
            }
        }
    }
}

int CountChars(SDL_Surface *bloc)
{
    Uint32 pixel;
    Uint8 red;
    int Count = 0;
    int is_empty;
    int x_max;
    int spaceSize = (SizeOfChar(bloc) / 4) * 3;
    int currentspaceSize = 0;
    char insertspace = 1;
    for (int i = 0; i < bloc->w; i++)
    {
        pixel = get_pixel(bloc, i, 0);
        red = getRed(pixel, bloc->format);
        currentspaceSize++;
        if (red == 0 || red == 255)
        {
            insertspace = 1;
            currentspaceSize = 0;
            is_empty = 1;
            x_max = i;
            while (is_empty && x_max < bloc->w)
            {
                x_max++;
                pixel = get_pixel(bloc, x_max, 0);
                red = getRed(pixel, bloc->format);

                if (red == 128)
                {
                    is_empty = 0;
                }
            }
            Count++;
            i = x_max;
        }
        if (insertspace && Count != 0 && currentspaceSize == spaceSize)
        {
            insertspace = 0;
            for (int j = 0; j < bloc->h; j++)
            {
                put_pixel(bloc, i, j, SDL_MapRGB(bloc->format, 128, 128, 0));
            }
            Count++;
        }
    }
    return Count;
}

int *Resize1(int *mat, int fx, int fy, int sx, int sy)
{
    int *res = malloc(sizeof(int) * fx * fy);
    if (!res)
        errx(1, "OOM Resize1");

    double cx = (double)sx / fx;
    double cy = (double)sy / fy;

    for (int y = 0; y < fy; y++)
        for (int x = 0; x < fx; x++)
            res[y * fx + x] =
                mat[(int)(y * cy) * sx + (int)(x * cx)];

    return res;
}


int ImageToMatrix(SDL_Surface ***chars,
                  int ***chars_matrix,
                  int *len,
                  int BlocNumber)
{
    int total = 0;

    /* Comptage réel des surfaces valides */
    for (int b = 0; b < BlocNumber; b++)
        for (int c = 0; c < len[b]; c++)
            if (chars[b][c])
                total++;

    *chars_matrix = malloc(sizeof(int *) * total);
    if (!*chars_matrix)
        errx(1, "OOM ImageToMatrix");

    int index = 0;

    for (int b = 0; b < BlocNumber; b++)
    {
        for (int c = 0; c < len[b]; c++)
        {
            SDL_Surface *s = chars[b][c];
            if (!s)
                continue; // ← ESPACE, on ignore

            int w = s->w;
            int h = s->h;

            if (w <= 0 || h <= 0)
                continue;

            int *raw = malloc(sizeof(int) * w * h);
            if (!raw)
                errx(1, "OOM raw");

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    Uint32 px = get_pixel(s, x, y);
                    Uint8 r, g, b;
                    SDL_GetRGB(px, s->format, &r, &g, &b);
                    raw[y * w + x] = (r < 128) ? 1 : 0;
                }
            }

            int *resized = Resize1(raw, IMAGE_SIZE, IMAGE_SIZE, w, h);
            free(raw);

            (*chars_matrix)[index++] = resized;
        }
    }

    return index;
}

void SaveMatrix(int **chars_matrix, char *filename)
{
    FILE *matrix = fopen(filename, "w");
    size_t size = IMAGE_SIZE;
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            fprintf(matrix, "%lf", (double)chars_matrix[0][i * size + j]);
        }
        fprintf(matrix, "\n");
    }
    fclose(matrix);
}

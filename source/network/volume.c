#include "volume.h"
#include <stdlib.h>

// Helper function to create a new Volume
Volume *create_volume(int width, int height, int depth)
{
    Volume *v = (Volume *)malloc(sizeof(Volume));
    v->width = width;
    v->height = height;
    v->depth = depth;
    v->data = (float *)calloc(width * height * depth, sizeof(float));
    return v;
}

// Helper function for freeing volumes
void free_volume(Volume *vol)
{
    if (vol)
    {
        free(vol->data);
        free(vol);
    }
}
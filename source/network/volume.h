#ifndef VOLUME_H_
#define VOLUME_H

// Structure to hold 3D volume of data
typedef struct
{
    int width, height, depth;
    float *data;
} Volume;

// Helper function to create a new Volume
Volume *create_volume(int width, int height, int depth);

// Helper function for freeing volumes
void free_volume(Volume *vol);

#endif // !VOLUME_H_
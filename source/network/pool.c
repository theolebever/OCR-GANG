#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "pool.h"

// Function to add a pooling layer
void add_pool_layer(Network *net, int layer_index, int in_w, int in_h, int in_d,
                    int pool_w, int pool_h, int stride)
{
    PoolLayer *pool = (PoolLayer *)malloc(sizeof(PoolLayer));
    pool->base.type = LAYER_POOL;
    pool->base.input = (Volume *)malloc(sizeof(Volume));
    pool->base.output = (Volume *)malloc(sizeof(Volume));
    pool->pool_width = pool_w;
    pool->pool_height = pool_h;
    pool->stride = stride;

    pool->base.input->width = in_w;
    pool->base.input->height = in_h;
    pool->base.input->depth = in_d;
    pool->base.input->data = (float *)calloc(in_w * in_h * in_d, sizeof(float));

    int output_w = (in_w - pool_w) / stride + 1;
    int output_h = (in_h - pool_h) / stride + 1;
    pool->base.output->width = output_w;
    pool->base.output->height = output_h;
    pool->base.output->depth = in_d;
    pool->base.output->data = (float *)calloc(output_w * output_h * in_d, sizeof(float));

    pool->max_indices = (int *)malloc(output_w * output_h * in_d * sizeof(int));

    if (!pool->base.input->data || !pool->base.output->data || !pool->max_indices)
    {
        perror("Memory allocation failed in add_pool_layer");
        exit(EXIT_FAILURE);
    }

    pool->base.forward = (void (*)(Layer *, Volume *))pool_forward;
    pool->base.backward = (void (*)(Layer *, float *))pool_backward;

    net->layers[layer_index] = (Layer *)pool;
}

void pool_forward(PoolLayer *layer, Volume *input)
{
    int out_w = layer->base.output->width;
    int out_h = layer->base.output->height;

    for (int d = 0; d < input->depth; d++)
    {
        for (int y = 0; y < out_h; y++)
        {
            for (int x = 0; x < out_w; x++)
            {
                float max_val = -INFINITY;
                int max_idx = -1;
                for (int py = 0; py < layer->pool_height; py++)
                {
                    for (int px = 0; px < layer->pool_width; px++)
                    {
                        int ix = x * layer->stride + px;
                        int iy = y * layer->stride + py;
                        if (ix < input->width && iy < input->height)
                        { // Bounds check
                            float val = input->data[(iy * input->width + ix) * input->depth + d];
                            if (val > max_val)
                            {
                                max_val = val;
                                max_idx = (iy * input->width + ix) * input->depth + d;
                            }
                        }
                    }
                }
                layer->base.output->data[(y * out_w + x) * input->depth + d] = max_val;
                layer->max_indices[(y * out_w + x) * input->depth + d] = max_idx;
            }
        }
    }
}

void pool_backward(PoolLayer *layer, float *upstream_gradient)
{
    int out_w = layer->base.output->width;
    int out_h = layer->base.output->height;
    int in_w = layer->base.input->width;
    int in_h = layer->base.input->height;
    int depth = layer->base.input->depth;

    // Initialize input gradients to zero
    float *input_gradients = calloc(in_w * in_h * depth, sizeof(float));

    for (int d = 0; d < depth; d++)
    {
        for (int y = 0; y < out_h; y++)
        {
            for (int x = 0; x < out_w; x++)
            {
                int out_idx = (y * out_w + x) * depth + d;
                int max_idx = layer->max_indices[out_idx];
                input_gradients[max_idx] += upstream_gradient[out_idx];
            }
        }
    }

    // Store input gradients for the previous layer
    memcpy(layer->base.input->data, input_gradients, in_w * in_h * depth * sizeof(float));
    free(input_gradients);
}
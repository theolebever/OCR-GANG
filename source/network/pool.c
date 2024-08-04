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

    net->layers[layer_index] = (Layer *)pool;
}

// Forward pass for pooling layer
void pool_forward(PoolLayer *layer, Volume *input)
{
    int in_w = input->width;
    int in_h = input->height;
    int in_d = input->depth;
    int out_w = layer->base.output->width;
    int out_h = layer->base.output->height;
    int pool_w = layer->pool_width;
    int pool_h = layer->pool_height;
    int stride = layer->stride;

    for (int d = 0; d < in_d; d++)
    {
        for (int y = 0; y < out_h; y++)
        {
            for (int x = 0; x < out_w; x++)
            {
                float max_val = -INFINITY;
                int max_idx = -1;
                for (int py = 0; py < pool_h; py++)
                {
                    for (int px = 0; px < pool_w; px++)
                    {
                        int in_y = y * stride + py;
                        int in_x = x * stride + px;
                        if (in_y < in_h && in_x < in_w)
                        {
                            int input_idx = (in_y * in_w + in_x) * in_d + d;
                            if (input->data[input_idx] > max_val)
                            {
                                max_val = input->data[input_idx];
                                max_idx = input_idx;
                            }
                        }
                    }
                }
                int out_idx = (y * out_w + x) * in_d + d;
                layer->base.output->data[out_idx] = max_val;
                layer->max_indices[out_idx] = max_idx;
            }
        }
    }
}

void pool_backward(Layer *layer, float *upstream_gradient)
{
    PoolLayer *pool = (PoolLayer *)layer;
    int in_w = layer->input->width;
    int in_h = layer->input->height;
    int in_d = layer->input->depth;
    int out_w = layer->output->width;
    int out_h = layer->output->height;

    // Initialize input gradients to zero
    float *input_gradients = calloc(in_w * in_h * in_d * out_w * out_h, sizeof(float));

    for (int d = 0; d < in_d; d++)
    {
        for (int y = 0; y < out_h; y++)
        {
            for (int x = 0; x < out_w; x++)
            {
                int out_idx = (y * out_w + x) * in_d + d;
                int max_idx = pool->max_indices[out_idx];
                float up = upstream_gradient[out_idx];
                input_gradients[max_idx] += up;
            }
        }
    }

    // Store input gradients for the previous layer
    memcpy(layer->input->data, input_gradients, in_w * in_h * in_d * sizeof(float));
    free(input_gradients);
}
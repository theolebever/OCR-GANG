#include "early_stop.h"
#include "OCR.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

EarlyStopping *init_early_stopping(Network *net, int patience)
{
    EarlyStopping *es = malloc(sizeof(EarlyStopping));
    es->best_val_loss = INFINITY;
    es->patience = patience;
    es->wait = 0;
    es->best_epoch = 0;

    // Count total parameters
    int total_params = 0;
    for (int i = 0; i < net->num_layers; i++)
    {
        if (net->layers[i]->type == LAYER_CONV)
        {
            ConvLayer *conv = (ConvLayer *)net->layers[i];
            total_params += conv->filter_width * conv->filter_height * conv->base.input->depth * conv->num_filters;
            total_params += conv->num_filters; // biases
        }
        else if (net->layers[i]->type == LAYER_FC)
        {
            FCLayer *fc = (FCLayer *)net->layers[i];
            total_params += fc->input_size * fc->output_size;
            total_params += fc->output_size; // biases
        }
    }

    es->best_params = malloc(total_params * sizeof(float));
    return es;
}

void free_early_stopping(EarlyStopping *es)
{
    free(es->best_params);
    free(es);
}

int should_stop(EarlyStopping *es, float val_loss, Network *net, int epoch)
{
    if (val_loss < es->best_val_loss)
    {
        es->best_val_loss = val_loss;
        es->wait = 0;
        es->best_epoch = epoch;

        // Save best parameters
        int idx = 0;
        for (int i = 0; i < net->num_layers; i++)
        {
            if (net->layers[i]->type == LAYER_CONV)
            {
                ConvLayer *conv = (ConvLayer *)net->layers[i];
                int weights_size = conv->filter_width * conv->filter_height * conv->base.input->depth * conv->num_filters;
                memcpy(es->best_params + idx, conv->weights, weights_size * sizeof(float));
                idx += weights_size;
                memcpy(es->best_params + idx, conv->biases, conv->num_filters * sizeof(float));
                idx += conv->num_filters;
            }
            else if (net->layers[i]->type == LAYER_FC)
            {
                FCLayer *fc = (FCLayer *)net->layers[i];
                int weights_size = fc->input_size * fc->output_size;
                memcpy(es->best_params + idx, fc->weights, weights_size * sizeof(float));
                idx += weights_size;
                memcpy(es->best_params + idx, fc->biases, fc->output_size * sizeof(float));
                idx += fc->output_size;
            }
        }
    }
    else
    {
        es->wait++;
        if (es->wait >= es->patience)
        {
            printf("Early stopping at epoch %d\n", epoch);
            return 1;
        }
    }
    return 0;
}
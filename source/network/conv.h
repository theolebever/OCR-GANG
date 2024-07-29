#ifndef CONV_H_
#define CONV_H_

#include "OCR.h"

void add_conv_layer(Network *net, int layer_index, int in_w, int in_h, int in_d,
                    int filter_w, int filter_h, int num_filters);
void conv_forward(Layer *layer, Volume *input);
void conv_backward(Layer *layer, float *upstream_gradient);

#endif // !CONV_H_
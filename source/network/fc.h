#ifndef FC_H_
#define FC_H_

#include "OCR.h"

void add_fc_layer(Network *net, int layer_index, int input_size, int output_size);
void fc_forward(FCLayer *layer, Volume *input);
void fc_backward(Layer *layer, float *upstream_gradient);

#endif // !FC_H_
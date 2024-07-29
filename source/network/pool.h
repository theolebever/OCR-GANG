#ifndef POOL_H_
#define POOL_H_

#include "OCR.h"

void add_pool_layer(Network *net, int layer_index, int in_w, int in_h, int in_d,
                    int pool_w, int pool_h, int stride);
void pool_forward(PoolLayer *layer, Volume *input);
void pool_backward(PoolLayer *layer, float *upstream_gradient);

#endif // !POOL_H_
#include "OCR.h"

#include <math.h>

typedef struct {
    float *best_params;
    float best_val_loss;
    int patience;
    int wait;
    int best_epoch;
} EarlyStopping;

EarlyStopping *init_early_stopping(Network *net, int patience);
void free_early_stopping(EarlyStopping *es);
int should_stop(EarlyStopping *es, float val_loss, Network *net, int epoch);
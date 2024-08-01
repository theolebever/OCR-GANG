#ifndef ADAM_H_
#define ADAM_H_

#define MAX_GRAD_NORM 5.0

typedef struct
{
    float *m; // First moment vector
    float *v; // Second moment vector
    float beta1;
    float beta2;
    float epsilon;
    int t; // Time step
} AdamOptimizer;

AdamOptimizer *init_adam(int param_count, float beta1, float beta2, float epsilon);
void free_adam(AdamOptimizer *adam);
void adam_update(AdamOptimizer *adam, float *params, float *grads, int param_count, float learning_rate);

#endif // !ADAM_H_
#include "adam.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

AdamOptimizer *init_adam(int param_count, float beta1, float beta2, float epsilon)
{
    AdamOptimizer *adam = malloc(sizeof(AdamOptimizer));
    if (!adam)
    {
        perror("Failed to allocate AdamOptimizer");
        exit(EXIT_FAILURE);
    }
    adam->m = calloc(param_count, sizeof(float));
    if (!adam->m)
    {
        perror("Failed to allocate memory for m");
        exit(EXIT_FAILURE);
    }
    adam->v = calloc(param_count, sizeof(float));
    if (!adam->v)
    {
        perror("Failed to allocate memory for v");
        exit(EXIT_FAILURE);
    }
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->epsilon = epsilon;
    adam->t = 0;
    return adam;
}

void adam_update(AdamOptimizer *adam, float *params, float *grads, int param_count, float learning_rate)
{
    adam->t++;
    float beta1_pow_t = powf(adam->beta1, adam->t);
    float beta2_pow_t = powf(adam->beta2, adam->t);
    float lr_t = learning_rate * sqrtf(1.0f - beta2_pow_t) / (1.0f - beta1_pow_t);

    for (int i = 0; i < param_count; i++)
    {
        adam->m[i] = adam->beta1 * adam->m[i] + (1.0f - adam->beta1) * grads[i];
        adam->v[i] = adam->beta2 * adam->v[i] + (1.0f - adam->beta2) * grads[i] * grads[i];

        float m_hat = adam->m[i] / (1.0f - beta1_pow_t);
        float v_hat = adam->v[i] / (1.0f - beta2_pow_t);

        params[i] -= lr_t * m_hat / (sqrtf(v_hat) + adam->epsilon);
    }
}

void free_adam(AdamOptimizer *adam)
{
    free(adam->m);
    free(adam->v);
    free(adam);
}
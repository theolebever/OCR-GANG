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
    // Gradient clipping
    float grad_norm = 0.0;
    for (int i = 0; i < param_count; i++)
    {
        grad_norm += grads[i] * grads[i];
    }
    grad_norm = sqrt(grad_norm);

    if (grad_norm > MAX_GRAD_NORM)
    {
        float scale = MAX_GRAD_NORM / (grad_norm + 1e-6);
        for (int i = 0; i < param_count; i++)
        {
            grads[i] *= scale;
        }
    }

    adam->t++;
    float lr_t = learning_rate * sqrt(1 - pow(adam->beta2, adam->t)) / (1 - pow(adam->beta1, adam->t));

    for (int i = 0; i < param_count; i++)
    {
        adam->m[i] = adam->beta1 * adam->m[i] + (1 - adam->beta1) * grads[i];
        adam->v[i] = adam->beta2 * adam->v[i] + (1 - adam->beta2) * grads[i] * grads[i];

        float m_hat = adam->m[i] / (1 - pow(adam->beta1, adam->t));
        float v_hat = adam->v[i] / (1 - pow(adam->beta2, adam->t));

        params[i] -= lr_t * m_hat / (sqrt(v_hat) + adam->epsilon);
    }
}

void free_adam(AdamOptimizer *adam)
{
    free(adam->m);
    free(adam->v);
    free(adam);
}
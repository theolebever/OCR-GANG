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

void free_adam(AdamOptimizer *adam)
{
    free(adam->m);
    free(adam->v);
    free(adam);
}
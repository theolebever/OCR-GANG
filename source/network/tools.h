#ifndef TOOLS_H_
#define TOOLS_H_

#include <stddef.h>
#include "../network/network.h"

void progressBar(int step, int nb);
double sigmoid(double x);
double dSigmoid(double x);
float expo(float x);
double init_weight();

int file_exists(const char *filename);
int file_empty(const char *filename);

void save_network(const char *filename, struct network *network);

void load_network(const char *filename, struct network *network);

void shuffle(int *array, size_t n);

char retrieve_char(size_t val);

size_t index_answer(struct network *net);

void expected_output(struct network *network, char c);

size_t expected_pos(char c);

char *update_path(const char *filepath, size_t len, char c, size_t index);

void input_from_txt(char *filepath, struct network *net);

void prepare_training();

#endif

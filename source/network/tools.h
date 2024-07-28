#ifndef TOOLS_H_
#define TOOLS_H_

#include "OCR.h"
#include "XOR.h"
#include "SDL/SDL.h"
#include "SDL/SDL_image.h"

void progressBar(int step, int nb);
double sigmoid(double x);
double sigmoid_derivative(double x);
float expo(float x);
double init_weight();
int file_exists(const char *filename);
int file_empty(const char *filename);
void save_network(const char *filename, struct fnn *net);
void load_network(const char *filename, struct fnn *net);
void shuffle(int *array, size_t n);
char retrieve_char(size_t val);
size_t expected_pos(char c);
char *update_path(const char *filepath, size_t len, char c, size_t index);
void input_from_txt(char *filepath, struct fnn *net);
void prepare_training();
void free_chars_matrix(int **chars_matrix, int chars_count);
void free_chars(SDL_Surface ***chars, int *charslen, int BlocCount);
void free_surfaces(SDL_Surface **surfaces, int count);
int input_image(float *input_layer, const int *image_data, size_t image_size);
void read_binary_image(const char *filepath, double *arr);
void xavier_init(float *weights, int fan_in, int fan_out);

#endif

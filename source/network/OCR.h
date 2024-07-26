#ifndef OCR_H_
#define OCR_H_

void perform_ocr(char *filepath);
void train_neural_network();
void free_chars_matrix(int **chars_matrix, int chars_count);
void free_chars(SDL_Surface ***chars, int *charslen, int BlocCount);
void free_surfaces(SDL_Surface **surfaces, int count);
#endif
#ifndef COMMON_H
#define COMMON_H

// Terminal color codes
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KWHT "\x1B[37m"

// Suppress unused parameter warnings
#define UNUSED(x) (void)(x)

// Adam optimizer hyperparameters
#define ADAM_BETA1  0.9
#define ADAM_BETA2  0.999
#define ADAM_EPS    1e-8

// File paths
#define XOR_WEIGHTS_PATH   "source/Xor/xorwb.txt"
#define XOR_DATA_PATH      "source/Xor/xordata.txt"
#define OCR_MLP_WEIGHTS    "source/OCR-data/ocrwb.txt"
#define OCR_CNN_WEIGHTS    "source/OCR-data/cnnwb.txt"

// Image processing
#define BW_THRESHOLD       180
#define IMAGE_SIZE         28
#define IMAGE_PIXELS       (IMAGE_SIZE * IMAGE_SIZE)

// Mathematical constants
#define MY_PI              3.14159265358979323846
#define MY_TWO_PI          6.28318530717958647692
#define MY_HALF_PI         1.57079632679489661923
#define MY_INV_LN2         1.4426950408889634
#define MY_LN2             0.6931471805599453

#endif

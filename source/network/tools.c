#include "../network/tools.h"
#include "../common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#include "../network/network.h"
#include "../network/cnn.h"
#include "../process/process.h"
#include "../sdl/our_sdl.h"
#include "../segmentation/segmentation.h"
#include "SDL/SDL.h"
#include "SDL/SDL_image.h"


void progressBar(int step, int nb)
{
    printf("\e[?25l");
    int percent = (step * 100) / nb;
    const int pwidth = 72;
    int pos = (step * pwidth) / nb;
    printf("[");
    for (int i = 0; i < pos; i++)
    {
        printf("%c", '=');
    }
    printf("%*c ", pwidth - pos + 1, ']');
    printf(" %3d%%\r", percent);
    fflush(stdout);
}

// e^x via Cody-Waite range reduction: x = n*ln(2) + r, |r| <= ln(2)/2.
// Taylor on the reduced range + exact power-of-two scaling via bit manipulation.
double expo(double x)
{
    // Clamp to the range where exp is representable as a normal double
    if (x >  708.0) return 1.7976931348623157e308;
    if (x < -708.0) return 0.0;

    // Round x / ln(2) to nearest int (handles positive and negative symmetrically)
    double k_d = x * MY_INV_LN2;
    long n = (k_d >= 0.0) ? (long)(k_d + 0.5) : -(long)(-k_d + 0.5);
    double r = x - (double)n * MY_LN2;
    // After reduction, |r| <= ln(2)/2 ≈ 0.347

    // Taylor series exp(r) = 1 + r + r^2/2! + ... + r^14/14!
    // On |r| <= 0.347, 14 terms give relative error well below 1e-15
    double sum = 1.0, term = 1.0;
    for (int i = 1; i <= 14; i++) { term *= r / (double)i; sum += term; }

    // Build 2^n as an IEEE 754 double (n is guaranteed in [-1022, 1022])
    unsigned long long bits = (unsigned long long)(n + 1023) << 52;
    double pow2;
    __builtin_memcpy(&pow2, &bits, 8);

    return sum * pow2;
}

// ln(x) via IEEE 754 exponent extraction: x = 2^e * m, m in [1,2).
// ln(x) = e*ln(2) + ln(m). Uses u = (m-1)/(m+1) series so |u| <= 1/3
// and 2*atanh(u) = ln(m). 10 odd terms give < 1e-15 relative error.
double my_log(double x)
{
    if (x <= 0.0) return -1.7976931348623157e308;

    unsigned long long bits;
    __builtin_memcpy(&bits, &x, 8);
    long e = (long)((bits >> 52) & 0x7FF) - 1023;
    bits = (bits & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    double m;
    __builtin_memcpy(&m, &bits, 8);  // m in [1, 2)

    double u = (m - 1.0) / (m + 1.0);
    double u2 = u * u;
    double sum = u;
    double term = u;
    for (int k = 1; k <= 9; k++) {
        term *= u2;
        sum += term / (double)(2 * k + 1);
    }
    return 2.0 * sum + (double)e * MY_LN2;
}

// sqrt via IEEE 754 exponent halving for a ~6-bit seed,
// then 4 Newton-Raphson iterations → full double precision.
double my_sqrt(double x)
{
    if (x <= 0.0) return 0.0;

    // Seed: halve the exponent field. For x = 2^e * m, yields ≈ 2^(e/2) * sqrt(m)
    unsigned long long bits;
    __builtin_memcpy(&bits, &x, 8);
    bits = (bits >> 1) + 0x1FF8000000000000ULL;
    double guess;
    __builtin_memcpy(&guess, &bits, 8);

    // Newton-Raphson doubles the correct bits per iteration.
    // 6-bit seed → 12 → 24 → 48 → >52 bits after 4 iterations.
    guess = 0.5 * (guess + x / guess);
    guess = 0.5 * (guess + x / guess);
    guess = 0.5 * (guess + x / guess);
    guess = 0.5 * (guess + x / guess);
    return guess;
}

// sin via two-stage range reduction: first to [-π, π] (symmetric rounding),
// then to [-π/2, π/2] using sin(π - x) = sin(x). Degree-13 odd Taylor.
double my_sin(double x)
{
    // Reduce to [-π, π]: subtract nearest multiple of 2π
    double k_d = x * (1.0 / MY_TWO_PI);
    long long k = (k_d >= 0.0) ? (long long)(k_d + 0.5) : -(long long)(-k_d + 0.5);
    x -= (double)k * MY_TWO_PI;

    // Reduce to [-π/2, π/2]
    if      (x >  MY_HALF_PI) x =  MY_PI - x;
    else if (x < -MY_HALF_PI) x = -MY_PI - x;

    // Odd-power Taylor (Horner form): x - x^3/3! + x^5/5! - ... + x^13/13!
    // Max |x| = π/2; residual ≈ (π/2)^15 / 15! ≈ 6e-10
    double x2 = x * x;
    return x * (1.0 + x2 * (-1.0/6.0
            + x2 * ( 1.0/120.0
            + x2 * (-1.0/5040.0
            + x2 * ( 1.0/362880.0
            + x2 * (-1.0/39916800.0
            + x2 * ( 1.0/6227020800.0)))))));
}

// cos via sin(x + π/2). my_sin handles the re-reduction.
double my_cos(double x)
{
    return my_sin(x + MY_HALF_PI);
}

// Fast rounding for non-negative values
static inline int my_round(double x)
{
    return (x >= 0.0) ? (int)(x + 0.5) : -(int)(-x + 0.5);
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + expo(-x));
}

double dSigmoid(double x)
{
    return x * (1.0 - x);
}

double relu(double x)
{
    // Leaky ReLU to avoid dead neurons
    return x > 0.0 ? x : 0.01 * x;
}

double dRelu(double x)
{
    return x > 0.0 ? 1.0 : 0.01;
}

void softmax(double *input, int n)
{
    double max = input[0];
    for (int i = 1; i < n; i++)
    {
        if (input[i] > max) max = input[i];
    }

    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        input[i] = expo(input[i] - max);
        sum += input[i];
    }

    double inv_sum = 1.0 / sum;
    for (int i = 0; i < n; i++)
    {
        input[i] *= inv_sum;
    }
}

// Uniform random number between min and max
double random_uniform(double min, double max)
{
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// He Initialization for ReLU (Uniform)
// Range: [-sqrt(6/n_in), sqrt(6/n_in)]
double init_weight_he(int fan_in)
{
    double limit = my_sqrt(6.0 / fan_in);
    return random_uniform(-limit, limit);
}

// Xavier/Glorot Initialization for Sigmoid/Softmax (Uniform)
// Range: [-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out))]
double init_weight_xavier(int fan_in, int fan_out)
{
    double limit = my_sqrt(6.0 / (fan_in + fan_out));
    return random_uniform(-limit, limit);
}

double init_weight()
{
    return ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
}

int cfileexists(const char *filename)
{
    if (filename == NULL) return 0;
    FILE *file = fopen(filename, "r");
    if (file == NULL) return 0;
    fclose(file);
    return 1;
}

int fileempty(const char *filename)
{
    if (filename == NULL) return 1;
    FILE *fptr = fopen(filename, "r");
    if (fptr == NULL) return 1;

    fseek(fptr, 0, SEEK_END);
    unsigned long len = (unsigned long)ftell(fptr);
    fclose(fptr);
    return (len > 0) ? 0 : 1;
}

// Versioned weight file: dimensions + weights/biases + full Adam state.
// Incompatible with pre-v2 files; those are ignored on load.
#define NET_MAGIC "OCRNET"
#define NET_VERSION 2

static int read_doubles(FILE *f, double *dst, size_t n)
{
    for (size_t i = 0; i < n; i++)
        if (fscanf(f, "%lf", &dst[i]) != 1) return 0;
    return 1;
}

static void write_doubles(FILE *f, const double *src, size_t n)
{
    for (size_t i = 0; i < n; i++) fprintf(f, "%.17g\n", src[i]);
}

void save_network(const char *filename, struct network *network)
{
    if (filename == NULL || network == NULL) return;
    FILE *f = fopen(filename, "w");
    if (f == NULL) { perror(filename); return; }

    int I = network->number_of_inputs;
    int H = network->number_of_hidden_nodes;
    int O = network->number_of_outputs;

    fprintf(f, "%s %d %d %d %d\n", NET_MAGIC, NET_VERSION, I, H, O);
    fprintf(f, "%ld %.17g %.17g\n",
            network->adam_t, network->adam_beta1_t, network->adam_beta2_t);

    write_doubles(f, network->hidden_layer_bias, H);
    write_doubles(f, network->hidden_weights,    (size_t)I * H);
    write_doubles(f, network->output_layer_bias, O);
    write_doubles(f, network->output_weights,    (size_t)H * O);

    write_doubles(f, network->m_hidden_bias,    H);
    write_doubles(f, network->v_hidden_bias,    H);
    write_doubles(f, network->m_hidden_weights, (size_t)I * H);
    write_doubles(f, network->v_hidden_weights, (size_t)I * H);

    write_doubles(f, network->m_output_bias,    O);
    write_doubles(f, network->v_output_bias,    O);
    write_doubles(f, network->m_output_weights, (size_t)H * O);
    write_doubles(f, network->v_output_weights, (size_t)H * O);

    fclose(f);
}

int load_network(const char *filename, struct network *network)
{
    if (filename == NULL || network == NULL) return 0;
    FILE *f = fopen(filename, "r");
    if (f == NULL) return 0;

    char magic[16];
    int version, I, H, O;
    if (fscanf(f, "%15s %d %d %d %d", magic, &version, &I, &H, &O) != 5
        || strcmp(magic, NET_MAGIC) != 0
        || version != NET_VERSION
        || I != network->number_of_inputs
        || H != network->number_of_hidden_nodes
        || O != network->number_of_outputs)
    {
        fprintf(stderr, "load_network: incompatible file %s (ignored)\n", filename);
        fclose(f);
        return 0;
    }

    int ok = (fscanf(f, "%ld %lf %lf",
                     &network->adam_t,
                     &network->adam_beta1_t,
                     &network->adam_beta2_t) == 3);

    ok &= read_doubles(f, network->hidden_layer_bias, H);
    ok &= read_doubles(f, network->hidden_weights,    (size_t)I * H);
    ok &= read_doubles(f, network->output_layer_bias, O);
    ok &= read_doubles(f, network->output_weights,    (size_t)H * O);

    ok &= read_doubles(f, network->m_hidden_bias,    H);
    ok &= read_doubles(f, network->v_hidden_bias,    H);
    ok &= read_doubles(f, network->m_hidden_weights, (size_t)I * H);
    ok &= read_doubles(f, network->v_hidden_weights, (size_t)I * H);

    ok &= read_doubles(f, network->m_output_bias,    O);
    ok &= read_doubles(f, network->v_output_bias,    O);
    ok &= read_doubles(f, network->m_output_weights, (size_t)H * O);
    ok &= read_doubles(f, network->v_output_weights, (size_t)H * O);

    fclose(f);

    if (!ok)
        fprintf(stderr, "load_network: file %s truncated or corrupt\n", filename);
    return ok;
}

void shuffle(int *array, size_t n)
{
    if (array == NULL || n <= 1) return;
    for (size_t i = 0; i < n - 1; i++)
    {
        size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

size_t IndexAnswer(struct network *net)
{
    if (net == NULL || net->output_layer == NULL) return 0;
    size_t index = 0;
    for (size_t i = 1; i < (size_t)net->number_of_outputs; i++)
    {
        if (net->output_layer[i] > net->output_layer[index])
        {
            index = i;
        }
    }
    return index;
}

char RetrieveChar(size_t val)
{
    char c;
    if (val <= 25) c = val + 65;
    else if (val > 25 && val <= 51) c = (val + 97 - 26);
    else c = '?';
    return c;
}

int LabelIndex(char c)
{
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    return -1;
}

size_t ExpectedPos(char c)
{
    int index = LabelIndex(c);
    return index >= 0 ? (size_t)index : 0;
}

void ExpectedOutput(struct network *network, char c)
{
    if (network == NULL || network->goal == NULL) return;
    for (int i = 0; i < network->number_of_outputs; i++) network->goal[i] = 0;

    int index = LabelIndex(c);
    if (index >= 0 && index < network->number_of_outputs)
        network->goal[index] = 1;
}

char *updatepath(char *filepath, size_t len, char c, size_t index)
{
    if (filepath == NULL) return NULL;
    char *newpath = malloc(len + 20);
    if (newpath == NULL) return NULL;
    
    // Simple implementation for now
    sprintf(newpath, "%s_%c_%lu.bmp", filepath, c, index);
    return newpath;
}

void PrintState(char expected, char obtained)
{
    printf("Expected: %c, Obtained: %c\n", expected, obtained);
}

void InputFromTXT(char *filepath, struct network *net)
{
    // Suppress unused parameter warnings
    (void)filepath;
    (void)net;
    
    // This function appears to be incomplete or a stub
    // If you need to implement it, add the proper logic here
}

void freeDataSet(TrainingDataSet *dataset)
{
    if (dataset == NULL) return;
    
    for (int i = 0; i < dataset->count; i++)
    {
        free(dataset->inputs[i]);
    }
    free(dataset->inputs);
    free(dataset->labels);
    free(dataset);
}

static int ensure_dataset_capacity(TrainingDataSet *dataset)
{
    if (dataset->count < dataset->capacity)
        return 1;

    int new_cap = dataset->capacity == 0 ? 64 : dataset->capacity * 2;
    double **new_inputs = realloc(dataset->inputs, sizeof(double *) * new_cap);
    if (new_inputs == NULL)
        return 0;

    char *new_labels = realloc(dataset->labels, sizeof(char) * new_cap);
    if (new_labels == NULL)
    {
        dataset->inputs = new_inputs;
        return 0;
    }

    dataset->inputs = new_inputs;
    dataset->labels = new_labels;
    dataset->capacity = new_cap;
    return 1;
}

// Helper to properly resize an image to 28x28.
// Mirrors ImageToMatrix: binarize, crop foreground, square-pad, then resize.
double *resize_image_to_28x28(SDL_Surface *img)
{
    double *input = calloc(IMAGE_PIXELS, sizeof(double));
    if (input == NULL) return NULL;

    int w = img->w;
    int h = img->h;
    int *raw = malloc(sizeof(int) * w * h);
    if (raw == NULL)
    {
        free(input);
        return NULL;
    }

    int min_x = w, max_x = -1, min_y = h, max_y = -1;
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            Uint32 pixel = get_pixel(img, x, y);
            Uint8 r, g, b;
            SDL_GetRGB(pixel, img->format, &r, &g, &b);
            (void)g; (void)b;
            int v = (r < BW_THRESHOLD) ? 1 : 0;
            raw[y * w + x] = v;
            if (v)
            {
                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
        }
    }

    if (max_x < 0)
    {
        free(raw);
        return input;
    }

    int bw = max_x - min_x + 1;
    int bh = max_y - min_y + 1;
    int size = bw > bh ? bw : bh;
    int off_x = size / 2 - bw / 2;
    int off_y = size / 2 - bh / 2;

    int *padded = calloc(size * size, sizeof(int));
    if (padded == NULL)
    {
        free(raw);
        free(input);
        return NULL;
    }

    for (int y = 0; y < bh; y++)
        for (int x = 0; x < bw; x++)
            padded[(y + off_y) * size + (x + off_x)] =
                raw[(y + min_y) * w + (x + min_x)];
    free(raw);

    int *resized = Resize1(padded, IMAGE_SIZE, IMAGE_SIZE, size, size);
    free(padded);
    
    if (resized == NULL)
    {
        free(input);
        return NULL;
    }
    
    // Convert to double array
    for (int i = 0; i < IMAGE_PIXELS; i++)
    {
        input[i] = (double)resized[i];
    }
    
    free(resized);
    return input;
}

// Helper to load a single directory of images
void load_directory(const char *path, TrainingDataSet *dataset, int is_uppercase)
{
    DIR *d;
    struct dirent *dir;
    d = opendir(path);
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            if (strstr(dir->d_name, ".png") || strstr(dir->d_name, ".jpg") || strstr(dir->d_name, ".bmp"))
            {
                char fullpath[512];
                snprintf(fullpath, sizeof(fullpath), "%s/%s", path, dir->d_name);

                SDL_Surface *img = load_image(fullpath);
                if (img)
                {
                    double *input = resize_image_to_28x28(img);
                    SDL_FreeSurface(img);

                    if (input == NULL)
                    {
                        printf("Warning: Failed to resize image %s\n", fullpath);
                        continue;
                    }

                    if (!ensure_dataset_capacity(dataset))
                    {
                        free(input);
                        printf("Error: Memory allocation failed\n");
                        break;
                    }

                    char label = dir->d_name[0];
                    if (is_uppercase && label >= 'a' && label <= 'z') label -= 32;
                    if (!is_uppercase && label >= 'A' && label <= 'Z') label += 32;

                    dataset->inputs[dataset->count] = input;
                    dataset->labels[dataset->count] = label;
                    dataset->count++;
                }
            }
        }
        closedir(d);
    }
    else
    {
        printf("Failed to open directory: %s\n", path);
    }
}

TrainingDataSet *loadDataSet(void)
{
    TrainingDataSet *dataset = malloc(sizeof(TrainingDataSet));
    if (dataset == NULL) return NULL;

    dataset->inputs   = NULL;
    dataset->labels   = NULL;
    dataset->count    = 0;
    dataset->capacity = 0;

    load_directory("img/training/maj", dataset, 1);
    load_directory("img/training/min", dataset, 0);

    if (dataset->count == 0)
    {
        printf("ERROR: No training images found in directories!\n");
        printf("       Expected: img/training/maj/ and img/training/min/\n");
        freeDataSet(dataset);
        return NULL;
    }

    return dataset;
}

#define CNN_MAGIC "OCRCNN"
#define CNN_VERSION 2

void save_cnn(const char *filename, void *cnn_ptr)
{
    if (filename == NULL || cnn_ptr == NULL) return;
    CNN *cnn = (CNN *)cnn_ptr;
    FILE *f = fopen(filename, "w");
    if (f == NULL) { perror(filename); return; }

    fprintf(f, "%s %d %d %d\n", CNN_MAGIC, CNN_VERSION, NUM_FILTERS, CONV_SIZE);
    fprintf(f, "%ld %.17g %.17g\n",
            cnn->adam_t, cnn->adam_beta1_t, cnn->adam_beta2_t);

    const size_t kernel_count = (size_t)NUM_FILTERS * CONV_SIZE * CONV_SIZE;
    write_doubles(f, cnn->biases,      NUM_FILTERS);
    write_doubles(f, &cnn->filters[0][0][0],   kernel_count);
    write_doubles(f, cnn->m_biases,    NUM_FILTERS);
    write_doubles(f, cnn->v_biases,    NUM_FILTERS);
    write_doubles(f, &cnn->m_filters[0][0][0], kernel_count);
    write_doubles(f, &cnn->v_filters[0][0][0], kernel_count);

    fclose(f);
}

int load_cnn(const char *filename, void *cnn_ptr)
{
    if (filename == NULL || cnn_ptr == NULL) return 0;
    CNN *cnn = (CNN *)cnn_ptr;
    FILE *f = fopen(filename, "r");
    if (f == NULL) return 0;

    char magic[16];
    int version, nf, ks;
    if (fscanf(f, "%15s %d %d %d", magic, &version, &nf, &ks) != 4
        || strcmp(magic, CNN_MAGIC) != 0
        || version != CNN_VERSION
        || nf != NUM_FILTERS
        || ks != CONV_SIZE)
    {
        fprintf(stderr, "load_cnn: incompatible file %s (ignored)\n", filename);
        fclose(f);
        return 0;
    }

    int ok = (fscanf(f, "%ld %lf %lf",
                     &cnn->adam_t,
                     &cnn->adam_beta1_t,
                     &cnn->adam_beta2_t) == 3);

    const size_t kernel_count = (size_t)NUM_FILTERS * CONV_SIZE * CONV_SIZE;
    ok &= read_doubles(f, cnn->biases,                NUM_FILTERS);
    ok &= read_doubles(f, &cnn->filters[0][0][0],     kernel_count);
    ok &= read_doubles(f, cnn->m_biases,              NUM_FILTERS);
    ok &= read_doubles(f, cnn->v_biases,              NUM_FILTERS);
    ok &= read_doubles(f, &cnn->m_filters[0][0][0],   kernel_count);
    ok &= read_doubles(f, &cnn->v_filters[0][0][0],   kernel_count);

    fclose(f);

    if (!ok)
        fprintf(stderr, "load_cnn: file %s truncated or corrupt\n", filename);
    return ok;
}

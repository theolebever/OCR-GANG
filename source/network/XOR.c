#include "XOR.h"
#include "../network/network.h"
#include "../network/tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define KGRN "\x1B[32m"
#define KRED "\x1B[31m"
#define KWHT "\x1B[37m"

static const char* NETWORK_FILE = "source/Xor/xorwb.txt";
static const char* RESULT_FILE = "source/Xor/xordata.txt";

static const int INPUT_NEURONS = 2;
static const int HIDDEN_NEURONS = 2;
static const int OUTPUT_NEURONS = 1;
static const int TRAINING_SETS = 4;

static const double TRAINING_INPUTS[] = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
static const double TRAINING_OUTPUTS[] = {0.0, 1.0, 1.0, 0.0};

static void train_network(struct network* network);
static void use_network(struct network* network);
static void save_training_results(struct network* network, FILE* result_file, int index);

void run_xor_demo()
{
    struct network* network = initialize_network(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, NETWORK_FILE);

    char answer[10];
    printf("Do you want to train the neural network or use it?\n1 = Train it\n2 = Use it\n");
    if (fgets(answer, sizeof(answer), stdin) == NULL) {
        fprintf(stderr, "Error reading input\n");
        exit(1);
    }

    switch(answer[0]) {
        case '1':
            train_network(network);
            break;
        case '2':
            use_network(network);
            break;
        default:
            printf("%sInvalid choice!%s\n", KRED, KWHT);
    }

    free_network(network);
}

static void train_network(struct network* network)
{
    FILE* result_file = fopen(RESULT_FILE, "w");
    if (result_file == NULL) {
        fprintf(stderr, "Error opening result file\n");
        exit(1);
    }

    int training_order[TRAINING_SETS];
    for (int i = 0; i < TRAINING_SETS; i++) {
        training_order[i] = i;
    }

    double errors[EARLY_STOPPING_WINDOW] = {0};
    int error_index = 0;

    printf("Started training...\n");
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double epoch_error = 0.0;
        shuffle(training_order, TRAINING_SETS);

        for (int i = 0; i < TRAINING_SETS; i++) {
            int index = training_order[i];
            network->input_layer[0] = TRAINING_INPUTS[2 * index];
            network->input_layer[1] = TRAINING_INPUTS[2 * index + 1];
            network->goal[0] = TRAINING_OUTPUTS[index];

            forward_pass(network);
            back_propagation(network);
            update_weights_and_biases(network);

            epoch_error += pow(network->goal[0] - network->output_layer[0], 2);
            save_training_results(network, result_file, index);
        }

        adaptive_learning_rate(network);
        epoch_error /= TRAINING_SETS;
        errors[error_index] = epoch_error;
        error_index = (error_index + 1) % EARLY_STOPPING_WINDOW;

        if (early_stopping(errors, EARLY_STOPPING_WINDOW, EARLY_STOPPING_THRESHOLD)) {
            printf("Early stopping at epoch %d\n", epoch);
            break;
        }

        if (epoch % 100 == 0) {
            printf("Epoch %d, Error: %f\n", epoch, epoch_error);
        }
        fprintf(result_file, "\n");
    }

    fclose(result_file);
    save_network(NETWORK_FILE, network);
    printf("%sTraining completed!%s\n", KGRN, KWHT);
}

static void use_network(struct network* network)
{
    printf("Using the trained network...\n");
    for (int i = 0; i < INPUT_NEURONS; i++) {
        printf("Please input number %d (0 or 1): ", i + 1);
        if (scanf("%lf", &network->input_layer[i]) != 1) {
            fprintf(stderr, "Invalid input\n");
            exit(1);
        }
        while (getchar() != '\n'); // Clear input buffer
    }

    forward_pass(network);
    printf("The neural network output: %f\n", network->output_layer[0]);
    printf("Interpreted result: %s\n", network->output_layer[0] > 0.5 ? "1" : "0");
}

static void save_training_results(struct network* network, FILE* result_file, int index)
{
    fprintf(result_file, "input: %f ^ %f => output = %f, expected: %f\n",
            network->input_layer[0], network->input_layer[1],
            network->output_layer[0], TRAINING_OUTPUTS[index]);
}
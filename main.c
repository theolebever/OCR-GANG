#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "SDL/SDL.h"
#include "SDL/SDL_image.h"
#include "err.h"
#include "source/GUI/gui.h"
#include "source/network/network.h"
#include "source/network/tools.h"
#include "source/process/process.h"
#include "source/sdl/our_sdl.h"
#include "source/segmentation/segmentation.h"

#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KWHT "\x1B[37m"
#define UNUSED(x) (void)(x)

/**
 * Implements the XOR neural network demo.
 * Allows training or using a neural network for the XOR operation.
 */
void XOR(void)
{
    // Neural network initialization
    struct network *network = InitializeNetwork(2, 4, 1, "source/Xor/xorwb.txt");

    if (network == NULL)
    {
        errx(1, "Failed to initialize neural network");
    }

    // Define training data
    static const int number_training_sets = 4;
    double training_inputs[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    double training_outputs[] = {0.0f, 1.0f, 1.0f, 0.0f};
    int trainingSetOrder[] = {0, 1, 2, 3};

    // File for saving training results
    FILE *result_file = NULL;

    printf("Finished all initialization!\n");

    // Get user choice - train or use the network
    char answer[8];
    printf("Do you want to train the neural network or use it?\n"
           "1 = Train it\n"
           "2 = Use it\n");

    if (fgets(answer, sizeof(answer), stdin) == NULL)
    {
        free(network);
        errx(1, "Error reading input!");
    }

    int choice = atoi(answer);

    if (choice == 1) // Train the network
    {
        result_file = fopen("source/Xor/xordata.txt", "w");
        if (result_file == NULL)
        {
            free(network);
            errx(1, "Failed to open result file");
        }

        printf("Started computing...\n");
        int nb = 10000;
        int step = 0;

        for (int n = 0; n < nb; n++)
        {
            step++;
            progressBar(step, nb);
            shuffle(trainingSetOrder, number_training_sets);

            for (int x = 0; x < number_training_sets; x++)
            {
                int index = trainingSetOrder[x];

                // Set input and expected output
                network->input_layer[0] = training_inputs[2 * index];
                network->input_layer[1] = training_inputs[2 * index + 1];
                network->goal[0] = training_outputs[index];

                // Forward pass
                forward_pass(network);

                // Back propagation
                back_propagation(network);
                updateweightsetbiases(network);

                // Log results
                fprintf(result_file,
                        "input : %f ^ %f => output = %f , expected : %f\n",
                        network->input_layer[0], network->input_layer[1],
                        network->output_layer[0], training_outputs[index]);
            }
            fprintf(result_file, "\n");
        }

        printf("\n");
        printf("\e[?25h"); // Show cursor

        // Save trained network weights and biases
        save_network("source/Xor/xorwb.txt", network);
        fclose(result_file);
    }
    else if (choice == 2) // Use the network
    {
        printf("%sNote: This feature is under development.%s\n", KRED, KWHT);

        char input_str[32];
        double input1, input2;

        printf("Please input the first number (0 or 1):\n");
        if (fgets(input_str, sizeof(input_str), stdin) == NULL)
        {
            free(network);
            errx(1, "Error reading input!");
        }
        input1 = atof(input_str);

        printf("Please input the second number (0 or 1):\n");
        if (fgets(input_str, sizeof(input_str), stdin) == NULL)
        {
            free(network);
            errx(1, "Error reading input!");
        }
        input2 = atof(input_str);

        // Set inputs and compute
        network->input_layer[0] = input1;
        network->input_layer[1] = input2;
        forward_pass(network);

        printf("The neural network returned: %f\n", network->output_layer[0]);
    }
    else
    {
        printf("Invalid option selected.\n");
    }

    freeNetwork(network);
}

/**
 * Starts the OCR process with the given image file.
 * @param filepath Path to the image file to process
 */
void StartOCR(char *filepath)
{
    if (filepath == NULL)
    {
        errx(1, "Invalid file path");
    }

    // Initialize neural network
    struct network *network = InitializeNetwork(28 * 28, 20, 52, "source/OCR/ocrwb.txt");
    if (network == NULL)
    {
        errx(1, "Failed to initialize neural network");
    }

    // Initialize SDL and load image
    init_sdl();
    SDL_Surface *image = load__image(filepath);
    if (image == NULL)
    {
        free(network);
        SDL_Quit();
        errx(1, "Failed to load image");
    }

    // Process image
    image = black_and_white(image);
    DrawRedLines(image);

    int BlocCount = CountBlocs(image);
    SDL_Surface ***chars = malloc(sizeof(SDL_Surface **) * BlocCount);
    SDL_Surface **blocs = malloc(sizeof(SDL_Surface *) * BlocCount);

    if (chars == NULL || blocs == NULL)
    {
        SDL_FreeSurface(image);
        free(chars);
        free(blocs);
        free(network);
        SDL_Quit();
        errx(1, "Memory allocation failed");
    }

    int *charslen = DivideIntoBlocs(image, blocs, chars, BlocCount);
    if (charslen == NULL)
    {
        SDL_FreeSurface(image);
        free(chars);
        free(blocs);
        free(network);
        SDL_Quit();
        errx(1, "Character segmentation failed");
    }

    // Save segmented image
    SDL_SaveBMP(image, "segmentation.bmp");

    // Free bloc surfaces
    for (int j = 0; j < BlocCount; ++j)
    {
        if (blocs[j] != NULL)
        {
            SDL_FreeSurface(blocs[j]);
        }
    }

    // Convert image to matrix for neural network processing
    int **chars_matrix = NULL;
    int chars_count = ImageToMatrix(chars, &chars_matrix, charslen, BlocCount);

    char *result = calloc(chars_count + 1, sizeof(char));
    if (result == NULL)
    {
        SDL_FreeSurface(image);
        free(chars);
        free(blocs);
        free(charslen);
        free(network);
        SDL_Quit();
        errx(1, "Memory allocation failed");
    }

    // Process each character
    for (int index = 0; index < chars_count; index++)
    {
        int is_space = InputImage(network, index, &chars_matrix);
        if (!is_space)
        {
            forward_pass(network);
            size_t index_answer = IndexAnswer(network);
            result[index] = RetrieveChar(index_answer);
        }
        else
        {
            result[index] = ' ';
        }
    }
    result[chars_count] = '\0'; // Ensure string is properly terminated

    // Cleanup and output result
    SDL_FreeSurface(image);
    free(chars);
    free(blocs);
    free(charslen);
    // Would need to free chars_matrix here

    printf("OCR Result: %s\n", result);
    free(result);
    free(network);
    SDL_Quit();
}

void PrintTrainingStats(char expected, char recognized, int *correct_count, int total_count)
{
    static int batch_correct = 0;
    static int batch_total = 0;
    static int last_reported_percent = 0;

    // Update counts
    if (expected == recognized)
    {
        (*correct_count)++;
        batch_correct++;
    }

    batch_total++;

    // Calculate overall accuracy
    float accuracy = (float)(*correct_count) / total_count * 100.0f;

    // Calculate batch accuracy (every 100 iterations)
    if (batch_total >= 100)
    {
        float batch_accuracy = (float)batch_correct / batch_total * 100.0f;
        int current_percent = (int)accuracy;

        // Only print if accuracy has changed by at least 1% or on specific intervals
        if (current_percent != last_reported_percent || total_count % 500 == 0)
        {
            printf("\rOverall Accuracy: %.2f%% | Last 100 samples: %.2f%% | Total samples: %d",
                   accuracy, batch_accuracy, total_count);
            fflush(stdout);
            last_reported_percent = current_percent;
        }

        // Reset batch counters
        batch_correct = 0;
        batch_total = 0;
    }
}

/**
 * Train the neural network for OCR with prepared training data.
 */
void TNeuralNetwork(void)
{
    int correct_count = 0;
    int total_count = 0;
    struct network *network = InitializeNetwork(28 * 28, 20, 52, "source/OCR/ocrwb.txt");
    if (network == NULL)
    {
        errx(1, "Failed to initialize neural network");
    }

    char *filepath = "img/training/maj/A0.txt";
    char expected_result[52] = {
        'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e',
        'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i', 'J', 'j',
        'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o',
        'P', 'p', 'Q', 'q', 'R', 'r', 'S', 's', 'T', 't',
        'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y',
        'Z', 'z'};

    printf("Starting neural network training...\n");

    int nb = 2500;
    int total_iterations = nb * 52;
    int current_iteration = 0;

    for (int number = 0; number < nb; number++)
    {
        for (int i = 0; i < 52; i++)
        {
            ExpectedOutput(network, expected_result[i]);
            InputFromTXT(filepath, network);
            forward_pass(network);

            char recognized = RetrieveChar(IndexAnswer(network));
            total_count++;

            // Replace PrintState with this:
            PrintTrainingStats(expected_result[i], recognized, &correct_count, total_count);
            // PrintState(expected_result[i], recognized);

            back_propagation(network);
            updateweightsetbiases(network);

            current_iteration++;
            // if (current_iteration % 100 == 0)
            // {
            //     progressBar(current_iteration, total_iterations);
            // }
        }
    }

    printf("\n%sTraining completed!%s\n", KGRN, KWHT);
    printf("\e[?25h"); // Show cursor

    save_network("source/OCR/ocrwb.txt", network);
    free(network);
}

/**
 * Main function - entry point of the program
 */
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        // Start GUI if no arguments provided
        printf("Starting GUI interface...\n");
        InitGUI(argc, argv);
        return 0;
    }

    // Process command line arguments
    if (strcmp(argv[1], "--XOR") == 0)
    {
        XOR();
    }
    else if (strcmp(argv[1], "--OCR") == 0)
    {
        if (argc < 3)
        {
            printf("Error: Missing image path for OCR.\n");
            printf("Usage: %s --OCR <image_path>\n", argv[0]);
            return 1;
        }

        if (cfileexists(argv[2]))
        {
            PrepareTraining();
            TNeuralNetwork();
            StartOCR(argv[2]);
        }
        else
        {
            printf("Error: There is no such image, please specify a correct path.\n");
            return 1;
        }
    }
    else if (strcmp(argv[1], "--train") == 0)
    {
        PrepareTraining();
        TNeuralNetwork();
    }
    else
    {
        // Display help if invalid argument
        printf("-----------------------\n");
        printf("Bienvenue dans OCR GANG\n");
        printf("-----------------------\n");
        printf("Arguments :\n");
        printf("    (Aucun) Lance l'interface utilisateur (GUI)\n");
        printf("    --train Lance l'entrainement du réseau de neurones\n");
        printf("    --OCR <image_path> Lance l'OCR sur l'image spécifiée\n");
        printf("    --XOR   Montre la fonction XOR\n");
    }

    return 0;
}
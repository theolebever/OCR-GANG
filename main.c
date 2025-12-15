#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "SDL/SDL.h"
#include "SDL/SDL_image.h"
#include "err.h"
#include "source/GUI/gui.h"
#include "source/network/network.h"
#include "source/network/tools.h"
#include "source/process/process.h"
#include "source/sdl/our_sdl.h"
#include "source/segmentation/segmentation.h"
#include "source/training/training.h"
#include "source/ocr/ocr.h"

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
    const double training_inputs[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    const double training_outputs[] = {0.0f, 1.0f, 1.0f, 0.0f};
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
        freeNetwork(network);
        errx(1, "Error reading input!");
    }

    char *end;
    long choice = strtol(answer, &end, 10);
    if (*end != '\n' && *end != '\0')
    {
        printf("Invalid input\n");
    }

    if (choice == 1) // Train the network
    {
        result_file = fopen("source/Xor/xordata.txt", "w");
        if (result_file == NULL)
        {
            freeNetwork(network);
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
            freeNetwork(network);
            errx(1, "Error reading input!");
        }
        input1 = atof(input_str);

        printf("Please input the second number (0 or 1):\n");
        if (fgets(input_str, sizeof(input_str), stdin) == NULL)
        {
            freeNetwork(network);
            errx(1, "Error reading input!");
        }
        input2 = atof(input_str);

        if ((input1 != 0.0 && input1 != 1.0) ||
            (input2 != 0.0 && input2 != 1.0))
        {
            printf("Inputs must be 0 or 1\n");
            freeNetwork(network);
            return;
        }

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

int main(int argc, char *argv[])
{
    srand((unsigned)time(NULL));

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
        TrainNetwork();
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
#include <stdio.h>
#include <string.h>

#include "SDL/SDL.h"
#include "SDL/SDL_image.h"
#include "err.h"
#include "source/GUI/gui.h"
#include "source/network/tools.h"
#include "source/process/process.h"
#include "source/sdl/our_sdl.h"
#include "source/segmentation/segmentation.h"
#include "source/network/XOR.h"
#include "source/network/OCR.h"
#include <math.h>

#define UNUSED(x) (void)(x)
#define NB_EPOCHS 1000

void print_usage()
{
    printf("OCR GANG - Usage:\n");
    printf("  No arguments: Launch GUI\n");
    printf("  --train: Train neural network\n");
    printf("  --OCR <image_path>: Perform OCR on specified image\n");
    printf("  --XOR: Demonstrate XOR function\n");
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        prepare_training();
        init_gui(argc, argv);
        return 0;
    }

    if (strcmp(argv[1], "--XOR") == 0)
    {
        run_xor_demo();
    }
    else if (strcmp(argv[1], "--OCR") == 0 && argc == 3)
    {
        printf("Not implemented\n");
    }
    else if (strcmp(argv[1], "--train") == 0)
    {
        prepare_training();
        Network *cnn = create_ocr_network();

        char *filematrix = "img/training/maj/A0.txt\0";
        char expected_result[52] = {'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E',
                                    'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                                    'J', 'j', 'K', 'k', 'L', 'I', 'M', 'm', 'N',
                                    'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                                    'S', 's', 'I', 't', 'U', 'u', 'V', 'v', 'W',
                                    'w', 'X', 'x', 'Y', 'y', 'Z', 'z'};
        train(cnn, filematrix, expected_result, 4, 10, 0.1);
        free_network_cnn(cnn);
    }
    else
    {
        print_usage();
    }
    return 0;
}
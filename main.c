#include <stdio.h>
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
#include "source/network/XOR.h"
#include "source/network/OCR.h"
#include <math.h>


#define UNUSED(x) (void)(x)

void print_usage() {
    printf("OCR GANG - Usage:\n");
    printf("  No arguments: Launch GUI\n");
    printf("  --train: Train neural network\n");
    printf("  --OCR <image_path>: Perform OCR on specified image\n");
    printf("  --XOR: Demonstrate XOR function\n");
}


int main(int argc, char **argv) {
    if (argc < 2) {
        prepare_training();
        init_gui(argc, argv);
        return 0;
    }

    if (strcmp(argv[1], "--XOR") == 0) {
        run_xor_demo();
    }
    else if (strcmp(argv[1], "--OCR") == 0 && argc == 3) {
        if (file_exists(argv[2])) {
            prepare_training();
            train_neural_network();
            perform_ocr(argv[2]);
        } else {
            printf("Error: Image file not found. Please specify a correct path.\n");
        }
    }
    else if (strcmp(argv[1], "--train") == 0) {
        prepare_training();
        train_neural_network();
    }
    else {
        print_usage();
    }

    return 0;
}
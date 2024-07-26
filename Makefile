# OCR_Gang's Makefile
#Author : marius.andre

CC=gcc

CPPFLAGS= `pkg-config --cflags sdl gtk+-3.0` -MMD
CFLAGS= -Wall -Wextra -Werror -std=c99
LDFLAGS= -rdynamic
LDLIBS= `pkg-config --libs sdl gtk+-3.0` -lSDL_image -lm

SRC= main.c source/process/process.c source/sdl/our_sdl.c source/segmentation/segmentation.c source/network/network.c source/network/tools.c source/GUI/gui.c source/network/OCR.c source/network/XOR.c
OBJ= $(SRC:.c=.o)
DEP= $(SRC:.c=.d)

all: main create

create:
	[ -d "source/Xor" ] || mkdir source/Xor
	[ -d "source/OCR" ] || mkdir source/OCR
	touch source/Xor/xordata.txt
	touch source/Xor/xorwb.txt
	touch source/OCR/ocrwb.txt

main: $(OBJ)

clean:
	rm -rf *.bmp img/temp/*.bmp source/Xor source/OCR *.tst img/training/maj/*.txt img/training/min/*.txt
	$(RM) $(OBJ) $(OBJ_TESTS) $(DEP) $(DEP_TESTS) main && clear
# END

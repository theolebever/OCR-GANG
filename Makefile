# OCR_Gang's Makefile
#Author : marius.andre

CC=gcc

CPPFLAGS= `pkg-config --cflags sdl gtk+-3.0` -MMD
CFLAGS= -Wall -Wextra -std=c99 -O3
LDFLAGS= -ldl -lm -rdynamic
LDLIBS= `pkg-config --libs sdl gtk+-3.0` -lSDL_image

SRC= main.c source/process/process.c source/sdl/our_sdl.c source/segmentation/segmentation.c source/network/network.c source/network/tools.c source/GUI/gui.c source/training/training.c source/ocr/ocr.c
OBJ= $(SRC:.c=.o)
DEP= $(SRC:.c=.d)

all: main create

create:
	[ -d "source/Xor" ] || mkdir source/Xor
	[ -d "source/OCR-data" ] || mkdir source/OCR-data
	touch source/Xor/xordata.txt
	touch source/Xor/xorwb.txt
	touch source/OCR-data/ocrwb.txt

main: $(OBJ)

debug: CFLAGS+= -g
debug: LDFLAGS+= -fsanitize=address
debug: LDLIBS+= -lasan
debug: all

clean:
	rm -rf *.bmp img/temp/*.bmp source/Xor source/OCR-data *.tst img/training/maj/*.txt img/training/min/*.txt
	$(RM) $(OBJ) $(OBJ_TESTS) $(DEP) $(DEP_TESTS) main && clear
# END

# Compiler and flags
CC := gcc
CPPFLAGS := $(shell pkg-config --cflags sdl gtk+-3.0) -MMD
CFLAGS := -Wall -Wextra -Werror -std=c99
LDFLAGS := -rdynamic
LDLIBS := $(shell pkg-config --libs sdl gtk+-3.0) -lSDL_image -lm

# Debug flags
DEBUG_CFLAGS := -g -O0 -DDEBUG -fsanitize=address
DEBUG_LDFLAGS := -fsanitize=address

# Release flags
RELEASE_CFLAGS := -O3

# Directories
SRC_DIR := source
BUILD_DIR := build
DEBUG_BUILD_DIR := build_debug

# Source files
SRC := main.c \
    $(SRC_DIR)/process/process.c \
    $(SRC_DIR)/sdl/our_sdl.c \
    $(SRC_DIR)/segmentation/segmentation.c \
    $(SRC_DIR)/network/tools.c \
    $(SRC_DIR)/network/OCR.c \
    $(SRC_DIR)/network/XOR.c \
    $(SRC_DIR)/network/early_stop.c \
    $(SRC_DIR)/network/pool.c \
    $(SRC_DIR)/network/fc.c \
    $(SRC_DIR)/network/conv.c \
    $(SRC_DIR)/network/volume.c \
    $(SRC_DIR)/network/adam.c \
    $(SRC_DIR)/network/training.c \
    $(SRC_DIR)/GUI/gui.c

# Object files
OBJ := $(SRC:%.c=$(BUILD_DIR)/%.o)
DEBUG_OBJ := $(SRC:%.c=$(DEBUG_BUILD_DIR)/%.o)

# Dependency files
DEP := $(OBJ:.o=.d)
DEBUG_DEP := $(DEBUG_OBJ:.o=.d)

# Executables
MAIN := main
DEBUG_MAIN := main_debug

# Phony targets
.PHONY: all debug clean create

# Default target
all: create $(MAIN)

# Debug target
debug: CFLAGS += $(DEBUG_CFLAGS)
debug: LDFLAGS += $(DEBUG_LDFLAGS)
debug: create $(DEBUG_MAIN)

# Create necessary directories
create:
	mkdir -p $(SRC_DIR)/Xor $(SRC_DIR)/OCR
	touch $(SRC_DIR)/Xor/xordata.txt $(SRC_DIR)/Xor/xorwb.txt $(SRC_DIR)/OCR/ocrwb.txt

# Compile the main executable (release version)
$(MAIN): CFLAGS += $(RELEASE_CFLAGS)
$(MAIN): $(OBJ)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Compile the debug executable
$(DEBUG_MAIN): $(DEBUG_OBJ)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Compile object files (release version)
$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Compile object files (debug version)
$(DEBUG_BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -rf $(BUILD_DIR) $(DEBUG_BUILD_DIR) $(MAIN) $(DEBUG_MAIN)
	rm -rf *.bmp img/temp/*.bmp $(SRC_DIR)/Xor $(SRC_DIR)/OCR *.tst img/training/maj/*.txt img/training/min/*.txt *.bin

# Include dependency files
-include $(DEP) $(DEBUG_DEP)

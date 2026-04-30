# OCR-GANG

OCR-GANG is an OCR project written in C for EPITA.

## Prerequisites

- `gcc`
- `make`
- `pkg-config`
- SDL 1.2 image development package, for example `libsdl-image1.2-dev`
- GTK+ 3 development package, for example `libgtk-3-dev`

The Makefile uses Unix shell command substitution for `pkg-config`, so build it from a Unix-like shell. On Windows/WSL, this works:

```sh
bash -lc "make"
```

## Build

From the project root:

```sh
make
```

This builds the `main` executable and creates the weight/data directories if they do not exist.

## Usage

```sh
./main
```

Launches the GUI.

```sh
./main --train
```

Trains the OCR model. Training images are loaded from:

```text
img/training/maj
img/training/min
```

The trained weights are saved to:

```text
source/OCR-data/ocrwb.txt
source/OCR-data/cnnwb.txt
```

```sh
./main --OCR <image_path>
```

Runs OCR on the given image.

```sh
./main --XOR
```

Runs the XOR neural-network demo.

## Authors

- Marius ANDRE
- Pierre MEGALLI
- Theo LE BEVER
- Maxence DE TORQUAT DE LA COULERIE

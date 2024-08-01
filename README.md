# Welcome to OCR-GANG 👋
![Version](https://img.shields.io/badge/version-2.0-blue.svg?cacheSeconds=2592000)

> Create an OCR in C language for EPITA

## Prérequis
* SDL_Image (libsdl-image1.2-dev)
* GTK+ 3.0 (libgtk-3-dev)

## Installation

Démarrer une console dans le répertoire du projet puis lancer les commandes suivantes:

```sh
mkdir build
cd build
cmake ..
make
Lancer le programme avec la commande ../OCRProject
```
## Usage

```sh
OCR GANG - Usage:
  No arguments: Launch GUI
  --train: Train neural network
  --OCR <image_path>: Perform OCR on specified image
  --XOR: Demonstrate XOR function
```

## Explications

L'OCR (Reconnaissance Optique de Caractères) de ce projet est construit en utilisant un [réseau de neurones convolutifs (CNN)](https://fr.wikipedia.org/wiki/R%C3%A9seau_neuronal_convolutif). Voici une brève explication de sa construction :

1. **Prétraitement de l'image**:
    - Les images d'entrée sont converties en [niveaux de gris](https://fr.wikipedia.org/wiki/Niveau_de_gris) et redimensionnées pour correspondre à la taille attendue par le réseau de neurones.
    - La [segmentation](https://fr.wikipedia.org/wiki/Segmentation_d%27image) est effectuée pour isoler les caractères individuels dans l'image.

2. **Réseau de Neurones Convolutifs**:
    - **[Convolution Layers](https://cs231n.github.io/convolutional-networks/#conv)**: Les couches convolutives sont utilisées pour extraire des caractéristiques de bas niveau des images (comme les contours).
    - **[Pooling Layers](https://cs231n.github.io/convolutional-networks/#pool)**: Les couches de pooling réduisent la dimensionnalité des données et permettent une certaine invariance de translation.
    - **[Fully Connected Layers](https://cs231n.github.io/convolutional-networks/#fc)**: Ces couches prennent les caractéristiques extraites et prédisent la classe des caractères.

3. **Entraînement**:
    - Le réseau est entraîné en utilisant des ensembles de données d'images de caractères étiquetées.
    - L'[optimiseur Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) est utilisé pour ajuster les poids du réseau en minimisant l'erreur de prédiction.

4. **Reconnaissance**:
    - Une fois entraîné, le réseau peut prendre une nouvelle image en entrée, segmenter les caractères, et prédire les lettres correspondantes.
    - Le résultat final est une chaîne de caractères reconnue à partir de l'image d'entrée.

## Données d'entraînement

Les données utilisées pour entraîner le modèle sont des images de caractères isolés, générées et étiquetées manuellement. Chaque image est redimensionnée et convertie en niveaux de gris avant d'être utilisée pour l'entraînement.

## Performance

TBD


## Authors

👤 **Marius ANDRE ** 👤 **Pierre MEGALLI ** 👤 **Théo LE BEVER ** 👤 **Maxence DE TORQUAT DE LA COULERIE **


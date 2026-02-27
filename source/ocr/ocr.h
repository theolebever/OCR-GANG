#ifndef OCR_H
#define OCR_H

#include <gtk/gtk.h>

// Starts the OCR process from CLI
void StartOCR(char *filepath);

// Starts the OCR process from GUI
int OCR(GtkButton *button, GtkTextBuffer *buffer);

#endif

#ifndef OCR_H
#define OCR_H

// Runs the full OCR pipeline and returns the recognized text (caller frees).
// Returns NULL on failure.
char *PerformOCR(const char *filepath);

// CLI entry point: runs OCR, prints the result, exits on failure.
void StartOCR(const char *filepath);

#endif

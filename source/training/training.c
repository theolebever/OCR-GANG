#include "training.h"
#include "../network/tools.h"
#include "../network/network.h"
#include <stdio.h>
#include <stdlib.h>
#include <err.h>

void PrintTrainingStats(char expected, char recognized, int *correct_count, int total_count)
{
    static int batch_correct = 0;
    static int batch_total = 0;
    static int last_reported_percent = 0;

    if (expected == recognized)
    {
        (*correct_count)++;
        batch_correct++;
    }

    batch_total++;
    float accuracy = (float)(*correct_count) / total_count * 100.0f;

    if (batch_total >= 100)
    {
        float batch_accuracy = (float)batch_correct / batch_total * 100.0f;
        int current_percent = (int)accuracy;

        if (current_percent != last_reported_percent || total_count % 500 == 0)
        {
            printf("\rOverall Accuracy: %.2f%% | Last 100 samples: %.2f%% | Total samples: %d",
                   accuracy, batch_accuracy, total_count);
            fflush(stdout);
            last_reported_percent = current_percent;
        }

        batch_correct = 0;
        batch_total = 0;
    }
}

void TrainNetwork(void)
{
    printf("Loading Dataset...\n");
    TrainingDataSet *dataset = loadDataSet();
    
    if (dataset == NULL)
    {
        errx(1, "Failed to load dataset!");
    }
    
    printf("\n=== DATASET ANALYSIS ===\n");
    printf("Total samples: %d\n", dataset->count);
    
    // Check class distribution
    int class_counts[52] = {0};
    for (int i = 0; i < dataset->count; i++)
    {
        size_t pos = ExpectedPos(dataset->labels[i]);
        if (pos < 52) class_counts[pos]++;
    }
    
    int classes_with_data = 0;
    int min_samples = dataset->count;
    int max_samples = 0;
    
    for (int i = 0; i < 52; i++)
    {
        if (class_counts[i] > 0)
        {
            classes_with_data++;
            if (class_counts[i] < min_samples) min_samples = class_counts[i];
            if (class_counts[i] > max_samples) max_samples = class_counts[i];
        }
    }
    
    printf("Classes with data: %d/52\n", classes_with_data);
    printf("Samples per class - Min: %d, Max: %d, Avg: %.1f\n", 
           min_samples, max_samples, (float)dataset->count / (float)classes_with_data);
    
    // WARNING for tiny datasets
    if (dataset->count < 1000)
    {
        printf("\n WARNING: Dataset is extremely small!\n");
        printf("   Recommended: At least 50-100 samples per class\n");
        printf("   Current: ~%.1f samples per class\n", (float)dataset->count / 52.0f);
        printf("   Network will struggle to learn effectively.\n");
    }

    // CRITICAL: Use smaller network for tiny datasets to prevent overfitting
    // Rule of thumb: hidden nodes should be ~sqrt(inputs * outputs)
    // sqrt(784 * 52) ≈ 200, but for tiny dataset use much less
    int hidden_nodes = OCR_HIDDEN_NODES;  // Very small to prevent overfitting
    
    printf("\n=== NETWORK CONFIGURATION ===\n");
    printf("Architecture: 784-%d-52\n", hidden_nodes);
    printf("Total parameters: %d\n", (784 * hidden_nodes) + hidden_nodes + (hidden_nodes * 52) + 52);
    
    struct network *net = InitializeNetwork(784, hidden_nodes, 52, "source/OCR-data/ocrwb.txt");
    
    if (net == NULL)
    {
        freeDataSet(dataset);
        errx(1, "Failed to initialize network!");
    }

    // For tiny datasets: many epochs with very low learning rate
    int epochs = 200;  // More epochs since we learn slowly
    
    // Create an index array for shuffling
    int *indices = malloc(sizeof(int) * dataset->count);
    for(int i = 0; i < dataset->count; i++) indices[i] = i;

    printf("\n=== TRAINING CONFIGURATION ===\n");
    printf("Epochs: %d\n", epochs);
    printf("Learning rate: %.5f (very conservative)\n", net->eta);
    printf("Momentum: %.2f\n", net->alpha);
    printf("Samples per epoch: %d\n\n", dataset->count);

    // Track best accuracy
    float best_accuracy = 0.0f;
    int epochs_without_improvement = 0;
    int save_interval = 10;
    
    printf("Starting Training...\n");
    printf("================================================================================\n");
    
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        shuffle(indices, dataset->count);
        int epoch_correct = 0;
        
        // Train on all samples
        for (int i = 0; i < dataset->count; i++)
        {
            int idx = indices[i];
            
            // Load input
            for(int j = 0; j < 784; j++) {
                net->input_layer[j] = dataset->inputs[idx][j];
            }

            // Set goal
            ExpectedOutput(net, dataset->labels[idx]);

            // Forward pass
            forward_pass(net);

            // Back propagation
            back_propagation(net);

            // Check prediction
            size_t answer_idx = IndexAnswer(net);
            char recognized = RetrieveChar(answer_idx);
            if (recognized == dataset->labels[idx])
            {
                epoch_correct++;
            }
        }
        
        float epoch_accuracy = (float)epoch_correct / dataset->count * 100.0f;
        
        // Print progress
        printf("Epoch %3d/%d | Accuracy: %6.2f%% (%3d/%3d correct)", 
               epoch + 1, epochs, epoch_accuracy, epoch_correct, dataset->count);
        
        // Track improvement
        if (epoch_accuracy > best_accuracy)
        {
            best_accuracy = epoch_accuracy;
            epochs_without_improvement = 0;
            printf(" * NEW BEST");
            save_network("source/OCR-data/ocrwb.txt", net);  // Save on improvement
        }
        else
        {
            epochs_without_improvement++;
        }
        
        printf("\n");
        
        // Periodic save
        if ((epoch + 1) % save_interval == 0)
        {
            save_network("source/OCR-data/ocrwb.txt", net);
        }
        
        // Learning rate decay - very gradual for tiny datasets
        if ((epoch + 1) % 50 == 0 && net->eta > 0.0001)
        {
            net->eta *= 0.9;  // Reduce by only 10%
            printf("    -> Learning rate adjusted to: %.6f\n", net->eta);
        }
        
        // Early stopping - more patient for tiny datasets
        if (epochs_without_improvement >= 30)
        {
            printf("\nEarly stopping: No improvement for 30 epochs.\n");
            break;
        }
    }

    printf("================================================================================\n");
    printf("\n=== TRAINING SUMMARY ===\n");
    printf("Best Accuracy Achieved: %.2f%%\n", best_accuracy);
    printf("Expected for this dataset size: 10-30%% (severely limited by data)\n");
    
    if (best_accuracy < 20.0f)
    {
        printf("\nCRITICAL: Accuracy is very low!\n");
        printf("   Primary issue: Dataset too small (only %d samples)\n", dataset->count);
        printf("   Solutions:\n");
        printf("   1. Collect more training data (aim for 2600+ samples)\n");
        printf("   2. Use data augmentation (rotate, scale, shift images)\n");
        printf("   3. Reduce number of classes (e.g., only uppercase OR lowercase)\n");
    }
    else if (best_accuracy < 50.0f)
    {
        printf("\nLow accuracy - dataset still too small\n");
        printf("   Need more samples for reliable OCR\n");
    }
    else
    {
        printf("\nReasonable accuracy for dataset size\n");
    }
    
    printf("\nSaving final model...\n");
    save_network("source/OCR-data/ocrwb.txt", net);
    printf("Model saved to: source/OCR-data/ocrwb.txt\n");
    
    free(indices);
    freeDataSet(dataset);
    freeNetwork(net);
}

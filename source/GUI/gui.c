#include "gui.h"

#include "../network/tools.h"
#include "../training/training.h"
#include "../ocr/ocr.h"

// GUI-owned state: kept file-scoped so other modules don't reach in via `extern`.
static gchar *filename = NULL;  // owned: free with g_free before replacing
static char  *text     = NULL;  // owned: free before replacing
static GtkWidget *parent;

void save_text(GtkButton *button, GtkTextBuffer *buffer)
{
    UNUSED(buffer);
    GtkWidget *toplevel = gtk_widget_get_toplevel(GTK_WIDGET(button));
    GtkWidget *dialog = gtk_file_chooser_dialog_new(
        "Save Text ", GTK_WINDOW(toplevel), GTK_FILE_CHOOSER_ACTION_SAVE,
        "Cancel", GTK_RESPONSE_CANCEL, "Save", GTK_RESPONSE_ACCEPT, NULL);
    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT)
    {
        gchar *save_path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        if (save_path != NULL)
        {
            g_file_set_contents(save_path,
                                text ? text : "",
                                text ? strlen(text) : 0,
                                NULL);
            g_free(save_path);
        }
    }
    gtk_widget_destroy(dialog);
}

void gui_load_image(GtkButton *button, GtkImage *image)
{
    UNUSED(button);
    if (filename == NULL || filename[0] == '\0')
        return;

    SDL_Surface *img = load_image(filename);
    if (img->w > 560 || img->h > 560)
    {
        float wi = img->w, hi = img->h;
        float best = (560.0f / wi < 560.0f / hi) ? 560.0f / wi : 560.0f / hi;
        SDL_Surface *scaled = resize(img, (int)(wi * best), (int)(hi * best));
        SDL_SaveBMP(scaled, "image_resize.bmp");
        SDL_FreeSurface(scaled);
        gtk_image_set_from_file(image, "image_resize.bmp");
    }
    else
    {
        gtk_image_set_from_file(image, filename);
    }
    SDL_FreeSurface(img);
}

void open_image(GtkButton *button, GtkLabel *text_label)
{
    GtkWidget *toplevel = gtk_widget_get_toplevel(GTK_WIDGET(button));
    GtkWidget *dialog = gtk_file_chooser_dialog_new(
        "Open image", GTK_WINDOW(toplevel), GTK_FILE_CHOOSER_ACTION_OPEN,
        "Open", GTK_RESPONSE_ACCEPT, "Cancel", GTK_RESPONSE_CANCEL, NULL);

    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT)
    {
        g_free(filename);
        filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        if (filename != NULL)
            gtk_label_set_text(text_label, filename);
    }
    gtk_widget_destroy(dialog);
}

int OCR(GtkButton *button, GtkTextBuffer *buffer)
{
    UNUSED(button);

    if (filename == NULL || filename[0] == '\0')
    {
        g_print("No file selected!\n");
        return EXIT_FAILURE;
    }

    char *result = PerformOCR(filename);
    if (result == NULL)
    {
        g_print("OCR Failed!\n");
        return EXIT_FAILURE;
    }

    g_print("OCR Done !\nResult: %s\n", result);

    free(text);
    text = result;  // transfers ownership
    gtk_text_buffer_set_text(buffer, text, strlen(text));

    return EXIT_SUCCESS;
}

int TrainNeuralNetwork()
{
    // Use the shared training function
    // Note: TrainNetwork currently prints to stdout. 
    // If we need GUI feedback, we might need to redirect stdout or modify TrainNetwork.
    // For now, we assume console output is acceptable as per original code structure.
    TrainNetwork();
    return EXIT_SUCCESS;
}

void InitGUI(int argc, char *argv[])
{
    // Init variables
    GtkWidget *main_window;
    SGlobalData data;
    // Init GTK
    gtk_init(&argc, &argv);
    // Build from .glade
    data.builder = gtk_builder_new();
    gtk_builder_add_from_file(data.builder, "gui.glade", NULL);
    // Get main_window
    main_window =
        GTK_WIDGET(gtk_builder_get_object(data.builder, "main_window"));
    parent = main_window;
    // Connect signals
    gtk_builder_connect_signals(data.builder, &data);
    g_signal_connect(main_window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

    gtk_window_set_title(GTK_WINDOW(main_window), "Welcome to OCR-GANG");
    gtk_widget_show_all(main_window);
    gtk_main();
}
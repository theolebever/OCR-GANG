#include "gui.h"

#include "../network/tools.h"
#include "../training/training.h"
#include "../ocr/ocr.h"

gchar *filename = "";
char *text = NULL;
GtkWidget *parent;

void save_text(GtkButton *button, GtkTextBuffer *buffer)
{
    UNUSED(button);
    UNUSED(buffer);
    GtkWidget *dialog;
    GtkWidget *toplevel = gtk_widget_get_toplevel(GTK_WIDGET(button));
    dialog = gtk_file_chooser_dialog_new(
        "Save Text ", GTK_WINDOW(toplevel), GTK_FILE_CHOOSER_ACTION_SAVE,
        "Cancel", GTK_RESPONSE_CANCEL, "Save", GTK_RESPONSE_ACCEPT, NULL);
    if (gtk_dialog_run(GTK_DIALOG(dialog)) == GTK_RESPONSE_ACCEPT)
    {
        char *filename;
        filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        /* set the contents of the file to the text from the buffer */
        if (filename != NULL)
            g_file_set_contents(filename, text ? text : "", text ? strlen(text) : 0, NULL);
    }
    gtk_widget_destroy(dialog);
}

void load_image(GtkButton *button, GtkImage *image)
{
    if (strcmp(filename, "") == 0)
        return;
    UNUSED(button);
    SDL_Surface *img = load__image((char *)filename);
    if (img->w > 560 && img->h > 560)
    {
        float wi = img->w;
        float hi = img->h;
        float max_h = 560.;
        float max_w = 560.;
        float best;
        if (max_w / wi < max_h / hi)
            best = max_w / wi;
        else
            best = max_h / hi;
        int new_w = wi * best;
        int new_h = hi * best;
        // printf("%d %d",new_w,new_h);
        SDL_Surface *new = resize(img, new_w, new_h);
        SDL_SaveBMP(new, "image_resize.bmp");
        gtk_image_set_from_file(GTK_IMAGE(image), "image_resize.bmp");
    }
    else
    {
        gtk_image_set_from_file(GTK_IMAGE(image), filename);
    }
}
// Colors for print
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KWHT "\x1B[37m"

void open_image(GtkButton *button, GtkLabel *text_label)
{
    GtkWidget *label = (GtkWidget *)text_label;
    GtkWidget *toplevel = gtk_widget_get_toplevel(GTK_WIDGET(button));
    // GtkFileFilter *filter = gtk_file_filter_new ();
    GtkWidget *dialog = gtk_file_chooser_dialog_new(
        ("Open image"), GTK_WINDOW(toplevel), GTK_FILE_CHOOSER_ACTION_OPEN,
        "Open", GTK_RESPONSE_ACCEPT, "Cancel", GTK_RESPONSE_CANCEL, NULL);

    // gtk_file_filter_add_pixbuf_formats (filter);
    // gtk_file_chooser_add_filter (GTK_FILE_CHOOSER (dialog),filter);

    switch (gtk_dialog_run(GTK_DIALOG(dialog)))
    {
    case GTK_RESPONSE_ACCEPT: {
        filename = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(dialog));
        gtk_label_set_text(GTK_LABEL(label), filename);
        break;
    }
    default:
        break;
    }
    gtk_widget_destroy(dialog);
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
// Simple get/put pixel for SDL
// Inspired by code from SDL documentation
// (http://www.libsdl.org/release/SDL-1.2.15/docs/html/guidevideo.html)

#include "our_sdl.h"

#include <err.h>

#include "SDL/SDL.h"
#include "SDL/SDL_image.h"

void init_sdl()
{
    // Init only the video part.
    // If it fails, die with an error message.
    if (SDL_Init(SDL_INIT_VIDEO) == -1)
        errx(1, "Could not initialize SDL: %s.\n", SDL_GetError());
}

SDL_Surface *load__image(char *path)
{
    SDL_Surface *img;

    // Load an image using SDL_image with format detection.
    // If it fails, die with an error message.
    img = IMG_Load(path);
    if (!img)
        errx(3, "can't load %s: %s", path, IMG_GetError());

    return img;
}

SDL_Surface *display_image(SDL_Surface *img)
{
    SDL_Surface *screen;

    // Set the window to the same size as the image
    screen = SDL_SetVideoMode(img->w, img->h, 0, SDL_SWSURFACE | SDL_ANYFORMAT);
    if (screen == NULL)
    {
        // error management
        errx(1, "Couldn't set %dx%d video mode: %s\n", img->w, img->h,
             SDL_GetError());
    }

    // Blit onto the screen surface
    if (SDL_BlitSurface(img, NULL, screen, NULL) < 0)
        warnx("BlitSurface error: %s\n", SDL_GetError());

    // Update the screen
    SDL_UpdateRect(screen, 0, 0, img->w, img->h);

    // return the screen for further uses
    return screen;
}

void wait_for_keypressed()
{
    SDL_Event event;

    // Wait for a key to be down.
    do
    {
        SDL_PollEvent(&event);
    } while (event.type != SDL_KEYDOWN);

    // Wait for a key to be up.
    do
    {
        SDL_PollEvent(&event);
    } while (event.type != SDL_KEYUP);
}

static inline Uint8 *pixel_ref(SDL_Surface *surf, unsigned x, unsigned y)
{
    int bpp = surf->format->BytesPerPixel;
    return (Uint8 *)surf->pixels + y * surf->pitch + x * bpp;
}

Uint32 get_pixel(SDL_Surface *surface, unsigned x, unsigned y)
{
    Uint8 *p = pixel_ref(surface, x, y);

    switch (surface->format->BytesPerPixel)
    {
    case 1:
        return *p;

    case 2:
        return *(Uint16 *)p;

    case 3:
        if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
            return p[0] << 16 | p[1] << 8 | p[2];
        else
            return p[0] | p[1] << 8 | p[2] << 16;

    case 4:
        return *(Uint32 *)p;
    }

    return 0;
}

void put_pixel(SDL_Surface *surface, unsigned x, unsigned y, Uint32 pixel)
{
    Uint8 *p = pixel_ref(surface, x, y);

    switch (surface->format->BytesPerPixel)
    {
    case 1:
        *p = pixel;
        break;

    case 2:
        *(Uint16 *)p = pixel;
        break;

    case 3:
        if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
        {
            p[0] = (pixel >> 16) & 0xff;
            p[1] = (pixel >> 8) & 0xff;
            p[2] = pixel & 0xff;
        }
        else
        {
            p[0] = pixel & 0xff;
            p[1] = (pixel >> 8) & 0xff;
            p[2] = (pixel >> 16) & 0xff;
        }
        break;

    case 4:
        *(Uint32 *)p = pixel;
        break;
    }
}

void update_surface(SDL_Surface *screen, SDL_Surface *image)
{
    if (SDL_BlitSurface(image, NULL, screen, NULL) < 0)
        warnx("BlitSurface error: %s\n", SDL_GetError());

    SDL_UpdateRect(screen, 0, 0, image->w, image->h);
}

Uint8 getRed(Uint32 pixel, SDL_PixelFormat *fmt)
{
    Uint32 temp;
    /* Get Red component */
    temp = pixel & fmt->Rmask; /* Isolate red component */
    temp = temp >> fmt->Rshift; /* Shift it down to 8-bit */
    temp = temp << fmt->Rloss; /* Expand to a full 8-bit number */
    return (Uint8)temp;
}

Uint8 getGreen(Uint32 pixel, SDL_PixelFormat *fmt)
{
    Uint32 temp;
    /* Get Green component */
    temp = pixel & fmt->Gmask; /* Isolate green component */
    temp = temp >> fmt->Gshift; /* Shift it down to 8-bit */
    temp = temp << fmt->Gloss; /* Expand to a full 8-bit number */
    return (Uint8)temp;
}

Uint8 getBlue(Uint32 pixel, SDL_PixelFormat *fmt)
{
    Uint32 temp;
    /* Get Blue component */
    temp = pixel & fmt->Bmask; /* Isolate blue component */
    temp = temp >> fmt->Bshift; /* Shift it down to 8-bit */
    temp = temp << fmt->Bloss; /* Expand to a full 8-bit number */
    return (Uint8)temp;
}

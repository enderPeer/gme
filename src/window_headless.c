/* window_headless.c - window.h without a display (Linux and other non-Windows
 * builds). The "window" is just a size: frames are rendered but not shown,
 * there is no input, and the window never closes by itself. Use it with
 * --frames N --screenshot FILE (render, save, exit) for tests on render nodes. */
#include "window.h"
#include <stdio.h>
#include <stdlib.h>

#ifndef _WIN32

struct Window { int w, h; };

Window* window_create(const char* title, int width, int height) {
    Window* w = (Window*)calloc(1, sizeof *w);
    if (!w) return NULL;
    w->w = width > 0 ? width : 1280;
    w->h = height > 0 ? height : 720;
    printf("headless: no display, rendering %dx%d off screen (%s)\n", w->w, w->h, title ? title : "");
    return w;
}

void window_destroy(Window* w) { free(w); }
int  window_pump(Window* w) { return w != NULL; }
void window_client_size(const Window* w, int* width, int* height) { *width = w->w; *height = w->h; }
void window_present(Window* w, const uint32_t* bgra, int width, int height) { (void)w; (void)bgra; (void)width; (void)height; }
void window_set_title(Window* w, const char* title) { (void)w; (void)title; }
int  window_key_down(const Window* w, int key) { (void)w; (void)key; return 0; }
int  window_key_pressed(const Window* w, int key) { (void)w; (void)key; return 0; }
void window_mouse_delta(Window* w, int* dx, int* dy) { (void)w; *dx = 0; *dy = 0; }
int  window_wheel(Window* w) { (void)w; return 0; }

#endif

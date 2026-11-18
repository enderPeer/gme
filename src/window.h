/* window.h - minimal native window: input + presenting a CPU-side BGRA frame. */
#ifndef GME_WINDOW_H
#define GME_WINDOW_H

#include <stdint.h>

#define KEY_SHIFT 0x10
#define KEY_CTRL  0x11
#define KEY_ESC   0x1B
#define KEY_SPACE 0x20
/* letters and digits use their ASCII upper-case code: 'W', 'A', '1' ... */

typedef struct Window Window;

Window* window_create(const char* title, int width, int height);
void    window_destroy(Window* w);
int     window_pump(Window* w);                 /* process events; 0 once closed */
void    window_client_size(const Window* w, int* width, int* height);
void    window_present(Window* w, const uint32_t* bgra, int width, int height);
void    window_set_title(Window* w, const char* title);
int     window_key_down(const Window* w, int key);
int     window_key_pressed(const Window* w, int key);  /* went down since last pump */
void    window_mouse_delta(Window* w, int* dx, int* dy); /* while right button held */
int     window_wheel(Window* w);                /* notches since last call */


#endif

/* image.h - tiny software image helpers + PNG writer (no dependencies). */
#ifndef GME_IMAGE_H
#define GME_IMAGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 24-bit PNG (uncompressed deflate) from a 0xAARRGGBB frame. 0 on success. */
int  write_png(const char* path, const uint32_t* argb, int width, int height);

typedef struct Image { int w, h; uint32_t* px; } Image;
int  img_create(Image* im, int w, int h, uint32_t fill);
void img_free(Image* im);
void img_blend(Image* im, int x, int y, uint32_t rgb, float alpha);
void img_disc(Image* im, float cx, float cy, float r, uint32_t rgb, float alpha);
void img_ring(Image* im, float cx, float cy, float r, uint32_t rgb, float alpha);

#ifdef __cplusplus
}
#endif
#endif

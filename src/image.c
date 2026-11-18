/* image.c - PNG writer (stored deflate, no zlib) + minimal raster helpers. */
#include "image.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t crc_table[256];

static uint32_t crc32_update(uint32_t c, const uint8_t* p, size_t n) {
    if (!crc_table[1]) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t v = i;
            for (int k = 0; k < 8; ++k) v = (v & 1) ? 0xEDB88320u ^ (v >> 1) : v >> 1;
            crc_table[i] = v;
        }
    }
    for (size_t i = 0; i < n; ++i) c = crc_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c;
}

static void put32(uint8_t* p, uint32_t v) { p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }

static void chunk(FILE* f, const char* type, const uint8_t* data, uint32_t len) {
    uint8_t b[4];
    put32(b, len);
    fwrite(b, 1, 4, f);
    fwrite(type, 1, 4, f);
    if (len) fwrite(data, 1, len, f);
    uint32_t c = crc32_update(0xFFFFFFFFu, (const uint8_t*)type, 4);
    if (len) c = crc32_update(c, data, len);
    put32(b, c ^ 0xFFFFFFFFu);
    fwrite(b, 1, 4, f);
}

int write_png(const char* path, const uint32_t* px, int w, int h) {
    const size_t row = (size_t)w * 3 + 1, raw_n = row * (size_t)h;
    uint8_t* raw = (uint8_t*)malloc(raw_n);
    if (!raw) return -1;
    for (int y = 0; y < h; ++y) {
        uint8_t* r = raw + row * (size_t)y;
        r[0] = 0;
        for (int x = 0; x < w; ++x) {
            const uint32_t c = px[(size_t)y * w + x];
            r[1 + 3 * x] = (uint8_t)(c >> 16);
            r[2 + 3 * x] = (uint8_t)(c >> 8);
            r[3 + 3 * x] = (uint8_t)c;
        }
    }
    const size_t blocks = (raw_n + 65534) / 65535;
    uint8_t* z = (uint8_t*)malloc(2 + blocks * 5 + raw_n + 4);
    if (!z) { free(raw); return -1; }
    size_t o = 0;
    z[o++] = 0x78; z[o++] = 0x01;
    uint32_t s1 = 1, s2 = 0;
    for (size_t i = 0; i < raw_n; i += 65535) {
        const size_t n = raw_n - i < 65535 ? raw_n - i : 65535;
        z[o++] = (uint8_t)(i + n == raw_n);
        z[o++] = (uint8_t)n; z[o++] = (uint8_t)(n >> 8);
        z[o++] = (uint8_t)~n; z[o++] = (uint8_t)(~n >> 8);
        memcpy(z + o, raw + i, n);
        for (size_t k = 0; k < n; ++k) { s1 = (s1 + raw[i + k]) % 65521u; s2 = (s2 + s1) % 65521u; }
        o += n;
    }
    put32(z + o, (s2 << 16) | s1);
    o += 4;

    FILE* f = fopen(path, "wb");
    if (!f) { free(raw); free(z); return -1; }
    static const uint8_t sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    fwrite(sig, 1, 8, f);
    uint8_t ihdr[13];
    put32(ihdr, (uint32_t)w);
    put32(ihdr + 4, (uint32_t)h);
    ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    chunk(f, "IHDR", ihdr, 13);
    chunk(f, "IDAT", z, (uint32_t)o);
    chunk(f, "IEND", NULL, 0);
    fclose(f);
    free(raw);
    free(z);
    return 0;
}

int img_create(Image* im, int w, int h, uint32_t fill) {
    im->w = w; im->h = h;
    im->px = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)w * h);
    if (!im->px) return -1;
    for (size_t i = 0; i < (size_t)w * h; ++i) im->px[i] = 0xFF000000u | fill;
    return 0;
}

void img_free(Image* im) { free(im->px); im->px = NULL; }

void img_blend(Image* im, int x, int y, uint32_t rgb, float a) {
    if (x < 0 || y < 0 || x >= im->w || y >= im->h || a <= 0.f) return;
    if (a > 1.f) a = 1.f;
    uint32_t* d = &im->px[(size_t)y * im->w + x];
    const uint32_t o = *d;
    uint32_t r = 0;
    for (int s = 0; s < 24; s += 8) {
        const float c = (float)((rgb >> s) & 255u) * a + (float)((o >> s) & 255u) * (1.f - a);
        r |= ((uint32_t)(c + 0.5f) & 255u) << s;
    }
    *d = 0xFF000000u | r;
}

void img_disc(Image* im, float cx, float cy, float r, uint32_t rgb, float a) {
    const int x0 = (int)floorf(cx - r - 1), x1 = (int)ceilf(cx + r + 1);
    const int y0 = (int)floorf(cy - r - 1), y1 = (int)ceilf(cy + r + 1);
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x) {
            const float d = sqrtf((x + 0.5f - cx) * (x + 0.5f - cx) + (y + 0.5f - cy) * (y + 0.5f - cy));
            const float cov = r + 0.5f - d; /* antialiased edge */
            if (cov > 0.f) img_blend(im, x, y, rgb, a * (cov > 1.f ? 1.f : cov));
        }
}

void img_ring(Image* im, float cx, float cy, float r, uint32_t rgb, float a) {
    const int n = (int)(r * 6.3f) + 16;
    for (int i = 0; i < n; ++i) {
        const float t = (float)i / n * 6.2831853f;
        img_blend(im, (int)(cx + r * cosf(t)), (int)(cy + r * sinf(t)), rgb, a);
    }
}

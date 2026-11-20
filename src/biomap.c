/* biomap.c - frozen vegetation snapshot (see biomap.h).
 *
 * File GMEBMAP1, little-endian, fixed 528-byte header then the cells:
 *    0  char[8] "GMEBMAP1"          8  u32 version (1)       12  u32 header bytes (528)
 *   16  u64 planet_hash            24  u32 seed             28  u32 sim_year
 *   32  u32 gates                  36  i32 nreg (0..8)      40  i32 shift (0..15)
 *   44  u32 fallback_on            48  8 x {i32 x0, z0, w, h; u32 offset}  (160 B)
 *  208  u32 fallback[64]          464  u8 palette[16][3]   512  u64 ncells
 *  520  u32 crc32 of the payload  524  u32 crc32 of bytes 0..523
 *  528  u32 cells[ncells]
 * File GMEBMAP2 (written when the biomap carries morphology), 832-byte header:
 *    0  char[8] "GMEBMAP2"          8  u32 version (2)       12  u32 header bytes (832)
 *   16 .. 519 as version 1 (520 = crc32 of the cells)
 *  524  u32 crc32 of the morph words
 *  528  u32 fallback_morph[64]    784  i32 atm_ppm[7]      812  u32 flags (bit 0: atm_ppm known)
 *  816  u32 reserved[3] (0)       828  u32 crc32 of bytes 0..827
 *  832  u32 cells[ncells], then u32 morph[ncells]
 * biomap_load reads both and returns 0 ok, -1 cannot open / short read / out
 * of memory, -2 not a valid header, -3 CRC mismatch (corrupt payload). */
#include "biomap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BIOMAP_VERSION   1u
#define BIOMAP_HDR       528u
#define BIOMAP_VERSION2  2u
#define BIOMAP_HDR2      832u
#define BIOMAP_MAX_CELLS (1ull << 28)   /* sanity limit: 1 GiB of cells */
#define IO_CELLS         65536u         /* cells per read/write block */

/* ---- crc32 (IEEE 802.3, reflected) ----------------------------------------- */
static void crc_table(uint32_t T[256]) {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        T[i] = c;
    }
}

static uint32_t crc_update(const uint32_t T[256], uint32_t crc, const uint8_t* p, size_t n) {
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = T[(crc ^ p[i]) & 255u] ^ (crc >> 8);
    return ~crc;
}

/* ---- little-endian fields ---------------------------------------------------- */
static void put32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
static void put64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
static uint32_t get32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t get64(const uint8_t* p) { return (uint64_t)get32(p) | ((uint64_t)get32(p + 4) << 32); }

/* regions must lie inside the cell array; w, h <= 65536 cells */
static int regions_ok(int32_t nreg, int32_t shift, const GmeBioRegion* reg, uint64_t ncells) {
    if (nreg < 0 || nreg > BIOMAP_MAX_REGIONS || shift < 0 || shift > 15) return 0;
    for (int r = 0; r < nreg; ++r) {
        if (reg[r].w <= 0 || reg[r].h <= 0 || reg[r].w > 65536 || reg[r].h > 65536) return 0;
        if ((uint64_t)reg[r].offset + (uint64_t)reg[r].w * (uint64_t)reg[r].h > ncells) return 0;
    }
    return 1;
}

/* crc32 of n words as little-endian bytes */
static uint32_t crc_words(const uint32_t T[256], const uint32_t* w, uint64_t n, uint8_t* blk) {
    uint32_t c = 0;
    for (uint64_t i = 0; i < n; i += IO_CELLS) {
        const size_t m = (size_t)(n - i < IO_CELLS ? n - i : IO_CELLS);
        for (size_t k = 0; k < m; ++k) put32(blk + 4 * k, w[i + k]);
        c = crc_update(T, c, blk, m * 4);
    }
    return c;
}

static int write_words(FILE* f, const uint32_t* w, uint64_t n, uint8_t* blk) {
    for (uint64_t i = 0; i < n; i += IO_CELLS) {
        const size_t m = (size_t)(n - i < IO_CELLS ? n - i : IO_CELLS);
        for (size_t k = 0; k < m; ++k) put32(blk + 4 * k, w[i + k]);
        if (fwrite(blk, 4, m, f) != m) return 0;
    }
    return 1;
}

/* reads n little-endian words and returns their crc32 in *crc; 1 ok */
static int read_words(FILE* f, const uint32_t T[256], uint32_t* w, uint64_t n, uint8_t* blk, uint32_t* crc) {
    uint32_t c = 0;
    for (uint64_t i = 0; i < n; i += IO_CELLS) {
        const size_t m = (size_t)(n - i < IO_CELLS ? n - i : IO_CELLS);
        if (fread(blk, 4, m, f) != m) return 0;
        c = crc_update(T, c, blk, m * 4);
        for (size_t k = 0; k < m; ++k) w[i + k] = get32(blk + 4 * k);
    }
    *crc = c;
    return 1;
}

int biomap_save(const char* path, const Biomap* b) {
    if (!path || !b) return -1;
    const uint64_t ncells = b->cells ? (uint64_t)b->ncells : 0;
    if (ncells > BIOMAP_MAX_CELLS || !regions_ok(b->nreg, b->shift, b->reg, ncells)) return -2;
    const int v2 = b->morph != NULL;
    uint32_t T[256];
    crc_table(T);

    uint8_t* blk = (uint8_t*)malloc((size_t)IO_CELLS * 4);
    if (!blk) return -1;
    const uint32_t pcrc = crc_words(T, b->cells, ncells, blk);

    uint8_t hd[BIOMAP_HDR2];
    const size_t hlen = v2 ? BIOMAP_HDR2 : BIOMAP_HDR;
    memset(hd, 0, sizeof hd);
    memcpy(hd, v2 ? BIOMAP_MAGIC2 : BIOMAP_MAGIC, 8);
    put32(hd + 8, v2 ? BIOMAP_VERSION2 : BIOMAP_VERSION);
    put32(hd + 12, (uint32_t)hlen);
    put64(hd + 16, b->planet_hash);
    put32(hd + 24, b->seed);
    put32(hd + 28, b->sim_year);
    put32(hd + 32, b->gates);
    put32(hd + 36, (uint32_t)b->nreg);
    put32(hd + 40, (uint32_t)b->shift);
    put32(hd + 44, b->fallback_on ? 1u : 0u);
    for (int r = 0; r < b->nreg; ++r) {
        uint8_t* q = hd + 48 + 20 * r;
        put32(q, (uint32_t)b->reg[r].x0); put32(q + 4, (uint32_t)b->reg[r].z0);
        put32(q + 8, (uint32_t)b->reg[r].w); put32(q + 12, (uint32_t)b->reg[r].h);
        put32(q + 16, b->reg[r].offset);
    }
    for (int i = 0; i < BIOMAP_FALLBACK; ++i) put32(hd + 208 + 4 * i, b->fallback[i]);
    memcpy(hd + 464, b->palette, BIOMAP_PALETTE * 3);
    put64(hd + 512, ncells);
    put32(hd + 520, pcrc);
    if (v2) {
        put32(hd + 524, crc_words(T, b->morph, ncells, blk));
        for (int i = 0; i < BIOMAP_FALLBACK; ++i) put32(hd + 528 + 4 * i, b->fallback_morph[i]);
        for (int i = 0; i < BIOMAP_ATM; ++i) put32(hd + 784 + 4 * i, (uint32_t)b->atm_ppm[i]);
        put32(hd + 812, b->atm_known ? 1u : 0u);
        put32(hd + 828, crc_update(T, 0, hd, 828));
    } else {
        put32(hd + 524, crc_update(T, 0, hd, 524));
    }

    /* write a temporary file, then replace: readers never see a partial biomap */
    const size_t pl = strlen(path);
    char* tmp = (char*)malloc(pl + 5);
    if (!tmp) { free(blk); return -1; }
    memcpy(tmp, path, pl);
    memcpy(tmp + pl, ".tmp", 5);
    FILE* f = fopen(tmp, "wb");
    int ok = f != NULL;
    if (ok) ok = fwrite(hd, 1, hlen, f) == hlen;
    if (ok) ok = write_words(f, b->cells, ncells, blk);
    if (ok && v2) ok = write_words(f, b->morph, ncells, blk);
    if (f && fclose(f)) ok = 0;
    free(blk);
    if (ok) {
        remove(path);
        ok = rename(tmp, path) == 0;
    }
    if (!ok) remove(tmp);
    free(tmp);
    return ok ? 0 : -1;
}

int biomap_load(const char* path, Biomap* b) {
    memset(b, 0, sizeof *b);
    FILE* f = path ? fopen(path, "rb") : NULL;
    if (!f) return -1;
    uint32_t T[256];
    crc_table(T);
    uint8_t hd[BIOMAP_HDR2];
    int rc = -2;
    int v2 = 0;
    uint8_t* blk = NULL;
    uint32_t* cells = NULL;
    uint32_t* morph = NULL;
    if (fread(hd, 1, BIOMAP_HDR, f) != BIOMAP_HDR) goto fail;
    v2 = !memcmp(hd, BIOMAP_MAGIC2, 8);
    if (v2) {
        if (get32(hd + 8) != BIOMAP_VERSION2 || get32(hd + 12) != BIOMAP_HDR2) goto fail;
        if (fread(hd + BIOMAP_HDR, 1, BIOMAP_HDR2 - BIOMAP_HDR, f) != BIOMAP_HDR2 - BIOMAP_HDR) goto fail;
        if (get32(hd + 828) != crc_update(T, 0, hd, 828)) goto fail;
    } else {
        if (memcmp(hd, BIOMAP_MAGIC, 8) || get32(hd + 8) != BIOMAP_VERSION || get32(hd + 12) != BIOMAP_HDR) goto fail;
        if (get32(hd + 524) != crc_update(T, 0, hd, 524)) goto fail;
    }

    Biomap m;
    memset(&m, 0, sizeof m);
    m.planet_hash = get64(hd + 16);
    m.seed = get32(hd + 24);
    m.sim_year = get32(hd + 28);
    m.gates = get32(hd + 32);
    m.nreg = (int32_t)get32(hd + 36);
    m.shift = (int32_t)get32(hd + 40);
    m.fallback_on = get32(hd + 44) ? 1u : 0u;
    if (m.nreg < 0 || m.nreg > BIOMAP_MAX_REGIONS) goto fail;
    for (int r = 0; r < m.nreg; ++r) {
        const uint8_t* q = hd + 48 + 20 * r;
        m.reg[r].x0 = (int32_t)get32(q); m.reg[r].z0 = (int32_t)get32(q + 4);
        m.reg[r].w = (int32_t)get32(q + 8); m.reg[r].h = (int32_t)get32(q + 12);
        m.reg[r].offset = get32(q + 16);
    }
    for (int i = 0; i < BIOMAP_FALLBACK; ++i) m.fallback[i] = get32(hd + 208 + 4 * i);
    memcpy(m.palette, hd + 464, BIOMAP_PALETTE * 3);
    if (v2) {
        for (int i = 0; i < BIOMAP_FALLBACK; ++i) m.fallback_morph[i] = get32(hd + 528 + 4 * i);
        for (int i = 0; i < BIOMAP_ATM; ++i) m.atm_ppm[i] = (int32_t)get32(hd + 784 + 4 * i);
        m.atm_known = get32(hd + 812) & 1u;
    }
    const uint64_t ncells = get64(hd + 512);
    if (ncells > BIOMAP_MAX_CELLS || !regions_ok(m.nreg, m.shift, m.reg, ncells)) goto fail;

    rc = -1;
    cells = (uint32_t*)malloc((size_t)(ncells ? ncells : 1) * sizeof(uint32_t));
    blk = (uint8_t*)malloc((size_t)IO_CELLS * 4);
    if (!cells || !blk) goto fail;
    uint32_t pcrc = 0, mcrc = 0;
    if (!read_words(f, T, cells, ncells, blk, &pcrc)) goto fail;
    if (v2) {
        morph = (uint32_t*)malloc((size_t)(ncells ? ncells : 1) * sizeof(uint32_t));
        if (!morph || !read_words(f, T, morph, ncells, blk, &mcrc)) goto fail;
    }
    rc = -2;
    if (fgetc(f) != EOF) goto fail;              /* trailing bytes: not what we wrote */
    rc = -3;
    if (pcrc != get32(hd + 520)) goto fail;
    if (v2 && mcrc != get32(hd + 524)) goto fail;

    fclose(f);
    free(blk);
    m.cells = cells;
    m.morph = morph;
    m.ncells = (size_t)ncells;
    *b = m;
    return 0;
fail:
    fclose(f);
    free(blk);
    free(cells);
    free(morph);
    return rc;
}

void biomap_free(Biomap* b) {
    if (!b) return;
    free(b->cells);
    free(b->morph);
    memset(b, 0, sizeof *b);
}

void biomap_view(const Biomap* b, GmeBioView* o) {
    memset(o, 0, sizeof *o);
    if (!b) return;
    const uint64_t ncells = b->cells ? (uint64_t)b->ncells : 0;
    if (ncells && regions_ok(b->nreg, b->shift, b->reg, ncells)) {   /* never hand out-of-range regions to a kernel */
        o->cells = b->cells;
        o->nreg = b->nreg;
        o->shift = b->shift;
        memcpy(o->reg, b->reg, sizeof(GmeBioRegion) * (size_t)b->nreg);
        o->morph = b->morph;                      /* NULL for a version 1 biomap: version 1 shapes */
    }
    o->seed = b->seed;
    o->fallback_on = b->fallback_on ? 1u : 0u;
    memcpy(o->fallback, b->fallback, sizeof o->fallback);
    if (b->morph) memcpy(o->fallback_morph, b->fallback_morph, sizeof o->fallback_morph);
}

/* Integer HSV wheel: index i = 1..15 has hue (i - 1) * 24 degrees (1 red,
 * 6 green, 11 blue, 13 purple), saturation 150/255, value 176/255. */
void biomap_default_palette(uint8_t pal[BIOMAP_PALETTE][3]) {
    pal[0][0] = pal[0][1] = pal[0][2] = 0;
    const int v = 176, s = 150;
    for (int i = 1; i < BIOMAP_PALETTE; ++i) {
        const int hue = (i - 1) * 1536 / 15;             /* 0..1535: six 256-step sectors */
        const int sec = hue >> 8, f = hue & 255;
        const int lo = v * (255 - s) / 255;
        const int dn = v * (255 - s * f / 255) / 255;
        const int up = v * (255 - s * (255 - f) / 255) / 255;
        int r, g, b;
        switch (sec) {
        case 0:  r = v;  g = up; b = lo; break;
        case 1:  r = dn; g = v;  b = lo; break;
        case 2:  r = lo; g = v;  b = up; break;
        case 3:  r = lo; g = dn; b = v;  break;
        case 4:  r = up; g = lo; b = v;  break;
        default: r = v;  g = lo; b = dn; break;
        }
        pal[i][0] = (uint8_t)r; pal[i][1] = (uint8_t)g; pal[i][2] = (uint8_t)b;
    }
}

/* Test pattern for T11: one 1024 x 1024-cell region (16 m cells, 16.4 km)
 * centred on (cx, cz), fallback off. Blocks of 2 x 2 cells (32 m) cycle through
 * the kinds none, mat, kelp, cover, shrub, tree; the palette index (0..15)
 * changes every 4 x 4 blocks, the cover (72..255) every few blocks, and
 * heights vary per block within each kind's range. Which kinds show depends
 * on the terrain: mats and kelp in water and on beaches, the rest on land. */
int biomap_make_test(Biomap* b, uint64_t planet_hash, uint32_t seed, int32_t cx, int32_t cz) {
    memset(b, 0, sizeof *b);
    const int32_t n = 1024, shift = 4, size = n << shift;
    uint32_t* cells = (uint32_t*)malloc((size_t)n * n * sizeof(uint32_t));
    if (!cells) return -1;
    static const uint32_t covers[4] = { 255, 192, 128, 72 };
    for (int32_t j = 0; j < n; ++j) {
        for (int32_t i = 0; i < n; ++i) {
            const int32_t bi = i >> 1, bj = j >> 1;    /* 32 m blocks */
            const uint32_t kind = (uint32_t)(bi + 2 * bj) % 6u;
            const uint32_t pal = (uint32_t)((bi >> 2) + 5 * (bj >> 2)) % 16u;
            const uint32_t cover = covers[(bi / 6 + bj / 3) % 4];
            const uint32_t var = (uint32_t)(bi * 7 + bj * 13) % 16u;
            uint32_t height = 0;
            switch (kind) {
            case BIO_KIND_KELP:  height = 3 + var; break;          /* 3..18 m */
            case BIO_KIND_SHRUB: height = 1 + (var & 3u); break;   /* 1..4 m */
            case BIO_KIND_TREE:  height = 5 + var; break;          /* 5..20 m */
            default: break;
            }
            cells[(size_t)j * n + i] = kind ? BIOCELL(kind, pal, cover, height, 0) : 0u;
        }
    }
    b->planet_hash = planet_hash;
    b->seed = seed;
    b->nreg = 1;
    b->shift = shift;
    b->fallback_on = 0;
    b->reg[0].x0 = ((cx - size / 2) >> 4) << 4;   /* 16 m aligned */
    b->reg[0].z0 = ((cz - size / 2) >> 4) << 4;
    b->reg[0].w = n;
    b->reg[0].h = n;
    b->reg[0].offset = 0;
    biomap_default_palette(b->palette);
    b->cells = cells;
    b->ncells = (size_t)n * n;
    return 0;
}

/* The T11 pattern as version 2: every 32 m block also gets a morphology word.
 * The crown class changes with each repeat of the six kinds along x and the
 * aspect every three blocks along z, so a walk across the pattern shows all
 * 5 x 8 shapes; leaf angle, porosity, pigment amount and Q band width vary on
 * coarser grids. */
int biomap_make_test2(Biomap* b, uint64_t planet_hash, uint32_t seed, int32_t cx, int32_t cz) {
    if (biomap_make_test(b, planet_hash, seed, cx, cz)) return -1;
    const int32_t n = b->reg[0].w;
    uint32_t* morph = (uint32_t*)malloc(b->ncells * sizeof(uint32_t));
    if (!morph) { biomap_free(b); return -1; }
    for (int32_t j = 0; j < n; ++j) {
        for (int32_t i = 0; i < n; ++i) {
            const int32_t bi = i >> 1, bj = j >> 1;    /* the pattern's 32 m blocks */
            const uint32_t crown = (uint32_t)(bi / 6) % (uint32_t)BIO_CROWN_N;
            const uint32_t aspect = (uint32_t)(bj / 3) % 8u;
            const uint32_t leafang = (uint32_t)(bi / 30 + bj / 24) % 4u;
            const uint32_t porosity = (uint32_t)(bi / 12 + bj / 12) % 4u;
            const uint32_t pig = 96u + 53u * ((uint32_t)(bi / 8 + bj / 8) % 4u);   /* 96 .. 255 */
            const uint32_t qw = (uint32_t)(bi / 16 + 3 * (bj / 16)) % 16u;
            morph[(size_t)j * n + i] = b->cells[(size_t)j * n + i] ? BIOMORPH(crown, aspect, leafang, porosity, pig, qw) : 0u;
        }
    }
    b->morph = morph;
    return 0;
}

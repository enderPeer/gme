/* density.h - the smooth terrain surface T(x, z) and the per-chunk-column
 * tiles that store it (docs/render-upgrade-spec.md sections 1.2, 2 and 3.5).
 *
 * GME_HD and integer-only, shared by the CPU (world.c: cpu tiles, validation,
 * dens_hash; main.c: self-tests D1/D2) and CUDA (gpu.cu: k_col_tiles;
 * trace.cuh: the column tracer), so CPU and GPU tiles are bit-identical.
 * Includes worldgen.h and flora.h and edits neither; gme_genesis and
 * gme_biotest never include it, so the voxel truth (gme_height, gme_material,
 * gme_voxel) and every life file stay untouched.
 *
 *   T(x, z)  the generator's own height expression without its intermediate
 *            floors, mean-shifted to h + 1, plus +-0.25 m micro relief, then
 *            clamped into [h, h + 2) and coast-locked: a column is drawn wet
 *            (T < sea_level + 1) iff its voxel column is wet (h < sea_level).
 *   storage  2 B per column: h - h_base (uint8) and the deviation
 *            T - (h + 1) in 1/128 m (int8), in one 3,456 B GmeColTile per
 *            chunk column with a 2-column apron (36 x 36 columns). */
#ifndef GME_DENSITY_H
#define GME_DENSITY_H

#include "worldgen.h"
#include "flora.h"

#ifdef __cplusplus
#define GME_STATIC_ASSERT(c, m) static_assert(c, m)
#else
#define GME_STATIC_ASSERT(c, m) _Static_assert(c, m)
#endif

#define GME_TILE_APRON 2
#define GME_TILE_N     36          /* columns x0-2 .. x0+33: bilinear corners + central-difference normals */
#define GME_TILE_COLS  (GME_TILE_N * GME_TILE_N)
#define GME_DEV_UNIT   128         /* deviation T-(h+1) in 1/128 m, int8 => T in [h, h+2) */

/* GmeColTile.flags */
#define GME_TILE_WET       1u      /* any wet column (h < sea_level) */
#define GME_TILE_TREES     2u      /* any shrub / tree slot */
#define GME_TILE_KELP      4u      /* any kelp slot */
#define GME_TILE_SATURATED 0x8000u /* h range > 255 inside the tile: must stay 0 */

/* Nothing in the world reaches above this (terrain + T margin + flora). */
#define GME_GLOBAL_TOP (GME_HEIGHT_MAX + 2 + GME_FLORA_REACH)

typedef struct GmeColTile {        /* one per chunk column (all layers); 3,456 B */
    uint32_t tag;                  /* gme_page_tag(cx, cz) (informational; publication is host-side) */
    int32_t  cx, cz;
    int16_t  h_base;               /* min gme_height over the 36x36 columns */
    uint16_t flags;                /* GME_TILE_* */
    int16_t  top, bot;             /* whole tile: max(T, flora top, water plane) ceil / min T floor, metres */
    int16_t  blk_top[16], blk_bot[16];   /* per 8 m block (= flora slot), row-major z*4+x, incl. 1-column overlap */
    GmeSlot  slot[16];             /* gme_slot() of the 4x4 slots, exactly as flora.h computes it (int32 fields) */
    uint16_t col[GME_TILE_COLS];   /* bits 0-7: h - h_base; bits 8-15: int8 dev q7. Index (z-z0)*36 + (x-x0), x0 = cx*32-2 */
    int16_t  lod4[9 * 9];          /* M3: T box-averaged over 4x4 columns at the 4 m lattice, q4 m; 0 in M1 */
    uint8_t  pad[106];
} GmeColTile;
GME_STATIC_ASSERT(sizeof(GmeColTile) == 3456, "GmeColTile must be 3456 bytes");

/* index = gme_page_index(cx, 0, cz, page_n, 1) */
typedef struct GmeTileJob { int32_t cx, cz; uint32_t index, pad; } GmeTileJob;

/* Publication key of a chunk column's tile in the world's tile_tag table.
 * Never 0 (0 = no tile), unlike gme_page_tag, which is 0 for column (0, 0).
 * Columns sharing a toroidal cell differ by multiples of page_n << 32768,
 * so their keys differ. */
GME_HD uint32_t gme_tile_key(int32_t cx, int32_t cz) {
    return 0x80000000u | ((uint32_t)cx & 0x7FFFu) | (((uint32_t)cz & 0x7FFFu) << 15);
}

/* ---- the surface ---------------------------------------------------------- */
/* h (== gme_height) and the continuous height T (q8 m) from one evaluation of
 * the noise: T is gme_height's expression without its three floors, shifted so
 * that mean(T) = h + 1, plus +-0.25 m micro relief. */
GME_HD void gme_surface_pair(int32_t x, int32_t z, uint32_t seed, const GmePlanet* p, int32_t* h_out, int32_t* t_q8) {
    const int32_t cont = gme_noise2(x, z, 11, seed ^ 0xA511E9B3u), det = gme_fbm2(x, z, 8, 5, seed);
    const int32_t mount = cont > 0 ? cont : 0, amp = 20 + ((mount * 110) >> 15);
    int32_t h = ((((cont * 80) >> 15) + (int32_t)(((int64_t)det * amp) >> 16)) * p->relief_q8) >> 8;  /* == gme_height */
    h = h > GME_HEIGHT_MAX ? GME_HEIGHT_MAX : (h < -GME_HEIGHT_MAX ? -GME_HEIGHT_MAX : h);
    int64_t t = (((int64_t)cont * 80) >> 7) + (((int64_t)det * (20 * 32768 + (int64_t)mount * 110)) >> 23); /* a+b, q8 */
    t = ((t * p->relief_q8) >> 8) - (p->relief_q8 - 128);            /* mean(T) = h + 1 */
    t += gme_noise2(x, z, 2, seed ^ 0xD3A5u) >> 9;                   /* +-0.25 m micro relief */
    *h_out = h;
    *t_q8 = (int32_t)t;
}

/* Stored deviation T - (h+1) in 1/128 m: rounded, clamped so T lies in
 * [h, h+2), and coast-locked: a dry column (h >= sea_level) keeps
 * T >= sea_level + 1 + 1/128 (above the water plane); wet columns satisfy
 * T < h + 2 <= sea_level + 1 already. */
GME_HD int32_t gme_tile_dev_q7(int32_t t_q8, int32_t h, const GmePlanet* p) {
    int32_t dev = (t_q8 - 256 * (h + 1) + 1) >> 1, lo = -128;        /* q7, rounded */
    if (h >= p->sea_level) { const int32_t c = 128 * (p->sea_level - h) + 1; if (c > lo) lo = c; }  /* coast lock */
    return dev < lo ? lo : (dev > 127 ? 127 : dev);                   /* T in [h, h+2) */
}

/* What is drawn (and what walking should use): T in q8 m. */
GME_HD int32_t gme_surface_q8(int32_t x, int32_t z, uint32_t seed, const GmePlanet* p) {
    int32_t h, t;
    gme_surface_pair(x, z, seed, p, &h, &t);
    return 256 * (h + 1) + 2 * gme_tile_dev_q7(t, h, p);
}

GME_HD void gme_col_tile_build_column(int32_t x, int32_t z, uint32_t seed, const GmePlanet* p, int32_t* h, int32_t* dev) {
    int32_t t;
    gme_surface_pair(x, z, seed, p, h, &t);
    *dev = gme_tile_dev_q7(t, *h, p);
}

/* ---- tile encoding -------------------------------------------------------- */
GME_HD int32_t gme_tile_h(const GmeColTile* t, int i)    { return (int32_t)t->h_base + (int32_t)(t->col[i] & 255u); }
GME_HD int32_t gme_tile_dev(const GmeColTile* t, int i)  { return (int32_t)(((uint32_t)t->col[i] >> 8) ^ 0x80u) - 0x80; }
GME_HD int32_t gme_tile_t_q8(const GmeColTile* t, int i) { return 256 * (gme_tile_h(t, i) + 1) + 2 * gme_tile_dev(t, i); }

/* Clears the record (padding and lod4 included, so CPU and GPU bytes match). */
GME_HD void gme_col_tile_init(GmeColTile* t, int32_t cx, int32_t cz) {
    uint32_t* w = (uint32_t*)t;
    for (unsigned i = 0; i < sizeof(GmeColTile) / 4; ++i) w[i] = 0u;
    t->tag = gme_page_tag(cx, cz);
    t->cx = cx;
    t->cz = cz;
}

/* min / max of h over the 36x36 columns */
GME_HD void gme_tile_h_range(const int32_t* h36, int32_t* hmin, int32_t* hmax) {
    int32_t mn = h36[0], mx = h36[0];
    for (int i = 1; i < GME_TILE_COLS; ++i) { mn = h36[i] < mn ? h36[i] : mn; mx = h36[i] > mx ? h36[i] : mx; }
    *hmin = mn;
    *hmax = mx;
}

/* column i (needs t->h_base); a saturated offset is clamped (flagged by the header) */
GME_HD void gme_tile_pack_col(GmeColTile* t, int i, int32_t h, int32_t dev) {
    int32_t o = h - (int32_t)t->h_base;
    o = o < 0 ? 0 : (o > 255 ? 255 : o);
    t->col[i] = (uint16_t)((uint32_t)o | (((uint32_t)dev & 255u) << 8));
}

/* Block b (0..15, z*4+x): its flora slot and bounds over the columns of the
 * block plus a 1-column overlap (every dual cell touching the block).
 * Returns the block's GME_TILE_WET / TREES / KELP flags. */
GME_HD uint32_t gme_tile_block(GmeColTile* t, int b, const int32_t* h36, const int32_t* dev36,
                               uint32_t seed, const GmePlanet* p, const GmeBioView* bv) {
    const int bx = b & 3, bz = b >> 2;
    GmeSlot* s = &t->slot[b];
    gme_slot(t->cx * 4 + bx, t->cz * 4 + bz, seed, p, bv, s);
    int32_t tmin = 0x7FFFFFFF, tmax = -0x7FFFFFFF - 1;
    uint32_t fl = 0u;
    for (int j = 8 * bz + 1; j <= 8 * bz + 10; ++j)
        for (int i = 8 * bx + 1; i <= 8 * bx + 10; ++i) {
            const int k = j * GME_TILE_N + i;
            const int32_t tq = 256 * (h36[k] + 1) + 2 * dev36[k];
            tmin = tq < tmin ? tq : tmin;
            tmax = tq > tmax ? tq : tmax;
            if (h36[k] < p->sea_level) fl |= GME_TILE_WET;
        }
    int32_t top = (tmax + 255) >> 8, bot = tmin >> 8;              /* ceil / floor, metres */
    if (fl & GME_TILE_WET) top = top > p->sea_level + 1 ? top : p->sea_level + 1;   /* water plane */
    if (s->trad > 0) {
        const int32_t ft = s->tcy + s->tvr + 2;                    /* crown top + |dev| <= 1 + rounding */
        top = top > ft ? top : ft;
        fl |= GME_TILE_TREES;
    }
    if (BIOCELL_KIND(s->cell) == BIO_KIND_KELP) {
        top = top > p->sea_level + 1 ? top : p->sea_level + 1;
        fl |= GME_TILE_KELP;
    }
    t->blk_top[b] = (int16_t)top;
    t->blk_bot[b] = (int16_t)bot;
    return fl;
}

/* whole-tile bounds and flags from the 16 blocks */
GME_HD void gme_tile_header(GmeColTile* t, const uint32_t* block_flags, int saturated) {
    int32_t top = t->blk_top[0], bot = t->blk_bot[0];
    uint32_t fl = saturated ? GME_TILE_SATURATED : 0u;
    for (int b = 0; b < 16; ++b) {
        top = t->blk_top[b] > top ? t->blk_top[b] : top;
        bot = t->blk_bot[b] < bot ? t->blk_bot[b] : bot;
        fl |= block_flags[b];
    }
    t->top = (int16_t)top;
    t->bot = (int16_t)bot;
    t->flags = (uint16_t)fl;
}

/* Bounds, slots and (from M3) lod4 from the 36x36 per-column values.
 * t must come from gme_col_tile_init. */
GME_HD void gme_col_tile_finish(GmeColTile* t, const int32_t* h36, const int32_t* dev36,
                                uint32_t seed, const GmePlanet* p, const GmeBioView* b) {
    int32_t hmin, hmax;
    gme_tile_h_range(h36, &hmin, &hmax);
    t->h_base = (int16_t)hmin;
    for (int i = 0; i < GME_TILE_COLS; ++i) gme_tile_pack_col(t, i, h36[i], dev36[i]);
    uint32_t fl[16];
    for (int k = 0; k < 16; ++k) fl[k] = gme_tile_block(t, k, h36, dev36, seed, p, b);
    gme_tile_header(t, fl, hmax - hmin > 255);
}

/* The tile of chunk column (cx, cz), serially (CPU reference; the GPU builds
 * the same bytes in parallel in k_col_tiles). Uses ~10 KB of stack. */
GME_HD void gme_col_tile_build(GmeColTile* t, int32_t cx, int32_t cz, uint32_t seed, const GmePlanet* p,
                               const GmeBioView* b) {
    int32_t h36[GME_TILE_COLS], dev36[GME_TILE_COLS];
    const int32_t x0 = cx * GME_CHUNK_DIM - GME_TILE_APRON, z0 = cz * GME_CHUNK_DIM - GME_TILE_APRON;
    for (int j = 0; j < GME_TILE_N; ++j)
        for (int i = 0; i < GME_TILE_N; ++i)
            gme_col_tile_build_column(x0 + i, z0 + j, seed, p, &h36[j * GME_TILE_N + i], &dev36[j * GME_TILE_N + i]);
    gme_col_tile_init(t, cx, cz);
    gme_col_tile_finish(t, h36, dev36, seed, p, b);
}

/* ---- analytic flora: float twins of gme_voxel's integer tests (D2) --------- */
/* Trunk radius: contains only the trunk column's voxel centres. */
GME_HD float gme_trunk_radius(int32_t trad) { return 0.18f + 0.06f * (float)trad; }

/* Crown: ellipsoid (dx^2 + dz^2) / (r^2 + r) + dy^2 / (v^2 + v) <= 1 around
 * (tx+.5, tcy+.5, tz+.5), the same inequality gme_voxel tests, in floats. */
GME_HD int gme_crown_inside_f(float dx, float dy, float dz, int32_t trad, int32_t tvr) {
    const float rr = (float)(trad * trad + trad), vv = (float)(tvr * tvr + tvr);
    return (dx * dx + dz * dz) * vv + dy * dy * rr <= rr * vv;
}

/* Trunk: vertical cylinder around (tx+.5, tz+.5) from y = tbase+1 to tcy+1
 * (flat ground; the renderer shifts the whole tree by the trunk column's dev). */
GME_HD int gme_trunk_inside_f(float dx, float y, float dz, const GmeSlot* s) {
    const float r = gme_trunk_radius(s->trad);
    return dx * dx + dz * dz <= r * r && y >= (float)(s->tbase + 1) && y <= (float)(s->tcy + 1);
}

/* Kelp strand length of column (x, z) of height h in a slot of cell `cell`:
 * the same integer rules as gme_voxel; KELP voxels are h < y <= h + len.
 * 0 = no strand. */
GME_HD int32_t gme_kelp_len(int32_t x, int32_t z, int32_t h, uint32_t cell, uint32_t seed, const GmePlanet* p) {
    if (BIOCELL_KIND(cell) != BIO_KIND_KELP || p->ocean_mat != MAT_WATER) return 0;
    const int32_t sea = p->sea_level;
    if (h >= sea - 1) return 0;                                    /* no water voxel below sea_level */
    const uint32_t colhash = gme_hash3(x, 0, z, seed ^ 0xF10Au);
    const uint32_t cover = (cell >> 8) & 255u;
    if (((colhash >> 16) & 255u) >= (cover >> 1)) return 0;
    int32_t len = (int32_t)((cell >> 16) & 255u);
    if (len > sea - h - 1) len = sea - h - 1;
    return (int32_t)(((uint32_t)len * (128u + ((colhash >> 24) & 127u))) >> 8);
}

/* Kelp capsule of radius GME_KELP_R whose total extent on the column axis is
 * [y0, y1] (y0 = the column's T, y1 = h + len + 1): segment endpoints are
 * inset by the radius. dx, dz: offsets from the column axis (x+.5, z+.5). */
#define GME_KELP_R 0.2f
GME_HD int gme_kelp_inside_f(float dx, float y, float dz, float y0, float y1) {
    float a = y0 + GME_KELP_R, b = y1 - GME_KELP_R;
    if (a > b) a = b = 0.5f * (y0 + y1);                           /* shorter than its diameter: a sphere */
    const float yc = y < a ? a : (y > b ? b : y);
    return dx * dx + dz * dz + (y - yc) * (y - yc) <= GME_KELP_R * GME_KELP_R;
}

#endif

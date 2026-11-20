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
 *            chunk column with a 2-column apron (36 x 36 columns).
 *   M3       lod4: T box-averaged on the tile's 9 x 9 lattice at 4 m (the LOD
 *            surface) with its bounds; gme_far_sample: one sample of the
 *            renderer's far-field clipmap (16 m / 64 m). */
#ifndef GME_DENSITY_H
#define GME_DENSITY_H

#include <math.h>
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
    int16_t  lod4[9 * 9];          /* M3: T box-averaged over 4x4 columns at the 4 m lattice (cx*32 + 4i, cz*32 + 4j), q4 m */
    int16_t  lod_top, lod_bot;     /* M3: ceil / floor (metres) of lod4 over the whole tile */
    int16_t  lod_btop[16];         /* M3: ceil (metres) of lod4 over each 8 m block's 3 x 3 lattice points */
    uint8_t  pad[70];
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

/* ---- M3: the 4 m LOD lattice (spec 3.8) ------------------------------------- */
/* lod4 at lattice point (i, j), i, j in 0..8, i.e. at (cx*32 + 4i, cz*32 + 4j):
 * the box average of T over the 4 x 4 columns around the point (columns
 * 4i-2 .. 4i+1 of the tile = apron indices 4i .. 4i+3), in 1/16 m, rounded.
 * Corner points sit on the tile edges and use the same columns in both
 * neighbouring tiles, so they share their samples exactly. */
#define GME_LOD_N 9
GME_HD int32_t gme_tile_lod4_at(const int32_t* h36, const int32_t* dev36, int i, int j) {
    int32_t s = 0;
    for (int b = 0; b < 4; ++b)
        for (int a = 0; a < 4; ++a) {
            const int k = (4 * j + b) * GME_TILE_N + 4 * i + a;
            s += 256 * (h36[k] + 1) + 2 * dev36[k];
        }
    return (s + 128) >> 8;                                         /* 16 values of q8 -> mean in q4 */
}

/* lod_top / lod_bot of the tile and lod_btop of each block from lod4 */
GME_HD void gme_tile_lod_bounds(GmeColTile* t) {
    int32_t mn = t->lod4[0], mx = t->lod4[0];
    for (int k = 1; k < GME_LOD_N * GME_LOD_N; ++k) {
        mn = t->lod4[k] < mn ? t->lod4[k] : mn;
        mx = t->lod4[k] > mx ? t->lod4[k] : mx;
    }
    t->lod_top = (int16_t)((mx + 15) >> 4);                        /* ceil / floor of q4 -> metres */
    t->lod_bot = (int16_t)(mn >> 4);
    for (int b = 0; b < 16; ++b) {
        const int i0 = 2 * (b & 3), j0 = 2 * (b >> 2);
        int32_t bm = t->lod4[j0 * GME_LOD_N + i0];
        for (int j = j0; j <= j0 + 2; ++j)
            for (int i = i0; i <= i0 + 2; ++i) bm = t->lod4[j * GME_LOD_N + i] > bm ? t->lod4[j * GME_LOD_N + i] : bm;
        t->lod_btop[b] = (int16_t)((bm + 15) >> 4);
    }
}

/* Bounds, slots and lod4 from the 36x36 per-column values.
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
    for (int k = 0; k < GME_LOD_N * GME_LOD_N; ++k) t->lod4[k] = (int16_t)gme_tile_lod4_at(h36, dev36, k % GME_LOD_N, k / GME_LOD_N);
    gme_tile_lod_bounds(t);
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

/* ---- M3: far-field clipmap samples (spec 3.8) ------------------------------- */
/* The renderer's far field: GME_FAR_LEVELS toroidal clipmap levels of
 * GME_FAR_N^2 samples at 16 m (L1, +-8.2 km) and 64 m (L2, +-32.8 km). It is
 * computed from the generator alone (no world residency). Sample (X, Z)
 * (absolute metres, multiples of the spacing) packs:
 *   bits  0-15  int16 height, q4: T (gme_surface_q8's value) box-averaged over
 *               4 x 4 sub-samples placed symmetrically about the point (column
 *               centres at +-1.5 / +-6.5 m for L1, +-6.5 / +-25.5 m for L2)
 *   bits 16-19  sub-samples whose ground voxel is the beach material, 0..15
 *   bits 20-23  palette index of the biomap cell at the sample point (gme_slot)
 *   bits 24-25  tissue: 0 none, 1 grass (cover / shrub / tree), 2 mat (mat / kelp)
 *   bits 26-31  tissue coverage 0..63: the share of the sub-samples where
 *               gme_voxel puts that tissue on the ground (cover / 256 of the
 *               eligible columns), plus the crown area of the slot's shrub / tree
 * Integer only, so CPU and GPU produce the same bits (self-test D6). */
#define GME_FAR_N      1024
#define GME_FAR_LEVELS 2
#define GME_FAR_SHIFT0 4                   /* L1 spacing 16 m; level l: 16 << 2l */

GME_HD int32_t gme_far_spacing(int level) { return 1 << (GME_FAR_SHIFT0 + 2 * level); }

/* Sub-sample (a, b) (0..3 each) of the far sample at (X, Z): its T (q8) and
 * its ground class bits (1 beach material, 2 grass-eligible, 4 mat-eligible).
 * L1 columns X - 7, X - 2, X + 1, X + 6 (centres -6.5 .. 6.5); L2 -26, -7, 6, 25. */
GME_HD void gme_far_sub(int32_t X, int32_t Z, int level, int a, int b, uint32_t seed, const GmePlanet* p, int32_t* tq8, uint32_t* cls) {
    const int32_t oa = level ? (a == 0 ? -26 : (a == 1 ? -7 : (a == 2 ? 6 : 25))) : (a == 0 ? -7 : (a == 1 ? -2 : (a == 2 ? 1 : 6)));
    const int32_t ob = level ? (b == 0 ? -26 : (b == 1 ? -7 : (b == 2 ? 6 : 25))) : (b == 0 ? -7 : (b == 1 ? -2 : (b == 2 ? 1 : 6)));
    int32_t h, t;
    gme_surface_pair(X + oa, Z + ob, seed, p, &h, &t);
    *tq8 = 256 * (h + 1) + 2 * gme_tile_dev_q7(t, h, p);
    *cls = (uint32_t)(h <= p->sea_level + 2 && h >= p->sea_level - 3) | ((uint32_t)(h > p->sea_level && gme_ground_mat(h, p) == p->surface_mat) << 1) |
           ((uint32_t)(h <= p->sea_level + 2) << 2);
}

/* The packed sample from its 16 sub-samples' sum of T (q8) and class counts
 * (the order of summation does not matter: the GPU sums them across lanes) */
GME_HD uint32_t gme_far_pack(int32_t X, int32_t Z, int32_t sum, int32_t nbeach, int32_t ngrass, int32_t nmat, uint32_t seed,
                             const GmePlanet* p, const GmeBioView* bv) {
    GmeSlot s;
    gme_slot(X >> GME_FLORA_SLOT_SHIFT, Z >> GME_FLORA_SLOT_SHIFT, seed, p, bv, &s);
    const uint32_t kind = BIOCELL_KIND(s.cell), cover = BIOCELL_COVER(s.cell);
    uint32_t tissue = 0u, cov = 0u;
    if (kind >= BIO_KIND_COVER && kind <= BIO_KIND_TREE) {
        tissue = 1u;
        cov = ((uint32_t)ngrass * cover * 63u + 2048u) >> 12;      /* 16 sub-samples x cover / 256 */
        if (s.trad > 0) cov += ((uint32_t)(s.trad * s.trad + s.trad) * 201u * 63u) >> 12;   /* crown pi a^2 / 64 m^2 */
    } else if (kind == BIO_KIND_MAT || kind == BIO_KIND_KELP) {
        tissue = 2u;
        cov = ((uint32_t)nmat * cover * 63u + 2048u) >> 12;
    }
    cov = cov > 63u ? 63u : cov;
    const int32_t hq4 = (sum + 128) >> 8;                           /* 16 values of q8 -> mean in q4 */
    return ((uint32_t)hq4 & 0xFFFFu) | ((uint32_t)(nbeach > 15 ? 15 : nbeach) << 16) | (BIOCELL_PAL(s.cell) << 20) |
           (tissue << 24) | (cov << 26);
}

GME_HD uint32_t gme_far_sample(int32_t X, int32_t Z, int level, uint32_t seed, const GmePlanet* p, const GmeBioView* bv) {
    int32_t sum = 0, nbeach = 0, ngrass = 0, nmat = 0;
    for (int k = 0; k < 16; ++k) {
        int32_t tq;
        uint32_t cls;
        gme_far_sub(X, Z, level, k & 3, k >> 2, seed, p, &tq, &cls);
        sum += tq;
        nbeach += (int32_t)(cls & 1u);
        ngrass += (int32_t)((cls >> 1) & 1u);
        nmat += (int32_t)((cls >> 2) & 1u);
    }
    return gme_far_pack(X, Z, sum, nbeach, ngrass, nmat, seed, p, bv);
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

/* ---- M5: drawn shapes of life (render only: the voxel truth and D2's twins above are unchanged) ---- */
/* Clumped crowns (spec 3.5): the crown ellipsoid displaced inward by a 3D value
 * noise n (trace.cuh shapes it into billows, delta = amp (1 - |2n - 1|)^2 in
 * [0, amp]: rounded clumps, narrow creases), amp = 0.35 trad m, on a lattice
 * of 0.42 a m (a = the horizontal radius), so a crown shows a few clumps
 * across; a noise seed per tree. Inward only: the clumped crown lies inside
 * the voxel crown. */
GME_HD float gme_clump_amp(int32_t trad) { return 0.35f * (float)trad; }
GME_HD float gme_clump_cell(int32_t trad) { return 0.42f * sqrtf((float)(trad * trad + trad)); }
GME_HD uint32_t gme_clump_seed(int32_t tx, int32_t tz, uint32_t seed) { return gme_hash3(tx, 0x7EE, tz, seed ^ 0xC1u); }
GME_HD float gme_clump_h01(int32_t x, int32_t y, int32_t z, uint32_t s) { return (float)(gme_hash3(x, y, z, s) & 0xFFFFu) * (1.f / 65535.f); }
/* value noise in [0, 1] at lattice coordinates (x, y, z) (smoothstep trilinear) and its gradient */
GME_HD float gme_clump_noise(float x, float y, float z, uint32_t s, float g[3]) {
    const float fx = floorf(x), fy = floorf(y), fz = floorf(z);
    const int32_t ix = (int32_t)fx, iy = (int32_t)fy, iz = (int32_t)fz;
    const float u = x - fx, v = y - fy, w = z - fz;
    const float su = u * u * (3.f - 2.f * u), sv = v * v * (3.f - 2.f * v), sw = w * w * (3.f - 2.f * w);
    const float du = 6.f * u * (1.f - u), dv = 6.f * v * (1.f - v), dw = 6.f * w * (1.f - w);
    const float c000 = gme_clump_h01(ix, iy, iz, s), c100 = gme_clump_h01(ix + 1, iy, iz, s);
    const float c010 = gme_clump_h01(ix, iy + 1, iz, s), c110 = gme_clump_h01(ix + 1, iy + 1, iz, s);
    const float c001 = gme_clump_h01(ix, iy, iz + 1, s), c101 = gme_clump_h01(ix + 1, iy, iz + 1, s);
    const float c011 = gme_clump_h01(ix, iy + 1, iz + 1, s), c111 = gme_clump_h01(ix + 1, iy + 1, iz + 1, s);
    const float x00 = c000 + (c100 - c000) * su, x10 = c010 + (c110 - c010) * su;
    const float x01 = c001 + (c101 - c001) * su, x11 = c011 + (c111 - c011) * su;
    const float y0 = x00 + (x10 - x00) * sv, y1 = x01 + (x11 - x01) * sv;
    if (g) {
        const float a00 = c100 - c000, a10 = c110 - c010, a01 = c101 - c001, a11 = c111 - c011;
        g[0] = du * ((a00 + (a10 - a00) * sv) * (1.f - sw) + (a01 + (a11 - a01) * sv) * sw);
        g[1] = dv * ((x10 - x00) * (1.f - sw) + (x11 - x01) * sw);
        g[2] = dw * (y1 - y0);
    }
    return y0 + (y1 - y0) * sw;
}

/* Kelp blades (spec 8, M5): the strand of a column is drawn as a thin stipe
 * (capsule of radius GME_KELP_STIPE_R on the column axis, its tip swaying by
 * at most GME_KELP_SWAY) carrying flat elliptical blades every
 * GME_KELP_BLADE_DY m, turning by the golden angle, tilted 30-42 deg upward.
 * Everything stays within 0.49 m of the column centre (D2 checks it), so a
 * strand never leaves its 1 m column and the walk's first hit is the nearest. */
#define GME_KELP_STIPE_R  0.06f
#define GME_KELP_SWAY     0.136f      /* |(0.11 sin, 0.08 cos)| */
#define GME_KELP_BLADE_L  0.40f       /* blade length (from the stipe outward) */
#define GME_KELP_BLADE_W  0.16f       /* blade width */
#define GME_KELP_BLADE_DY 0.6f        /* blade spacing along the strand */
#define GME_KELP_BLADES   64          /* at most */
typedef struct GmeKelpBlade {
    float cx, cy, cz;                 /* centre */
    float ux, uy, uz;                 /* unit long axis (outward and up) */
    float wx, wz;                     /* unit short axis (horizontal) */
} GmeKelpBlade;
GME_HD int gme_kelp_nblades(float y0, float y1) {
    int n = (int)((y1 - y0) / GME_KELP_BLADE_DY);
    return n < 1 ? 1 : (n > GME_KELP_BLADES ? GME_KELP_BLADES : n);
}
/* blade i of n of the strand from (ax, y0, az) to (ax + swx, y1, az + swz); hs = the column's hash */
GME_HD void gme_kelp_blade(int i, int n, float ax, float y0, float az, float swx, float y1, float swz, uint32_t hs, GmeKelpBlade* b) {
    const float f = ((float)i + 0.5f) / (float)n;
    const float phi = (float)(hs & 1023u) * (6.2831853f / 1024.f) + 2.3999632f * (float)i;
    const float th = 0.5236f + 0.2094f * (float)((hs >> (8 + i % 21)) & 7u) * (1.f / 7.f);   /* 30 .. 42 deg */
    const float cp = cosf(phi), sp = sinf(phi), ct = cosf(th), st = sinf(th);
    b->ux = ct * cp; b->uy = st; b->uz = ct * sp;
    b->wx = -sp; b->wz = cp;
    b->cx = ax + swx * f + b->ux * (0.5f * GME_KELP_BLADE_L);
    b->cy = y0 + (y1 - y0) * f + b->uy * (0.5f * GME_KELP_BLADE_L);
    b->cz = az + swz * f + b->uz * (0.5f * GME_KELP_BLADE_L);
}

#endif

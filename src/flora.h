/* flora.h - vegetation voxels from a frozen biomap (docs/biosphere-spec.md 7.3).
 *
 * GME_HD and integer-only, shared by the CPU generator (world.c) and the CUDA
 * generation kernel (gpu.cu), so vegetated worlds stay bit-identical on the
 * CPU and every GPU. Usage, per column and then per voxel:
 *
 *     h = gme_height(x, z, seed, p);
 *     gme_column(x, z, h, seed, p, view, &col);        // once per column
 *     m = gme_voxel(x, y, z, &col, seed, p);            // per voxel
 *
 * Invariant: with view->cells == NULL and view->fallback_on == 0, gme_voxel
 * returns exactly gme_material (no vegetation anywhere).
 *
 * One biomap cell per 8 m slot, looked up at the slot centre, so a slot never
 * straddles two cells (regions) or two environment classes (fallback):
 *   kind 1 mat    ground voxel (y == h) on seabed or beach -> MAT_MAT
 *   kind 2 kelp   strands h < y < min(h + height, sea_level) in water -> MAT_KELP,
 *                 plus the kind-1 mat on the bed
 *   kind 3 cover  ground voxel on land soil -> MAT_GRASS, tufts at h + 1 -> MAT_LEAF
 *   kind 4/5      shrub / tree: at most one per slot with probability cover/256,
 *                 trunk -> MAT_WOOD, ellipsoid crown -> MAT_LEAF (air only, trunk on
 *                 land), grass on the soil under them like kind 3
 * All flora of a column lies in [h, max(h, sea_level) + GME_FLORA_REACH], which
 * lets generators skip the biomap work for chunks outside that span
 * (gme_column_span) without changing a single voxel. */
#ifndef GME_FLORA_H
#define GME_FLORA_H

#include "worldgen.h"
#include "biomap.h"

#define GME_FLORA_SLOT_SHIFT 3    /* 8 m slots: biomap lookup + at most one tree each */
#define GME_FLORA_TREE_MAX   32   /* tallest tree, metres above its base */
#define GME_FLORA_REACH      72   /* flora never extends above max(h, sea_level) + this */

typedef struct GmeColumn {
    int32_t  h;          /* surface height of this column (gme_height) */
    uint32_t cell;       /* packed biomap cell (biomap.h) of this column's slot; 0 = bare */
    int32_t  tx, tz;     /* trunk column of the slot's shrub / tree */
    int32_t  tbase;      /* surface height at the trunk */
    int32_t  theight;    /* crown top above tbase, metres */
    int32_t  trad;       /* crown radius (horizontal), 1..3; 0 = no shrub / tree in this slot */
    uint32_t colhash;    /* per-column random bits */
    int32_t  tcy, tvr;   /* crown centre y and vertical radius */
} GmeColumn;

GME_HD int gme_is_standing_flora(uint32_t m) {
    return m == MAT_LEAF || m == MAT_WOOD || m == MAT_KELP;
}

/* every material that only life produces */
GME_HD int gme_is_flora(uint32_t m) {
    return m == MAT_GRASS || m == MAT_MAT || gme_is_standing_flora(m);
}

/* Material of the ground voxel (depth 0) of a column of height h; mirrors the
 * depth-0 branch of gme_material (caves only start at depth 7). */
GME_HD int32_t gme_ground_mat(int32_t h, const GmePlanet* p) {
    if (h <= p->sea_level + 2 && h >= p->sea_level - 3) return p->beach_mat;
    return p->soil_depth > 0 ? p->surface_mat : MAT_REGOLITH;
}

/* Environment class 0..63 of a column, the key of the biomap fallback table
 * (the exporter evaluates it at cell centres with the cell's height):
 *   bits 4-5  moisture: 1 km noise in quartiles, wetter up to 16 m above sea level
 *   bits 2-3  height band: (h - sea_level) / 40, clamped 0..3
 *   bit  1    soil: the ground voxel is MAT_DIRT
 *   bit  0    granite province (the same test as gme_material) */
GME_HD uint32_t gme_env_bin(int32_t x, int32_t z, int32_t h, uint32_t seed, const GmePlanet* p) {
    const int32_t rel = h - p->sea_level;
    int32_t m = gme_noise2(x, z, 10, seed ^ 0xB10C0001u);
    if (rel < 16) m += 16384;
    int32_t mq = (m + 16384) >> 13;                    /* class edges at -8192, 0, 8192 */
    mq = mq < 0 ? 0 : (mq > 3 ? 3 : mq);
    int32_t band = rel / 40;
    band = band < 0 ? 0 : (band > 3 ? 3 : band);
    const int32_t prov = gme_noise2(x, z, 9, seed ^ 0x6A417E00u);
    const uint32_t granite = (((prov + 32768) >> 8) < p->granite_q8) ? 1u : 0u;
    const uint32_t soil = gme_ground_mat(h, p) == MAT_DIRT ? 1u : 0u;
    return ((uint32_t)mq << 4) | ((uint32_t)band << 2) | (soil << 1) | granite;
}

/* Biomap cell of the region containing (x, z); *inside = 0 if none does. */
GME_HD uint32_t gme_bio_region_cell(const GmeBioView* b, int32_t x, int32_t z, int* inside) {
    const int n = b->nreg < BIOMAP_MAX_REGIONS ? b->nreg : BIOMAP_MAX_REGIONS;
    for (int r = 0; r < n; ++r) {
        /* negative offsets wrap to huge values and fail the bounds test */
        const uint32_t i = (uint32_t)(x - b->reg[r].x0) >> b->shift;
        const uint32_t j = (uint32_t)(z - b->reg[r].z0) >> b->shift;
        if (i < (uint32_t)b->reg[r].w && j < (uint32_t)b->reg[r].h) {
            *inside = 1;
            return b->cells[b->reg[r].offset + j * (uint32_t)b->reg[r].w + i];
        }
    }
    *inside = 0;
    return 0u;
}

/* The part of gme_column shared by all 64 columns of an 8 m slot: its biomap
 * cell and its shrub / tree. Generators may compute it once per slot. */
typedef struct GmeSlot {
    uint32_t cell;                       /* 0 = bare (also for unknown kinds) */
    int32_t  tx, tz, tbase, theight, trad, tcy, tvr;   /* as in GmeColumn; trad 0 = no tree */
} GmeSlot;

GME_HD void gme_slot(int32_t sx, int32_t sz, uint32_t seed, const GmePlanet* p, const GmeBioView* b, GmeSlot* s) {
    s->cell = 0u;
    s->tx = 0; s->tz = 0; s->tbase = 0; s->theight = 0; s->trad = 0; s->tcy = 0; s->tvr = 0;
    const int32_t px = sx * 8 + 4, pz = sz * 8 + 4;   /* slot centre */
    uint32_t cell = 0u;
    int inside = 0;
    if (b->cells) cell = gme_bio_region_cell(b, px, pz, &inside);
    if (!inside) {
        if (!b->fallback_on) return;
        /* outside the simulated regions: statistical vegetation by environment
         * class, sampled at the slot centre so the whole slot agrees */
        cell = b->fallback[gme_env_bin(px, pz, gme_height(px, pz, seed, p), seed, p) & 63u];
    }
    const uint32_t kind = BIOCELL_KIND(cell);
    if (kind == BIO_KIND_NONE || kind > BIO_KIND_TREE) return;   /* unknown kinds: bare */
    s->cell = cell;
    if (kind < BIO_KIND_SHRUB) return;
    const uint32_t sh = gme_hash3(sx, 0x5107, sz, seed ^ 0xF10Bu);
    if ((sh & 255u) >= BIOCELL_COVER(cell)) return;              /* slot left empty */
    /* height 75..100% of the cell's: shrubs 2..4 m, trees 4..TREE_MAX m */
    int32_t ht = (int32_t)((BIOCELL_HEIGHT(cell) * (192u + ((sh >> 24) & 63u))) >> 8);
    const int32_t lo = kind == BIO_KIND_TREE ? 4 : 2, hi = kind == BIO_KIND_TREE ? GME_FLORA_TREE_MAX : 4;
    ht = ht < lo ? lo : (ht > hi ? hi : ht);
    int32_t trad = ht / 4;
    trad = trad < 1 ? 1 : (trad > 3 ? 3 : trad);
    int32_t vr = ht / 3;                                         /* taller trees: taller crowns */
    vr = vr < trad ? trad : (vr > 8 ? 8 : vr);
    /* trunk jittered inside the slot with a margin of trad, so the whole
     * crown stays inside this slot */
    const uint32_t span = (uint32_t)(8 - 2 * trad);
    const int32_t tx = sx * 8 + trad + (int32_t)(((sh >> 8) & 255u) % span);
    const int32_t tz = sz * 8 + trad + (int32_t)(((sh >> 16) & 255u) % span);
    const int32_t tb = gme_height(tx, tz, seed, p);              /* the one extra height call, vegetated slots only */
    if (tb <= p->sea_level) return;                              /* land plants only */
    s->tx = tx; s->tz = tz; s->tbase = tb; s->theight = ht; s->trad = trad;
    s->tvr = vr; s->tcy = tb + ht - vr;
}

/* Column without vegetation: gme_voxel then equals gme_material. */
GME_HD void gme_column_bare(int32_t h, GmeColumn* c) {
    c->h = h; c->cell = 0u;
    c->tx = 0; c->tz = 0; c->tbase = 0; c->theight = 0; c->trad = 0;
    c->colhash = 0u;
    c->tcy = 0; c->tvr = 0;
}

/* Column (x, z) of height h in the slot s = gme_slot(x >> 3, z >> 3). */
GME_HD void gme_column_in_slot(int32_t x, int32_t z, int32_t h, uint32_t seed, const GmeSlot* s, GmeColumn* c) {
    c->h = h; c->cell = s->cell;
    c->tx = s->tx; c->tz = s->tz; c->tbase = s->tbase; c->theight = s->theight; c->trad = s->trad;
    c->tcy = s->tcy; c->tvr = s->tvr;
    c->colhash = s->cell ? gme_hash3(x, 0, z, seed ^ 0xF10Au) : 0u;
}

GME_HD void gme_column(int32_t x, int32_t z, int32_t h, uint32_t seed, const GmePlanet* p,
                       const GmeBioView* b, GmeColumn* c) {
    GmeSlot s;
    gme_slot(x >> GME_FLORA_SLOT_SHIFT, z >> GME_FLORA_SLOT_SHIFT, seed, p, b, &s);
    gme_column_in_slot(x, z, h, seed, &s, c);
}

/* Can the column's flora touch any voxel y0..y1? (No: gme_column_bare gives
 * the same voxels there, without the biomap work.) */
GME_HD int gme_flora_in_span(int32_t h, int32_t y0, int32_t y1, const GmePlanet* p, const GmeBioView* b) {
    const int32_t top = (h > p->sea_level ? h : p->sea_level) + GME_FLORA_REACH;
    return y1 >= h && y0 <= top && (b->cells || b->fallback_on);
}

/* gme_column for the voxels y0..y1 of the column only. Exact for those voxels. */
GME_HD void gme_column_span(int32_t x, int32_t z, int32_t h, int32_t y0, int32_t y1, uint32_t seed,
                            const GmePlanet* p, const GmeBioView* b, GmeColumn* c) {
    if (gme_flora_in_span(h, y0, y1, p, b)) gme_column(x, z, h, seed, p, b, c);
    else gme_column_bare(h, c);
}

GME_HD uint8_t gme_voxel(int32_t x, int32_t y, int32_t z, const GmeColumn* c, uint32_t seed, const GmePlanet* p) {
    const uint8_t base = gme_material(x, y, z, c->h, seed, p);
    const uint32_t cell = c->cell;
    const int32_t h = c->h;
    if (!cell || y < h) return base;   /* no life in this column, or underground: geology only */
    const uint32_t kind = cell & 15u, cover = (cell >> 8) & 255u;
    const int32_t sea = p->sea_level;
    if (kind <= BIO_KIND_KELP) {       /* mat, kelp: wet ground and water */
        if (y == h) return (h <= sea + 2 && (c->colhash & 255u) < cover) ? (uint8_t)MAT_MAT : base;
        if (kind == BIO_KIND_KELP && base == MAT_WATER && y < sea && ((c->colhash >> 16) & 255u) < (cover >> 1)) {
            /* strand length 50..100% of the cell height, and of the water
             * above the bed, so the tops stay ragged below the surface */
            int32_t len = (int32_t)((cell >> 16) & 255u);
            if (len > sea - h - 1) len = sea - h - 1;        /* h < y < sea here: sea - h - 1 >= 1 */
            len = (int32_t)(((uint32_t)len * (128u + ((c->colhash >> 24) & 127u))) >> 8);
            if (y <= h + len) return (uint8_t)MAT_KELP;
        }
        return base;
    }
    /* ground cover, shrubs, trees: land */
    if (y == h) return (h > sea && base == p->surface_mat && (c->colhash & 255u) < cover) ? (uint8_t)MAT_GRASS : base;
    if (base != MAT_AIR || y > h + GME_FLORA_REACH) return base;
    if (kind == BIO_KIND_COVER) {      /* tufts on a quarter of the grass */
        const int grass = h > sea && gme_ground_mat(h, p) == p->surface_mat && (c->colhash & 255u) < cover;
        return (y == h + 1 && grass && ((c->colhash >> 8) & 255u) < (cover >> 2)) ? (uint8_t)MAT_LEAF : base;
    }
    if (!c->trad) return base;         /* no shrub / tree in this slot */
    const int32_t dx = x - c->tx, dz = z - c->tz, dy = y - c->tcy;
    if (dx == 0 && dz == 0 && y > c->tbase && y <= c->tcy) return (uint8_t)MAT_WOOD;
    /* crown: ellipsoid (dx^2 + dz^2) / (r^2 + r) + dy^2 / (v^2 + v) <= 1 */
    const int32_t rr = c->trad * c->trad + c->trad, vv = c->tvr * c->tvr + c->tvr;
    if ((dx * dx + dz * dz) * vv + dy * dy * rr <= rr * vv) return (uint8_t)MAT_LEAF;
    return base;
}

#endif

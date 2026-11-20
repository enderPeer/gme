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
 * (gme_column_span) without changing a single voxel.
 *
 * Version 2 (docs/render-upgrade-spec.md 5.3): a biomap with morphology
 * (GmeBioView.morph != NULL) gives every cell a BIOMORPH word. A shrub / tree
 * of a cell whose word is valid takes its crown class and aspect bin
 * (aspect = BIOMORPH_ASPECT_Q8 of the bin, 0.5 .. 3; bin 2 is 1 : 1):
 *   r   horizontal radius 1..3, a^2 = r^2 + r as in version 1. It starts at the
 *       version 1 radius clamp(ht / 4, 1, 3) (a column: one less), which is
 *       also the trunk's jitter margin, so every crown stays inside its slot.
 *   ch  crown rows. The crown fills the rows yb .. yt, yt = tbase + ht, yb =
 *       yt - ch + 1.
 *       ellipsoid: the version 1 crown stretched by the aspect, ch = round(
 *       aspect x (2 vr + 1)) with version 1's vr = clamp(ht / 3, r, 8), so the
 *       aspect 1 : 1 (bin 2) is the version 1 plant, voxel for voxel.
 *       cone, column, dome: height : width, ch = round(aspect x (2 r + 1)).
 *       All four: at least 2 rows, at most the plant's height; a crown that
 *       would be taller narrows first (r - 1 while r > 1).
 *       umbrella: a cap, ch = half the rows of aspect x (2 r + 1), 1 .. r + 1
 *       and at most ht - 1; r stays the version 1 radius.
 *   With X = dx^2 + dz^2 and u = y - yb + 1/2 (a voxel centre's height above
 *   the bottom face of row yb), a voxel of those rows is MAT_LEAF iff
 *     ellipsoid  X / a^2 + (u - ch / 2)^2 / ((ch^2 - 1) / 4) <= 1  (version 1's
 *                inequality when ch = 2 vr + 1)
 *     cone       X / a^2 <= ((ch - u) / ch)^2   widest at the base, a point on top
 *     umbrella   X / a^2 <= u / ch              a bowl: flat and widest on top, its
 *                tip on a bare trunk
 *     column     X <= a^2                       a cylinder
 *     dome       X / a^2 + (u / ch)^2 <= 1      upper half of an ellipsoid
 *   all evaluated in integers (gme_crown_v2_inside). The trunk is MAT_WOOD from
 *   tbase + 1 up to the row below the umbrella's cap, else up to the lower
 *   middle row of the crown, yb + (ch - 1) / 2 (version 1's trunk for the
 *   ellipsoid at 1 : 1).
 * A cell without a valid word, and every cell of a version 1 biomap, keeps the
 * version 1 shape, voxel for voxel. Porosity, leaf angle, pigment amount and Q
 * band width never change a voxel: they are carried for the renderer only. */
#ifndef GME_FLORA_H
#define GME_FLORA_H

#include "worldgen.h"
#include "biomap.h"

#define GME_FLORA_SLOT_SHIFT 3    /* 8 m slots: biomap lookup + at most one tree each */
#define GME_FLORA_TREE_MAX   32   /* tallest tree, metres above its base */
#define GME_FLORA_REACH      72   /* flora never extends above max(h, sea_level) + this */

/* GmeSlot.theight / GmeColumn.theight, the shape word of a slot:
 *   bits 0-7    ht: the plant's top above tbase, metres (0 = no shrub / tree)
 *   -- a version 1 slot stops here (bits 8-31 are 0) --
 *   bit  11     version 2: the slot's cell carries evolved morphology
 *   bits 8-10   crown class (BIO_CROWN_*)
 *   bits 12-19  mean pigment amount, 20-21 porosity bin, 22-23 leaf angle bin,
 *   bits 24-27  Q band width bin (renderer only; set for every kind of cell)
 * In a version 2 slot tvr is the number of crown rows ch (not a radius) and tcy
 * the trunk's top row (not the crown's centre). */
#define GME_SHAPE_V2          0x800u
#define GME_SHAPE_HT(s)       ((int32_t)((uint32_t)(s) & 255u))
#define GME_SHAPE_IS_V2(s)    (((uint32_t)(s) >> 11) & 1u)
#define GME_SHAPE_CLASS(s)    (((uint32_t)(s) >> 8) & 7u)
#define GME_SHAPE_PIG(s)      (((uint32_t)(s) >> 12) & 255u)
#define GME_SHAPE_POROSITY(s) (((uint32_t)(s) >> 20) & 3u)
#define GME_SHAPE_LEAFANG(s)  (((uint32_t)(s) >> 22) & 3u)
#define GME_SHAPE_QWIDTH(s)   (((uint32_t)(s) >> 24) & 15u)

typedef struct GmeColumn {
    int32_t  h;          /* surface height of this column (gme_height) */
    uint32_t cell;       /* packed biomap cell (biomap.h) of this column's slot; 0 = bare */
    int32_t  tx, tz;     /* trunk column of the slot's shrub / tree */
    int32_t  tbase;      /* surface height at the trunk */
    int32_t  theight;    /* shape word: crown top above tbase (metres) in bits 0-7, version 2 above */
    int32_t  trad;       /* crown radius (horizontal), 1..3; 0 = no shrub / tree in this slot */
    uint32_t colhash;    /* per-column random bits */
    int32_t  tcy, tvr;   /* version 1: crown centre y and vertical radius; version 2: trunk top row and crown rows */
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

/* The same with the cell's morphology word (version 2; 0 for a version 1 view). */
GME_HD uint32_t gme_bio_region_cell2(const GmeBioView* b, int32_t x, int32_t z, int* inside, uint32_t* morph) {
    const int n = b->nreg < BIOMAP_MAX_REGIONS ? b->nreg : BIOMAP_MAX_REGIONS;
    for (int r = 0; r < n; ++r) {
        const uint32_t i = (uint32_t)(x - b->reg[r].x0) >> b->shift;
        const uint32_t j = (uint32_t)(z - b->reg[r].z0) >> b->shift;
        if (i < (uint32_t)b->reg[r].w && j < (uint32_t)b->reg[r].h) {
            const uint32_t k = b->reg[r].offset + j * (uint32_t)b->reg[r].w + i;
            *inside = 1;
            *morph = b->morph ? b->morph[k] : 0u;
            return b->cells[k];
        }
    }
    *inside = 0;
    *morph = 0u;
    return 0u;
}

/* Shape word (bits 8-27, no height) of a morphology word; 0 = version 1. */
GME_HD uint32_t gme_shape_word(uint32_t m) {
    if (!(m & BIOMORPH_VALID)) return 0u;
    uint32_t cls = BIOMORPH_CROWN(m);
    if (cls >= (uint32_t)BIO_CROWN_N) cls = BIO_CROWN_ELLIPSOID;  /* unknown classes: the ellipsoid */
    return GME_SHAPE_V2 | (cls << 8) | (BIOMORPH_PIG(m) << 12) | (BIOMORPH_POROSITY(m) << 20) |
           (BIOMORPH_LEAFANG(m) << 22) | (BIOMORPH_QWIDTH(m) << 24);
}

/* Version 2 crown of a plant ht (2..32) metres tall whose version 1 radius is
 * rn (1..3): the horizontal radius r (1..rn) and the crown's rows ch (1..ht;
 * the ellipsoid of a 2 m shrub keeps version 1's 3 rows, the lowest in the ground). */
GME_HD void gme_crown_v2(uint32_t cls, uint32_t aspect_bin, int32_t ht, int32_t rn, int32_t* r_out, int32_t* ch_out) {
    const int32_t aq = (int32_t)BIOMORPH_ASPECT_Q8(aspect_bin);  /* aspect, q8: 128 (0.5) .. 768 (3); bin 2 = 256 */
    int32_t r = rn;
    if (cls == BIO_CROWN_ELLIPSOID) {
        /* the version 1 crown (2 vr + 1 rows) stretched by the aspect: at 1 : 1 it is the version 1 crown */
        int32_t vr = ht / 3;
        vr = vr < rn ? rn : (vr > 8 ? 8 : vr);
        const int32_t rows1 = 2 * vr + 1, cap = ht > rows1 ? ht : rows1;
        const int32_t want = (aq * rows1 + 128) >> 8;
        while (r > 1 && want * (2 * r + 1) > cap * (2 * rn + 1)) --r;   /* taller than the plant: narrower instead */
        const int32_t ch = want > cap ? cap : want;
        *r_out = r;
        *ch_out = ch < 2 ? 2 : ch;
        return;
    }
    if (cls == BIO_CROWN_UMBRELLA) {
        /* a cap: half the rows the aspect asks for, never deeper than r + 1, on at least one row of bare trunk */
        int32_t t = (((aq * (2 * r + 1) + 128) >> 8) + 1) >> 1;
        const int32_t tmax = r + 1 < ht - 1 ? r + 1 : ht - 1;
        t = t > tmax ? tmax : t;
        *r_out = r;
        *ch_out = t < 1 ? 1 : t;
        return;
    }
    if (cls == BIO_CROWN_COLUMN && r > 1) --r;                   /* narrow */
    while (r > 1 && ((aq * (2 * r + 1) + 128) >> 8) > ht) --r;  /* taller than the plant: narrower first */
    int32_t ch = (aq * (2 * r + 1) + 128) >> 8;
    ch = ch > ht ? ht : ch;
    *r_out = r;
    *ch_out = ch < 2 ? 2 : ch;
}

/* Is the voxel centre at row y, X = dx^2 + dz^2 from the trunk axis, inside the
 * version 2 crown of radius r whose ch rows end at row yt? Integer only. */
GME_HD int gme_crown_v2_inside(uint32_t cls, int32_t X, int32_t y, int32_t yt, int32_t ch, int32_t r) {
    const int32_t yb = yt - ch + 1;
    if (y < yb || y > yt) return 0;
    const int32_t rr = r * r + r, k = 2 * (y - yb) + 1;          /* k = 2 u: 1, 3, .., 2 ch - 1 */
    switch (cls) {
    case BIO_CROWN_CONE:     { const int32_t g = 2 * ch - k; return X * 4 * ch * ch <= rr * g * g; }
    case BIO_CROWN_UMBRELLA: return X * 2 * ch <= rr * k;
    case BIO_CROWN_COLUMN:   return X <= rr;
    case BIO_CROWN_DOME:     return X * 4 * ch * ch + k * k * rr <= rr * 4 * ch * ch;
    default:                 { const int32_t e = k - ch, vv = ch * ch - 1; return X * vv + e * e * rr <= rr * vv; }
    }
}

/* The part of gme_column shared by all 64 columns of an 8 m slot: its biomap
 * cell and its shrub / tree. Generators may compute it once per slot. */
typedef struct GmeSlot {
    uint32_t cell;                       /* 0 = bare (also for unknown kinds) */
    int32_t  tx, tz, tbase, theight, trad, tcy, tvr;   /* as in GmeColumn; trad 0 = no tree */
} GmeSlot;

/* The shrub / tree of a slot from its shape word sw (gme_shape_word; 0 = version
 * 1) and aspect bin, its height ht, version 1 radius rn and trunk (tx, tz)
 * standing on tb. */
GME_HD void gme_slot_tree(GmeSlot* s, uint32_t sw, uint32_t aspect_bin, int32_t ht, int32_t rn, int32_t tx, int32_t tz, int32_t tb) {
    s->tx = tx; s->tz = tz; s->tbase = tb;
    if (!sw) {                                                   /* version 1: ellipsoid, taller trees: taller crowns */
        int32_t vr = ht / 3;
        vr = vr < rn ? rn : (vr > 8 ? 8 : vr);
        s->theight = ht; s->trad = rn;
        s->tvr = vr; s->tcy = tb + ht - vr;
        return;
    }
    const uint32_t cls = GME_SHAPE_CLASS(sw);
    int32_t r, ch;
    gme_crown_v2(cls, aspect_bin, ht, rn, &r, &ch);
    s->theight = (int32_t)(sw | (uint32_t)ht);
    s->trad = r;
    s->tvr = ch;
    /* trunk top: the row under an umbrella's cap, else the crown's lower middle row */
    s->tcy = cls == BIO_CROWN_UMBRELLA ? tb + ht - ch : tb + ht - ch + 1 + (ch - 1) / 2;
}

GME_HD void gme_slot(int32_t sx, int32_t sz, uint32_t seed, const GmePlanet* p, const GmeBioView* b, GmeSlot* s) {
    s->cell = 0u;
    s->tx = 0; s->tz = 0; s->tbase = 0; s->theight = 0; s->trad = 0; s->tcy = 0; s->tvr = 0;
    const int32_t px = sx * 8 + 4, pz = sz * 8 + 4;   /* slot centre */
    uint32_t cell = 0u, mw = 0u;
    int inside = 0;
    if (b->cells) cell = gme_bio_region_cell2(b, px, pz, &inside, &mw);
    if (!inside) {
        if (!b->fallback_on) return;
        /* outside the simulated regions: statistical vegetation by environment
         * class, sampled at the slot centre so the whole slot agrees */
        const uint32_t eb = gme_env_bin(px, pz, gme_height(px, pz, seed, p), seed, p) & 63u;
        cell = b->fallback[eb];
        mw = b->morph ? b->fallback_morph[eb] : 0u;
    }
    const uint32_t kind = BIOCELL_KIND(cell);
    if (kind == BIO_KIND_NONE || kind > BIO_KIND_TREE) return;   /* unknown kinds: bare */
    s->cell = cell;
    /* version 2: the cell's morphology (0 for a version 1 biomap and for cells without a valid word) */
    const uint32_t sw = gme_shape_word(mw);
    s->theight = (int32_t)sw;
    if (kind < BIO_KIND_SHRUB) return;
    const uint32_t sh = gme_hash3(sx, 0x5107, sz, seed ^ 0xF10Bu);
    if ((sh & 255u) >= BIOCELL_COVER(cell)) return;              /* slot left empty */
    /* height 75..100% of the cell's: shrubs 2..4 m, trees 4..TREE_MAX m */
    int32_t ht = (int32_t)((BIOCELL_HEIGHT(cell) * (192u + ((sh >> 24) & 63u))) >> 8);
    const int32_t lo = kind == BIO_KIND_TREE ? 4 : 2, hi = kind == BIO_KIND_TREE ? GME_FLORA_TREE_MAX : 4;
    ht = ht < lo ? lo : (ht > hi ? hi : ht);
    int32_t trad = ht / 4;
    trad = trad < 1 ? 1 : (trad > 3 ? 3 : trad);
    /* trunk jittered inside the slot with a margin of trad (the version 1 radius; a
     * version 2 crown is never wider), so the whole crown stays inside this slot */
    const uint32_t span = (uint32_t)(8 - 2 * trad);
    const int32_t tx = sx * 8 + trad + (int32_t)(((sh >> 8) & 255u) % span);
    const int32_t tz = sz * 8 + trad + (int32_t)(((sh >> 16) & 255u) % span);
    const int32_t tb = gme_height(tx, tz, seed, p);              /* the one extra height call, vegetated slots only */
    if (tb <= p->sea_level) return;                              /* land plants only */
    gme_slot_tree(s, sw, BIOMORPH_ASPECT(mw), ht, trad, tx, tz, tb);
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
    if (GME_SHAPE_IS_V2(c->theight))   /* version 2: the crown class of the cell's morphology */
        return gme_crown_v2_inside(GME_SHAPE_CLASS(c->theight), dx * dx + dz * dz, y, c->tbase + GME_SHAPE_HT(c->theight), c->tvr, c->trad)
                   ? (uint8_t)MAT_LEAF : base;
    /* crown: ellipsoid (dx^2 + dz^2) / (r^2 + r) + dy^2 / (v^2 + v) <= 1 */
    const int32_t rr = c->trad * c->trad + c->trad, vv = c->tvr * c->tvr + c->tvr;
    if ((dx * dx + dz * dz) * vv + dy * dy * rr <= rr * vv) return (uint8_t)MAT_LEAF;
    return base;
}

#endif

/* fauna_scene.h - creatures in the 3D view (docs/biosphere-spec.md 7.5).
 *
 * The motile organisms of a GMEFAUN1 snapshot (fauna.h) are drawn as spheres.
 * fauna_scene_build (host, once per file) places them on the world the
 * renderer draws and bins them into a hash grid of 32 m cells (= chunk
 * columns): every cell owns a contiguous run of sphere records (a sphere that
 * reaches into a neighbouring cell is listed in both). The renderer uploads
 * the records and the grid (renderer_set_fauna) and its kernels walk the grid
 * along each ray. Creatures never touch the chunk
 * pool or the tiles: generation, validation and every hash are unaffected.
 *
 * Placement (the file carries no heights):
 *   walker   on the drawn ground: T = the bilinear patch of gme_surface_q8 at
 *            the 4 column centres around it (exactly what the column tracer
 *            draws near the camera), centre at T + r sqrt(1 + |grad T|^2), so
 *            the sphere touches the slope. A walker whose ground is under the
 *            water plane (T < sea_level + 1) is dropped (on a frozen sea it
 *            stands on the ice).
 *   swimmer  at depth below the water plane y_w = sea_level + 1, kept between
 *            the seabed (T + r) and the surface (y_w - r - 0.05); where the
 *            water is shallower than the body it rests on the bed. A swimmer
 *            over land (T >= y_w) is dropped.
 *   size     radius = half the record's diameter, clamped to FAUNA_R_MIN ..
 *            FAUNA_R_MAX (small animals stay visible).
 *   swarm    FAUNA_SWARM: FAUNA_SWARM_N spheres of radius max(R_MIN, 0.42 r)
 *            on a jittered ring of radius max(1.6 r, 2.6 r_member) around the
 *            record's position, each placed by the rules above.
 *   motion   fauna_motion: an idle sway and hop (walkers) or drift (swimmers)
 *            of at most `amp` metres, amp and speed scaled by the motor gene;
 *            time 0 (--still) holds it.
 *   classic  the voxel view has no smooth surface: there a walker stands on
 *            the top face of its column's ground voxel (yv), a swimmer is
 *            kept above the voxel seabed.
 * Shared by C (fauna_scene.c, main.c) and CUDA (render.cu). */
#ifndef GME_FAUNA_SCENE_H
#define GME_FAUNA_SCENE_H

#include <math.h>
#include <stdint.h>
#include "worldgen.h"

#define FAUNA_FAR        1024.f   /* creatures are drawn up to this path length, metres */
#define FAUNA_SHADOW_FAR 128.f    /* ... and shadow a point up to this far along the sun ray */
#define FAUNA_R_MIN      0.2f
#define FAUNA_R_MAX      6.0f
#define FAUNA_SWARM_N    6
#define FAUNA_MACRO_SHIFT 6       /* occupancy list for the frame's early out: 64 cells = 2048 m */

/* FaunaSphere.attr */
#define FA_PAL(a)     ((a) & 15u)             /* palette index = the pigment's hue bin (0 = the default bin) */
#define FA_SWIMS      0x10u
#define FA_HUNTER     0x20u
#define FA_ARMORED    0x40u
#define FA_SWARM      0x80u                   /* one of a swarm's spheres */
#define FA_MOTOR(a)   (((a) >> 8) & 255u)     /* locomotion gene 0..255 */
#define FA_PHASE(a)   (((a) >> 16) & 511u)    /* phase of the idle motion, 0..511 */
#define FA_FIRST      0x02000000u             /* the first (or only) sphere of its creature */
#define FA_BRIGHT(a)  ((a) >> 26)             /* per-creature brightness, 0..63 */

typedef struct FaunaSphere {       /* 36 B */
    int32_t  cx, cz;               /* 32 m cell (chunk column) of the rest position */
    float    lx, y, lz;            /* rest centre: metres inside the cell (0..32), absolute height */
    float    r;                    /* drawn radius, metres */
    float    amp;                  /* idle motion amplitude, metres (0 = none) */
    uint32_t attr;                 /* FA_* */
    float    yv;                   /* rest height in the classic view: on the voxel ground (top face of h) */
} FaunaSphere;

typedef struct FaunaCell {         /* 24 B; one slot of the hash grid, count 0 = empty */
    int32_t  cx, cz;
    uint32_t first, count;         /* grid records [first, first + count): the spheres that can touch the cell */
    float    ylo, yhi;             /* height range of those spheres, motion included (absolute metres) */
} FaunaCell;

GME_HD uint32_t fauna_cell_hash(int32_t cx, int32_t cz) { return gme_hash3(cx, 0x0FA0, cz, 0xFA17A5EDu); }

/* the slot of cell (cx, cz), or -1 (open addressing, linear probing; the table is at most a quarter full) */
GME_HD int32_t fauna_cell_find(const FaunaCell* cells, uint32_t mask, int32_t cx, int32_t cz) {
    uint32_t h = fauna_cell_hash(cx, cz) & mask;
    for (int k = 0; k < 64; ++k) {
        const FaunaCell* c = cells + h;
        if (!c->count) return -1;
        if (c->cx == cx && c->cz == cz) return (int32_t)h;
        h = (h + 1u) & mask;
    }
    return -1;
}

/* Offset of a sphere's centre from its rest position at time t (seconds): at
 * most amp horizontally; a walker hops 0 .. amp upward (never below its rest
 * height), a swimmer drifts within +-amp / 2 vertically. */
GME_HD void fauna_motion(float amp, uint32_t attr, float t, float* ox, float* oy, float* oz) {
    if (!(amp > 0.f)) { *ox = 0.f; *oy = 0.f; *oz = 0.f; return; }
    const float ph = (float)FA_PHASE(attr) * (6.2831853f / 512.f);
    const float w = 0.5f + 2.5f * (float)FA_MOTOR(attr) * (1.f / 255.f);   /* rad / s */
    if (attr & FA_SWIMS) {
        *ox = 0.7f * amp * sinf(0.6f * w * t + ph);
        *oz = 0.7f * amp * cosf(0.47f * w * t + 1.7f * ph);
        *oy = 0.5f * amp * sinf(0.33f * w * t + 2.3f * ph);
    } else {
        const float h = sinf(w * t + ph);
        *ox = 0.3f * amp * sinf(0.5f * w * t + 1.3f * ph);
        *oz = 0.3f * amp * cosf(0.43f * w * t + 1.7f * ph);
        *oy = amp * h * h;
    }
}

#ifndef __CUDACC__
#include "fauna.h"

typedef struct FaunaScene {
    FaunaSphere* sph;              /* nsph spheres, one record each (the creatures in file order) */
    uint32_t     nsph;
    FaunaSphere* grid;             /* ngrid records grouped by cell: what the renderer uploads */
    uint32_t     ngrid;
    FaunaCell*   cells;            /* ncells slots (a power of two) */
    uint32_t     ncells;
    /* what the build did */
    uint32_t     records;          /* records in the file */
    uint32_t     kept;             /* creatures placed */
    uint32_t     walkers, swimmers, swarms;       /* ... of them */
    uint32_t     drop_swim_land;   /* swimmers over land */
    uint32_t     drop_walk_water;  /* walkers over water */
    uint32_t     drop_range;       /* outside the world's coordinate range */
    uint32_t     used_cells;       /* occupied 32 m cells */
} FaunaScene;

/* Places the creatures of f on the world of (seed, pl). 0 ok (out->nsph may be 0), -1 out of memory. */
int      fauna_scene_build(const Fauna* f, uint32_t seed, const GmePlanet* pl, FaunaScene* out);
void     fauna_scene_free(FaunaScene* s);
/* creatures (not spheres) whose rest position lies within `radius` metres of (x, z) */
uint32_t fauna_scene_within(const FaunaScene* s, double x, double z, double radius);
#endif

#endif

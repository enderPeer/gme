/* render.h - CUDA renderer that draws straight from the GPU chunk pool.
 *
 * Two paths (docs/render-upgrade-spec.md):
 *   RQ_CLASSIC  today's voxel ray-marcher (k_render), byte-for-byte unchanged
 *   RQ_LOW..REF the smooth-surface renderer: a 2.5D column tracer over the
 *               world's column tiles (density.h, trace.cuh), analytic trees,
 *               shrubs and kelp, voxel mode for caves and missing tiles. M1
 *               shades it with the classic lighting model on smooth normals
 *               (k_smooth); the presets differ from M2 on. */
#ifndef GME_RENDER_H
#define GME_RENDER_H

#include <stdint.h>
#include "worldgen.h"
#include "biomap.h"
#include "density.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RENDER_MAX_BODIES 16

/* Light and sky of the home planet (world-space directions). */
typedef struct RenderSky {
    float sun_dir[3], sun_color[3], sun_intensity, sun_size;
    float zenith[3], horizon[3];
    int   nbodies;
    float body_dir[RENDER_MAX_BODIES][3];
    float body_ang[RENDER_MAX_BODIES];
    float body_color[RENDER_MAX_BODIES][3];
} RenderSky;

/* What the renderer needs from the world (filled by world_get_scene). */
typedef struct RenderScene {
    const GmePageEntry* page;  /* host copy of the lookup table */
    uint32_t page_n, layers;
    int32_t  min_cy;
    int      page_dirty;       /* table changed since last frame */
    const uint8_t* pool;       /* device chunk pool */
    float    max_dist;         /* metres; loaded radius */
    /* vegetation tint: grass, leaves, mats and kelp inside a biomap region take
     * the colour of their cell's palette index (0 = keep the material colour) */
    const uint32_t* bio_cells; /* device biomap cells, NULL = no tint */
    int32_t  bio_nreg, bio_shift;
    GmeBioRegion bio_reg[BIOMAP_MAX_REGIONS];
    float    bio_palette[BIOMAP_PALETTE][3];   /* linear RGB */
    /* smooth surface (density.h) */
    const GmeColTile* tiles;   /* device, page_n * page_n records (toroidal like page) */
    const uint32_t*   tile_tag;/* host copy: gme_tile_key of the published tile per cell, 0 = none */
    int       tile_dirty;      /* tile_tag changed since last frame */
    GmePlanet planet;          /* analytic fallbacks */
    uint32_t  seed;
    GmeBioView bio;            /* cells = device pointer (gpu_world_biomap) or NULL */
    const uint32_t* page_dirty_idx; uint32_t page_dirty_n;   /* M4 scatter lists (unused in M1) */
    const uint32_t* tile_dirty_idx; uint32_t tile_dirty_n;
} RenderScene;

enum { RQ_CLASSIC = 0, RQ_LOW, RQ_MEDIUM, RQ_HIGH, RQ_REF, RQ_COUNT };
enum { RD_NONE = 0, RD_NORMALS, RD_ALBEDO, RD_GI, RD_DEV, RD_BAND0 };   /* RD_BAND0 + b: band b (M2) */

typedef struct RenderCamera {
    int32_t cx, cy, cz;        /* chunk */
    float   lx, ly, lz;        /* local offset in chunk */
    float   yaw, pitch, fov_y; /* radians */
    int     shadows;
    const RenderSky* sky;
    int      quality;          /* RQ_*; RQ_CLASSIC = today's k_render */
    int      debug;            /* RD_* (new path only) */
    uint32_t frame;            /* jitter / animation index */
    float    sun_dir[3];       /* world space (M2 --time); all 0 = the sky's direction */
    float    ev;               /* fixed EV100 (M2), NAN = auto */
} RenderCamera;

typedef struct RendererConfig { int max_w, max_h, quality; } RendererConfig;

/* --metrics: counters of one frame (renderer_metrics). The new path is
 * traced with the frame's camera next to the classic voxel DDA. */
typedef struct RenderMetrics {
    uint64_t pixels;
    uint64_t terrain;          /* new path: primary hit on the smooth surface */
    uint64_t axis_aligned;     /* ... whose shading normal has max |n_i| > 0.9995 */
    uint64_t level;            /* ... exactly (0, 1, 0): a flat 3x3 stencil (clamp plateau, coast-locked flat) */
    uint64_t flora, water, sky, voxel_mode;   /* other primary outcomes (water = water_t set) */
    uint64_t cracks;           /* classic hits terrain at t < 0.9 max_dist, new path misses */
    uint64_t cracks_below;     /* ... and the classic hit lies below the smooth surface (a true hole) */
    uint64_t classic_terrain;  /* classic primary hits on terrain voxels */
    uint64_t classic_axis;     /* ... always axis-aligned */
    uint64_t nan;              /* NaN in the new path's colour or normal */
    /* diagnostic image (host, valid until the next draw): terrain grey by sun
     * angle, red = axis-aligned normal (orange: exactly level), green = flora, blue = voxel mode,
     * black = sky, magenta = classic hit / new miss, yellow = a true hole */
    const uint32_t* image;
    int image_w, image_h;
} RenderMetrics;

#define RENDER_PERF_STAGES 7
/* --perf: per-stage GPU times (cudaEvent pairs) over every frame drawn since
 * renderer_create: p50 / p95 / max in ms. Stages: upload (page table + tile
 * tags), primary (k_primary, or k_render in classic mode), surface (k_surface),
 * sun (k_sun), compose (k_compose), readback, gpu_total (first upload ..
 * readback). */
typedef struct RenderPerf {
    int      nstages;
    const char* name[RENDER_PERF_STAGES];
    double   p50[RENDER_PERF_STAGES], p95[RENDER_PERF_STAGES], max[RENDER_PERF_STAGES];
    uint32_t frames;
} RenderPerf;

typedef struct Renderer Renderer;

/* Allocates the fixed buffers for frames up to max_w x max_h now, so call it
 * before world_create (the pool is then sized from the VRAM really left). */
Renderer* renderer_create(const RendererConfig* cfg);
void      renderer_destroy(Renderer* r);
/* Renders a w x h BGRA frame; *pixels points to host memory valid until next call. */
int       renderer_draw(Renderer* r, const RenderScene* s, const RenderCamera* c,
                        int w, int h, const uint32_t** pixels);
float     renderer_gpu_ms(const Renderer* r);
/* --metrics of the last frame drawn (runs both tracers once more). 0 ok. */
int       renderer_metrics(Renderer* r, RenderMetrics* out);
/* --perf statistics; per-frame samples are kept from renderer_create on.
 * renderer_perf_reset drops them (e.g. after --settle). */
void      renderer_perf(const Renderer* r, RenderPerf* out);
void      renderer_perf_reset(Renderer* r);
/* last frame's per-stage GPU ms (RENDER_PERF_STAGES values) */
const float* renderer_perf_last(const Renderer* r);

#ifdef __cplusplus
}
#endif
#endif

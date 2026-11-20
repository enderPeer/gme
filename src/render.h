/* render.h - CUDA renderer that draws straight from the GPU chunk pool.
 *
 * Two paths (docs/render-upgrade-spec.md):
 *   RQ_CLASSIC  today's voxel ray-marcher (k_render), byte-for-byte unchanged
 *   RQ_LOW..REF the smooth-surface renderer: a 2.5D column tracer over the
 *               world's column tiles (density.h, trace.cuh), analytic trees,
 *               shrubs and kelp, voxel mode for caves and missing tiles.
 *               M2 lights it spectrally (spectral.h, atmo.cu): the star
 *               through a Hillaire-style atmosphere (sky-view LUT, aerial
 *               perspective), GGX + energy-compensated diffuse, refraction and
 *               spectral water, sky SH x heightfield AO ambient, histogram
 *               exposure, CAT16, AgX. Low and Medium are direct-light presets
 *               (High / Ref equal Medium until M4 adds indirect light).
 *               M3 adds distance: each tile's 4 m lattice past lod_fine, a
 *               renderer-owned far-field clipmap (16 m to 8 km, 64 m to 32 km,
 *               curvature drop) beyond the loaded radius and over tiles not yet
 *               streamed in, and long-range sun shadows (8 km).
 *               M4 adds indirect light and the frame pipeline (gi.cu): half-res
 *               GI rays in 8 bands with a world-space radiance cache and SVGF,
 *               traced reflections on water, ice and wet ground, a filtered sun
 *               visibility, Halton-jittered primary rays resolved by TAA / TAAU,
 *               the High and Ref presets, the reference transport of --accum
 *               (4-bounce paths, no denoising), the page / tile tag scatter and
 *               an asynchronous readback (renderer_draw returns the frame
 *               before the one it queued).
 *               M5 colours life by the pigment of its hue bin (spectral.h:
 *               Soret + Q band; leaf reflectance / transmittance, mats, grass,
 *               kelp) and draws clumped crowns, kelp blades and grass tufts
 *               (shading); crowns are a leaf-area medium for their own light,
 *               and for every shadow ray in High.
 *               Version 2 biomaps (biomap.h morphology; spec 5.3) give shrubs
 *               and trees one of five crown classes (flora.h), every cell its
 *               own pigment amount and Q band width, and a crown's leaf medium
 *               its porosity and leaf angle. Creatures (fauna_scene.h) are
 *               spheres drawn over both paths (renderer_set_fauna). */
#ifndef GME_RENDER_H
#define GME_RENDER_H

#include <stdint.h>
#include "worldgen.h"
#include "biomap.h"
#include "density.h"
#include "fauna_scene.h"
#include "spectral.h"

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
    /* M4 scatter lists: the page entries / tile tags (indices into page / tile_tag)
     * that changed since the last world_get_scene. When page_dirty (tile_dirty) is
     * set and the list is NULL, the whole table changed (or the list overflowed):
     * the renderer uploads all of it. */
    const uint32_t* page_dirty_idx; uint32_t page_dirty_n;
    const uint32_t* tile_dirty_idx; uint32_t tile_dirty_n;
} RenderScene;

enum { RQ_CLASSIC = 0, RQ_LOW, RQ_MEDIUM, RQ_HIGH, RQ_REF, RQ_COUNT };
/* hit kinds of the smooth path (renderer_read_linear, probe): trace.cuh's HK_* */
enum { RK_NONE = 0, RK_TERRAIN, RK_CROWN, RK_TRUNK, RK_KELP, RK_VOXEL, RK_ICE, RK_LOD, RK_FAR, RK_DIST, RK_FAUNA };
/* debug views: RD_BAND0 + b shows band b; RD_GI the indirect irradiance (M4; the
 * sky ambient's AO without GI), RD_AGE the TAA history age (red 1 frame, yellow 2,
 * green 3-7, blue >= 8; disoccluded pixels must be red / yellow), RD_VIS the
 * filtered sun visibility, RD_REFL the traced reflections, RD_AO the heightfield AO */
enum { RD_NONE = 0, RD_NORMALS, RD_ALBEDO, RD_GI, RD_DEV, RD_BAND0 };
/* RD_PALETTE (M5): the albedo with the old HSV palette tint of tissue (the pigment model replaced it; debug only) */
enum { RD_AGE = RD_BAND0 + 16, RD_VIS, RD_REFL, RD_AO, RD_PALETTE };

typedef struct RenderCamera {
    int32_t cx, cy, cz;        /* chunk */
    float   lx, ly, lz;        /* local offset in chunk */
    float   yaw, pitch, fov_y; /* radians */
    int     shadows;
    const RenderSky* sky;
    int      quality;          /* RQ_*; RQ_CLASSIC = today's k_render */
    int      debug;            /* RD_* (new path only) */
    uint32_t frame;            /* jitter / animation index */
    float    sun_dir[3];       /* world space toward the star; all 0 = the sky's direction */
    float    ev;               /* fixed EV100 (M2), NAN = auto exposure */
    float    dt;               /* seconds since the previous frame (exposure adaptation); <= 0 snaps */
    float    wb;               /* CAT16 degree D (0 = the raw alien light, 1 = full D65); NAN = 0.8 */
    /* M3 distance (0 = defaults) */
    int      far_off;          /* 1: no far field (M2: missing tiles in voxel mode, edge fade at the loaded radius) */
    float    far_ring;         /* near / far handover radius in m (test knob; 0 = the loaded radius) */
    float    lod_fine;         /* fine -> 4 m LOD distance in m (0 = the preset's: Low 300, Medium 600, High 900) */
    /* M4 frame pipeline (0 = defaults) */
    float    scale;            /* internal render scale of the smooth path, TAAU to w x h (0 = 1; Low multiplies it by 0.67) */
    int      gi_off;           /* 1: no indirect light rays (sky SH x heightfield AO, as Low) */
    int      ref;              /* 1: reference transport (--accum; the Ref preset always): full-res 4-bounce paths, no
                                * denoising, binary sun visibility, frames averaged while the camera stands still */
    float    anim;             /* animation time of waves and kelp, seconds (main.c: real time in the interactive
                                * viewer, frame / 60 for --frames and the test modes; --still holds it) */
    int      reset;            /* 1: drop every temporal history (TAA, GI, V, accumulation) this frame */
    int      furnace;          /* D4 only (renderer_d4_furnace) */
} RenderCamera;

/* max_w x max_h: the largest output frame; quality: the starting preset (RQ_CLASSIC allocates nothing before
 * world_create: a classic session sizes its pool as before the upgrade, and the first smooth frame allocates
 * what it needs); scale: the starting --scale of the smooth path (0 = 1), whose internal buffers are sized
 * from it (they grow when Ref or a larger scale is selected) */
typedef struct RendererConfig { int max_w, max_h, quality; float scale; } RendererConfig;

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
    /* M3 distance */
    uint64_t lod, far;         /* primary hits on the 4 m LOD lattice / the far-field clipmap */
    uint64_t distant;          /* below the geometric horizon beyond the far field (sky-view LUT ground) */
    uint64_t below_horizon;    /* pixels looking below the geometric horizon (camera above the water) */
    uint64_t sky_below_horizon;/* ... that see the sky: must be 0 */
    uint64_t sun_lit;          /* shadow rays (k_sun) that reach the sun */
    uint64_t shadow_near, shadow_mid, shadow_far;   /* ... blocked by an occluder at < 512 m, 512 m - 2 km, >= 2 km */
    /* diagnostic image (host, valid until the next draw): terrain grey by sun
     * angle, red = axis-aligned normal (orange: exactly level), LOD bluish, far
     * field brownish, distant ground purple, green = flora, blue = voxel mode,
     * black = sky, cyan = sky below the horizon, magenta = classic hit / new
     * miss, yellow = a true hole */
    const uint32_t* image;
    int image_w, image_h;
    /* M3 shadow-distance image (same size): the frame in grey, shadowed pixels
     * by occluder distance: blue < 512 m, red 512 m - 2 km, yellow >= 2 km */
    const uint32_t* shadow_image;
} RenderMetrics;

#define RENDER_PERF_STAGES 12
/* --perf: per-stage GPU times (cudaEvent pairs) over every frame drawn since
 * renderer_create: p50 / p95 / max in ms. Stages: upload (page table + tile
 * tags: scatter of the changed entries, or the whole table), far (far-field
 * clipmap update + the near walk's ceiling), atmo (sky-view when the sun moved,
 * aerial froxels), primary (k_primary, or k_render in classic mode), surface
 * (k_surface, guides), sun (k_sun), gi (secondary rays: GI + reflections: ray
 * generation, trace, NEE, shading; the reference paths), denoise (SVGF temporal
 * + a-trous + upsample, radiance cache update, sun visibility filter), compose
 * (white point, k_compose, histogram, exposure), taa (TAA / TAAU or the
 * reference accumulation, tonemap), readback (queued copy of the frame),
 * gpu_total (first upload .. readback). frame kernels = far .. taa. */
enum { RS_UPLOAD = 0, RS_FAR, RS_ATMO, RS_PRIMARY, RS_SURFACE, RS_SUN, RS_GI, RS_DENOISE, RS_COMPOSE, RS_TAA, RS_READBACK,
       RS_TOTAL };
typedef struct RenderPerf {
    int      nstages;
    const char* name[RENDER_PERF_STAGES];
    double   p50[RENDER_PERF_STAGES], p95[RENDER_PERF_STAGES], max[RENDER_PERF_STAGES];
    uint32_t frames;
} RenderPerf;

/* Probe (key L): light at the crosshair hit of the last frame. Band values are
 * band-mean spectral irradiance (W m^-2 nm^-1) on the hit surface. */
typedef struct RenderProbe {
    int    valid;              /* 0: the crosshair sees the sky */
    int    kind;               /* hit kind (RK_*): 1 terrain, 2 crown, 3 trunk, 4 kelp, 5 voxel, 6 ice, 7 LOD, 8 far field, 9 distant,
                                * 10 a creature */
    int    material;           /* dominant MAT_* */
    float  pos[3];             /* absolute world position */
    float  normal[3];
    float  depth;              /* metres below the water plane (0 = above) */
    float  sun_vis;            /* shadow-ray visibility 0/1 */
    float  ao;
    float  sun_elev_deg;
    float  e_sun[SPEC_NB], e_sky[SPEC_NB];   /* on the surface: direct (with shadow, n.l, water) and sky (SH x AO, water) */
    float  e_hor[SPEC_NB];     /* unshadowed global horizontal irradiance at the water plane / ground (no water) */
    /* M5 */
    int    material1;          /* the second material and its weight */
    float  w1;
    int    pal;                /* biomap palette index = the pigment's hue bin (0 = the default bin) */
    float  cover;              /* the cell's cover 0..1 */
    float  tsun;               /* crowns: transmittance of the own crown toward the sun (in e_sun); else 1 */
    /* version 2 (biomap morphology) */
    int    pig_level;          /* pigment level of the cell, 1..15 = PIG 17 x level; 0 = no morphology (PIG 200, Q band 25 nm) */
    int    qw_bin;             /* Q band width bin 0..15 (15 + 3 bin nm), with pig_level > 0 */
} RenderProbe;

typedef struct Renderer Renderer;

/* Allocates the fixed buffers of the starting preset for frames up to max_w x
 * max_h now (none for a classic start), so call it before world_create (the
 * pool is then sized from the VRAM really left). */
Renderer* renderer_create(const RendererConfig* cfg);
void      renderer_destroy(Renderer* r);
/* Spectral light of the planet (spectral_build_atmo): uploads the LUTs and
 * tables. Without it the smooth path is not drawn (classic is). 0 ok. */
int       renderer_set_atmo(Renderer* r, const SpectralAtmo* a);
/* Creatures (fauna_scene.h): uploads the sphere records grouped by cell and
 * the hash grid of a FaunaScene (its grid, ngrid, cells, ncells; copies;
 * nsph == 0 removes them) and the palette of the classic tint (sRGB per
 * palette index). Both paths draw them from the next frame on: the smooth
 * one as spheres with a skin of the palette index's pigment (primary rays
 * and the sun's shadow ray see them; GI and reflection rays do not), the
 * classic one with a plain tint. Without creatures no kernel of theirs runs
 * and no frame changes. Call it before world_create (36 + 16 B of VRAM per
 * sphere, 24 B per grid slot). 0 ok. */
int       renderer_set_fauna(Renderer* r, const FaunaSphere* sph, uint32_t nsph, const FaunaCell* cells, uint32_t ncells,
                             const uint8_t palette[16][3]);
/* Queues a w x h BGRA frame and returns (*pixels, host memory valid until the
 * next call) the frame queued by the call before it, which the GPU finished
 * while this one was queued (M4 asynchronous readback, 2 frames in flight:
 * each frame's kernels write its own device frame, a copy stream reads it back
 * while the next frame's kernels run). The first frame, a frame whose size
 * differs from the one before, and every frame in sync mode are waited for and
 * returned themselves. renderer_finish waits for the last one. */
int       renderer_draw(Renderer* r, const RenderScene* s, const RenderCamera* c,
                        int w, int h, const uint32_t** pixels);
/* M4: waits for the frame queued by the last renderer_draw and returns it
 * (screenshots, the end of a run). 0 ok. */
int       renderer_finish(Renderer* r, const uint32_t** pixels);
/* M4: the GPU event recorded after the last queued frame's kernels (a
 * cudaEvent_t). The world's generation streams wait for it before they write
 * chunk slots or tile records, which the frame may still be reading
 * (world_set_render_fence). */
void*     renderer_fence(const Renderer* r);
/* M4: drops every temporal history (TAA, GI, sun visibility, radiance cache, accumulation) */
void      renderer_reset_history(Renderer* r);
/* M4: NaN pixels found (shaded colour, GI, TAA) in the frames finished so far */
uint64_t  renderer_nan_count(const Renderer* r);
/* M4: radiance cache lookups at secondary-ray hits so far, and how many found a cell (32-bit device counters) */
void      renderer_cache_stats(const Renderer* r, uint64_t* lookups, uint64_t* hits);
/* M4: frames averaged by the reference transport (RenderCamera.ref) so far */
uint32_t  renderer_accum_frames(const Renderer* r);
/* M4: 1 = every renderer_draw waits for its own frame and returns it (A/B timing, tests) */
void      renderer_set_sync(Renderer* r, int sync);
float     renderer_gpu_ms(const Renderer* r);
/* exposure of the last frame as EV100 (k = 1 / (1.2 2^EV100)) */
float     renderer_ev100(const Renderer* r);
/* --metrics of the last frame drawn (runs both tracers once more). 0 ok. */
int       renderer_metrics(Renderer* r, RenderMetrics* out);
/* --perf statistics over the per-frame samples kept since renderer_create (a
 * ring of the last 65,536 frames). renderer_perf_reset drops them (e.g. after
 * --settle). */
void      renderer_perf(const Renderer* r, RenderPerf* out);
void      renderer_perf_reset(Renderer* r);
/* last frame's per-stage GPU ms (RENDER_PERF_STAGES values) */
const float* renderer_perf_last(const Renderer* r);
/* light at the crosshair of the last frame drawn (smooth path). 0 ok. */
int       renderer_probe(Renderer* r, RenderProbe* out);
/* D3: the GPU sky-view LUT read back at world directions (SpectralGpuSkyFn,
 * ctx = Renderer*, which must have an atmosphere). 0 ok. */
int       renderer_d3_sky(void* ctx, const SpectralAtmo* a, const float sun[3], float cam_h, const float* dirs, int n, float* out);
/* M3: per pixel of the last frame (smooth path), 4 floats: linear luminance
 * before exposure, path length (-1 sky), horizontal distance of a straight
 * hit from the camera (-1 if none), hit kind + 16 x water mode. out holds
 * 4 * w * h floats (w, h <= the frame drawn). 0 ok. */
int       renderer_read_linear(Renderer* r, float* out, int* w, int* h);
/* M3: far-field samples rebuilt for the last frame (0 = no update) */
uint32_t  renderer_far_rebuilt(const Renderer* r);
/* D6: builds the far field around (cam_x, cam_z) (absolute metres), moves the
 * camera by (move_x, move_z) (the toroidal update), reads both levels back
 * and compares nprobe random storage slots per level with the CPU's
 * gme_far_sample of the window sample they must hold, and every mip cell with
 * the min / max of its samples. bio: host view (its ncells cells are uploaded).
 * step_ms: then 64 camera steps of 16 m (after 64 warm-up steps): GPU ms of the far stage per step that
 * rebuilt one L1 column (avg, max) and one L1 + one L2 column (avg, max).
 * The far field is rebuilt by the next frame. 0 ok. */
int       renderer_far_selftest(Renderer* r, uint32_t seed, const GmePlanet* planet, const GmeBioView* bio, size_t ncells,
                                int32_t cam_x, int32_t cam_z, int32_t move_x, int32_t move_z, uint32_t nprobe,
                                uint64_t* checked, uint64_t* bad, uint64_t* mip_bad, float step_ms[4]);

/* D4 (M4) furnace: a flat synthetic world (renderer-owned tiles, no water, no
 * flora) under a uniform sky of radiance L per g-point, no sun, every surface
 * Lambertian with albedo 0.5, drawn with the Medium pipeline (GI rays, radiance
 * cache, SVGF, TAA) for `frames` frames. Ratio = linear luminance of each
 * terrain pixel / (0.5 x the luminance of L): mean, min, max, the number of
 * pixels, and NaN pixels. ref: the reference transport instead. 0 ok. */
int       renderer_d4_furnace(Renderer* r, uint32_t seed, const GmePlanet* planet, int frames, int ref, double* ratio_mean,
                              double* ratio_min, double* ratio_max, uint64_t* pixels, uint64_t* nan);

/* M4 disocclusion check of the last frame (--disocc-test): every pixel whose
 * surface the previous frame did not see (reprojected off screen, or the
 * previous frame's surface there is another one: depth off by more than 5 % +
 * 0.25 m, normals more than 45 deg apart, or another class) is disoccluded; it
 * must carry no history older than 2 frames: TAA age, sun visibility history
 * and (half-res) GI history length. */
typedef struct RenderDisocc {
    uint64_t pixels, disoccluded, taa_old, vis_old;   /* full res: old = history age > 2 frames */
    uint64_t gi_pixels, gi_disoccluded, gi_old;      /* half res GI samples */
    uint32_t taa_age_max, gi_age_max;                /* largest age among the disoccluded pixels */
} RenderDisocc;
int       renderer_disocc(Renderer* r, RenderDisocc* out);

#ifdef __cplusplus
}
#endif

/* ---- M4 frame pipeline: shared by render.cu and gi.cu (CUDA only) ------------------------ */
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* surface classes of the guides (disocclusion: a pixel whose class changed has no history) */
enum { GC_SKY = 0, GC_TERRAIN, GC_FLORA, GC_KELP, GC_VOXEL, GC_ICE, GC_WATER, GC_UNDER, GC_DIST, GC_FAUNA };
/* guide flags (bits 28..31 of the guide's second word) */
#define GG_GI    0x10000000u   /* the point has GI (half-res guide) / uses GI (full-res guide) */
#define GG_REFL  0x20000000u   /* the point has a traced reflection */
#define GI_DISOCC_N 9          /* renderer_disocc's device counters (gi.cu) */

/* unit normal <-> 12 + 12 bit octahedral code (bits 0..23 of a guide word) */
__device__ __forceinline__ uint32_t gg_oct_pack(float nx, float ny, float nz) {
    const float s = 1.f / (fabsf(nx) + fabsf(ny) + fabsf(nz) + 1e-20f);
    float px = nx * s, pz = nz * s;
    if (ny < 0.f) {
        const float qx = (1.f - fabsf(pz)) * (px >= 0.f ? 1.f : -1.f), qz = (1.f - fabsf(px)) * (pz >= 0.f ? 1.f : -1.f);
        px = qx; pz = qz;
    }
    const uint32_t ix = (uint32_t)__float2int_rn((fminf(fmaxf(px, -1.f), 1.f) * 0.5f + 0.5f) * 4095.f);
    const uint32_t iz = (uint32_t)__float2int_rn((fminf(fmaxf(pz, -1.f), 1.f) * 0.5f + 0.5f) * 4095.f);
    return ix | (iz << 12);
}
__device__ __forceinline__ void gg_oct_unpack(uint32_t v, float n[3]) {
    const float px = (float)(v & 4095u) * (2.f / 4095.f) - 1.f, pz = (float)((v >> 12) & 4095u) * (2.f / 4095.f) - 1.f;
    float x = px, y = 1.f - fabsf(px) - fabsf(pz), z = pz;
    if (y < 0.f) {
        x = (1.f - fabsf(pz)) * (px >= 0.f ? 1.f : -1.f);
        z = (1.f - fabsf(px)) * (pz >= 0.f ? 1.f : -1.f);
    }
    const float il = rsqrtf(x * x + y * y + z * z);
    n[0] = x * il; n[1] = y * il; n[2] = z * il;
}

/* TAA history pixel (spec 6.3: RGBA16F): exposed linear sRGB + the history age in
 * frames (w < 0: a debug view's display value). Values above fp16's range are
 * clamped (AgX saturates at 2^4.03 anyway). */
typedef uint2 TaaPx;
__device__ __forceinline__ float4 taa_ld(const TaaPx* p, size_t i) {
    const uint2 v = p[i];
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&v.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&v.y));
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ void taa_st(TaaPx* p, size_t i, float4 c) {
    const float m = 65504.f;
    const __half2 a = __floats2half2_rn(fminf(fmaxf(c.x, -m), m), fminf(fmaxf(c.y, -m), m));
    const __half2 b = __floats2half2_rn(fminf(fmaxf(c.z, -m), m), fminf(fmaxf(c.w, -m), m));
    uint2 v;
    v.x = *reinterpret_cast<const uint32_t*>(&a);
    v.y = *reinterpret_cast<const uint32_t*>(&b);
    p[i] = v;
}

/* The camera of one frame, for reprojection: position relative to its chunk,
 * orthonormal basis, field of view, internal size, sub-pixel jitter and the
 * half-res grid's representative pixel offset. */
struct GiCam {
    int   cx, cy, cz;
    float o[3];
    float fw[3], rt[3], up[3];
    float th, aspect;
    int   w, h;
    float jx, jy;
    int   hx, hy;
};

/* World-space radiance cache (spec 4.6): 2^20 cells, key = (position cell of
 * size 0.25 * 2^L m with L from the camera distance, normal octant); a cell
 * keeps 8 bands of irradiance (fp16, EMA with weight 1 / min(n, 32)). meta.x =
 * key (0 = empty), meta.y = frame << 8 | n. Written once per frame and cell by
 * gi.cu's k_cache_update, read at secondary-ray hits (render.cu). */
#define GI_CACHE_BITS  20
#define GI_CACHE_N     (1u << GI_CACHE_BITS)
#define GI_CACHE_PROBE 8
struct GiCache { uint2* meta; uint4* e; };

__device__ __forceinline__ uint32_t gi_hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16;
    return x;
}
/* slot and key of the cell holding point (px, py, pz) (metres relative to chunk
 * (cx, cy, cz)), normal n, d metres from the camera */
__device__ __forceinline__ void gi_cache_cell(int cx, int cy, int cz, float px, float py, float pz, float nx, float ny, float nz, float d,
                                              uint32_t* slot, uint32_t* key) {
    int L = (int)floorf(log2f(1.f + d * 0.125f) + 0.5f);
    L = L < 0 ? 0 : (L > 6 ? 6 : L);
    const float is = 4.f / (float)(1 << L);                     /* 1 / cell size */
    const int32_t m = 128 >> L;                                 /* cells per chunk */
    const int32_t ix = cx * m + (int32_t)floorf(px * is), iy = cy * m + (int32_t)floorf(py * is), iz = cz * m + (int32_t)floorf(pz * is);
    const uint32_t oct = (nx >= 0.f ? 1u : 0u) | (ny >= 0.f ? 2u : 0u) | (nz >= 0.f ? 4u : 0u);
    uint32_t h = gi_hash32((uint32_t)ix ^ gi_hash32((uint32_t)iy ^ gi_hash32((uint32_t)iz ^ gi_hash32((uint32_t)L | (oct << 4)))));
    *slot = h & (GI_CACHE_N - 1u);
    *key = gi_hash32(h ^ 0x9E3779B9u) | 1u;
}
/* irradiance (8 bands, W/m^2) of the cell, false if absent */
__device__ __forceinline__ bool gi_cache_get(const GiCache& C, uint32_t slot, uint32_t key, float E[8]) {
#pragma unroll 1
    for (int k = 0; k < GI_CACHE_PROBE; ++k) {
        const uint32_t s = (slot + (uint32_t)k) & (GI_CACHE_N - 1u);
        const uint2 m = __ldg(C.meta + s);
        if (m.x == key) {
            if (!(m.y & 255u)) return false;
            const uint4 v = __ldg(C.e + s);
            const uint32_t w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const __half2 h2 = *reinterpret_cast<const __half2*>(&w[b]);
                const float2 f = __half22float2(h2);
                E[2 * b] = f.x; E[2 * b + 1] = f.y;
            }
            return true;
        }
    }
    return false;
}
/* 8 bands <-> 8 x fp16 */
__device__ __forceinline__ uint4 gi_pack8(const float* e) {
    uint4 v;
    uint32_t* w = &v.x;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const __half2 h2 = __floats2half2_rn(e[2 * b], e[2 * b + 1]);
        w[b] = *reinterpret_cast<const uint32_t*>(&h2);
    }
    return v;
}
__device__ __forceinline__ void gi_unpack8(uint4 v, float* e) {
    const uint32_t w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&w[b]));
        e[2 * b] = f.x; e[2 * b + 1] = f.y;
    }
}

/* Everything gi.cu's passes read and write in one frame (render.cu fills it).
 * Internal size w x h (cur.w, cur.h), half-res grid w2 x h2, output ow x oh. */
struct GiFrame {
    GiCam cur, prev;
    int   dcx, dcy, dcz;               /* cur chunk - prev chunk */
    int   hist_ok;                     /* the previous frame's buffers are valid and comparable */
    int   w2, h2, ow, oh;
    uint32_t frame;
    /* guides: x = straight-ray distance t of the visible surface (+inf sky), y = normal oct 12+12 | class << 24 | GG_* */
    const uint2* fg; const uint2* fg_prev;   /* internal res */
    const uint2* hg; const uint2* hg_prev;   /* half res (GI sample points) */
    /* GI (half res, 8 bands fp16) */
    int   gi_on, atrous;               /* a-trous passes: 0 = no filtering (reference: e_noisy is full res already) */
    const uint4* e_noisy;
    uint4* e_hist; const uint4* e_hist_prev;
    float4* m_hist; const float4* m_hist_prev;   /* luminance moments m1, m2, history length n, - */
    uint4* e_a; uint4* e_b; float* v_a; float* v_b; float* v_t;
    uint4* e_full;                     /* internal res, x = -1 (band 0 negative): no GI value here */
    float lw[8];                       /* luminance weight per band (variance / edge stopping) */
    /* sun visibility (internal res) */
    int   vis_on;
    const unsigned char* v_smp;        /* this frame's binary sample (255 lit) */
    __half* v_hist; const __half* v_hist_prev; __half* v_out;
    unsigned char* v_age; const unsigned char* v_age_prev;
    /* TAA (internal colour -> output-res history) */
    const float4* color;               /* linear sRGB, unexposed; w < 0: a debug view's display value */
    TaaPx* taa; const TaaPx* taa_prev; /* exposed linear sRGB, w = history age (frames); -1: display value (RGBA16F) */
    const float* k;                    /* device: this frame's exposure multiplier (FrameState.k) */
    const float* k_prev;               /* device: the exposure the history was resolved with (0 = unknown) */
    float ah_still, gamma_still;       /* history weight / clip width of a still pixel with a history >= 16 frames */
    int   taau;                        /* output size != internal size */
    /* traced reflections (render.cu k_sec_shade): temporally filtered here; refl_mode 1 half res (hg guides),
     * 2 full res (fg guides); band 0 = -1 marks "no reflection" */
    int   refl_mode;
    const uint4* r_cur; uint4* r_hist; const uint4* r_hist_prev;
    /* reference accumulation */
    float4* acc; uint32_t acc_n;
    /* NaN counter (device) */
    unsigned int* nan;
    /* radiance cache */
    GiCache cache;
};

struct GiPipe;
GiPipe* gi_create(void);
void    gi_destroy(GiPipe* g);
/* the radiance cache (allocated by gi_create; NULL members if VRAM ran out) */
GiCache gi_cache(GiPipe* g);
void    gi_cache_clear(GiPipe* g, cudaStream_t s);
/* passes; each returns 0 or -1 (launch error, printed) */
int     gi_denoise(GiPipe* g, const GiFrame& f, cudaStream_t s);    /* temporal, a-trous, cache update, upsample */
int     gi_vis_filter(const GiFrame& f, cudaStream_t s);             /* sun visibility: temporal + 3x3 a-trous */
int     gi_refl_filter(const GiFrame& f, cudaStream_t s);            /* traced reflections: temporal (alpha 0.2), clamped */
int     gi_resolve(const GiFrame& f, cudaStream_t s);                /* TAA / TAAU, or the reference accumulation (f.acc) */
int     gi_disocc(const GiFrame& f, const TaaPx* taa_cur, const unsigned char* v_age, const float4* m_hist_cur,
                  unsigned long long* acc, cudaStream_t s);          /* renderer_disocc's counters */
#endif
#endif

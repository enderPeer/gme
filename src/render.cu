// render.cu - real-time voxel ray-marcher.
//
// One thread per pixel walks the voxel grid (Amanatides-Woo DDA) directly in
// the device chunk pool. Uniform chunks (sky, solid rock, open water) and
// chunks not yet loaded are crossed in a single step, so a ray only does
// per-voxel work near surfaces. Everything is computed relative to the
// camera's chunk, so precision is the same everywhere in the world.
#include "render.h"
#include "flora.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STEPS 1536

// ---- small float3 helpers --------------------------------------------------
__device__ __forceinline__ float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 operator*(float3 a, float3 b) { return f3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ __forceinline__ float3 operator*(float3 a, float s)  { return f3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float  dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 norm3(float3 a) { return a * rsqrtf(dot3(a, a)); }
__device__ __forceinline__ float3 mix3(float3 a, float3 b, float t) { return a + (b - a) * t; }
__device__ __forceinline__ float  sat(float x) { return fminf(fmaxf(x, 0.f), 1.f); }

struct KParams {
    const GmePageEntry* page;
    const uint8_t* pool;
    uint32_t n, layers;
    int32_t  min_cy;
    int32_t  cx, cy, cz;
    float3   o;              // camera, relative to its chunk origin
    float3   fw, rt, up;
    float    th, aspect;
    float3   sun;
    int      shadows;
    RenderSky sky;
    float    max_dist;
    int      w, h;
    uint32_t* out;
    const uint32_t* bio_cells;   // device biomap cells (NULL = no vegetation tint)
    int32_t  bio_nreg, bio_shift;
    GmeBioRegion bio_reg[BIOMAP_MAX_REGIONS];
    float3   bio_pal[BIOMAP_PALETTE];
};

struct Hit { float t; int mat, axis, vx, vy, vz; float water_t; };

// chunk lookup, chunk coords relative to camera chunk
__device__ __forceinline__ int lookup(const KParams& p, int rcx, int rcy, int rcz) {
    const int layer = p.cy + rcy - p.min_cy;
    if (layer < 0) return -2 - MAT_STONE;          // below the world: solid
    if (layer >= (int)p.layers) return -2 - MAT_AIR; // above: sky
    const int ax = p.cx + rcx, az = p.cz + rcz;
    const GmePageEntry e = p.page[gme_page_index(ax, layer, az, p.n, p.layers)];
    return e.tag == gme_page_tag(ax, az) ? e.val : GME_PAGE_UNLOADED;
}

__device__ bool trace(const KParams& p, float3 o, float3 d, float tmax, Hit& h) {
    if (fabsf(d.x) < 1e-7f) d.x = copysignf(1e-7f, d.x);
    if (fabsf(d.y) < 1e-7f) d.y = copysignf(1e-7f, d.y);
    if (fabsf(d.z) < 1e-7f) d.z = copysignf(1e-7f, d.z);
    const float ix = 1.f / d.x, iy = 1.f / d.y, iz = 1.f / d.z;
    const int sx = d.x > 0 ? 1 : -1, sy = d.y > 0 ? 1 : -1, sz = d.z > 0 ? 1 : -1;
    const float tdx = fabsf(ix), tdy = fabsf(iy), tdz = fabsf(iz);
    const int top = (p.min_cy + (int)p.layers - p.cy) * GME_CHUNK_DIM;

    int vx = (int)floorf(o.x), vy = (int)floorf(o.y), vz = (int)floorf(o.z);
    float tmx = ((float)(vx + (sx > 0)) - o.x) * ix;
    float tmy = ((float)(vy + (sy > 0)) - o.y) * iy;
    float tmz = ((float)(vz + (sz > 0)) - o.z) * iz;
    float t = 0.f;
    int axis = 1;
    h.water_t = -1.f;

    for (int i = 0; i < MAX_STEPS; ++i) {
        if (t > tmax) return false;
        if (vy >= top && d.y >= 0.f) return false;
        const int rcx = vx >> 5, rcy = vy >> 5, rcz = vz >> 5;
        const int e = lookup(p, rcx, rcy, rcz);
        int mat;
        if (e >= 0) mat = p.pool[(size_t)e * GME_CHUNK_VOXELS + GME_VOXEL_INDEX(vx & 31, vy & 31, vz & 31)];
        else mat = (e == GME_PAGE_UNLOADED) ? MAT_AIR : (-2 - e);

        if (mat == MAT_WATER) {
            if (h.water_t < 0.f) h.water_t = t;
        } else if (mat != MAT_AIR) {
            h.t = t; h.mat = mat; h.axis = axis; h.vx = vx; h.vy = vy; h.vz = vz;
            return true;
        }

        if (e < 0) { // uniform or unloaded chunk: jump to where the ray leaves it
            const float ex = ((float)(rcx * 32 + (sx > 0 ? 32 : 0)) - o.x) * ix;
            const float ey = ((float)(rcy * 32 + (sy > 0 ? 32 : 0)) - o.y) * iy;
            const float ez = ((float)(rcz * 32 + (sz > 0 ? 32 : 0)) - o.z) * iz;
            float te;
            if (ex < ey && ex < ez) { te = ex; axis = 0; }
            else if (ey < ez)       { te = ey; axis = 1; }
            else                    { te = ez; axis = 2; }
            t = te;
            vx = (int)floorf(o.x + d.x * te);
            vy = (int)floorf(o.y + d.y * te);
            vz = (int)floorf(o.z + d.z * te);
            if (axis == 0) vx = rcx * 32 + (sx > 0 ? 32 : -1);
            else if (axis == 1) vy = rcy * 32 + (sy > 0 ? 32 : -1);
            else vz = rcz * 32 + (sz > 0 ? 32 : -1);
            tmx = ((float)(vx + (sx > 0)) - o.x) * ix;
            tmy = ((float)(vy + (sy > 0)) - o.y) * iy;
            tmz = ((float)(vz + (sz > 0)) - o.z) * iz;
            continue;
        }

        if (tmx < tmy) {
            if (tmx < tmz) { vx += sx; t = tmx; tmx += tdx; axis = 0; }
            else           { vz += sz; t = tmz; tmz += tdz; axis = 2; }
        } else {
            if (tmy < tmz) { vy += sy; t = tmy; tmy += tdy; axis = 1; }
            else           { vz += sz; t = tmz; tmz += tdz; axis = 2; }
        }
    }
    return false;
}

__device__ __forceinline__ float3 v3(const float* a) { return f3(a[0], a[1], a[2]); }

// sky: planet's atmosphere gradient, its star, and the other planets
__device__ float3 sky(const KParams& p, float3 d, bool bodies) {
    const RenderSky& s = p.sky;
    const float k = sqrtf(sat(d.y));
    float3 c = mix3(v3(s.horizon), v3(s.zenith), k);
    if (d.y < 0.f) c = c * (1.f + d.y * 0.5f);
    const float cs = dot3(d, p.sun);
    const float edge = cosf(s.sun_size);
    const float3 sc = v3(s.sun_color) * s.sun_intensity;
    if (bodies && cs > edge) c = sc * 6.f;                             // star disc
    const float glow = sat(cs);
    c = c + sc * (powf(glow, 12.f) * 0.18f + powf(glow, 200.f) * 0.6f); // corona / scattering
    if (bodies) {
        for (int i = 0; i < s.nbodies; ++i) {
            const float cb = dot3(d, v3(s.body_dir[i]));
            const float r = s.body_ang[i];
            if (cb > cosf(r)) {
                // lit side faces the star
                const float3 bd = v3(s.body_dir[i]);
                const float3 off = norm3(d - bd * cb);
                const float lit = 0.25f + 0.75f * sat(0.5f + 0.5f * dot3(off, p.sun));
                c = mix3(c, v3(s.body_color[i]) * lit * s.sun_intensity, 0.92f);
            }
        }
    }
    return c;
}

__device__ float3 albedo(int m) {
    switch (m) {
    case MAT_STONE:    return f3(0.36f, 0.36f, 0.38f);  // basalt
    case MAT_DIRT:     return f3(0.40f, 0.28f, 0.18f);
    case MAT_GRASS:    return f3(0.28f, 0.52f, 0.20f);
    case MAT_SAND:     return f3(0.82f, 0.76f, 0.52f);
    case MAT_GRANITE:  return f3(0.62f, 0.56f, 0.53f);
    case MAT_IRON_ORE: return f3(0.55f, 0.24f, 0.14f);
    case MAT_SULFIDE:  return f3(0.68f, 0.60f, 0.24f);
    case MAT_ICE:      return f3(0.82f, 0.90f, 0.97f);
    case MAT_REGOLITH: return f3(0.52f, 0.46f, 0.40f);
    case MAT_MAT:      return f3(0.30f, 0.22f, 0.30f);  // microbial mat
    case MAT_LEAF:     return f3(0.25f, 0.45f, 0.18f);
    case MAT_WOOD:     return f3(0.35f, 0.25f, 0.15f);
    case MAT_KELP:     return f3(0.20f, 0.35f, 0.18f);
    default:           return f3(1.f, 0.f, 1.f);
    }
}

// Palette index of the biomap cell over absolute column (x, z), 0 outside the
// regions. Looked up at the 8 m slot centre, like generation (flora.h).
__device__ uint32_t bio_palette_at(const KParams& p, int32_t x, int32_t z) {
    const int32_t px = (x >> GME_FLORA_SLOT_SHIFT) * 8 + 4, pz = (z >> GME_FLORA_SLOT_SHIFT) * 8 + 4;
    for (int r = 0; r < p.bio_nreg; ++r) {
        const uint32_t i = (uint32_t)(px - p.bio_reg[r].x0) >> p.bio_shift;
        const uint32_t j = (uint32_t)(pz - p.bio_reg[r].z0) >> p.bio_shift;
        if (i < (uint32_t)p.bio_reg[r].w && j < (uint32_t)p.bio_reg[r].h)
            return BIOCELL_PAL(p.bio_cells[p.bio_reg[r].offset + j * (uint32_t)p.bio_reg[r].w + i]);
    }
    return 0u;
}

__global__ void __launch_bounds__(128) k_render(KParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;

    const float u = (2.f * (x + 0.5f) / p.w - 1.f) * p.th * p.aspect;
    const float v = (1.f - 2.f * (y + 0.5f) / p.h) * p.th;
    const float3 d = norm3(p.fw + p.rt * u + p.up * v);
    const float3 sky_c = sky(p, d, true);
    const float3 fog_c = sky(p, d, false);
    const float fog_k = 1.f / (p.max_dist * 0.55f);

    Hit h;
    float3 col;
    const bool hit = trace(p, p.o, d, p.max_dist, h);
    if (hit) {
        float3 n = f3(0.f, 0.f, 0.f);
        if (h.axis == 0) n.x = d.x > 0 ? -1.f : 1.f;
        else if (h.axis == 1) n.y = d.y > 0 ? -1.f : 1.f;
        else n.z = d.z > 0 ? -1.f : 1.f;

        // stable per-voxel tint (absolute coords)
        const int32_t ax = p.cx * 32 + h.vx, az = p.cz * 32 + h.vz;
        const uint32_t hv = gme_hash3(ax, p.cy * 32 + h.vy, az, 0x5EEDu);
        float3 base = albedo(h.mat);
        if (p.bio_cells && (h.mat == MAT_GRASS || h.mat == MAT_LEAF || h.mat == MAT_MAT || h.mat == MAT_KELP)) {
            const uint32_t pi = bio_palette_at(p, ax, az);   // pigment colour from the life simulation
            if (pi) base = mix3(base, p.bio_pal[pi], 0.7f);
        }
        if ((h.mat == MAT_GRASS || h.mat == MAT_MAT) && h.axis != 1) base = mix3(base, albedo(MAT_DIRT), 0.55f);
        base = base * (0.88f + 0.12f * (float)(hv & 255u) / 255.f);

        const float ndl = sat(dot3(n, p.sun));
        float shadow = 1.f;
        if (p.shadows && ndl > 0.f) {
            Hit hs;
            const float3 hp = p.o + d * h.t + n * 0.01f;
            if (trace(p, hp, p.sun, 512.f, hs)) shadow = 0.f;
        }
        // star light + skylight (ambient takes the atmosphere's colour; airless = dark shadows)
        const float amb = 0.30f + 0.14f * n.y + 0.04f * n.x;
        const float3 skylight = mix3(f3(0.08f, 0.08f, 0.09f), v3(p.sky.zenith) * 0.9f + f3(0.22f, 0.22f, 0.20f), 0.9f);
        const float3 light = v3(p.sky.sun_color) * (p.sky.sun_intensity * ndl * shadow * 1.15f) + skylight * amb;
        col = base * light;
        const float fog = fmaxf(1.f - expf(-h.t * fog_k * h.t * fog_k), 0.f);
        col = mix3(col, fog_c, sat(fog));
    } else {
        col = sky_c;
    }

    if (h.water_t >= 0.f) { // translucent water: absorb along the path, reflect sky at the surface
        const float path = (hit ? h.t : p.max_dist) - h.water_t;
        const float a = 1.f - expf(-path * 0.07f);
        const float3 deep = f3(0.04f, 0.20f, 0.32f) * (0.55f + 0.45f * p.sun.y);
        float3 under = mix3(col, deep, a);
        if (h.water_t > 0.f) {
            const float c = fabsf(d.y);
            const float fres = 0.02f + 0.98f * powf(1.f - c, 5.f);
            const float3 refl = sky(p, f3(d.x, fabsf(d.y), d.z), true);
            under = mix3(under, refl, fres * 0.7f);
            const float fog = 1.f - expf(-h.water_t * fog_k * h.water_t * fog_k);
            under = mix3(under, fog_c, sat(fog));
        }
        col = under;
    }

    // crosshair
    const int dx = x - p.w / 2, dy = y - p.h / 2;
    if ((dy == 0 && abs(dx) < 9 && abs(dx) > 2) || (dx == 0 && abs(dy) < 9 && abs(dy) > 2))
        col = f3(1.f, 1.f, 1.f) - col;

    const uint32_t r = (uint32_t)(sqrtf(sat(col.x)) * 255.f + 0.5f);
    const uint32_t g = (uint32_t)(sqrtf(sat(col.y)) * 255.f + 0.5f);
    const uint32_t b = (uint32_t)(sqrtf(sat(col.z)) * 255.f + 0.5f);
    p.out[(size_t)y * p.w + x] = 0xFF000000u | (r << 16) | (g << 8) | b;
}

// ===========================================================================
// Smooth-surface path (docs/render-upgrade-spec.md M1 - M4). Everything above
// this line is the classic renderer, kept byte for byte (--quality classic).
//
// Wavefront (spec 3.1, 4); M4 additions marked +:
//   upload      + the page entries / tile tags the world changed (one scatter
//               list, k_scatter); the whole table only when it is new or more
//               than 1/8 of it changed
//   far field   (M3) k_far_update rebuilds the clipmap rows / columns that
//               entered the windows, k_far_mipA / B the min / max mips,
//               k_near_bound the near walk's ceiling
//   atmo.cu     sky-view LUT + SH when the sun moved, aerial froxels (out to
//               the far field's edge)
//   k_primary   column tracer -> hit record: fine 1 m patches to lod_fine, the
//               4 m lattice to the ring, the far field to 32 km (curved). The
//               water plane (curved beyond the loaded radius) refracts the ray
//               (Gerstner normals, n = 1.333) and the walk continues in the
//               same thread; from below: Snell window, else total internal
//               reflection. Below the geometric horizon beyond the far field:
//               the distant ground.
//   k_surface   smooth normals, depth-anchored material ids (2 dominant +
//               weight; far: the far material model), heightfield AO -> G-buffer
//               (24 B / pixel)
//   k_sun       one shadow ray toward a jittered point of the stellar disc;
//               under water along the refracted sun direction to the surface;
//               M3: through the LOD tiles and the far field to 8 km;
//               + the disc point moves every frame (R2), the binary sample
//               goes to the visibility filter (gi.cu)
//   +k_sec_gen  per half-res pixel (its representative pixel rotates over the
//               2 x 2 every frame): cosine GI rays (IGN + R2), the reflection
//               ray of water (mirror at the Gerstner normal), ice and the wet
//               tidal band (mirror / GGX); the half-res guide
//   +k_sec_trace  the column tracer: GI fine 8 m, LOD 256 m, crowns 64 m;
//               reflections fine 64 m, LOD + far field to the horizon; the
//               water plane ends rays from above
//   +k_sec_nee  the hit's normal and materials, its sun visibility (coarse)
//   +k_sec_shade  rho / pi (sun NEE + radiance-cache irradiance) per band, the
//               sky without the disc on a miss; E_ind = pi mean L (8 bands);
//               reference: 4-bounce paths instead (no cache, no denoising)
//   +gi.cu      SVGF (temporal, a-trous, upsample), radiance cache update,
//               sun visibility filter
//   k_white     CAT16 white: global horizontal irradiance at the camera's ground;
//               + each g-point's share of its band's sky irradiance
//   k_compose   spectral shading in 32 g-points (spectral.h): the star through
//               the atmosphere, GGX + energy-compensated diffuse, sky SH x AO
//               (+ Medium / High / Ref: the GI irradiance), wet tidal band,
//               water volume (band transmittance + in-scatter), water surface
//               (Fresnel, sky reflection (+ traced), glint), + caustics (High),
//               aerial perspective, the far field's fade into the sky-view at
//               30-32 km, sky, stellar disc, planets -> XYZ -> CAT16 -> linear sRGB
//   k_lum_hist, k_exposure   log2 Y histogram of every 4th pixel, mean of the
//               5-95 % range, adaptation
//   +gi.cu      TAA / TAAU into the output-resolution history (or the reference
//               accumulation)
//   k_tonemap   AgX, sRGB OETF, dither, crosshair -> BGRA (+ the history-age view)
//   readback    + asynchronous: 2 pinned frames, renderer_draw returns the
//               frame before the one it queued
// M5 life colours and shapes: tissue spectra from the pigment of each cell's
// hue bin (spec 5.1: leaves rho_L / tau_L, mats, grass mix(soil, leaf, 0.8
// cover), kelp); clumped crowns for primary rays (normals from the billows,
// leaf-cluster bumps, the clumps' depth as occlusion); a crown is a leaf-area
// medium for its own light (k_surface: the sun's e^-(sigma chord) through it,
// k_sun: shadow rays from where they leave it, k_compose: back-lit
// transmission tau / pi E); High's shadow rays see other crowns as that medium
// too; kelp blades transmit (two-sided); grass tufts as a normal fuzz and a
// sheen lobe; --debug palette keeps the old HSV tint
// Version 2 biomaps (spec 5.3; biomap.h morphology): crowns by class (trace.cuh:
// ellipsoid, cone, umbrella, column, dome), tissue from the cell's own pigment
// amount and Q band width (ShadeTab.pig_A2; the G-buffer carries both), the
// crown's porosity and leaf angle in its leaf-area medium.
// Creatures (fauna_scene.h): spheres found through a hash grid of 32 m cells,
// in passes that run only while some are near the camera: k_fauna_anim (their
// positions of the frame), k_fauna_primary (after k_primary: a nearer sphere
// replaces the hit record), k_fauna_sun (after k_sun: their shadows); k_surface
// and k_compose shade them as skin of their palette index's pigment. Classic:
// k_fauna_classic over k_render's frame (k_render itself is untouched).
// ===========================================================================
#include "trace.cuh"
static_assert(HK_TERRAIN == RK_TERRAIN && HK_LOD == RK_LOD && HK_FAR == RK_FAR && HK_DIST == RK_DIST && HK_FAUNA == RK_FAUNA,
              "render.h RK_* = trace.cuh HK_*");

// Gerstner normals only (no displacement), spec 3.6: a stochastic directional
// spectrum of WAVES components (8 bands of 20 .. 0.3 m, WAVE_PER components per
// band with jittered wavelengths and directions spread about a 5 m/s wind by a
// cos^2s spreading function, random phases), sorted by wave number. Single
// sinusoids per band drew a woven lattice in the sun glint and a grid in the
// caustics; the band slope variances are unchanged (Cox & Munk budget).
#define WAVE_BANDS 8
#define WAVE_PER   8
#define WAVES   (WAVE_BANDS * WAVE_PER)
#define WATER_N 1.333f

struct Waves {
    float kx[WAVES], kz[WAVES]; // wave vector
    float dx[WAVES], dz[WAVES]; // unit direction
    float k[WAVES];             // |k|, ascending
    float s[WAVES];             // slope amplitude k A
    float q[WAVES];             // crest steepness of the normal
    float pm[WAVES];            // gain of the slow phase noise (decorrelates the components further)
    float tail[WAVES];          // sum of 0.5 s^2 over components i.. (the variance faded out beyond i)
    float off[WAVES];           // phase at the camera chunk origin (host, double precision)
    float alpha0;               // roughness of the unresolved capillary waves
};

// spectral shading tables (device memory, read through the cache)
struct ShadeTab {
    float rho[SPEC_NMAT][SPEC_NB];
    float alpha[SPEC_NMAT];
    int   ior[SPEC_NMAT];
    float pyrite_f0[SPEC_NB];
    float e_lut[SPEC_NIOR][SPEC_EL_N][SPEC_EL_N];
    float e_avg[SPEC_NIOR][SPEC_EL_N];
    float water_lnT[SPEC_NB][SPEC_WT_N];
    float water_a[SPEC_NB], water_b[SPEC_NB], water_bm[SPEC_NB];
    float basis[3][SPEC_NB];
    float cmf_band[SPEC_NB][3];
    float dbg_rgb[9];           // XYZ (equal-energy white, Y = 1) -> linear sRGB (D65)
    // M5 pigmented tissue per hue bin (index 0 = the default bin), spec 5.1
    float leaf_rho[SPEC_NHUE][SPEC_NB];
    float leaf_tau[SPEC_NHUE][SPEC_NB];
    float mat_tis[SPEC_NHUE][SPEC_NB];
    // version 2 (biomap morphology, spec 5.3): the absorptance of a cell's own pigment amount (level - 1)
    // and Q band width bin; leaf rho / tau and the mat follow as rho_L = 0.04 + 0.44 (1 - A) ...
    float pig_A2[SPEC_NHUE][SPEC_NPIG][SPEC_NQW][SPEC_NB];
    // creatures: the skin of each hue bin, a darker and less saturated leaf
    float skin[SPEC_NHUE][SPEC_NB];
};
// a creature's skin in the material tables: the row of MAT_AIR, which never is a surface (id 15, the
// one MAT_* leaves free, is MATX_SOILMAT)
#define RMAT_SKIN 0
static_assert(RMAT_SKIN == MAT_AIR && MAT_COUNT == 15, "RMAT_SKIN uses the table row of MAT_AIR; 15 is MATX_SOILMAT");

// per-frame state on the device: white balance, exposure, light at the water level
struct FrameState {
    float cat[9];               // XYZ -> linear sRGB with CAT16
    float k;                    // exposure multiplier
    float log2k;
    int   init;
    float ev100;
    float white[3];             // XYZ of the white (global horizontal irradiance at the camera's ground)
    float eup[SPEC_NG];         // sky irradiance on a horizontal plane at the water level, W/m^2
    float esun0[SPEC_NG];       // sun beam irradiance at the water level (normal to the beam), W/m^2
    // M4
    float gsplit[SPEC_NG];      // each g-point's share of its band's sky irradiance (8-band GI -> g-points)
    float gglob[SPEC_NG];       // ... of its band's global horizontal irradiance (sun + sky; band radiance -> XYZ)
    unsigned int nan;           // NaN pixels found so far (shaded colour, GI, TAA)
    float k_prev;               // the exposure of the previous frame (its TAA history), 0 = none
    unsigned long long cache_lookups, cache_hits;   // radiance cache lookups at GI / reflection hits so far, and hits
};

// M4 secondary rays (GI, reflections, reference paths): one slot per ray, SoA.
// r.y flags: bit 0 active (to trace), 1-2 type, 3 path (reference: the slot
// carries a path; finished paths keep it for the gather), 4-7 hit kind
// (HK_*, HKX_WATER), 8 sun visible (NEE), 9-11 bounce, 12 starts in voxel
// mode, 16-23 beach share of a far material. Trace: r.z, r.w = hit record,
// h = (t, u, v, -). NEE: r.z = shading normal (oct), r.w = material pack.
enum { ST_GI = 0, ST_MIRROR = 1, ST_GLOSSY = 2 };
#define SF_ACTIVE 1u
#define SF_PATH   8u
#define SF_VIS    0x100u
#define SF_VOXEL  0x1000u
#define HKX_WATER 15
struct SecBuf {
    float4* o;                  // origin (relative to the camera chunk), w = tmax
    uint4*  r;
    float4* h;
    uint4*  thr;                // reference: path throughput, 8 x fp16
    uint4*  acc;                // reference: path radiance, 8 x fp16
};
// a grid of ray-generating pixels: half res (Medium / High GI: pixel (i, j) is
// the internal pixel (2i + hx, 2j + hy)) or full res (reference, High reflections)
struct SecGrid {
    int    gw, gh, K;           // grid size, slots per grid pixel
    int    half;
    int    gi_n;                // GI rays per pixel (slots 0 .. gi_n - 1)
    int    refl;                // slot gi_n: one reflection ray
    int    glossy;              // ice / wet: GGX-sampled reflection (else mirror)
    int    base;                // first slot in SecBuf
    uint4* e_out;               // GI irradiance per grid pixel (8 x fp16, band 0 = -1: none)
    uint4* r_out;               // reflected radiance per grid pixel (band 0 = -1: none)
    uint2* hg;                  // half-res guide (NULL for full-res grids)
};

struct ProbeOut {
    int   valid, kind, mat;
    float pos[3], n[3], depth, vis, ao, sun_y;
    float e_sun[SPEC_NB], e_sky[SPEC_NB], e_hor[SPEC_NB];
    int   pal, mat1;              // M5: hue bin (palette index) and the second material
    float cover, w1, tsun;        // ... the cell's cover, the second material's weight, a crown's own transmittance
    int   pig, qw;                // version 2: pigment level (0 = none) and Q band width bin of the cell
};

enum { GF_HIT = 1u, GF_TERRAIN = 2u, GF_SHADOW = 4u, GF_UNDER = 8u };

// Creatures (fauna_scene.h): the device arrays and the frame's camera chunk and time. n = 0: none near
// the camera (their kernels are not launched).
struct FaunaDev {
    const FaunaSphere* sph;         // records grouped by cell
    const float4*      pos;         // this frame (k_fauna_anim): each record's moving centre relative to the camera chunk, radius
    const FaunaCell*   cells;
    uint32_t mask;                  // slots - 1 of the hash grid
    uint32_t n;                     // spheres
    float    ylo, yhi;              // height range of all spheres (absolute metres)
    int32_t  cx, cy, cz;            // camera chunk
    float    time;                  // seconds (idle motion; 0 = still)
    int      classic;               // rest heights of the classic view (on the voxel ground)
};
struct FaunaPal { float3 c[BIOMAP_PALETTE]; };   // classic tint per palette index (linear RGB)
// water mode of a pixel's path: bits 24..26 of b.x (miss records) or b.y (G-buffer)
enum { WM_NONE = 0, WM_STRAIGHT, WM_REFR, WM_UNDER, WM_UNDER_OUT, WM_UNDER_TIR };

// (The column tracer's context of the frame, TCtx, is not a kernel parameter: it is uploaded to constant
// memory (g_tc) and kept next to these in the Renderer (lastt). SParams itself is above 4 KB, the kernel
// parameter limit of CUDA toolkits before 12.1, since the 64-component wave table: see the assert below.)
struct SParams {
    KParams k;                  // camera, classic sky, vegetation tint (shared with the classic kernel)
    int     debug;              // RD_*
    unsigned long long* dbg;    // --metrics: NaN counter of the shaded colour (NULL = off)
    // M2
    AtmoParams a;               // spectral atmosphere (atmo.cu)
    Waves   w;
    const ShadeTab* tab;
    FrameState* fs;
    float4* color;              // linear sRGB after CAT16, before exposure (w < 0: a debug view's display value)
    unsigned int* hist;         // 64 bins of log2 Y over [-8, 16]
    ProbeOut* probe;
    float3  sun;                // toward the star (world)
    float3  sun_r;              // toward the star from under the water (refracted at a flat surface)
    float   sun_t;              // (1 - F) mu_s / mu_r: beam irradiance just below the surface per unit above
    float   cam_h;              // camera altitude above the water plane (m, >= 0)
    float   pix;                // radians per pixel
    int     refract, jitter, ao_far;
    float   wb, ev, dt;
    uint32_t frame;
    int     ow, oh;                 // output size (k_tonemap); k.w x k.h is the internal size (Low: 0.67)
    uint32_t* outp;                 // output BGRA, ow x oh
    // M3
    float   mu_h;                   // direction cosine of the geometric horizon (the sea sphere seen from the camera)
    unsigned long long* sacc;       // --metrics: k_sun bins the occluder distance here (NULL = off)
    float*  sdist;                  // --metrics: k_sun's occluder distance per pixel (-1 lit, 0 untested; NULL = off)
    // M4 frame pipeline
    float   jx, jy;                 // sub-pixel jitter of the primary rays (internal pixels)
    int     dither_t;               // temporal dithers (TAA resolves them): frame-dependent offset, else static
    int     furnace;                // D4: uniform sky, no sun, albedo 0.5 Lambertian, no air path
    int     gi;                     // 0: sky SH x heightfield AO; 1: GI irradiance from efull
    int     ref;                    // reference transport
    int     refl;                   // 0: sky reflections; 1: traced, half res (hx, hy); 2: traced, full res
    int     vis_mode;               // 0: the shadow bit; 1: the filtered visibility (vis); 2: binary samples (vsmp)
    int     caustics;               // High: caustics on sunlit points under water
    int     w2, h2, hx, hy;         // half-res grid and this frame's representative pixel offset
    uint2*  fg;                     // full-res guide (k_surface writes it)
    unsigned char* vsmp;            // binary sun visibility samples (k_sun writes them)
    const __half* vis;              // filtered sun visibility
    const uint4* efull;             // GI irradiance per internal pixel (8 x fp16; band 0 = -1: none)
    const uint4* refl_buf;          // reflected radiance (half or full res; band 0 = -1: none)
    const uint2* hg;                // half-res guide (refl == 1)
    GiCache cache;
    const TaaPx* taa;               // resolved output (k_tonemap, output res; RGBA16F)
    unsigned int* nan;              // NaN counter (FrameState.nan)
    // M5
    int     lad;                    // High: shadow rays (k_sun, k_sec_nee) treat crowns as a leaf-area medium
    // creatures
    FaunaDev fauna;
};

// every kernel takes SParams as a __grid_constant__ parameter (CUDA 12.1+: up to 32,764 bytes)
static_assert(sizeof(SParams) <= 32000, "SParams must fit the kernel parameter space");

// ---- small helpers -------------------------------------------------------------
__device__ __forceinline__ float frc(float x) { return x - floorf(x); }
__device__ __forceinline__ float ign(float x, float y) { return frc(52.9829189f * frc(0.06711056f * x + 0.00583715f * y)); }
__device__ __forceinline__ float smooth01(float e0, float e1, float x) {
    const float t = sat((x - e0) / (e1 - e0));
    return t * t * (3.f - 2.f * t);
}
__device__ __forceinline__ float3 pixel_dir(const KParams& p, int x, int y) {
    const float u = (2.f * (x + 0.5f) / p.w - 1.f) * p.th * p.aspect;
    const float v = (1.f - 2.f * (y + 0.5f) / p.h) * p.th;
    return norm3(p.fw + p.rt * u + p.up * v);
}
// M4: the frame's Halton-jittered primary ray of pixel (x, y) (= pixel_dir without jitter)
__device__ __forceinline__ float3 sp_dir(const SParams& sp, int x, int y) {
    const KParams& p = sp.k;
    const float u = (2.f * ((float)x + 0.5f + sp.jx) / p.w - 1.f) * p.th * p.aspect;
    const float v = (1.f - 2.f * ((float)y + 0.5f + sp.jy) / p.h) * p.th;
    return norm3(p.fw + p.rt * u + p.up * v);
}
// per-pixel dither in [0, 1): static IGN, moved by the golden ratio every frame
// when TAA resolves it (g_tc.dith = frac(0.618 frame), M4)
__device__ __forceinline__ float dith(float x, float y) { return frc(ign(x, y) + g_tc.dith); }
// M3 per-pixel switch distances of a primary ray, dithered +-5 %: the fine
// surface ends at lod_t, the analytic flora at flora_t (temporal since M4)
__device__ __forceinline__ float lod_t(int x, int y) { return g_tc.lod_fine * (0.95f + 0.1f * dith((float)x + 3.7f, (float)y + 11.3f)); }
__device__ __forceinline__ float flora_t(int x, int y) { return g_tc.flora_far * (0.95f + 0.1f * dith((float)x + 9.1f, (float)y + 4.7f)); }

// Voxel at absolute (ax, ay, az) of a column of height h: pool byte, uniform
// chunk, or (chunk not resident yet) the analytic gme_voxel, exactly as
// generation computes it.
__device__ __noinline__ int sm_voxel_analytic(int32_t ax, int32_t ay, int32_t az, int32_t h) {
    GmeSlot s;
    gme_slot(ax >> GME_FLORA_SLOT_SHIFT, az >> GME_FLORA_SLOT_SHIFT, g_tc.seed, &g_tc.planet, &g_tc.bio, &s);
    GmeColumn col;
    gme_column_in_slot(ax, az, h, g_tc.seed, &s, &col);
    return gme_voxel(ax, ay, az, &col, g_tc.seed, &g_tc.planet);
}
__device__ __forceinline__ int sm_voxel(int32_t ax, int32_t ay, int32_t az, int32_t h) {
    const int e = tr_lookup((ax >> 5) - g_tc.cx, (ay >> 5) - g_tc.cy, (az >> 5) - g_tc.cz);
    if (e >= 0) return g_tc.pool[(size_t)e * GME_CHUNK_VOXELS + GME_VOXEL_INDEX(ax & 31, ay & 31, az & 31)];
    if (e != GME_PAGE_UNLOADED) return -2 - e;
    return sm_voxel_analytic(ax, ay, az, h);   // chunk not resident yet (rare): exactly as generation
}
__device__ __forceinline__ float vox_bright(int32_t ax, int32_t ay, int32_t az) {   // classic per-voxel brightness
    return 0.88f + 0.12f * (float)(gme_hash3(ax, ay, az, 0x5EEDu) & 255u) / 255.f;
}
__device__ __forceinline__ bool is_tissue(int m) { return m == MAT_GRASS || m == MAT_MAT || m == MAT_LEAF || m == MAT_KELP || m == 15; }
__device__ __forceinline__ bool p_has_bio(const SParams& sp) { return sp.k.bio_cells != NULL; }
// M5: the biomap cell over absolute column (x, z) (its palette index = the pigment's hue bin, its
// cover), looked up at the 8 m slot centre like generation; 0 outside the regions
__device__ uint32_t bio_cell_at(const KParams& p, int32_t x, int32_t z) {
    const int32_t px = (x >> GME_FLORA_SLOT_SHIFT) * 8 + 4, pz = (z >> GME_FLORA_SLOT_SHIFT) * 8 + 4;
    for (int r = 0; r < p.bio_nreg; ++r) {
        const uint32_t i = (uint32_t)(px - p.bio_reg[r].x0) >> p.bio_shift;
        const uint32_t j = (uint32_t)(pz - p.bio_reg[r].z0) >> p.bio_shift;
        if (i < (uint32_t)p.bio_reg[r].w && j < (uint32_t)p.bio_reg[r].h)
            return __ldg(p.bio_cells + p.bio_reg[r].offset + j * (uint32_t)p.bio_reg[r].w + i);
    }
    return 0u;
}
// the biomap cell generation used for absolute column (x, z): its tile's slot (regions and the
// fallback table alike), else the region lookup (tile not published); and the slot's shape word
// (version 2: pigment amount, Q band width; 0 without a tile or without morphology)
struct SlotLook { uint32_t cell; int32_t sw; };
__device__ __forceinline__ SlotLook sm_slot_look(const KParams& p, int32_t x, int32_t z) {
    SlotLook L;
    const GmeColTile* t = tr_tile((x >> 5) - g_tc.cx, (z >> 5) - g_tc.cz);
    if (t) {
        const GmeSlot* s = &t->slot[((z & 31) >> 3) * 4 + ((x & 31) >> 3)];
        L.cell = __ldg(&s->cell);
        L.sw = __ldg(&s->theight);
    } else {
        L.cell = p.bio_cells ? bio_cell_at(p, x, z) : 0u;
        L.sw = 0;
    }
    return L;
}
__device__ __forceinline__ float cell_cover(uint32_t c) { return (float)BIOCELL_COVER(c) * (1.f / 255.f); }

// ---- water surface ---------------------------------------------------------------
// Gerstner normal at relative (x, z). Waves shorter than about two pixel
// footprints fade out; their slope variance moves into the glint roughness.
// smooth value noise in [-1, 1] on a lattice of size 1 (absolute lattice: the
// pattern does not move with the camera chunk)
__device__ __forceinline__ float vnoise(float x, float z, int32_t ox, int32_t oz) {
    const float fx = floorf(x), fz = floorf(z);
    const int32_t ix = (int32_t)fx + ox, iz = (int32_t)fz + oz;
    float u = x - fx, v = z - fz;
    u = u * u * (3.f - 2.f * u); v = v * v * (3.f - 2.f * v);
    const float a = (float)(gme_hash3(ix, 0, iz, 0x3A7Eu) & 1023u), b = (float)(gme_hash3(ix + 1, 0, iz, 0x3A7Eu) & 1023u);
    const float c = (float)(gme_hash3(ix, 0, iz + 1, 0x3A7Eu) & 1023u), d = (float)(gme_hash3(ix + 1, 0, iz + 1, 0x3A7Eu) & 1023u);
    return ((a + (b - a) * u) + ((c + (d - c) * u) - (a + (b - a) * u)) * v) * (2.f / 1023.f) - 1.f;
}
__device__ float3 water_normal(const SParams& sp, float x, float z, float foot, float* a2) {
    float gx = 0.f, gz = 0.f, gy = 0.f, lost = 0.f;
    // slow phase noise (two octaves, 64 m and 16 m) breaks the coherence of the 8
    // waves; anchored to absolute coordinates in 32768 m blocks (float precision)
    const float ax = x + 32.f * (float)(g_tc.cx & 1023), az = z + 32.f * (float)(g_tc.cz & 1023);
    const int32_t bx = g_tc.cx >> 10, bz = g_tc.cz >> 10;
    const float pn = 2.6f * vnoise(ax * (1.f / 64.f), az * (1.f / 64.f), bx * 512, bz * 512) +
                     1.3f * vnoise(ax * (1.f / 16.f), az * (1.f / 16.f), bx * 2048 + 7919, bz * 2048);
#pragma unroll 4
    for (int i = 0; i < WAVES; ++i) {
        const float f = sat(2.f - 0.63661977f * foot * sp.w.k[i]);
        if (f <= 0.f) { lost += sp.w.tail[i]; break; }   // sorted by k: every shorter one is faded out too
        float ph = sp.w.kx[i] * x + sp.w.kz[i] * z + sp.w.off[i] + pn * sp.w.pm[i];
        ph -= 6.2831853f * rintf(ph * 0.15915494f);       // into [-pi, pi]: the hardware sine is accurate there
        float sn, cs;
        __sincosf(ph, &sn, &cs);
        const float amp = sp.w.s[i] * f;
        gx += sp.w.dx[i] * amp * cs;
        gz += sp.w.dz[i] * amp * cs;
        gy += sp.w.q[i] * amp * sn;
        lost += 0.5f * sp.w.s[i] * sp.w.s[i] * (1.f - f * f);
    }
    if (a2) *a2 = sp.w.alpha0 * sp.w.alpha0 + lost;
    if (g_tc.far_on) {                                 // beyond the loaded radius the sea follows the curvature (M3)
        const float dx = x - g_tc.camx, dz = z - g_tc.camz;
        if (dx * dx + dz * dz > g_tc.drop_r2) { gx -= 2.f * g_tc.c2 * dx * (1.f - gy); gz -= 2.f * g_tc.c2 * dz * (1.f - gy); }
    }
    return norm3(f3(-gx, 1.f - gy, -gz));
}
// direction of a path after the water plane at distance tw (mode WM_*)
__device__ float3 water_second(const SParams& sp, float3 d, float3 nw, int mode) {
    if (!sp.refract) return d;                         // Low: straight on
    float t[3];
    const float dv[3] = { d.x, d.y, d.z };
    if (mode == WM_REFR) {
        if (dot3(nw, d) > -0.02f) nw = f3(0.f, 1.f, 0.f);
        const float n[3] = { nw.x, nw.y, nw.z };
        spec_refract(dv, n, 1.f / WATER_N, t);         // from air: always transmits
        return norm3(f3(t[0], fminf(t[1], -0.01f), t[2]));
    }
    float3 n = f3(-nw.x, -nw.y, -nw.z);                // from below: the normal faces the camera
    if (dot3(n, d) > -0.02f) n = f3(0.f, -1.f, 0.f);
    const float nn[3] = { n.x, n.y, n.z };
    if (mode == WM_UNDER_OUT && spec_refract(dv, nn, WATER_N, t) && t[1] > 0.f) return norm3(f3(t[0], t[1], t[2]));
    const float3 r = d - n * (2.f * dot3(d, n));        // total internal reflection
    return norm3(f3(r.x, fminf(r.y, -0.01f), r.z));
}
__device__ __forceinline__ bool path_split(int wm) { return wm == WM_REFR || wm == WM_UNDER_OUT || wm == WM_UNDER_TIR; }
__device__ __forceinline__ float water_foot(const SParams& sp, float3 d, float tw) { return tw * sp.pix / fmaxf(fabsf(d.y), 0.05f); }
__device__ float3 path_dir2(const SParams& sp, float3 d, float tw, int wm) {
    const float3 pw = sp.k.o + d * tw;
    return water_second(sp, d, water_normal(sp, pw.x, pw.z, water_foot(sp, d, tw), NULL), wm);
}
// hit position along a (possibly bent) path
__device__ __forceinline__ float3 path_pos(const SParams& sp, float3 d, float t, float tw, int wm, float3* d2) {
    if (path_split(wm) && tw >= 0.f && t >= tw) {
        *d2 = path_dir2(sp, d, tw, wm);
        return sp.k.o + d * tw + *d2 * (t - tw);
    }
    *d2 = d;
    return sp.k.o + d * t;
}

// ---- creatures (fauna_scene.h, biosphere spec 7.5) --------------------------------------------
// Spheres found through a hash grid of 32 m cells (= chunk columns). Nothing here touches the
// classic kernel or the walk: k_fauna_anim, k_fauna_primary, k_fauna_sun and k_fauna_classic are
// extra passes, launched only while creatures are near the camera.
// k_fauna_anim: once per frame, the moving centre of every sphere (fauna_motion at the frame's time)
// relative to the camera chunk, and its radius: the rays then test plain spheres
__global__ void __launch_bounds__(256) k_fauna_anim(FaunaDev F, float4* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= F.n) return;
    const FaunaSphere s = F.sph[i];
    float ox, oy, oz;
    fauna_motion(s.amp, s.attr, F.time, &ox, &oy, &oz);
    out[i] = make_float4((float)((s.cx - F.cx) * 32) + s.lx + ox, (F.classic ? s.yv : s.y) - (float)(32 * F.cy) + oy,
                         (float)((s.cz - F.cz) * 32) + s.lz + oz, s.r);
}
// The nearest creature along o + d t (d a unit vector), t in (t0, t1): its sphere (s; -1 none) and the
// hit's t. The walk visits the cells the ray crosses; a cell lists every sphere that can touch it, so
// the walk ends with the first cell that holds the nearest hit inside its own span. pix > 0 (primary
// rays): radians per pixel and tpre the path length before o; a sphere under 0.6 pixels in radius is
// drawn that large (small animals stay visible), one under 0.15 pixels not at all. any: the first hit
// found ends the walk, and a ray that starts inside a sphere is blocked (shadow rays).
struct FaunaHit { int s; float t; };
__device__ __noinline__ FaunaHit fauna_trace(const FaunaDev F, float3 o, float3 d, float t0, float t1, float pix, float tpre, bool any) {
    FaunaHit H;
    H.s = -1;
    H.t = t1;
    const float tmin = t0, yo = (float)(32 * F.cy);
    if (fabsf(d.y) > 1e-6f) {                           // the slab of heights that holds every sphere
        const float iy = 1.f / d.y, s0 = (F.ylo - yo - o.y) * iy, s1 = (F.yhi - yo - o.y) * iy;
        t0 = fmaxf(t0, fminf(s0, s1));
        t1 = fminf(t1, fmaxf(s0, s1));
    } else if (o.y < F.ylo - yo || o.y > F.yhi - yo) {
        return H;
    }
    if (!(t1 > t0)) return H;
    if (fabsf(d.x) < 1e-7f) d.x = copysignf(1e-7f, d.x);
    if (fabsf(d.z) < 1e-7f) d.z = copysignf(1e-7f, d.z);
    const float ix = 1.f / d.x, iz = 1.f / d.z;
    const bool px = d.x > 0.f, pz = d.z > 0.f;
    const float tn = t0 + 1e-3f;
    int rcx = (int)floorf((o.x + d.x * tn) * (1.f / 32.f)), rcz = (int)floorf((o.z + d.z * tn) * (1.f / 32.f));
    float t = t0, best = t1;
    for (int step = 0; step < 96 && t < best; ++step) {
        const float ex = ((float)((rcx + (px ? 1 : 0)) * 32) - o.x) * ix, ez = ((float)((rcz + (pz ? 1 : 0)) * 32) - o.z) * iz;
        const float te = fminf(fminf(ex, ez), best);
        const int32_t ci = fauna_cell_find(F.cells, F.mask, F.cx + rcx, F.cz + rcz);
        if (ci >= 0) {
            const FaunaCell c = F.cells[ci];
            const float ya = o.y + d.y * t, yb = o.y + d.y * te;
            if (fminf(ya, yb) <= c.yhi - yo && fmaxf(ya, yb) >= c.ylo - yo) {
                for (uint32_t k = 0; k < c.count; ++k) {
                    const uint32_t si = c.first + k;
                    const float4 q = __ldg(F.pos + si);              // the sphere of this frame
                    const float3 pc = f3(q.x - o.x, q.y - o.y, q.z - o.z);
                    const float bq = dot3(pc, d);
                    float r = q.w;
                    if (pix > 0.f) {
                        const float rp = pix * (tpre + fmaxf(bq, 0.f));   // one pixel at the sphere
                        if (r < 0.15f * rp) continue;
                        r = fmaxf(r, 0.6f * rp);
                    }
                    const float3 pp = pc - d * bq;
                    const float disc = r * r - dot3(pp, pp);
                    if (disc <= 0.f) continue;
                    const float sq = sqrtf(disc);
                    float ts = bq - sq;
                    if (ts <= tmin) {
                        if (!any || bq + sq <= tmin) continue;      // behind the origin (or the eye is inside it)
                        ts = tmin;
                    }
                    if (ts >= best) continue;
                    best = ts;
                    H.s = (int)si;
                    H.t = ts;
                    if (any) return H;
                }
            }
        }
        if (best <= te) break;                          // the nearest hit lies in the cells visited so far
        if (ex < ez) rcx += px ? 1 : -1; else rcz += pz ? 1 : -1;
        t = te;
    }
    return H;
}

// Classic mode: creatures as plainly tinted spheres over k_render's frame (k_render itself is untouched;
// this pass runs only while creatures are near the camera). A sphere nearer than the classic DDA's hit
// along the pixel's ray gets the classic light (sun x shadow + ambient), fog and water tint; hunters
// darker, a highlight on armour.
__global__ void __launch_bounds__(128) k_fauna_classic(KParams p, FaunaDev F, FaunaPal P) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    const float u = (2.f * (x + 0.5f) / p.w - 1.f) * p.th * p.aspect;
    const float v = (1.f - 2.f * (y + 0.5f) / p.h) * p.th;
    const float3 d = norm3(p.fw + p.rt * u + p.up * v);
    const FaunaHit fh = fauna_trace(F, p.o, d, 0.f, fminf(p.max_dist, FAUNA_FAR), 2.f * p.th / (float)p.h, 0.f, false);
    if (fh.s < 0) return;
    Hit h;
    if (trace(p, p.o, d, fh.t, h)) return;             // the world is nearer
    const FaunaSphere s = F.sph[fh.s];
    const float4 q = F.pos[fh.s];
    const float3 pos = p.o + d * fh.t;
    const float3 n = norm3(pos - f3(q.x, q.y, q.z));
    const uint32_t pi = FA_PAL(s.attr);
    float3 base = pi ? mix3(f3(0.46f, 0.43f, 0.40f), P.c[pi], 0.6f) : f3(0.46f, 0.43f, 0.40f);
    base = base * (((s.attr & FA_HUNTER) ? 0.6f : 1.f) * (0.88f + 0.12f * (float)FA_BRIGHT(s.attr) * (1.f / 63.f)));
    const float ndl = sat(dot3(n, p.sun));
    float shadow = 1.f;
    if (p.shadows && ndl > 0.f) {
        Hit hs;
        if (trace(p, pos + n * 0.02f, p.sun, 512.f, hs)) shadow = 0.f;
    }
    const float amb = 0.30f + 0.14f * n.y + 0.04f * n.x;
    const float3 skylight = mix3(f3(0.08f, 0.08f, 0.09f), v3(p.sky.zenith) * 0.9f + f3(0.22f, 0.22f, 0.20f), 0.9f);
    float3 col = base * (v3(p.sky.sun_color) * (p.sky.sun_intensity * ndl * shadow * 1.15f) + skylight * amb);
    if ((s.attr & FA_ARMORED) && ndl > 0.f)
        col = col + v3(p.sky.sun_color) * (p.sky.sun_intensity * shadow * 0.35f * powf(sat(dot3(n, norm3(p.sun - d))), 48.f));
    const float3 fog_c = sky(p, d, false);
    const float fog_k = 1.f / (p.max_dist * 0.55f);
    col = mix3(col, fog_c, sat(fmaxf(1.f - expf(-fh.t * fog_k * fh.t * fog_k), 0.f)));
    if (h.water_t >= 0.f) {                            // in the water: k_render's absorption and surface
        const float path = fh.t - h.water_t;
        const float a = 1.f - expf(-path * 0.07f);
        const float3 deep = f3(0.04f, 0.20f, 0.32f) * (0.55f + 0.45f * p.sun.y);
        float3 under = mix3(col, deep, a);
        if (h.water_t > 0.f) {
            const float c = fabsf(d.y);
            const float fres = 0.02f + 0.98f * powf(1.f - c, 5.f);
            const float3 refl = sky(p, f3(d.x, fabsf(d.y), d.z), true);
            under = mix3(under, refl, fres * 0.7f);
            const float fog = 1.f - expf(-h.water_t * fog_k * h.water_t * fog_k);
            under = mix3(under, fog_c, sat(fog));
        }
        col = under;
    }
    const int dx = x - p.w / 2, dy = y - p.h / 2;       // the crosshair stays on top
    if ((dy == 0 && abs(dx) < 9 && abs(dx) > 2) || (dx == 0 && abs(dy) < 9 && abs(dy) > 2))
        col = f3(1.f, 1.f, 1.f) - col;
    const uint32_t r = (uint32_t)(sqrtf(sat(col.x)) * 255.f + 0.5f);
    const uint32_t g = (uint32_t)(sqrtf(sat(col.y)) * 255.f + 0.5f);
    const uint32_t b = (uint32_t)(sqrtf(sat(col.z)) * 255.f + 0.5f);
    p.out[(size_t)y * p.w + x] = 0xFF000000u | (r << 16) | (g << 8) | b;
}

// ---- primary rays ---------------------------------------------------------------
// Cave-mouth test of a terrain hit (spec 3.4): an eligible corner column
// (T_i >= y_hit - 0.25) whose voxel at the hit's depth is air.
__device__ __noinline__ bool sm_cave(const THit& h, float3 pos) {
    const GmeColTile* t = g_tc.tiles + gme_page_index(g_tc.cx + h.rcx, 0, g_tc.cz + h.rcz, g_tc.n, 1);
    const int32_t hb = (int32_t)__ldg(&t->h_base);
    const float yabs = pos.y + (float)(32 * g_tc.cy);
    const int32_t ax0 = (g_tc.cx + h.rcx) * 32 - GME_TILE_APRON + h.i, az0 = (g_tc.cz + h.rcz) * 32 - GME_TILE_APRON + h.j;
    for (int k = 0; k < 4; ++k) {
        const int ci = k & 1, cj = k >> 1;
        const uint32_t cv = __ldg(&t->col[(h.j + cj) * GME_TILE_N + h.i + ci]);
        const int32_t hc = hb + (int32_t)(cv & 255u);
        const int32_t dv = (int32_t)((cv >> 8) ^ 0x80u) - 0x80;
        if ((float)(256 * (hc + 1) + 2 * dv) * (1.f / 256.f) < yabs - 0.25f) continue;
        if (sm_voxel(ax0 + ci, min(hc, (int32_t)floorf(yabs - (float)dv * (1.f / GME_DEV_UNIT))), az0 + cj, hc) == MAT_AIR)
            return true;
    }
    return false;
}
// one straight segment of a primary path: the column walk (fine surface before
// tl, LOD after, flora to tf, the far field), cave mouths continued in voxel
// mode until the ray is above the surface again
__device__ __noinline__ bool sm_walk(float3 o, float3 d, float t0, float tmax, float tl, float tf, THit& h) {
    for (;;) {
        if (!trace_smooth_t<false, FM_CLUMP>(o, d, t0, tmax, tl, tf, h)) return false;   // M5: clumped crowns
        if (h.kind != HK_TERRAIN || !sm_cave(h, o + d * h.t)) return true;
        float tx;                                      // cave mouth: the grid's voxels from here
        const int r = trace_voxels_call(o, d, h.t, tmax, false, true, h, tx);
        if (r == VX_HIT) return true;
        if (r == VX_END || !(tx > t0 + 0.01f)) return false;   // t0 grows every pass: the loop ends
        t0 = tx;
    }
}
// The second segment of a bent path from pw: hit fields of the walk, t made total
__device__ __noinline__ bool sm_walk2(float3 pw, float3 d2, float tw, float tmax, float tl, float tf, THit& h) {
    THit h2;
    h2.water_t = -1.f; h2.tice = -1.f; h2.kind = HK_NONE;
    if (!sm_walk(pw, d2, 0.f, tmax - tw, tl - tw, tf - tw, h2)) return false;
    const float wt = h.water_t;
    h = h2;
    h.t = tw + h2.t;
    h.water_t = wt;
    return true;
}
// Primary ray (spec 3.2, 3.6): a camera below the surface starts in voxel mode;
// the water plane y_w (one query per ray) bends the path. wm = WM_*, tw = the
// path's distance to the water plane (-1 none). tl: the fine surface ends here
// (per-pixel dithered lod_fine). Beyond the loaded radius the sea is curved;
// a ray below the geometric horizon that leaves the far field without a hit
// ends on the distant ground (HK_DIST).
__device__ bool sm_trace2(const SParams& sp, float3 o, float3 d, float tmax, float tl, float tf, THit& h, int& wm, float& tw) {
    h.water_t = -1.f;
    h.tice = -1.f;
    h.kind = HK_NONE;
    wm = WM_NONE;
    tw = -1.f;
    float t0 = 0.f, T;
    bool under = false;
    if (tr_surface(o.x, o.z, T)) {
        if (o.y < T) {                                 // underground (a cave): voxels first
            float tx;
            const int r = trace_voxels_call(o, d, 0.f, tmax, false, true, h, tx);
            if (r == VX_HIT) return true;
            if (r == VX_END) return false;
            t0 = tx;
        } else if (o.y < g_tc.yw && T < g_tc.yw && !g_tc.ice) {
            under = true;                              // the camera is in the water
        }
    }
    if (under) {
        wm = WM_UNDER;
        if (d.y > 0.f) {
            const float ts = (g_tc.yw - o.y) / d.y;
            if (ts < tmax) {
                if (sm_walk(o, d, t0, ts, tl, tf, h)) return true;  // under water, before the surface
                tw = ts;
                h.water_t = ts;
                const float3 pw = o + d * ts;
                const float3 nw = water_normal(sp, pw.x, pw.z, water_foot(sp, d, ts), NULL);
                float tt[3];
                const float dv[3] = { d.x, d.y, d.z }, nn[3] = { -nw.x, -nw.y, -nw.z };
                wm = (!sp.refract || (dot3(nw, d) > 0.02f && spec_refract(dv, nn, WATER_N, tt) && tt[1] > 0.f)) ? WM_UNDER_OUT
                                                                                                                : WM_UNDER_TIR;
                return sm_walk2(pw, water_second(sp, d, nw, wm), ts, tmax, tl, tf, h);
            }
        }
        return sm_walk(o, d, t0, tmax, tl, tf, h);
    }
    if (d.y < 0.f && o.y > g_tc.yw) {                  // the water plane
        float ts = (g_tc.yw - o.y) / d.y;
        const float hl2 = d.x * d.x + d.z * d.z;
        const float Cw = o.y - g_tc.yw - g_tc.c2 * g_tc.drop_r2;
        if (g_tc.far_on && hl2 * ts * ts > g_tc.drop_r2 && Cw > 0.f) {
            // beyond the loaded radius: y_w - c2 (r^2 - max_dist^2) with r = hl t (camera rays)
            const float disc = d.y * d.y - 4.f * g_tc.c2 * hl2 * Cw;
            ts = disc >= 0.f ? 2.f * Cw / (-d.y + sqrtf(disc)) : 3e38f;  // the near root; none: above the horizon
        }
        if (ts < tmax && tr_ground(o.x + d.x * ts, o.z + d.z * ts, ts < tl, T) && T < g_tc.yw) {
            if (g_tc.ice) {
                h.tice = ts;
                tmax = ts;
            } else {
                if (sm_walk(o, d, t0, ts, tl, tf, h)) return true;  // something above the water first
                tw = ts;
                h.water_t = ts;
                if (!sp.refract) {                                  // Low: straight on through the water
                    wm = WM_STRAIGHT;
                    return sm_walk(o, d, ts, tmax, tl, tf, h);
                }
                wm = WM_REFR;
                const float3 pw = o + d * ts;
                return sm_walk2(pw, path_dir2(sp, d, ts, WM_REFR), ts, tmax, tl, tf, h);
            }
        }
    }
    if (!sm_walk(o, d, t0, tmax, tl, tf, h)) {
        if (h.tice < 0.f) {
            if (h.water_t >= 0.f) { wm = WM_STRAIGHT; tw = h.water_t; return false; }   // water voxels (no far field)
            if (g_tc.far_on && d.y < sp.mu_h) {        // below the horizon beyond the far field: the distant ground
                h.t = tmax;
                h.kind = HK_DIST;
                return true;
            }
            return false;
        }
        h.t = h.tice;
        h.kind = HK_ICE;                               // an opaque ice sheet on the sea
        return true;
    }
    if (h.water_t >= 0.f && wm == WM_NONE) { wm = WM_STRAIGHT; tw = h.water_t; }
    return true;
}

// ---- G-buffer (24 B per pixel) ------------------------------------------------
// a: t (-1 = sky), water_t, shading normal, geometric normal (octahedral 2 x snorm16)
// b.x: material pack: m0 4 | m1 4 | w1 6 | palette 4 | brightness 6 | AO 8
// b.y: flags (GF_*) | dev q7 << 8 | water mode << 24
struct GBuf { float4* a; uint2* b; };

__device__ __forceinline__ uint32_t oct_pack(float3 n) {
    const float s = 1.f / (fabsf(n.x) + fabsf(n.y) + fabsf(n.z));
    float px = n.x * s, pz = n.z * s;
    if (n.y < 0.f) {
        const float qx = (1.f - fabsf(pz)) * copysignf(1.f, px), qz = (1.f - fabsf(px)) * copysignf(1.f, pz);
        px = qx; pz = qz;
    }
    const int32_t ix = __float2int_rn(fminf(fmaxf(px, -1.f), 1.f) * 32767.f);
    const int32_t iz = __float2int_rn(fminf(fmaxf(pz, -1.f), 1.f) * 32767.f);
    return ((uint32_t)ix & 0xFFFFu) | ((uint32_t)iz << 16);
}
__device__ __forceinline__ float3 oct_unpack(uint32_t v) {
    const float px = (float)(int16_t)(v & 0xFFFFu) * (1.f / 32767.f), pz = (float)(int16_t)(v >> 16) * (1.f / 32767.f);
    float3 n = f3(px, 1.f - fabsf(px) - fabsf(pz), pz);
    if (n.y < 0.f) {
        const float qx = (1.f - fabsf(pz)) * copysignf(1.f, px), qz = (1.f - fabsf(px)) * copysignf(1.f, pz);
        n.x = qx; n.z = qz;
    }
    return norm3(n);
}
__device__ __forceinline__ uint32_t mat_pack(int m0, int m1, float w1, int pal, float bright, float ao) {
    return ((uint32_t)m0 & 15u) | (((uint32_t)m1 & 15u) << 4) | ((uint32_t)min(63, __float2int_rn(sat(w1) * 63.f)) << 8) |
           (((uint32_t)pal & 15u) << 14) | ((uint32_t)min(63, max(0, __float2int_rn((bright - 0.88f) * (63.f / 0.12f)))) << 18) |
           ((uint32_t)min(255, __float2int_rn(sat(ao) * 255.f)) << 24);
}

// Hit record (k_primary -> k_surface), stored in place in the G-buffer arrays:
// a = t (-1 = none), water_t, u, v (terrain, LOD) or voxel x, z; b.x = kind | axis << 4
// | mat << 6 (voxel) or kind | rcx << 4 | rcz << 14 (10-bit each), water mode
// << 24; b.y = i | j << 8 (terrain, LOD cell, kelp; i = block for flora) or voxel y.
// Far-field and distant-ground hits need only t.
__device__ __forceinline__ void hit_pack(const THit& h, bool hit, float4& a, uint2& b) {
    a = make_float4(hit ? h.t : -1.f, h.water_t, 0.f, 0.f);
    b = make_uint2(0u, 0u);
    if (!hit) return;
    if (h.kind == HK_VOXEL) {
        b.x = (uint32_t)HK_VOXEL | ((uint32_t)h.axis << 4) | ((uint32_t)h.mat << 6);
        b.y = (uint32_t)h.vy;
        a.z = (float)h.vx; a.w = (float)h.vz;
    } else if (h.kind >= HK_FAR) {
        b.x = (uint32_t)h.kind;
    } else {
        b.x = (uint32_t)h.kind | (((uint32_t)h.rcx & 1023u) << 4) | (((uint32_t)h.rcz & 1023u) << 14);
        b.y = (uint32_t)h.i | ((uint32_t)h.j << 8);
        a.z = h.u; a.w = h.v;
    }
}
__device__ __forceinline__ void hit_unpack(float4 a, uint2 b, THit& h) {
    h.t = a.x; h.water_t = a.y;
    h.kind = (int)(b.x & 15u);
    if (h.kind == HK_VOXEL) {
        h.axis = (int)((b.x >> 4) & 3u); h.mat = (int)((b.x >> 6) & 255u);
        h.vx = (int)a.z; h.vy = (int)b.y; h.vz = (int)a.w;
    } else {
        h.rcx = (int)(((b.x >> 4) & 1023u) ^ 512u) - 512;
        h.rcz = (int)(((b.x >> 14) & 1023u) ^ 512u) - 512;
        h.i = (int)(b.y & 255u); h.j = (int)((b.y >> 8) & 255u);
        h.u = a.z; h.v = a.w;
    }
}

// k_primary: the column tracer (traversal only) -> hit record per pixel. (128, 6) as spec 3.1 (80 registers):
// measured on rig B against (128, 8) with 64 registers: -6 .. -9.5 % on near views and the streaming flight,
// +2 .. +5 % on the far / peak views; the near ground views are the common case
__global__ void __launch_bounds__(128, 6) k_primary(const __grid_constant__ SParams sp, GBuf g) {
    const KParams& p = sp.k;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    THit h;
    int wm;
    float tw;
    const float3 d = sp_dir(sp, x, y);
    // the fine / LOD and flora switches (dithered per pixel); the far field's edge
    const float tmax = g_tc.far_on ? g_tc.r2 * rsqrtf(fmaxf(d.x * d.x + d.z * d.z, 1e-8f)) : p.max_dist;
    const bool hit = sm_trace2(sp, p.o, d, tmax, lod_t(x, y), flora_t(x, y), h, wm, tw);
    float4 a;
    uint2 b;
    hit_pack(h, hit, a, b);
    a.y = wm == WM_UNDER ? 0.f : tw;
    b.x |= (uint32_t)wm << 24;
    const size_t i = (size_t)y * p.w + x;
    g.a[i] = a;
    g.b[i] = b;
}

// k_fauna_primary: creatures on the primary paths (after k_primary; only while creatures are near the
// camera). The pixel's path, straight or bent at the water plane as k_primary took it, against the
// sphere grid up to the walk's hit (FAUNA_FAR at most); a nearer sphere replaces the hit record: kind
// HK_FAUNA, b.y = the sphere. A sphere in front of the water plane takes the water off the path.
__global__ void __launch_bounds__(128) k_fauna_primary(const __grid_constant__ SParams sp, GBuf g) {
    const KParams& p = sp.k;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    const size_t i = (size_t)y * p.w + x;
    float4 a = g.a[i];
    int wm = (int)((g.b[i].x >> 24) & 7u);
    const float thit = a.x >= 0.f ? a.x : 3.0e38f, tw = a.y;
    const bool split = path_split(wm) && tw >= 0.f;
    const float3 d = sp_dir(sp, x, y);
    FaunaHit fh = fauna_trace(sp.fauna, p.o, d, 0.f, fminf(fminf(thit, split ? tw : 3.0e38f), FAUNA_FAR), sp.pix, 0.f, false);
    float tf = fh.t;
    if (fh.s >= 0) {
        if ((wm == WM_REFR || wm == WM_STRAIGHT) && (tw < 0.f || tf < tw)) { wm = WM_NONE; a.y = -1.f; }   // in front of the water
        else if (wm == WM_UNDER_OUT || wm == WM_UNDER_TIR) { wm = WM_UNDER; a.y = 0.f; }                     // in the water, before the surface
    } else if (split && thit > tw && tw < FAUNA_FAR) {  // the path beyond the water plane
        fh = fauna_trace(sp.fauna, p.o + d * tw, path_dir2(sp, d, tw, wm), 0.f, fminf(thit, FAUNA_FAR) - tw, sp.pix, tw, false);
        tf = tw + fh.t;
    }
    if (fh.s < 0) return;
    a.x = tf;
    a.z = 0.f;
    a.w = 0.f;
    g.a[i] = a;
    g.b[i] = make_uint2((uint32_t)HK_FAUNA | ((uint32_t)wm << 24), (uint32_t)fh.s);
}

// ---- surfaces ----------------------------------------------------------------------
struct SurfInfo {
    float3 n, ng;               // shading and geometric normal
    int    m0, m1, pal;         // two dominant materials, palette index (M5: the pigment's hue bin)
    float  w1, bright, ao, dev;
    float  wb;                  // far model: share of beach ground blended into m0 (LOD / far hits; else 0)
    float  cover;               // M5: the cell's cover 0..1 (grass: mix(soil, leaf, 0.8 cover))
    float  tsun;                // M5 crowns: the own crown's transmittance toward the sun, e^-(sigma chord); else 1
};
// SurfInfo.pal beyond its 4 palette bits (no extra registers in the kernels): version 2: bits 4-7 the cell's
// pigment level (0 = the version 1 look), bits 8-11 its Q band width bin; creatures: bit 12 hunter, bit 13 armoured
__device__ __forceinline__ int pal_look(uint32_t cell, int32_t sw) {
    return (int)BIOCELL_PAL(cell) | (GME_SHAPE_IS_V2(sw) ? (SPEC_PIG_LEVEL(GME_SHAPE_PIG(sw)) << 4) | ((int)GME_SHAPE_QWIDTH(sw) << 8) : 0);
}

// heightfield AO (spec 4.6): horizon angles above the tangent plane from tile T
// samples at 2 m (and 8 m) in 4 directions
__device__ float hf_ao(const SParams& sp, float3 p, float3 n) {
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float dx = (k == 0) ? 1.f : (k == 2 ? -1.f : 0.f), dz = (k == 1) ? 1.f : (k == 3 ? -1.f : 0.f);
        const float st = -(n.x * dx + n.z * dz) / fmaxf(n.y, 0.05f);      // tangent-plane slope along the direction
        float hm = -1e9f, T;
        if (tr_surface(p.x + 2.f * dx, p.z + 2.f * dz, T)) hm = fmaxf(hm, (T - p.y) * 0.5f);
        if (sp.ao_far && tr_surface(p.x + 8.f * dx, p.z + 8.f * dz, T)) hm = fmaxf(hm, (T - p.y) * 0.125f);
        const float e = fmaxf(hm - st, 0.f);
        sum += hm > -1e8f ? 1.f - e * rsqrtf(1.f + e * e) : 1.f;
    }
    return 0.25f * sum;
}

__device__ void sm_terrain_mat(const SParams& sp, const GmeColTile* t, int rcx, int rcz, int i, int j, float u, float v, float3 pos,
                               SurfInfo& si, float fp);
// Terrain hit: shading normal (corner gradients by central differences, blended
// bilinearly), geometric normal of the patch, and the depth-anchored materials
// (spec 3.3, 3.4; sm_terrain_mat). fp: the pixel footprint at the hit (m; large
// for secondary rays: the cheap material path).
__device__ void sm_terrain(const SParams& sp, const THit& h, float3 pos, SurfInfo& si, bool want_ao = true, float fp = 1e3f) {
    const GmeColTile* t = g_tc.tiles + gme_page_index(g_tc.cx + h.rcx, 0, g_tc.cz + h.rcz, g_tc.n, 1);
    const int i = h.i, j = h.j;
    const float u = h.u, v = h.v;
    int32_t q[4][4];                                  // 256*(h-h_base) + 2*dev, rows j-1..j+2, cols i-1..i+2
    for (int r = 0; r < 4; ++r)
        for (int s = 0; s < 4; ++s)
            q[r][s] = ((r == 0 || r == 3) && (s == 0 || s == 3)) ? 0
                    : tr_col_q8(__ldg(&t->col[(j - 1 + r) * GME_TILE_N + i - 1 + s]));
    const float g = 1.f / 512.f;                       // q8 central difference -> slope
    const float gx00 = (float)(q[1][2] - q[1][0]) * g, gz00 = (float)(q[2][1] - q[0][1]) * g;
    const float gx10 = (float)(q[1][3] - q[1][1]) * g, gz10 = (float)(q[2][2] - q[0][2]) * g;
    const float gx01 = (float)(q[2][2] - q[2][0]) * g, gz01 = (float)(q[3][1] - q[1][1]) * g;
    const float gx11 = (float)(q[2][3] - q[2][1]) * g, gz11 = (float)(q[3][2] - q[1][2]) * g;
    const float gx = (gx00 + (gx10 - gx00) * u) + ((gx01 + (gx11 - gx01) * u) - (gx00 + (gx10 - gx00) * u)) * v;
    const float gz = (gz00 + (gz10 - gz00) * u) + ((gz01 + (gz11 - gz01) * u) - (gz00 + (gz10 - gz00) * u)) * v;
    si.n = norm3(f3(-gx, 1.f, -gz));
    const float bb = (float)(q[1][2] - q[1][1]) * (1.f / 256.f), cc = (float)(q[2][1] - q[1][1]) * (1.f / 256.f);
    const float kk = (float)(q[1][1] - q[1][2] - q[2][1] + q[2][2]) * (1.f / 256.f);
    si.ng = norm3(f3(-(bb + kk * v), 1.f, -(cc + kk * u)));
    sm_terrain_mat(sp, t, h.rcx, h.rcz, i, j, u, v, pos, si, fp);
    si.ao = want_ao ? hf_ao(sp, pos, si.n) : 1.f;
}

// ---- ground materials: border shapes and the coverage of life on the ground --------------------
// A mat on sand (MAT_MAT) and a mat on soil (MATX_SOILMAT: the deeper seabed) differ only in the
// ground they cover (mat_rho); the id fits the G-buffer's 4 bits (MAT_COUNT = 15).
#define MATX_SOILMAT 15
#define LEAFY 0.85f                 // leaf share of a fully grassed surface (the rest: its soil between the blades)
// Visual coverage of a cell's ground tissue: 1 - e^(-6 cover). The voxel truth puts a grass / mat voxel on a
// column with probability cover / 256, and the evolved islands' cells have cover ~0.1: drawn literally (and with
// the cover again in the albedo) that read as bare soil and as dark 1 m squares. Tufts and mats spread beyond
// their voxel's square metre: the coverage field draws them as organic patches of this mean coverage.
__device__ __forceinline__ float vis_cover(float c) { return 1.f - __expf(-6.f * c); }
// patch noise q in [0, 1] (world anchored, 2.3 m + 0.85 m value noise) and the quantiles of its distribution at
// v = k / 16 (measured on 2M samples), so a threshold at quantile v covers the share v of the ground
__constant__ float c_patch_q[17] = { 0.0100f, 0.2589f, 0.3144f, 0.3544f, 0.3882f, 0.4185f, 0.4467f, 0.4739f, 0.5005f,
                                     0.5271f, 0.5544f, 0.5826f, 0.6129f, 0.6466f, 0.6867f, 0.7421f, 0.9890f };
__device__ float patch_q(float3 pos) {
    const float axw = pos.x + 32.f * (float)(g_tc.cx & 1023), azw = pos.z + 32.f * (float)(g_tc.cz & 1023);
    const int32_t bx = g_tc.cx >> 10, bz = g_tc.cz >> 10;
    return 0.5f + 0.5f * (0.62f * vnoise(axw * (1.f / 2.3f), azw * (1.f / 2.3f), bx * 14247 + 101, bz * 14247 + 7) +
                          0.38f * vnoise(axw * (1.f / 0.85f) + 17.3f, azw * (1.f / 0.85f), bx * 38551 + 5, bz * 38551 + 77));
}
// coverage at the point of a patch field of mean v: 1 inside a patch (q below the v-quantile), soft edges; the mean
// where patches are sub-pixel (fp: pixel footprint, m)
__device__ float patch_cover(float3 pos, float v, float fp) {
    const float fade = smooth01(0.35f, 1.4f, fp);
    if (fade >= 1.f || v <= 0.f) return v;
    const float x = sat(v) * 16.f;
    const int k = min((int)x, 15);
    const float th = c_patch_q[k] + (c_patch_q[k + 1] - c_patch_q[k]) * (x - (float)k);
    const float e = 0.07f + 0.05f * fp;
    const float g = smooth01(th + e, th - e, patch_q(pos));
    return g + (v - g) * fade;
}
// cubic B-spline weights of the 4 taps around a fraction t (C2; sum 1)
__device__ __forceinline__ void bspl_w(float t, float w[4]) {
    const float t2 = t * t, t3 = t2 * t, s = 1.f - t;
    w[0] = s * s * s * (1.f / 6.f);
    w[1] = (3.f * t3 - 6.f * t2 + 4.f) * (1.f / 6.f);
    w[2] = (-3.f * t3 + 3.f * t2 + 3.f * t + 1.f) * (1.f / 6.f);
    w[3] = t3 * (1.f / 6.f);
}

// The depth-anchored materials of dual cell (i, j) (tile array index of its lower corner) at patch point (u, v),
// height pos.y (spec 3.4). A column is eligible if T_i >= y_hit - 0.25 (drops low columns under a point high on a
// cliff); its voxel is taken at the hit's depth below that column's surface, so cliffs show the grid's real layering.
// Up close (fp < 1.5 m) the border between two ground materials is the 0.5 contour of their cubic B-spline
// indicators over the 4 x 4 columns around the cell, moved by a 1.3 m noise (organic, not the voxels' 1 m
// staircase that the corner weights w^2 drew), and the column nearest the hit keeps its own material within ~0.35 m
// of its centre (single columns stay round spots instead of vanishing); farther, and for secondary rays, the 4
// corners weighted w^2. Life on the ground (grass, mats) belongs to its ground material's area: where the cell of
// the nearest column has ground cover / shrubs / trees (on soil) or mats / kelp (at most 2 m above the sea), that
// ground becomes the tissue with the coverage of the patch field (vis_cover) and of the tissue voxels themselves.
__device__ void sm_terrain_mat(const SParams& sp, const GmeColTile* t, int rcx, int rcz, int i, int j, float u, float v, float3 pos,
                               SurfInfo& si, float fp) {
    const GmePlanet& P = g_tc.planet;
    const int32_t hb = (int32_t)__ldg(&t->h_base);
    const float yabs = pos.y + (float)(32 * g_tc.cy);
    const int32_t ax0 = (g_tc.cx + rcx) * 32 - GME_TILE_APRON + i, az0 = (g_tc.cz + rcz) * 32 - GME_TILE_APRON + j;
    const bool fine = fp < 1.5f;
    const int ru = u >= 0.5f ? 1 : 0, rv = v >= 0.5f ? 1 : 0;   // the column nearest the hit
    float bu[4], bw[4];
    if (fine) { bspl_w(u, bu); bspl_w(v, bw); }
    int ids[6];
    float ws[6];
    int nid = 0, mnear = -1, cnear = -1;
    float wsum = 0.f, bsum = 0.f, dsum = 0.f, wgr = 0.f, wmt = 0.f;
    const int r0 = fine ? -1 : 0, r1 = fine ? 2 : 1;
    for (int r = r0; r <= r1; ++r)
        for (int s = r0; s <= r1; ++s) {
            const uint32_t cv = __ldg(&t->col[(j + r) * GME_TILE_N + i + s]);
            const int32_t hc = hb + (int32_t)(cv & 255u);
            const int32_t dv = (int32_t)((cv >> 8) ^ 0x80u) - 0x80;
            float w;
            if (fine) w = bu[s + 1] * bw[r + 1];
            else { const float wb = (s ? u : 1.f - u) * (r ? v : 1.f - v); w = wb * wb + 1e-4f; }
            if ((unsigned)r <= 1u && (unsigned)s <= 1u) dsum += (s ? u : 1.f - u) * (r ? v : 1.f - v) * (float)dv * (1.f / GME_DEV_UNIT);
            const float Tc = (float)(256 * (hc + 1) + 2 * dv) * (1.f / 256.f);
            if (Tc < yabs - 0.25f) continue;
            const int32_t iy = min(hc, (int32_t)floorf(yabs - (float)dv * (1.f / GME_DEV_UNIT)));
            const int m = sm_voxel(ax0 + s, iy, az0 + r, hc);
            if (m == MAT_AIR) continue;
            // the class of the border: the ground the tissue sits on
            const int c = m == MAT_GRASS ? P.surface_mat : (m == MAT_MAT ? gme_ground_mat(hc, &P) : m);
            if (m == MAT_GRASS) wgr += w;
            if (m == MAT_MAT) wmt += w;
            if (s == ru && r == rv) { mnear = m; cnear = c; }
            wsum += w;
            bsum += w * vox_bright(ax0 + s, iy, az0 + r);
            int k = 0;
            while (k < nid && ids[k] != c) ++k;
            if (k == nid && nid < 6) { ids[nid] = c; ws[nid] = 0.f; ++nid; }
            if (k < nid) ws[k] += w;
        }
    // the nearest column's own material within ~0.35 m of its centre
    const float du = u - (float)ru, dvv = v - (float)rv, dn = sqrtf(du * du + dvv * dvv);
    const float guard = fine ? 0.78f * smooth01(0.46f, 0.22f, dn) : 0.f;
    if (fine && cnear >= 0)
        for (int k = 0; k < nid; ++k) if (ids[k] == cnear) ws[k] = fmaxf(ws[k], guard * wsum);
    int a0 = 0;
    for (int k = 1; k < nid; ++k) if (ws[k] > ws[a0]) a0 = k;
    int a1 = a0;
    for (int k = 0; k < nid; ++k) if (k != a0 && (a1 == a0 || ws[k] > ws[a1])) a1 = k;
    int m0 = nid ? ids[a0] : MAT_STONE, m1 = nid ? ids[a1] : MAT_STONE;
    float w1 = 0.f;
    if (nid && a1 != a0) {
        if (fine) {                                     // the 0.5 contour, moved by a 1.3 m noise (+-0.3 m), soft edges
            const float axw = pos.x + 32.f * (float)(g_tc.cx & 1023), azw = pos.z + 32.f * (float)(g_tc.cz & 1023);
            const float nz = vnoise(axw * (1.f / 1.3f), azw * (1.f / 1.3f), (g_tc.cx >> 10) * 25206 + 31, (g_tc.cz >> 10) * 25206 + 3);
            const float sd = (ws[a1] - ws[a0]) / (ws[a0] + ws[a1]);
            const float e = 0.06f + 0.12f * fp;
            w1 = smooth01(-e, e, sd + 0.24f * nz * (1.f - smooth01(0.4f, 1.2f, fp)));
        } else {
            w1 = ws[a1] / (ws[a0] + ws[a1]);
        }
    }
    si.bright = wsum > 0.f ? bsum / wsum : 1.f;
    si.dev = dsum;
    si.pal = 0;
    si.cover = 0.f;
    // life on the ground: the cells (8 m slots) at two points warped by independent noises (+-6 m at 13 m and +-2 m at
    // 4 m scale), so the biomap's 16 m cells (a checkerboard of covers on the evolved islands) meet along organic
    // borders with a transition of half the coverage; the hue of the first
    const float axw = pos.x + 32.f * (float)(g_tc.cx & 1023), azw = pos.z + 32.f * (float)(g_tc.cz & 1023);
    const int32_t bwx = g_tc.cx >> 10, bwz = g_tc.cz >> 10;
    SlotLook looks[2];                                  // (version 2: each with its slot's shape word, for pal_look)
    uint32_t cells[2];
    for (int q = 0; q < 2; ++q) {
        const float o = 37.1f * (float)q;
        const float wpx = 6.f * vnoise(axw * (1.f / 13.f) + o, azw * (1.f / 13.f), bwx * 2521 + 13, bwz * 2521 + 71) +
                          2.f * vnoise(axw * 0.25f, azw * 0.25f + o, bwx * 8192 + 5, bwz * 8192 + 3);
        const float wpz = 6.f * vnoise(axw * (1.f / 13.f) + 41.3f, azw * (1.f / 13.f) + 7.9f + o, bwx * 2521 + 59, bwz * 2521 + 23) +
                          2.f * vnoise(axw * 0.25f + 11.7f + o, azw * 0.25f, bwx * 8192 + 9, bwz * 8192 + 1);
        looks[q] = sm_slot_look(sp.k, g_tc.cx * 32 + (int32_t)floorf(pos.x + wpx), g_tc.cz * 32 + (int32_t)floorf(pos.z + wpz));
        cells[q] = looks[q].cell;
    }
    // where each tissue grows, gated on the drawn (smooth) height of the hit, not on the nearest column's integer h
    // (its 1 m staircase drew saw-tooth borders): ground cover above the waterline; mats and the kelp bed from under
    // the water up to ~2 m above it (the voxels' h <= sea + 2, T ~ h + 1)
    const float ywl = (float)(P.sea_level + 1);
    const float gate_gr = smooth01(ywl, ywl + 0.3f, yabs), gate_mt = smooth01(ywl + 2.6f, ywl + 2.f, yabs);
    int tis = 0;
    float cvs = 0.f;
    for (int q = 0; q < 2; ++q) {                       // the tissue (either lookup) and its mean coverage target
        const uint32_t k = BIOCELL_KIND(cells[q]);
        const int tq = (k >= BIO_KIND_COVER && k <= BIO_KIND_TREE && gate_gr > 0.f) ? MAT_GRASS
                     : (((k == BIO_KIND_MAT || k == BIO_KIND_KELP) && gate_mt > 0.f) ? MAT_MAT : 0);
        if (tq && !tis) tis = tq;
        if (tq && tq == tis) cvs += 0.5f * vis_cover(cell_cover(cells[q]));
    }
    if (tis) {
        // the tissue grows on: grass on the soil; mats on any ground (sand on the beach, soil on the deeper bed)
        const int g = P.surface_mat;
        const bool e0 = tis == MAT_GRASS ? m0 == g : (m0 == P.beach_mat || m0 == g || m0 == MAT_REGOLITH);
        const bool e1 = tis == MAT_GRASS ? m1 == g : (m1 == P.beach_mat || m1 == g || m1 == MAT_REGOLITH);
        if (e0 || (e1 && w1 > 0.f)) {
            // its voxels' share of that ground (indicator, with the nearest column's spot), and the patch field
            float wg = 0.f;
            for (int k = 0; k < nid; ++k) {
                const bool el = tis == MAT_GRASS ? ids[k] == g : (ids[k] == P.beach_mat || ids[k] == g || ids[k] == MAT_REGOLITH);
                if (el) wg += ws[k];
            }
            float ind = wg > 0.f ? (tis == MAT_GRASS ? wgr : wmt) / wg : 0.f;
            if (mnear == tis) ind = fmaxf(ind, guard);
            if (fine) ind = smooth01(0.3f, 0.6f, ind);
            // the tissue voxels show as tufts / spots of 0.6 coverage on top of the patch field (whole cells of
            // voxels would otherwise draw the cells' straight borders)
            si.cover = (tis == MAT_GRASS ? gate_gr : gate_mt) * fmaxf(0.6f * ind, fine ? patch_cover(pos, cvs, fp) : cvs);
            const int tm0 = tis == MAT_MAT && m0 != P.beach_mat ? MATX_SOILMAT : tis;
            const int tm1 = tis == MAT_MAT && m1 != P.beach_mat ? MATX_SOILMAT : tis;
            if (e0) m0 = tm0;
            if (e1) m1 = tm1;
            { const SlotLook lk = looks[BIOCELL_KIND(cells[0]) ? 0 : 1]; si.pal = pal_look(lk.cell, lk.sw); }
        }
    }
    if (si.pal == 0 && (is_tissue(m0) || is_tissue(m1))) si.pal = pal_look(looks[0].cell, looks[0].sw);
    si.m0 = m0;
    si.m1 = m1;
    si.w1 = w1;
}

// ---- M3: LOD and far-field surfaces ---------------------------------------------------
// lod4 lattice value (relative metres) at lattice point (I, J) of tile (rcx, rcz),
// I, J in -1..9 (reaching into the neighbouring tiles); NAN if that tile is missing
__device__ __forceinline__ float lod_lat(const GmeColTile* t, int rcx, int rcz, int I, int J) {
    if (I < 0 || I > 8 || J < 0 || J > 8) {
        if (I < 0) { --rcx; I += 8; } else if (I > 8) { ++rcx; I -= 8; }
        if (J < 0) { --rcz; J += 8; } else if (J > 8) { ++rcz; J -= 8; }
        t = tr_tile(rcx, rcz);
        if (!t) return NAN;
    }
    return (float)__ldg(&t->lod4[J * GME_LOD_N + I]) * (1.f / 16.f) + g_tc.ybase_m;
}
// central difference of three lattice values (spacing s); one-sided if a neighbour is missing
__device__ __forceinline__ float cdiff(float m, float c, float p, float s) {
    if (isnan(m)) return (p - c) / s;
    if (isnan(p)) return (c - m) / s;
    return (p - m) / (2.f * s);
}
// smooth gradient of far-field level l at relative (x, z): corner gradients by
// central differences, blended bilinearly (C0 across cells, like the fine surface)
__device__ void far_grad(int l, float x, float z, float& gx, float& gz) {
    const FarLevel& L = g_tc.far[l];
    const float u = (x - L.ox) * L.iS, v = (z - L.oz) * L.iS;
    const int fi = (int)floorf(u), fj = (int)floorf(v);
    const float a = u - (float)fi, b = v - (float)fj;
    float q[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int s = 0; s < 4; ++s) q[r][s] = ((r == 0 || r == 3) && (s == 0 || s == 3)) ? 0.f : far_h(L, fi - 1 + s, fj - 1 + r);
    const float g = 0.5f * L.iS;
    const float gx00 = (q[1][2] - q[1][0]) * g, gz00 = (q[2][1] - q[0][1]) * g;
    const float gx10 = (q[1][3] - q[1][1]) * g, gz10 = (q[2][2] - q[0][2]) * g;
    const float gx01 = (q[2][2] - q[2][0]) * g, gz01 = (q[3][1] - q[1][1]) * g;
    const float gx11 = (q[2][3] - q[2][1]) * g, gz11 = (q[3][2] - q[1][2]) * g;
    gx = (gx00 + (gx10 - gx00) * a) + ((gx01 + (gx11 - gx01) * a) - (gx00 + (gx10 - gx00) * a)) * b;
    gz = (gz00 + (gz10 - gz00) * a) + ((gz01 + (gz11 - gz01) * a) - (gz00 + (gz10 - gz00) * a)) * b;
}
// surface of level l alone (L1 corners morphed toward L2 as in the march), no drop
__device__ float far_level_h(int l, float x, float z) {
    const FarLevel& L = g_tc.far[l];
    const float u = (x - L.ox) * L.iS, v = (z - L.oz) * L.iS;
    const int fi = (int)floorf(u), fj = (int)floorf(v);
    FarCell c;
    far_cell(l, fi, fj, c);
    return bil4(c, u - (float)fi, v - (float)fj);
}
// heightfield AO of level l (the near hf_ao at the level's scale: horizons at S and 2S)
__device__ float far_ao(int l, float x, float z, float3 n) {
    const float S = g_tc.far[l].S, h0 = far_level_h(l, x, z);
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float dx = (k == 0) ? 1.f : (k == 2 ? -1.f : 0.f), dz = (k == 1) ? 1.f : (k == 3 ? -1.f : 0.f);
        const float st = -(n.x * dx + n.z * dz) / fmaxf(n.y, 0.05f);
        const float hm = fmaxf((far_level_h(l, x + S * dx, z + S * dz) - h0) / S, (far_level_h(l, x + 2.f * S * dx, z + 2.f * S * dz) - h0) / (2.f * S));
        const float e = fmaxf(hm - st, 0.f);
        sum += 1.f - e * rsqrtf(1.f + e * e);
    }
    return 0.25f * sum;
}
// The far model of a surface's materials from a level's samples around (x, z):
// the share of beach ground, the coverage of grass and of mat, and the
// palette of the corner with the most tissue (bilinear weights).
struct FarMat { float beach, grass, mat, best; int pal; };
__device__ void far_mat_level(int l, float x, float z, FarMat& f) {
    const FarLevel& L = g_tc.far[l];
    const float u = (x - L.ox) * L.iS, v = (z - L.oz) * L.iS;
    const int fi = (int)floorf(u), fj = (int)floorf(v);
    const float a = u - (float)fi, b = v - (float)fj;
    f.beach = f.grass = f.mat = f.best = 0.f;
    f.pal = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t s = far_raw(L, fi + (k & 1), fj + (k >> 1));
        const float w = ((k & 1) ? a : 1.f - a) * ((k >> 1) ? b : 1.f - b);
        f.beach += w * (float)((s >> 16) & 15u) * (1.f / 15.f);
        const uint32_t tis = (s >> 24) & 3u;
        const float cov = w * (float)(s >> 26) * (1.f / 63.f);
        if (tis == 1u) f.grass += cov; else if (tis == 2u) f.mat += cov;
        if (tis && cov > f.best) { f.best = cov; f.pal = (int)((s >> 20) & 15u); }
    }
}
// The far material of level l (moving to L2 by w1): m0 = the land surface, the
// beach share as a third weight (si.wb, k_compose blends it into m0), m1 = the
// dominant tissue with its coverage. Continuous: no dither, nothing to resolve.
__device__ void far_material(const SParams& sp, int l, float w1, float x, float z, SurfInfo& si) {
    FarMat f;
    far_mat_level(l, x, z, f);
    if (w1 > 0.f) {
        FarMat g;
        far_mat_level(1, x, z, g);
        f.beach += (g.beach - f.beach) * w1;
        f.grass += (g.grass - f.grass) * w1;
        f.mat += (g.mat - f.mat) * w1;
        if (g.best * w1 > f.best * (1.f - w1)) f.pal = g.pal;
    }
    const GmePlanet& P = g_tc.planet;
    si.m0 = P.soil_depth > 0 ? P.surface_mat : MAT_REGOLITH;
    si.wb = f.beach;
    si.m1 = f.grass >= f.mat ? MAT_GRASS : MAT_MAT;
    si.w1 = vis_cover(fmaxf(f.grass, f.mat));          // the visual coverage, as the near field's patch field (mean)
    si.pal = f.pal;
    si.cover = 1.f;                                    // the covered part is all tissue
    si.bright = 0.94f;                                 // the mean of the classic per-voxel brightness
    si.dev = 0.f;
}

// LOD hit (4 m lattice of a published tile): smooth normals from the lattice
// (across tiles), the fine surface's depth-anchored materials and AO under the
// hit; near the ring the normal, materials and AO move to the far model with
// the geometry's morph weight.
__device__ void sm_lod(const SParams& sp, const THit& h, float3 pos, int px, int py, SurfInfo& si, bool want_ao = true, float fp = 1e3f) {
    const GmeColTile* t = g_tc.tiles + gme_page_index(g_tc.cx + h.rcx, 0, g_tc.cz + h.rcz, g_tc.n, 1);
    float gxc[4], gzc[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int I = h.i + (k & 1), J = h.j + (k >> 1);
        const float c = lod_lat(t, h.rcx, h.rcz, I, J);
        gxc[k] = cdiff(lod_lat(t, h.rcx, h.rcz, I - 1, J), c, lod_lat(t, h.rcx, h.rcz, I + 1, J), 4.f);
        gzc[k] = cdiff(lod_lat(t, h.rcx, h.rcz, I, J - 1), c, lod_lat(t, h.rcx, h.rcz, I, J + 1), 4.f);
    }
    const float u = h.u, v = h.v;
    float gx = (gxc[0] + (gxc[1] - gxc[0]) * u) + ((gxc[2] + (gxc[3] - gxc[2]) * u) - (gxc[0] + (gxc[1] - gxc[0]) * u)) * v;
    float gz = (gzc[0] + (gzc[1] - gzc[0]) * u) + ((gzc[2] + (gzc[3] - gzc[2]) * u) - (gzc[0] + (gzc[1] - gzc[0]) * u)) * v;
    const float w = g_tc.far_on ? lod_w(far_r(pos.x, pos.z)) : 0.f;
    if (w > 0.f) {
        float fx, fz;
        far_grad(0, pos.x, pos.z, fx, fz);
        gx += (fx - gx) * w;
        gz += (fz - gz) * w;
    }
    si.n = norm3(f3(-gx, 1.f, -gz));
    FarCell c;
    lod_cell(t, h.rcx, h.rcz, h.i, h.j, c);
    const float bb = (c.h10 - c.h00) * 0.25f, cc = (c.h01 - c.h00) * 0.25f, kk = (c.h00 - c.h10 - c.h01 + c.h11) * 0.25f;
    si.ng = norm3(f3(-(bb + kk * v), 1.f, -(cc + kk * u)));
    // the fine surface under the hit: its dual cell, height, materials and AO
    const float lx = pos.x - (float)(h.rcx * 32), lz = pos.z - (float)(h.rcz * 32);
    const int DX = min(max((int)floorf(lx - 0.5f), -1), 31), DZ = min(max((int)floorf(lz - 0.5f), -1), 31);
    const float uf = sat(lx - ((float)DX + 0.5f)), vf = sat(lz - ((float)DZ + 0.5f));
    const int k0 = (DZ + 2) * GME_TILE_N + DX + 2;
    const int32_t base = tr_base(t);
    const FarCell fc = { tr_T(t, k0, base), tr_T(t, k0 + 1, base), tr_T(t, k0 + GME_TILE_N, base), tr_T(t, k0 + GME_TILE_N + 1, base) };
    const float3 pf = f3(pos.x, bil4(fc, uf, vf), pos.z);
    if (w > 0.f && dith((float)px + 13.7f, (float)py + 5.9f) < w) {
        far_material(sp, 0, 0.f, pos.x, pos.z, si);
    } else {
        sm_terrain_mat(sp, t, h.rcx, h.rcz, DX + 2, DZ + 2, uf, vf, pf, si, fp);
        // beyond the analytic flora (spec 3.1, 3.5): the block's shrub / tree as a canopy tint, its crown's share of the slot
        const GmeSlot* sl = &t->slot[min(max((int)(lz * 0.125f), 0), 3) * 4 + min(max((int)(lx * 0.125f), 0), 3)];
        const int32_t trad = __ldg(&sl->trad);
        if (h.t > flora_t(px, py) && trad > 0) {
            si.m1 = MAT_LEAF;
            si.w1 = fminf(1.f, 0.0490874f * (float)(trad * trad + trad));   // pi (r^2 + r) / 64 m^2
            si.pal = pal_look(__ldg(&sl->cell), __ldg(&sl->theight));
        }
    }
    if (!want_ao) { si.ao = 1.f; return; }
    si.ao = hf_ao(sp, pf, si.n);
    if (w > 0.f) si.ao += (far_ao(0, pos.x, pos.z, si.n) - si.ao) * w;
}

// Far-field hit: smooth normal of the level (L1 moving to L2 in its band) plus
// the curvature drop's slope; the far material model; the level's AO.
__device__ void sm_far(const SParams& sp, float3 pos, int px, int py, SurfInfo& si, bool want_ao = true) {
    const float dx = pos.x - g_tc.camx, dz = pos.z - g_tc.camz, r2 = dx * dx + dz * dz, r = sqrtf(r2);
    const int l = r < g_tc.r1 ? 0 : 1;
    const float w1 = l == 0 ? far_w1(r) : 0.f;
    float gx, gz;
    far_grad(l, pos.x, pos.z, gx, gz);
    if (w1 > 0.f) {
        float fx, fz;
        far_grad(1, pos.x, pos.z, fx, fz);
        gx += (fx - gx) * w1;
        gz += (fz - gz) * w1;
    }
    const float sx = r2 > g_tc.drop_r2 ? 2.f * g_tc.c2 * dx : 0.f, sz = r2 > g_tc.drop_r2 ? 2.f * g_tc.c2 * dz : 0.f;
    si.n = norm3(f3(-(gx - sx), 1.f, -(gz - sz)));
    const FarLevel& L = g_tc.far[l];
    const float u = (pos.x - L.ox) * L.iS, v = (pos.z - L.oz) * L.iS;
    const int fi = (int)floorf(u), fj = (int)floorf(v);
    FarCell c;
    far_cell(l, fi, fj, c);
    const float a = u - (float)fi, b = v - (float)fj;
    const float bb = (c.h10 - c.h00) * L.iS, cc = (c.h01 - c.h00) * L.iS, kk = (c.h00 - c.h10 - c.h01 + c.h11) * L.iS;
    si.ng = norm3(f3(-(bb + kk * b - sx), 1.f, -(cc + kk * a - sz)));
    far_material(sp, l, w1, pos.x, pos.z, si);
    if (!want_ao) { si.ao = 1.f; return; }
    si.ao = far_ao(l, pos.x, pos.z, si.n);
    if (w1 > 0.f) si.ao += (far_ao(1, pos.x, pos.z, si.n) - si.ao) * w1;
}

// Normal + material of a non-terrain hit (flora, kelp, ice, voxel mode). tclump:
// the primary ray's distance to the hit along its last straight segment (the
// clumped crown's fade, as the walk); < 0: the plain crown ellipsoid (secondary rays)
__device__ void sm_other(const SParams& sp, const THit& h, float3 d, float3 pos, SurfInfo& si, float tclump = -1.f) {
    const KParams& p = sp.k;
    si.dev = 0.f;
    si.ao = 1.f;
    si.w1 = 0.f;
    si.pal = 0;
    si.cover = 0.f;
    si.tsun = 1.f;
    if (h.kind == HK_VOXEL) {                          // classic voxel faces
        float3 n = f3(0.f, 0.f, 0.f);
        if (h.axis == 0) n.x = d.x > 0 ? -1.f : 1.f;
        else if (h.axis == 1) n.y = d.y > 0 ? -1.f : 1.f;
        else n.z = d.z > 0 ? -1.f : 1.f;
        si.n = n; si.ng = n;
        const int32_t ax = g_tc.cx * 32 + h.vx, ay = g_tc.cy * 32 + h.vy, az = g_tc.cz * 32 + h.vz;
        si.m0 = si.m1 = h.mat;
        si.bright = vox_bright(ax, ay, az);
        if (is_tissue(h.mat)) { const SlotLook c = sm_slot_look(p, ax, az); si.pal = pal_look(c.cell, c.sw); si.cover = 1.f; }
        return;
    }
    if (h.kind == HK_ICE) {
        si.n = f3(0.f, 1.f, 0.f); si.ng = si.n;
        const int32_t ax = g_tc.cx * 32 + (int32_t)floorf(pos.x), az = g_tc.cz * 32 + (int32_t)floorf(pos.z);
        si.m0 = si.m1 = MAT_ICE;
        si.bright = vox_bright(ax, g_tc.planet.sea_level, az);
        return;
    }
    const GmeColTile* t = g_tc.tiles + gme_page_index(g_tc.cx + h.rcx, 0, g_tc.cz + h.rcz, g_tc.n, 1);
    if (h.kind == HK_KELP) {                           // M5: the stipe (radial normal) or a blade (its plane, facing the ray)
        const int32_t x = (g_tc.cx + h.rcx) * 32 - GME_TILE_APRON + h.i, z = (g_tc.cz + h.rcz) * 32 - GME_TILE_APRON + h.j;
        const int b = ((h.j - GME_TILE_APRON) >> 3) * 4 + ((h.i - GME_TILE_APRON) >> 3);
        const uint32_t cell = __ldg(&t->slot[b].cell);
        float3 pa, pb;
        uint32_t hb;
        si.n = f3(0.f, 1.f, 0.f);
        if (tr_kelp(t, h.j * GME_TILE_N + h.i, x, z, cell, pa, pb, hb)) {
            const int part = (int)floorf(h.u + 0.5f);
            if (part >= 0) {
                GmeKelpBlade bl;
                tr_kelp_blade(pa, pb, hb, part, bl);
                si.n = tr_blade_normal(bl);
                if (dot3(si.n, d) > 0.f) si.n = si.n * -1.f;
            } else {
                const float3 ba = pb - pa;
                const float s = sat(dot3(pos - pa, ba) / fmaxf(dot3(ba, ba), 1e-6f));
                si.n = norm3(pos - (pa + ba * s));
            }
        }
        si.ng = si.n;
        si.m0 = si.m1 = MAT_KELP;
        si.bright = vox_bright(x, 0, z) * (0.94f + 0.06f * (float)(gme_hash3(x, 0, z, 0x5EEDu) >> 24) / 255.f);
        si.pal = pal_look(cell, __ldg(&t->slot[b].theight));
        si.cover = cell_cover(cell);
        return;
    }
    // crown / trunk of the block's shrub or tree
    const GmeSlot* s = &t->slot[h.i];
    const int32_t tx = __ldg(&s->tx), tz = __ldg(&s->tz), tcy = __ldg(&s->tcy), trad = __ldg(&s->trad), tvr = __ldg(&s->tvr);
    const int kt = (tz - (g_tc.cz + h.rcz) * 32 + GME_TILE_APRON) * GME_TILE_N + (tx - (g_tc.cx + h.rcx) * 32 + GME_TILE_APRON);
    const float dl = (float)gme_tile_dev(t, kt) * (1.f / GME_DEV_UNIT);
    const float3 ce = f3((float)(tx - 32 * g_tc.cx) + 0.5f, (float)(tcy - 32 * g_tc.cy) + 0.5f + dl, (float)(tz - 32 * g_tc.cz) + 0.5f);
    si.bright = vox_bright(tx, tcy, tz);
    if (h.kind == HK_CROWN) {
        // M5: the clumped surface's normal = grad(d_ellipsoid + delta); the clump's depth darkens the gaps
        // (leaf-area medium: e^-(2 sigma delta)); the own crown's transmittance toward the sun
        CrownG c;
        tr_crown_of(t, h.i, h.rcx, h.rcz, c);
        const float3 q = pos - c.ce;
        float3 n = tr_crown_normal(c, q);              // the ellipsoid's gradient; version 2: of the class's solid
        if (tclump >= 0.f) {
            const float amp = gme_clump_amp(trad) * tr_crown_ampk(c) * tr_clump_fade(tclump, gme_clump_cell(trad));
            if (amp > 0.f) {
                float3 g;
                const float del = tr_clump_delta(c, q, amp, tclump * g_tc.pix, &g);
                n = norm3(n + g);
                si.ao = __expf(-2.f * tr_crown_sigma_mean(c) * del);
            }
            // leaf clusters: bumps of 2.5 cm on a 12 cm lattice and 1 cm on a 5 cm one, where they span several pixels
            // (TAA resolves them; finer, they would shimmer); the clumps' brightness varies a little
            const float fl = sat(1.5f - tclump * g_tc.pix / 0.03f), fl2 = sat(1.5f - tclump * g_tc.pix / 0.0125f);
            if (fl > 0.f) {
                float gl[3], gm[3];
                gme_clump_noise(q.x * 8.3f, q.y * 8.3f, q.z * 8.3f, c.cs ^ 0x51EAFu, gl);
                float3 bump = f3(gl[0], gl[1], gl[2]) * (0.025f * 8.3f * fl);
                if (fl2 > 0.f) {
                    gme_clump_noise(q.x * 20.f + 3.1f, q.y * 20.f, q.z * 20.f, c.cs ^ 0xB1ADEu, gm);
                    bump = bump + f3(gm[0], gm[1], gm[2]) * (0.01f * 20.f * fl2);
                }
                n = norm3(n - bump);
            }
            const float ic = 1.f / gme_clump_cell(trad);
            si.bright = 0.88f + 0.12f * gme_clump_noise(q.x * ic + 5.7f, q.y * ic, q.z * ic, c.cs ^ 0xB71Eu, NULL);
        }
        si.n = n;
        si.tsun = __expf(-tr_crown_sigma(c, sp.sun) * tr_crown_exit(c, pos, sp.sun));
        si.m0 = si.m1 = MAT_LEAF;
        const uint32_t cell = __ldg(&s->cell);
        si.pal = pal_look(cell, __ldg(&s->theight));
        si.cover = cell_cover(cell);
        (void)tvr;
    } else {
        si.n = norm3(f3(pos.x - ce.x, 0.f, pos.z - ce.z));
        si.m0 = si.m1 = MAT_WOOD;
    }
    si.ng = si.n;
}

// A creature's sphere: its normal, the skin of its palette index (the pigment's hue bin), its flags.
// Out of line and by value: k_surface's other paths keep their code.
__device__ __noinline__ SurfInfo sm_fauna(const FaunaSphere* sph, const float4* cen, uint32_t idx, float3 pos) {
    SurfInfo si;
    const FaunaSphere s = sph[idx];
    const float4 q = cen[idx];
    si.n = norm3(pos - f3(q.x, q.y, q.z));
    si.ng = si.n;
    si.m0 = si.m1 = RMAT_SKIN;
    si.w1 = 0.f;
    si.pal = (int)FA_PAL(s.attr) | ((s.attr & FA_HUNTER) ? 0x1000 : 0) | ((s.attr & FA_ARMORED) ? 0x2000 : 0);
    si.bright = 0.88f + 0.12f * (float)FA_BRIGHT(s.attr) * (1.f / 63.f);
    si.ao = 0.55f + 0.45f * sat(0.5f + 0.5f * si.n.y);   // the ground under it (the sky ambient of Low; GI finds it itself)
    si.dev = 0.f;
    si.cover = 0.f;
    si.tsun = 1.f;
    si.wb = 0.f;
    return si;
}

// M4 full-res guide word of a pixel: the straight-ray distance of its visible
// surface (the water plane where the path crosses it), normal, class, flags
__device__ __forceinline__ uint2 guide_word(float t, float3 n, int cls, uint32_t fl) {
    return make_uint2(__float_as_uint(t), gg_oct_pack(n.x, n.y, n.z) | ((uint32_t)cls << 24) | fl);
}
__device__ __forceinline__ int kind_class(int kind) {
    switch (kind) {
    case HK_TERRAIN: case HK_LOD: case HK_FAR: return GC_TERRAIN;
    case HK_CROWN: case HK_TRUNK: return GC_FLORA;
    case HK_KELP: return GC_KELP;
    case HK_VOXEL: return GC_VOXEL;
    case HK_ICE: return GC_ICE;
    case HK_DIST: return GC_DIST;
    case HK_FAUNA: return GC_FAUNA;
    default: return GC_SKY;
    }
}
#define GI_BIG_T 3.0e38f

// k_surface: hit record -> G-buffer (normals, material ids, AO), in place; M4: the full-res guide.
// Launch bounds 128 x 5: the kernel must keep its 96 registers (5 blocks per SM on sm_86; 98, which the
// version 2 branches of the analytic voxel fallback would cost it, leave room for 4 only: 12 % slower).
__global__ void __launch_bounds__(128, 5) k_surface(const __grid_constant__ SParams sp, GBuf g) {
    const KParams& p = sp.k;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    const size_t i = (size_t)y * p.w + x;
    float4 a = g.a[i];
    if (!(a.x >= 0.f)) {                               // miss: the record already is the G-buffer entry
        if (sp.fg) {
            const int wm = (int)((g.b[i].x >> 24) & 7u);
            const bool surf = wm == WM_STRAIGHT || wm == WM_REFR || wm == WM_UNDER_OUT || wm == WM_UNDER_TIR;
            sp.fg[i] = surf ? guide_word(a.y, f3(0.f, 1.f, 0.f), GC_WATER, 0u)
                            : guide_word(GI_BIG_T, f3(0.f, 1.f, 0.f), wm == WM_UNDER ? GC_UNDER : GC_SKY, 0u);
        }
        return;
    }
    uint2 b = g.b[i];
    const int wm = (int)((b.x >> 24) & 7u);
    b.x &= 0x00FFFFFFu;
    THit h;
    hit_unpack(a, b, h);
    const float3 d = sp_dir(sp, x, y);
    float3 d2;
    const float3 pos = path_pos(sp, d, h.t, a.y, wm, &d2);
    SurfInfo si;
    si.wb = 0.f;
    si.cover = 0.f;
    si.tsun = 1.f;
    if (h.kind == HK_TERRAIN) sm_terrain(sp, h, pos, si, true, h.t * sp.pix);
    else if (h.kind == HK_LOD) sm_lod(sp, h, pos, x, y, si, true, h.t * sp.pix);
    else if (h.kind == HK_FAR) sm_far(sp, pos, x, y, si);
    else if (h.kind == HK_DIST) {                      // shaded from the sky-view LUT (k_compose)
        si.n = si.ng = f3(0.f, 1.f, 0.f);
        si.m0 = si.m1 = MAT_REGOLITH; si.w1 = 0.f; si.pal = 0; si.bright = 1.f; si.ao = 1.f; si.dev = 0.f;
    }
    else if (h.kind == HK_FAUNA) si = sm_fauna(sp.fauna.sph, sp.fauna.pos, b.y, pos);   // a creature (k_fauna_primary): b.y = its sphere
    else sm_other(sp, h, d2, pos, si, (path_split(wm) && a.y >= 0.f && h.t >= a.y) ? h.t - a.y : h.t);
    // M5 grass (spec 3.5): tufts are shading, not geometry: a fuzz of the shading normal from two octaves of
    // value noise (0.25 / 0.125 m, world-anchored), by the grass share and cover, gone where tufts are sub-pixel
    if ((h.kind == HK_TERRAIN || h.kind == HK_LOD) && (si.m0 == MAT_GRASS || si.m1 == MAT_GRASS)) {
        const float wg = (si.m0 == MAT_GRASS ? 1.f - si.w1 : 0.f) + (si.m1 == MAT_GRASS ? si.w1 : 0.f);
        const float fz = wg * si.cover * sat(1.5f - h.t * sp.pix / 0.12f);
        if (fz > 0.01f) {
            const float axw = pos.x + 32.f * (float)(g_tc.cx & 1023), azw = pos.z + 32.f * (float)(g_tc.cz & 1023);
            const int32_t bx = g_tc.cx >> 10, bz = g_tc.cz >> 10;
            const float s1 = 4.f, s2 = 8.f;                          // lattice cells per metre
            const float gx = vnoise(axw * s1, azw * s1, bx * 131072, bz * 131072 + 17) + 0.5f * vnoise(axw * s2, azw * s2, bx * 262144 + 5, bz * 262144);
            const float gz = vnoise(axw * s1 + 37.3f, azw * s1, bx * 131072 + 11, bz * 131072) + 0.5f * vnoise(axw * s2, azw * s2 + 19.7f, bx * 262144, bz * 262144 + 3);
            si.n = norm3(si.n + f3(gx, 0.f, gz) * (0.45f * fz));
        }
    }
    const bool under = wm == WM_STRAIGHT || wm == WM_REFR || wm == WM_UNDER || wm == WM_UNDER_TIR;
    a.z = __uint_as_float(oct_pack(si.n));
    a.w = __uint_as_float(oct_pack(si.ng));
    b.x = mat_pack(si.m0, si.m1, si.w1, si.pal, si.bright, si.ao);
    const int32_t dq = max(-128, min(127, __float2int_rn(si.dev * (float)GME_DEV_UNIT)));
    const bool terr = h.kind == HK_TERRAIN || h.kind == HK_LOD || h.kind == HK_FAR;
    // bits 8-15: the terrain's dev (debug view), for LOD / far hits the beach share of the far material, for
    // crowns (M5) the own crown's transmittance toward the sun, for creatures their flags; bits 27-31 (M5): the
    // cell's cover; version 2: bits 4-7 the cell's pigment level (0 = none), bits 20-23 its Q band width bin
    const uint32_t b8 = (h.kind == HK_LOD || h.kind == HK_FAR) ? (uint32_t)min(255, __float2int_rn(sat(si.wb) * 255.f))
                      : (h.kind == HK_CROWN ? (uint32_t)min(255, __float2int_rn(sat(si.tsun) * 255.f))
                      : (h.kind == HK_FAUNA ? ((uint32_t)si.pal >> 12) & 3u : ((uint32_t)dq & 255u)));
    b.y = GF_HIT | (terr ? GF_TERRAIN : 0u) | (under ? GF_UNDER : 0u) | (b8 << 8) |
          ((uint32_t)wm << 24) | ((uint32_t)h.kind << 16) | ((uint32_t)min(31, __float2int_rn(sat(si.cover) * 31.f)) << 27) |
          ((uint32_t)si.pal & 0xF0u) | (((uint32_t)si.pal & 0xF00u) << 12);
    g.a[i] = a;
    g.b[i] = b;
    if (sp.fg) {
        uint2 gw;
        if (wm == WM_STRAIGHT || wm == WM_REFR || wm == WM_UNDER_OUT || wm == WM_UNDER_TIR) gw = guide_word(a.y, f3(0.f, 1.f, 0.f), GC_WATER, 0u);
        else if (wm == WM_UNDER) gw = guide_word(h.t, si.n, GC_UNDER, 0u);
        else gw = guide_word(h.t, si.n, kind_class(h.kind), (sp.gi && h.kind != HK_DIST) ? GG_GI : 0u);
        sp.fg[i] = gw;
    }
}

// direction toward a point of the stellar disc (IGN jitter, uniform on the disc;
// M4: an R2 offset per frame, the visibility filter averages the disc)
__device__ __forceinline__ float3 sun_jitter(const SParams& sp, float3 l, int x, int y) {
    if (!sp.jitter) return l;
    const float u1 = frc(ign((float)x, (float)y) + 0.7548776662f * (float)(sp.frame & 1023u) * (float)sp.dither_t);
    const float u2 = frc(ign((float)x + 47.f, (float)y + 13.f) + 0.5698402910f * (float)(sp.frame & 1023u) * (float)sp.dither_t);
    const float r = sp.a.sun_theta * sqrtf(u1), ph = 6.2831853f * u2;
    const float3 t1 = norm3(fabsf(l.y) < 0.99f ? f3(-l.z, 0.f, l.x) : f3(1.f, 0.f, 0.f));
    const float3 t2 = f3(l.y * t1.z - l.z * t1.y, l.z * t1.x - l.x * t1.z, l.x * t1.y - l.y * t1.x);
    float sn, cs;
    sincosf(ph, &sn, &cs);
    return norm3(l + t1 * (r * cs) + t2 * (r * sn));
}

// shadow rays (spec 3.1, 3.7): the fine surface to 32 m (LOD hits: the 4 m
// lattice from the start), crowns to 256 m, the LOD tiles to the ring, then
// the far field (long-range shadows) to 8 km. Without the far field: 512 m, as in M2.
#define SUN_FINE  32.f
#define SUN_FLORA 256.f
#define SUN_FAR   8000.f

// k_sun: one any-hit shadow ray per lit pixel toward the stellar disc (column
// tracer, flora included). Under water the ray follows the refracted sun
// direction up to the water plane, then the true one (spec 3.7). M3: 80
// registers (launch bounds 128 x 6, spec 3.1) measured 3-9 % faster than 64
// with the long-range walk (k_primary uses the same bound since the review).
// M5: crowns. A crown pixel's own crown is a leaf-area medium for its light (k_surface keeps its
// transmittance toward the sun, k_compose its back-lit transmission), so its shadow ray starts where
// the ray leaves the own crown, front- and back-lit alike. kLad (High): other crowns along the ray
// attenuate by e^-(sigma chord) (the sample is fractional) instead of blocking it.
template <int kLad>
__global__ void __launch_bounds__(128, 6) k_sun(const __grid_constant__ SParams sp, GBuf g) {
    const KParams& p = sp.k;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    const size_t i = (size_t)y * p.w + x;
    const float4 a = g.a[i];
    const uint32_t fl = g.b[i].y;
    if (sp.vsmp) sp.vsmp[i] = 255;                     // M4: the binary sample of the visibility filter (lit unless shadowed)
    if (!(fl & GF_HIT) || sp.sun.y <= -0.01f) return;
    const int kind = (int)((fl >> 16) & 15u);
    if (kind == HK_DIST) return;
    const int wm = (int)((fl >> 24) & 7u);
    const bool under = (fl & GF_UNDER) != 0;
    const float3 l = sun_jitter(sp, sp.sun, x, y);
    const float3 ld = under ? sp.sun_r : l;
    if (kind != HK_CROWN && dot3(oct_unpack(__float_as_uint(a.z)), ld) <= 0.f) return;
    float3 d2;
    const float3 pos = path_pos(sp, sp_dir(sp, x, y), a.x, a.y, wm, &d2);
    float3 hp;
    if (kind == HK_CROWN) {                            // from the exit of the own crown toward the sun
        CrownG c;
        hp = tr_crown_at(pos, c) ? pos + ld * (tr_crown_exit(c, pos, ld) + 0.02f) : pos + oct_unpack(__float_as_uint(a.w)) * 0.03f;
    } else {
        const float off = kind == HK_FAR ? 0.25f : (kind == HK_LOD ? 0.08f : 0.03f);
        hp = pos + oct_unpack(__float_as_uint(a.w)) * off;
    }
    const bool far = g_tc.far_on != 0;
    // an LOD hit lies on the 4 m lattice, which may run below the fine surface: its shadow ray walks the lattice
    // from the start (also without the far field), else it would shadow itself on the fine surface
    const float smax = far ? SUN_FAR : 512.f, tl = kind == HK_LOD ? 0.f : (far ? SUN_FINE : 3e38f), tf = far ? SUN_FLORA : 3e38f;
    THit hs;
    hs.water_t = -1.f;
    hs.tau = 0.f;
    bool shadow;
    float occ = -1.f;
    constexpr int FM = kLad ? FM_LAD : FM_PLAIN;
    if (kind == HK_FAR) {                              // far-field points: the far field only
        shadow = tr_far(hp, ld, 0.f, smax, hs);
        occ = hs.t;
    } else if (under && ld.y > 0.f) {
        const float ts = fmaxf((g_tc.yw - hp.y) / ld.y, 0.f);
        shadow = trace_smooth<FM>(hp, ld, 0.f, fminf(ts, smax), tl, tf, hs);
        occ = hs.t;
        if (!shadow && ts < smax) {
            hs.water_t = -1.f;
            shadow = trace_smooth<FM>(hp + ld * ts, l, 0.f, smax - ts, tl - ts, tf - ts, hs);
            occ = ts + hs.t;
        }
    } else {
        shadow = trace_smooth<FM>(hp, ld, 0.f, smax, tl, tf, hs);
        occ = hs.t;
    }
    const float vis = shadow ? 0.f : (kLad ? __expf(-hs.tau) : 1.f);   // LAD: the crowns' transmittance
    if (vis < 0.5f) g.b[i].y = fl | GF_SHADOW;
    if (sp.vsmp) sp.vsmp[i] = (unsigned char)__float2int_rn(vis * 255.f);
    if (sp.sacc) atomicAdd(sp.sacc + (!shadow ? 0 : (occ < 512.f ? 1 : (occ < 2000.f ? 2 : 3))), 1ull);   // --metrics
    if (sp.sdist) sp.sdist[i] = shadow ? fmaxf(occ, 1e-3f) : -1.f;
}

// k_fauna_sun: creatures in the sun's way (after k_sun; only while creatures are near the camera). The
// shadow ray of every pixel k_sun left lit, against the sphere grid over FAUNA_SHADOW_FAR m of the
// path k_sun's ray takes (under water: the refracted direction up to the plane, then the true one).
__global__ void __launch_bounds__(128) k_fauna_sun(const __grid_constant__ SParams sp, GBuf g) {
    const KParams& p = sp.k;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    const size_t i = (size_t)y * p.w + x;
    const float4 a = g.a[i];
    const uint32_t fl = g.b[i].y;
    if (!(fl & GF_HIT) || (fl & GF_SHADOW) || sp.sun.y <= -0.01f || a.x > 2.f * FAUNA_FAR) return;
    const int kind = (int)((fl >> 16) & 15u);
    if (kind == HK_DIST || kind == HK_FAR) return;
    const int wm = (int)((fl >> 24) & 7u);
    const bool under = (fl & GF_UNDER) != 0;
    const float3 l = sun_jitter(sp, sp.sun, x, y);      // the disc point k_sun took
    const float3 ld = under ? sp.sun_r : l;
    if (kind != HK_CROWN && dot3(oct_unpack(__float_as_uint(a.z)), ld) <= 0.f) return;
    float3 d2;
    const float3 pos = path_pos(sp, sp_dir(sp, x, y), a.x, a.y, wm, &d2);
    const float3 hp = kind == HK_CROWN ? pos + ld * 0.05f : pos + oct_unpack(__float_as_uint(a.w)) * 0.03f;
    FaunaHit fh;
    if (under && ld.y > 0.f) {
        const float ts = fmaxf((g_tc.yw - hp.y) / ld.y, 0.f);
        fh = fauna_trace(sp.fauna, hp, ld, 0.f, fminf(ts, FAUNA_SHADOW_FAR), 0.f, 0.f, true);
        if (fh.s < 0 && ts < FAUNA_SHADOW_FAR) fh = fauna_trace(sp.fauna, hp + ld * ts, l, 0.f, FAUNA_SHADOW_FAR - ts, 0.f, 0.f, true);
    } else {
        fh = fauna_trace(sp.fauna, hp, ld, 0.f, FAUNA_SHADOW_FAR, 0.f, 0.f, true);
    }
    if (fh.s < 0) return;
    g.b[i].y = fl | GF_SHADOW;
    if (sp.vsmp) sp.vsmp[i] = 0;
}

// ---- M4 secondary rays: GI and reflections (spec 3.1, 4.6) --------------------------------
// GI rays: fine 1 m patches to 8 m, the 4 m lattice to 256 m (the far field where
// tiles are missing or past the ring), crowns to 64 m, then the sky. Reflection
// rays: fine to 64 m, the lattice and the far field to the horizon, flora to 512 m.
#define GI_TMAX   256.f
#define GI_FINE   8.f
#define GI_FLORA  64.f
#define RF_FINE   64.f
#define RF_FLORA  512.f

// cosine-weighted direction around n from two uniforms
__device__ __forceinline__ float3 cos_dir(float3 n, float u1, float u2) {
    const float r = sqrtf(u1), ph = 6.2831853f * u2;
    float sn, cs;
    __sincosf(ph, &sn, &cs);
    const float3 t1 = norm3(fabsf(n.y) < 0.99f ? f3(-n.z, 0.f, n.x) : f3(1.f, 0.f, 0.f));
    const float3 t2 = f3(n.y * t1.z - n.z * t1.y, n.z * t1.x - n.x * t1.z, n.x * t1.y - n.y * t1.x);
    return norm3(t1 * (r * cs) + t2 * (r * sn) + n * sqrtf(fmaxf(0.f, 1.f - u1)));
}
// mirror (a = 0) or GGX-sampled reflection of view direction d at normal n
__device__ __forceinline__ float3 refl_dir(float3 d, float3 n, float a, float u1, float u2) {
    float3 h = n;
    if (a > 0.f) {
        const float th = atanf(a * sqrtf(u1 / fmaxf(1.f - u1, 1e-6f))), ph = 6.2831853f * u2;
        float sn, cs, st, ct;
        __sincosf(ph, &sn, &cs);
        __sincosf(th, &st, &ct);
        const float3 t1 = norm3(fabsf(n.y) < 0.99f ? f3(-n.z, 0.f, n.x) : f3(1.f, 0.f, 0.f));
        const float3 t2 = f3(n.y * t1.z - n.z * t1.y, n.z * t1.x - n.x * t1.z, n.x * t1.y - n.y * t1.x);
        h = norm3(t1 * (st * cs) + t2 * (st * sn) + n * ct);
    }
    float3 r = d - h * (2.f * dot3(d, h));
    if (dot3(r, n) < 0.01f) r = d - n * (2.f * dot3(d, n));   // below the surface: the mirror direction
    return norm3(r);
}
// reference paths: a small hash RNG per pixel, frame, slot and bounce
__device__ __forceinline__ uint32_t pcg_next(uint32_t& s) {
    s = s * 747796405u + 2891336453u;
    uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
__device__ __forceinline__ float rng01(uint32_t& s) { return (float)(pcg_next(s) >> 8) * (1.f / 16777216.f); }

__device__ __forceinline__ void sec_put(SecBuf sb, int slot, float3 o, float tmax, float3 d, uint32_t flags) {
    sb.o[slot] = make_float4(o.x, o.y, o.z, tmax);
    sb.r[slot] = make_uint4(oct_pack(d), flags, 0u, 0u);
}

// k_sec_gen: per grid pixel, the primary point of its (representative) pixel ->
// GI rays (cosine, IGN + R2 frame offset; reference: hash RNG) and one
// reflection ray (water: mirror at the Gerstner normal; ice / wet ground: mirror
// or GGX), the half-res guide. Reference: the paths' throughput and radiance.
// (launch bounds 128 x 8: 64 registers, as before the version 2 crown calls)
__global__ void __launch_bounds__(128, 8) k_sec_gen(const __grid_constant__ SParams sp, GBuf g, SecBuf sb, SecGrid G) {
    const KParams& p = sp.k;
    const int gi = blockIdx.x * blockDim.x + threadIdx.x, gj = blockIdx.y * blockDim.y + threadIdx.y;
    if (gi >= G.gw || gj >= G.gh) return;
    const int x = G.half ? min(2 * gi + sp.hx, p.w - 1) : gi, y = G.half ? min(2 * gj + sp.hy, p.h - 1) : gj;
    const size_t pi = (size_t)y * p.w + x, gidx = (size_t)gj * G.gw + gi;
    const float4 a = g.a[pi];
    const uint2 b = g.b[pi];
    const float3 d = sp_dir(sp, x, y);
    const bool hit = (b.y & GF_HIT) != 0;
    const int wm = (int)(((hit ? b.y : b.x) >> 24) & 7u);
    const int kind = hit ? (int)((b.y >> 16) & 15u) : HK_NONE;
    bool do_gi = false, do_refl = false, glossy = false;
    float3 go = f3(0.f, 0.f, 0.f), gn = f3(0.f, 1.f, 0.f), ro = f3(0.f, 0.f, 0.f), rd = f3(0.f, 1.f, 0.f);
    float rtmax = 0.f;
    const float tfar = g_tc.far_on ? g_tc.r2 : p.max_dist;
    if ((wm == WM_STRAIGHT || wm == WM_REFR) && a.y >= 0.f) {   // the water surface seen from above: its reflection
        const float tw = a.y;
        const float3 pw = p.o + d * tw;
        const float3 nw = water_normal(sp, pw.x, pw.z, water_foot(sp, d, tw) * (G.half ? 2.f : 1.f), NULL);
        float3 r = d - nw * (2.f * dot3(d, nw));
        // a wave facet that sends the view ray downward sends it into the next wave: the ray meets the water plane
        // within a metre (k_sec_trace: HKX_WATER, the sky reflected once more, with its Fresnel). Clamped up to the
        // horizon instead, such facets reflected the far shore as brown dashes on open water (more of them with the
        // stochastic wave spectrum, whose slopes are Gaussian)
        r.y = r.y < 0.f ? fminf(r.y, -0.02f) : fmaxf(r.y, 0.01f);
        rd = norm3(r);
        ro = pw + f3(0.f, 0.02f, 0.f);
        rtmax = tfar;
        do_refl = G.refl != 0;
    } else if (hit && wm == WM_NONE && kind != HK_DIST) {
        const float3 pos = p.o + d * a.x;
        const float3 n = oct_unpack(__float_as_uint(a.z)), ng = oct_unpack(__float_as_uint(a.w));
        const float off = (kind == HK_FAR ? 0.25f : (kind == HK_LOD ? 0.08f : 0.03f)) + 5e-4f * a.x;
        go = pos + ng * off;
        gn = n;
        do_gi = G.gi_n > 0;
        // reflections: ice, and the wet tidal band of ground (spec 4.4 / 4.6)
        float ya = pos.y;
        if (g_tc.far_on) ya += far_drop(pos.x, pos.z);
        const float wet = (kind == HK_TERRAIN || kind == HK_VOXEL || kind == HK_LOD || kind == HK_FAR) && !sp.furnace
                              ? smooth01(g_tc.yw + 1.5f, g_tc.yw, ya) : 0.f;
        if (G.refl && (kind == HK_ICE || wet > 0.02f)) {
            do_refl = true;
            glossy = G.glossy != 0 || kind != HK_ICE;
            ro = go;
            rtmax = tfar;
            const float al = kind == HK_ICE ? 0.08f : 0.05f;
            const float u1 = frc(ign((float)x + 7.f, (float)y + 29.f) + 0.7548776662f * (float)(sp.frame & 1023u));
            const float u2 = frc(ign((float)x + 31.f, (float)y + 3.f) + 0.5698402910f * (float)(sp.frame & 1023u));
            rd = refl_dir(d, n, glossy ? al : 0.f, u1, u2);
        }
    }
    if (G.hg) {
        uint2 hw = sp.fg ? sp.fg[pi] : make_uint2(0u, 0u);
        hw.y = (hw.y & 0x0FFFFFFFu) | (do_gi ? GG_GI : 0u) | (do_refl ? GG_REFL : 0u);
        G.hg[gidx] = hw;
    }
    uint32_t rs = gi_hash32((uint32_t)gidx * 9781u + sp.frame * 6271u + 1u);
    CrownG crown;                                       // M5: a crown point's GI rays start where they leave its own crown
    const bool in_crown = do_gi && kind == HK_CROWN && tr_crown_at(p.o + d * a.x, crown);
    for (int s = 0; s < G.K; ++s) {
        const int slot = G.base + (int)gidx * G.K + s;
        const bool is_gi = s < G.gi_n;
        const bool on = is_gi ? do_gi : (s == G.gi_n && do_refl);
        if (!on) { sb.r[slot] = make_uint4(0u, 0u, 0u, 0u); continue; }
        uint32_t fl = SF_ACTIVE | (sp.ref ? SF_PATH : 0u) | (kind == HK_VOXEL && is_gi ? SF_VOXEL : 0u);
        if (is_gi) {
            float u1, u2;
            if (sp.ref) { u1 = rng01(rs); u2 = rng01(rs); }
            else {
                const float fi = (float)((sp.frame * (uint32_t)G.gi_n + (uint32_t)s) & 4095u);
                u1 = frc(ign((float)x, (float)y) + 0.7548776662f * fi);
                u2 = frc(ign((float)x + 47.f, (float)y + 13.f) + 0.5698402910f * fi);
            }
            float3 dir = cos_dir(gn, u1, u2);
            const float3 ng = oct_unpack(__float_as_uint(a.w));
            const float dn = dot3(dir, ng);
            if (dn < 0.01f) dir = norm3(dir + ng * (0.01f - dn) * 1.01f);   // never into the geometric surface
            if (in_crown) {
                const float3 pc = p.o + d * a.x;
                sec_put(sb, slot, pc + dir * (tr_crown_exit(crown, pc, dir) + 0.02f), GI_TMAX, dir, fl | ((uint32_t)ST_GI << 1));
            } else {
                sec_put(sb, slot, go, GI_TMAX, dir, fl | ((uint32_t)ST_GI << 1));
            }
        } else {
            sec_put(sb, slot, ro, rtmax, rd, fl | ((uint32_t)(glossy ? ST_GLOSSY : ST_MIRROR) << 1));
        }
        if (sp.ref) {
            const float one[8] = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
            sb.thr[slot] = gi_pack8(one);
            sb.acc[slot] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

// k_sec_trace: one secondary ray per thread -> hit record (in place)
__global__ void __launch_bounds__(128, 6) k_sec_trace(const __grid_constant__ SParams sp, SecBuf sb, int n0, int n) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int slot = n0 + k;
    uint4 r = sb.r[slot];
    if (!(r.y & SF_ACTIVE)) return;
    const float4 o4 = sb.o[slot];
    const float3 o = f3(o4.x, o4.y, o4.z), d = oct_unpack(r.x);
    const int type = (int)((r.y >> 1) & 3u);
    const float tl = type == ST_GI ? GI_FINE : RF_FINE, tf = type == ST_GI ? GI_FLORA : RF_FLORA, tmax = o4.w;
    THit h;
    h.water_t = -1.f;
    h.tice = -1.f;
    h.kind = HK_NONE;
    bool hit = false;
    float t0 = 0.f;
    if (r.y & SF_VOXEL) {                                   // from a cave wall: the grid's voxels until above the surface
        float tx;
        const int vr = trace_voxels_call(o, d, 0.f, tmax, false, true, h, tx);
        if (vr == VX_HIT) hit = true;
        else if (vr == VX_EXIT) t0 = tx;
        else t0 = tmax;
    }
    if (!hit && t0 < tmax) hit = trace_smooth(o, d, t0, tmax, tl, tf, h);
    // the water plane (or the ice sheet), seen from above
    if (d.y < 0.f && o.y > g_tc.yw) {
        const float ts = (g_tc.yw - o.y) / d.y;
        float T;
        if (ts < (hit ? h.t : tmax) && tr_ground(o.x + d.x * ts, o.z + d.z * ts, ts < tl, T) && T < g_tc.yw) {
            hit = true;
            h.t = ts;
            h.kind = g_tc.ice ? HK_ICE : HKX_WATER;
        }
    }
    float4 a;
    uint2 b;
    if (hit && h.kind == HKX_WATER) { a = make_float4(h.t, -1.f, 0.f, 0.f); b = make_uint2((uint32_t)HKX_WATER, 0u); }
    else hit_pack(h, hit, a, b);
    r.y = (r.y & ~0xF0u) | ((hit ? (uint32_t)(h.kind & 15) : 0u) << 4);
    r.z = b.x;
    r.w = b.y;
    sb.r[slot] = r;
    sb.h[slot] = make_float4(hit ? a.x : -1.f, a.z, a.w, 0.f);
}

// k_sec_nee: the surface at a secondary hit (normal, materials) and its sun
// visibility over coarse geometry (fine 16 m, then the lattice; crowns to 64 m;
// the far field to 2 km)
// M5: r.y bits 13-15 the sun visibility as 0..7 (kLad, High: crowns along the shadow ray attenuate
// by e^-(sigma chord); else 0 or 7), bits 24-28 the cell's cover (grass)
template <int kLad>
__global__ void __launch_bounds__(128, 6) k_sec_nee(const __grid_constant__ SParams sp, SecBuf sb, int n0, int n) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int slot = n0 + k;
    uint4 r = sb.r[slot];
    if (!(r.y & SF_ACTIVE)) return;
    const int kind = (int)((r.y >> 4) & 15u);
    if (kind == HK_NONE || kind == HKX_WATER) return;
    const float4 o4 = sb.o[slot], h4 = sb.h[slot];
    const float3 d = oct_unpack(r.x);
    const float3 pos = f3(o4.x, o4.y, o4.z) + d * h4.x;
    THit h;
    hit_unpack(make_float4(h4.x, -1.f, h4.y, h4.z), make_uint2(r.z, r.w), h);
    SurfInfo si;
    si.wb = 0.f;
    si.cover = 0.f;
    si.tsun = 1.f;
    if (kind == HK_TERRAIN) sm_terrain(sp, h, pos, si, false);
    else if (kind == HK_LOD) sm_lod(sp, h, pos, 0, 0, si, false);
    else if (kind == HK_FAR) sm_far(sp, pos, 0, 0, si, false);
    else sm_other(sp, h, d, pos, si);
    float vis = 0.f;
    if (!sp.furnace && sp.sun.y > -0.01f && dot3(si.n, sp.sun) > 0.f && dot3(si.ng, sp.sun) > -0.05f) {
        const float off = kind == HK_FAR ? 0.25f : (kind == HK_LOD ? 0.08f : 0.04f);
        const float3 hp = pos + si.ng * off;
        const bool far = g_tc.far_on != 0;
        const float smax = far ? 2000.f : 512.f;
        THit hs;
        hs.water_t = -1.f;
        hs.tau = 0.f;
        if (kind == HK_FAR) vis = tr_far(hp, sp.sun, 0.f, smax, hs) ? 0.f : 1.f;
        else if (!trace_smooth<kLad ? FM_LAD : FM_PLAIN>(hp, sp.sun, 0.f, smax, kind == HK_LOD ? 0.f : 16.f, GI_FLORA, hs))
            vis = kLad ? __expf(-hs.tau) : 1.f;
    }
    const uint32_t vq = (uint32_t)min(7, __float2int_rn(vis * 7.f));
    r.z = oct_pack(si.n);
    r.w = mat_pack(si.m0, si.m1, si.w1, si.pal, si.bright, 1.f);
    r.y = (r.y & 0xE0001EFFu) | (vq ? SF_VIS : 0u) | (vq << 13) | ((uint32_t)min(255, __float2int_rn(sat(si.wb) * 255.f)) << 16) |
          ((uint32_t)min(31, __float2int_rn(sat(si.cover) * 31.f)) << 24);
    sb.r[slot] = r;
}

// ---- spectral shading helpers ---------------------------------------------------
__device__ __forceinline__ void sh_basis(float3 n, float Y[9]) {
    Y[0] = 0.282095f;
    Y[1] = 0.488603f * n.y; Y[2] = 0.488603f * n.z; Y[3] = 0.488603f * n.x;
    Y[4] = 1.092548f * n.x * n.y; Y[5] = 1.092548f * n.y * n.z; Y[6] = 0.315392f * (3.f * n.z * n.z - 1.f);
    Y[7] = 1.092548f * n.x * n.z; Y[8] = 0.546274f * (n.x * n.x - n.y * n.y);
}
// irradiance from order-2 SH radiance coefficients c[9] (cosine lobe convolution)
__device__ __forceinline__ float sh_irr(const float* c, const float Y[9]) {
    return fmaxf(3.14159265f * __ldg(c) * Y[0] +
                 2.0943951f * (__ldg(c + 1) * Y[1] + __ldg(c + 2) * Y[2] + __ldg(c + 3) * Y[3]) +
                 0.78539816f * (__ldg(c + 4) * Y[4] + __ldg(c + 5) * Y[5] + __ldg(c + 6) * Y[6] + __ldg(c + 7) * Y[7] + __ldg(c + 8) * Y[8]),
                 0.f);
}
__device__ __forceinline__ float e_spec(const ShadeTab* T, int ior, float mu, float alpha) {
    const float fx = sat(mu) * (SPEC_EL_N - 1), fy = sat(alpha) * (SPEC_EL_N - 1);
    const int i = min((int)fx, SPEC_EL_N - 2), j = min((int)fy, SPEC_EL_N - 2);
    const float u = fx - (float)i, v = fy - (float)j;
    const float* r0 = &T->e_lut[ior][j][0];
    const float* r1 = &T->e_lut[ior][j + 1][0];
    return (__ldg(r0 + i) * (1.f - u) + __ldg(r0 + i + 1) * u) * (1.f - v) + (__ldg(r1 + i) * (1.f - u) + __ldg(r1 + i + 1) * u) * v;
}
__device__ __forceinline__ float e_avg(const ShadeTab* T, int ior, float alpha) {
    const float fy = sat(alpha) * (SPEC_EL_N - 1);
    const int j = min((int)fy, SPEC_EL_N - 2);
    const float v = fy - (float)j;
    return __ldg(&T->e_avg[ior][j]) * (1.f - v) + __ldg(&T->e_avg[ior][j + 1]) * v;
}
// band transmittance of pure + coloured water over d metres (log-spaced table)
__device__ float water_T(const ShadeTab* T, int b, float d) {
    if (!(d > 0.f)) return 1.f;
    if (d >= SPEC_WT_D1) return 0.f;
    const float L = __logf(SPEC_WT_D1 / SPEC_WT_D0);
    const float x = __logf(d / SPEC_WT_D0) / L * (float)(SPEC_WT_N - 1);
    if (x <= 0.f) return __expf(__ldg(&T->water_lnT[b][0]) * d / SPEC_WT_D0);
    const int i = min((int)x, SPEC_WT_N - 2);
    const float d0 = SPEC_WT_D0 * __expf(L * (float)i / (SPEC_WT_N - 1)), d1 = SPEC_WT_D0 * __expf(L * (float)(i + 1) / (SPEC_WT_N - 1));
    const float l0 = __ldg(&T->water_lnT[b][i]), l1 = __ldg(&T->water_lnT[b][i + 1]);
    return __expf(l0 + (l1 - l0) * (d - d0) / (d1 - d0));
}
// integral of e^(-k s) over [0, len] (len < 0: to infinity)
__device__ __forceinline__ float wint(float k, float len) {
    if (len < 0.f) return k > 1e-6f ? 1.f / k : 1e6f;
    const float x = k * len;
    if (fabsf(x) < 1e-4f) return len * (1.f - 0.5f * x);
    return (1.f - __expf(-x)) / k;
}
// one underwater segment (spec 4.5): radiance L at its far end seen from its
// start (depth z0, direction u, length len; len < 0 = deep water), plus the sun
// (beam irradiance ew0 just below the surface) and sky (downwelling ed0) light
// scattered into it. Band b.
struct WSeg { float tb, ksun, ksky; };   // L' = L tb + ew0 ksun + ed0 ksky
__device__ WSeg water_seg_band(const SParams& sp, int b, float z0, float3 u, float len) {
    const ShadeTab* T = sp.tab;
    const float wa = __ldg(&T->water_a[b]), wb = __ldg(&T->water_b[b]), wbm = __ldg(&T->water_bm[b]);
    const float kap = wa + 0.2f * wb;
    const float mur = fmaxf(sp.sun_r.y, 0.05f);
    const float r = -u.y;                              // depth gained per metre of path
    const float c = dot3(sp.sun_r, u);
    const float beta = wbm * spec_phase_water_mol(c) + (wb - wbm) * spec_phase_hg(c, 0.92f);
    WSeg w;
    w.tb = len < 0.f ? 0.f : water_T(T, b, len) * __expf(-0.2f * wb * len);
    w.ksun = beta * __expf(-kap * z0 / mur) * wint(kap * (1.f + r / mur), len);
    w.ksky = wb * 0.0795775f * __expf(-1.2f * kap * z0) * wint(kap * (1.f + 1.2f * r), len);
    return w;
}
__device__ __forceinline__ float water_apply(const WSeg& w, float L, float ew0, float ed0) { return L * w.tb + ew0 * w.ksun + ed0 * w.ksky; }
// per-g sky radiance from the sky-view buffer (bilinear)
struct SkyTap { int o00, o10, o01, o11; float w00, w10, w01, w11; };
__device__ SkyTap sky_tap(const AtmoParams& P, float3 d) {
    float x, y;
    spec_sky_tc(P, d.x, d.y, d.z, &x, &y);
    const float fx = fminf(fmaxf(x * SPEC_SV_W - 0.5f, 0.f), (float)(SPEC_SV_W - 1));
    const float fy = fminf(fmaxf(y * SPEC_SV_H - 0.5f, 0.f), (float)(SPEC_SV_H - 1));
    const int i = min((int)fx, SPEC_SV_W - 2), j = min((int)fy, SPEC_SV_H - 2);
    const float u = fx - (float)i, v = fy - (float)j;
    SkyTap t;
    t.o00 = (j * SPEC_SV_W + i) * SPEC_NG; t.o10 = t.o00 + SPEC_NG; t.o01 = t.o00 + SPEC_SV_W * SPEC_NG; t.o11 = t.o01 + SPEC_NG;
    t.w00 = (1.f - u) * (1.f - v); t.w10 = u * (1.f - v); t.w01 = (1.f - u) * v; t.w11 = u * v;
    return t;
}
__device__ __forceinline__ float sky_g(const AtmoParams& P, const SkyTap& t, int g) {
    return __ldg(P.sv_spec + t.o00 + g) * t.w00 + __ldg(P.sv_spec + t.o10 + g) * t.w10 + __ldg(P.sv_spec + t.o01 + g) * t.w01 +
           __ldg(P.sv_spec + t.o11 + g) * t.w11;
}
// classic RGB of pigmented tissue (render.cu albedo()): only the --debug palette view keeps it, with the
// old HSV palette tint (M5: the pigment model replaced it)
__device__ __forceinline__ float3 tissue_rgb(int m) {
    switch (m) {
    case MAT_GRASS: return f3(0.28f, 0.52f, 0.20f);
    case MAT_MAT: case MATX_SOILMAT: return f3(0.30f, 0.22f, 0.30f);
    case MAT_LEAF:  return f3(0.25f, 0.45f, 0.18f);
    default:        return f3(0.20f, 0.35f, 0.18f);   // kelp
    }
}
__device__ __forceinline__ int hue_bin(int pal) { return (pal & 15) ? (pal & 15) : SPEC_HUE_DEFAULT; }
// spectral albedo of material m in band b. M5 pigmented tissue (spec 5.1) by its hue bin: leaves and kelp
// rho_L, mats 0.03 + rho_sand (1 - A)^2; cover: the tissue's coverage of its ground at the point (the coverage
// field of sm_terrain_mat; 1 for a voxel / far tissue part): grass mix(soil, rho_L, LEAFY cover), mats
// mix(ground, mat, cover) on sand (MAT_MAT) or soil (MATX_SOILMAT); grass and mats on steep ground show the soil
// version 2 (pal bits 4-7: the cell's pigment level 1..15, 0 = no morphology; bits 8-11: its Q band width bin): the
// same tissue laws from the absorptance of that pigment amount and band width (ShadeTab.pig_A2)
__device__ __forceinline__ float mat_rho(const SParams& sp, int m, int pal, float cover, float steep, int b) {
    const ShadeTab* T = sp.tab;
    float r;
    if (is_tissue(m)) {
        const int hb = hue_bin(pal);
        // version 2: 1 - A of the cell's own pigment amount and Q band width, else the hue bin's tables
        const bool v2 = (pal & 0xF0) != 0;
        const float t1 = v2 ? 1.f - __ldg(&T->pig_A2[hb][((pal >> 4) & 15) - 1][(pal >> 8) & 15][b]) : 0.f;
        if (m == MAT_MAT || m == MATX_SOILMAT) {
            const float gr = __ldg(&T->rho[(m == MAT_MAT ? g_tc.planet.beach_mat : g_tc.planet.surface_mat) & 15][b]);
            const float mt = v2 ? 0.03f + __ldg(&T->rho[MAT_SAND][b]) * t1 * t1 : __ldg(&T->mat_tis[hb][b]);
            r = gr + (mt - gr) * cover;
        } else if (m == MAT_GRASS) {
            const float soil = __ldg(&T->rho[g_tc.planet.surface_mat & 15][b]);
            r = soil + ((v2 ? 0.04f + 0.44f * t1 : __ldg(&T->leaf_rho[hb][b])) - soil) * LEAFY * cover;
        } else r = v2 ? 0.04f + 0.44f * t1 : __ldg(&T->leaf_rho[hb][b]);
        if (m == MAT_GRASS || m == MAT_MAT || m == MATX_SOILMAT) r += (__ldg(&T->rho[MAT_DIRT][b]) - r) * 0.55f * steep;
    } else {
        r = __ldg(&T->rho[m & 15][b]);
    }
    return r;
}
// a creature's albedo in band b: the skin of its hue bin (pal), its brightness; one that hunts is darker
__device__ __forceinline__ float skin_rho(const SParams& sp, int pal, float bright, bool hunter, int b) {
    return __ldg(&sp.tab->skin[hue_bin(pal)][b]) * bright * (hunter ? 0.55f : 1.f);
}
// the classic colours and tint of the --debug palette view (linear RGB; a copy of the classic albedo(),
// which stays untouched)
__device__ __forceinline__ float3 palette_rgb(const SParams& sp, int m, int pal) {
    float3 c;
    switch (m) {
    case MAT_STONE:    c = f3(0.36f, 0.36f, 0.38f); break;
    case MAT_DIRT:     c = f3(0.40f, 0.28f, 0.18f); break;
    case MAT_SAND:     c = f3(0.82f, 0.76f, 0.52f); break;
    case MAT_GRANITE:  c = f3(0.62f, 0.56f, 0.53f); break;
    case MAT_IRON_ORE: c = f3(0.55f, 0.24f, 0.14f); break;
    case MAT_SULFIDE:  c = f3(0.68f, 0.60f, 0.24f); break;
    case MAT_ICE:      c = f3(0.82f, 0.90f, 0.97f); break;
    case MAT_REGOLITH: c = f3(0.52f, 0.46f, 0.40f); break;
    case MAT_WOOD:     c = f3(0.35f, 0.25f, 0.15f); break;
    default:           c = is_tissue(m) ? tissue_rgb(m) : f3(1.f, 0.f, 1.f); break;
    }
    if (is_tissue(m) && pal) c = mix3(c, sp.k.bio_pal[pal & 15], 0.7f);
    return c;
}

// per-pixel description of a lit surface point
struct Lit {
    float3 n;
    float  nl, spec, specw, dvis, vh, comp, amb_spec, ao, vis;
    int    cond;
    int    m0, m1, pal;
    float  w1, bright, steep, wet;
    float  wb;                   // far material model: beach share blended into m0 (M3)
    bool   under;
    float  depth;                // below the water plane
    float  As, Hs;               // sun columns at the point (at the surface above it when under water)
    float  Y[9];                 // SH basis of n
    // M4
    int    gi;                   // the ambient is the GI irradiance egi (per band, split to g-points), not SH x AO
    float  egi[SPEC_NB];
    float  caus;                 // caustic factor of the sun under water (High), 1 otherwise
    // M5
    float  cover;                // the cell's cover 0..1 (grass albedo)
    float  tw;                   // leaf transmission of the sun: crowns sat(-2 n.l) (x the own crown's e^-(sigma chord)
                                 // in vis), kelp blades sat(-n.l); 0 for other surfaces
    float  tamb;                 // kelp: the ambient from the other side through the blade (two-sided)
    float  sheen;                // grass: the Charlie sheen lobe (D V) x weight
    int    tavg;                 // terrain: the ambient specular is the lobe's cosine-averaged albedo E_avg (lit_setup)
};
// GGX + energy-compensated diffuse terms of a lit point (spec 4.4)
// (L.pal: bits 4-11 as SurfInfo.pal; alpha_k: roughness factor, 1 but for a creature's armour)
__device__ void lit_setup(const SParams& sp, Lit& L, float3 v, float3 l, float alpha_k = 1.f) {
    L.sheen = 0.f;
    if (sp.furnace) {                                   // D4: Lambertian, no specular lobe
        L.nl = fmaxf(dot3(L.n, l), 0.f); L.vh = 1.f; L.dvis = 0.f; L.cond = 0; L.spec = 0.f; L.specw = 0.f;
        L.comp = 0.31830989f; L.amb_spec = 0.f;
        return;
    }
    const ShadeTab* T = sp.tab;
    const int mm = L.w1 > 0.5f ? L.m1 : L.m0;
    float alpha = __ldg(&T->alpha[L.m0 & 15]) * (1.f - L.w1) + __ldg(&T->alpha[L.m1 & 15]) * L.w1;
    alpha = fmaxf(alpha * alpha_k, 0.02f);
    const int ior = __ldg(&T->ior[mm & 15]);
    const float iorn = ior == SPEC_IOR_ROCK ? 1.55f : (ior == SPEC_IOR_LEAF ? 1.45f : (ior == SPEC_IOR_ICE ? 1.31f : WATER_N));
    const float nv = fmaxf(dot3(L.n, v), 1e-3f), nl = fmaxf(dot3(L.n, l), 0.f);
    const float3 hv = norm3(l + v);
    const float nh = fmaxf(dot3(L.n, hv), 0.f), vh = fmaxf(dot3(v, hv), 0.f);
    L.nl = nl;
    L.vh = vh;
    L.dvis = spec_ggx_d(nh, alpha) * spec_ggx_vis(nl, nv, alpha);
    L.cond = mm == MAT_SULFIDE;
    L.spec = L.cond ? 0.f : L.dvis * spec_fresnel(vh, iorn);
    const float Eo = e_spec(T, ior, nv, alpha), Ei = e_spec(T, ior, nl, alpha), Ea = e_avg(T, ior, alpha);
    L.comp = (1.f - Ei) * (1.f - Eo) / fmaxf(1.f - Ea, 1e-3f) * 0.31830989f;
    // the ambient specular: the irradiance x the lobe's albedo. E(mu_o) grows toward grazing view (0.02 head-on,
    // 0.1 at mu_o -> 0 for rock); on terrain the C0 shading normal turns sharply away from the camera along creases,
    // so E(mu_o) drew a network of bright sky-blue veins on shadowed slopes at low sun and rim lines on ridges. The
    // drawn heightfield is smoother than real ground, whose visible facets spread the view angle: terrain takes the
    // cosine average E_avg (no view dependence, measured: the veins go, the mean level drops ~1 %)
    L.amb_spec = L.tavg ? Ea : Eo;
    L.specw = L.wet > 0.f ? L.wet * spec_ggx_d(nh, 0.05f) * spec_ggx_vis(nl, nv, 0.05f) * spec_schlick(0.02f, vh) : 0.f;
    // M5 grass: the blades' sheen (Charlie distribution, alpha 0.5, Ashikhmin's visibility), its colour the albedo
    const float wg = (L.m0 == MAT_GRASS ? 1.f - L.w1 : 0.f) + (L.m1 == MAT_GRASS ? L.w1 : 0.f);
    if (wg > 0.f && nl > 0.f) {
        const float s2 = fmaxf(1.f - nh * nh, 0.f);
        L.sheen = 0.5f * wg * L.cover * (0.63662f * s2) / (4.f * fmaxf(nl + nv - nl * nv, 1e-3f));
    }
}
// albedo of a lit point in band b (two materials, brightness, wet darkening)
__device__ __forceinline__ float lit_rho(const SParams& sp, const Lit& L, int b) {
    if (sp.furnace) return 0.5f;
    float r0 = mat_rho(sp, L.m0, L.pal, L.cover, L.steep, b);
    if (L.wb > 0.f) r0 += (mat_rho(sp, g_tc.planet.beach_mat, L.pal, L.cover, L.steep, b) - r0) * L.wb;
    float r = r0 * (1.f - L.w1) + mat_rho(sp, L.m1, L.pal, L.cover, L.steep, b) * L.w1;
    r *= L.bright;
    if (L.wet > 0.f) r += (0.493f * r / (1.f - 0.472f * r) - r) * L.wet;   // Lekner & Dorf
    return r;
}
// radiance of a lit point in g-point g (band b, albedo rho, leaf transmittance tau): direct sun + sky
// ambient; twd / twa: water transmittance of the sun path / of the diffuse light. M5 leaves: the sun
// transmitted through crowns and kelp blades (tau / pi E tw, spec 4.6), kelp's two-sided ambient, grass sheen
__device__ __forceinline__ float lit_radiance(const SParams& sp, const Lit& L, int g, int b, float rho, float tau, float twd, float twa) {
    const AtmoParams& P = sp.a;
    const float spec = L.cond ? L.dvis * spec_schlick(__ldg(&sp.tab->pyrite_f0[b]), L.vh) : L.spec;
    float esun, eamb, eamb_t = 0.f;
    if (L.under) {
        esun = __ldg(&sp.fs->esun0[g]) * sp.sun_t * twd * L.vis * L.caus;
        const float eu = 0.934f * twa * __ldg(&sp.fs->eup[g]);
        eamb = eu * (0.5f + 0.5f * L.n.y);
        eamb_t = eu * (0.5f - 0.5f * L.n.y) * L.tamb;
    } else {
        esun = sp.furnace ? 0.f : P.g_e[g] * __expf(-L.As * P.g_a[g] - L.Hs * P.g_b[g]) * L.vis;
        eamb = L.gi ? L.egi[b] * __ldg(&sp.fs->gsplit[g]) : sh_irr(P.sh + 9 * g, L.Y);   // M4: GI irradiance, AO = 1
    }
    return esun * (L.nl * (spec + L.specw + rho * (L.comp + L.sheen)) + tau * L.tw * 0.31830989f) +
           eamb * L.ao * (rho * (1.f - L.amb_spec) + L.amb_spec) * 0.31830989f + tau * eamb_t * 0.31830989f;
}

// ---- M4: secondary-ray shading -------------------------------------------------------------
__device__ __forceinline__ void air_cols(const AtmoParams& P, float h0, float dy, float t, float& A, float& Hz);
// band radiances of the sky (no disc) along d: the per-g sky-view buffer summed per band
__device__ __forceinline__ void sky_bands(const AtmoParams& P, float3 d, float L[SPEC_NB]) {
    const SkyTap t = sky_tap(P, d);
#pragma unroll
    for (int b = 0; b < SPEC_NB; ++b) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < SPEC_GPB; ++j) s += sky_g(P, t, b * SPEC_GPB + j);
        L[b] = s;
    }
}
// The irradiance at a secondary hit (8 bands): the radiance cache's cell (statistics: one atomic pair per warp),
// else half the sky SH irradiance
__device__ void sec_cache_irr(const SParams& sp, float3 pos, float3 n, float ec[SPEC_NB]) {
    const AtmoParams& P = sp.a;
    bool have = false;
    const float3 dc = pos - sp.k.o;
    if (sp.cache.meta) {
        uint32_t slot, key;
        gi_cache_cell(g_tc.cx, g_tc.cy, g_tc.cz, pos.x, pos.y, pos.z, n.x, n.y, n.z, sqrtf(dot3(dc, dc)), &slot, &key);
        have = gi_cache_get(sp.cache, slot, key, ec);
        const unsigned m = __activemask(), nh = __popc(__ballot_sync(m, have));
        if ((int)(threadIdx.x & 31u) == __ffs(m) - 1) {
            atomicAdd(&sp.fs->cache_lookups, (unsigned long long)__popc(m));
            atomicAdd(&sp.fs->cache_hits, (unsigned long long)nh);
        }
    }
    if (!have) {
        float Y[9];
        sh_basis(n, Y);
#pragma unroll
        for (int b = 0; b < SPEC_NB; ++b) {
            float s = 0.f;
#pragma unroll
            for (int j = 0; j < SPEC_GPB; ++j) s += sh_irr(P.sh + 9 * (b * SPEC_GPB + j), Y);
            ec[b] = 0.5f * s;
        }
    }
}
// band radiances (band-integrated) -> XYZ (no 683) with the per-g colour weights of the global light's spectrum
// at the ground (the reference transport's reflection paths carry bands)
__device__ float3 bands_xyz_g(const SParams& sp, const float* L) {
    const AtmoParams& P = sp.a;
    float X = 0.f, Y = 0.f, Z = 0.f;
    for (int g = 0; g < SPEC_NG; ++g) {
        const float l = L[g / SPEC_GPB] * __ldg(&sp.fs->gglob[g]);
        X += l * P.g_cmf[g][0]; Y += l * P.g_cmf[g][1]; Z += l * P.g_cmf[g][2];
    }
    return f3(X, Y, Z);
}
// The radiance a reflection ray brings back, as XYZ (no 683) summed per g-point with the direct path's colour
// weights (the band-mean CMFs turned the CH4-shaped bands red: pink water), its hit shaded with the BRDF of the
// surface on screen (lit_radiance: GGX + energy-compensated diffuse + ambient specular): the sun with the ray's
// visibility, the radiance cache's band irradiance split over its g-points; the air path per g-point.
__device__ float3 sec_reflect_xyz(const SParams& sp, float4 o4, uint4 r, float4 h4) {
    const AtmoParams& P = sp.a;
    const float3 d = oct_unpack(r.x);
    const int kind = (int)((r.y >> 4) & 15u);
    if (kind == HK_NONE) return spec_sky_xyz(P, d.x, d.y, d.z);
    if (kind == HKX_WATER) return spec_sky_xyz(P, d.x, fmaxf(-d.y, 0.01f), d.z) * spec_fresnel(fmaxf(-d.y, 1e-3f), WATER_N);
    const float3 pos = f3(o4.x, o4.y, o4.z) + d * h4.x;
    Lit M;
    M.n = oct_unpack(r.z);
    const uint32_t mp = r.w;
    M.m0 = mp & 15u; M.m1 = (mp >> 4) & 15u; M.w1 = ((mp >> 8) & 63u) / 63.f; M.pal = (mp >> 14) & 15u;
    M.bright = 0.88f + 0.12f * ((mp >> 18) & 63u) / 63.f;
    M.steep = sat((0.9f - M.n.y) * 5.f);
    M.wet = 0.f;
    M.wb = (float)((r.y >> 16) & 255u) * (1.f / 255.f);
    M.cover = (float)((r.y >> 24) & 31u) * (1.f / 31.f);
    M.under = false; M.gi = 1; M.caus = 1.f; M.tw = 0.f; M.tamb = 0.f; M.ao = 1.f; M.depth = 0.f;
    M.tavg = kind == HK_TERRAIN || kind == HK_LOD || kind == HK_FAR;
    const float vq = (r.y & SF_VIS) ? (float)((r.y >> 13) & 7u) * (1.f / 7.f) : 0.f;
    M.vis = sp.furnace ? 0.f : vq * sat(0.5f + sp.sun.y / (2.f * P.sun_theta));
    const float2 cs = spec_columns(P, fmaxf(pos.y - g_tc.yw, 0.f), sp.sun.y);
    M.As = cs.x; M.Hs = cs.y;
    sec_cache_irr(sp, pos, M.n, M.egi);
    lit_setup(sp, M, f3(-d.x, -d.y, -d.z), sp.sun);
    float A = 0.f, Hz = 0.f;
    if (!sp.furnace) air_cols(P, fmaxf(o4.y - g_tc.yw, 0.f), d.y, h4.x, A, Hz);
    const SkyTap tap = sky_tap(P, d);
    float X = 0.f, Y = 0.f, Z = 0.f;
    for (int b = 0; b < SPEC_NB; ++b) {
        const float rho = lit_rho(sp, M, b);
#pragma unroll 1
        for (int j = 0; j < SPEC_GPB; ++j) {
            const int g = b * SPEC_GPB + j;
            float L = lit_radiance(sp, M, g, b, rho, 0.f, 1.f, 1.f);
            if (!sp.furnace) {
                const float T = __expf(-A * P.g_a[g] - Hz * P.g_b[g]);
                L = L * T + sky_g(P, tap, g) * (1.f - T);           // in-scatter ~ the sky along the ray
            }
            X += L * P.g_cmf[g][0]; Y += L * P.g_cmf[g][1]; Z += L * P.g_cmf[g][2];
        }
    }
    return f3(X, Y, Z);
}

// Radiance per band (band-integrated, W m^-2 sr^-1) arriving back along a secondary
// ray: the sky on a miss; the water surface's Fresnel sky reflection; at a surface
// rho / pi (sun NEE with the ray's visibility + the radiance cache's irradiance, or
// 0.5 x the sky SH where the cache has no cell; nee_only: the reference path adds
// the further bounces itself). Reflection rays also get the air path. surf: a
// surface was hit (rho, n, pos valid).
__device__ void sec_radiance(const SParams& sp, float4 o4, uint4 r, float4 h4, bool nee_only, float L[SPEC_NB], float rho[SPEC_NB],
                             float3& n, float3& pos, bool& surf) {
    const AtmoParams& P = sp.a;
    const float3 d = oct_unpack(r.x);
    const int kind = (int)((r.y >> 4) & 15u), type = (int)((r.y >> 1) & 3u);
    surf = false;
    if (kind == HK_NONE) { sky_bands(P, d, L); return; }
    if (kind == HKX_WATER) {
        const float F = spec_fresnel(fmaxf(-d.y, 1e-3f), WATER_N);
        sky_bands(P, f3(d.x, fmaxf(-d.y, 0.01f), d.z), L);
#pragma unroll
        for (int b = 0; b < SPEC_NB; ++b) L[b] *= F;
        return;
    }
    surf = true;
    pos = f3(o4.x, o4.y, o4.z) + d * h4.x;
    n = oct_unpack(r.z);
    const uint32_t mp = r.w;
    Lit M;
    M.m0 = mp & 15u; M.m1 = (mp >> 4) & 15u; M.w1 = ((mp >> 8) & 63u) / 63.f; M.pal = (mp >> 14) & 15u;
    M.bright = 0.88f + 0.12f * ((mp >> 18) & 63u) / 63.f;
    M.steep = sat((0.9f - n.y) * 5.f);
    M.wet = 0.f;
    M.wb = (float)((r.y >> 16) & 255u) * (1.f / 255.f);
    M.cover = (float)((r.y >> 24) & 31u) * (1.f / 31.f);   // (secondary hits: the hue bin's default pigment)
#pragma unroll
    for (int b = 0; b < SPEC_NB; ++b) rho[b] = lit_rho(sp, M, b);
    float es[SPEC_NB];
#pragma unroll
    for (int b = 0; b < SPEC_NB; ++b) es[b] = 0.f;
    const float nl = fmaxf(dot3(n, sp.sun), 0.f);
    if ((r.y & SF_VIS) && nl > 0.f && !sp.furnace) {
        const float2 cs = spec_columns(P, fmaxf(pos.y - g_tc.yw, 0.f), sp.sun.y);
        const float f = nl * sat(0.5f + sp.sun.y / (2.f * P.sun_theta)) * (float)((r.y >> 13) & 7u) * (1.f / 7.f);
        for (int g = 0; g < SPEC_NG; ++g) es[g / SPEC_GPB] += P.g_e[g] * __expf(-cs.x * P.g_a[g] - cs.y * P.g_b[g]) * f;
    }
    float ec[SPEC_NB];
    if (!nee_only) sec_cache_irr(sp, pos, n, ec);          // the radiance cache, else half the sky SH irradiance
#pragma unroll
    for (int b = 0; b < SPEC_NB; ++b) L[b] = rho[b] * 0.31830989f * (es[b] + (nee_only ? 0.f : ec[b]));
    if (type != ST_GI && !sp.furnace) {                     // reflection rays: the air path (transmittance, in-scatter ~ the sky)
        float A, Hz;
        air_cols(P, fmaxf(o4.y - g_tc.yw, 0.f), d.y, h4.x, A, Hz);
        float Ls[SPEC_NB];
        sky_bands(P, d, Ls);
#pragma unroll
        for (int b = 0; b < SPEC_NB; ++b) {
            float tw = 0.f, ew = 0.f;
#pragma unroll
            for (int j = 0; j < SPEC_GPB; ++j) {
                const int g = b * SPEC_GPB + j;
                tw += P.g_e[g] * __expf(-A * P.g_a[g] - Hz * P.g_b[g]);
                ew += P.g_e[g];
            }
            const float T = ew > 0.f ? tw / ew : 1.f;
            L[b] = L[b] * T + Ls[b] * (1.f - T);
        }
    }
}

// k_sec_shade: per grid pixel, the radiance of its slots -> GI irradiance
// E_b = pi x mean L_b over its GI rays (cosine-weighted) and the reflected
// radiance. Reference: one bounce of each path (radiance += throughput x
// (NEE, or the sky)); throughput x= albedo and a new cosine ray at a surface,
// until `last`, which gathers the paths.
__global__ void __launch_bounds__(128) k_sec_shade(const __grid_constant__ SParams sp, SecBuf sb, SecGrid G, int bounce, int last) {
    const int gi = blockIdx.x * blockDim.x + threadIdx.x, gj = blockIdx.y * blockDim.y + threadIdx.y;
    if (gi >= G.gw || gj >= G.gh) return;
    const size_t gidx = (size_t)gj * G.gw + gi;
    float E[SPEC_NB], R[SPEC_NB];
    int ngi = 0;
    bool has_r = false;
#pragma unroll
    for (int b = 0; b < SPEC_NB; ++b) { E[b] = 0.f; R[b] = 0.f; }
    for (int s = 0; s < G.K; ++s) {
        const int slot = G.base + (int)gidx * G.K + s;
        uint4 r = sb.r[slot];
        const bool is_gi = s < G.gi_n;
        float L[SPEC_NB], rho[SPEC_NB];
        float3 n = f3(0.f, 1.f, 0.f), pos = f3(0.f, 0.f, 0.f);
        bool surf;
        if (!sp.ref) {
            if (!(r.y & SF_ACTIVE)) continue;
            if (!is_gi) {                                   // the reflection: XYZ per g-point, the surface's own BRDF
                const float3 xr = sec_reflect_xyz(sp, sb.o[slot], r, sb.h[slot]);
                R[0] = xr.x; R[1] = xr.y; R[2] = xr.z;
                has_r = true;
                continue;
            }
            sec_radiance(sp, sb.o[slot], r, sb.h[slot], false, L, rho, n, pos, surf);
        } else {
            if (!(r.y & SF_PATH)) continue;
            float thr[SPEC_NB], acc[SPEC_NB];
            gi_unpack8(sb.acc[slot], acc);
            if (r.y & SF_ACTIVE) {
                gi_unpack8(sb.thr[slot], thr);
                sec_radiance(sp, sb.o[slot], r, sb.h[slot], true, L, rho, n, pos, surf);
#pragma unroll
                for (int b = 0; b < SPEC_NB; ++b) acc[b] += thr[b] * L[b];
                sb.acc[slot] = gi_pack8(acc);
                if (surf && !last) {                        // continue: cosine-weighted, throughput x albedo
#pragma unroll
                    for (int b = 0; b < SPEC_NB; ++b) thr[b] *= rho[b];
                    uint32_t rs = gi_hash32((uint32_t)slot * 7919u + sp.frame * 104729u + (uint32_t)bounce * 15485863u + 7u);
                    const float u1 = rng01(rs), u2 = rng01(rs);
                    const float3 dir = cos_dir(n, u1, u2);
                    const float4 h4 = sb.h[slot];
                    const float off = 0.04f + 5e-4f * h4.x;
                    sec_put(sb, slot, pos + n * off, GI_TMAX, dir, SF_ACTIVE | SF_PATH | ((uint32_t)ST_GI << 1) | ((uint32_t)min(bounce + 1, 7) << 9));
                    sb.thr[slot] = gi_pack8(thr);
                } else {
                    sb.r[slot] = make_uint4(r.x, (r.y & ~SF_ACTIVE), r.z, r.w);
                }
            }
            if (!last) continue;
#pragma unroll
            for (int b = 0; b < SPEC_NB; ++b) L[b] = acc[b];
        }
        if (is_gi) {
#pragma unroll
            for (int b = 0; b < SPEC_NB; ++b) E[b] += L[b];
            ++ngi;
        } else {                                            // reference reflection paths (bands) -> XYZ
            const float3 xr = bands_xyz_g(sp, L);
            R[0] = xr.x; R[1] = xr.y; R[2] = xr.z;
            has_r = true;
        }
    }
    if (sp.ref && !last) return;
    const float none[SPEC_NB] = { -1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    if (G.e_out) {
        if (ngi) {
            const float k = 3.14159265f / (float)ngi;
#pragma unroll
            for (int b = 0; b < SPEC_NB; ++b) E[b] *= k;
        }
        G.e_out[gidx] = gi_pack8(ngi ? E : none);
    }
    if (G.r_out) G.r_out[gidx] = gi_pack8(has_r ? R : none);
}

// ---- white point (CAT16) and the light at the water level ----------------------------
__device__ void mat3_inv_f(const float* m, float* o) {
    const float d = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    const float id = 1.f / d;
    o[0] = (m[4] * m[8] - m[5] * m[7]) * id; o[1] = (m[2] * m[7] - m[1] * m[8]) * id; o[2] = (m[1] * m[5] - m[2] * m[4]) * id;
    o[3] = (m[5] * m[6] - m[3] * m[8]) * id; o[4] = (m[0] * m[8] - m[2] * m[6]) * id; o[5] = (m[2] * m[3] - m[0] * m[5]) * id;
    o[6] = (m[3] * m[7] - m[4] * m[6]) * id; o[7] = (m[1] * m[6] - m[0] * m[7]) * id; o[8] = (m[0] * m[4] - m[1] * m[3]) * id;
}
// one block of SPEC_NG threads: per g-point irradiance at the water level, the
// white (global horizontal irradiance at the camera's ground) and CAT16 to D65
__global__ void k_white(const __grid_constant__ SParams sp) {
    const AtmoParams& P = sp.a;
    FrameState* fs = sp.fs;
    const int g = threadIdx.x;
    __shared__ float sw[3][SPEC_NG];
    __shared__ float seup[SPEC_NG], sglob[SPEC_NG];
    if (g < SPEC_NG) {
        float Y[9];
        sh_basis(f3(0.f, 1.f, 0.f), Y);
        const float eup = sh_irr(P.sh + 9 * g, Y);
        const float2 c0 = spec_columns(P, 0.f, sp.sun.y);
        fs->eup[g] = eup;
        seup[g] = eup;
        fs->esun0[g] = P.g_e[g] * __expf(-c0.x * P.g_a[g] - c0.y * P.g_b[g]);
        const float2 cc = spec_columns(P, sp.cam_h, sp.sun.y);
        const float e = P.g_e[g] * __expf(-cc.x * P.g_a[g] - cc.y * P.g_b[g]) * fmaxf(sp.sun.y, 0.f) + eup;
        sglob[g] = e;
        for (int c = 0; c < 3; ++c) sw[c][g] = e * P.g_cmf[g][c];
    }
    __syncthreads();
    if (g < SPEC_NG) {                                  // M4: the band's sky irradiance split over its g-points (GI is 8 bands)
        const int b0 = (g / SPEC_GPB) * SPEC_GPB;
        float s = 0.f, sg = 0.f;
        for (int j = 0; j < SPEC_GPB; ++j) { s += seup[b0 + j]; sg += sglob[b0 + j]; }
        fs->gsplit[g] = s > 1e-20f ? seup[g] / s : 1.f / (float)SPEC_GPB;
        fs->gglob[g] = sg > 1e-20f ? sglob[g] / sg : 1.f / (float)SPEC_GPB;
    }
    if (g != 0) return;
    float w[3] = { 0.f, 0.f, 0.f };
    for (int k = 0; k < SPEC_NG; ++k) for (int c = 0; c < 3; ++c) w[c] += sw[c][k];
    for (int c = 0; c < 3; ++c) fs->white[c] = w[c];
    const float M16[9] = { 0.401288f, 0.650173f, -0.051461f, -0.250268f, 1.204414f, 0.045854f, -0.002079f, 0.048952f, 0.953127f };
    const float X2R[9] = { 3.2406f, -1.5372f, -0.4986f, -0.9689f, 1.8758f, 0.0415f, 0.0557f, -0.2040f, 1.0570f };
    float Mi[9], gain[3] = { 1.f, 1.f, 1.f };
    mat3_inv_f(M16, Mi);
    const float D = sat(sp.wb);
    if (w[1] > 1e-20f) {
        const float wn[3] = { w[0] / w[1], 1.f, w[2] / w[1] }, d65[3] = { 0.95047f, 1.f, 1.08883f };
        for (int r = 0; r < 3; ++r) {
            const float rw = M16[3 * r] * wn[0] + M16[3 * r + 1] * wn[1] + M16[3 * r + 2] * wn[2];
            const float rr = M16[3 * r] * d65[0] + M16[3 * r + 1] * d65[1] + M16[3 * r + 2] * d65[2];
            gain[r] = rw > 1e-6f ? D * rr / rw + 1.f - D : 1.f;
        }
    }
    float C[9];                                         // Mi diag(gain) M16
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) C[3 * r + c] = Mi[3 * r] * gain[0] * M16[c] + Mi[3 * r + 1] * gain[1] * M16[3 + c] + Mi[3 * r + 2] * gain[2] * M16[6 + c];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) fs->cat[3 * r + c] = X2R[3 * r] * C[c] + X2R[3 * r + 1] * C[3 + c] + X2R[3 * r + 2] * C[6 + c];
}

// air path columns from altitude h0 along direction cosine dy over t metres
// (flat: the curvature drop is 0.7 m over the loaded radius)
__device__ __forceinline__ void air_cols(const AtmoParams& P, float h0, float dy, float t, float& A, float& Hz) {
    const float x1 = t * dy / P.H, x2 = t * dy / P.Hh;
    const float e1 = fabsf(x1) < 1e-4f ? 1.f - 0.5f * x1 : (1.f - __expf(-x1)) / x1;
    const float e2 = fabsf(x2) < 1e-4f ? 1.f - 0.5f * x2 : (1.f - __expf(-x2)) / x2;
    A = t * __expf(-h0 / P.H) * e1 / P.H;
    Hz = t * __expf(-h0 / P.Hh) * e2 / P.Hh;
}

// the sky behind a primary ray from the camera (no surface): scattered light,
// the stellar disc (limb darkened, through the atmosphere) and the planets
__device__ float3 sky_xyz_full(const SParams& sp, float3 d) {
    const AtmoParams& P = sp.a;
    float3 c = spec_sky_xyz(P, d.x, d.y, d.z);
    const float cd = dot3(d, sp.sun);
    const float ct = cosf(P.sun_theta);
    const RenderSky& s = sp.k.sky;
    int body = -1;
    float lit = 0.f;
    for (int k = 0; k < s.nbodies; ++k) {
        const float3 bd = v3(s.body_dir[k]);
        const float cb = dot3(d, bd);
        if (cb > cosf(s.body_ang[k])) {
            const float3 off = norm3(d - bd * cb);
            body = k;
            lit = 0.25f + 0.75f * sat(0.5f + 0.5f * dot3(off, sp.sun));
        }
    }
    if (cd <= ct && body < 0) return c;
    const float2 cv = spec_columns(P, sp.cam_h, d.y);
    if (cv.x > 1e3f) return c;
    float disc = 0.f;
    if (cd > ct) {                                      // limb darkening I ~ 1 - 0.6 (1 - mu), normalised
        const float r = acosf(fminf(cd, 1.f)) / P.sun_theta;
        const float mu = sqrtf(fmaxf(0.f, 1.f - r * r));
        disc = (0.4f + 0.6f * mu) / (0.8f * 3.14159265f * P.sun_theta * P.sun_theta);
    }
    const ShadeTab* T = sp.tab;
    for (int g = 0; g < SPEC_NG; ++g) {
        const int b = g / SPEC_GPB;
        float f = disc;
        if (body >= 0 && disc == 0.f) {                 // a planet: reflected starlight (its colour, smooth spectrum)
            const float* bc = s.body_color[body];
            const float rho = sat(bc[0] * __ldg(&T->basis[0][b]) + bc[1] * __ldg(&T->basis[1][b]) + bc[2] * __ldg(&T->basis[2][b]));
            f = rho * lit * 0.31830989f;
        }
        const float L = P.g_e[g] * __expf(-cv.x * P.g_a[g] - cv.y * P.g_b[g]) * f;
        c.x += L * P.g_cmf[g][0]; c.y += L * P.g_cmf[g][1]; c.z += L * P.g_cmf[g][2];
    }
    return c;
}

// M4: the traced reflection at internal pixel (x, y) (surface class cls) as XYZ
// (no 683; k_sec_shade stores XYZ in channels 0-2): full res, or the half-res
// samples of the same class, bilinear
__device__ bool refl_xyz(const SParams& sp, int x, int y, int cls, float3& xyz) {
    if (!sp.refl || !sp.refl_buf) return false;
    float R[SPEC_NB];
    if (sp.refl == 2) {
        gi_unpack8(sp.refl_buf[(size_t)y * sp.k.w + x], R);
        if (!(R[0] >= 0.f)) return false;
    } else {
        const float c0 = ((float)x - (float)sp.hx) * 0.5f, c1 = ((float)y - (float)sp.hy) * 0.5f;
        const int i0 = (int)floorf(c0), j0 = (int)floorf(c1);
        const float fx = c0 - (float)i0, fy = c1 - (float)j0;
        float ws = 0.f;
#pragma unroll
        for (int b = 0; b < SPEC_NB; ++b) R[b] = 0.f;
        for (int k = 0; k < 4; ++k) {
            const int a = i0 + (k & 1), b = j0 + (k >> 1);
            if (a < 0 || b < 0 || a >= sp.w2 || b >= sp.h2) continue;
            const size_t q = (size_t)b * sp.w2 + a;
            const uint2 hq = sp.hg[q];
            if (!(hq.y & GG_REFL) || (int)((hq.y >> 24) & 15u) != cls) continue;
            float Rq[SPEC_NB];
            gi_unpack8(sp.refl_buf[q], Rq);
            if (!(Rq[0] >= 0.f)) continue;
            const float w = ((k & 1) ? fx : 1.f - fx) * ((k >> 1) ? fy : 1.f - fy) + 1e-4f;
#pragma unroll
            for (int bb = 0; bb < SPEC_NB; ++bb) R[bb] += w * Rq[bb];
            ws += w;
        }
        if (ws <= 0.f) return false;
#pragma unroll
        for (int b = 0; b < SPEC_NB; ++b) R[b] /= ws;
    }
    xyz = f3(R[0], R[1], R[2]);
    return true;
}
// band irradiances (8 bands) -> XYZ (no 683) of a white Lambertian surface lit by them (debug view)
__device__ float3 bands_xyz_lambert(const SParams& sp, const float* E) {
    const ShadeTab* T = sp.tab;
    float X = 0.f, Y = 0.f, Z = 0.f;
    for (int b = 0; b < SPEC_NB; ++b) {
        const float l = E[b] * 0.31830989f / (float)SPEC_BW;
        X += l * __ldg(&T->cmf_band[b][0]); Y += l * __ldg(&T->cmf_band[b][1]); Z += l * __ldg(&T->cmf_band[b][2]);
    }
    return f3(X, Y, Z);
}
// High: caustics on a sunlit point under water (spec 4.5). A refracted sun ray
// leaves the surface displaced by c grad(eta) at depth D (c = D (1 - 1/n), D the
// path length along the refracted sun), so the beam irradiance scales with
// 1 / |det(I + c H)|, H the Hessian of the Gerstner heights where the point's sun
// ray entered the water. The spec's 1 / (1 + c Laplacian) is the trace of it; the
// full determinant draws the caustic network as lines instead of a lattice of
// dots. Waves much shorter than the depth are blurred out, and so are waves whose
// pattern is finer than about 4 pixel footprints on the ground (foot, m: no
// moire in the distance, where the caustics average to 1). Clamped to [0.2, 5].
__device__ float caustic(const SParams& sp, float3 pos, float depth, float foot) {
    const float mur = fmaxf(sp.sun_r.y, 0.05f);
    const float D = depth / mur;
    const float3 ps = pos + sp.sun_r * D;
    const float ax = ps.x + 32.f * (float)(g_tc.cx & 1023), az = ps.z + 32.f * (float)(g_tc.cz & 1023);
    const int32_t bx = g_tc.cx >> 10, bz = g_tc.cz >> 10;
    const float pn = 2.6f * vnoise(ax * (1.f / 64.f), az * (1.f / 64.f), bx * 512, bz * 512) +
                     1.3f * vnoise(ax * (1.f / 16.f), az * (1.f / 16.f), bx * 2048 + 7919, bz * 2048);
    float hxx = 0.f, hzz = 0.f, hxz = 0.f;               // eta = sum A sin(phase), A = s / k
#pragma unroll 4
    for (int i = 0; i < WAVES; ++i) {
        const float f = sat(1.5f - 0.02f * D * sp.w.k[i]) * sat(2.f - 1.2732395f * foot * sp.w.k[i]);
        if (f <= 0.f) break;                             // sorted by k: the shorter ones are blurred out too
        float ph = sp.w.kx[i] * ps.x + sp.w.kz[i] * ps.z + sp.w.off[i] + pn * sp.w.pm[i];
        ph -= 6.2831853f * rintf(ph * 0.15915494f);
        const float a = sp.w.s[i] * f / sp.w.k[i] * __sinf(ph);
        hxx -= a * sp.w.kx[i] * sp.w.kx[i];
        hzz -= a * sp.w.kz[i] * sp.w.kz[i];
        hxz -= a * sp.w.kx[i] * sp.w.kz[i];
    }
    const float c = D * (1.f - 1.f / WATER_N);
    const float det = (1.f + c * hxx) * (1.f + c * hzz) - c * c * hxz * hxz;
    return fminf(fmaxf(1.f / fmaxf(fabsf(det), 1e-3f), 0.2f), 5.f);
}

// k_compose: spectral shading of the G-buffer -> linear sRGB (white-balanced, unexposed)
__global__ void __launch_bounds__(128, 4) k_compose(const __grid_constant__ SParams sp, GBuf gb) {
    const KParams& p = sp.k;
    const AtmoParams& P = sp.a;
    const ShadeTab* T = sp.tab;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    const size_t i = (size_t)y * p.w + x;
    const float4 a = gb.a[i];
    const uint2 b = gb.b[i];
    const float3 d = sp_dir(sp, x, y);
    const bool hit = (b.y & GF_HIT) != 0;
    const int wm = (int)(((hit ? b.y : b.x) >> 24) & 7u);
    const float tw = a.y;
    const int kind = hit ? (int)((b.y >> 16) & 15u) : HK_NONE;
    const int dbg = sp.debug;

    // debug views shown as display values (no exposure / AgX)
    if (dbg == RD_NORMALS || dbg == RD_ALBEDO || dbg == RD_DEV || dbg == RD_AO || dbg == RD_VIS || dbg == RD_GI || dbg == RD_REFL ||
        dbg == RD_PALETTE) {
        float3 c = f3(0.05f, 0.05f, 0.1f);
        if ((dbg == RD_GI && sp.gi) || dbg == RD_REFL) {
            // M4: luminance of the GI irradiance (of the reflected radiance x pi) over the sky's horizontal
            // irradiance at the water level: 0.5 grey = open flat ground; blue: no value at this pixel
            float yr = 0.f;
            for (int g = 0; g < SPEC_NG; ++g) yr += __ldg(&sp.fs->eup[g]) * P.g_cmf[g][1];
            float3 s = f3(0.f, 0.f, 0.f);
            bool have = false;
            if (dbg == RD_GI) {
                float E[SPEC_NB];
                if (hit && sp.efull && !(b.y & GF_UNDER)) {
                    gi_unpack8(sp.efull[i], E);
                    if (E[0] >= 0.f) { s = bands_xyz_lambert(sp, E) * 3.14159265f; have = true; }
                }
            } else {
                have = refl_xyz(sp, x, y, (wm == WM_STRAIGHT || wm == WM_REFR) ? GC_WATER : kind_class(kind), s);
                s = s * 3.14159265f;
            }
            const float v = 0.5f * s.y / fmaxf(yr, 1e-20f);
            c = have ? f3(v, v, v) : f3(0.05f, 0.05f, 0.3f);
        } else if (hit) {
            const float3 n = oct_unpack(__float_as_uint(a.z));
            const uint32_t mp = b.x;
            if (dbg == RD_NORMALS) c = n * 0.5f + f3(0.5f, 0.5f, 0.5f);
            else if (dbg == RD_PALETTE) {                  // M5: the classic colours with the old HSV palette tint
                const int m0 = mp & 15u, m1 = (mp >> 4) & 15u, pal = (mp >> 14) & 15u;
                const float w1 = ((mp >> 8) & 63u) / 63.f, br = 0.88f + 0.12f * ((mp >> 18) & 63u) / 63.f;
                c = mix3(palette_rgb(sp, m0, pal), palette_rgb(sp, m1, pal), w1) * br;   // linear: k_tonemap encodes it
            } else if (dbg == RD_ALBEDO) {
                Lit L;
                L.m0 = mp & 15u; L.m1 = (mp >> 4) & 15u; L.w1 = ((mp >> 8) & 63u) / 63.f; L.pal = (mp >> 14) & 15u;
                L.bright = 0.88f + 0.12f * ((mp >> 18) & 63u) / 63.f; L.steep = sat((0.9f - n.y) * 5.f); L.wet = 0.f;
                L.wb = (kind == HK_LOD || kind == HK_FAR) ? (float)((b.y >> 8) & 255u) * (1.f / 255.f) : 0.f;
                L.cover = (float)((b.y >> 27) & 31u) * (1.f / 31.f);
                L.pal |= (int)(b.y & 0xF0u) | (int)((b.y >> 12) & 0xF00u);   // version 2: pigment level, Q band width bin
                float X = 0.f, Y = 0.f, Z = 0.f, W = 0.f;
                for (int k = 0; k < SPEC_NB; ++k) {
                    const float r = kind == HK_FAUNA ? skin_rho(sp, L.pal, L.bright, (b.y & 0x100u) != 0, k) : lit_rho(sp, L, k);
                    X += r * __ldg(&T->cmf_band[k][0]); Y += r * __ldg(&T->cmf_band[k][1]); Z += r * __ldg(&T->cmf_band[k][2]);
                    W += __ldg(&T->cmf_band[k][1]);
                }
                X /= W; Y /= W; Z /= W;
                c = f3(T->dbg_rgb[0] * X + T->dbg_rgb[1] * Y + T->dbg_rgb[2] * Z, T->dbg_rgb[3] * X + T->dbg_rgb[4] * Y + T->dbg_rgb[5] * Z,
                       T->dbg_rgb[6] * X + T->dbg_rgb[7] * Y + T->dbg_rgb[8] * Z);
            } else if (dbg == RD_DEV) {
                const float dv = (float)(int8_t)((b.y >> 8) & 255u) * (1.f / GME_DEV_UNIT);
                c = kind == HK_TERRAIN ? f3(sat(dv), 0.25f, sat(-dv)) : f3(0.1f, 0.1f, 0.1f);
            } else if (dbg == RD_VIS) {
                const float v = sp.vis_mode == 1 && sp.vis ? __half2float(sp.vis[i])
                              : (sp.vis_mode == 2 && sp.vsmp ? (float)sp.vsmp[i] / 255.f : ((b.y & GF_SHADOW) ? 0.f : 1.f));
                c = f3(v, v, v);
            } else {
                c = f3((mp >> 24) / 255.f, (mp >> 24) / 255.f, (mp >> 24) / 255.f);   // the heightfield AO
            }
        }
        sp.color[i] = make_float4(c.x, c.y, c.z, -1.f);
        return;
    }

    const bool cam_under = wm >= WM_UNDER;
    const float yw = g_tc.yw;
    const float zc = fmaxf(yw - p.o.y, 0.f);
    float3 d2 = d, nw = f3(0.f, 1.f, 0.f);
    float a2w = 0.f, fv = 0.f, glint = 0.f;
    if (tw >= 0.f && wm != WM_NONE && wm != WM_UNDER) {   // the water surface on the path
        const float3 pw = p.o + d * tw;
        nw = water_normal(sp, pw.x, pw.z, water_foot(sp, d, tw), &a2w);
        if (path_split(wm)) d2 = water_second(sp, d, nw, wm);
        if (!cam_under) {                                  // seen from above: Fresnel, glint
            float3 nn = dot3(nw, d) < -0.02f ? nw : f3(0.f, 1.f, 0.f);
            const float cv = fmaxf(-dot3(nn, d), 1e-3f);
            fv = spec_fresnel(cv, WATER_N);
            if (sp.sun.y > 0.f) {
                const float3 v = f3(-d.x, -d.y, -d.z), hv = norm3(sp.sun + v);
                const float nl = fmaxf(dot3(nn, sp.sun), 0.f), aw = sqrtf(a2w);
                glint = spec_ggx_d(fmaxf(dot3(nn, hv), 0.f), aw) * spec_ggx_vis(nl, cv, aw) * spec_fresnel(fmaxf(dot3(v, hv), 0.f), WATER_N) * nl *
                        sat(0.5f + sp.sun.y / (2.f * P.sun_theta));
            }
        } else {                                           // seen from below: transmission into the water
            fv = spec_fresnel(fmaxf(dot3(nw, d), 1e-3f), 1.f / WATER_N);
        }
    }
    float X = 0.f, Y = 0.f, Z = 0.f;                   // spectral part (no 683)
    float3 xadd = f3(0.f, 0.f, 0.f);                    // sky-view (XYZ) terms on the camera side of the air path
    float t_air = -1.f;                                  // air path from the camera (camera above water)
    if (!cam_under && !sp.furnace) t_air = wm == WM_NONE ? (hit ? a.x : -1.f) : tw;

    if (!(hit && kind != HK_DIST) && wm == WM_NONE) {   // the sky (and below the horizon beyond the far field: its ground)
        const float3 s = sky_xyz_full(sp, d);
        X = s.x; Y = s.y; Z = s.z;
    } else {
        Lit L;
        L.under = false; L.wet = 0.f; L.wb = 0.f; L.gi = 0; L.caus = 1.f; L.tavg = 0;
        L.cover = 0.f; L.tw = 0.f; L.tamb = 0.f; L.sheen = 0.f;
        float3 vdir = d2;
        float rfl_w = 0.f;                                 // weight of the traced reflection (wet film, ice)
        int leaf_h = -1;                                   // M5: hue bin of a transmitting leaf (crown, kelp blade), -1 none
        if (hit) {
            const float3 pos = (path_split(wm) && tw >= 0.f && a.x >= tw) ? p.o + d * tw + d2 * (a.x - tw) : p.o + d * a.x;
            L.n = oct_unpack(__float_as_uint(a.z));
            const uint32_t mp = b.x;
            L.m0 = mp & 15u; L.m1 = (mp >> 4) & 15u; L.w1 = ((mp >> 8) & 63u) / 63.f; L.pal = (mp >> 14) & 15u;
            L.bright = 0.88f + 0.12f * ((mp >> 18) & 63u) / 63.f;
            L.ao = (float)(mp >> 24) / 255.f;
            L.under = (b.y & GF_UNDER) != 0;
            L.wb = (kind == HK_LOD || kind == HK_FAR) ? (float)((b.y >> 8) & 255u) * (1.f / 255.f) : 0.f;
            L.cover = (float)((b.y >> 27) & 31u) * (1.f / 31.f);
            L.pal |= (int)(b.y & 0xF0u) | (int)((b.y >> 12) & 0xF00u);   // version 2: the cell's pigment level, Q band width bin
            // M3: beyond the loaded radius the ground drops with the curvature and its vertical tilts by r / R_p
            float ya = pos.y, mus = sp.sun.y;
            if (g_tc.far_on) {
                const float dx = pos.x - g_tc.camx, dz = pos.z - g_tc.camz, r2 = dx * dx + dz * dz;
                if (r2 > g_tc.drop_r2) {
                    ya += g_tc.c2 * (r2 - g_tc.drop_r2);
                    const float3 up = norm3(f3(2.f * g_tc.c2 * dx, 1.f, 2.f * g_tc.c2 * dz));
                    mus = dot3(sp.sun, up);
                }
            }
            L.depth = fmaxf(yw - ya, 0.f);
            // M4: the filtered sun visibility (Medium / High), binary samples (reference), else the shadow bit
            const float vs = sp.vis_mode == 1 ? __half2float(sp.vis[i]) : (sp.vis_mode == 2 ? (float)sp.vsmp[i] * (1.f / 255.f)
                                                                                           : ((b.y & GF_SHADOW) ? 0.f : 1.f));
            L.vis = vs * sat(0.5f + mus / (2.f * P.sun_theta));
            L.steep = sat((0.9f - L.n.y) * 5.f);
            L.wet = (!L.under && !sp.furnace && (kind == HK_TERRAIN || kind == HK_VOXEL || kind == HK_LOD || kind == HK_FAR))
                        ? smooth01(yw + 1.5f, yw, ya) : 0.f;
            const float2 cs = spec_columns(P, L.under ? 0.f : fmaxf(ya - yw, 0.f), mus);
            L.As = cs.x; L.Hs = cs.y;
            sh_basis(L.n, L.Y);
            // M4: the GI irradiance of this pixel replaces the sky SH x AO ambient (M5: a crown keeps its clumps' occlusion:
            // its GI rays start outside the own crown)
            if (sp.gi && !L.under && sp.efull) {
                gi_unpack8(sp.efull[i], L.egi);
                if (L.egi[0] >= 0.f) { L.gi = 1; if (kind != HK_CROWN) L.ao = 1.f; }
            }
            L.tavg = (b.y & GF_TERRAIN) ? 1 : 0;           // terrain: a view-independent ambient specular (lit_setup)
            // M5 leaves: a crown is a leaf-area medium for its light: the sun reaches a crown point through the own
            // crown (k_surface: e^-(sigma chord)), reflected on the sunny side, transmitted (tau / pi, spec 4.6) where
            // the sun is behind; kelp blades transmit (two-sided, cosine) under water
            if (!sp.furnace && kind == HK_CROWN) {
                L.vis *= (float)((b.y >> 8) & 255u) * (1.f / 255.f);
                L.tw = sat(-2.f * dot3(L.n, sp.sun));
                leaf_h = hue_bin(L.pal);
            } else if (!sp.furnace && kind == HK_KELP) {
                L.tw = sat(-dot3(L.n, L.under ? sp.sun_r : sp.sun));
                L.tamb = 1.f;
                leaf_h = hue_bin(L.pal);
            }
            if (sp.caustics && L.under && sp.sun.y > 0.f && L.vis > 0.f)
                L.caus = caustic(sp, pos, L.depth, a.x * sp.pix / fmaxf(fabsf(dot3(vdir, L.n)), 0.1f));
            // (a creature's armour, bit 9 of b.y: glossier)
            lit_setup(sp, L, f3(-vdir.x, -vdir.y, -vdir.z), L.under ? sp.sun_r : sp.sun, (kind == HK_FAUNA && (b.y & 0x200u)) ? 0.4f : 1.f);
            const float3 v = f3(-vdir.x, -vdir.y, -vdir.z);
            float3 rx;
            const bool traced = (L.wet > 0.f || kind == HK_ICE) && refl_xyz(sp, x, y, kind_class(kind), rx);
            if (L.wet > 0.f) {                         // the wet film reflects the sky (M4: the traced reflection)
                const float fw = L.wet * spec_schlick(0.02f, fmaxf(dot3(L.n, v), 0.f));
                if (traced) {
                    xadd = xadd + rx * fw;
                } else {
                    float3 r = L.n * (2.f * dot3(L.n, v)) - v;
                    r.y = fmaxf(r.y, 0.01f);
                    xadd = xadd + spec_sky_xyz(P, r.x, r.y, r.z) * (fw * L.ao);
                }
            }
            if (kind == HK_ICE && traced) rfl_w = spec_fresnel(fmaxf(dot3(L.n, v), 1e-3f), 1.31f);
            if (rfl_w > 0.f) xadd = xadd + rx * rfl_w;
        }
        // air path of the camera (above water) and of the far end seen through the window (below)
        float Av = 0.f, Hv = 0.f, A2 = 0.f, H2 = 0.f;
        if (!cam_under && t_air > 0.f) air_cols(P, sp.cam_h, d.y, t_air, Av, Hv);
        if (wm == WM_UNDER_OUT && hit) air_cols(P, 0.f, d2.y, a.x - tw, A2, H2);
        SkyTap tap;
        const bool sky_below = !hit && wm == WM_UNDER_OUT;   // the sky through the Snell window
        if (sky_below) tap = sky_tap(P, d2);
        // ... and the stellar disc in it (limb darkened, through the atmosphere from the surface up), like a primary
        // sky ray above the water; the waves' normals break it into glints
        float disc_f = 0.f;
        float2 dcol = make_float2(0.f, 0.f);
        if (sky_below && !sp.furnace && sp.sun.y > 0.f) {
            const float cd = dot3(d2, sp.sun);
            if (cd > cosf(P.sun_theta)) {
                const float rr = acosf(fminf(cd, 1.f)) / P.sun_theta, mu = sqrtf(fmaxf(0.f, 1.f - rr * rr));
                disc_f = (0.4f + 0.6f * mu) / (0.8f * 3.14159265f * P.sun_theta * P.sun_theta);
                dcol = spec_columns(P, 0.f, d2.y);
            }
        }
        float tx = 0.f, ty = 0.f, tz = 0.f, wx = 0.f, wy = 0.f, wz = 0.f;
        const int dbg_band = (dbg >= RD_BAND0 && dbg < RD_BAND0 + SPEC_NB) ? dbg - RD_BAND0 : -1;
        float band_l = 0.f;
        for (int bd = 0; bd < SPEC_NB; ++bd) {
            float rho = 0.f, tau = 0.f, twd = 1.f, twa = 1.f;
            if (hit) {
                rho = kind == HK_FAUNA ? skin_rho(sp, L.pal, L.bright, (b.y & 0x100u) != 0, bd) : lit_rho(sp, L, bd);
                if (leaf_h >= 0)
                    tau = ((L.pal & 0xF0) ? 0.42f * (1.f - __ldg(&T->pig_A2[leaf_h][((L.pal >> 4) & 15) - 1][(L.pal >> 8) & 15][bd]))
                                          : __ldg(&T->leaf_tau[leaf_h][bd])) * L.bright;
                if (L.under) {
                    const float bs = __ldg(&T->water_b[bd]), mur = fmaxf(sp.sun_r.y, 0.05f);
                    twd = water_T(T, bd, L.depth / mur) * __expf(-0.2f * bs * L.depth / mur);
                    twa = water_T(T, bd, 1.2f * L.depth) * __expf(-0.24f * bs * L.depth);
                }
            }
            // the band's water segments (the per-g part is the light that enters them)
            WSeg s1, s2;
            if (wm == WM_STRAIGHT || wm == WM_REFR || wm == WM_UNDER_TIR) s1 = water_seg_band(sp, bd, 0.f, d2, hit ? a.x - tw : -1.f);
            else if (wm == WM_UNDER) s1 = water_seg_band(sp, bd, zc, d, hit ? a.x : -1.f);
            else if (wm == WM_UNDER_OUT) s1 = water_seg_band(sp, bd, zc, d, tw);
            if (wm == WM_UNDER_TIR) s2 = water_seg_band(sp, bd, zc, d, tw);
#pragma unroll 1
            for (int j = 0; j < SPEC_GPB; ++j) {          // not unrolled: ~19 % faster (smaller code, fewer live values)
                const int g = bd * SPEC_GPB + j;
                float Lg = 0.f;
                if (hit) Lg = lit_radiance(sp, L, g, bd, rho, tau, twd, twa);
                else if (sky_below) {
                    Lg = sky_g(P, tap, g);
                    if (disc_f > 0.f) Lg += P.g_e[g] * __expf(-dcol.x * P.g_a[g] - dcol.y * P.g_b[g]) * disc_f;
                }
                const float ew0 = __ldg(&sp.fs->esun0[g]) * sp.sun_t, ed0 = 0.934f * __ldg(&sp.fs->eup[g]);
                switch (wm) {
                case WM_STRAIGHT:
                case WM_REFR:
                    Lg = water_apply(s1, Lg, ew0, ed0);
                    Lg = Lg * (1.f - fv) * (1.f / (WATER_N * WATER_N)) + __ldg(&sp.fs->esun0[g]) * glint;
                    break;
                case WM_UNDER:
                    Lg = water_apply(s1, Lg, ew0, ed0);
                    break;
                case WM_UNDER_TIR:
                    Lg = water_apply(s2, water_apply(s1, Lg, ew0, ed0), ew0, ed0);
                    break;
                case WM_UNDER_OUT:
                    if (hit) Lg *= __expf(-A2 * P.g_a[g] - H2 * P.g_b[g]);
                    Lg = water_apply(s1, Lg * (1.f - fv) * (WATER_N * WATER_N), ew0, ed0);
                    break;
                default:
                    break;
                }
                if (!cam_under) {
                    const float ta = __expf(-Av * P.g_a[g] - Hv * P.g_b[g]);
                    Lg *= ta;
                    const float e = P.g_e[g];
                    tx += ta * e * P.g_cmf[g][0]; ty += ta * e * P.g_cmf[g][1]; tz += ta * e * P.g_cmf[g][2];
                    wx += e * P.g_cmf[g][0]; wy += e * P.g_cmf[g][1]; wz += e * P.g_cmf[g][2];
                }
                if (bd == dbg_band) band_l += Lg;
                X += Lg * P.g_cmf[g][0]; Y += Lg * P.g_cmf[g][1]; Z += Lg * P.g_cmf[g][2];
            }
        }
        if (!cam_under) {
            if (wm == WM_STRAIGHT || wm == WM_REFR) {          // the sky in the water surface (M4: the traced reflection)
                float3 s;
                if (!refl_xyz(sp, x, y, GC_WATER, s)) {
                    float3 r = d - nw * (2.f * dot3(d, nw));
                    const float dn = r.y < 0.f ? spec_fresnel(fminf(-r.y, 1.f), WATER_N) : 1.f;   // into the next wave:
                    r.y = fmaxf(fabsf(r.y), 0.01f);                // reflected once more (as k_sec_trace's HKX_WATER)
                    r = norm3(r);
                    s = spec_sky_xyz(P, r.x, r.y, r.z) * dn;
                }
                xadd = xadd + s * fv;
            }
            X += xadd.x * (wx > 0.f ? tx / wx : 1.f); Y += xadd.y * (wy > 0.f ? ty / wy : 1.f); Z += xadd.z * (wz > 0.f ? tz / wz : 1.f);
            if (t_air > 0.f) {                                // aerial perspective (in-scatter along the air path)
                const float3 s = spec_aerial_xyz(P, ((float)x + 0.5f + sp.jx) / p.w, ((float)y + 0.5f + sp.jy) / p.h, t_air);
                X += s.x; Y += s.y; Z += s.z;
            }
            // edge of the drawn world fades into the sky-view: M3 the far field's last 2 km (spec 3.8: "beyond 30 km
            // the terrain blends into the sky-view LUT's below-horizon radiance"); without it the loaded radius (M2)
            float f = 0.f;
            if (sp.furnace) {
            } else if (g_tc.far_on) {
                const float hl = sqrtf(d.x * d.x + d.z * d.z);
                const float re = (wm != WM_NONE && tw >= 0.f) ? tw * hl : ((kind == HK_FAR && hit) ? a.x * hl : -1.f);
                f = re > 0.f ? smooth01(g_tc.r2 - 2000.f, g_tc.r2, re) : 0.f;
            } else if (t_air > 0.f) {
                f = smooth01(0.85f * p.max_dist, p.max_dist, t_air);
            }
            if (f > 0.f) {
                const float3 s = spec_sky_xyz(P, d.x, d.y, d.z);
                X += (s.x - X) * f; Y += (s.y - Y) * f; Z += (s.z - Z) * f;
            }
        }
        if (dbg_band >= 0) { const float v = band_l * (106.86f / SPEC_NB) / (float)SPEC_BW; X = v * 0.95047f; Y = v; Z = v * 1.08883f; }
    }
    X *= 683.f; Y *= 683.f; Z *= 683.f;
    const float* M = sp.fs->cat;
    float r = __ldg(M) * X + __ldg(M + 1) * Y + __ldg(M + 2) * Z;
    float g = __ldg(M + 3) * X + __ldg(M + 4) * Y + __ldg(M + 5) * Z;
    float bl = __ldg(M + 6) * X + __ldg(M + 7) * Y + __ldg(M + 8) * Z;
    const float mn = fminf(r, fminf(g, bl));
    if (mn < 0.f) {                                     // out of gamut: desaturate toward the luminance
        const float yl = 0.2126f * r + 0.7152f * g + 0.0722f * bl;
        if (yl <= 0.f) { r = g = bl = 0.f; }
        else { const float k = yl / (yl - mn); r = yl + (r - yl) * k; g = yl + (g - yl) * k; bl = yl + (bl - yl) * k; }
    }
    if (isnan(r) || isnan(g) || isnan(bl) || isinf(r) || isinf(g) || isinf(bl)) {
        if (sp.dbg) atomicAdd(sp.dbg, 1ull);
        if (sp.nan) atomicAdd(sp.nan, 1u);
        r = g = bl = 0.f;
    }
    sp.color[i] = make_float4(r, g, bl, 1.f);
}

// ---- exposure (spec 4.8) --------------------------------------------------------------
#define HIST_BINS 64
__global__ void __launch_bounds__(256) k_lum_hist(const __grid_constant__ SParams sp) {
    __shared__ unsigned int h[HIST_BINS];
    const int t = threadIdx.y * blockDim.x + threadIdx.x;
    if (t < HIST_BINS) h[t] = 0u;
    __syncthreads();
    const int x = 4 * (blockIdx.x * blockDim.x + threadIdx.x), y = 4 * (blockIdx.y * blockDim.y + threadIdx.y);
    if (x < sp.k.w && y < sp.k.h) {
        const float4 c = sp.color[(size_t)y * sp.k.w + x];
        if (c.w > 0.f) {
            const float yl = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
            const int bin = yl > 0.f ? min(HIST_BINS - 1, max(0, (int)floorf((log2f(yl) + 8.f) * (HIST_BINS / 24.f)))) : 0;
            atomicAdd(&h[bin], 1u);
        }
    }
    __syncthreads();
    if (t < HIST_BINS && h[t]) atomicAdd(&sp.hist[t], h[t]);
}
// mean log2 luminance of the 5-95 % range -> k = 0.18 / Y_avg (or the fixed EV),
// adapted in log space: tau 0.5 s while the image darkens, 2 s while it brightens
__global__ void k_exposure(const __grid_constant__ SParams sp) {
    if (threadIdx.x != 0) return;
    FrameState* fs = sp.fs;
    float n = 0.f;
    for (int k = 0; k < HIST_BINS; ++k) n += (float)sp.hist[k];
    float target = fs->log2k;
    if (n > 0.f) {
        const float lo = 0.05f * n, hi = 0.95f * n;
        float acc = 0.f, s = 0.f, w = 0.f;
        for (int k = 0; k < HIST_BINS; ++k) {
            const float c = (float)sp.hist[k];
            const float ov = fmaxf(0.f, fminf(acc + c, hi) - fmaxf(acc, lo));
            s += ov * (-8.f + (k + 0.5f) * (24.f / HIST_BINS));
            w += ov;
            acc += c;
        }
        target = log2f(0.18f) - (w > 0.f ? s / w : 0.f);
    }
    if (!isnan(sp.ev)) target = -log2f(1.2f) - sp.ev;
    float cur = fs->log2k;
    if (!fs->init || !(sp.dt > 0.f) || !isnan(sp.ev)) cur = target;
    else cur += (target - cur) * (1.f - __expf(-sp.dt / (target < cur ? 0.5f : 2.0f)));
    fs->k_prev = fs->init ? fs->k : 0.f;                // M4: the exposure the TAA history was resolved with
    fs->log2k = cur;
    fs->k = exp2f(cur);
    fs->ev100 = -cur - log2f(1.2f);
    fs->init = 1;
    for (int k = 0; k < HIST_BINS; ++k) sp.hist[k] = 0u;
}
// AgX, sRGB OETF, +-0.5 LSB triangular dither, crosshair (inverted) -> BGRA, from
// the resolved output-resolution image (M4: TAA / TAAU or the reference
// accumulation, already exposed; w < 0: a debug view's display value).
// RD_AGE tints the image by the TAA history age.
__global__ void __launch_bounds__(128) k_tonemap(const __grid_constant__ SParams sp) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= sp.ow || y >= sp.oh) return;
    const float4 c = taa_ld(sp.taa, (size_t)y * sp.ow + x);
    float r = c.x, g = c.y, b = c.z;
    if (c.w >= 0.f) spec_agx(&r, &g, &b);
    if (sp.debug == RD_AGE && c.w >= 0.f) {             // red 1 frame, yellow 2, green 3-7, blue >= 8 (on the image's grey)
        const float l = 0.25f + 0.5f * sat(0.2126f * r + 0.7152f * g + 0.0722f * b);
        const float3 t = c.w < 1.5f ? f3(1.f, 0.f, 0.f) : (c.w < 2.5f ? f3(1.f, 1.f, 0.f) : (c.w < 7.5f ? f3(0.f, 1.f, 0.f) : f3(0.2f, 0.4f, 1.f)));
        r = l * t.x; g = l * t.y; b = l * t.z;
    }
    r = spec_srgb_oetf(r); g = spec_srgb_oetf(g); b = spec_srgb_oetf(b);
    const int dx = x - sp.ow / 2, dy = y - sp.oh / 2;
    if ((dy == 0 && abs(dx) < 9 && abs(dx) > 2) || (dx == 0 && abs(dy) < 9 && abs(dy) > 2)) { r = 1.f - r; g = 1.f - g; b = 1.f - b; }
    const float n = 0.5f * (ign((float)x, (float)y) + ign((float)x + 5.588238f, (float)y + 3.1415927f) - 1.f);
    const uint32_t R = (uint32_t)min(255, max(0, (int)floorf(r * 255.f + 0.5f + n)));
    const uint32_t G = (uint32_t)min(255, max(0, (int)floorf(g * 255.f + 0.5f + n)));
    const uint32_t B = (uint32_t)min(255, max(0, (int)floorf(b * 255.f + 0.5f + n)));
    sp.outp[(size_t)y * sp.ow + x] = 0xFF000000u | (R << 16) | (G << 8) | B;
}

// ---- probe (key L): irradiance at the crosshair hit ---------------------------------------
__global__ void k_probe(const __grid_constant__ SParams sp, GBuf gb) {
    if (threadIdx.x != 0) return;
    const KParams& p = sp.k;
    const AtmoParams& P = sp.a;
    const ShadeTab* T = sp.tab;
    ProbeOut* o = sp.probe;
    const int x = p.w / 2, y = p.h / 2;
    const size_t i = (size_t)y * p.w + x;
    const float4 a = gb.a[i];
    const uint2 b = gb.b[i];
    o->valid = (b.y & GF_HIT) != 0;
    o->sun_y = sp.sun.y;
    if (!o->valid) return;
    const int wm = (int)((b.y >> 24) & 7u);
    const float3 d = sp_dir(sp, x, y);
    float3 d2;
    const float3 pos = path_pos(sp, d, a.x, a.y, wm, &d2);
    const float3 n = oct_unpack(__float_as_uint(a.z));
    const bool under = (b.y & GF_UNDER) != 0;
    const float yw = g_tc.yw;
    const float depth = under ? fmaxf(yw - pos.y, 0.f) : 0.f;
    const float vs = sp.vis_mode == 1 ? __half2float(sp.vis[i]) : (sp.vis_mode == 2 ? (float)sp.vsmp[i] / 255.f : ((b.y & GF_SHADOW) ? 0.f : 1.f));
    const float vis = vs * sat(0.5f + sp.sun.y / (2.f * P.sun_theta));
    float egi[SPEC_NB];                                  // M4: the GI irradiance of the pixel, if it has one
    bool gi = false;
    if (sp.gi && sp.efull && !under) { gi_unpack8(sp.efull[i], egi); gi = egi[0] >= 0.f; }
    const float ao = (float)(b.x >> 24) / 255.f;
    o->kind = (int)((b.y >> 16) & 15u);
    o->mat = (int)(b.x & 15u);
    o->mat1 = (int)((b.x >> 4) & 15u);
    o->w1 = (float)((b.x >> 8) & 63u) / 63.f;
    o->pal = (int)((b.x >> 14) & 15u);
    o->cover = (float)((b.y >> 27) & 31u) / 31.f;
    o->pig = (int)((b.y >> 4) & 15u);
    o->qw = (int)((b.y >> 20) & 15u);
    o->tsun = o->kind == HK_CROWN ? (float)((b.y >> 8) & 255u) / 255.f : 1.f;   // M5: the sun through the own crown
    o->pos[0] = pos.x + 32.f * g_tc.cx; o->pos[1] = pos.y + 32.f * g_tc.cy; o->pos[2] = pos.z + 32.f * g_tc.cz;
    o->n[0] = n.x; o->n[1] = n.y; o->n[2] = n.z;
    o->depth = depth; o->vis = vis; o->ao = ao;
    float Y[9], Yu[9];
    sh_basis(n, Y);
    sh_basis(f3(0.f, 1.f, 0.f), Yu);
    const float hpos = under ? 0.f : fmaxf(pos.y - yw, 0.f);
    const float2 cs = spec_columns(P, hpos, sp.sun.y);
    // M5 leaves: a crown point gets the sun through its own crown, on either side (reflected / transmitted);
    // a kelp blade on either face
    const float nls = dot3(n, under ? sp.sun_r : sp.sun);
    const float nl = o->kind == HK_CROWN ? fmaxf(nls, sat(-2.f * nls)) : (o->kind == HK_KELP ? fabsf(nls) : fmaxf(nls, 0.f));
    const float visx = vis * o->tsun;
    for (int bd = 0; bd < SPEC_NB; ++bd) {
        float es = 0.f, ek = 0.f, eh = 0.f, twd = 1.f, twa = 1.f;
        if (under) {
            const float bs = T->water_b[bd], mur = fmaxf(sp.sun_r.y, 0.05f);
            twd = water_T(T, bd, depth / mur) * __expf(-0.2f * bs * depth / mur);
            twa = water_T(T, bd, 1.2f * depth) * __expf(-0.24f * bs * depth);
        }
        for (int j = 0; j < SPEC_GPB; ++j) {
            const int g = bd * SPEC_GPB + j;
            const float tsun = __expf(-cs.x * P.g_a[g] - cs.y * P.g_b[g]);
            const float eup = sh_irr(P.sh + 9 * g, Yu);
            if (under) {
                es += P.g_e[g] * tsun * sp.sun_t * twd * visx * nl;
                ek += 0.934f * eup * twa * (0.5f + 0.5f * n.y) * ao;
            } else {
                es += P.g_e[g] * tsun * visx * nl;
                ek += gi ? egi[bd] * __ldg(&sp.fs->gsplit[g]) : sh_irr(P.sh + 9 * g, Y) * ao;
            }
            eh += P.g_e[g] * tsun * fmaxf(sp.sun.y, 0.f) + eup;
        }
        o->e_sun[bd] = es / (float)SPEC_BW;
        o->e_sky[bd] = ek / (float)SPEC_BW;
        o->e_hor[bd] = eh / (float)SPEC_BW;
    }
}

// --metrics: both tracers per pixel (k_crack in the spec). Counter slots:
enum { MT_PIXELS, MT_TERRAIN, MT_AXIS, MT_LEVEL, MT_FLORA, MT_WATER, MT_SKY, MT_VOXEL, MT_CRACK, MT_CRACK_BELOW,
       MT_CTERRAIN, MT_NAN, MT_LOD, MT_FAR, MT_DIST, MT_BELOW, MT_SKY_BELOW, MT_SH_LIT, MT_SH_NEAR, MT_SH_MID, MT_SH_FAR,
       MT_COUNT };

__global__ void __launch_bounds__(128) k_metrics(const __grid_constant__ SParams sp, unsigned long long* acc) {
    const KParams& p = sp.k;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    const float u = (2.f * (x + 0.5f) / p.w - 1.f) * p.th * p.aspect;
    const float v = (1.f - 2.f * (y + 0.5f) / p.h) * p.th;
    const float3 d = norm3(p.fw + p.rt * u + p.up * v);

    Hit hc;                                            // classic voxel DDA
    const bool ch = trace(p, p.o, d, p.max_dist, hc);
    const bool cterr = ch && !gme_is_standing_flora((uint32_t)hc.mat);
    THit h;
    SurfInfo si;
    int wm;
    float tw;
    const float tmax = g_tc.far_on ? g_tc.r2 * rsqrtf(fmaxf(d.x * d.x + d.z * d.z, 1e-8f)) : p.max_dist;   // as k_primary
    const bool nh = sm_trace2(sp, p.o, d, tmax, lod_t(x, y), flora_t(x, y), h, wm, tw);
    const bool bent = path_split(wm) && tw >= 0.f;
    float3 d2;
    const float3 hp = path_pos(sp, d, nh ? h.t : 0.f, tw, wm, &d2);
    if (nh && h.kind == HK_TERRAIN) sm_terrain(sp, h, hp, si);
    else if (nh && h.kind == HK_LOD) sm_lod(sp, h, hp, x, y, si);
    else if (nh && h.kind == HK_FAR) sm_far(sp, hp, x, y, si);
    atomicAdd(acc + MT_PIXELS, 1ull);
    if (cterr) atomicAdd(acc + MT_CTERRAIN, 1ull);
    // M3: below the geometric horizon (the sea sphere seen from a camera above the water) nothing may be sky
    const bool below = p.o.y >= g_tc.yw && d.y < sp.mu_h;
    if (below) atomicAdd(acc + MT_BELOW, 1ull);
    // diagnostic image: terrain grey by normal (red = axis-aligned), LOD bluish, far field brownish, distant
    // ground purple, flora green, voxel mode blue, sky black; silhouette misses magenta, true holes yellow,
    // sky below the geometric horizon cyan
    float3 col = f3(0.f, 0.f, 0.f);
    if (nh) {
        if (h.kind == HK_TERRAIN || h.kind == HK_LOD || h.kind == HK_FAR) {
            atomicAdd(acc + (h.kind == HK_TERRAIN ? MT_TERRAIN : (h.kind == HK_LOD ? MT_LOD : MT_FAR)), 1ull);
            const float l = 0.25f + 0.6f * sat(dot3(si.n, p.sun));
            col = h.kind == HK_TERRAIN ? f3(l, l, l) : (h.kind == HK_LOD ? f3(0.8f * l, 0.85f * l, l) : f3(l, 0.8f * l, 0.6f * l));
            if (h.kind == HK_TERRAIN) {
                const float m = fmaxf(fabsf(si.n.x), fmaxf(fabsf(si.n.y), fabsf(si.n.z)));
                if (m > 0.9995f) { atomicAdd(acc + MT_AXIS, 1ull); col = f3(1.f, 0.f, 0.f); }
                if (si.n.x == 0.f && si.n.z == 0.f) { atomicAdd(acc + MT_LEVEL, 1ull); col = f3(1.f, 0.55f, 0.f); }
            }
            if (isnan(si.n.x) || isnan(si.n.y) || isnan(si.n.z) || isnan(h.t)) atomicAdd(acc + MT_NAN, 1ull);
        } else if (h.kind == HK_DIST) {
            atomicAdd(acc + MT_DIST, 1ull);
            col = f3(0.6f, 0.2f, 0.8f);
        } else if (h.kind == HK_VOXEL) {
            atomicAdd(acc + MT_VOXEL, 1ull);
            col = f3(0.2f, 0.3f, 1.f);
        } else {
            atomicAdd(acc + MT_FLORA, 1ull);
            col = f3(0.1f, 0.7f, 0.1f);
        }
    } else {
        atomicAdd(acc + MT_SKY, 1ull);
        if (below && wm == WM_NONE) { atomicAdd(acc + MT_SKY_BELOW, 1ull); col = f3(0.f, 1.f, 1.f); }
    }
    if (wm != WM_NONE) atomicAdd(acc + MT_WATER, 1ull);
    // cracks: the classic hit is compared with the straight ray only (refraction bends the new path)
    if (cterr && hc.t < 0.9f * p.max_dist && !nh && !bent) {
        atomicAdd(acc + MT_CRACK, 1ull);
        col = f3(1.f, 0.f, 1.f);
        const float3 q = p.o + d * hc.t;               // a true hole: the ray passed under the smooth surface
        float T;
        if (tr_surface(q.x, q.z, T) && T > q.y + 0.05f) { atomicAdd(acc + MT_CRACK_BELOW, 1ull); col = f3(1.f, 1.f, 0.f); }
    }
    const uint32_t r = (uint32_t)(sat(col.x) * 255.f + 0.5f), g = (uint32_t)(sat(col.y) * 255.f + 0.5f);
    const uint32_t b = (uint32_t)(sat(col.z) * 255.f + 0.5f);
    p.out[(size_t)y * p.w + x] = 0xFF000000u | (r << 16) | (g << 8) | b;
}

// ---- far field (M3, spec 3.8): renderer-owned toroidal clipmap ----------------------------
// Level l keeps the GME_FAR_N^2 samples of the window [c - 512, c + 511]^2
// around the camera's sample c (absolute sample index; spacing 16 << 2l m) at
// storage index (index & 1023). When the camera moves, k_far_update rebuilds
// the rows and columns that enter the window; the mips are then rebuilt.
struct FarRect { int32_t x0, z0, nx, nz; };            // absolute sample rectangle

// 16 lanes per sample, one sub-sample each (gme_far_sub), summed across the
// lanes (integer sums: the same bits as the CPU's gme_far_sample), lane 0
// adds the biomap slot and stores the packed sample
__global__ void __launch_bounds__(256) k_far_update(uint32_t* __restrict__ s, int level, FarRect rc, uint32_t seed, GmePlanet planet,
                                                    GmeBioView bio) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x, i = gid >> 4;
    const int k = (int)(gid & 15u);
    const bool ok = i < (uint32_t)rc.nx * (uint32_t)rc.nz;
    const int32_t ax = rc.x0 + (int32_t)(i % (uint32_t)rc.nx), az = rc.z0 + (int32_t)(i / (uint32_t)rc.nx);
    const int32_t S = gme_far_spacing(level);
    int32_t tq = 0;
    uint32_t cls = 0u;
    if (ok) gme_far_sub(ax * S, az * S, level, k & 3, k >> 2, seed, &planet, &tq, &cls);
    int32_t nb = (int32_t)(cls & 1u), ng = (int32_t)((cls >> 1) & 1u), nm = (int32_t)((cls >> 2) & 1u);
    for (int o = 8; o > 0; o >>= 1) {                  // within each half-warp (one sample)
        tq += __shfl_xor_sync(0xFFFFFFFFu, tq, o);
        nb += __shfl_xor_sync(0xFFFFFFFFu, nb, o);
        ng += __shfl_xor_sync(0xFFFFFFFFu, ng, o);
        nm += __shfl_xor_sync(0xFFFFFFFFu, nm, o);
    }
    if (ok && k == 0)
        s[(((uint32_t)az & (GME_FAR_N - 1u)) << 10) + ((uint32_t)ax & (GME_FAR_N - 1u))] = gme_far_pack(ax * S, az * S, tq, nb, ng, nm, seed, &planet, &bio);
}
// 8 x 8 mip: max / min (q4) over storage samples [8i, 8i + 8]^2 (one row and column of overlap),
// for the cells [cx0, cx0 + ncx) x [cz0, cz0 + ncz) (storage cell indices, wrapping)
__global__ void __launch_bounds__(128) k_far_mipA(const uint32_t* __restrict__ s, short2* __restrict__ mA, int cx0, int ncx, int cz0, int ncz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncx * ncz) return;
    const int cx = (cx0 + i % ncx) & (GME_FAR_N / 8 - 1), cz = (cz0 + i / ncx) & (GME_FAR_N / 8 - 1);
    int mx = -32768, mn = 32767;
    for (int b = 0; b <= 8; ++b) {
        const uint32_t row = (uint32_t)((cz * 8 + b) & (GME_FAR_N - 1)) << 10;
        for (int a = 0; a <= 8; ++a) {
            const int v = (int)(int16_t)(s[row + ((cx * 8 + a) & (GME_FAR_N - 1))] & 0xFFFFu);
            mx = max(mx, v);
            mn = min(mn, v);
        }
    }
    mA[cz * (GME_FAR_N / 8) + cx] = make_short2((short)mx, (short)mn);
}
// 64 x 64 mip from the 8 x 8 one (whose overlap covers the last row / column), and the level's maximum
__global__ void __launch_bounds__(256) k_far_mipB(const short2* __restrict__ mA, short2* __restrict__ mB, int* gmax) {
    const int i = threadIdx.x, cx = i & 15, cz = i >> 4;
    int mx = -32768, mn = 32767;
    for (int b = 0; b < 8; ++b)
        for (int a = 0; a < 8; ++a) {
            const short2 v = mA[(cz * 8 + b) * (GME_FAR_N / 8) + cx * 8 + a];
            mx = max(mx, (int)v.x);
            mn = min(mn, (int)v.y);
        }
    mB[i] = make_short2((short)mx, (short)mn);
    __shared__ int sm[256];
    sm[i] = mx;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (i < w) sm[i] = max(sm[i], sm[i + w]);
        __syncthreads();
    }
    if (i == 0) *gmax = sm[0];
}
__global__ void k_set_int(int* p, int v) { *p = v; }
// Ceiling of the near walk (metres, absolute): the highest published tile
// (fine top with flora and the water plane, LOD top) and the L1 far field
// within the ring, which stands in for tiles that have not streamed in.
// (camu, camv): the camera in L1 sample units relative to the window origin (w0x, w0z).
__global__ void __launch_bounds__(256) k_near_bound(const uint32_t* __restrict__ tag, const GmeColTile* __restrict__ tiles, uint32_t ntiles,
                                                    const short2* __restrict__ mA0, int32_t w0x, int32_t w0z, float camu, float camv,
                                                    float ring_s, int* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int m = -32768;
    if (i < ntiles && tag && __ldg(tag + i)) m = max((int)__ldg(&tiles[i].top), (int)__ldg(&tiles[i].lod_top));
    if (mA0 && i < (GME_FAR_N / 8) * (GME_FAR_N / 8)) {
        const int sx = (int)(i & (GME_FAR_N / 8 - 1)) * 8, sz = (int)(i / (GME_FAR_N / 8)) * 8;
        const float du = (float)((sx - w0x) & (GME_FAR_N - 1)) + 4.f - camu, dv = (float)((sz - w0z) & (GME_FAR_N - 1)) + 4.f - camv;
        if (du * du + dv * dv < ring_s * ring_s) m = max(m, ((int)__ldg(&mA0[i]).x + 15) >> 4);
    }
    for (int o = 16; o > 0; o >>= 1) m = max(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
    if ((threadIdx.x & 31u) == 0 && m > -32768) atomicMax(out, m);
}

// ---------------------------------------------------------------------------
// M4 page / tile tag scatter (spec 6.1), one launch for both lists: (index, tag,
// value) entries of the changed page entries, then (index, tag) of the changed
// tile tags
__global__ void __launch_bounds__(256) k_scatter(const uint4* __restrict__ pe, uint32_t np, const uint2* __restrict__ te, uint32_t nt,
                                                 GmePageEntry* __restrict__ page, uint32_t* __restrict__ tag) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < np) {
        const uint4 v = pe[i];
        GmePageEntry e;
        e.tag = v.y;
        e.val = (int32_t)v.z;
        page[v.x] = e;
    } else if (i - np < nt) {
        const uint2 v = te[i - np];
        tag[v.x] = v.y;
    }
}
// D4: the uniform sky of the furnace (per-g sky-view buffer and its SH)
__global__ void k_furnace_sky(float* sv, float* sh, const float* L) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SPEC_SV_W * SPEC_SV_H * SPEC_NG) sv[i] = L[i % SPEC_NG];
    if (i < SPEC_NG * 9) sh[i] = (i % 9) == 0 ? L[i / 9] * 3.5449077f : 0.f;   // c0 = L sqrt(4 pi)
}

#define RB_SLOTS 2                       // frames in flight: the asynchronous readback (spec 6.1)
// One frame in flight. The kernels of frame N write the slot's own device frame
// (d_px) and a snapshot of the frame state (d_fs) on the render stream; the copy
// stream then reads both back into pinned host memory while frame N + 1's
// kernels run (the copy engine overlaps the SMs). The slot is reused two frames
// later, after harvest() has waited for its copy.
struct FrameSlot {
    uint32_t*   h_px;                    // pinned output frame
    uint32_t*   d_px;                    // device output frame (k_tonemap / k_render write it)
    FrameState* h_fs;                    // pinned FrameState readback
    FrameState* d_fs;                    // device snapshot of the frame state (the next frame's kernels write r->d_fs)
    TCtx*       h_tc;                    // pinned tracer context staging (an async copy from pageable memory may sync)
    uint8_t*    h_stage;                 // pinned scatter staging (page + tile entries)
    size_t      px_cap, stage_cap;
    cudaEvent_t ev[RENDER_PERF_STAGES];  // stage boundaries (ev[RS_TOTAL] = the end of the readback, copy stream)
    cudaEvent_t kdone;                   // after the frame's last kernel (the world's fence; the copy waits for it)
    cudaEvent_t done;                    // after the frame's readback (copy stream)
    int         pending;                 // queued, not yet harvested
    int         has_frame;               // h_px holds (or will hold, once harvested) a frame of ow x oh
    int         ow, oh, smooth;
};

// preset of a frame (spec 6.2, M4)
struct Preset {
    float scale;                         // internal render scale
    int   gi;                            // 0 none (sky SH x AO), 1 half-res GI + SVGF, 2 reference paths
    int   spp;                           // GI rays per half-res pixel
    int   atrous;                        // a-trous passes
    int   refl;                          // 0 sky, 1 half res, 2 full res
    int   glossy;                        // GGX-sampled ice reflections (wet ground always)
    int   vis;                           // 1 filtered sun visibility, 2 binary samples (reference)
    int   caustics;
    int   ref;                           // reference transport (accumulation)
    int   taa;                           // TAA / TAAU (else: no history at all, debug views)
};

struct Renderer {
    size_t        cap, ocap, hcap, tcap; // internal, output, half-res, TAA history pixels allocated
    GmePageEntry* d_page;
    size_t        page_bytes;
    uint32_t*     d_tile_tag;
    size_t        tile_tag_bytes;
    uint8_t*      d_scatter;
    size_t        scatter_cap;
    cudaStream_t  stream;
    cudaStream_t  cstream;               // readback copies (overlap the next frame's kernels)
    FrameSlot     slot[RB_SLOTS];
    uint32_t      nq;                    // frames queued
    int           sync;                  // renderer_set_sync
    int           last_slot;
    float         ms;                    // frame kernels of the last finished frame (title / renderer_gpu_ms)
    // M3 far field (renderer-owned clipmap, spec 3.8)
    uint32_t*     d_far[GME_FAR_LEVELS];
    short2*       d_farA[GME_FAR_LEVELS];
    short2*       d_farB[GME_FAR_LEVELS];
    int*          d_farmax;                 // [0], [1] level max (q4, absolute), [2] near-walk ceiling (m, absolute)
    int           far_ok;                   // allocated
    int           far_tried;                // alloc_far ran (renderer_create for a smooth start, else the first smooth frame)
    int           far_built[GME_FAR_LEVELS];
    int32_t       far_c[GME_FAR_LEVELS][2]; // camera sample of the built window
    uint32_t      far_seed;
    GmePlanet     far_planet;
    const uint32_t* far_bio;
    uint32_t      far_rebuilt;              // samples rebuilt for the last frame
    int           nb_dirty;                 // the near walk's ceiling needs a rebuild (tile tags uploaded)
    double        nb_x, nb_z;               // ... camera position and ring of the last one
    float         nb_ring;
    float4*       d_lin;                    // renderer_read_linear scratch
    float*        d_sdist;                  // --metrics: occluder distance per pixel
    uint32_t*     h_shadow;                 // --metrics: the shadow-distance image (host)
    uint32_t*     h_metrics;                // --metrics: the diagnostic image (host)
    GBuf          g;       // smooth path G-buffer, cap pixels
    float4*       d_color; // linear sRGB before exposure, cap pixels
    // M4 frame pipeline buffers (internal: cap, half: hcap, output: ocap)
    uint2*        d_fg[2];                  // full-res guides, by frame parity
    unsigned char* d_vsmp;                  // binary sun visibility samples
    __half*       d_vh[2];                  // sun visibility history
    __half*       d_vout;                   // filtered sun visibility
    unsigned char* d_vage[2];
    uint4*        d_efull;                  // GI irradiance per internal pixel
    uint2*        d_hg[2];                  // half-res guides
    uint4*        d_en;                     // noisy GI (half res)
    uint4*        d_eh[2];                  // GI history (first a-trous pass)
    float4*       d_mh[2];                  // moments + history length
    uint4*        d_ea; uint4* d_eb;        // a-trous ping-pong
    float*        d_va; float* d_vb;
    uint4*        d_refl;                   // reflected radiance (half or full res)
    uint4*        d_rh[2];                  // ... temporally filtered (gi.cu), by frame parity
    size_t        refl_cap;
    SecBuf        sb;                       // secondary ray slots
    size_t        sb_cap;
    int           sb_ref;                   // throughput / radiance allocated
    TaaPx*        d_taa[2];                 // resolved output history, RGBA16F (tcap)
    float4*       d_accum;                  // reference accumulation (ocap)
    GiPipe*       gi;                       // allocated with the first smooth frame when the session starts in classic
    unsigned long long* d_dacc;             // renderer_disocc counters
    // temporal state
    int           hpar;                     // buffer parity of the last frame
    int           hist_valid;               // the last frame's buffers can be the next one's history
    GiCam         prev_cam;
    Preset        prev_pr;
    int           prev_q, prev_dbg, prev_ow, prev_oh;
    double        prev_pos[3];
    float         prev_yaw, prev_pitch;
    uint32_t      acc_n;                    // reference frames averaged
    GiFrame       lastf;                    // the last frame's view for gi.cu (renderer_disocc)
    int           cache_seed_ok;
    uint32_t      cache_seed;
    float         lw[SPEC_NB];              // band luminance weights
    // D4 furnace
    float*        d_fsky;                   // [SV_H][SV_W][NG]
    float*        d_fsh;                    // [NG][9]
    float*        d_fL;                     // [NG]
    // --perf samples: a ring of the last RENDER_PERF_RING frames, RENDER_PERF_STAGES floats each (bounded:
    // an interactive session runs for hours)
    float*        perf;
    uint32_t      perf_n, perf_head;
    float         last[RENDER_PERF_STAGES];
    // the last frame, for --metrics and the probe
    SParams       lastp;
    TCtx          lastt;   // ... and its tracer context (g_tc)
    int           have_last;
    unsigned long long* d_acc;
    // spectral light (renderer_set_atmo)
    AtmoGPU*      atmo;
    ShadeTab*     d_tab;
    FrameState*   d_fs;
    unsigned int* d_hist;
    ProbeOut*     d_probe;
    FrameState    h_fs;    // read back with every frame (exposure, NaN count)
    Waves         waves;   // directions, wave numbers, amplitudes; per-frame phases from w_omega / w_phi
    double        w_omega[WAVES], w_phi[WAVES];
    // creatures (renderer_set_fauna): device copies of the spheres, cell lists and hash grid
    FaunaSphere*  d_fsph;
    float4*       d_fpos;                   // per frame: moving centres + radii (k_fauna_anim)
    FaunaCell*    d_fcells;
    uint32_t      fauna_n, fauna_mask;
    float         fauna_ylo, fauna_yhi;
    int32_t*      fauna_macro;              // host: the occupied 2 km macro cells (x, z pairs): the frame's early out
    uint32_t      fauna_nmacro;
    FaunaPal      fauna_pal;                // classic tint
};

static int rck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return 0;
    fprintf(stderr, "render: %s: %s\n", what, cudaGetErrorString(e));
    return 1;
}

template <typename T> static int dalloc(T** p, size_t n, const char* what) {
    cudaFree(*p);
    *p = NULL;
    return rck(cudaMalloc((void**)p, n * sizeof(T)), what);
}

static void free_sec(Renderer* r) {
    cudaFree(r->sb.o); cudaFree(r->sb.r); cudaFree(r->sb.h); cudaFree(r->sb.thr); cudaFree(r->sb.acc);
    memset(&r->sb, 0, sizeof r->sb);
    r->sb_cap = 0;
    r->sb_ref = 0;
}

// Buffers of frames with internal size iw x ih (0 x 0: classic, which needs only
// the output frames) and output size ow x oh, taa: the output-resolution TAA
// history (smooth path): grown when needed (the streams are drained first); the
// temporal histories restart. sec_slots / refl_px: secondary ray slots and
// reflection samples this preset needs.
static int ensure_buffers(Renderer* r, int iw, int ih, int ow, int oh, int taa, size_t sec_slots, int sec_ref, size_t refl_px) {
    const size_t px = (size_t)iw * ih, opx = (size_t)ow * oh, hpx = (size_t)((iw + 1) / 2) * ((ih + 1) / 2);
    const bool grow_i = px > r->cap, grow_o = opx > r->ocap, grow_h = hpx > r->hcap, grow_t = taa && opx > r->tcap;
    const bool grow_s = sec_slots > r->sb_cap || (sec_ref && !r->sb_ref), grow_r = refl_px > r->refl_cap;
    if (!grow_i && !grow_o && !grow_h && !grow_t && !grow_s && !grow_r) return 0;
    if (rck(cudaStreamSynchronize(r->stream), "sync before allocation") || rck(cudaStreamSynchronize(r->cstream), "sync before allocation"))
        return -1;
    r->hist_valid = 0;
    if (grow_i) {
        const size_t n = px > r->cap ? px : r->cap;
        cudaFree(r->d_lin); r->d_lin = NULL;
        cudaFree(r->d_sdist); r->d_sdist = NULL;
        free(r->h_shadow); r->h_shadow = NULL;
        free(r->h_metrics); r->h_metrics = NULL;
        if (dalloc(&r->g.a, n, "alloc G-buffer") || dalloc(&r->g.b, n, "alloc G-buffer") || dalloc(&r->d_color, n, "alloc colour buffer") ||
            dalloc(&r->d_fg[0], n, "alloc guides") || dalloc(&r->d_fg[1], n, "alloc guides") || dalloc(&r->d_vsmp, n, "alloc visibility") ||
            dalloc(&r->d_vh[0], n, "alloc visibility") || dalloc(&r->d_vh[1], n, "alloc visibility") || dalloc(&r->d_vout, n, "alloc visibility") ||
            dalloc(&r->d_vage[0], n, "alloc visibility") || dalloc(&r->d_vage[1], n, "alloc visibility") ||
            dalloc(&r->d_efull, n, "alloc GI")) { r->cap = 0; return -1; }
        r->cap = n;
    }
    if (grow_o) {
        const size_t n = opx;
        for (int k = 0; k < RB_SLOTS; ++k) {
            FrameSlot& S = r->slot[k];
            if (S.pending) { cudaEventSynchronize(S.done); S.pending = 0; }
            S.has_frame = 0;                            // its frame is gone with the buffer
            cudaFreeHost(S.h_px);
            S.h_px = NULL;
            S.px_cap = 0;
            if (rck(cudaHostAlloc(&S.h_px, n * 4, cudaHostAllocDefault), "alloc frame") || dalloc(&S.d_px, n, "alloc frame")) {
                r->ocap = 0;
                return -1;
            }
            S.px_cap = n;
        }
        cudaFree(r->d_accum);                           // the reference accumulation: allocated with its first frame
        r->d_accum = NULL;
        r->ocap = n;
    }
    if (grow_t) {                                       // RGBA16F history (spec 6.3), smooth frames only
        if (dalloc(&r->d_taa[0], opx, "alloc TAA history") || dalloc(&r->d_taa[1], opx, "alloc TAA history")) { r->tcap = 0; return -1; }
        r->tcap = opx;
    }
    if (grow_h) {
        const size_t n = hpx;
        if (dalloc(&r->d_hg[0], n, "alloc GI") || dalloc(&r->d_hg[1], n, "alloc GI") || dalloc(&r->d_en, n, "alloc GI") ||
            dalloc(&r->d_eh[0], n, "alloc GI") || dalloc(&r->d_eh[1], n, "alloc GI") || dalloc(&r->d_mh[0], n, "alloc GI") ||
            dalloc(&r->d_mh[1], n, "alloc GI") || dalloc(&r->d_ea, n, "alloc GI") || dalloc(&r->d_eb, n, "alloc GI") ||
            dalloc(&r->d_va, n, "alloc GI") || dalloc(&r->d_vb, n, "alloc GI")) { r->hcap = 0; return -1; }
        r->hcap = n;
    }
    if (grow_s) {
        const size_t n = sec_slots > r->sb_cap ? sec_slots : r->sb_cap;
        const int ref = sec_ref || r->sb_ref;
        free_sec(r);
        if (dalloc(&r->sb.o, n, "alloc secondary rays") || dalloc(&r->sb.r, n, "alloc secondary rays") || dalloc(&r->sb.h, n, "alloc secondary rays") ||
            (ref && (dalloc(&r->sb.thr, n, "alloc paths") || dalloc(&r->sb.acc, n, "alloc paths")))) { free_sec(r); return -1; }
        r->sb_cap = n;
        r->sb_ref = ref;
    }
    if (grow_r) {
        if (dalloc(&r->d_refl, refl_px, "alloc reflections") || dalloc(&r->d_rh[0], refl_px, "alloc reflections") ||
            dalloc(&r->d_rh[1], refl_px, "alloc reflections")) { r->refl_cap = 0; return -1; }
        r->refl_cap = refl_px;
    }
    return 0;
}

// the far-field clipmap: 2 levels x (4 MiB samples + mips), allocated with the
// other fixed buffers before world_create (spec 6.3)
static void alloc_far(Renderer* r) {
    const size_t ns = (size_t)GME_FAR_N * GME_FAR_N, na = (size_t)(GME_FAR_N / 8) * (GME_FAR_N / 8), nb = (size_t)(GME_FAR_N / 64) * (GME_FAR_N / 64);
    int bad = rck(cudaMalloc(&r->d_farmax, sizeof(int) * 4), "alloc far field");
    for (int l = 0; l < GME_FAR_LEVELS && !bad; ++l)
        bad = rck(cudaMalloc(&r->d_far[l], ns * sizeof(uint32_t)), "alloc far field") ||
              rck(cudaMalloc(&r->d_farA[l], na * sizeof(short2)), "alloc far field") ||
              rck(cudaMalloc(&r->d_farB[l], nb * sizeof(short2)), "alloc far field");
    r->far_ok = !bad;
    if (bad) fprintf(stderr, "render: no far field (not enough VRAM)\n");
}

// the preset of a frame (spec 6.2): Low 0.67 scale + TAAU, sky SH x AO, sky
// reflections, hard shadows; Medium half-res GI 1 spp + 3 a-trous, half-res
// reflections, filtered disc shadows; High 2 spp + 4 a-trous, full-res
// reflections with the GGX lobe, caustics; Ref = High's scene with the
// reference transport. --gi 0 drops the GI rays; ref (--accum) keeps the scene
// of the chosen preset and switches the transport.
static Preset preset_of(const RenderCamera* c) {
    Preset p;
    memset(&p, 0, sizeof p);
    const int q = c->quality;
    const float us = c->scale > 0.f ? fminf(fmaxf(c->scale, 0.25f), 1.f) : 1.f;
    p.scale = q == RQ_LOW ? us * 0.67f : us;
    p.taa = 1;
    if (q == RQ_LOW) return p;
    p.vis = 1;
    p.gi = c->gi_off ? 0 : 1;
    p.spp = q >= RQ_HIGH ? 2 : 1;
    p.atrous = q >= RQ_HIGH ? 4 : 3;
    p.refl = q >= RQ_HIGH ? 2 : 1;
    p.glossy = q >= RQ_HIGH;
    p.caustics = q >= RQ_HIGH;
    if (q == RQ_REF || c->ref) {                      // the reference transport: full res, no denoising, accumulation
        p.ref = 1;
        p.scale = 1.f;
        p.gi = c->gi_off ? 0 : 2;
        p.refl = 2;
        p.vis = 2;
        p.atrous = 0;
    }
    return p;
}
// secondary ray slots and reflection samples of a preset at internal size iw x ih
static void preset_needs(const Preset& p, int iw, int ih, size_t* slots, int* ref, size_t* refl) {
    const size_t px = (size_t)iw * ih, hp = (size_t)((iw + 1) / 2) * ((ih + 1) / 2);
    *ref = p.ref;
    if (p.ref) { *slots = 2 * px; *refl = px; return; }
    size_t s = 0;
    if (p.gi == 1) s += hp * (size_t)(p.spp + (p.refl == 1 ? 1 : 0));
    else if (p.refl == 1) s += hp;
    if (p.refl == 2) s += px;
    *slots = s;
    *refl = p.refl == 2 ? px : (p.refl == 1 ? hp : 0);
}

// internal render size of the smooth path for output w x h at render scale s (renderer_draw)
static void internal_size(int w, int h, float s, int* iw, int* ih) {
    *iw = w; *ih = h;
    if (s >= 1.f) return;
    int x = (int)((float)w * s + 0.5f), y = (int)((float)h * s + 0.5f);
    const int mx = w < 64 ? w : 64, my = h < 36 ? h : 36;
    x = x < mx ? mx : (x > w ? w : x);
    y = y < my ? my : (y > h ? h : y);
    *iw = x; *ih = y;
}

// the smooth path's fixed parts: the radiance cache (gi.cu) and the far-field clipmap. Allocated by
// renderer_create for a smooth start, else with the first smooth frame (after world_create: a classic session
// keeps the pre-upgrade renderer's footprint, spec 6.4; the creature buffers of renderer_set_fauna come on top)
static void alloc_smooth_fixed(Renderer* r) {
    if (!r->gi) r->gi = gi_create();
    if (!r->far_tried) { r->far_tried = 1; alloc_far(r); }
}

extern "C" Renderer* renderer_create(const RendererConfig* cfg) {
    Renderer* r = (Renderer*)calloc(1, sizeof(Renderer));
    if (!r) return NULL;
    int prio_least = 0, prio_greatest = 0;          // the render stream gets the greatest priority (spec 6.1)
    if (rck(cudaDeviceGetStreamPriorityRange(&prio_least, &prio_greatest), "stream priorities") ||
        rck(cudaStreamCreateWithPriority(&r->stream, cudaStreamNonBlocking, prio_greatest), "stream") ||
        rck(cudaStreamCreateWithPriority(&r->cstream, cudaStreamNonBlocking, prio_greatest), "copy stream") ||
        rck(cudaMalloc(&r->d_acc, sizeof(unsigned long long) * MT_COUNT), "alloc metrics") ||
        rck(cudaMalloc(&r->d_dacc, sizeof(unsigned long long) * GI_DISOCC_N), "alloc metrics")) {
        renderer_destroy(r);
        return NULL;
    }
    for (int k = 0; k < RB_SLOTS; ++k) {
        for (int i = 0; i < RENDER_PERF_STAGES; ++i)
            if (rck(cudaEventCreate(&r->slot[k].ev[i]), "event")) { renderer_destroy(r); return NULL; }
        if (rck(cudaEventCreateWithFlags(&r->slot[k].done, cudaEventDisableTiming), "event") ||
            rck(cudaEventCreateWithFlags(&r->slot[k].kdone, cudaEventDisableTiming), "event") ||
            rck(cudaHostAlloc(&r->slot[k].h_fs, sizeof(FrameState), cudaHostAllocDefault), "alloc frame state") ||
            rck(cudaHostAlloc(&r->slot[k].h_tc, sizeof(TCtx), cudaHostAllocDefault), "alloc tracer context")) {
            renderer_destroy(r);
            return NULL;
        }
    }
    r->last_slot = -1;
    // every fixed buffer of the starting preset now (spec 6.3: before world_create), sized from its internal
    // resolution (Low and --scale < 1 render smaller; Ref, High's extra buffers and a larger scale grow
    // lazily). A classic start allocates nothing here: its pool is sized as before the upgrade (spec 6.4; less
    // the creature buffers when a creature file is loaded), and the first smooth frame (key V / K) allocates
    // what it needs.
    if (cfg && cfg->max_w > 0 && cfg->max_h > 0 && cfg->quality != RQ_CLASSIC) {
        RenderCamera c;
        memset(&c, 0, sizeof c);
        c.quality = cfg->quality;
        c.scale = cfg->scale;
        const Preset p = preset_of(&c);
        int iw, ih;
        internal_size(cfg->max_w, cfg->max_h, p.scale, &iw, &ih);
        size_t slots, refl;
        int ref;
        preset_needs(p, iw, ih, &slots, &ref, &refl);
        if (ensure_buffers(r, iw, ih, cfg->max_w, cfg->max_h, 1, slots, ref, refl)) {
            renderer_destroy(r);
            return NULL;
        }
        alloc_smooth_fixed(r);
    }
    return r;
}

extern "C" void renderer_destroy(Renderer* r) {
    if (!r) return;
    if (r->stream) cudaStreamSynchronize(r->stream);
    if (r->cstream) cudaStreamSynchronize(r->cstream);
    for (int k = 0; k < RB_SLOTS; ++k) {
        FrameSlot& S = r->slot[k];
        cudaFreeHost(S.h_px); cudaFreeHost(S.h_fs); cudaFreeHost(S.h_stage); cudaFreeHost(S.h_tc);
        cudaFree(S.d_px); cudaFree(S.d_fs);
        for (int i = 0; i < RENDER_PERF_STAGES; ++i) if (S.ev[i]) cudaEventDestroy(S.ev[i]);
        if (S.done) cudaEventDestroy(S.done);
        if (S.kdone) cudaEventDestroy(S.kdone);
    }
    cudaFree(r->d_page);
    cudaFree(r->d_tile_tag);
    cudaFree(r->d_scatter);
    cudaFree(r->d_acc);
    cudaFree(r->d_dacc);
    cudaFree(r->g.a);
    cudaFree(r->g.b);
    cudaFree(r->d_color);
    cudaFree(r->d_lin);
    cudaFree(r->d_sdist);
    free(r->h_shadow);
    free(r->h_metrics);
    for (int k = 0; k < 2; ++k) {
        cudaFree(r->d_fg[k]); cudaFree(r->d_vh[k]); cudaFree(r->d_vage[k]); cudaFree(r->d_hg[k]); cudaFree(r->d_eh[k]);
        cudaFree(r->d_mh[k]); cudaFree(r->d_taa[k]); cudaFree(r->d_rh[k]);
    }
    cudaFree(r->d_vsmp); cudaFree(r->d_vout); cudaFree(r->d_efull); cudaFree(r->d_en); cudaFree(r->d_ea); cudaFree(r->d_eb);
    cudaFree(r->d_va); cudaFree(r->d_vb); cudaFree(r->d_refl); cudaFree(r->d_accum);
    cudaFree(r->d_fsky); cudaFree(r->d_fsh); cudaFree(r->d_fL);
    free_sec(r);
    gi_destroy(r->gi);
    for (int l = 0; l < GME_FAR_LEVELS; ++l) { cudaFree(r->d_far[l]); cudaFree(r->d_farA[l]); cudaFree(r->d_farB[l]); }
    cudaFree(r->d_farmax);
    cudaFree(r->d_tab);
    cudaFree(r->d_fs);
    cudaFree(r->d_hist);
    cudaFree(r->d_probe);
    cudaFree(r->d_fsph);
    cudaFree(r->d_fpos);
    cudaFree(r->d_fcells);
    free(r->fauna_macro);
    atmo_destroy(r->atmo);
    if (r->stream) cudaStreamDestroy(r->stream);
    if (r->cstream) cudaStreamDestroy(r->cstream);
    free(r->perf);
    free(r);
}

// XYZ (equal-energy white) -> linear sRGB with a full CAT16 to D65 (debug albedo view)
static void host_cat_ee(const SpectralAtmo* a, float out[9]) {
    double E[3] = { 0, 0, 0 };
    for (int b = 0; b < SPEC_NB; ++b) for (int c = 0; c < 3; ++c) E[c] += a->cmf[b][c];
    const double M16[3][3] = { { 0.401288, 0.650173, -0.051461 }, { -0.250268, 1.204414, 0.045854 }, { -0.002079, 0.048952, 0.953127 } };
    const double X2R[3][3] = { { 3.2406, -1.5372, -0.4986 }, { -0.9689, 1.8758, 0.0415 }, { 0.0557, -0.2040, 1.0570 } };
    const double d65[3] = { 0.95047, 1.0, 1.08883 }, wn[3] = { E[0] / E[1], 1.0, E[2] / E[1] };
    double gain[3], Mi[3][3];
    for (int r = 0; r < 3; ++r) {
        const double rw = M16[r][0] * wn[0] + M16[r][1] * wn[1] + M16[r][2] * wn[2];
        const double rr = M16[r][0] * d65[0] + M16[r][1] * d65[1] + M16[r][2] * d65[2];
        gain[r] = rr / rw;
    }
    const double (*m)[3] = M16;
    const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                       m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    Mi[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det; Mi[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
    Mi[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det; Mi[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
    Mi[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det; Mi[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
    Mi[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det; Mi[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
    Mi[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            double s = 0;
            for (int k = 0; k < 3; ++k) {
                double t = 0;
                for (int j = 0; j < 3; ++j) t += Mi[k][j] * gain[j] * M16[j][c];
                s += X2R[r][k] * t;
            }
            out[3 * r + c] = (float)s;
        }
}

extern "C" int renderer_set_atmo(Renderer* r, const SpectralAtmo* a) {
    if (r->stream) cudaStreamSynchronize(r->stream);   // (a lazy build after classic frames: nothing in flight reads these)
    if (r->cstream) cudaStreamSynchronize(r->cstream);
    atmo_destroy(r->atmo);
    r->atmo = atmo_create(a);
    if (!r->atmo) return -1;
    ShadeTab* t = (ShadeTab*)calloc(1, sizeof(ShadeTab));
    if (!t) return -1;
    memcpy(t->rho, a->mat_rho, sizeof t->rho);
    memcpy(t->alpha, a->mat_alpha, sizeof t->alpha);
    memcpy(t->ior, a->mat_ior, sizeof t->ior);
    t->alpha[MATX_SOILMAT] = t->alpha[MAT_MAT];         // a mat on soil: the mat's optics (mat_rho blends its ground)
    t->ior[MATX_SOILMAT] = t->ior[MAT_MAT];
    memcpy(t->pyrite_f0, a->pyrite_f0, sizeof t->pyrite_f0);
    memcpy(t->e_lut, a->e_lut, sizeof t->e_lut);
    memcpy(t->e_avg, a->e_avg, sizeof t->e_avg);
    for (int b = 0; b < SPEC_NB; ++b) {
        for (int k = 0; k < SPEC_WT_N; ++k) t->water_lnT[b][k] = logf(fmaxf(a->water_T[b][k], 1e-30f));
        t->water_a[b] = a->water_a[b]; t->water_b[b] = a->water_b[b]; t->water_bm[b] = a->water_bm[b];
        for (int c = 0; c < 3; ++c) { t->basis[c][b] = a->rgb_basis[c][b]; t->cmf_band[b][c] = (float)a->cmf[b][c]; }
    }
    host_cat_ee(a, t->dbg_rgb);
    memcpy(t->leaf_rho, a->leaf_rho, sizeof t->leaf_rho);   // M5: pigmented tissue per hue bin
    memcpy(t->leaf_tau, a->leaf_tau, sizeof t->leaf_tau);
    memcpy(t->mat_tis, a->mat_tis, sizeof t->mat_tis);
    memcpy(t->pig_A2, a->pig_A2, sizeof t->pig_A2);         // version 2: by pigment amount and Q band width
    for (int h = 0; h < SPEC_NHUE; ++h) {                   // creatures: a skin is the hue bin's leaf, darker and less saturated
        float m = 0.f;
        for (int b = 0; b < SPEC_NB; ++b) m += a->leaf_rho[h][b];
        m /= (float)SPEC_NB;
        for (int b = 0; b < SPEC_NB; ++b) t->skin[h][b] = 0.8f * (0.6f * a->leaf_rho[h][b] + 0.4f * m);
    }
    for (int b = 0; b < SPEC_NB; ++b) t->rho[RMAT_SKIN][b] = t->skin[SPEC_HUE_DEFAULT][b];
    t->alpha[RMAT_SKIN] = 0.6f;
    t->ior[RMAT_SKIN] = SPEC_IOR_LEAF;
    {                                                   // M4: band luminance weights (GI variance and edge stopping)
        double s = 0;
        for (int b = 0; b < SPEC_NB; ++b) s += a->cmf[b][1];
        for (int b = 0; b < SPEC_NB; ++b) r->lw[b] = (float)(a->cmf[b][1] / (s > 0 ? s : 1.0));
    }
    int bad = rck(cudaMalloc(&r->d_tab, sizeof(ShadeTab)), "shade tables") ||
              rck(cudaMemcpy(r->d_tab, t, sizeof(ShadeTab), cudaMemcpyHostToDevice), "shade tables") ||
              rck(cudaMalloc(&r->d_fs, sizeof(FrameState)), "frame state") ||
              rck(cudaMemset(r->d_fs, 0, sizeof(FrameState)), "frame state") ||
              rck(cudaMalloc(&r->d_hist, sizeof(unsigned int) * HIST_BINS), "histogram") ||
              rck(cudaMemset(r->d_hist, 0, sizeof(unsigned int) * HIST_BINS), "histogram") ||
              rck(cudaMalloc(&r->d_probe, sizeof(ProbeOut)), "probe");
    for (int k = 0; k < RB_SLOTS && !bad; ++k)          // per-slot snapshots of the frame state (asynchronous readback)
        bad = rck(cudaMalloc(&r->slot[k].d_fs, sizeof(FrameState)), "frame state") ||
              rck(cudaMemset(r->slot[k].d_fs, 0, sizeof(FrameState)), "frame state");
    free(t);
    if (bad) return -1;
    // Gerstner waves (spec 3.6): 8 bands of 20 m .. 0.3 m about a 5 m/s wind, deep-water dispersion with the
    // planet's gravity; slope variance of the resolved waves ~60 % of Cox & Munk's 0.003 + 0.00512 U, the rest
    // is the capillary roughness; the longest waves are gentler (fetch-limited sea). Each band is WAVE_PER
    // components (a stochastic directional spectrum): wavelengths spread over the band (log-uniform, jittered),
    // directions stratified over +-100 deg about the wind and weighted by cos^2s((theta - wind) / 2) (s 6 for the
    // longest band down to 1.5: short waves are more isotropic), random phases; the weights sum to the band's
    // slope variance. A fixed seed: the sea is the same in every run.
    static const float slope[WAVE_BANDS] = { 0.035f, 0.05f, 0.06f, 0.065f, 0.065f, 0.06f, 0.055f, 0.05f };
    const double grav = a->g > 0 ? a->g : 9.81, PI2 = 6.283185307179586;
    const double oct = log2(20.0 / 0.3) / (WAVE_BANDS - 1);             // band spacing in octaves
    uint32_t rs = 0x9E3779B9u;
    struct WaveC { double k, th, s, ph, pm; } wc[WAVES];
    for (int b = 0; b < WAVE_BANDS; ++b) {
        const double lam_b = 20.0 * pow(0.3 / 20.0, b / (double)(WAVE_BANDS - 1));
        const double sexp = 6.0 - 4.5 * b / (double)(WAVE_BANDS - 1);  // spreading exponent s
        double wsum = 0, wt[WAVE_PER];
        int perm[WAVE_PER];                                               // directions in a shuffled order vs wavelengths
        for (int j = 0; j < WAVE_PER; ++j) perm[j] = j;
        for (int j = WAVE_PER - 1; j > 0; --j) { rs = rs * 1664525u + 1013904223u; const int q = (int)((rs >> 8) % (uint32_t)(j + 1)); const int t = perm[j]; perm[j] = perm[q]; perm[q] = t; }
        for (int j = 0; j < WAVE_PER; ++j) {
            double u5[5];
            for (int q = 0; q < 5; ++q) { rs = rs * 1664525u + 1013904223u; u5[q] = (double)(rs >> 8) * (1.0 / 16777216.0); }
            const double u = ((j + 0.5) / WAVE_PER - 0.5) + 0.8 / WAVE_PER * (u5[0] - 0.5);
            const double lam = lam_b * pow(2.0, -u * oct);
            const double dth = 1.75 * (((perm[j] + 0.5) / WAVE_PER - 0.5) * 2.0 + 1.6 / WAVE_PER * (u5[1] - 0.5));
            WaveC& c = wc[b * WAVE_PER + j];
            c.k = PI2 / lam;
            c.th = 0.7 + dth;
            c.ph = PI2 * u5[2];
            c.pm = (u5[3] < 0.5 ? -1.0 : 1.0) * (0.6 + 2.0 * fabs(u5[3] - 0.5));
            wt[j] = pow(fmax(cos(0.5 * dth), 0.0), 2.0 * sexp) * (0.6 + 0.8 * u5[4]);
            wsum += wt[j];
        }
        for (int j = 0; j < WAVE_PER; ++j) wc[b * WAVE_PER + j].s = slope[b] * sqrt(wt[j] / wsum);
    }
    for (int i = 1; i < WAVES; ++i)                       // ascending wave number: the shaders stop at the first faded one
        for (int j = i; j > 0 && wc[j].k < wc[j - 1].k; --j) { const WaveC t = wc[j]; wc[j] = wc[j - 1]; wc[j - 1] = t; }
    double tail = 0;
    for (int i = WAVES - 1; i >= 0; --i) { tail += 0.5 * wc[i].s * wc[i].s; r->waves.tail[i] = (float)tail; }
    for (int i = 0; i < WAVES; ++i) {
        const double k = wc[i].k, th = wc[i].th;
        r->waves.k[i] = (float)k;
        r->waves.dx[i] = (float)cos(th); r->waves.dz[i] = (float)sin(th);
        r->waves.kx[i] = (float)(k * cos(th)); r->waves.kz[i] = (float)(k * sin(th));
        r->waves.s[i] = (float)wc[i].s;
        r->waves.q[i] = 0.35f;
        r->waves.pm[i] = (float)wc[i].pm;
        r->w_omega[i] = sqrt(grav * k);
        r->w_phi[i] = wc[i].ph;
    }
    r->waves.alpha0 = 0.1f;
    memset(&r->h_fs, 0, sizeof r->h_fs);
    return 0;
}

extern "C" int renderer_set_fauna(Renderer* r, const FaunaSphere* sph, uint32_t nsph, const FaunaCell* cells, uint32_t ncells,
                                  const uint8_t palette[16][3]) {
    if (r->stream) cudaStreamSynchronize(r->stream);
    cudaFree(r->d_fsph); cudaFree(r->d_fpos); cudaFree(r->d_fcells);
    free(r->fauna_macro);
    r->d_fsph = NULL; r->d_fpos = NULL; r->d_fcells = NULL; r->fauna_macro = NULL;
    r->fauna_n = 0; r->fauna_nmacro = 0;
    for (int i = 0; i < BIOMAP_PALETTE; ++i) {              // sRGB -> linear, gamma 2 like the classic encoding
        const float c[3] = { palette ? palette[i][0] / 255.f : 0.f, palette ? palette[i][1] / 255.f : 0.f, palette ? palette[i][2] / 255.f : 0.f };
        r->fauna_pal.c[i] = make_float3(c[0] * c[0], c[1] * c[1], c[2] * c[2]);
    }
    if (!sph || !nsph || !cells || !ncells || (ncells & (ncells - 1u))) return 0;
    // the occupied macro cells (2 km) for the frame's early out, the height range of all spheres
    int32_t* macro = (int32_t*)malloc(sizeof(int32_t) * 2 * (size_t)ncells);
    if (!macro) return -1;
    uint32_t nm = 0;
    float ylo = 3.0e38f, yhi = -3.0e38f;
    for (uint32_t i = 0; i < ncells; ++i) {
        if (!cells[i].count) continue;
        ylo = fminf(ylo, cells[i].ylo);
        yhi = fmaxf(yhi, cells[i].yhi);
        const int32_t mx = cells[i].cx >> FAUNA_MACRO_SHIFT, mz = cells[i].cz >> FAUNA_MACRO_SHIFT;
        uint32_t k = nm;
        while (k > 0 && (macro[2 * (k - 1)] != mx || macro[2 * (k - 1) + 1] != mz)) --k;   // newest first: cells of a place lie together
        if (!k) { macro[2 * nm] = mx; macro[2 * nm + 1] = mz; ++nm; }
    }
    if (rck(cudaMalloc(&r->d_fsph, sizeof(FaunaSphere) * (size_t)nsph), "alloc creatures") ||
        rck(cudaMalloc(&r->d_fpos, sizeof(float4) * (size_t)nsph), "alloc creatures") ||
        rck(cudaMalloc(&r->d_fcells, sizeof(FaunaCell) * (size_t)ncells), "alloc creatures") ||
        rck(cudaMemcpy(r->d_fsph, sph, sizeof(FaunaSphere) * (size_t)nsph, cudaMemcpyHostToDevice), "upload creatures") ||
        rck(cudaMemcpy(r->d_fcells, cells, sizeof(FaunaCell) * (size_t)ncells, cudaMemcpyHostToDevice), "upload creatures")) {
        cudaFree(r->d_fsph); cudaFree(r->d_fpos); cudaFree(r->d_fcells);
        r->d_fsph = NULL; r->d_fpos = NULL; r->d_fcells = NULL;
        free(macro);
        return -1;
    }
    r->fauna_macro = macro;
    r->fauna_nmacro = nm;
    r->fauna_n = nsph;
    r->fauna_mask = ncells - 1u;
    r->fauna_ylo = ylo;
    r->fauna_yhi = yhi;
    return 0;
}
// creatures of this frame: the device arrays if any creature's 2 km macro cell touches the camera's
// neighbourhood (FAUNA_FAR < 2 km), else n = 0 (no creature kernel is launched)
static FaunaDev fauna_frame(const Renderer* r, const RenderCamera* c, int classic) {
    FaunaDev F;
    memset(&F, 0, sizeof F);
    if (!r->fauna_n) return F;
    const int32_t mx = c->cx >> FAUNA_MACRO_SHIFT, mz = c->cz >> FAUNA_MACRO_SHIFT;
    bool hit = false;
    for (uint32_t k = 0; k < r->fauna_nmacro && !hit; ++k)
        hit = abs(r->fauna_macro[2 * k] - mx) <= 1 && abs(r->fauna_macro[2 * k + 1] - mz) <= 1;
    if (!hit) return F;
    F.sph = r->d_fsph; F.pos = r->d_fpos; F.cells = r->d_fcells;
    F.mask = r->fauna_mask; F.n = r->fauna_n;
    F.ylo = r->fauna_ylo; F.yhi = r->fauna_yhi;
    F.cx = c->cx; F.cy = c->cy; F.cz = c->cz;
    F.time = c->anim;
    F.classic = classic;
    return F;
}

extern "C" float renderer_gpu_ms(const Renderer* r) { return r->ms; }
extern "C" float renderer_ev100(const Renderer* r) { return r->h_fs.ev100; }
extern "C" uint32_t renderer_far_rebuilt(const Renderer* r) { return r->far_rebuilt; }

static int64_t floor_div64(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

// The far field of this frame (spec 3.8): rebuilds the samples that entered
// each level's window (all of it on the first frame, a jump or a new
// generator), then that level's mips; the near walk's ceiling; and the
// tracer's view of the levels. t.ring must be set. 0 ok.
static int far_frame(Renderer* r, const RenderScene* s, const RenderCamera* c, TCtx& t) {
    r->far_rebuilt = 0;
    if (r->far_seed != s->seed || memcmp(&r->far_planet, &s->planet, sizeof(GmePlanet)) || r->far_bio != s->bio.cells) {
        for (int l = 0; l < GME_FAR_LEVELS; ++l) r->far_built[l] = 0;
        r->nb_dirty = 1;
        r->far_seed = s->seed;
        r->far_planet = s->planet;
        r->far_bio = s->bio.cells;
    }
    const double camx = 32.0 * c->cx + c->lx, camz = 32.0 * c->cz + c->lz;
    uint32_t l1_rebuilt = 0;
    for (int l = 0; l < GME_FAR_LEVELS; ++l) {
        const int32_t S = gme_far_spacing(l);
        const int32_t ccx = (int32_t)floor(camx / S), ccz = (int32_t)floor(camz / S);
        const int32_t dx = ccx - r->far_c[l][0], dz = ccz - r->far_c[l][1];
        FarRect rc[2];
        int nr = 0;
        if (!r->far_built[l] || abs(dx) >= GME_FAR_N || abs(dz) >= GME_FAR_N) {
            rc[nr++] = { ccx - GME_FAR_N / 2, ccz - GME_FAR_N / 2, GME_FAR_N, GME_FAR_N };
        } else {                                          // the columns and rows that entered [c - 512, c + 511]
            if (dx) rc[nr++] = { dx > 0 ? r->far_c[l][0] + GME_FAR_N / 2 : ccx - GME_FAR_N / 2, ccz - GME_FAR_N / 2, abs(dx), GME_FAR_N };
            if (dz) rc[nr++] = { ccx - GME_FAR_N / 2, dz > 0 ? r->far_c[l][1] + GME_FAR_N / 2 : ccz - GME_FAR_N / 2, GME_FAR_N, abs(dz) };
        }
        for (int k = 0; k < nr; ++k) {
            const uint32_t n = (uint32_t)rc[k].nx * (uint32_t)rc[k].nz;
            k_far_update<<<(16 * n + 255) / 256, 256, 0, r->stream>>>(r->d_far[l], l, rc[k], s->seed, s->planet, s->bio);
            r->far_rebuilt += n;
        }
        for (int k = 0; k < nr; ++k) {                    // the 8 x 8 mip cells whose samples (with overlap) were rebuilt
            const int full_x = rc[k].nx >= GME_FAR_N - 8, full_z = rc[k].nz >= GME_FAR_N - 8;
            const int cx0 = full_x ? 0 : (int)(((rc[k].x0 & (GME_FAR_N - 1)) - 8) >> 3), ncx = full_x ? GME_FAR_N / 8 : (rc[k].nx + 15) / 8 + 1;
            const int cz0 = full_z ? 0 : (int)(((rc[k].z0 & (GME_FAR_N - 1)) - 8) >> 3), ncz = full_z ? GME_FAR_N / 8 : (rc[k].nz + 15) / 8 + 1;
            const int nx = ncx < GME_FAR_N / 8 ? ncx : GME_FAR_N / 8, nz = ncz < GME_FAR_N / 8 ? ncz : GME_FAR_N / 8;
            k_far_mipA<<<(nx * nz + 127) / 128, 128, 0, r->stream>>>(r->d_far[l], r->d_farA[l], cx0, nx, cz0, nz);
        }
        if (nr) k_far_mipB<<<1, 256, 0, r->stream>>>(r->d_farA[l], r->d_farB[l], r->d_farmax + l);
        if (l == 0) l1_rebuilt = r->far_rebuilt;
        r->far_built[l] = 1;
        r->far_c[l][0] = ccx;
        r->far_c[l][1] = ccz;
        // the tracer's frame: the window's origin rounded down to 64 samples, so mip cells are frame-aligned
        const int64_t fx = floor_div64((int64_t)ccx - GME_FAR_N / 2, 64) * 64, fz = floor_div64((int64_t)ccz - GME_FAR_N / 2, 64) * 64;
        FarLevel& L = t.far[l];
        L.s = r->d_far[l]; L.mA = r->d_farA[l]; L.mB = r->d_farB[l];
        L.ox = (float)(fx * S - 32ll * c->cx);
        L.oz = (float)(fz * S - 32ll * c->cz);
        L.bx = (int32_t)(((fx % GME_FAR_N) + GME_FAR_N) % GME_FAR_N);
        L.bz = (int32_t)(((fz % GME_FAR_N) + GME_FAR_N) % GME_FAR_N);
        L.S = (float)S;
        L.iS = 1.f / (float)S;
    }
    // the near walk's ceiling: published tiles + L1 inside the ring (+ 200 m); rebuilt when the tile tags, L1,
    // the ring or the camera (by more than 16 m: the margin covers it) changed
    if (r->nb_dirty || l1_rebuilt || t.ring != r->nb_ring || fabs(camx - r->nb_x) > 16.0 || fabs(camz - r->nb_z) > 16.0) {
        k_set_int<<<1, 1, 0, r->stream>>>(r->d_farmax + 2, -32768);
        const uint32_t nt = s->page_n * s->page_n, nmax = nt > (GME_FAR_N / 8) * (GME_FAR_N / 8) ? nt : (GME_FAR_N / 8) * (GME_FAR_N / 8);
        const int32_t w0x = r->far_c[0][0] - GME_FAR_N / 2, w0z = r->far_c[0][1] - GME_FAR_N / 2;
        k_near_bound<<<(nmax + 255) / 256, 256, 0, r->stream>>>(r->d_tile_tag, s->tiles, nt, r->d_farA[0], w0x, w0z,
                                                                (float)(camx / 16.0 - w0x), (float)(camz / 16.0 - w0z), (t.ring + 200.f) / 16.f,
                                                                r->d_farmax + 2);
        r->nb_dirty = 0;
        r->nb_x = camx;
        r->nb_z = camz;
        r->nb_ring = t.ring;
    }
    return rck(cudaGetLastError(), "far field");
}

// A finished (or finishing) frame slot: waits for its readback, records its
// stage times (--perf; a bounded ring) and keeps its frame state (exposure, NaN
// count). 0 ok.
#define RENDER_PERF_RING 65536u
static int harvest(Renderer* r, FrameSlot& S) {
    if (!S.pending) return 0;
    S.pending = 0;
    if (rck(cudaEventSynchronize(S.done), "frame")) return -1;
    float st[RENDER_PERF_STAGES] = { 0 };
    for (int i = 0; i < RS_TOTAL; ++i) cudaEventElapsedTime(&st[i], S.ev[i], S.ev[i + 1]);
    cudaEventElapsedTime(&st[RS_TOTAL], S.ev[0], S.ev[RS_TOTAL]);
    cudaEventElapsedTime(&r->ms, S.ev[RS_FAR], S.ev[RS_READBACK]);
    memcpy(r->last, st, sizeof st);
    if (!r->perf) r->perf = (float*)malloc(sizeof(float) * RENDER_PERF_STAGES * RENDER_PERF_RING);
    if (r->perf) {
        memcpy(r->perf + (size_t)RENDER_PERF_STAGES * r->perf_head, r->last, sizeof r->last);
        r->perf_head = (r->perf_head + 1u) % RENDER_PERF_RING;
        if (r->perf_n < RENDER_PERF_RING) ++r->perf_n;
    }
    if (S.smooth) r->h_fs = *S.h_fs;
    return 0;
}
// waits for every frame in flight (in queue order) and records them
static int drain(Renderer* r) {
    if (r->last_slot < 0) return 0;
    for (int k = 1; k <= RB_SLOTS; ++k)
        if (harvest(r, r->slot[(r->last_slot + k) % RB_SLOTS])) return -1;
    return 0;
}

// Page table and tile tags (spec 6.1, M4): the entries the world changed since
// the last frame are scattered (pinned staging, one copy, one k_scatter); the
// whole table is uploaded only when it was reallocated, more than 1/8 of it
// changed, or the world kept no list. While the world streams, the scatter
// waits for an SM slot at the head of the frame until a resident generation
// block retires (stream priorities order pending blocks, they never preempt
// running ones); that wait, not the copy or the scatter, dominates the stage's
// p95 in the flight.
static int upload_tables(Renderer* r, const RenderScene* s, FrameSlot& S) {
    const size_t ne = (size_t)s->page_n * s->page_n * s->layers, pb = sizeof(GmePageEntry) * ne;
    int full = s->page_dirty && (!s->page_dirty_idx || s->page_dirty_n > ne / 8);
    if (pb != r->page_bytes) {
        cudaFree(r->d_page);
        r->d_page = NULL;
        if (rck(cudaMalloc(&r->d_page, pb), "alloc page table")) { r->page_bytes = 0; return -1; }
        r->page_bytes = pb;
        full = 1;
    }
    const int smooth_ok = s->tiles && s->tile_tag;
    const size_t nt = (size_t)s->page_n * s->page_n, tb = sizeof(uint32_t) * nt;
    int tfull = 0;
    if (smooth_ok) {
        tfull = s->tile_dirty && (!s->tile_dirty_idx || s->tile_dirty_n > nt / 8);
        if (tb != r->tile_tag_bytes) {
            cudaFree(r->d_tile_tag);
            r->d_tile_tag = NULL;
            if (rck(cudaMalloc(&r->d_tile_tag, tb), "alloc tile tags")) { r->tile_tag_bytes = 0; return -1; }
            r->tile_tag_bytes = tb;
            tfull = 1;
        }
    }
    const uint32_t np = (!full && s->page_dirty && s->page_dirty_idx) ? s->page_dirty_n : 0;
    const uint32_t ntl = (smooth_ok && !tfull && s->tile_dirty && s->tile_dirty_idx) ? s->tile_dirty_n : 0;
    const size_t bytes = (size_t)np * 16 + (size_t)ntl * 8;
    if (bytes) {                                       // the host side first: the scatter list in pinned staging
        if (bytes > S.stage_cap) {
            cudaFreeHost(S.h_stage);
            S.h_stage = NULL;
            const size_t cap = bytes + bytes / 2 + 4096;
            if (rck(cudaHostAlloc(&S.h_stage, cap, cudaHostAllocWriteCombined), "alloc scatter staging")) { S.stage_cap = 0; return -1; }
            S.stage_cap = cap;
        }
        if (bytes > r->scatter_cap) {
            cudaFree(r->d_scatter);
            r->d_scatter = NULL;
            const size_t cap = bytes + bytes / 2 + 4096;
            if (rck(cudaMalloc(&r->d_scatter, cap), "alloc scatter list")) { r->scatter_cap = 0; return -1; }
            r->scatter_cap = cap;
        }
        uint4* pe = (uint4*)S.h_stage;
        for (uint32_t k = 0; k < np; ++k) {
            const uint32_t idx = s->page_dirty_idx[k];
            const GmePageEntry e = s->page[idx];
            pe[k] = make_uint4(idx, e.tag, (uint32_t)e.val, 0u);
        }
        uint2* te = (uint2*)(S.h_stage + (size_t)np * 16);
        for (uint32_t k = 0; k < ntl; ++k) {
            const uint32_t idx = s->tile_dirty_idx[k];
            te[k] = make_uint2(idx, s->tile_tag[idx]);
        }
    }
    // the GPU side of the upload stage starts here (--perf times GPU work, not the CPU's staging above)
    cudaEventRecord(S.ev[RS_UPLOAD], r->stream);
    if (full && rck(cudaMemcpyAsync(r->d_page, s->page, pb, cudaMemcpyHostToDevice, r->stream), "upload page")) return -1;
    if (tfull && rck(cudaMemcpyAsync(r->d_tile_tag, s->tile_tag, tb, cudaMemcpyHostToDevice, r->stream), "upload tile tags")) return -1;
    if (bytes) {
        if (rck(cudaMemcpyAsync(r->d_scatter, S.h_stage, bytes, cudaMemcpyHostToDevice, r->stream), "upload scatter list")) return -1;
        k_scatter<<<(np + ntl + 255) / 256, 256, 0, r->stream>>>((const uint4*)r->d_scatter, np, (const uint2*)(r->d_scatter + (size_t)np * 16),
                                                                ntl, r->d_page, r->d_tile_tag);
    }
    if (tfull || ntl) r->nb_dirty = 1;
    return rck(cudaGetLastError(), "page / tile scatter");
}

static float radinv(uint32_t i, uint32_t base) {
    float f = 1.f, v = 0.f;
    while (i) { f /= (float)base; v += f * (float)(i % base); i /= base; }
    return v;
}
static bool display_debug(int d) { return d == RD_NORMALS || d == RD_ALBEDO || d == RD_DEV || d == RD_AO || d == RD_VIS; }

extern "C" int renderer_draw(Renderer* r, const RenderScene* s, const RenderCamera* c,
                             int w, int h, const uint32_t** pixels) {
    if (!s->pool || w <= 0 || h <= 0) return -1;
    const int smooth_ok = s->tiles && s->tile_tag;
    const int classic = c->quality == RQ_CLASSIC || !smooth_ok || !r->atmo;
    const Preset pr = preset_of(c);
    int iw = w, ih = h;
    if (!classic) internal_size(w, h, pr.scale, &iw, &ih);
    size_t slots = 0, refl = 0;
    int ref = 0;
    if (!classic) preset_needs(pr, iw, ih, &slots, &ref, &refl);
    const int si = (int)(r->nq % RB_SLOTS);
    FrameSlot& S = r->slot[si];
    if (S.pending && harvest(r, S)) return -1;              // the frame queued two calls ago (its readback is long done)
    // classic needs only the output frames (no internal buffers, no TAA history)
    if (ensure_buffers(r, classic ? 0 : iw, classic ? 0 : ih, w, h, !classic, slots, ref, refl)) return -1;
    if (S.px_cap < (size_t)w * h) return -1;
    if (!classic) {
        alloc_smooth_fixed(r);                               // a classic start allocates these with its first smooth frame
        if (!r->gi) return -1;
    }
    if (upload_tables(r, s, S)) return -1;                   // records S.ev[RS_UPLOAD] before its first GPU command

    KParams p;
    p.page = r->d_page;
    p.pool = s->pool;
    p.n = s->page_n;
    p.layers = s->layers;
    p.min_cy = s->min_cy;
    p.cx = c->cx; p.cy = c->cy; p.cz = c->cz;
    p.o = make_float3(c->lx, c->ly, c->lz);
    const float cp = cosf(c->pitch), sp = sinf(c->pitch), cyw = cosf(c->yaw), syw = sinf(c->yaw);
    p.fw = make_float3(cp * cyw, sp, cp * syw);
    p.rt = make_float3(-syw, 0.f, cyw);                  // normalize(cross(fw, up_world))
    p.up = make_float3(-sp * cyw, cp, -sp * syw);         // cross(rt, fw)
    p.th = tanf(c->fov_y * 0.5f);
    p.aspect = (float)w / (float)h;
    if (c->sky) {
        p.sky = *c->sky;
    } else {
        memset(&p.sky, 0, sizeof p.sky);
        const float sd[3] = { 0.45f, 0.75f, 0.30f }, sc[3] = { 1.0f, 0.95f, 0.86f };
        const float zc[3] = { 0.22f, 0.45f, 0.82f }, hc[3] = { 0.70f, 0.80f, 0.93f };
        memcpy(p.sky.sun_dir, sd, 12); memcpy(p.sky.sun_color, sc, 12);
        memcpy(p.sky.zenith, zc, 12); memcpy(p.sky.horizon, hc, 12);
        p.sky.sun_intensity = 1.f; p.sky.sun_size = 0.03f;
    }
    {
        const float* s = p.sky.sun_dir;
        const float l = sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
        p.sun = make_float3(s[0] / l, s[1] / l, s[2] / l);
    }
    p.shadows = c->shadows;
    p.max_dist = s->max_dist;
    p.w = iw; p.h = ih;
    p.out = S.d_px;
    p.bio_cells = s->bio_cells;
    p.bio_nreg = s->bio_cells ? (s->bio_nreg < BIOMAP_MAX_REGIONS ? s->bio_nreg : BIOMAP_MAX_REGIONS) : 0;
    p.bio_shift = s->bio_shift;
    memcpy(p.bio_reg, s->bio_reg, sizeof p.bio_reg);
    for (int i = 0; i < BIOMAP_PALETTE; ++i)
        p.bio_pal[i] = make_float3(s->bio_palette[i][0], s->bio_palette[i][1], s->bio_palette[i][2]);

    // M4 temporal state: the previous frame's buffers are this frame's history when
    // the configuration is the same and the camera moved at most 2 m (spec 4.7)
    const double pos_abs[3] = { 32.0 * c->cx + c->lx, 32.0 * c->cy + c->ly, 32.0 * c->cz + c->lz };
    const double dmx = pos_abs[0] - r->prev_pos[0], dmy = pos_abs[1] - r->prev_pos[1], dmz = pos_abs[2] - r->prev_pos[2];
    const double moved = sqrt(dmx * dmx + dmy * dmy + dmz * dmz);
    const bool disp = display_debug(c->debug);
    const bool same = !classic && r->hist_valid && !c->reset && r->prev_q == c->quality && r->prev_pr.gi == pr.gi && r->prev_pr.ref == pr.ref &&
                      r->prev_pr.refl == pr.refl && r->prev_pr.vis == pr.vis && r->prev_ow == w && r->prev_oh == h && r->prev_cam.w == iw &&
                      r->prev_cam.h == ih && r->prev_dbg == c->debug;
    const bool hist_ok = same && moved <= 2.0;
    const bool still = hist_ok && moved == 0.0 && r->prev_yaw == c->yaw && r->prev_pitch == c->pitch;
    r->acc_n = (pr.ref && still) ? r->acc_n + 1 : 1;
    const int cur = r->hpar ^ 1, prv = r->hpar;
    float jx = 0.f, jy = 0.f;
    if (!classic && !disp) {
        const uint32_t k = pr.ref ? r->acc_n : (c->frame % 8u) + 1u;     // Halton (2, 3): 8 phases; reference: the whole sequence
        jx = radinv(k, 2) - 0.5f;
        jy = radinv(k, 3) - 0.5f;
    }
    static const int HX[4] = { 0, 1, 1, 0 }, HY[4] = { 0, 1, 0, 1 };
    const int w2 = (iw + 1) / 2, h2 = (ih + 1) / 2;

    if (!classic) {   // the column tracer's and the light's view of the same frame (also kept for --metrics / the probe)
        SParams& q = r->lastp;
        q.k = p;
        TCtx& t = r->lastt;
        t.page = r->d_page; t.pool = s->pool; t.tiles = s->tiles; t.tile_tag = r->d_tile_tag;
        t.n = s->page_n; t.layers = s->layers; t.min_cy = s->min_cy;
        t.cx = c->cx; t.cy = c->cy; t.cz = c->cz;
        t.ybase_q8 = -256 * GME_CHUNK_DIM * c->cy;
        t.yw = (float)(s->planet.sea_level + 1 - GME_CHUNK_DIM * c->cy);
        t.ytop = (float)(GME_GLOBAL_TOP - GME_CHUNK_DIM * c->cy);
        t.ice = s->planet.ocean_top_mat == MAT_ICE || s->planet.ocean_mat == MAT_ICE;
        t.time = c->anim;
        t.dith = disp ? 0.f : (float)(0.6180339887 * (double)c->frame - floor(0.6180339887 * (double)c->frame));
        t.seed = s->seed;
        t.planet = s->planet;
        t.bio = s->bio;
        // M3: LOD and far field (spec 3.1, 3.8)
        t.far_on = r->far_ok && !c->far_off;
        t.far_max = r->d_farmax;
        t.ybase_m = -(float)(GME_CHUNK_DIM * c->cy);
        t.camx = c->lx;
        t.camz = c->lz;
        // L1 holds the samples [c - 512, c + 511] of its 16 m grid around the camera's sample c, so it is valid to
        // 8160 m in every direction: the L1 / L2 boundary r1 stays inside that window (minus a sample), and so does
        // the ring, whose LOD -> L1 morph and band bounds read L1 up to the ring (a loaded radius beyond ~6.6 km,
        // i.e. 40+ GB GPUs or --far-ring, hands over to the far field at the cap)
        const float r1_max = (float)((GME_FAR_N / 2 - 2) * gme_far_spacing(0)) - (float)gme_far_spacing(0);   // 8144 m
        t.ring = fminf(c->far_ring > 0.f ? c->far_ring : s->max_dist, r1_max - 1500.f);
        t.ring_m0 = 0.85f * t.ring;
        t.ring_iw = 1.f / fmaxf(t.ring - 6.f - t.ring_m0, 1.f);
        t.r1 = fminf(fmaxf(8000.f, t.ring + 1500.f), r1_max);
        t.r1_m0 = t.r1 - 1000.f;
        t.r1_iw = 1.f / (t.r1 - 23.f - t.r1_m0);
        t.r2 = 32000.f;
        t.c2 = 0.5f / (r->atmo ? atmo_params(r->atmo)->R : 6.371e6f);
        t.drop_r2 = s->max_dist * s->max_dist;
        {
            const float lf = c->lod_fine > 0.f ? c->lod_fine : (c->quality == RQ_LOW ? 300.f : (c->quality >= RQ_HIGH ? 900.f : 600.f));
            t.lod_fine = fminf(lf, t.ring_m0);            // the morph band near the ring is always LOD
        }
        t.flora_far = 1500.f;
        q.debug = c->debug;
        q.dbg = NULL;
        q.sacc = NULL;
        q.sdist = NULL;
        // the star: the camera's time-of-day direction, else the sky's
        float3 sun = p.sun;
        const float sl = sqrtf(c->sun_dir[0] * c->sun_dir[0] + c->sun_dir[1] * c->sun_dir[1] + c->sun_dir[2] * c->sun_dir[2]);
        if (sl > 1e-6f) sun = make_float3(c->sun_dir[0] / sl, c->sun_dir[1] / sl, c->sun_dir[2] / sl);
        q.k.sun = sun;
        q.sun = sun;
        if (sun.y > 0.f) {                                // refracted at a flat water surface
            const float st = sqrtf(fmaxf(0.f, 1.f - sun.y * sun.y)) / WATER_N, ct = sqrtf(fmaxf(0.f, 1.f - st * st));
            const float hl = sqrtf(sun.x * sun.x + sun.z * sun.z);
            q.sun_r = hl > 1e-6f ? make_float3(sun.x / hl * st, ct, sun.z / hl * st) : make_float3(0.f, 1.f, 0.f);
            q.sun_t = (1.f - spec_fresnel(sun.y, WATER_N)) * sun.y / ct;
        } else {
            q.sun_r = make_float3(0.f, 1.f, 0.f);
            q.sun_t = 0.f;
        }
        const float cam_y = (float)(GME_CHUNK_DIM * c->cy) + c->ly;
        q.cam_h = fmaxf(cam_y - (float)(s->planet.sea_level + 1), 0.f);
        {                                              // the geometric horizon: rays below it meet the sea sphere
            const float R = r->atmo ? atmo_params(r->atmo)->R : 6.371e6f, hc = q.cam_h;
            q.mu_h = -sqrtf(hc * (2.f * R + hc)) / (R + hc);
        }
        q.refract = 1;                                 // every preset refracts (straight rays through water are longer)
        q.jitter = c->quality >= RQ_MEDIUM;
        q.ao_far = c->quality >= RQ_MEDIUM;
        q.wb = isnan(c->wb) ? 0.8f : c->wb;
        q.ev = c->ev;
        q.dt = c->dt;
        q.frame = c->frame;
        q.w = r->waves;
        const double X0 = 32.0 * c->cx, Z0 = 32.0 * c->cz, tsec = (double)c->anim;
        for (int i = 0; i < WAVES; ++i) {
            const double ph = q.w.kx[i] * X0 + q.w.kz[i] * Z0 - r->w_omega[i] * tsec + r->w_phi[i];
            q.w.off[i] = (float)(ph - 6.283185307179586 * floor(ph / 6.283185307179586));
        }
        q.ow = w;
        q.oh = h;
        q.outp = S.d_px;
        q.pix = 2.f * p.th / (float)p.h;
        q.tab = r->d_tab;
        q.fs = r->d_fs;
        q.color = r->d_color;
        q.hist = r->d_hist;
        q.probe = r->d_probe;
        // M4
        q.jx = jx; q.jy = jy;
        q.dither_t = disp ? 0 : 1;
        q.furnace = c->furnace;
        q.gi = pr.gi != 0;
        q.ref = pr.ref;
        q.refl = pr.refl;
        q.vis_mode = c->shadows ? pr.vis : 0;
        q.caustics = pr.caustics;
        q.w2 = w2; q.h2 = h2;
        q.hx = pr.ref ? 0 : HX[c->frame & 3u];
        q.hy = pr.ref ? 0 : HY[c->frame & 3u];
        q.fg = r->d_fg[cur];
        q.vsmp = r->d_vsmp;
        q.vis = r->d_vout;
        q.efull = r->d_efull;
        q.refl_buf = pr.ref ? r->d_refl : r->d_rh[cur];     // reference: unfiltered
        q.hg = r->d_hg[cur];
        q.cache = gi_cache(r->gi);
        q.taa = r->d_taa[cur];
        q.nan = &r->d_fs->nan;
        // M5: High's shadow rays see crowns as a leaf-area medium (spec 4.6); clumped crowns fade by the pixel size
        q.lad = c->quality >= RQ_HIGH && !c->furnace;
        t.pix = q.pix;
        q.fauna = fauna_frame(r, c, 0);                // creatures near the camera (n = 0: none)
        r->have_last = 1;
    }

    // classic: k_render alone. smooth: far field -> atmo -> k_primary -> k_surface
    // -> k_sun -> secondary rays (GI, reflections; reference paths) -> SVGF,
    // visibility filter -> k_white, k_compose, k_lum_hist, k_exposure -> TAA /
    // accumulation, k_tonemap. One event per stage boundary (RS_*).
    const dim3 block(16, 8);
    const dim3 grid((p.w + block.x - 1) / block.x, (p.h + block.y - 1) / block.y);
    const dim3 ogrid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    cudaEventRecord(S.ev[RS_FAR], r->stream);
    if (classic) {
        r->far_rebuilt = 0;
        for (int k = RS_ATMO; k <= RS_SUN; ++k) cudaEventRecord(S.ev[k], r->stream);
        k_render<<<grid, block, 0, r->stream>>>(p);
        {                                              // creatures over the classic frame (only while some are near)
            const FaunaDev F = fauna_frame(r, c, 1);
            if (F.n) {
                k_fauna_anim<<<(F.n + 255) / 256, 256, 0, r->stream>>>(F, r->d_fpos);
                k_fauna_classic<<<grid, block, 0, r->stream>>>(p, F, r->fauna_pal);
            }
        }
        for (int k = RS_GI; k <= RS_TAA; ++k) cudaEventRecord(S.ev[k], r->stream);
        r->hist_valid = 0;
    } else {
        SParams& q = r->lastp;
        TCtx& qt = r->lastt;
        if (qt.far_on && far_frame(r, s, c, qt)) return -1;
        if (!qt.far_on) r->far_rebuilt = 0;
        if (c->reset || !r->cache_seed_ok || r->cache_seed != s->seed) {   // a new world or a reset: an empty radiance cache
            gi_cache_clear(r->gi, r->stream);
            r->cache_seed_ok = 1;
            r->cache_seed = s->seed;
        }
        cudaEventRecord(S.ev[RS_ATMO], r->stream);
        const float sun3[3] = { q.sun.x, q.sun.y, q.sun.z };
        const float fw[3] = { p.fw.x, p.fw.y, p.fw.z }, rt[3] = { p.rt.x, p.rt.y, p.rt.z }, up[3] = { p.up.x, p.up.y, p.up.z };
        // aerial perspective out to the far field's edge (M3), else the loaded radius
        const float ap_dist = qt.far_on ? qt.r2 + 1000.f : p.max_dist;
        if (atmo_frame(r->atmo, sun3, q.cam_h, ap_dist, fw, rt, up, p.th, p.aspect, r->stream, 0) < 0) return -1;
        q.a = *atmo_params(r->atmo);
        if (c->furnace && r->d_fL) {                     // D4: a uniform sky
            if (!r->d_fsky && (dalloc(&r->d_fsky, (size_t)SPEC_SV_W * SPEC_SV_H * SPEC_NG, "furnace sky") || dalloc(&r->d_fsh, SPEC_NG * 9, "furnace sky")))
                return -1;
            k_furnace_sky<<<(SPEC_SV_W * SPEC_SV_H * SPEC_NG + 255) / 256, 256, 0, r->stream>>>(r->d_fsky, r->d_fsh, r->d_fL);
            q.a.sv_spec = r->d_fsky;
            q.a.sh = r->d_fsh;
        }
        *S.h_tc = qt;                                     // pinned staging: a truly asynchronous copy (the slot is reused after
        if (rck(cudaMemcpyToSymbolAsync(g_tc, S.h_tc, sizeof(TCtx), 0, cudaMemcpyHostToDevice, r->stream), "tracer context"))
            return -1;                                    // harvest() waited for it)
        cudaEventRecord(S.ev[RS_PRIMARY], r->stream);
        k_primary<<<grid, block, 0, r->stream>>>(q, r->g);
        if (q.fauna.n) {                                // creatures on the primary paths
            k_fauna_anim<<<(q.fauna.n + 255) / 256, 256, 0, r->stream>>>(q.fauna, r->d_fpos);
            k_fauna_primary<<<grid, block, 0, r->stream>>>(q, r->g);
        }
        cudaEventRecord(S.ev[RS_SURFACE], r->stream);
        k_surface<<<grid, block, 0, r->stream>>>(q, r->g);
        cudaEventRecord(S.ev[RS_SUN], r->stream);
        if (p.shadows) {
            if (q.lad) k_sun<1><<<grid, block, 0, r->stream>>>(q, r->g);
            else k_sun<0><<<grid, block, 0, r->stream>>>(q, r->g);
            if (q.fauna.n) k_fauna_sun<<<grid, block, 0, r->stream>>>(q, r->g);   // ... and in the sun's way
        }
        // ---- secondary rays: GI and reflections (spec 4.6) ----
        cudaEventRecord(S.ev[RS_GI], r->stream);
        const dim3 hgrid((w2 + block.x - 1) / block.x, (h2 + block.y - 1) / block.y);
        if (pr.ref) {                                     // reference: per pixel one GI path and one reflection path, 4 bounces
            SecGrid G = { iw, ih, 2, 0, pr.gi ? 1 : 0, 1, 1, 0, r->d_efull, r->d_refl, NULL };
            k_sec_gen<<<grid, block, 0, r->stream>>>(q, r->g, r->sb, G);
            const int n = iw * ih * 2;
            for (int b = 0; b < 4; ++b) {
                k_sec_trace<<<(n + 127) / 128, 128, 0, r->stream>>>(q, r->sb, 0, n);
                if (q.lad) k_sec_nee<1><<<(n + 127) / 128, 128, 0, r->stream>>>(q, r->sb, 0, n); else k_sec_nee<0><<<(n + 127) / 128, 128, 0, r->stream>>>(q, r->sb, 0, n);
                k_sec_shade<<<grid, block, 0, r->stream>>>(q, r->sb, G, b, b == 3);
            }
        } else {
            int base = 0;
            if (pr.gi == 1 || pr.refl == 1) {             // half res: GI rays (+ the reflection ray at Medium)
                const int gin = pr.gi == 1 ? pr.spp : 0, K = gin + (pr.refl == 1 ? 1 : 0);
                SecGrid G = { w2, h2, K, 1, gin, pr.refl == 1, pr.glossy, 0, gin ? r->d_en : NULL, pr.refl == 1 ? r->d_refl : NULL, r->d_hg[cur] };
                k_sec_gen<<<hgrid, block, 0, r->stream>>>(q, r->g, r->sb, G);
                const int n = w2 * h2 * K;
                k_sec_trace<<<(n + 127) / 128, 128, 0, r->stream>>>(q, r->sb, 0, n);
                if (q.lad) k_sec_nee<1><<<(n + 127) / 128, 128, 0, r->stream>>>(q, r->sb, 0, n); else k_sec_nee<0><<<(n + 127) / 128, 128, 0, r->stream>>>(q, r->sb, 0, n);
                k_sec_shade<<<hgrid, block, 0, r->stream>>>(q, r->sb, G, 0, 1);
                base = n;
            }
            if (pr.refl == 2) {                           // High: full-res reflections with the GGX lobe
                SecGrid G = { iw, ih, 1, 0, 0, 1, pr.glossy, base, NULL, r->d_refl, NULL };
                k_sec_gen<<<grid, block, 0, r->stream>>>(q, r->g, r->sb, G);
                const int n = iw * ih;
                k_sec_trace<<<(n + 127) / 128, 128, 0, r->stream>>>(q, r->sb, base, n);
                if (q.lad) k_sec_nee<1><<<(n + 127) / 128, 128, 0, r->stream>>>(q, r->sb, base, n); else k_sec_nee<0><<<(n + 127) / 128, 128, 0, r->stream>>>(q, r->sb, base, n);
                k_sec_shade<<<grid, block, 0, r->stream>>>(q, r->sb, G, 0, 1);
            }
        }
        // ---- denoising: SVGF on the GI, the radiance cache, the sun visibility filter ----
        cudaEventRecord(S.ev[RS_DENOISE], r->stream);
        GiFrame f;
        memset(&f, 0, sizeof f);
        GiCam& cc = f.cur;
        cc.cx = c->cx; cc.cy = c->cy; cc.cz = c->cz;
        cc.o[0] = c->lx; cc.o[1] = c->ly; cc.o[2] = c->lz;
        cc.fw[0] = p.fw.x; cc.fw[1] = p.fw.y; cc.fw[2] = p.fw.z;
        cc.rt[0] = p.rt.x; cc.rt[1] = p.rt.y; cc.rt[2] = p.rt.z;
        cc.up[0] = p.up.x; cc.up[1] = p.up.y; cc.up[2] = p.up.z;
        cc.th = p.th; cc.aspect = p.aspect; cc.w = iw; cc.h = ih; cc.jx = jx; cc.jy = jy; cc.hx = q.hx; cc.hy = q.hy;
        f.prev = r->prev_cam;
        f.dcx = c->cx - r->prev_cam.cx; f.dcy = c->cy - r->prev_cam.cy; f.dcz = c->cz - r->prev_cam.cz;
        f.hist_ok = hist_ok;
        f.w2 = w2; f.h2 = h2; f.ow = w; f.oh = h;
        f.frame = c->frame;
        f.fg = r->d_fg[cur]; f.fg_prev = r->d_fg[prv];
        f.hg = r->d_hg[cur]; f.hg_prev = r->d_hg[prv];
        f.gi_on = pr.gi == 1;
        f.atrous = pr.atrous;
        f.e_noisy = r->d_en; f.e_hist = r->d_eh[cur]; f.e_hist_prev = r->d_eh[prv];
        f.m_hist = r->d_mh[cur]; f.m_hist_prev = r->d_mh[prv];
        f.e_a = r->d_ea; f.e_b = r->d_eb; f.v_a = r->d_va; f.v_b = r->d_vb; f.e_full = r->d_efull;
        for (int b = 0; b < SPEC_NB; ++b) f.lw[b] = r->lw[b];
        f.vis_on = q.vis_mode == 1;
        f.v_smp = r->d_vsmp; f.v_hist = r->d_vh[cur]; f.v_hist_prev = r->d_vh[prv]; f.v_out = r->d_vout;
        f.v_age = r->d_vage[cur]; f.v_age_prev = r->d_vage[prv];
        f.color = r->d_color; f.taa = r->d_taa[cur]; f.taa_prev = r->d_taa[prv];
        f.k = &r->d_fs->k; f.k_prev = &r->d_fs->k_prev;
        f.taau = iw != w || ih != h;
        f.ah_still = 0.95f;                               // TAA: a pixel that stands still with >= 16 frames of history
        f.gamma_still = 1.25f;
        f.nan = &r->d_fs->nan;
        f.cache = gi_cache(r->gi);
        if (pr.ref) {
            if (!r->d_accum && dalloc(&r->d_accum, r->ocap, "alloc accumulation")) return -1;
            f.acc = r->d_accum;
            f.acc_n = r->acc_n;
        }
        f.refl_mode = pr.ref ? 0 : pr.refl;
        f.r_cur = r->d_refl; f.r_hist = r->d_rh[cur]; f.r_hist_prev = r->d_rh[prv];
        if (pr.gi == 1 && gi_denoise(r->gi, f, r->stream)) return -1;
        if (f.vis_on && gi_vis_filter(f, r->stream)) return -1;
        if (f.refl_mode && gi_refl_filter(f, r->stream)) return -1;
        // ---- shading ----
        cudaEventRecord(S.ev[RS_COMPOSE], r->stream);
        k_white<<<1, SPEC_NG, 0, r->stream>>>(q);
        k_compose<<<grid, block, 0, r->stream>>>(q, r->g);
        const dim3 hb(16, 16), hg(((p.w + 3) / 4 + 15) / 16, ((p.h + 3) / 4 + 15) / 16);
        k_lum_hist<<<hg, hb, 0, r->stream>>>(q);
        k_exposure<<<1, 32, 0, r->stream>>>(q);
        // ---- TAA / TAAU or the reference accumulation, tonemap ----
        cudaEventRecord(S.ev[RS_TAA], r->stream);
        if (gi_resolve(f, r->stream)) return -1;
        k_tonemap<<<ogrid, block, 0, r->stream>>>(q);
        // this frame becomes the next one's history
        r->lastf = f;
        r->hpar = cur;
        r->hist_valid = 1;
        r->prev_cam = cc;
        r->prev_pr = pr;
    }
    r->prev_q = c->quality;
    r->prev_dbg = c->debug;
    r->prev_ow = w;
    r->prev_oh = h;
    r->prev_pos[0] = pos_abs[0]; r->prev_pos[1] = pos_abs[1]; r->prev_pos[2] = pos_abs[2];
    r->prev_yaw = c->yaw;
    r->prev_pitch = c->pitch;
    // the frame state of this frame: a snapshot on the render stream (the next frame's kernels write d_fs)
    if (!classic && rck(cudaMemcpyAsync(S.d_fs, r->d_fs, sizeof(FrameState), cudaMemcpyDeviceToDevice, r->stream), "frame state"))
        return -1;
    cudaEventRecord(S.ev[RS_READBACK], r->stream);
    cudaEventRecord(S.kdone, r->stream);
    // the readback on the copy stream (spec 6.1): it overlaps the next frame's kernels, which write the other
    // slot's frame
    if (rck(cudaGetLastError(), classic ? "k_render" : "smooth path kernels") ||
        rck(cudaStreamWaitEvent(r->cstream, S.kdone, 0), "readback") ||
        rck(cudaMemcpyAsync(S.h_px, S.d_px, (size_t)w * h * 4, cudaMemcpyDeviceToHost, r->cstream), "readback") ||
        (!classic && rck(cudaMemcpyAsync(S.h_fs, S.d_fs, sizeof(FrameState), cudaMemcpyDeviceToHost, r->cstream), "frame state")))
        return -1;
    cudaEventRecord(S.ev[RS_TOTAL], r->cstream);
    cudaEventRecord(S.done, r->cstream);
    S.pending = 1;
    S.has_frame = 1;
    S.ow = w;
    S.oh = h;
    S.smooth = !classic;
    r->last_slot = si;
    r->nq++;
    // Return the frame queued by the call before this one (it finished while this one was queued; its slot is
    // harvested here if it was not yet). Only the first frame, a size change or the sync mode wait for this one.
    FrameSlot& Pv = r->slot[(si + RB_SLOTS - 1) % RB_SLOTS];
    if (r->sync || !Pv.has_frame || Pv.ow != w || Pv.oh != h) {
        if (harvest(r, Pv) || harvest(r, S)) return -1;
        *pixels = S.h_px;
    } else {
        if (harvest(r, Pv)) return -1;
        *pixels = Pv.h_px;
    }
    return 0;
}

extern "C" int renderer_finish(Renderer* r, const uint32_t** pixels) {
    if (r->last_slot < 0) return -1;
    if (drain(r)) return -1;                               // in queue order: the older slot first
    if (pixels) *pixels = r->slot[r->last_slot].h_px;
    return 0;
}
// the world's generation waits for the last frame's kernels (they read chunk slots and tiles), not its readback
extern "C" void* renderer_fence(const Renderer* r) { return r->last_slot < 0 ? NULL : (void*)r->slot[r->last_slot].kdone; }
extern "C" void renderer_reset_history(Renderer* r) { r->hist_valid = 0; r->acc_n = 0; }
extern "C" uint64_t renderer_nan_count(const Renderer* r) { return r->h_fs.nan; }
extern "C" void renderer_cache_stats(const Renderer* r, uint64_t* lookups, uint64_t* hits) {
    *lookups = r->h_fs.cache_lookups;
    *hits = r->h_fs.cache_hits;
}
extern "C" uint32_t renderer_accum_frames(const Renderer* r) { return r->acc_n; }
extern "C" void renderer_set_sync(Renderer* r, int sync) { r->sync = sync; }

// --metrics: the frame in grey (exposed luminance) with every shadowed pixel
// coloured by its occluder's distance along the sun ray: blue < 512 m,
// red 512 m - 2 km, yellow >= 2 km (M3: long-range shadows)
__global__ void __launch_bounds__(128) k_shadow_img(const __grid_constant__ SParams sp, const float* sdist, uint32_t* out) {
    const KParams& p = sp.k;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    const size_t i = (size_t)y * p.w + x;
    const float4 c = sp.color[i];
    const float yl = spec_srgb_oetf(sat((0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z) * __ldg(&sp.fs->k)));
    const float o = sdist[i];
    float3 col = f3(yl, yl, yl);
    if (o > 0.f) col = o < 512.f ? f3(0.25f * yl, 0.35f * yl, 0.5f + 0.5f * yl) : (o < 2000.f ? f3(1.f, 0.15f, 0.1f) : f3(1.f, 0.9f, 0.1f));
    const uint32_t r = (uint32_t)(sat(col.x) * 255.f + 0.5f), g = (uint32_t)(sat(col.y) * 255.f + 0.5f), b = (uint32_t)(sat(col.z) * 255.f + 0.5f);
    out[i] = 0xFF000000u | (r << 16) | (g << 8) | b;
}

extern "C" int renderer_metrics(Renderer* r, RenderMetrics* out) {
    memset(out, 0, sizeof *out);
    if (!r->have_last || drain(r)) return -1;             // the frames in flight first: k_metrics reuses the last one's buffer
    SParams q = r->lastp;
    const dim3 block(16, 8);
    const dim3 grid((q.k.w + block.x - 1) / block.x, (q.k.h + block.y - 1) / block.y);
    unsigned long long acc[MT_COUNT];
    const size_t px = (size_t)q.k.w * q.k.h;
    if (rck(cudaMemsetAsync(r->d_acc, 0, sizeof acc, r->stream), "metrics") ||
        rck(cudaMemcpyToSymbolAsync(g_tc, &r->lastt, sizeof(TCtx), 0, cudaMemcpyHostToDevice, r->stream), "tracer context"))
        return -1;
    if (!r->h_metrics && !(r->h_metrics = (uint32_t*)malloc(r->cap * sizeof(uint32_t)))) return -1;
    k_metrics<<<grid, block, 0, r->stream>>>(q, r->d_acc);
    if (rck(cudaMemcpyAsync(r->h_metrics, q.k.out, px * 4, cudaMemcpyDeviceToHost, r->stream), "metrics image")) return -1;
    q.dbg = r->d_acc + MT_NAN;          // NaN in the shaded colour (re-renders into the device buffers only)
    q.sacc = r->d_acc + MT_SH_LIT;      // k_sun: occluder distances (lit, < 512 m, < 2 km, beyond)
    if (!r->d_sdist && rck(cudaMalloc(&r->d_sdist, r->cap * sizeof(float)), "alloc shadow distances")) return -1;
    if (!r->h_shadow && !(r->h_shadow = (uint32_t*)malloc(r->cap * sizeof(uint32_t)))) return -1;
    if (rck(cudaMemsetAsync(r->d_sdist, 0, px * sizeof(float), r->stream), "shadow distances")) return -1;
    q.sdist = r->d_sdist;
    q.debug = RD_NONE;
    k_primary<<<grid, block, 0, r->stream>>>(q, r->g);
    if (q.fauna.n) k_fauna_primary<<<grid, block, 0, r->stream>>>(q, r->g);
    k_surface<<<grid, block, 0, r->stream>>>(q, r->g);
    if (q.k.shadows) {
        if (q.lad) k_sun<1><<<grid, block, 0, r->stream>>>(q, r->g); else k_sun<0><<<grid, block, 0, r->stream>>>(q, r->g);
        if (q.fauna.n) k_fauna_sun<<<grid, block, 0, r->stream>>>(q, r->g);
    }
    k_white<<<1, SPEC_NG, 0, r->stream>>>(q);
    k_compose<<<grid, block, 0, r->stream>>>(q, r->g);
    k_shadow_img<<<grid, block, 0, r->stream>>>(q, r->d_sdist, q.k.out);
    if (rck(cudaMemcpyAsync(r->h_shadow, q.k.out, px * 4, cudaMemcpyDeviceToHost, r->stream), "shadow image")) return -1;
    if (rck(cudaGetLastError(), "k_metrics") ||
        rck(cudaMemcpyAsync(acc, r->d_acc, sizeof acc, cudaMemcpyDeviceToHost, r->stream), "metrics") ||
        rck(cudaStreamSynchronize(r->stream), "metrics")) return -1;
    out->image = r->h_metrics;
    out->image_w = q.k.w;
    out->image_h = q.k.h;
    out->pixels = acc[MT_PIXELS];
    out->terrain = acc[MT_TERRAIN];
    out->axis_aligned = acc[MT_AXIS];
    out->flora = acc[MT_FLORA];
    out->water = acc[MT_WATER];
    out->sky = acc[MT_SKY];
    out->voxel_mode = acc[MT_VOXEL];
    out->cracks = acc[MT_CRACK];
    out->cracks_below = acc[MT_CRACK_BELOW];
    out->classic_terrain = acc[MT_CTERRAIN];
    out->classic_axis = acc[MT_CTERRAIN];
    out->nan = acc[MT_NAN];
    out->level = acc[MT_LEVEL];
    out->lod = acc[MT_LOD];
    out->far = acc[MT_FAR];
    out->distant = acc[MT_DIST];
    out->below_horizon = acc[MT_BELOW];
    out->sky_below_horizon = acc[MT_SKY_BELOW];
    out->sun_lit = acc[MT_SH_LIT];
    out->shadow_near = acc[MT_SH_NEAR];
    out->shadow_mid = acc[MT_SH_MID];
    out->shadow_far = acc[MT_SH_FAR];
    out->shadow_image = r->h_shadow;
    return 0;
}

// per pixel of the last frame: linear luminance before exposure (M4: of the
// resolved image, TAA / accumulation, divided by the exposure), path length
// (-1 sky), horizontal distance of a straight hit from the camera (-1 if not
// one), kind + 16 x water mode
__global__ void __launch_bounds__(128) k_read_lin(const __grid_constant__ SParams sp, GBuf g, float4* out) {
    const KParams& p = sp.k;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    const size_t i = (size_t)y * p.w + x;
    const float4 a = g.a[i];
    const uint2 b = g.b[i];
    const int ox = min(x * sp.ow / p.w, sp.ow - 1), oy = min(y * sp.oh / p.h, sp.oh - 1);
    const float4 c = taa_ld(sp.taa, (size_t)oy * sp.ow + ox);
    const float k = c.w >= 0.f ? __ldg(&sp.fs->k) : 1.f;
    const bool hit = (b.y & GF_HIT) != 0;
    const int wm = (int)(((hit ? b.y : b.x) >> 24) & 7u), kind = hit ? (int)((b.y >> 16) & 15u) : 0;
    const float3 d = sp_dir(sp, x, y);
    const float rr = (hit && wm == WM_NONE) ? a.x * sqrtf(d.x * d.x + d.z * d.z) : -1.f;
    out[i] = make_float4((0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z) / k, hit ? a.x : -1.f, rr, (float)(kind + 16 * wm));
}

extern "C" int renderer_read_linear(Renderer* r, float* out, int* w, int* h) {
    if (!r->have_last) return -1;
    const SParams& q = r->lastp;
    const size_t px = (size_t)q.k.w * q.k.h;
    if (!r->d_lin && rck(cudaMalloc(&r->d_lin, r->cap * sizeof(float4)), "alloc linear readback")) return -1;
    const dim3 block(16, 8);
    const dim3 grid((q.k.w + block.x - 1) / block.x, (q.k.h + block.y - 1) / block.y);
    k_read_lin<<<grid, block, 0, r->stream>>>(q, r->g, r->d_lin);
    if (rck(cudaGetLastError(), "k_read_lin") || rck(cudaMemcpyAsync(out, r->d_lin, px * sizeof(float4), cudaMemcpyDeviceToHost, r->stream), "linear readback") ||
        rck(cudaStreamSynchronize(r->stream), "linear readback")) return -1;
    *w = q.k.w;
    *h = q.k.h;
    return 0;
}

extern "C" int renderer_far_selftest(Renderer* r, uint32_t seed, const GmePlanet* planet, const GmeBioView* hbio, size_t ncells,
                                     int32_t cam_x, int32_t cam_z, int32_t move_x, int32_t move_z, uint32_t nprobe,
                                     uint64_t* checked, uint64_t* bad, uint64_t* mip_bad, float step_ms[4]) {
    *checked = *bad = *mip_bad = 0;
    for (int k = 0; k < 4; ++k) step_ms[k] = 0.f;
    if (!r->far_ok) return -1;
    uint32_t* d_cells = NULL;
    uint32_t* d_morph = NULL;
    RenderScene s;
    memset(&s, 0, sizeof s);
    s.seed = seed;
    s.planet = *planet;
    s.bio = *hbio;
    s.bio.cells = NULL;
    s.bio.morph = NULL;
    if (hbio->cells && ncells) {
        if (rck(cudaMalloc(&d_cells, ncells * sizeof(uint32_t)), "D6 cells") ||
            rck(cudaMemcpy(d_cells, hbio->cells, ncells * sizeof(uint32_t), cudaMemcpyHostToDevice), "D6 cells")) {
            cudaFree(d_cells);
            return -1;
        }
        s.bio.cells = d_cells;
        if (hbio->morph) {                                // version 2: the morphology words (crown radii differ)
            if (rck(cudaMalloc(&d_morph, ncells * sizeof(uint32_t)), "D6 morphology") ||
                rck(cudaMemcpy(d_morph, hbio->morph, ncells * sizeof(uint32_t), cudaMemcpyHostToDevice), "D6 morphology")) {
                cudaFree(d_cells);
                cudaFree(d_morph);
                return -1;
            }
            s.bio.morph = d_morph;
        }
    }
    TCtx t;
    memset(&t, 0, sizeof t);
    t.ring = 3000.f;
    RenderCamera c;
    memset(&c, 0, sizeof c);
    for (int l = 0; l < GME_FAR_LEVELS; ++l) r->far_built[l] = 0;
    int rc = 0;
    for (int pass = 0; pass < 2 && !rc; ++pass) {            // full build, then the toroidal update of a move
        const int32_t X = cam_x + (pass ? move_x : 0), Z = cam_z + (pass ? move_z : 0);
        c.cx = (int32_t)floor_div64(X, 32); c.cz = (int32_t)floor_div64(Z, 32);
        c.lx = (float)(X - 32 * c.cx) + 0.5f; c.lz = (float)(Z - 32 * c.cz) + 0.5f;
        rc = far_frame(r, &s, &c, t);
    }
    uint32_t* hs = (uint32_t*)malloc(sizeof(uint32_t) * GME_FAR_N * GME_FAR_N);
    short2* ha = (short2*)malloc(sizeof(short2) * (GME_FAR_N / 8) * (GME_FAR_N / 8));
    if (!rc && (!hs || !ha)) rc = -1;
    if (!rc) rc = rck(cudaStreamSynchronize(r->stream), "D6");
    uint32_t rng = 0x2545F491u ^ seed;
    for (int l = 0; l < GME_FAR_LEVELS && !rc; ++l) {
        if (rck(cudaMemcpy(hs, r->d_far[l], sizeof(uint32_t) * GME_FAR_N * GME_FAR_N, cudaMemcpyDeviceToHost), "D6 readback") ||
            rck(cudaMemcpy(ha, r->d_farA[l], sizeof(short2) * (GME_FAR_N / 8) * (GME_FAR_N / 8), cudaMemcpyDeviceToHost), "D6 readback")) {
            rc = -1;
            break;
        }
        const int32_t S = gme_far_spacing(l), w0x = r->far_c[l][0] - GME_FAR_N / 2, w0z = r->far_c[l][1] - GME_FAR_N / 2;
        for (uint32_t k = 0; k < nprobe; ++k) {               // random slots, half of them in the rows / columns the move rebuilt
            rng = rng * 1664525u + 1013904223u;
            int32_t ix = (int32_t)((rng >> 8) & (GME_FAR_N - 1));
            rng = rng * 1664525u + 1013904223u;
            int32_t iz = (int32_t)((rng >> 8) & (GME_FAR_N - 1));
            if (k & 1) {                                     // a slot of the last entered column (x) or row (z)
                if (k & 2) ix = (int32_t)((move_x >= 0 ? w0x + GME_FAR_N - 1 : w0x) & (GME_FAR_N - 1));
                else iz = (int32_t)((move_z >= 0 ? w0z + GME_FAR_N - 1 : w0z) & (GME_FAR_N - 1));
            }
            const int32_t ax = w0x + ((ix - w0x) & (GME_FAR_N - 1)), az = w0z + ((iz - w0z) & (GME_FAR_N - 1));
            const uint32_t ref = gme_far_sample(ax * S, az * S, l, seed, planet, hbio);
            ++*checked;
            *bad += hs[(size_t)iz * GME_FAR_N + ix] != ref;
        }
        for (int i = 0; i < (GME_FAR_N / 8) * (GME_FAR_N / 8); ++i) {   // every 8 x 8 mip cell
            const int cx = i % (GME_FAR_N / 8), cz = i / (GME_FAR_N / 8);
            int mx = -32768, mn = 32767;
            for (int b = 0; b <= 8; ++b)
                for (int a = 0; a <= 8; ++a) {
                    const int v = (int)(int16_t)(hs[(size_t)((cz * 8 + b) & (GME_FAR_N - 1)) * GME_FAR_N + ((cx * 8 + a) & (GME_FAR_N - 1))] & 0xFFFFu);
                    mx = v > mx ? v : mx;
                    mn = v < mn ? v : mn;
                }
            *mip_bad += ha[i].x != mx || ha[i].y != mn;
        }
    }
    // the update's cost per camera step of 16 m in x (one L1 column; every fourth step also one L2 column),
    // GPU time of the whole far stage, without the world's streaming on the GPU: avg / max of each kind over
    // 64 steps after 64 warm-up steps
    cudaEvent_t e0 = NULL, e1 = NULL;
    if (!rc && !rck(cudaEventCreate(&e0), "D6 event") && !rck(cudaEventCreate(&e1), "D6 event")) {
        int32_t X = cam_x + move_x;
        const int32_t Z = cam_z + move_z;
        int n1 = 0, n2 = 0;
        for (int k = 0; k < 128 && !rc; ++k) {           // 64 untimed steps first: the GPU clocks up
            X += 16;
            c.cx = (int32_t)floor_div64(X, 32);
            c.lx = (float)(X - 32 * c.cx) + 0.5f;
            c.cz = (int32_t)floor_div64(Z, 32);
            c.lz = (float)(Z - 32 * c.cz) + 0.5f;
            cudaEventRecord(e0, r->stream);
            rc = far_frame(r, &s, &c, t);
            cudaEventRecord(e1, r->stream);
            if (rc || rck(cudaEventSynchronize(e1), "D6 timing")) { rc = -1; break; }
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (k < 64) continue;
            if (r->far_rebuilt == GME_FAR_N) { step_ms[0] += ms; step_ms[1] = fmaxf(step_ms[1], ms); ++n1; }
            else if (r->far_rebuilt == 2 * GME_FAR_N) { step_ms[2] += ms; step_ms[3] = fmaxf(step_ms[3], ms); ++n2; }
        }
        if (n1) step_ms[0] /= (float)n1;
        if (n2) step_ms[2] /= (float)n2;
    }
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    free(hs);
    free(ha);
    cudaFree(d_cells);
    cudaFree(d_morph);
    for (int l = 0; l < GME_FAR_LEVELS; ++l) r->far_built[l] = 0;   // the next frame builds its own scene's far field
    return rc;
}

// D4 (M4): the furnace. A flat synthetic world of 32 x 32 published tiles (T = 1 m
// everywhere, no flora, no water), every page entry unloaded (materials from the
// analytic generator; the furnace overrides the albedo anyway), the far field
// off; a uniform sky of radiance L_g = 0.03 g_e[g], no sun, albedo 0.5,
// Lambertian, no air path. Drawn `frames` frames with a still camera.
extern "C" int renderer_d4_furnace(Renderer* r, uint32_t seed, const GmePlanet* planet, int frames, int ref, double* ratio_mean,
                                   double* ratio_min, double* ratio_max, uint64_t* pixels, uint64_t* nan) {
    *ratio_mean = *ratio_min = *ratio_max = 0.0;
    *pixels = *nan = 0;
    if (!r->atmo) return -1;
    const int N = 32, layers = 16, W = 640, H = 360;
    const size_t np = (size_t)N * N * layers, nt = (size_t)N * N;
    GmePageEntry* page = (GmePageEntry*)malloc(sizeof(GmePageEntry) * np);
    uint32_t* tag = (uint32_t*)calloc(nt, sizeof(uint32_t));
    GmeColTile* tiles = (GmeColTile*)calloc(nt, sizeof(GmeColTile));
    float* lin = (float*)malloc(sizeof(float) * 4 * W * H);
    GmeColTile* d_tiles = NULL;
    uint8_t* d_pool = NULL;
    int rc = -1;
    if (!page || !tag || !tiles || !lin) goto done;
    for (size_t i = 0; i < np; ++i) { page[i].tag = 0; page[i].val = GME_PAGE_UNLOADED; }
    for (int az = -N / 2; az < N / 2; ++az)
        for (int ax = -N / 2; ax < N / 2; ++ax) {
            const uint32_t idx = gme_page_index(ax, 0, az, (uint32_t)N, 1);
            tag[idx] = gme_tile_key(ax, az);
            GmeColTile& t = tiles[idx];
            t.tag = gme_page_tag(ax, az); t.cx = ax; t.cz = az;
            t.h_base = 0; t.flags = 0; t.top = 2; t.bot = 0;
            for (int k = 0; k < 16; ++k) { t.blk_top[k] = 2; t.blk_bot[k] = 0; t.lod_btop[k] = 2; }
            for (int k = 0; k < GME_TILE_COLS; ++k) t.col[k] = 0;   // h = h_base = 0, dev 0: T = h + 1 = 1 m
            for (int k = 0; k < GME_LOD_N * GME_LOD_N; ++k) t.lod4[k] = 16;
            t.lod_top = 2; t.lod_bot = 0;
        }
    if (rck(cudaMalloc(&d_tiles, sizeof(GmeColTile) * nt), "D4 tiles") || rck(cudaMalloc(&d_pool, GME_CHUNK_VOXELS), "D4 pool") ||
        rck(cudaMemcpy(d_tiles, tiles, sizeof(GmeColTile) * nt, cudaMemcpyHostToDevice), "D4 tiles"))
        goto done;
    {
        const AtmoParams* P = atmo_params(r->atmo);
        float L[SPEC_NG];
        double X = 0, Y = 0, Z = 0;
        for (int g = 0; g < SPEC_NG; ++g) {
            L[g] = 0.03f * P->g_e[g];
            X += 683.0 * L[g] * P->g_cmf[g][0]; Y += 683.0 * L[g] * P->g_cmf[g][1]; Z += 683.0 * L[g] * P->g_cmf[g][2];
        }
        if (!r->d_fL && dalloc(&r->d_fL, SPEC_NG, "furnace sky")) goto done;
        if (rck(cudaMemcpy(r->d_fL, L, sizeof L, cudaMemcpyHostToDevice), "furnace sky")) goto done;
        GmePlanet pl = *planet;
        pl.sea_level = -200;                              // no water anywhere near the plane
        pl.ocean_top_mat = MAT_WATER;
        pl.ocean_mat = MAT_WATER;
        RenderScene sc;
        memset(&sc, 0, sizeof sc);
        sc.page = page; sc.page_n = (uint32_t)N; sc.layers = (uint32_t)layers; sc.min_cy = -8; sc.page_dirty = 1;
        sc.pool = d_pool; sc.max_dist = 480.f;
        sc.tiles = d_tiles; sc.tile_tag = tag; sc.tile_dirty = 1;
        sc.planet = pl; sc.seed = seed;
        RenderCamera cam;
        memset(&cam, 0, sizeof cam);
        cam.cx = 0; cam.cy = 1; cam.cz = 0; cam.lx = 16.f; cam.ly = 18.f; cam.lz = 16.f;   // 49 m above the plane
        cam.yaw = 0.3f; cam.pitch = -0.95f; cam.fov_y = 1.15f; cam.shadows = 1;
        cam.quality = RQ_MEDIUM; cam.ev = 12.f; cam.wb = 0.8f; cam.far_off = 1; cam.furnace = 1; cam.ref = ref;
        const int sync0 = r->sync;
        r->sync = 1;
        renderer_reset_history(r);
        const uint32_t* px;
        int ok = 1;
        for (int k = 0; k < frames && ok; ++k) {
            cam.frame = (uint32_t)k;
            cam.reset = k == 0;
            ok = !renderer_draw(r, &sc, &cam, W, H, &px);
            sc.page_dirty = 0;
            sc.tile_dirty = 0;
        }
        int w = 0, h = 0;
        if (ok && !renderer_read_linear(r, lin, &w, &h)) {
            const float* M = r->h_fs.cat;
            const double R = M[0] * X + M[1] * Y + M[2] * Z, G = M[3] * X + M[4] * Y + M[5] * Z, B = M[6] * X + M[7] * Y + M[8] * Z;
            const double ye = 0.5 * (0.2126 * R + 0.7152 * G + 0.0722 * B);
            double s = 0, mn = 1e30, mx = -1e30;
            uint64_t n = 0;
            for (int i = 0; i < w * h; ++i) {
                const int kind = (int)lin[4 * i + 3] & 15;
                if (kind != RK_TERRAIN && kind != RK_LOD) continue;
                const double q = lin[4 * i] / ye;
                s += q; mn = q < mn ? q : mn; mx = q > mx ? q : mx;
                ++n;
            }
            if (n) { *ratio_mean = s / (double)n; *ratio_min = mn; *ratio_max = mx; *pixels = n; rc = 0; }
        }
        *nan = renderer_nan_count(r);
        r->sync = sync0;
        renderer_reset_history(r);
        r->cache_seed_ok = 0;
    }
done:
    if (r->stream) cudaStreamSynchronize(r->stream);
    cudaFree(d_tiles);
    cudaFree(d_pool);
    free(page); free(tag); free(tiles); free(lin);
    return rc;
}

extern "C" int renderer_disocc(Renderer* r, RenderDisocc* out) {
    memset(out, 0, sizeof *out);
    if (!r->have_last || !r->hist_valid) return -1;
    const GiFrame& f = r->lastf;
    if (gi_disocc(f, f.taa, f.vis_on ? f.v_age : NULL, f.gi_on ? f.m_hist : NULL, r->d_dacc, r->stream)) return -1;
    unsigned long long a[GI_DISOCC_N];
    if (rck(cudaMemcpyAsync(a, r->d_dacc, sizeof a, cudaMemcpyDeviceToHost, r->stream), "disocclusion counters") ||
        rck(cudaStreamSynchronize(r->stream), "disocclusion counters")) return -1;
    out->pixels = a[0]; out->disoccluded = a[1]; out->taa_old = a[2]; out->vis_old = a[3];
    out->gi_pixels = a[4]; out->gi_disoccluded = a[5]; out->gi_old = a[6];
    out->taa_age_max = (uint32_t)a[7]; out->gi_age_max = (uint32_t)a[8];
    return 0;
}

extern "C" int renderer_probe(Renderer* r, RenderProbe* out) {
    memset(out, 0, sizeof *out);
    if (!r->have_last || !r->d_probe) return -1;
    if (rck(cudaMemcpyToSymbolAsync(g_tc, &r->lastt, sizeof(TCtx), 0, cudaMemcpyHostToDevice, r->stream), "tracer context"))
        return -1;
    k_probe<<<1, 32, 0, r->stream>>>(r->lastp, r->g);
    ProbeOut po;
    if (rck(cudaGetLastError(), "k_probe") ||
        rck(cudaMemcpyAsync(&po, r->d_probe, sizeof po, cudaMemcpyDeviceToHost, r->stream), "probe") ||
        rck(cudaStreamSynchronize(r->stream), "probe")) return -1;
    out->valid = po.valid;
    out->sun_elev_deg = asinf(fminf(fmaxf(po.sun_y, -1.f), 1.f)) * 57.29578f;
    if (!po.valid) return 0;
    out->kind = po.kind;
    out->material = po.mat;
    out->material1 = po.mat1;
    out->w1 = po.w1;
    out->pal = po.pal;
    out->cover = po.cover;
    out->tsun = po.tsun;
    out->pig_level = po.pig;
    out->qw_bin = po.qw;
    memcpy(out->pos, po.pos, sizeof out->pos);
    memcpy(out->normal, po.n, sizeof out->normal);
    out->depth = po.depth;
    out->sun_vis = po.vis;
    out->ao = po.ao;
    memcpy(out->e_sun, po.e_sun, sizeof out->e_sun);
    memcpy(out->e_sky, po.e_sky, sizeof out->e_sky);
    memcpy(out->e_hor, po.e_hor, sizeof out->e_hor);
    return 0;
}

extern "C" int renderer_d3_sky(void* ctx, const SpectralAtmo* a, const float sun[3], float cam_h, const float* dirs, int n, float* out) {
    (void)a;
    Renderer* r = (Renderer*)ctx;
    if (!r || !r->atmo) return -1;
    return atmo_sky_probe(r->atmo, sun, cam_h, dirs, n, out, r->stream);
}

static int cmp_float(const void* a, const void* b) {
    const float x = *(const float*)a, y = *(const float*)b;
    return (x > y) - (x < y);
}

extern "C" void renderer_perf(const Renderer* r, RenderPerf* out) {
    static const char* names[RENDER_PERF_STAGES] = { "upload", "far", "atmo", "primary", "surface", "sun", "gi", "denoise", "compose", "taa",
                                                     "readback", "gpu_total" };
    memset(out, 0, sizeof *out);
    out->nstages = RENDER_PERF_STAGES;
    out->frames = r->perf_n;
    float* v = r->perf_n ? (float*)malloc(sizeof(float) * r->perf_n) : NULL;
    for (int s = 0; s < RENDER_PERF_STAGES; ++s) {
        out->name[s] = names[s];
        if (!v) continue;
        for (uint32_t i = 0; i < r->perf_n; ++i) v[i] = r->perf[(size_t)RENDER_PERF_STAGES * i + s];
        qsort(v, r->perf_n, sizeof(float), cmp_float);
        const uint32_t n = r->perf_n;
        out->p50[s] = v[(n - 1) / 2];
        out->p95[s] = v[(uint32_t)((n - 1) * 0.95 + 0.5)];
        out->max[s] = v[n - 1];
    }
    free(v);
}

extern "C" void renderer_perf_reset(Renderer* r) { r->perf_n = 0; r->perf_head = 0; }
extern "C" const float* renderer_perf_last(const Renderer* r) { return r->last; }

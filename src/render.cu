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
// Smooth-surface path (docs/render-upgrade-spec.md M1). Everything above this
// line is the classic renderer, kept byte for byte (--quality classic).
// Today's lighting model on the column tracer's smooth surface (smooth shading
// normals, depth-anchored materials, analytic flora), as a wavefront (spec 3.1):
// k_primary (column tracer -> hit record), k_surface (normals + material blend
// -> G-buffer), k_sun (shadow rays), k_compose (lighting).
// ===========================================================================
#include "trace.cuh"

struct SParams {
    KParams k;                  // camera, sky, fog, vegetation tint (shared with the classic kernel)
    TCtx    t;                  // column tracer
    int     debug;              // RD_*
    unsigned long long* dbg;    // --metrics: NaN counter of the shaded colour (NULL = off)
};

struct SurfInfo {
    float3 n;                   // shading normal
    float3 ng;                  // geometric normal (ray offsets)
    float3 base;                // albedo
    float  dev;                 // terrain: bilinear stored deviation T-(h+1), metres
    int    cave;                // terrain: an eligible corner voxel is air (cave mouth)
};

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

// classic material colour: life tint, side-face dirt on grass / mats (here by
// slope), per-voxel brightness
__device__ float3 sm_albedo(const KParams& p, int m, int32_t ax, int32_t ay, int32_t az, float steep) {
    float3 base = albedo(m);
    if (p.bio_cells && (m == MAT_GRASS || m == MAT_LEAF || m == MAT_MAT || m == MAT_KELP)) {
        const uint32_t pi = bio_palette_at(p, ax, az);
        if (pi) base = mix3(base, p.bio_pal[pi], 0.7f);
    }
    if ((m == MAT_GRASS || m == MAT_MAT) && steep > 0.f) base = mix3(base, albedo(MAT_DIRT), 0.55f * steep);
    const uint32_t hv = gme_hash3(ax, ay, az, 0x5EEDu);
    return base * (0.88f + 0.12f * (float)(hv & 255u) / 255.f);
}

// Terrain hit: shading normal (corner gradients by central differences, blended
// bilinearly), geometric normal of the patch, and the depth-anchored material
// blend of the 4 corner columns (spec 3.3, 3.4).
__device__ void sm_terrain(const SParams& p, const THit& h, float3 pos, SurfInfo& si) {
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

    // materials: corner i is eligible if T_i >= y_hit - 0.25 (drops low columns
    // under a point high on a cliff); its voxel is taken at the hit's depth below
    // that column's surface, so cliffs show the grid's real layering
    const int32_t hb = (int32_t)__ldg(&t->h_base);
    const float yabs = pos.y + (float)(32 * g_tc.cy);
    const float steep = sat((0.9f - si.n.y) * 5.f);
    const int32_t ax0 = (g_tc.cx + h.rcx) * 32 - GME_TILE_APRON + i, az0 = (g_tc.cz + h.rcz) * 32 - GME_TILE_APRON + j;
    float3 acc = f3(0.f, 0.f, 0.f);
    float wsum = 0.f, dsum = 0.f;
    si.cave = 0;
    for (int k = 0; k < 4; ++k) {
        const int ci = k & 1, cj = k >> 1;
        const float w = (ci ? u : 1.f - u) * (cj ? v : 1.f - v);
        const uint32_t cv = __ldg(&t->col[(j + cj) * GME_TILE_N + i + ci]);
        const int32_t hc = hb + (int32_t)(cv & 255u);
        const int32_t dv = (int32_t)((cv >> 8) ^ 0x80u) - 0x80;
        dsum += w * (float)dv * (1.f / GME_DEV_UNIT);
        const float Tc = (float)(256 * (hc + 1) + 2 * dv) * (1.f / 256.f);
        if (Tc < yabs - 0.25f) continue;
        const int32_t iy = min(hc, (int32_t)floorf(yabs - (float)dv * (1.f / GME_DEV_UNIT)));
        const int m = sm_voxel(ax0 + ci, iy, az0 + cj, hc);
        if (m == MAT_AIR) { si.cave = 1; continue; }
        const float w2 = w * w + 1e-4f;
        acc = acc + sm_albedo(p.k, m, ax0 + ci, iy, az0 + cj, steep) * w2;
        wsum += w2;
    }
    si.base = wsum > 0.f ? acc * (1.f / wsum) : albedo(MAT_STONE);
    si.dev = dsum;
}

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

// Primary ray: a camera below the surface starts in voxel mode; terrain hits at
// a cave mouth (sm_cave) continue in voxel mode until the ray is above the
// surface again, then the walk resumes. The water plane y_w (spec 3.6) is one
// query per ray: the first downward crossing over wet ground (bilinear T < y_w)
// sets h.water_t (M1: the classic see-through look), or ends the ray there as
// an ice hit.
__device__ __forceinline__ bool sm_trace(float3 o, float3 d, float tmax, THit& h) {
    h.water_t = -1.f;
    h.tice = -1.f;
    h.kind = HK_NONE;
    float t0 = 0.f, T;
    if (tr_surface(o.x, o.z, T)) {
        if (o.y < T) {                                 // underground (a cave): voxels first
            float tx;
            const int r = trace_voxels_call(o, d, 0.f, tmax, false, true, h, tx);
            if (r == VX_HIT) return true;
            if (r == VX_END) return false;
            t0 = tx;
        } else if (o.y < g_tc.yw && T < g_tc.yw) {
            h.water_t = 0.f;                           // the camera is in the water
        }
    }
    if (d.y < 0.f && o.y > g_tc.yw) {                  // the water plane
        const float tw = (g_tc.yw - o.y) / d.y;
        if (tw < tmax && tr_surface(o.x + d.x * tw, o.z + d.z * tw, T) && T < g_tc.yw) {
            if (g_tc.ice) { h.tice = tw; tmax = tw; }
            else h.water_t = tw;
        }
    }
    for (;;) {
        if (!trace_smooth_t<false>(o, d, t0, tmax, h)) {
            if (h.tice < 0.f) return false;
            h.t = h.tice; h.kind = HK_ICE;             // an opaque ice sheet on the sea
            return true;
        }
        if (h.kind != HK_TERRAIN || !sm_cave(h, o + d * h.t)) return true;
        float tx;                                      // cave mouth: the grid's voxels from here
        const int r = trace_voxels_call(o, d, h.t, tmax, false, true, h, tx);
        if (r == VX_HIT) return true;
        if (r == VX_END || !(tx > t0 + 0.01f)) return false;   // t0 grows every pass: the loop ends
        t0 = tx;
    }
}

// Normal + albedo of a non-terrain hit (flora, kelp, ice, voxel mode).
__device__ void sm_other(const SParams& p, const THit& h, float3 d, float3 pos, SurfInfo& si) {
    si.cave = 0;
    si.dev = 0.f;
    if (h.kind == HK_VOXEL) {                          // classic voxel shading
        float3 n = f3(0.f, 0.f, 0.f);
        if (h.axis == 0) n.x = d.x > 0 ? -1.f : 1.f;
        else if (h.axis == 1) n.y = d.y > 0 ? -1.f : 1.f;
        else n.z = d.z > 0 ? -1.f : 1.f;
        si.n = n; si.ng = n;
        const int32_t ax = g_tc.cx * 32 + h.vx, ay = g_tc.cy * 32 + h.vy, az = g_tc.cz * 32 + h.vz;
        si.base = sm_albedo(p.k, h.mat, ax, ay, az, h.axis != 1 ? 1.f : 0.f);
        return;
    }
    if (h.kind == HK_ICE) {
        si.n = f3(0.f, 1.f, 0.f); si.ng = si.n;
        const int32_t ax = g_tc.cx * 32 + (int32_t)floorf(pos.x), az = g_tc.cz * 32 + (int32_t)floorf(pos.z);
        si.base = sm_albedo(p.k, MAT_ICE, ax, g_tc.planet.sea_level, az, 0.f);
        return;
    }
    const GmeColTile* t = g_tc.tiles + gme_page_index(g_tc.cx + h.rcx, 0, g_tc.cz + h.rcz, g_tc.n, 1);
    if (h.kind == HK_KELP) {
        const int32_t x = (g_tc.cx + h.rcx) * 32 - GME_TILE_APRON + h.i, z = (g_tc.cz + h.rcz) * 32 - GME_TILE_APRON + h.j;
        const int b = ((h.j - GME_TILE_APRON) >> 3) * 4 + ((h.i - GME_TILE_APRON) >> 3);
        float3 pa, pb;
        if (tr_kelp(t, h.j * GME_TILE_N + h.i, x, z, __ldg(&t->slot[b].cell), pa, pb)) {
            const float3 ba = pb - pa;
            const float s = sat(dot3(pos - pa, ba) / fmaxf(dot3(ba, ba), 1e-6f));
            si.n = norm3(pos - (pa + ba * s));
        } else {
            si.n = f3(0.f, 1.f, 0.f);
        }
        si.ng = si.n;
        const uint32_t hv = gme_hash3(x, 0, z, 0x5EEDu);
        float3 base = sm_albedo(p.k, MAT_KELP, x, 0, z, 0.f);
        si.base = base * (0.94f + 0.06f * (float)(hv >> 24) / 255.f);
        return;
    }
    // crown / trunk of the block's shrub or tree
    const GmeSlot* s = &t->slot[h.i];
    const int32_t tx = __ldg(&s->tx), tz = __ldg(&s->tz), tcy = __ldg(&s->tcy), trad = __ldg(&s->trad), tvr = __ldg(&s->tvr);
    const int kt = (tz - (g_tc.cz + h.rcz) * 32 + GME_TILE_APRON) * GME_TILE_N + (tx - (g_tc.cx + h.rcx) * 32 + GME_TILE_APRON);
    const float dl = (float)gme_tile_dev(t, kt) * (1.f / GME_DEV_UNIT);
    const float3 ce = f3((float)(tx - 32 * g_tc.cx) + 0.5f, (float)(tcy - 32 * g_tc.cy) + 0.5f + dl, (float)(tz - 32 * g_tc.cz) + 0.5f);
    const uint32_t hv = gme_hash3(tx, tcy, tz, 0x5EEDu);
    if (h.kind == HK_CROWN) {
        const float ia2 = 1.f / (float)(trad * trad + trad), ib2 = 1.f / (float)(tvr * tvr + tvr);
        si.n = norm3(f3((pos.x - ce.x) * ia2, (pos.y - ce.y) * ib2, (pos.z - ce.z) * ia2));
        si.base = albedo(MAT_LEAF);
        if (p.k.bio_cells) {
            const uint32_t pi = bio_palette_at(p.k, tx, tz);
            if (pi) si.base = mix3(si.base, p.k.bio_pal[pi], 0.7f);
        }
    } else {
        si.n = norm3(f3(pos.x - ce.x, 0.f, pos.z - ce.z));
        si.base = albedo(MAT_WOOD);
    }
    si.ng = si.n;
    si.base = si.base * (0.88f + 0.12f * (float)(hv & 255u) / 255.f);
}

// ---- G-buffer (24 B per pixel) ------------------------------------------------
// a: t (-1 = sky), water_t, shading normal, geometric normal (octahedral 2 x snorm16)
// b: albedo (3 x 10-bit unorm, linear), flags (GF_*) | dev q7 << 8 (--debug dev)
enum { GF_HIT = 1u, GF_TERRAIN = 2u, GF_SHADOW = 4u };
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
__device__ __forceinline__ uint32_t alb_pack(float3 c) {
    return (uint32_t)(sat(c.x) * 1023.f + 0.5f) | ((uint32_t)(sat(c.y) * 1023.f + 0.5f) << 10) |
           ((uint32_t)(sat(c.z) * 1023.f + 0.5f) << 20);
}
__device__ __forceinline__ float3 alb_unpack(uint32_t v) {
    return f3((float)(v & 1023u), (float)((v >> 10) & 1023u), (float)((v >> 20) & 1023u)) * (1.f / 1023.f);
}
__device__ __forceinline__ float3 pixel_dir(const KParams& p, int x, int y) {
    const float u = (2.f * (x + 0.5f) / p.w - 1.f) * p.th * p.aspect;
    const float v = (1.f - 2.f * (y + 0.5f) / p.h) * p.th;
    return norm3(p.fw + p.rt * u + p.up * v);
}

// Hit record (k_primary -> k_surface), stored in place in the G-buffer arrays:
// a = t (-1 = none), water_t, u, v (terrain) or voxel x, z; b.x = kind | axis << 3
// | mat << 5 (voxel) or kind | rcx << 3 | rcz << 13 (10-bit each); b.y = i | j << 8
// (terrain, kelp; i = block for flora) or voxel y.
__device__ __forceinline__ void hit_pack(const THit& h, bool hit, float4& a, uint2& b) {
    a = make_float4(hit ? h.t : -1.f, h.water_t, 0.f, 0.f);
    b = make_uint2(0u, 0u);
    if (!hit) return;
    if (h.kind == HK_VOXEL) {
        b.x = (uint32_t)HK_VOXEL | ((uint32_t)h.axis << 3) | ((uint32_t)h.mat << 5);
        b.y = (uint32_t)h.vy;
        a.z = (float)h.vx; a.w = (float)h.vz;
    } else {
        b.x = (uint32_t)h.kind | (((uint32_t)h.rcx & 1023u) << 3) | (((uint32_t)h.rcz & 1023u) << 13);
        b.y = (uint32_t)h.i | ((uint32_t)h.j << 8);
        a.z = h.u; a.w = h.v;
    }
}
__device__ __forceinline__ void hit_unpack(float4 a, uint2 b, THit& h) {
    h.t = a.x; h.water_t = a.y;
    h.kind = (int)(b.x & 7u);
    if (h.kind == HK_VOXEL) {
        h.axis = (int)((b.x >> 3) & 3u); h.mat = (int)((b.x >> 5) & 255u);
        h.vx = (int)a.z; h.vy = (int)b.y; h.vz = (int)a.w;
    } else {
        h.rcx = (int)(((b.x >> 3) & 1023u) ^ 512u) - 512;
        h.rcz = (int)(((b.x >> 13) & 1023u) ^ 512u) - 512;
        h.i = (int)(b.y & 255u); h.j = (int)((b.y >> 8) & 255u);
        h.u = a.z; h.v = a.w;
    }
}

// k_primary: the column tracer (traversal only) -> hit record per pixel
__global__ void __launch_bounds__(128, 8) k_primary(const __grid_constant__ SParams sp, GBuf g) {
    const KParams& p = sp.k;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    THit h;
    const bool hit = sm_trace(p.o, pixel_dir(p, x, y), p.max_dist, h);
    float4 a;
    uint2 b;
    hit_pack(h, hit, a, b);
    const size_t i = (size_t)y * p.w + x;
    g.a[i] = a;
    g.b[i] = b;
}

// k_surface: hit record -> G-buffer (normals + depth-anchored material blend), in place
__global__ void __launch_bounds__(128, 4) k_surface(const __grid_constant__ SParams sp, GBuf g) {
    const KParams& p = sp.k;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    const size_t i = (size_t)y * p.w + x;
    float4 a = g.a[i];
    if (!(a.x >= 0.f)) return;                         // miss: the record already is the G-buffer entry
    uint2 b = g.b[i];
    THit h;
    hit_unpack(a, b, h);
    const float3 d = pixel_dir(p, x, y);
    const float3 pos = p.o + d * h.t;
    SurfInfo si;
    if (h.kind == HK_TERRAIN) sm_terrain(sp, h, pos, si);
    else sm_other(sp, h, d, pos, si);
    a.z = __uint_as_float(oct_pack(si.n));
    a.w = __uint_as_float(oct_pack(si.ng));
    b.x = alb_pack(si.base);
    const int32_t dq = max(-128, min(127, __float2int_rn(si.dev * (float)GME_DEV_UNIT)));
    b.y = GF_HIT | (h.kind == HK_TERRAIN ? GF_TERRAIN : 0u) | (((uint32_t)dq & 255u) << 8);
    g.a[i] = a;
    g.b[i] = b;
}

// k_sun: one any-hit shadow ray per lit pixel (column tracer, flora included)
__global__ void __launch_bounds__(128, 8) k_sun(const __grid_constant__ SParams sp, GBuf g) {
    const KParams& p = sp.k;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    const size_t i = (size_t)y * p.w + x;
    const float4 a = g.a[i];
    if (!(a.x >= 0.f)) return;
    if (dot3(oct_unpack(__float_as_uint(a.z)), p.sun) <= 0.f) return;
    const float3 hp = p.o + pixel_dir(p, x, y) * a.x + oct_unpack(__float_as_uint(a.w)) * 0.03f;
    THit hs;
    hs.water_t = -1.f;
    if (trace_smooth(hp, p.sun, 0.f, 512.f, hs)) g.b[i].y |= GF_SHADOW;
}

// k_compose: today's lighting model (sun + skylight + fog + classic water) on the G-buffer
__global__ void __launch_bounds__(128) k_compose(const __grid_constant__ SParams sp, GBuf g) {
    const KParams& p = sp.k;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.w || y >= p.h) return;
    const size_t i = (size_t)y * p.w + x;
    const float4 a = g.a[i];
    const uint2 b = g.b[i];
    const float3 d = pixel_dir(p, x, y);
    const float3 fog_c = sky(p, d, false);
    const float fog_k = 1.f / (p.max_dist * 0.55f);
    const bool hit = (b.y & GF_HIT) != 0;
    float water_t = a.y;
    float3 col;
    if (hit) {
        const float3 n = oct_unpack(__float_as_uint(a.z));
        const float3 base = alb_unpack(b.x);
        const float ndl = sat(dot3(n, p.sun));
        const float shadow = (b.y & GF_SHADOW) ? 0.f : 1.f;
        // star light + skylight (ambient takes the atmosphere's colour; airless = dark shadows)
        const float amb = 0.30f + 0.14f * n.y + 0.04f * n.x;
        const float3 skylight = mix3(f3(0.08f, 0.08f, 0.09f), v3(p.sky.zenith) * 0.9f + f3(0.22f, 0.22f, 0.20f), 0.9f);
        const float3 light = v3(p.sky.sun_color) * (p.sky.sun_intensity * ndl * shadow * 1.15f) + skylight * amb;
        col = base * light;
        const float fog = fmaxf(1.f - expf(-a.x * fog_k * a.x * fog_k), 0.f);
        col = mix3(col, fog_c, sat(fog));
        if (sp.debug == RD_NORMALS) col = n * 0.5f + f3(0.5f, 0.5f, 0.5f);
        else if (sp.debug == RD_ALBEDO) col = base;
        else if (sp.debug == RD_DEV) {
            const float dv = (float)(int8_t)((b.y >> 8) & 255u) * (1.f / GME_DEV_UNIT);
            col = (b.y & GF_TERRAIN) ? f3(sat(dv), 0.25f, sat(-dv)) : f3(0.1f, 0.1f, 0.1f);
        }
    } else {
        col = sky(p, d, true);
    }
    if (sp.debug != RD_NONE) water_t = -1.f;

    if (water_t >= 0.f && (!hit || water_t < a.x)) { // translucent water: absorb along the path, reflect sky at the surface
        const float path = (hit ? a.x : p.max_dist) - water_t;
        const float aa = 1.f - expf(-path * 0.07f);
        const float3 deep = f3(0.04f, 0.20f, 0.32f) * (0.55f + 0.45f * p.sun.y);
        float3 under = mix3(col, deep, aa);
        if (water_t > 0.f) {
            const float c = fabsf(d.y);
            const float fres = 0.02f + 0.98f * powf(1.f - c, 5.f);
            const float3 refl = sky(p, f3(d.x, fabsf(d.y), d.z), true);
            under = mix3(under, refl, fres * 0.7f);
            const float fog = 1.f - expf(-water_t * fog_k * water_t * fog_k);
            under = mix3(under, fog_c, sat(fog));
        }
        col = under;
    }

    // crosshair
    const int dx = x - p.w / 2, dy = y - p.h / 2;
    if ((dy == 0 && abs(dx) < 9 && abs(dx) > 2) || (dx == 0 && abs(dy) < 9 && abs(dy) > 2))
        col = f3(1.f, 1.f, 1.f) - col;

    if (sp.dbg && (isnan(col.x) || isnan(col.y) || isnan(col.z))) atomicAdd(sp.dbg, 1ull);
    const uint32_t r = (uint32_t)(sqrtf(sat(col.x)) * 255.f + 0.5f);
    const uint32_t gg = (uint32_t)(sqrtf(sat(col.y)) * 255.f + 0.5f);
    const uint32_t bb = (uint32_t)(sqrtf(sat(col.z)) * 255.f + 0.5f);
    p.out[i] = 0xFF000000u | (r << 16) | (gg << 8) | bb;
}

// --metrics: both tracers per pixel (k_crack in the spec). Counter slots:
enum { MT_PIXELS, MT_TERRAIN, MT_AXIS, MT_LEVEL, MT_FLORA, MT_WATER, MT_SKY, MT_VOXEL, MT_CRACK, MT_CRACK_BELOW,
       MT_CTERRAIN, MT_NAN, MT_COUNT };

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
    const bool nh = sm_trace(p.o, d, p.max_dist, h);
    if (nh && h.kind == HK_TERRAIN) sm_terrain(sp, h, p.o + d * h.t, si);
    atomicAdd(acc + MT_PIXELS, 1ull);
    if (cterr) atomicAdd(acc + MT_CTERRAIN, 1ull);
    // diagnostic image: terrain grey by normal (red = axis-aligned), flora green,
    // voxel mode blue, sky black; silhouette misses magenta, true holes yellow
    float3 col = f3(0.f, 0.f, 0.f);
    if (nh) {
        if (h.kind == HK_TERRAIN) {
            atomicAdd(acc + MT_TERRAIN, 1ull);
            const float m = fmaxf(fabsf(si.n.x), fmaxf(fabsf(si.n.y), fabsf(si.n.z)));
            const float l = 0.25f + 0.6f * sat(dot3(si.n, p.sun));
            col = f3(l, l, l);
            if (m > 0.9995f) { atomicAdd(acc + MT_AXIS, 1ull); col = f3(1.f, 0.f, 0.f); }
            if (si.n.x == 0.f && si.n.z == 0.f) { atomicAdd(acc + MT_LEVEL, 1ull); col = f3(1.f, 0.55f, 0.f); }
            if (isnan(si.n.x) || isnan(si.n.y) || isnan(si.n.z) || isnan(h.t)) atomicAdd(acc + MT_NAN, 1ull);
        } else if (h.kind == HK_VOXEL) {
            atomicAdd(acc + MT_VOXEL, 1ull);
            col = f3(0.2f, 0.3f, 1.f);
        } else {
            atomicAdd(acc + MT_FLORA, 1ull);
            col = f3(0.1f, 0.7f, 0.1f);
        }
    } else {
        atomicAdd(acc + MT_SKY, 1ull);
    }
    if (h.water_t >= 0.f && (!nh || h.water_t < h.t)) atomicAdd(acc + MT_WATER, 1ull);
    if (cterr && hc.t < 0.9f * p.max_dist && !nh) {
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

// ---------------------------------------------------------------------------
struct Renderer {
    size_t        cap;     // pixels
    uint32_t*     d_px;
    uint32_t*     h_px;
    GmePageEntry* d_page;
    size_t        page_bytes;
    uint32_t*     d_tile_tag;
    size_t        tile_tag_bytes;
    cudaStream_t  stream;
    cudaEvent_t   ev[RENDER_PERF_STAGES];  // stage boundaries: uploads, primary (k_render), surface, sun, compose, readback
    float         ms;      // frame kernels (title / renderer_gpu_ms)
    GBuf          g;       // smooth path G-buffer, cap pixels
    // --perf samples: RENDER_PERF_STAGES floats per frame
    float*        perf;
    uint32_t      perf_n, perf_cap;
    float         last[RENDER_PERF_STAGES];
    // the last frame, for --metrics
    SParams       lastp;
    int           have_last;
    unsigned long long* d_acc;
};

static int rck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return 0;
    fprintf(stderr, "render: %s: %s\n", what, cudaGetErrorString(e));
    return 1;
}

static int alloc_frame(Renderer* r, size_t px) {
    cudaFree(r->d_px);
    cudaFreeHost(r->h_px);
    cudaFree(r->g.a);
    cudaFree(r->g.b);
    r->d_px = NULL; r->h_px = NULL; r->g.a = NULL; r->g.b = NULL; r->cap = 0;
    if (rck(cudaMalloc(&r->d_px, px * 4), "alloc frame") ||
        rck(cudaHostAlloc(&r->h_px, px * 4, cudaHostAllocDefault), "alloc frame") ||
        rck(cudaMalloc(&r->g.a, px * sizeof(float4)), "alloc G-buffer") ||
        rck(cudaMalloc(&r->g.b, px * sizeof(uint2)), "alloc G-buffer")) return -1;
    r->cap = px;
    return 0;
}

extern "C" Renderer* renderer_create(const RendererConfig* cfg) {
    Renderer* r = (Renderer*)calloc(1, sizeof(Renderer));
    if (!r) return NULL;
    int prio_least = 0, prio_greatest = 0;          // the render stream gets the greatest priority (spec 6.1)
    if (rck(cudaDeviceGetStreamPriorityRange(&prio_least, &prio_greatest), "stream priorities") ||
        rck(cudaStreamCreateWithPriority(&r->stream, cudaStreamNonBlocking, prio_greatest), "stream") ||
        rck(cudaMalloc(&r->d_acc, sizeof(unsigned long long) * MT_COUNT), "alloc metrics")) {
        renderer_destroy(r);
        return NULL;
    }
    for (int i = 0; i < RENDER_PERF_STAGES; ++i)
        if (rck(cudaEventCreate(&r->ev[i]), "event")) { renderer_destroy(r); return NULL; }
    if (cfg && cfg->max_w > 0 && cfg->max_h > 0 && alloc_frame(r, (size_t)cfg->max_w * cfg->max_h)) {
        renderer_destroy(r);
        return NULL;
    }
    return r;
}

extern "C" void renderer_destroy(Renderer* r) {
    if (!r) return;
    if (r->stream) cudaStreamSynchronize(r->stream);
    cudaFree(r->d_px);
    cudaFreeHost(r->h_px);
    cudaFree(r->d_page);
    cudaFree(r->d_tile_tag);
    cudaFree(r->d_acc);
    cudaFree(r->g.a);
    cudaFree(r->g.b);
    for (int i = 0; i < RENDER_PERF_STAGES; ++i) if (r->ev[i]) cudaEventDestroy(r->ev[i]);
    if (r->stream) cudaStreamDestroy(r->stream);
    free(r->perf);
    free(r);
}

extern "C" float renderer_gpu_ms(const Renderer* r) { return r->ms; }

extern "C" int renderer_draw(Renderer* r, const RenderScene* s, const RenderCamera* c,
                             int w, int h, const uint32_t** pixels) {
    if (!s->pool || w <= 0 || h <= 0) return -1;
    const size_t px = (size_t)w * h;
    if (px > r->cap && alloc_frame(r, px)) return -1;
    cudaEventRecord(r->ev[0], r->stream);
    const size_t pb = sizeof(GmePageEntry) * (size_t)s->page_n * s->page_n * s->layers;
    int upload = s->page_dirty;
    if (pb != r->page_bytes) {
        cudaFree(r->d_page);
        r->d_page = NULL;
        if (rck(cudaMalloc(&r->d_page, pb), "alloc page table")) { r->page_bytes = 0; return -1; }
        r->page_bytes = pb;
        upload = 1;
    }
    if (upload && rck(cudaMemcpyAsync(r->d_page, s->page, pb, cudaMemcpyHostToDevice, r->stream), "upload page"))
        return -1;
    // published column tiles (smooth path): uploaded with the page table whenever they change
    const int smooth_ok = s->tiles && s->tile_tag;
    if (smooth_ok) {
        const size_t tb = sizeof(uint32_t) * (size_t)s->page_n * s->page_n;
        int tup = s->tile_dirty;
        if (tb != r->tile_tag_bytes) {
            cudaFree(r->d_tile_tag);
            r->d_tile_tag = NULL;
            if (rck(cudaMalloc(&r->d_tile_tag, tb), "alloc tile tags")) { r->tile_tag_bytes = 0; return -1; }
            r->tile_tag_bytes = tb;
            tup = 1;
        }
        if (tup && rck(cudaMemcpyAsync(r->d_tile_tag, s->tile_tag, tb, cudaMemcpyHostToDevice, r->stream), "upload tile tags"))
            return -1;
    }

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
    p.w = w; p.h = h;
    p.out = r->d_px;
    p.bio_cells = s->bio_cells;
    p.bio_nreg = s->bio_cells ? (s->bio_nreg < BIOMAP_MAX_REGIONS ? s->bio_nreg : BIOMAP_MAX_REGIONS) : 0;
    p.bio_shift = s->bio_shift;
    memcpy(p.bio_reg, s->bio_reg, sizeof p.bio_reg);
    for (int i = 0; i < BIOMAP_PALETTE; ++i)
        p.bio_pal[i] = make_float3(s->bio_palette[i][0], s->bio_palette[i][1], s->bio_palette[i][2]);

    if (smooth_ok) {   // the column tracer's view of the same frame (also kept for --metrics)
        SParams& q = r->lastp;
        q.k = p;
        TCtx& t = q.t;
        t.page = r->d_page; t.pool = s->pool; t.tiles = s->tiles; t.tile_tag = r->d_tile_tag;
        t.n = s->page_n; t.layers = s->layers; t.min_cy = s->min_cy;
        t.cx = c->cx; t.cy = c->cy; t.cz = c->cz;
        t.ybase_q8 = -256 * GME_CHUNK_DIM * c->cy;
        t.yw = (float)(s->planet.sea_level + 1 - GME_CHUNK_DIM * c->cy);
        t.ytop = (float)(GME_GLOBAL_TOP - GME_CHUNK_DIM * c->cy);
        t.ice = s->planet.ocean_top_mat == MAT_ICE || s->planet.ocean_mat == MAT_ICE;
        t.time = (float)c->frame * (1.f / 60.f);
        t.seed = s->seed;
        t.planet = s->planet;
        t.bio = s->bio;
        q.debug = c->debug;
        q.dbg = NULL;
        r->have_last = 1;
    }

    // classic: k_render alone. smooth: k_primary -> k_sun -> k_compose (wavefront
    // split: the traversal kernels stay small, spec 3.1). One event per stage.
    const dim3 block(16, 8);
    const dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    const int classic = c->quality == RQ_CLASSIC || !smooth_ok;
    cudaEventRecord(r->ev[1], r->stream);
    if (classic) {
        k_render<<<grid, block, 0, r->stream>>>(p);
        cudaEventRecord(r->ev[2], r->stream);
        cudaEventRecord(r->ev[3], r->stream);
        cudaEventRecord(r->ev[4], r->stream);
    } else {
        if (rck(cudaMemcpyToSymbolAsync(g_tc, &r->lastp.t, sizeof(TCtx), 0, cudaMemcpyHostToDevice, r->stream), "tracer context"))
            return -1;
        k_primary<<<grid, block, 0, r->stream>>>(r->lastp, r->g);
        cudaEventRecord(r->ev[2], r->stream);
        k_surface<<<grid, block, 0, r->stream>>>(r->lastp, r->g);
        cudaEventRecord(r->ev[3], r->stream);
        if (p.shadows) k_sun<<<grid, block, 0, r->stream>>>(r->lastp, r->g);
        cudaEventRecord(r->ev[4], r->stream);
        k_compose<<<grid, block, 0, r->stream>>>(r->lastp, r->g);
    }
    cudaEventRecord(r->ev[5], r->stream);
    if (rck(cudaGetLastError(), classic ? "k_render" : "k_primary / k_sun / k_compose") ||
        rck(cudaMemcpyAsync(r->h_px, r->d_px, px * 4, cudaMemcpyDeviceToHost, r->stream), "readback")) return -1;
    cudaEventRecord(r->ev[6], r->stream);
    if (rck(cudaStreamSynchronize(r->stream), "sync")) return -1;
    float st[RENDER_PERF_STAGES] = { 0 };
    for (int i = 0; i < RENDER_PERF_STAGES - 1; ++i) cudaEventElapsedTime(&st[i], r->ev[i], r->ev[i + 1]);
    cudaEventElapsedTime(&st[RENDER_PERF_STAGES - 1], r->ev[0], r->ev[RENDER_PERF_STAGES - 1]);
    cudaEventElapsedTime(&r->ms, r->ev[1], r->ev[5]);
    memcpy(r->last, st, sizeof st);
    if (r->perf_n == r->perf_cap) {
        const uint32_t nc = r->perf_cap ? r->perf_cap * 2 : 4096;
        float* np = (float*)realloc(r->perf, sizeof(float) * RENDER_PERF_STAGES * nc);
        if (np) { r->perf = np; r->perf_cap = nc; }
    }
    if (r->perf_n < r->perf_cap) memcpy(r->perf + (size_t)RENDER_PERF_STAGES * r->perf_n++, r->last, sizeof r->last);
    *pixels = r->h_px;
    return 0;
}

extern "C" int renderer_metrics(Renderer* r, RenderMetrics* out) {
    memset(out, 0, sizeof *out);
    if (!r->have_last) return -1;
    SParams q = r->lastp;
    const dim3 block(16, 8);
    const dim3 grid((q.k.w + block.x - 1) / block.x, (q.k.h + block.y - 1) / block.y);
    unsigned long long acc[MT_COUNT];
    const size_t px = (size_t)q.k.w * q.k.h;
    if (rck(cudaMemsetAsync(r->d_acc, 0, sizeof acc, r->stream), "metrics") ||
        rck(cudaMemcpyToSymbolAsync(g_tc, &r->lastp.t, sizeof(TCtx), 0, cudaMemcpyHostToDevice, r->stream), "tracer context"))
        return -1;
    k_metrics<<<grid, block, 0, r->stream>>>(q, r->d_acc);
    if (rck(cudaMemcpyAsync(r->h_px, r->d_px, px * 4, cudaMemcpyDeviceToHost, r->stream), "metrics image")) return -1;
    q.dbg = r->d_acc + MT_NAN;          // NaN in the shaded colour (re-renders into the device frame only)
    q.debug = RD_NONE;
    k_primary<<<grid, block, 0, r->stream>>>(q, r->g);
    k_surface<<<grid, block, 0, r->stream>>>(q, r->g);
    if (q.k.shadows) k_sun<<<grid, block, 0, r->stream>>>(q, r->g);
    k_compose<<<grid, block, 0, r->stream>>>(q, r->g);
    if (rck(cudaGetLastError(), "k_metrics") ||
        rck(cudaMemcpyAsync(acc, r->d_acc, sizeof acc, cudaMemcpyDeviceToHost, r->stream), "metrics") ||
        rck(cudaStreamSynchronize(r->stream), "metrics")) return -1;
    out->image = r->h_px;
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
    return 0;
}

static int cmp_float(const void* a, const void* b) {
    const float x = *(const float*)a, y = *(const float*)b;
    return (x > y) - (x < y);
}

extern "C" void renderer_perf(const Renderer* r, RenderPerf* out) {
    static const char* names[RENDER_PERF_STAGES] = { "upload", "primary", "surface", "sun", "compose", "readback", "gpu_total" };
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

extern "C" void renderer_perf_reset(Renderer* r) { r->perf_n = 0; }
extern "C" const float* renderer_perf_last(const Renderer* r) { return r->last; }

// trace.cuh - device code of the smooth-surface renderer
// (docs/render-upgrade-spec.md section 3):
//
//   trace_smooth  the 2.5D column tracer. Level 0 walks chunk columns (32 m
//                 tiles), level 1 the 8 m blocks (= flora slots), level 2 the
//                 1 m dual cells, whose corners are column centres; each dual
//                 cell is an exact ray / bilinear-patch quadratic. Trees and
//                 shrubs are analytic (ellipsoid crown + cylinder trunk from the
//                 tile's GmeSlot), kelp is per-column capsules. (The water
//                 plane at sea_level + 1 is one query per primary ray, in
//                 render.cu's sm_trace; M1 keeps the classic see-through look.)
//   trace_voxels  a copy of the classic DDA (render.cu `trace`, which stays
//                 untouched) for caves, a camera below the surface and tiles
//                 that have not streamed in yet.
//
// Coordinates are relative to the camera chunk, as in the classic renderer.
// Included by render.cu after its float3 helpers; it does not touch the
// classic kernel.
#ifndef GME_TRACE_CUH
#define GME_TRACE_CUH

#include "density.h"

#define TR_VOXEL_STEPS 1536   // voxel-mode steps per call (classic MAX_STEPS)

enum { HK_NONE = 0, HK_TERRAIN, HK_CROWN, HK_TRUNK, HK_KELP, HK_VOXEL, HK_ICE };
enum { VX_END = 0, VX_HIT = 1, VX_EXIT = 2 };

struct TCtx {
    const GmePageEntry* page;
    const uint8_t*      pool;
    const GmeColTile*   tiles;
    const uint32_t*     tile_tag;   // device copy of the world's tile_tag
    uint32_t n, layers;
    int32_t  min_cy;
    int32_t  cx, cy, cz;            // camera chunk
    int32_t  ybase_q8;              // -256 * 32 * cy: absolute q8 height -> relative
    float    yw;                    // water plane y = sea_level + 1, relative
    float    ytop;                  // GME_GLOBAL_TOP, relative
    int      ice;                   // the plane is an opaque ice sheet
    float    time;                  // seconds (kelp sway)
    uint32_t seed;
    GmePlanet  planet;
    GmeBioView bio;                 // cells = device pointer or NULL
};

// The tracer's context of the current frame, in constant memory (set per frame
// by the renderer on its stream): every access is a constant-bank operand, so
// the walk spends no registers on it.
__constant__ TCtx g_tc;

struct THit {
    float t;
    int   kind;                     // HK_*
    int   rcx, rcz;                 // tile (chunk column relative to the camera) of terrain / flora / kelp hits
    int   i, j;                     // terrain: dual cell's lower corner (tile array); crown/trunk: i = block; kelp: column
    float u, v;                     // terrain: patch coordinates in [0, 1]
    int   vx, vy, vz, axis, mat;    // voxel hits (relative voxel, face axis), material
    float water_t;                  // first entry into water, -1 = none
    float tice;                     // primary rays: the ice sheet's t (ends the walk), -1 = none
};

// ---- tiles ------------------------------------------------------------------
__device__ __forceinline__ const GmeColTile* tr_tile(int rcx, int rcz) {
    const int32_t ax = g_tc.cx + rcx, az = g_tc.cz + rcz;
    const uint32_t ti = gme_page_index(ax, 0, az, g_tc.n, 1);
    return __ldg(g_tc.tile_tag + ti) == gme_tile_key(ax, az) ? g_tc.tiles + ti : (const GmeColTile*)0;
}

// 256 * (h - h_base) + 2 * dev of a packed column
__device__ __forceinline__ int32_t tr_col_q8(uint32_t cv) {
    return 256 * (int32_t)(cv & 255u) + 2 * ((int32_t)((cv >> 8) ^ 0x80u) - 0x80);
}
// T of tile column k in metres relative to the camera chunk; base = 256 * (h_base + 1) + ybase_q8
__device__ __forceinline__ float tr_T(const GmeColTile* t, int k, int32_t base) {
    return (float)(base + tr_col_q8(__ldg(&t->col[k]))) * (1.f / 256.f);
}
__device__ __forceinline__ int32_t tr_base(const GmeColTile* t) {
    return 256 * ((int32_t)__ldg(&t->h_base) + 1) + g_tc.ybase_q8;
}

// bilinear T at local tile point (lx, lz) in [0, 32]
__device__ __forceinline__ float tr_bilinear(const GmeColTile* t, float lx, float lz) {
    const int DX = min(max((int)floorf(lx - 0.5f), -1), 31), DZ = min(max((int)floorf(lz - 0.5f), -1), 31);
    const float u = fminf(fmaxf(lx - ((float)DX + 0.5f), 0.f), 1.f), v = fminf(fmaxf(lz - ((float)DZ + 0.5f), 0.f), 1.f);
    const int k = (DZ + 2) * GME_TILE_N + DX + 2;
    const int32_t b = tr_base(t);
    const float T00 = tr_T(t, k, b), T10 = tr_T(t, k + 1, b), T01 = tr_T(t, k + GME_TILE_N, b), T11 = tr_T(t, k + GME_TILE_N + 1, b);
    return (T00 + (T10 - T00) * u) + ((T01 + (T11 - T01) * u) - (T00 + (T10 - T00) * u)) * v;
}

// the smooth surface at relative (x, z); false if its tile is not published
__device__ bool tr_surface(float x, float z, float& T) {
    const int rcx = (int)floorf(x * (1.f / 32.f)), rcz = (int)floorf(z * (1.f / 32.f));
    const GmeColTile* t = tr_tile(rcx, rcz);
    if (!t) return false;
    T = tr_bilinear(t, x - (float)(rcx * 32), z - (float)(rcz * 32));
    return true;
}

// ---- exact intersections ------------------------------------------------------
// Smallest s in [0, S] where f(s) = A s^2 + B s + C turns negative (f(0) = C),
// -1 if none: the ray / bilinear-patch hit of one dual cell.
__device__ __forceinline__ float tr_patch_root(float A, float B, float C, float S) {
    if (C < 0.f) return 0.f;
    const float fS = (A * S + B) * S + C;
    if (fS >= 0.f) {                                   // both ends above: only a convex dip can cross
        if (!(A > 0.f)) return -1.f;
        const float sv = -B / (2.f * A);
        if (!(sv > 0.f && sv < S) || C + 0.5f * B * sv >= 0.f) return -1.f;
    }
    const float disc = fmaxf(B * B - 4.f * A * C, 0.f);
    const float q = -0.5f * (B + copysignf(sqrtf(disc), B));
    float r = 3.0e38f;
    if (q != 0.f) { const float r2 = C / q; if (r2 >= 0.f) r = r2; }
    if (A != 0.f) { const float r1 = q / A; if (r1 >= 0.f && r1 < r) r = r1; }
    if (r == 3.0e38f) return fS < 0.f ? S * C / (C - fS) : -1.f;   // degenerate: secant estimate
    if (r > S) return fS < 0.f ? S : -1.f;                          // rounding at the far end
    return r;
}

// ray from p (t' = 0) against the ellipsoid around ce with inverse radii ia (x, z), ib (y);
// entry t' in [0, tmax] (0 if p is inside), -1 none
__device__ __forceinline__ float tr_hit_ellipsoid(float3 p, float3 d, float3 ce, float ia, float ib, float tmax) {
    const float px = (p.x - ce.x) * ia, py = (p.y - ce.y) * ib, pz = (p.z - ce.z) * ia;
    const float qx = d.x * ia, qy = d.y * ib, qz = d.z * ia;
    const float qq = qx * qx + qy * qy + qz * qz, pq = px * qx + py * qy + pz * qz;
    const float pp = px * px + py * py + pz * pz - 1.f;
    const float disc = pq * pq - qq * pp;
    if (disc < 0.f) return -1.f;
    const float sq = sqrtf(disc);
    const float tin = (-pq - sq) / qq, tout = (-pq + sq) / qq;
    if (tout < 0.f || tin > tmax) return -1.f;
    return fmaxf(tin, 0.f);
}

// vertical cylinder around (ax, az) of radius r between y0 and y1 (no caps)
__device__ __forceinline__ float tr_hit_cylinder(float3 p, float3 d, float ax, float az, float r, float y0, float y1, float tmax) {
    const float px = p.x - ax, pz = p.z - az;
    const float qq = d.x * d.x + d.z * d.z;
    if (qq < 1e-12f) return -1.f;
    const float pq = px * d.x + pz * d.z, pp = px * px + pz * pz - r * r;
    const float disc = pq * pq - qq * pp;
    if (disc < 0.f) return -1.f;
    const float tin = (-pq - sqrtf(disc)) / qq;
    if (tin < 0.f || tin > tmax) return -1.f;
    const float y = p.y + d.y * tin;
    return (y >= y0 && y <= y1) ? tin : -1.f;
}

// capsule with segment pa..pb and radius r (I. Quilez's ray-capsule)
__device__ __forceinline__ float tr_hit_capsule(float3 ro, float3 rd, float3 pa, float3 pb, float r, float tmax) {
    const float3 ba = pb - pa, oa = ro - pa;
    const float baba = dot3(ba, ba), bard = dot3(ba, rd), baoa = dot3(ba, oa);
    const float rdoa = dot3(rd, oa), oaoa = dot3(oa, oa);
    const float a = baba - bard * bard;
    float b = baba * rdoa - baoa * bard;
    float cc = baba * oaoa - baoa * baoa - r * r * baba;
    float h = b * b - a * cc;
    float t = -1.f;
    if (h >= 0.f) {
        if (a > 1e-12f) {
            t = (-b - sqrtf(h)) / a;
            const float y = baoa + t * bard;
            if (!(y > 0.f && y < baba)) t = -1.f;     // not the body: try the caps
            else if (t < 0.f || t > tmax) return -1.f;
        }
        if (t < 0.f) {
            const float y = baoa + fmaxf(t, 0.f) * bard;
            const float3 oc = (y <= 0.f) ? oa : ro - pb;
            b = dot3(rd, oc);
            cc = dot3(oc, oc) - r * r;
            h = b * b - cc;
            if (h > 0.f) {
                t = -b - sqrtf(h);
                if (t < 0.f || t > tmax) t = -1.f;
            } else {
                t = -1.f;
            }
        }
    }
    return t;
}

// kelp strand of tile column k (absolute x, z): capsule axis bottom/top (relative). false = none
__device__ __forceinline__ bool tr_kelp(const GmeColTile* t, int k, int32_t x, int32_t z, uint32_t cell,
                                        float3& pa, float3& pb) {
    const int32_t h = gme_tile_h(t, k);
    const int32_t len = gme_kelp_len(x, z, h, cell, g_tc.seed, &g_tc.planet);
    if (len <= 0) return false;
    const float y0 = tr_T(t, k, tr_base(t));
    const float y1 = (float)(h + len + 1 - 32 * g_tc.cy);
    float a = y0 + GME_KELP_R, b = y1 - GME_KELP_R;
    if (a > b) a = b = 0.5f * (y0 + y1);
    const float axx = (float)(x - 32 * g_tc.cx) + 0.5f, azz = (float)(z - 32 * g_tc.cz) + 0.5f;
    // visual-only sway of the tip, < 0.3 m so the capsule stays inside its column
    const uint32_t hs = gme_hash3(x, 7, z, g_tc.seed ^ 0x5EA7u);
    const float ph = (float)(hs & 1023u) * (6.2831853f / 1024.f);
    const float sw = 0.22f * __sinf(g_tc.time * 0.9f + ph), sw2 = 0.12f * __cosf(g_tc.time * 0.7f + 1.7f * ph);
    pa = f3(axx, a, azz);
    pb = f3(axx + sw, b, azz + sw2);
    return true;
}

// ---- voxel mode -------------------------------------------------------------------
__device__ __forceinline__ int tr_lookup(int rcx, int rcy, int rcz) {
    const int layer = g_tc.cy + rcy - g_tc.min_cy;
    if (layer < 0) return -2 - MAT_STONE;            // below the world: solid
    if (layer >= (int)g_tc.layers) return -2 - MAT_AIR; // above: sky
    const int ax = g_tc.cx + rcx, az = g_tc.cz + rcz;
    const GmePageEntry e = g_tc.page[gme_page_index(ax, layer, az, g_tc.n, g_tc.layers)];
    return e.tag == gme_page_tag(ax, az) ? e.val : GME_PAGE_UNLOADED;
}

// A copy of the classic DDA over [t0, t1]. flora_solid: LEAF, WOOD and KELP are
// solid voxels (tiles not streamed in yet: drawn as today) or ignored (caves:
// they are drawn analytically). check_exit: leave when the ray is 0.5 m above
// the smooth surface again (camera underground, cave mouths). Water is
// transparent and sets h.water_t. Returns VX_HIT, VX_EXIT (t_out) or VX_END.
__device__ __forceinline__ int trace_voxels(float3 o, float3 d, float t0, float t1, bool flora_solid,
                                         bool check_exit, THit& h, float& t_out) {
    if (fabsf(d.x) < 1e-7f) d.x = copysignf(1e-7f, d.x);
    if (fabsf(d.y) < 1e-7f) d.y = copysignf(1e-7f, d.y);
    if (fabsf(d.z) < 1e-7f) d.z = copysignf(1e-7f, d.z);
    const float ix = 1.f / d.x, iy = 1.f / d.y, iz = 1.f / d.z;
    const int sx = d.x > 0 ? 1 : -1, sy = d.y > 0 ? 1 : -1, sz = d.z > 0 ? 1 : -1;
    const float tdx = fabsf(ix), tdy = fabsf(iy), tdz = fabsf(iz);
    const int top = (g_tc.min_cy + (int)g_tc.layers - g_tc.cy) * GME_CHUNK_DIM;

    const float ts = t0 + 1e-4f;
    int vx = (int)floorf(o.x + d.x * ts), vy = (int)floorf(o.y + d.y * ts), vz = (int)floorf(o.z + d.z * ts);
    float tmx = ((float)(vx + (sx > 0)) - o.x) * ix;
    float tmy = ((float)(vy + (sy > 0)) - o.y) * iy;
    float tmz = ((float)(vz + (sz > 0)) - o.z) * iz;
    float t = t0;
    int axis = 1;

    for (int i = 0; i < TR_VOXEL_STEPS; ++i) {
        if (t >= t1) return VX_END;                    // [t0, t1): the next column / tile starts at t1
        if (vy >= top && d.y >= 0.f) return VX_END;
        const int rcx = vx >> 5, rcy = vy >> 5, rcz = vz >> 5;
        const int e = tr_lookup(rcx, rcy, rcz);
        int mat;
        if (e >= 0) mat = g_tc.pool[(size_t)e * GME_CHUNK_VOXELS + GME_VOXEL_INDEX(vx & 31, vy & 31, vz & 31)];
        else mat = (e == GME_PAGE_UNLOADED) ? MAT_AIR : (-2 - e);

        if (mat == MAT_WATER) {
            if (h.water_t < 0.f) h.water_t = t;
        } else if (mat != MAT_AIR && (flora_solid || !gme_is_standing_flora((uint32_t)mat))) {
            h.t = t; h.kind = HK_VOXEL; h.mat = mat; h.axis = axis; h.vx = vx; h.vy = vy; h.vz = vz;
            return VX_HIT;
        }
        if (check_exit && i > 0) {
            float T;
            const float3 q = o + d * t;
            if (tr_surface(q.x, q.z, T) && q.y > T + 0.5f) { t_out = t; return VX_EXIT; }
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
    return VX_END;
}

// Out-of-line voxel mode: the primary ray's rare entries (camera below the
// surface, cave mouths) and its missing tiles. Sun rays inline their own copy
// (trace_smooth); for k_primary the call keeps the walk's registers lower.
__device__ __noinline__ int trace_voxels_call(float3 o, float3 d, float t0, float t1, bool flora_solid, bool check_exit,
                                              THit& h, float& t_out) {
    return trace_voxels(o, d, t0, t1, flora_solid, check_exit, h, t_out);
}

// ---- column mode --------------------------------------------------------------------
// The walk keeps little state: cell indices and entry t per level. Each exit
// t is recomputed from its cell index (exact boundaries, no accumulated DDA
// error, and no stepping state in registers). Coordinates inside a tile are
// tile-local: ox, oz = ray origin - tile corner.

// t at which the ray leaves cell i (cells of size S starting at `off`) in local
// coordinates with origin oo, inverse direction idd, moving toward + if pos
__device__ __forceinline__ float tr_exit(int i, float S, float off, float oo, float idd, bool pos) {
    return ((float)(i + (pos ? 1 : 0)) * S + off - oo) * idd;
}

// Shrub / tree of block b over [tb, tend]: analytic crown + trunk, shifted onto
// T by the trunk column's deviation, in tile-local coordinates (ox, oz = ray
// origin - tile corner). Returns the nearer hit's kind and lowers tend to it;
// HK_NONE if neither is hit.
__device__ __forceinline__ int tr_flora(const GmeColTile* tile, int b, int rcx, int rcz,
                                        float ox, float oy, float oz, float3 d, float tb, float& tend) {
    const GmeSlot* s = &tile->slot[b];
    const int32_t trad = __ldg(&s->trad), tcy = __ldg(&s->tcy), tvr = __ldg(&s->tvr);
    const int32_t lx = __ldg(&s->tx) - (g_tc.cx + rcx) * 32, lz = __ldg(&s->tz) - (g_tc.cz + rcz) * 32;   // 0..31
    const float dl = (float)((int32_t)((__ldg(&tile->col[(lz + GME_TILE_APRON) * GME_TILE_N + lx + GME_TILE_APRON]) >> 8) ^ 0x80u) - 0x80) *
                     (1.f / GME_DEV_UNIT);
    const float3 pb0 = f3(ox + d.x * tb, oy + d.y * tb, oz + d.z * tb);   // segment start: well-conditioned quadratics
    const float3 ce = f3((float)lx + 0.5f, (float)(tcy - 32 * g_tc.cy) + 0.5f + dl, (float)lz + 0.5f);
    int fk = HK_NONE;
    const float tf = tr_hit_ellipsoid(pb0, d, ce, rsqrtf((float)(trad * trad + trad)), rsqrtf((float)(tvr * tvr + tvr)), tend - tb);
    if (tf >= 0.f) { tend = tb + tf; fk = HK_CROWN; }
    const float y0 = (float)(__ldg(&s->tbase) + 1 - 32 * g_tc.cy) + dl, y1 = (float)(tcy + 1 - 32 * g_tc.cy) + dl;
    const float tt = tr_hit_cylinder(pb0, d, ce.x, ce.z, gme_trunk_radius(trad), y0, y1, tend - tb);
    if (tt >= 0.f) { tend = tb + tt; fk = HK_TRUNK; }
    return fk;
}

// Kelp of block b over [tb, tend] (tile-local origin ox, oy, oz): a walk over
// the block's columns along the ray; each column's strand is a capsule inside
// its own 1 m footprint, so the first capsule hit in walk order is the nearest.
// Lowers tend and returns true (h.i, h.j = the strand's column) on a hit.
// A pass of its own, before the dual cells, so its temporaries never share
// registers with the terrain walk.
__device__ __forceinline__ bool tr_kelp_block(const GmeColTile* tile, int b, int rcx, int rcz, float ox, float oy, float oz,
                                              float3 d, float ix, float iz, float tb, float& tend, THit& h) {
    const bool px = d.x > 0.f, pz = d.z > 0.f;
    const int bx8 = (b & 3) * 8, bz8 = (b >> 2) * 8;
    const uint32_t cell = __ldg(&tile->slot[b].cell);
    int lx = min(max((int)floorf(ox + d.x * tb), bx8), bx8 + 7), lz = min(max((int)floorf(oz + d.z * tb), bz8), bz8 + 7);
    float s0 = tb;
    for (;;) {
        const float ex = tr_exit(lx, 1.f, 0.f, ox, ix, px), ez = tr_exit(lz, 1.f, 0.f, oz, iz, pz);
        const float s1 = fminf(fminf(ex, ez), tend);
        if (s1 > s0 && fminf(oy + d.y * s0, oy + d.y * s1) < g_tc.yw) {
            float3 pa, pb;
            if (tr_kelp(tile, (lz + GME_TILE_APRON) * GME_TILE_N + lx + GME_TILE_APRON, (g_tc.cx + rcx) * 32 + lx,
                        (g_tc.cz + rcz) * 32 + lz, cell, pa, pb)) {
                // tr_kelp works relative to the camera chunk: shift the segment start there
                const float3 p0 = f3(ox + (float)(rcx * 32) + d.x * s0, oy + d.y * s0, oz + (float)(rcz * 32) + d.z * s0);
                const float tk = tr_hit_capsule(p0, d, pa, pb, GME_KELP_R, tend - s0);
                if (tk >= 0.f) {
                    tend = s0 + tk;
                    h.i = lx + GME_TILE_APRON; h.j = lz + GME_TILE_APRON;
                    return true;
                }
            }
        }
        if (s1 >= tend) break;
        if (ex < ez) lx += px ? 1 : -1; else lz += pz ? 1 : -1;
        s0 = s1;
        if ((unsigned)(lx - bx8) > 7u || (unsigned)(lz - bz8) > 7u) break;
    }
    return false;
}

// One 8 m block (index b) of a tile over the ray segment [tb, tbe]: flora of
// its slot, its kelp, then its dual cells. Tile-local origin (ox, oy, oz).
// true = hit (h filled). (The water plane is one query per ray: sm_trace.)
__device__ __forceinline__ bool tr_block(const GmeColTile* tile, int b, int rcx, int rcz,
                                         float ox, float oy, float oz, float3 d, float ix, float iz,
                                         float tb, float tbe, THit& h) {
    const bool px = d.x > 0.f, pz = d.z > 0.f;
    float tend = tbe;
    int fk = __ldg(&tile->slot[b].trad) > 0 ? tr_flora(tile, b, rcx, rcz, ox, oy, oz, d, tb, tend) : HK_NONE;
    if (BIOCELL_KIND(__ldg(&tile->slot[b].cell)) == BIO_KIND_KELP && fminf(oy + d.y * tb, oy + d.y * tend) < g_tc.yw &&
        tr_kelp_block(tile, b, rcx, rcz, ox, oy, oz, d, ix, iz, tb, tend, h))
        fk = HK_KELP;
    const int32_t base = tr_base(tile);

    // dual cells (corners at column centres) along [tb, tend]: exact ray / bilinear-patch hits
    int DX = min(max((int)floorf(ox + d.x * tb - 0.5f), (b & 3) * 8 - 1), (b & 3) * 8 + 7);
    int DZ = min(max((int)floorf(oz + d.z * tb - 0.5f), (b >> 2) * 8 - 1), (b >> 2) * 8 + 7);
    float s0 = tb;
    for (;;) {
        const float s1 = fminf(fminf(tr_exit(DX, 1.f, 0.5f, ox, ix, px), tr_exit(DZ, 1.f, 0.5f, oz, iz, pz)), tend);
        if (s1 > s0) {
            const int k = (DZ + GME_TILE_APRON) * GME_TILE_N + DX + GME_TILE_APRON;
            const float ya = oy + d.y * s0;
            const float T00 = tr_T(tile, k, base), T10 = tr_T(tile, k + 1, base);
            const float T01 = tr_T(tile, k + GME_TILE_N, base), T11 = tr_T(tile, k + GME_TILE_N + 1, base);
            if (fminf(ya, oy + d.y * s1) <= fmaxf(fmaxf(T00, T10), fmaxf(T01, T11))) {
                const float bb = T10 - T00, cc = T01 - T00, kk = T00 - T10 - T01 + T11;
                const float u0 = ox + d.x * s0 - ((float)DX + 0.5f), v0 = oz + d.z * s0 - ((float)DZ + 0.5f);
                const float A = -kk * d.x * d.z;
                const float B = d.y - bb * d.x - cc * d.z - kk * (u0 * d.z + v0 * d.x);
                const float C = ya - (T00 + bb * u0 + cc * v0 + kk * u0 * v0);
                const float r = tr_patch_root(A, B, C, s1 - s0);
                if (r >= 0.f) {
                    h.t = s0 + r; h.kind = HK_TERRAIN; h.rcx = rcx; h.rcz = rcz;
                    h.i = DX + GME_TILE_APRON; h.j = DZ + GME_TILE_APRON;
                    h.u = fminf(fmaxf(u0 + d.x * r, 0.f), 1.f);
                    h.v = fminf(fmaxf(v0 + d.z * r, 0.f), 1.f);
                    return true;
                }
            }
        }
        if (s1 >= tend) break;
        if (tr_exit(DX, 1.f, 0.5f, ox, ix, px) < tr_exit(DZ, 1.f, 0.5f, oz, iz, pz)) DX += px ? 1 : -1;
        else DZ += pz ? 1 : -1;
        s0 = s1;
        if ((unsigned)(DX - (b & 3) * 8 + 1) > 8u || (unsigned)(DZ - (b >> 2) * 8 + 1) > 8u) break;
    }
    if (fk != HK_NONE) {
        h.t = tend; h.kind = fk; h.rcx = rcx; h.rcz = rcz;
        if (fk != HK_KELP) h.i = b;
        return true;
    }
    return false;
}

enum { TS_MISS = 0, TS_HIT = 1, TS_VOXEL = 2 };

// Column-mode walk over [t, tmax]: tiles -> blocks -> dual cells. Returns
// TS_HIT (h filled), TS_MISS, or TS_VOXEL with [t, te] = a tile that is not
// published, which the caller crosses in voxel mode (no call inside this loop,
// which keeps the walk's registers low).
__device__ __forceinline__ int tr_columns(float3 o, float3 d, float& t, float& te, float tmax, THit& h) {
    const float ix = 1.f / d.x, iz = 1.f / d.z;
    const bool px = d.x > 0.f, pz = d.z > 0.f;
    const float tn = t + 1e-3f + t * 1e-6f;            // start tile: a point just past t, never the one behind
    int rcx = (int)floorf((o.x + d.x * tn) * (1.f / 32.f)), rcz = (int)floorf((o.z + d.z * tn) * (1.f / 32.f));
    float ox = o.x - (float)(rcx * 32), oz = o.z - (float)(rcz * 32);   // origin, tile-local
    while (t < tmax) {                                 // te > t every step but at a corner: always ends
        te = fminf(fminf(tr_exit(0, 32.f, 0.f, ox, ix, px), tr_exit(0, 32.f, 0.f, oz, iz, pz)), tmax);
        const float y0 = o.y + d.y * t;
        if (d.y >= 0.f && y0 > g_tc.ytop) return TS_MISS;            // above everything, climbing
        if (te > t) {
            const GmeColTile* tile = tr_tile(rcx, rcz);
            if (!tile) return TS_VOXEL;                               // not streamed in yet: voxels, as today
            if (fminf(y0, o.y + d.y * te) <= (float)(__ldg(&tile->top) - 32 * g_tc.cy)) {
                int b = min(max((int)floorf((oz + d.z * t) * 0.125f), 0), 3) * 4 +
                        min(max((int)floorf((ox + d.x * t) * 0.125f), 0), 3);
                float tb = t;
                for (;;) {
                    const float tbe = fminf(fminf(tr_exit(b & 3, 8.f, 0.f, ox, ix, px), tr_exit(b >> 2, 8.f, 0.f, oz, iz, pz)), te);
                    if (tbe > tb && fminf(o.y + d.y * tb, o.y + d.y * tbe) <= (float)(__ldg(&tile->blk_top[b]) - 32 * g_tc.cy) &&
                        tr_block(tile, b, rcx, rcz, ox, o.y, oz, d, ix, iz, tb, tbe, h))
                        return TS_HIT;
                    if (tbe >= te) break;
                    const bool sx = tr_exit(b & 3, 8.f, 0.f, ox, ix, px) < tr_exit(b >> 2, 8.f, 0.f, oz, iz, pz);
                    const int nb = sx ? (b & 3) + (px ? 1 : -1) : (b >> 2) + (pz ? 1 : -1);
                    if ((unsigned)nb > 3u) break;
                    b = sx ? (b & 12) | nb : (b & 3) | (nb << 2);
                    tb = tbe;
                }
            }
        }
        if (tr_exit(0, 32.f, 0.f, ox, ix, px) < tr_exit(0, 32.f, 0.f, oz, iz, pz)) { rcx += px ? 1 : -1; ox -= px ? 32.f : -32.f; }
        else                                                                  { rcz += pz ? 1 : -1; oz -= pz ? 32.f : -32.f; }
        t = te;
    }
    return TS_MISS;
}

// The column tracer over [t0, tmax], tiles that have not streamed in crossed in
// voxel mode (where WATER voxels set h.water_t). Returns the nearest hit, so it
// serves sun rays (any-hit) as well. true = hit (h filled).
template <bool kInlineVoxels>
__device__ __forceinline__ bool trace_smooth_t(float3 o, float3 d, float t0, float tmax, THit& h) {
    if (fabsf(d.x) < 1e-7f) d.x = copysignf(1e-7f, d.x);
    if (fabsf(d.z) < 1e-7f) d.z = copysignf(1e-7f, d.z);
    for (int k = 0; k < 256 && t0 < tmax; ++k) {
        float te = tmax;
        const int r = tr_columns(o, d, t0, te, tmax, h);
        if (r != TS_VOXEL) return r == TS_HIT;
        float tx;
        if ((kInlineVoxels ? trace_voxels(o, d, t0, te, true, false, h, tx)
                           : trace_voxels_call(o, d, t0, te, true, false, h, tx)) == VX_HIT) return true;
        t0 = te;
    }
    return false;
}

// sun rays: everything inline (k_sun keeps its hit record in registers)
__device__ __forceinline__ bool trace_smooth(float3 o, float3 d, float t0, float tmax, THit& h) {
    return trace_smooth_t<true>(o, d, t0, tmax, h);
}

#endif

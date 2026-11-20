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
//                 untouched) for caves and a camera below the surface (and,
//                 without the far field, tiles that have not streamed in).
//   M3 distance   past a distance tl the walk uses each tile's 4 m lattice
//                 (lod4, bilinear 4 m cells; flora to tf); near the loaded
//                 radius ("ring") the lattice morphs into the far field. From
//                 the ring on, and over tiles that have not streamed in, tr_far
//                 marches the renderer's far-field clipmap (L1 16 m to 8 km,
//                 L2 64 m to 32 km; min / max mips; exact patch hits including
//                 the curvature drop c2 (r^2 - max_dist^2)).
//   M5 life       flora modes of the walk (FM_*): primary rays see clumped
//                 crowns (the ellipsoid displaced inward by billowed value
//                 noise, sphere-traced in the shell; tr_clump), High's shadow
//                 rays a leaf-area medium (e^-(sigma chord) per crown, FM_LAD),
//                 the rest the plain ellipsoid. Kelp strands are a thin stipe
//                 carrying flat blades (tr_kelp_strand), all inside the column.
//   version 2     crowns of a biomap with morphology (flora.h): ellipsoid,
//                 cone, umbrella, column or dome by the slot's shape word, the
//                 float solids of density.h (GmeCrownQ) in every flora mode
//                 (tr_flora2); porosity and leaf angle shape the leaf medium.
//
// Coordinates are relative to the camera chunk, as in the classic renderer.
// Included by render.cu after its float3 helpers; it does not touch the
// classic kernel.
#ifndef GME_TRACE_CUH
#define GME_TRACE_CUH

#include "density.h"

#define TR_VOXEL_STEPS 1536   // voxel-mode steps per call (classic MAX_STEPS)
#define TR_FAR_STEPS   2048   // far-field march steps per call

// hit kinds (4 bits in the hit record and the G-buffer). LOD: the 4 m lattice
// of a published tile; FAR: the far-field clipmap; DIST: below the geometric
// horizon beyond the far field (shaded with the sky-view LUT's ground).
// HK_FAUNA: a creature's sphere (render.cu k_fauna_primary; never produced by the walk itself).
enum { HK_NONE = 0, HK_TERRAIN, HK_CROWN, HK_TRUNK, HK_KELP, HK_VOXEL, HK_ICE, HK_LOD, HK_FAR, HK_DIST, HK_FAUNA };
enum { VX_END = 0, VX_HIT = 1, VX_EXIT = 2 };

// One far-field clipmap level (spec 3.8): GME_FAR_N^2 packed samples
// (density.h gme_far_sample) in toroidal storage (absolute sample index & 1023),
// min / max mips over 8 x 8 (A) and 64 x 64 (B) storage samples, each including
// the next row and column, so a mip cell bounds every bilinear patch of its
// samples. The tracer works in "frame" sample indices: frame (0, 0) is the
// absolute sample F (a multiple of 64, so mip cells are frame-aligned), at
// (ox, oz) metres relative to the camera chunk.
struct FarLevel {
    const uint32_t* s;          // samples, [z & 1023][x & 1023]
    const short2*   mA;         // 128 x 128 cells: x = max, y = min (q4, absolute)
    const short2*   mB;         // 16 x 16 cells
    float   ox, oz;             // metres of frame sample (0, 0), relative to the camera chunk
    int32_t bx, bz;             // storage index of frame sample (0, 0)
    float   S, iS;              // spacing and its inverse
};

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
    // M3: LOD and far field (spec 3.8). Distances are horizontal, from the camera.
    int      far_on;                // the far field exists (else: M2 behaviour, missing tiles in voxel mode)
    FarLevel far[GME_FAR_LEVELS];
    const int* far_max;             // device: [0], [1] level max (q4, absolute); [2] near-walk ceiling (metres, absolute)
    float    ybase_m;               // -32 cy: absolute metres -> relative
    float    camx, camz;            // the camera, relative to its chunk
    float    ring;                  // near (tiles) / far handover radius (max_dist; --far-ring overrides)
    float    ring_m0, ring_iw;      // LOD -> L1 morph: weight (r - ring_m0) * ring_iw, 1 from ring - 6 m on
    float    r1, r1_m0, r1_iw;      // L1 / L2 boundary and the L1 -> L2 morph, 1 from r1 - 23 m on
    float    r2;                    // far field edge
    float    c2;                    // curvature: drop c2 (r^2 - drop_r2), with c2 = 1 / (2 R_p)
    float    drop_r2;               // max_dist^2: the drop is 0 inside the loaded radius
    float    lod_fine;              // primary rays: fine surface up to this distance (dithered +-5 % per pixel)
    float    flora_far;             // primary rays: analytic flora on LOD tiles up to this distance
    float    dith;                  // M4: temporal offset of the per-pixel dithers (frac(0.618 frame); 0 = static)
    float    pix;                   // M5: radians per pixel (clumped crowns fade out where clumps are sub-pixel)
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
    float tau;                      // M5 LAD shadow rays (FM_LAD): optical depth of the crowns crossed so far
};

// M5 flora modes of the walk: crowns as ellipsoids (shadow, GI, reflection
// rays), clumped crowns (primary rays: the ellipsoid displaced inward by a 3D
// value noise, sphere-traced in the shell), or a leaf-area medium (High's
// shadow rays: crowns attenuate by e^-(sigma chord) instead of blocking)
enum { FM_PLAIN = 0, FM_CLUMP = 1, FM_LAD = 2 };
#define TR_LAD_OPAQUE 7.f           // optical depth from which a LAD ray counts as blocked (T < 1e-3)

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

// ---- far field and LOD (M3, spec 3.8) -----------------------------------------------
// Surfaces by horizontal distance r from the camera:
//   r < ring      published tiles: the fine 1 m surface (primary rays to lod_fine,
//                 shadow rays to 32 m), then the 4 m lattice (lod4). In
//                 [ring_m0, ring] the lattice corners move to the L1 surface, so
//                 the near field meets the far field without a step.
//   ring .. r1    L1 (16 m); in [r1_m0, r1] its corners move to L2.
//   r1 .. r2      L2 (64 m).
// Beyond the loaded radius the far field drops by the curvature
// c2 (r^2 - max_dist^2): the drop and its slope are 0 at the loaded radius.
struct FarCell { float h00, h10, h01, h11; };

__device__ __forceinline__ float bil4(const FarCell& c, float a, float b) {
    const float lo = c.h00 + (c.h10 - c.h00) * a, hi = c.h01 + (c.h11 - c.h01) * a;
    return lo + (hi - lo) * b;
}
__device__ __forceinline__ uint32_t far_raw(const FarLevel& L, int fi, int fj) {   // frame sample (fi, fj)
    const uint32_t ix = (uint32_t)(L.bx + fi) & (GME_FAR_N - 1u), iz = (uint32_t)(L.bz + fj) & (GME_FAR_N - 1u);
    return __ldg(L.s + ((size_t)iz << 10) + ix);
}
__device__ __forceinline__ float far_hm(uint32_t v) { return (float)(int16_t)(v & 0xFFFFu) * (1.f / 16.f) + g_tc.ybase_m; }
__device__ __forceinline__ float far_h(const FarLevel& L, int fi, int fj) { return far_hm(far_raw(L, fi, fj)); }
__device__ __forceinline__ float far_r(float x, float z) {
    const float dx = x - g_tc.camx, dz = z - g_tc.camz;
    return sqrtf(dx * dx + dz * dz);
}
__device__ __forceinline__ float far_w1(float r) { return sat((r - g_tc.r1_m0) * g_tc.r1_iw); }    // L1 -> L2
__device__ __forceinline__ float lod_w(float r) { return sat((r - g_tc.ring_m0) * g_tc.ring_iw); }  // LOD -> L1
__device__ __forceinline__ float far_drop(float x, float z) {
    const float dx = x - g_tc.camx, dz = z - g_tc.camz;
    return g_tc.c2 * fmaxf(dx * dx + dz * dz - g_tc.drop_r2, 0.f);
}

// The 4 samples of level M's cell that contains the point (x0, z0) + 1/8 cell,
// and the point's cell coordinates. Used for corners of finer cells nested in it.
__device__ __forceinline__ void far_parent(const FarLevel& M, float x0, float z0, FarCell& m, float& a, float& b) {
    const float u = (x0 - M.ox) * M.iS, v = (z0 - M.oz) * M.iS;
    const int fi = (int)floorf(u + 0.125f), fj = (int)floorf(v + 0.125f);
    a = u - (float)fi; b = v - (float)fj;
    m.h00 = far_h(M, fi, fj); m.h10 = far_h(M, fi + 1, fj); m.h01 = far_h(M, fi, fj + 1); m.h11 = far_h(M, fi + 1, fj + 1);
}

// Corners (relative metres, no drop) of frame cell (fi, fj) of level l. L1
// cells in the L1 -> L2 band morph each corner toward the L2 surface by its own
// distance, so neighbouring cells share corners and the surface stays bilinear.
__device__ __forceinline__ void far_cell(int l, int fi, int fj, FarCell& c) {
    const FarLevel& L = g_tc.far[l];
    c.h00 = far_h(L, fi, fj); c.h10 = far_h(L, fi + 1, fj); c.h01 = far_h(L, fi, fj + 1); c.h11 = far_h(L, fi + 1, fj + 1);
    if (l != 0) return;
    const float x0 = L.ox + (float)fi * L.S, z0 = L.oz + (float)fj * L.S;
    if (far_r(x0 + 0.5f * L.S, z0 + 0.5f * L.S) + 0.71f * L.S <= g_tc.r1_m0) return;
    FarCell m;
    float a, b;
    far_parent(g_tc.far[1], x0, z0, m, a, b);                   // an L1 cell is a quarter of an L2 cell
    c.h00 += (bil4(m, a, b) - c.h00) * far_w1(far_r(x0, z0));
    c.h10 += (bil4(m, a + 0.25f, b) - c.h10) * far_w1(far_r(x0 + L.S, z0));
    c.h01 += (bil4(m, a, b + 0.25f) - c.h01) * far_w1(far_r(x0, z0 + L.S));
    c.h11 += (bil4(m, a + 0.25f, b + 0.25f) - c.h11) * far_w1(far_r(x0 + L.S, z0 + L.S));
}
// far-field surface (no drop) at relative (x, z); the level by distance
__device__ float far_surface(float x, float z) {
    const int l = far_r(x, z) < g_tc.r1 ? 0 : 1;
    const FarLevel& L = g_tc.far[l];
    const float u = (x - L.ox) * L.iS, v = (z - L.oz) * L.iS;
    const int fi = (int)floorf(u), fj = (int)floorf(v);
    FarCell c;
    far_cell(l, fi, fj, c);
    return bil4(c, u - (float)fi, v - (float)fj);
}

// 4 m LOD cell (CX, CZ) of a tile (corners on the lod4 lattice), relative
// metres. Near the ring the corners move to the L1 surface (an LOD cell is a
// sixteenth of an L1 cell), reaching it 6 m before the ring.
__device__ __forceinline__ void lod_cell(const GmeColTile* tile, int rcx, int rcz, int CX, int CZ, FarCell& c) {
    const int k = CZ * GME_LOD_N + CX;
    c.h00 = (float)__ldg(&tile->lod4[k]) * (1.f / 16.f) + g_tc.ybase_m;
    c.h10 = (float)__ldg(&tile->lod4[k + 1]) * (1.f / 16.f) + g_tc.ybase_m;
    c.h01 = (float)__ldg(&tile->lod4[k + GME_LOD_N]) * (1.f / 16.f) + g_tc.ybase_m;
    c.h11 = (float)__ldg(&tile->lod4[k + GME_LOD_N + 1]) * (1.f / 16.f) + g_tc.ybase_m;
    if (!g_tc.far_on) return;
    const float x0 = (float)(rcx * 32 + 4 * CX), z0 = (float)(rcz * 32 + 4 * CZ);
    if (far_r(x0 + 2.f, z0 + 2.f) + 2.9f <= g_tc.ring_m0) return;
    FarCell m;
    float a, b;
    far_parent(g_tc.far[0], x0, z0, m, a, b);
    c.h00 += (bil4(m, a, b) - c.h00) * lod_w(far_r(x0, z0));
    c.h10 += (bil4(m, a + 0.25f, b) - c.h10) * lod_w(far_r(x0 + 4.f, z0));
    c.h01 += (bil4(m, a, b + 0.25f) - c.h01) * lod_w(far_r(x0, z0 + 4.f));
    c.h11 += (bil4(m, a + 0.25f, b + 0.25f) - c.h11) * lod_w(far_r(x0 + 4.f, z0 + 4.f));
}
// the LOD surface at relative (x, z); false if its tile is not published
__device__ bool tr_surface_lod(float x, float z, float& T) {
    const int rcx = (int)floorf(x * (1.f / 32.f)), rcz = (int)floorf(z * (1.f / 32.f));
    const GmeColTile* t = tr_tile(rcx, rcz);
    if (!t) return false;
    const float lx = (x - (float)(rcx * 32)) * 0.25f, lz = (z - (float)(rcz * 32)) * 0.25f;
    const int CX = min(max((int)floorf(lx), 0), 7), CZ = min(max((int)floorf(lz), 0), 7);
    FarCell c;
    lod_cell(t, rcx, rcz, CX, CZ, c);
    T = bil4(c, sat(lx - (float)CX), sat(lz - (float)CZ));
    return true;
}
// Highest L1 sample over the tile (rcx, rcz): bounds the morphed LOD surface
__device__ __forceinline__ float lod_band_top(int rcx, int rcz) {
    const FarLevel& L = g_tc.far[0];
    const int fi = (int)floorf(((float)(rcx * 32) - L.ox) * L.iS + 0.03125f), fj = (int)floorf(((float)(rcz * 32) - L.oz) * L.iS + 0.03125f);
    float m = -3e38f;
#pragma unroll
    for (int j = 0; j < 3; ++j)
#pragma unroll
        for (int i = 0; i < 3; ++i) m = fmaxf(m, far_h(L, fi + i, fj + j));
    return m;
}
// The drawn ground (no drop) under relative (x, z): the fine surface (fine) or
// the LOD surface of a published tile inside the ring, else the far field.
// Without the far field: the fine surface only, as in M2. false: none.
__device__ bool tr_ground(float x, float z, bool fine, float& T) {
    if (!g_tc.far_on) return tr_surface(x, z, T);
    const float r = far_r(x, z);
    if (r < g_tc.ring && (fine ? tr_surface(x, z, T) : tr_surface_lod(x, z, T))) return true;
    if (r >= g_tc.r2) return false;
    T = far_surface(x, z);
    return true;
}

// mip max of the cell of size 2^k samples (k = 3: 8 x 8, k = 6: 64 x 64) holding frame sample (fi, fj)
__device__ __forceinline__ float far_mip_max(const FarLevel& L, int fi, int fj, int k) {
    const int n = GME_FAR_N >> k;
    const int ix = ((L.bx >> k) + (fi >> k)) & (n - 1), iz = ((L.bz >> k) + (fj >> k)) & (n - 1);
    const short2 m = __ldg((k == 3 ? L.mA : L.mB) + iz * n + ix);
    return (float)m.x * (1.f / 16.f) + g_tc.ybase_m;
}
// t at which the ray leaves the cell of size 2^k samples holding frame sample f (one axis)
__device__ __forceinline__ float far_exit(int f, int k, float S, float off, float oo, float idd, bool pos) {
    const int cb = ((f >> k) + (pos ? 1 : 0)) << k;
    return ((float)cb * S + off - oo) * idd;
}
// t interval where the ray's horizontal distance from the camera is below R:
// r(t)^2 = qq + 2 qd t + hl2 t^2. false if never.
__device__ __forceinline__ bool disc_span(float qq, float qd, float hl2, float R, float& ta, float& tb) {
    const float c = qq - R * R;
    if (hl2 < 1e-12f) { ta = -3e38f; tb = 3e38f; return c < 0.f; }
    const float disc = qd * qd - hl2 * c;
    if (disc <= 0.f) return false;
    const float qn = -(qd + copysignf(sqrtf(disc), qd));
    const float r0 = qn / hl2, r1 = c / qn;
    ta = fminf(r0, r1);
    tb = fmaxf(r0, r1);
    return true;
}
// lower bound of the ray's height plus the curvature drop over [ta, tb]
// (the drop is convex in t: its minimum is at the closest approach)
__device__ __forceinline__ float far_ylow(float oy, float dy, float qq, float qd, float hl2, float ta, float tb) {
    const float yl = oy + dy * (dy >= 0.f ? ta : tb);
    const float tc = hl2 > 1e-12f ? fminf(fmaxf(-qd / hl2, ta), tb) : ta;
    return yl + g_tc.c2 * fmaxf(qq + (2.f * qd + hl2 * tc) * tc - g_tc.drop_r2, 0.f);
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

// kelp strand of tile column k (absolute x, z): the stipe's axis from its foot on the
// smooth bed (pa) to its swaying tip (pb), relative to the camera chunk, and the
// column hash of its blades (density.h gme_kelp_blade). false = none
__device__ __forceinline__ bool tr_kelp(const GmeColTile* t, int k, int32_t x, int32_t z, uint32_t cell,
                                        float3& pa, float3& pb, uint32_t& hb) {
    const int32_t h = gme_tile_h(t, k);
    const int32_t len = gme_kelp_len(x, z, h, cell, g_tc.seed, &g_tc.planet);
    if (len <= 0) return false;
    const float y0 = tr_T(t, k, tr_base(t));
    const float y1 = fmaxf((float)(h + len + 1 - 32 * g_tc.cy), y0 + 0.3f);
    const float axx = (float)(x - 32 * g_tc.cx) + 0.5f, azz = (float)(z - 32 * g_tc.cz) + 0.5f;
    // visual-only sway of the tip (<= GME_KELP_SWAY: the blades stay inside the column)
    const uint32_t hs = gme_hash3(x, 7, z, g_tc.seed ^ 0x5EA7u);
    const float ph = (float)(hs & 1023u) * (6.2831853f / 1024.f);
    const float sw = 0.11f * __sinf(g_tc.time * 0.9f + ph), sw2 = 0.08f * __cosf(g_tc.time * 0.7f + 1.7f * ph);
    pa = f3(axx, y0, azz);
    pb = f3(axx + sw, y1, azz + sw2);
    hb = gme_hash3(x, 0, z, g_tc.seed ^ 0xF10Au);
    return true;
}
// blade i of a strand (pa, pb, hb) as drawn
__device__ __forceinline__ void tr_kelp_blade(float3 pa, float3 pb, uint32_t hb, int i, GmeKelpBlade& bl) {
    gme_kelp_blade(i, gme_kelp_nblades(pa.y, pb.y), pa.x, pa.y, pa.z, pb.x - pa.x, pb.y, pb.z - pa.z, hb, &bl);
}
__device__ __forceinline__ float3 tr_blade_normal(const GmeKelpBlade& bl) {   // u x w (w horizontal: w_y = 0)
    return norm3(f3(bl.uy * bl.wz, bl.uz * bl.wx - bl.ux * bl.wz, -bl.uy * bl.wx));
}
// One kelp strand along the ray p0 + d t, t in [0, tmax]: the stipe (thin capsule)
// and the blades whose height range the ray crosses there (at most 24, the
// nearer end first). Nearest t, -1 none; part = the blade, -1 the stipe. Out
// of line: the walk's registers stay untouched by its temporaries.
__device__ __noinline__ float tr_kelp_strand(float3 p0, float3 d, float tmax, float3 pa, float3 pb, uint32_t hb, int& part) {
    float best = tr_hit_capsule(p0, d, pa, f3(pb.x, pb.y - GME_KELP_STIPE_R, pb.z), GME_KELP_STIPE_R, tmax);
    part = -1;
    if (best < 0.f) best = 3.0e38f;
    const float tm = fminf(best, tmax);
    const float ylo = fminf(p0.y, p0.y + d.y * tm), yhi = fmaxf(p0.y, p0.y + d.y * tm);
    const int n = gme_kelp_nblades(pa.y, pb.y);
    const float fn = (float)n / fmaxf(pb.y - pa.y, 1e-3f);
    // blade i spans y in [A_i, A_i + L sin 42deg], A_i = pa.y + (i + 0.5) (pb.y - pa.y) / n
    int i0 = max((int)floorf((ylo - 0.29f - pa.y) * fn - 0.5f), 0), i1 = min((int)ceilf((yhi + 0.02f - pa.y) * fn - 0.5f), n - 1);
    if (i1 - i0 > 23) { if (d.y < 0.f) i0 = i1 - 23; else i1 = i0 + 23; }
    for (int i = i0; i <= i1; ++i) {
        GmeKelpBlade bl;
        tr_kelp_blade(pa, pb, hb, i, bl);
        const float3 N = tr_blade_normal(bl);
        const float dn = dot3(d, N);
        if (fabsf(dn) < 1e-6f) continue;
        const float3 c = f3(bl.cx, bl.cy, bl.cz);
        const float t = dot3(c - p0, N) / dn;
        if (!(t >= 0.f) || t >= best || t > tmax) continue;
        const float3 q = p0 + d * t - c;
        const float ua = (q.x * bl.ux + q.y * bl.uy + q.z * bl.uz) * (2.f / GME_KELP_BLADE_L);
        const float wb = (q.x * bl.wx + q.z * bl.wz) * (2.f / GME_KELP_BLADE_W);
        if (ua * ua + wb * wb <= 1.f) { best = t; part = i; }
    }
    return best < 3.0e37f ? best : -1.f;
}

// ---- M5 crowns: geometry of a slot's crown, the clumped surface, the leaf medium ----
// Version 2 (spec 5.3): a slot whose shape word says so (flora.h GME_SHAPE_*) carries one of five crown
// classes. Its ellipsoid goes through the same ellipsoid code as version 1 (centre and vertical radius
// from the crown's rows); cone, umbrella, column and dome are quadrics of revolution clipped to the
// crown's rows (density.h GmeCrownQ), with the chord, exit, distance estimate and normal below, so
// every ray type treats them as it treats the ellipsoid. The shape word's porosity bin thins the
// leaf-area medium, its leaf angle bin turns G into Goudriaan's G(theta) = phi1 + phi2 |cos theta|.
// CrownG keeps its version 1 size (the kernels' registers): a version 2 crown packs its class, rows and
// leaf angle into the high bits of trad, and the quadric is rebuilt from them where it is needed.
struct CrownG {
    float3   ce;                    // centre (relative to the camera chunk), shifted onto T like the drawn tree;
                                    // a quadric class: on the trunk axis at the bottom face of the crown's lowest row
    float    ia, ib;                // inverse radii: horizontal sqrt(trad^2 + trad), vertical sqrt(tvr^2 + tvr)
    float    sigma;                 // leaf-area extinction G LAD = G 3 LAI / (4 b), 1/m (G = 1/2, x the porosity factor)
    int32_t  trad;                  // bits 0-7 the radius. Version 2: bits 8-10 the crown class, bit 11 set,
                                    // bits 12-19 the crown's rows, bits 22-23 the leaf angle bin (the shape word's)
    uint32_t cs;                    // clump noise seed of the tree
};
#define CROWN_R(c)     ((c).trad & 255)
#define CROWN_QUAD(c)  ((c).trad & 0x700)          // != 0: cone, umbrella, column or dome (a quadric about ce)
#define CROWN_V2(c)    ((c).trad & 0x800)
#define CROWN_CLS(c)   ((uint32_t)((c).trad >> 8) & 7u)
__device__ __forceinline__ float crown_sigma(float ib) { return SPEC_LEAF_G * 0.75f * SPEC_CROWN_LAI * ib; }
// Leaf angle (version 2): the factor on sigma along a direction of vertical cosine mu. Goudriaan's
// G(theta) = phi1 + phi2 mu, phi1 = 0.5 - 0.633 chi - 0.33 chi^2, phi2 = 0.877 (1 - 2 phi1), chi =
// -0.4 + (bin + 0.5) / 4, taken relative to the mid bin 2, which is the version 1 crown (random
// leaves, G = 1/2: factor exactly 1): 1 + 2 (phi1 - phi1_mid) (1 - 1.754 mu). Erect leaves (bins 0, 1)
// catch more of a low sun, flat ones (bin 3) more of a high one.
__device__ __forceinline__ float tr_leaf_g2(int32_t packed, float mu) {
    const int bin = (packed >> 22) & 3;
    if (bin == 2) return 1.f;
    const float chi = -0.4f + ((float)bin + 0.5f) * 0.25f;
    const float d1 = (0.5f - 0.633f * chi - 0.33f * chi * chi) - 0.340869f;   // phi1 - phi1 of bin 2 (chi 0.225)
    return 1.f + 2.f * d1 * (1.f - 1.754f * mu);
}
// the crown's extinction along d, and its mean over directions (version 1: sigma, random leaves)
__device__ __forceinline__ float tr_crown_sigma(const CrownG& c, float3 d) {
    return CROWN_V2(c) ? c.sigma * tr_leaf_g2(c.trad, fabsf(d.y)) : c.sigma;
}
__device__ __forceinline__ float tr_crown_sigma_mean(const CrownG& c) { return CROWN_V2(c) ? c.sigma * tr_leaf_g2(c.trad, 0.5f) : c.sigma; }
// clump amplitude factor: thin shapes keep their outline
__device__ __forceinline__ float tr_crown_ampk(const CrownG& c) {
    const uint32_t cls = CROWN_CLS(c);
    return (cls == BIO_CROWN_ELLIPSOID || cls == BIO_CROWN_DOME) ? 1.f : (cls == BIO_CROWN_UMBRELLA ? 0.5f : 0.6f);
}
__device__ __forceinline__ void tr_crown_quadric(int32_t packed, GmeCrownQ& q) {
    gme_crown_quadric((uint32_t)(packed >> 8) & 7u, packed & 255, (packed >> 12) & 255, &q);
}
// Version 2 crown from the slot's shape word sw, radius r and rows ch: (ax, az) the trunk axis, yfoot the
// plant's foot (tbase, relative, shifted onto T). Out of line and by value, like every version 2 branch
// below: the version 1 paths of the kernels keep their code and registers.
__device__ __noinline__ CrownG tr_crown_v2(int32_t sw, int32_t r, int32_t ch, float ax, float yfoot, float az) {
    CrownG c;
    c.cs = 0u;
    const uint32_t cls = GME_SHAPE_CLASS(sw);
    const float H = (float)ch, yb = yfoot + (float)(GME_SHAPE_HT(sw) - ch + 1);   // bottom face of the crown's lowest row
    float lad;
    c.trad = r | (sw & 0x00C00F00) | (ch << 12);
    c.ia = rsqrtf((float)(r * r + r));
    if (cls == BIO_CROWN_ELLIPSOID) {
        c.ce = f3(ax, yb + 0.5f * H, az);
        c.ib = rsqrtf(0.25f * (float)(ch * ch - 1));
        lad = 0.75f * SPEC_CROWN_LAI * c.ib;
    } else {
        c.ce = f3(ax, yb, az);
        c.ib = 1.f / H;
        lad = SPEC_CROWN_LAI / (H * gme_crown_fill(cls));       // LAI over the footprint / the crown's volume
    }
    const uint32_t pb = GME_SHAPE_POROSITY(sw);                  // closed .. open crown
    c.sigma = SPEC_LEAF_G * lad * (pb == 0u ? 1.f : (pb == 1u ? 0.7f : (pb == 2u ? 0.45f : 0.25f)));
    return c;
}
// the crown of block b of the tile (rcx, rcz); false = no shrub / tree there
__device__ __forceinline__ bool tr_crown_of(const GmeColTile* t, int b, int rcx, int rcz, CrownG& c) {
    const GmeSlot* s = &t->slot[b];
    const int32_t trad = __ldg(&s->trad);
    if (trad <= 0) return false;
    const int32_t tx = __ldg(&s->tx), tz = __ldg(&s->tz), tcy = __ldg(&s->tcy), tvr = __ldg(&s->tvr);
    const int32_t lx = tx - (g_tc.cx + rcx) * 32, lz = tz - (g_tc.cz + rcz) * 32;
    const float dl = (float)((int32_t)((__ldg(&t->col[(lz + GME_TILE_APRON) * GME_TILE_N + lx + GME_TILE_APRON]) >> 8) ^ 0x80u) - 0x80) *
                     (1.f / GME_DEV_UNIT);
    const int32_t sw = __ldg(&s->theight);
    if (GME_SHAPE_IS_V2(sw)) {
        c = tr_crown_v2(sw, trad, tvr, (float)(rcx * 32 + lx) + 0.5f, (float)(__ldg(&s->tbase) - 32 * g_tc.cy) + dl, (float)(rcz * 32 + lz) + 0.5f);
        c.cs = gme_clump_seed(tx, tz, g_tc.seed);
        return true;
    }
    c.cs = gme_clump_seed(tx, tz, g_tc.seed);
    c.ce = f3((float)(rcx * 32 + lx) + 0.5f, (float)(tcy - 32 * g_tc.cy) + 0.5f + dl, (float)(rcz * 32 + lz) + 0.5f);
    c.ia = rsqrtf((float)(trad * trad + trad));
    c.ib = rsqrtf((float)(tvr * tvr + tvr));
    c.sigma = crown_sigma(c.ib);
    c.trad = trad;
    return true;
}
// the crown of the slot holding relative point p (a crown lies inside its slot)
__device__ __forceinline__ bool tr_crown_at(float3 p, CrownG& c) {
    const int rcx = (int)floorf(p.x * (1.f / 32.f)), rcz = (int)floorf(p.z * (1.f / 32.f));
    const GmeColTile* t = tr_tile(rcx, rcz);
    if (!t) return false;
    const int bx = min(max((int)floorf((p.x - (float)(rcx * 32)) * 0.125f), 0), 3), bz = min(max((int)floorf((p.z - (float)(rcz * 32)) * 0.125f), 0), 3);
    return tr_crown_of(t, bz * 4 + bx, rcx, rcz, c);
}
// the part of the ray p + d t inside the crown's solid: [t0, t1]; false = it misses
__device__ __forceinline__ bool tr_crown_chord(const CrownG& c, float3 p, float3 d, float& t0, float& t1) {
    if (CROWN_QUAD(c)) {
        GmeCrownQ q;
        tr_crown_quadric(c.trad, q);
        return gme_crown_q_chord(&q, p.x - c.ce.x, p.y - c.ce.y, p.z - c.ce.z, d.x, d.y, d.z, &t0, &t1) != 0;
    }
    const float px = (p.x - c.ce.x) * c.ia, py = (p.y - c.ce.y) * c.ib, pz = (p.z - c.ce.z) * c.ia;
    const float qx = d.x * c.ia, qy = d.y * c.ib, qz = d.z * c.ia;
    const float qq = qx * qx + qy * qy + qz * qz, pq = px * qx + py * qy + pz * qz, pp = px * px + py * py + pz * pz - 1.f;
    const float disc = pq * pq - qq * pp;
    if (disc <= 0.f) return false;
    const float sq = sqrtf(disc);
    t0 = (-pq - sq) / qq;
    t1 = (-pq + sq) / qq;
    return true;
}
// version 2 quadrics, out of line (packed = CrownG.trad): the exit distance from crown-local p, the
// outward normal at crown-local p
__device__ __noinline__ float tr_crown_exit2(int32_t packed, float3 p, float3 d) {
    GmeCrownQ q;
    tr_crown_quadric(packed, q);
    float t0, t1;
    return gme_crown_q_chord(&q, p.x, p.y, p.z, d.x, d.y, d.z, &t0, &t1) ? fmaxf(t1, 0.f) : 0.f;
}
__device__ __noinline__ float3 tr_crown_normal2(int32_t packed, float3 p) {
    GmeCrownQ q;
    tr_crown_quadric(packed, q);
    int face;
    gme_crown_q_de(&q, p.x, p.y, p.z, &face);
    if (face) return f3(0.f, (float)face, 0.f);           // a flat cap
    const float w = p.y - q.uc;
    const float3 g = f3(2.f * q.ia2 * p.x, 2.f * q.k2 * w + q.k1, 2.f * q.ia2 * p.z);
    const float l2 = dot3(g, g);
    return l2 > 1e-12f ? g * rsqrtf(l2) : f3(0.f, 1.f, 0.f);
}
// distance from p along unit d to where the ray leaves the crown's solid (0 if it never is inside ahead)
__device__ __forceinline__ float tr_crown_exit(const CrownG& c, float3 p, float3 d) {
    if (CROWN_QUAD(c)) return tr_crown_exit2(c.trad, p - c.ce, d);
    const float px = (p.x - c.ce.x) * c.ia, py = (p.y - c.ce.y) * c.ib, pz = (p.z - c.ce.z) * c.ia;
    const float qx = d.x * c.ia, qy = d.y * c.ib, qz = d.z * c.ia;
    const float qq = qx * qx + qy * qy + qz * qz, pq = px * qx + py * qy + pz * qz, pp = px * px + py * py + pz * pz - 1.f;
    const float disc = pq * pq - qq * pp;
    if (disc <= 0.f) return 0.f;
    return fmaxf((-pq + sqrtf(disc)) / qq, 0.f);
}
// clumps fade out where they are smaller than a few pixels (t: distance along the primary ray)
__device__ __forceinline__ float tr_clump_fade(float t, float cell) { return sat(2.f - t * g_tc.pix / (0.25f * cell)); }
// inward displacement (m) of the clumped surface at crown-local point q (m from the centre), and its gradient
// (1/m): billows of two octaves of value noise, delta = amp (0.72 m1^2 + 0.28 m2^2), m = 1 - |2n - 1|:
// rounded clumps at the crown's surface with smaller ones on them, narrow deep creases between (n = 0.5).
// Band-limited by the pixel footprint fp (m): |s| becomes sqrt(s^2 + w^2) with w = 1.5 fp / the octave's
// cell, so a crease is never narrower than about a pixel and an octave fades out (m -> 0) once its clumps
// are smaller than a few pixels (no shimmering under the jittered TAA).
#define CLUMP_O2 2.37f
__device__ __forceinline__ float tr_clump_delta(const CrownG& c, float3 q, float amp, float fp, float3* grad) {
    const float ic = 1.f / gme_clump_cell(CROWN_R(c)), ic2 = ic * CLUMP_O2;
    const float w1 = 1.5f * fp * ic, w2 = 1.5f * fp * ic2;
    float g[3], g2[3];
    const float n = gme_clump_noise(q.x * ic, q.y * ic, q.z * ic, c.cs, grad ? g : NULL);
    const float s = 2.f * n - 1.f, r = sqrtf(s * s + w1 * w1), m = fmaxf(1.f - r, 0.f);
    float m2 = 0.f, s2 = 0.f, r2 = 1.f;
    if (w2 < 1.f) {                                     // the second octave while it is resolved
        const float n2 = gme_clump_noise(q.x * ic2 + 17.3f, q.y * ic2, q.z * ic2, c.cs ^ 0x9E3779B9u, grad ? g2 : NULL);
        s2 = 2.f * n2 - 1.f;
        r2 = sqrtf(s2 * s2 + w2 * w2);
        m2 = fmaxf(1.f - r2, 0.f);
    }
    if (grad) {                                         // d/dq of amp (0.72 m^2 + 0.28 m2^2), dm/ds = -s / r
        const float k = -4.f * 0.72f * amp * m * (s / fmaxf(r, 1e-6f)) * ic;
        const float k2 = -4.f * 0.28f * amp * m2 * (s2 / fmaxf(r2, 1e-6f)) * ic2;
        *grad = f3(g[0] * k + (m2 > 0.f ? g2[0] * k2 : 0.f), g[1] * k + (m2 > 0.f ? g2[1] * k2 : 0.f), g[2] * k + (m2 > 0.f ? g2[2] * k2 : 0.f));
    }
    return amp * (0.72f * m * m + 0.28f * m2 * m2);
}
// signed distance estimate to the crown ellipsoid (I. Quilez's bound) at crown-local q
__device__ __forceinline__ float tr_ellipsoid_de(const CrownG& c, float3 q) {
    const float kx = q.x * c.ia, ky = q.y * c.ib, kz = q.z * c.ia;
    const float k0 = sqrtf(kx * kx + ky * ky + kz * kz);
    const float k1 = sqrtf(kx * kx * c.ia * c.ia + ky * ky * c.ib * c.ib + kz * kz * c.ia * c.ia);
    return k0 * (k0 - 1.f) / fmaxf(k1, 1e-6f);
}
// the crown's outward normal at crown-local q
__device__ __forceinline__ float3 tr_crown_normal(const CrownG& c, float3 q) {
    if (CROWN_QUAD(c)) return tr_crown_normal2(c.trad, q);
    return norm3(f3(q.x * c.ia * c.ia, q.y * c.ib * c.ib, q.z * c.ia * c.ia));
}
// Clumped crown along p0 + d t' entering the solid at tin: sphere tracing of
// d_solid + delta = 0 inside the shell, at most 12 steps; -1 if the ray leaves
// the solid through the gaps. tabs: distance of p0 along the primary ray (fade).
__device__ __noinline__ float tr_clump(float3 p0, float3 d, const CrownG& c, float tin, float tabs) {
    const float cell = gme_clump_cell(CROWN_R(c));
    const float amp = gme_clump_amp(CROWN_R(c)) * tr_crown_ampk(c) * tr_clump_fade(tabs + tin, cell);
    if (amp <= 0.f) return tin;
    GmeCrownQ cq;                                        // version 2 quadric classes: the solid's distance estimate
    if (CROWN_QUAD(c)) tr_crown_quadric(c.trad, cq);
    const float tout = tin + tr_crown_exit(c, p0 + d * tin, d);
    const float lip = 1.f + 3.5f * amp / cell;           // |grad delta| <= ~14 amp / cell at worst; typical far less
    const float fp = (tabs + tin) * g_tc.pix;            // pixel footprint at the crown (band limit of the billows)
    float t = tin;
    for (int i = 0; i < 12; ++i) {
        const float3 q = p0 + d * t - c.ce;
        const float F = (CROWN_QUAD(c) ? gme_crown_q_de(&cq, q.x, q.y, q.z, NULL) : tr_ellipsoid_de(c, q)) + tr_clump_delta(c, q, amp, fp, NULL);
        if (F < 2e-3f) return t;
        t += F / lip;
        if (t > tout) return -1.f;
    }
    return t;                                            // close to the surface: accept
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
// HK_NONE if neither is hit. M5 kMode: FM_PLAIN the crown ellipsoid; FM_CLUMP
// the clumped crown (primary rays; a ray may pass through its gaps); FM_LAD the
// crown as a leaf-area medium: h.tau += sigma x chord, blocking only once the
// optical depth passes TR_LAD_OPAQUE (High's shadow rays). Trunks are opaque.
// Version 2 crown (any class) of block b over [tb, tend], as tr_flora below: the hit's kind, the new
// tend, the LAD ray's optical depth. Out of line and by value: the walk's registers (and its tend, its
// hit record) stay those of the version 1 path.
struct FloraHit { int kind; float tend, tau; };
template <int kMode>
__device__ __noinline__ FloraHit tr_flora2(const GmeColTile* tile, int b, int rcx, int rcz,
                                           float ox, float oy, float oz, float3 d, float tb, float tend, float tau) {
    const GmeSlot* s = &tile->slot[b];
    const int32_t trad = __ldg(&s->trad), tcy = __ldg(&s->tcy), tbase = __ldg(&s->tbase);
    const int32_t tx = __ldg(&s->tx), tz = __ldg(&s->tz);
    const int32_t lx = tx - (g_tc.cx + rcx) * 32, lz = tz - (g_tc.cz + rcz) * 32;
    const float dl = (float)((int32_t)((__ldg(&tile->col[(lz + GME_TILE_APRON) * GME_TILE_N + lx + GME_TILE_APRON]) >> 8) ^ 0x80u) - 0x80) *
                     (1.f / GME_DEV_UNIT);
    const float3 pb0 = f3(ox + d.x * tb, oy + d.y * tb, oz + d.z * tb);
    // tile-local, like the ray
    CrownG c = tr_crown_v2(__ldg(&s->theight), trad, __ldg(&s->tvr), (float)lx + 0.5f, (float)(tbase - 32 * g_tc.cy) + dl, (float)lz + 0.5f);
    int fk = HK_NONE;
    float t0, t1;
    if (tr_crown_chord(c, pb0, d, t0, t1) && t1 > 0.f && t0 <= tend - tb) {
        const float tin = fmaxf(t0, 0.f);
        if (kMode == FM_LAD) {                           // the chord through the crown, clipped to the segment
            const float tout = fminf(t1, tend - tb);
            if (tout > tin) {
                tau += tr_crown_sigma(c, d) * (tout - tin);
                if (tau > TR_LAD_OPAQUE) { tend = tb + tin; fk = HK_CROWN; }
            }
        } else {
            float tf = tin;
            if (kMode == FM_CLUMP) {
                c.cs = gme_clump_seed(tx, tz, g_tc.seed);
                tf = tr_clump(pb0, d, c, tin, tb);
            }
            if (tf >= 0.f) { tend = tb + tf; fk = HK_CROWN; }
        }
    }
    const float y0 = (float)(tbase + 1 - 32 * g_tc.cy) + dl, y1 = (float)(tcy + 1 - 32 * g_tc.cy) + dl;
    const float tt = tr_hit_cylinder(pb0, d, (float)lx + 0.5f, (float)lz + 0.5f, gme_trunk_radius(trad), y0, y1, tend - tb);
    if (tt >= 0.f) { tend = tb + tt; fk = HK_TRUNK; }
    FloraHit r;
    r.kind = fk; r.tend = tend; r.tau = tau;
    return r;
}

template <int kMode>
__device__ __forceinline__ int tr_flora(const GmeColTile* tile, int b, int rcx, int rcz,
                                        float ox, float oy, float oz, float3 d, float tb, float& tend, THit& h) {
    const GmeSlot* s = &tile->slot[b];
    if (GME_SHAPE_IS_V2(__ldg(&s->theight))) {          // version 2 classes
        const FloraHit r = tr_flora2<kMode>(tile, b, rcx, rcz, ox, oy, oz, d, tb, tend, kMode == FM_LAD ? h.tau : 0.f);
        if (kMode == FM_LAD) h.tau = r.tau;
        tend = r.tend;
        return r.kind;
    }
    const int32_t trad = __ldg(&s->trad), tcy = __ldg(&s->tcy), tvr = __ldg(&s->tvr);
    const int32_t lx = __ldg(&s->tx) - (g_tc.cx + rcx) * 32, lz = __ldg(&s->tz) - (g_tc.cz + rcz) * 32;   // 0..31
    const float dl = (float)((int32_t)((__ldg(&tile->col[(lz + GME_TILE_APRON) * GME_TILE_N + lx + GME_TILE_APRON]) >> 8) ^ 0x80u) - 0x80) *
                     (1.f / GME_DEV_UNIT);
    const float3 pb0 = f3(ox + d.x * tb, oy + d.y * tb, oz + d.z * tb);   // segment start: well-conditioned quadratics
    const float3 ce = f3((float)lx + 0.5f, (float)(tcy - 32 * g_tc.cy) + 0.5f + dl, (float)lz + 0.5f);
    const float ia = rsqrtf((float)(trad * trad + trad)), ib = rsqrtf((float)(tvr * tvr + tvr));
    int fk = HK_NONE;
    if (kMode == FM_LAD) {                             // the chord through the crown, clipped to the segment
        const float px = (pb0.x - ce.x) * ia, py = (pb0.y - ce.y) * ib, pz = (pb0.z - ce.z) * ia;
        const float qx = d.x * ia, qy = d.y * ib, qz = d.z * ia;
        const float qq = qx * qx + qy * qy + qz * qz, pq = px * qx + py * qy + pz * qz, pp = px * px + py * py + pz * pz - 1.f;
        const float disc = pq * pq - qq * pp;
        if (disc > 0.f) {
            const float sq = sqrtf(disc);
            const float tin = fmaxf((-pq - sq) / qq, 0.f), tout = fminf((-pq + sq) / qq, tend - tb);
            if (tout > tin) {
                h.tau += crown_sigma(ib) * (tout - tin);
                if (h.tau > TR_LAD_OPAQUE) { tend = tb + tin; fk = HK_CROWN; }
            }
        }
    } else {
        float tf = tr_hit_ellipsoid(pb0, d, ce, ia, ib, tend - tb);
        if (kMode == FM_CLUMP && tf >= 0.f) {
            CrownG c;
            c.ce = ce; c.ia = ia; c.ib = ib; c.sigma = 0.f; c.trad = trad;
            c.cs = gme_clump_seed(__ldg(&s->tx), __ldg(&s->tz), g_tc.seed);
            tf = tr_clump(pb0, d, c, tf, tb);
        }
        if (tf >= 0.f) { tend = tb + tf; fk = HK_CROWN; }
    }
    const float y0 = (float)(__ldg(&s->tbase) + 1 - 32 * g_tc.cy) + dl, y1 = (float)(tcy + 1 - 32 * g_tc.cy) + dl;
    const float tt = tr_hit_cylinder(pb0, d, ce.x, ce.z, gme_trunk_radius(trad), y0, y1, tend - tb);
    if (tt >= 0.f) { tend = tb + tt; fk = HK_TRUNK; }
    return fk;
}

// Kelp of block b over [tb, tend] (tile-local origin ox, oy, oz): a walk over
// the block's columns along the ray; each column's strand (stipe + blades, M5)
// lies inside its own 1 m footprint, so the first hit in walk order is the
// nearest. Lowers tend and returns true (h.i, h.j = the strand's column, h.u =
// its blade, -1 the stipe) on a hit. A pass of its own, before the dual cells,
// so its temporaries never share registers with the terrain walk.
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
            uint32_t hb;
            if (tr_kelp(tile, (lz + GME_TILE_APRON) * GME_TILE_N + lx + GME_TILE_APRON, (g_tc.cx + rcx) * 32 + lx,
                        (g_tc.cz + rcz) * 32 + lz, cell, pa, pb, hb)) {
                // tr_kelp works relative to the camera chunk: shift the segment start there
                const float3 p0 = f3(ox + (float)(rcx * 32) + d.x * s0, oy + d.y * s0, oz + (float)(rcz * 32) + d.z * s0);
                int part;
                const float tk = tr_kelp_strand(p0, d, tend - s0, pa, pb, hb, part);
                if (tk >= 0.f) {
                    tend = s0 + tk;
                    h.i = lx + GME_TILE_APRON; h.j = lz + GME_TILE_APRON;
                    h.u = (float)part;
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
template <int kMode>
__device__ __forceinline__ bool tr_block(const GmeColTile* tile, int b, int rcx, int rcz,
                                         float ox, float oy, float oz, float3 d, float ix, float iz,
                                         float tb, float tbe, THit& h) {
    const bool px = d.x > 0.f, pz = d.z > 0.f;
    float tend = tbe;
    int fk = __ldg(&tile->slot[b].trad) > 0 ? tr_flora<kMode>(tile, b, rcx, rcz, ox, oy, oz, d, tb, tend, h) : HK_NONE;
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

// One 8 m block in LOD mode over [tb, tbe]: the slot's shrub / tree while
// tb < tf, then the 2 x 2 bilinear 4 m cells of the lod4 lattice (morphed
// toward L1 near the ring). true = hit (h filled).
template <int kMode>
__device__ __forceinline__ bool tr_block_lod(const GmeColTile* tile, int b, int rcx, int rcz, float ox, float oy, float oz,
                                             float3 d, float ix, float iz, float tb, float tbe, float tf, THit& h) {
    const bool px = d.x > 0.f, pz = d.z > 0.f;
    float tend = tbe;
    const int fk = (tb < tf && __ldg(&tile->slot[b].trad) > 0) ? tr_flora<kMode>(tile, b, rcx, rcz, ox, oy, oz, d, tb, tend, h) : HK_NONE;
    const int bx2 = (b & 3) * 2, bz2 = (b >> 2) * 2;
    int CX = min(max((int)floorf((ox + d.x * tb) * 0.25f), bx2), bx2 + 1);
    int CZ = min(max((int)floorf((oz + d.z * tb) * 0.25f), bz2), bz2 + 1);
    float s0 = tb;
    for (;;) {
        const float ex = tr_exit(CX, 4.f, 0.f, ox, ix, px), ez = tr_exit(CZ, 4.f, 0.f, oz, iz, pz);
        const float s1 = fminf(fminf(ex, ez), tend);
        if (s1 > s0) {
            FarCell c;
            lod_cell(tile, rcx, rcz, CX, CZ, c);
            const float ya = oy + d.y * s0;
            if (fminf(ya, oy + d.y * s1) <= fmaxf(fmaxf(c.h00, c.h10), fmaxf(c.h01, c.h11))) {
                const float bb = c.h10 - c.h00, cc = c.h01 - c.h00, kk = c.h00 - c.h10 - c.h01 + c.h11;
                const float u0 = (ox + d.x * s0) * 0.25f - (float)CX, v0 = (oz + d.z * s0) * 0.25f - (float)CZ;
                const float du = d.x * 0.25f, dv = d.z * 0.25f;
                const float A = -kk * du * dv;
                const float B = d.y - bb * du - cc * dv - kk * (u0 * dv + v0 * du);
                const float C = ya - (c.h00 + bb * u0 + cc * v0 + kk * u0 * v0);
                const float r = tr_patch_root(A, B, C, s1 - s0);
                if (r >= 0.f) {
                    h.t = s0 + r; h.kind = HK_LOD; h.rcx = rcx; h.rcz = rcz; h.i = CX; h.j = CZ;
                    h.u = sat(u0 + du * r); h.v = sat(v0 + dv * r);
                    return true;
                }
            }
        }
        if (s1 >= tend) break;
        if (ex < ez) CX += px ? 1 : -1; else CZ += pz ? 1 : -1;
        s0 = s1;
        if ((unsigned)(CX - bx2) > 1u || (unsigned)(CZ - bz2) > 1u) break;
    }
    if (fk != HK_NONE) {
        h.t = tend; h.kind = fk; h.rcx = rcx; h.rcz = rcz; h.i = b;
        return true;
    }
    return false;
}

enum { TS_MISS = 0, TS_HIT = 1, TS_VOXEL = 2 };

// Column-mode walk over [t, tmax]: tiles -> blocks -> dual cells (before tl)
// or 4 m LOD cells (from tl on; flora while t < tf). Returns TS_HIT (h
// filled), TS_MISS, or TS_VOXEL with [t, te] = a tile that is not published,
// which the caller crosses in the far field (M3) or in voxel mode (no call
// inside this loop, which keeps the walk's registers low).
template <int kMode>
__device__ __forceinline__ int tr_columns(float3 o, float3 d, float& t, float& te, float tmax, float tl, float tf, THit& h) {
    const float ix = 1.f / d.x, iz = 1.f / d.z;
    const bool px = d.x > 0.f, pz = d.z > 0.f;
    const float tn = t + 1e-3f + t * 1e-6f;            // start tile: a point just past t, never the one behind
    int rcx = (int)floorf((o.x + d.x * tn) * (1.f / 32.f)), rcz = (int)floorf((o.z + d.z * tn) * (1.f / 32.f));
    float ox = o.x - (float)(rcx * 32), oz = o.z - (float)(rcz * 32);   // origin, tile-local
    // ceiling of the near field: published tiles (and the far field's L1 inside the ring)
    const float ytop = g_tc.far_on ? fminf(g_tc.ytop, (float)__ldg(g_tc.far_max + 2) + g_tc.ybase_m) : g_tc.ytop;
    while (t < tmax) {                                 // te > t every step but at a corner: always ends
        te = fminf(fminf(tr_exit(0, 32.f, 0.f, ox, ix, px), tr_exit(0, 32.f, 0.f, oz, iz, pz)), tmax);
        const float y0 = o.y + d.y * t;
        if (d.y >= 0.f && y0 > ytop) return TS_MISS;                 // above everything, climbing
        if (te > t) {
            const GmeColTile* tile = tr_tile(rcx, rcz);
            if (!tile) return TS_VOXEL;                               // not streamed in yet: far field (M3) / voxels
            float top = (float)(__ldg(&tile->top) - 32 * g_tc.cy), band = -3e38f;
            const bool lod = te > tl;
            if (lod) {
                top = fmaxf(top, (float)(__ldg(&tile->lod_top) - 32 * g_tc.cy));
                if (g_tc.far_on && far_r((float)(rcx * 32 + 16), (float)(rcz * 32 + 16)) + 23.f > g_tc.ring_m0) {
                    band = lod_band_top(rcx, rcz);
                    top = fmaxf(top, band);
                }
            }
            if (fminf(y0, o.y + d.y * te) <= top) {
                int b = min(max((int)floorf((oz + d.z * t) * 0.125f), 0), 3) * 4 +
                        min(max((int)floorf((ox + d.x * t) * 0.125f), 0), 3);
                float tb = t;
                for (;;) {
                    const float tbe = fminf(fminf(tr_exit(b & 3, 8.f, 0.f, ox, ix, px), tr_exit(b >> 2, 8.f, 0.f, oz, iz, pz)), te);
                    if (tbe > tb) {
                        const float ylo = fminf(o.y + d.y * tb, o.y + d.y * tbe);
                        if (tb >= tl) {                                      // LOD block
                            const float btop = fmaxf(fmaxf((float)(__ldg(&tile->blk_top[b]) - 32 * g_tc.cy),
                                                           (float)(__ldg(&tile->lod_btop[b]) - 32 * g_tc.cy)), band);
                            if (ylo <= btop && tr_block_lod<kMode>(tile, b, rcx, rcz, ox, o.y, oz, d, ix, iz, tb, tbe, tf, h)) return TS_HIT;
                        } else if (ylo <= (float)(__ldg(&tile->blk_top[b]) - 32 * g_tc.cy) &&
                                   tr_block<kMode>(tile, b, rcx, rcz, ox, o.y, oz, d, ix, iz, tb, tbe, h)) {
                            return TS_HIT;
                        }
                    }
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

// ---- far-field march (M3) ------------------------------------------------------------
// The ray over [t0, t1] against the far field: L1 while the ray is within r1 of
// the camera, else L2, ending at r2. The ray's height plus the curvature drop
// is compared with the clipmap; 64 x 64 and 8 x 8 mip cells are skipped while
// that sum stays above their maximum, and each remaining sample cell is solved
// exactly (bilinear patch + the drop's quadratic in t is still a quadratic).
// true = hit (h.kind = HK_FAR, h.t).
__device__ __noinline__ bool tr_far(float3 o, float3 d, float t0, float t1, THit& h) {
    if (fabsf(d.x) < 1e-7f) d.x = copysignf(1e-7f, d.x);
    if (fabsf(d.z) < 1e-7f) d.z = copysignf(1e-7f, d.z);
    const float qx = o.x - g_tc.camx, qz = o.z - g_tc.camz;
    const float hl2 = d.x * d.x + d.z * d.z, qd = qx * d.x + qz * d.z, qq = qx * qx + qz * qz;
    float a1, b1, a2, b2;
    if (!disc_span(qq, qd, hl2, g_tc.r2, a2, b2)) return false;
    t0 = fmaxf(t0, a2);
    t1 = fminf(t1, b2);
    if (!(t1 > t0)) return false;
    if (!disc_span(qq, qd, hl2, g_tc.r1, a1, b1)) a1 = b1 = -3e38f;
    const float ix = 1.f / d.x, iz = 1.f / d.z;
    const bool px = d.x > 0.f, pz = d.z > 0.f;
    const float m1 = (float)__ldg(g_tc.far_max + 1) * (1.f / 16.f) + g_tc.ybase_m;
    const float m0 = fmaxf((float)__ldg(g_tc.far_max) * (1.f / 16.f) + g_tc.ybase_m, m1);   // L1 morphs toward L2
    const float rm2 = g_tc.r1_m0 * g_tc.r1_m0;
    float t = t0;
    for (int it = 0; it < TR_FAR_STEPS && t < t1; ++it) {
        const int l = (t >= a1 && t < b1) ? 0 : 1;
        const float tseg = fminf(t1, l == 0 ? b1 : (t < a1 ? a1 : t1));   // where this level's part of the ray ends
        if (far_ylow(o.y, d.y, qq, qd, hl2, t, tseg) > (l == 0 ? m0 : m1)) { t = tseg; continue; }   // above it all
        const FarLevel& L = g_tc.far[l];
        const float tn = t + 1e-3f + t * 1e-6f;
        const int fi = (int)floorf((o.x + d.x * tn - L.ox) * L.iS), fj = (int)floorf((o.z + d.z * tn - L.oz) * L.iS);
        if ((unsigned)fi >= (unsigned)(GME_FAR_N + 62) || (unsigned)fj >= (unsigned)(GME_FAR_N + 62)) return false;
        // L1 cells that may reach the L1 -> L2 band are bounded by L2's mips as well
        const bool band = l == 0 && fmaxf(qq + (2.f * qd + hl2 * t) * t, qq + (2.f * qd + hl2 * tseg) * tseg) > rm2;
        bool skip = false;
        for (int k = 6; k >= 3; k -= 3) {                              // 64 x 64, then 8 x 8 samples
            const float te = fminf(fminf(far_exit(fi, k, L.S, L.ox, o.x, ix, px), far_exit(fj, k, L.S, L.oz, o.z, iz, pz)), tseg);
            float mx = far_mip_max(L, fi, fj, k);
            if (band) {
                const FarLevel& M = g_tc.far[1];
                const float cx = L.ox + ((float)((fi >> k) << k) + 0.5f * (float)(1 << k)) * L.S;
                const float cz = L.oz + ((float)((fj >> k) << k) + 0.5f * (float)(1 << k)) * L.S;
                mx = fmaxf(mx, far_mip_max(M, (int)floorf((cx - M.ox) * M.iS), (int)floorf((cz - M.oz) * M.iS), k));
            }
            if (far_ylow(o.y, d.y, qq, qd, hl2, t, te) > mx) { t = fmaxf(te, tn); skip = true; break; }
        }
        if (skip) continue;
        // one sample cell: exact ray / bilinear patch with the drop
        const float te = fminf(fminf(far_exit(fi, 0, L.S, L.ox, o.x, ix, px), far_exit(fj, 0, L.S, L.oz, o.z, iz, pz)), tseg);
        FarCell c;
        far_cell(l, fi, fj, c);
        if (far_ylow(o.y, d.y, qq, qd, hl2, t, te) <= fmaxf(fmaxf(c.h00, c.h10), fmaxf(c.h01, c.h11))) {
            const float S = te - t;
            const float u0 = (o.x + d.x * t - L.ox) * L.iS - (float)fi, v0 = (o.z + d.z * t - L.oz) * L.iS - (float)fj;
            const float du = d.x * L.iS, dv = d.z * L.iS;
            const float bb = c.h10 - c.h00, cc = c.h01 - c.h00, kk = c.h00 - c.h10 - c.h01 + c.h11;
            const float ra2 = qq + (2.f * qd + hl2 * t) * t, rb = qd + hl2 * t;
            const float cd = (ra2 + (2.f * rb + hl2 * S) * S > g_tc.drop_r2) ? g_tc.c2 : 0.f;   // beyond the loaded radius
            const float A = -kk * du * dv + cd * hl2;
            const float B = d.y - bb * du - cc * dv - kk * (u0 * dv + v0 * du) + 2.f * cd * rb;
            const float C = (o.y + d.y * t) - (c.h00 + bb * u0 + cc * v0 + kk * u0 * v0) + cd * (ra2 - g_tc.drop_r2);
            const float r = tr_patch_root(A, B, C, S);
            if (r >= 0.f) {
                h.t = t + r;
                h.kind = HK_FAR;
                return true;
            }
        }
        t = fmaxf(te, tn);
    }
    return false;
}

// The smooth-surface walk over [t0, tmax] (tl: the LOD starts at this t, tf:
// flora up to this t). Inside the ring the column tracer walks the published
// tiles; tiles that have not streamed in are crossed in the far field (M3; voxel
// mode without it, where WATER voxels set h.water_t). From the ring on, or for
// a ray that starts outside it, the far field. Returns the nearest hit, so it
// serves sun rays (any-hit) as well. true = hit (h filled).
template <bool kInlineVoxels, int kMode = FM_PLAIN>
__device__ __forceinline__ bool trace_smooth_t(float3 o, float3 d, float t0, float tmax, float tl, float tf, THit& h) {
    if (fabsf(d.x) < 1e-7f) d.x = copysignf(1e-7f, d.x);
    if (fabsf(d.z) < 1e-7f) d.z = copysignf(1e-7f, d.z);
    float tn = tmax;                                   // end of the near part
    if (g_tc.far_on) {
        const float qx = o.x - g_tc.camx, qz = o.z - g_tc.camz;
        float a, b;
        tn = (disc_span(qx * qx + qz * qz, qx * d.x + qz * d.z, d.x * d.x + d.z * d.z, g_tc.ring, a, b) && a <= t0 && t0 < b)
                 ? fminf(b, tmax) : t0;                // outside the ring: far field only
    }
    const float tf0 = t0;
    for (int k = 0; k < 256 && t0 < tn; ++k) {
        float te = tn;
        const int r = tr_columns<kMode>(o, d, t0, te, tn, tl, tf, h);
        if (r == TS_HIT) return true;
        if (r == TS_MISS) break;
        if (g_tc.far_on) {
            if (tr_far(o, d, t0, te, h)) return true;
        } else {
            float tx;
            if ((kInlineVoxels ? trace_voxels(o, d, t0, te, true, false, h, tx)
                               : trace_voxels_call(o, d, t0, te, true, false, h, tx)) == VX_HIT) return true;
        }
        t0 = te;
    }
    if (!g_tc.far_on || !(tmax > tn)) return false;
    return tr_far(o, d, fmaxf(tn, tf0), tmax, h);
}

// sun rays: the walk inline (k_sun keeps its hit record in registers); voxel
// mode, now only used without the far field, out of line. kMode FM_LAD: crowns
// are a leaf-area medium (h.tau must start at 0; the ray's transmittance is
// e^-tau when it reaches the sun)
template <int kMode = FM_PLAIN>
__device__ __forceinline__ bool trace_smooth(float3 o, float3 d, float t0, float tmax, float tl, float tf, THit& h) {
    return trace_smooth_t<false, kMode>(o, d, t0, tmax, tl, tf, h);
}

#endif

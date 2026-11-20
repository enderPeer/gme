// gi.cu - M4 frame pipeline (docs/render-upgrade-spec.md 4.6, 4.7): the
// screen-space passes and the world-space radiance cache of the smooth path.
//
//   SVGF (GI irradiance, half res, 8 bands fp16)
//     k_gi_temporal  reprojects each GI sample point into the previous frame
//                    (bilinear over the previous half-res samples that pass the
//                    surface test: tangent-plane distance <= 0.02 t + 0.05 m,
//                    n.n' >= 0.9, same class); alpha = max(1 / n, 0.05),
//                    luminance moments alpha max(1 / n, 0.2), variance from the
//                    moments (x4, with a 3x3 spatial estimate, while n < 4)
//     k_gi_atrous    5x5 B3 a-trous, steps 1 2 4 (Medium) / 1 2 4 8 (High);
//                    weights: tangent-plane distance, max(0, n.n')^64,
//                    exp(-|dY| / (4 sqrt(var) + 1e-4)); the first pass's output
//                    is the next frame's history
//     k_cache_update the radiance cache (render.h GiCache): one writer per cell
//                    and frame (atomicCAS on the frame stamp), EMA 1 / min(n, 32)
//                    of the denoised irradiance
//     k_gi_upsample  joint-bilateral 2x2 to the internal resolution
//   sun visibility   k_vis_temporal (alpha 0.2) + k_vis_spatial (3x3 a-trous)
//   reflections      k_refl_temporal: the traced reflections along their
//                    surface's reprojection, clamped to this frame's 3x3 range,
//                    alpha 0.2
//   TAA / TAAU       k_taa: Halton-jittered internal samples resolved into an
//                    output-resolution history: the current frame is a Gaussian
//                    of its 3 x 3 jittered samples (TAA sigma 0.4 px, TAAU 0.47
//                    internal px x the upscale), Catmull-Rom history fetch along
//                    the reprojection (kept when any of the previous 2 x 2
//                    samples is the same surface), YCoCg 3x3 variance clip
//                    (gamma 1.25), history weight 0.9 (0.8 under fast motion,
//                    0.95 for a still pixel with >= 16 frames), luminance-weighted
//                    blend; disocclusion and > 2 m camera moves reset it.
//   reference        k_accum: the plain mean of the frames since the last reset
//   disocclusion     k_disocc / k_disocc_gi: the check of renderer_disocc
//
// Reprojection needs no motion-vector buffer: the scene is static apart from
// the camera, so each pass rebuilds a pixel's world point from its guide (the
// straight-ray distance t of the visible surface) and projects it with the
// previous frame's camera (GiCam). Waves and kelp animate in place.
#include "render.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GI_BIG 3.0e38f

static int gck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return 0;
    fprintf(stderr, "gi: %s: %s\n", what, cudaGetErrorString(e));
    return 1;
}

__device__ __forceinline__ float gsat(float x) { return fminf(fmaxf(x, 0.f), 1.f); }

// ---- guides ----------------------------------------------------------------------
struct Gd { float t; float n[3]; int cls; uint32_t fl; };
__device__ __forceinline__ void gd_unpack(uint2 g, Gd& d) {
    d.t = __uint_as_float(g.x);
    d.cls = (int)((g.y >> 24) & 15u);
    d.fl = g.y & 0xF0000000u;
    gg_oct_unpack(g.y, d.n);
}
// continuous sample coordinate of internal pixel (x, y) of camera c (pixel x covers [x, x + 1))
__device__ __forceinline__ void cam_dir(const GiCam& c, float sx, float sy, float d[3]) {
    const float u = (2.f * sx / (float)c.w - 1.f) * c.th * c.aspect, v = (1.f - 2.f * sy / (float)c.h) * c.th;
    float x = c.fw[0] + c.rt[0] * u + c.up[0] * v, y = c.fw[1] + c.rt[1] * u + c.up[1] * v, z = c.fw[2] + c.rt[2] * u + c.up[2] * v;
    const float il = rsqrtf(x * x + y * y + z * z);
    d[0] = x * il; d[1] = y * il; d[2] = z * il;
}
// world point (relative to camera c's chunk) of a guide sample at (sx, sy)
__device__ __forceinline__ void cam_point(const GiCam& c, float sx, float sy, float t, float P[3]) {
    float d[3];
    cam_dir(c, sx, sy, d);
    P[0] = c.o[0] + d[0] * t; P[1] = c.o[1] + d[1] * t; P[2] = c.o[2] + d[2] * t;
}
// Projects q (relative to camera c's position; a direction if dir) into camera
// c: continuous unjittered pixel coordinates (pixel centre at i + 0.5) and the
// distance. false: behind the camera or off screen.
__device__ __forceinline__ bool cam_project(const GiCam& c, float qx, float qy, float qz, float* X, float* Y) {
    const float z = qx * c.fw[0] + qy * c.fw[1] + qz * c.fw[2];
    if (!(z > 1e-5f)) return false;
    const float u = (qx * c.rt[0] + qy * c.rt[1] + qz * c.rt[2]) / z, v = (qx * c.up[0] + qy * c.up[1] + qz * c.up[2]) / z;
    *X = (u / (c.th * c.aspect) + 1.f) * 0.5f * (float)c.w;
    *Y = (1.f - v / c.th) * 0.5f * (float)c.h;
    return *X >= 0.f && *X < (float)c.w && *Y >= 0.f && *Y < (float)c.h;
}
// Reprojects the current frame's guide sample at (sx, sy) with distance t into
// the previous frame: X, Y (prev internal pixels), the point in the previous
// chunk's coordinates (P) and its distance from the previous camera. Sky (t
// huge): the direction only.
__device__ __forceinline__ bool reproject(const GiFrame& f, float sx, float sy, float t, float* X, float* Y, float P[3], float* tp) {
    if (t >= 1e30f) {
        float d[3];
        cam_dir(f.cur, sx, sy, d);
        P[0] = P[1] = P[2] = 0.f;
        *tp = GI_BIG;
        return cam_project(f.prev, d[0], d[1], d[2], X, Y);
    }
    float Q[3];
    cam_point(f.cur, sx, sy, t, Q);
    P[0] = Q[0] + 32.f * (float)f.dcx; P[1] = Q[1] + 32.f * (float)f.dcy; P[2] = Q[2] + 32.f * (float)f.dcz;
    const float qx = P[0] - f.prev.o[0], qy = P[1] - f.prev.o[1], qz = P[2] - f.prev.o[2];
    *tp = sqrtf(qx * qx + qy * qy + qz * qz);
    return cam_project(f.prev, qx, qy, qz, X, Y);
}
// Is the previous frame's guide sample g (at prev sample coordinate (sx, sy)) the
// same surface as point P (prev chunk coordinates) with normal n and class cls?
// tol: plane-distance tolerance per metre of distance and constant; nmin: n.n'
__device__ __forceinline__ bool same_surface(const GiFrame& f, uint2 g, float sx, float sy, const float P[3], const float n[3], int cls,
                                             float tp, float tol_rel, float tol_abs, float nmin) {
    Gd d;
    gd_unpack(g, d);
    if (d.cls != cls) return false;
    if (cls == GC_SKY) return true;
    if (n[0] * d.n[0] + n[1] * d.n[1] + n[2] * d.n[2] < nmin) return false;
    float Pp[3];
    cam_point(f.prev, sx, sy, d.t, Pp);
    const float pd = fabsf(d.n[0] * (P[0] - Pp[0]) + d.n[1] * (P[1] - Pp[1]) + d.n[2] * (P[2] - Pp[2]));
    return pd <= tol_rel * tp + tol_abs;
}

// half-res grid: the full-res sample pixel of half pixel (i, j) of camera c, continuous coordinate
__device__ __forceinline__ float half_sx(const GiCam& c, int i) { return (float)min(2 * i + c.hx, c.w - 1) + 0.5f + c.jx; }
__device__ __forceinline__ float half_sy(const GiCam& c, int j) { return (float)min(2 * j + c.hy, c.h - 1) + 0.5f + c.jy; }

__device__ __forceinline__ float lum8(const GiFrame& f, const float* e) {
    float s = 0.f;
#pragma unroll
    for (int b = 0; b < 8; ++b) s += f.lw[b] * e[b];
    return s;
}
__device__ __forceinline__ bool bad8(const float* e) {
    bool b = false;
#pragma unroll
    for (int k = 0; k < 8; ++k) b |= !(fabsf(e[k]) < 1e30f);
    return b;
}

// ---- SVGF: temporal ---------------------------------------------------------------
__global__ void __launch_bounds__(128) k_gi_temporal(const GiFrame f) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= f.w2 || j >= f.h2) return;
    const size_t idx = (size_t)j * f.w2 + i;
    const uint2 g = f.hg[idx];
    if (!(g.y & GG_GI)) {
        f.e_a[idx] = make_uint4(0u, 0u, 0u, 0u);
        f.v_a[idx] = 0.f;
        f.m_hist[idx] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }
    Gd d;
    gd_unpack(g, d);
    float e[8], h[8];
    gi_unpack8(f.e_noisy[idx], e);
    if (bad8(e)) {
        if (f.nan) atomicAdd(f.nan, 1u);
#pragma unroll
        for (int b = 0; b < 8; ++b) e[b] = 0.f;
    }
    const float l = lum8(f, e);
    float m1 = l, m2 = l * l, n = 1.f, wsum = 0.f;
#pragma unroll
    for (int b = 0; b < 8; ++b) h[b] = 0.f;
    if (f.hist_ok) {
        float X, Y, P[3], tp;
        if (reproject(f, half_sx(f.cur, i), half_sy(f.cur, j), d.t, &X, &Y, P, &tp)) {
            const float c0 = (X - (float)f.prev.hx - 0.5f - f.prev.jx) * 0.5f, c1 = (Y - (float)f.prev.hy - 0.5f - f.prev.jy) * 0.5f;
            const int i0 = (int)floorf(c0), j0 = (int)floorf(c1);
            const float fx = c0 - (float)i0, fy = c1 - (float)j0;
            float hm1 = 0.f, hm2 = 0.f, hn = 0.f;
            for (int k = 0; k < 4; ++k) {
                const int a = i0 + (k & 1), b = j0 + (k >> 1);
                if (a < 0 || b < 0 || a >= f.w2 || b >= f.h2) continue;
                const size_t q = (size_t)b * f.w2 + a;
                const uint2 gp = f.hg_prev[q];
                if (!(gp.y & GG_GI)) continue;
                if (!same_surface(f, gp, half_sx(f.prev, a), half_sy(f.prev, b), P, d.n, d.cls, tp, 0.02f, 0.05f, 0.9f)) continue;
                const float w = ((k & 1) ? fx : 1.f - fx) * ((k >> 1) ? fy : 1.f - fy);
                if (w <= 0.f) continue;
                float ep[8];
                gi_unpack8(f.e_hist_prev[q], ep);
                const float4 mp = f.m_hist_prev[q];
#pragma unroll
                for (int bb = 0; bb < 8; ++bb) h[bb] += w * ep[bb];
                hm1 += w * mp.x; hm2 += w * mp.y; hn += w * mp.z;
                wsum += w;
            }
            if (wsum > 1e-3f) {
                const float iw = 1.f / wsum;
#pragma unroll
                for (int bb = 0; bb < 8; ++bb) h[bb] *= iw;
                hm1 *= iw; hm2 *= iw; hn *= iw;
                n = fminf(floorf(hn + 0.5f), 254.f) + 1.f;
                const float a = fmaxf(1.f / n, 0.05f), am = fmaxf(1.f / n, 0.2f);
#pragma unroll
                for (int bb = 0; bb < 8; ++bb) e[bb] = h[bb] + (e[bb] - h[bb]) * a;
                m1 = hm1 + (l - hm1) * am;
                m2 = hm2 + (l * l - hm2) * am;
            }
        }
    }
    float var = fmaxf(m2 - m1 * m1, 0.f);
    if (n < 4.f) {                                          // short history: a 3x3 spatial estimate, x4
        float s1 = 0.f, s2 = 0.f, sw = 0.f;
        for (int b = -1; b <= 1; ++b)
            for (int a = -1; a <= 1; ++a) {
                const int x = i + a, y = j + b;
                if (x < 0 || y < 0 || x >= f.w2 || y >= f.h2) continue;
                const size_t q = (size_t)y * f.w2 + x;
                const uint2 gq = f.hg[q];
                if (!(gq.y & GG_GI) || (int)((gq.y >> 24) & 15u) != d.cls) continue;
                float eq[8];
                gi_unpack8(f.e_noisy[q], eq);
                if (bad8(eq)) continue;
                const float lq = lum8(f, eq);
                s1 += lq; s2 += lq * lq; sw += 1.f;
            }
        if (sw > 0.f) { s1 /= sw; s2 /= sw; var = fmaxf(var, s2 - s1 * s1); }
        var *= 4.f;
    }
    f.e_a[idx] = gi_pack8(e);
    f.v_a[idx] = var;
    f.m_hist[idx] = make_float4(m1, m2, n, 0.f);
}

// ---- SVGF: a-trous ---------------------------------------------------------------
// 5x5 B3 kernel at step s. Edge stopping: the tap's distance from the centre's
// tangent plane against the pixel footprint, the normals, and the luminance
// against the centre's (3x3-prefiltered) standard deviation.
__global__ void __launch_bounds__(128) k_gi_atrous(const GiFrame f, const uint4* __restrict__ ein, const float* __restrict__ vin,
                                                   uint4* __restrict__ eout, float* __restrict__ vout, int step, int first) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= f.w2 || j >= f.h2) return;
    const size_t idx = (size_t)j * f.w2 + i;
    const uint2 g = f.hg[idx];
    if (!(g.y & GG_GI)) {
        eout[idx] = make_uint4(0u, 0u, 0u, 0u);
        vout[idx] = 0.f;
        if (first) f.e_hist[idx] = make_uint4(0u, 0u, 0u, 0u);
        return;
    }
    Gd d;
    gd_unpack(g, d);
    float Pc[3];
    cam_point(f.cur, half_sx(f.cur, i), half_sy(f.cur, j), d.t, Pc);
    float ec[8];
    gi_unpack8(ein[idx], ec);
    const float lc = lum8(f, ec);
    // 3x3 Gaussian of the variance (the edge-stopping sigma)
    float vs = 0.f, vw = 0.f;
    for (int b = -1; b <= 1; ++b)
        for (int a = -1; a <= 1; ++a) {
            const int x = i + a, y = j + b;
            if (x < 0 || y < 0 || x >= f.w2 || y >= f.h2) continue;
            const size_t q = (size_t)y * f.w2 + x;
            if (!(f.hg[q].y & GG_GI)) continue;
            const float w = (a ? 0.5f : 1.f) * (b ? 0.5f : 1.f);
            vs += w * vin[q];
            vw += w;
        }
    const float sl = 4.f * sqrtf(fmaxf(vs / fmaxf(vw, 1e-6f), 0.f)) + 1e-4f;
    // pixel footprint (half res) at the centre's distance
    const float foot = 2.f * f.cur.th / (float)f.h2 * fmaxf(d.t, 0.1f);
    const float k3[3] = { 0.375f, 0.25f, 0.0625f };
    float acc[8], wsum = 0.f, vacc = 0.f;
#pragma unroll
    for (int b = 0; b < 8; ++b) acc[b] = 0.f;
    for (int bj = -2; bj <= 2; ++bj)
        for (int bi = -2; bi <= 2; ++bi) {
            const int x = i + bi * step, y = j + bj * step;
            if (x < 0 || y < 0 || x >= f.w2 || y >= f.h2) continue;
            const size_t q = (size_t)y * f.w2 + x;
            float w = k3[abs(bi)] * k3[abs(bj)];
            float eq[8];
            if (bi || bj) {
                const uint2 gq = f.hg[q];
                if (!(gq.y & GG_GI)) continue;
                Gd dq;
                gd_unpack(gq, dq);
                if (dq.cls != d.cls) continue;
                const float nd = fmaxf(0.f, d.n[0] * dq.n[0] + d.n[1] * dq.n[1] + d.n[2] * dq.n[2]);
                float n2 = nd * nd, n4 = n2 * n2, n8 = n4 * n4, n16 = n8 * n8, n32 = n16 * n16;
                const float wn = n32 * n32;                 // ^64
                if (wn < 1e-4f) continue;
                float Pq[3];
                cam_point(f.cur, half_sx(f.cur, x), half_sy(f.cur, y), dq.t, Pq);
                const float pd = fabsf(d.n[0] * (Pq[0] - Pc[0]) + d.n[1] * (Pq[1] - Pc[1]) + d.n[2] * (Pq[2] - Pc[2]));
                const float dist = sqrtf((float)(bi * bi + bj * bj)) * (float)step;
                const float wz = __expf(-pd / (foot * dist + 1e-3f));
                gi_unpack8(ein[q], eq);
                const float wl = __expf(-fabsf(lum8(f, eq) - lc) / sl);
                w *= wn * wz * wl;
                if (w < 1e-6f) continue;
            } else {
#pragma unroll
                for (int b = 0; b < 8; ++b) eq[b] = ec[b];
            }
#pragma unroll
            for (int b = 0; b < 8; ++b) acc[b] += w * eq[b];
            vacc += w * w * vin[q];
            wsum += w;
        }
    const float iw = 1.f / fmaxf(wsum, 1e-9f);
#pragma unroll
    for (int b = 0; b < 8; ++b) acc[b] *= iw;
    const uint4 o = gi_pack8(acc);
    eout[idx] = o;
    vout[idx] = vacc * iw * iw;
    if (first) f.e_hist[idx] = o;
}

// ---- radiance cache --------------------------------------------------------------
__global__ void __launch_bounds__(128) k_cache_update(const GiFrame f, const uint4* __restrict__ e) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= f.w2 || j >= f.h2) return;
    const size_t idx = (size_t)j * f.w2 + i;
    const uint2 g = f.hg[idx];
    if (!(g.y & GG_GI) || !f.cache.meta) return;
    Gd d;
    gd_unpack(g, d);
    float P[3];
    cam_point(f.cur, half_sx(f.cur, i), half_sy(f.cur, j), d.t, P);
    uint32_t slot, key;
    gi_cache_cell(f.cur.cx, f.cur.cy, f.cur.cz, P[0], P[1], P[2], d.n[0], d.n[1], d.n[2], d.t, &slot, &key);
    const uint32_t fr = 1u + (f.frame % 0xFFFFFEu);
    int found = -1, empty = -1, oldest = -1;
    uint32_t oldest_age = 0;
    for (int k = 0; k < GI_CACHE_PROBE; ++k) {
        const uint32_t s = (slot + (uint32_t)k) & (GI_CACHE_N - 1u);
        const uint2 m = f.cache.meta[s];
        if (m.x == key) { found = (int)s; break; }
        if (!m.x) { if (empty < 0) empty = (int)s; continue; }
        const uint32_t age = (fr - (m.y >> 8)) & 0xFFFFFFu;
        if (age > oldest_age) { oldest_age = age; oldest = (int)s; }
    }
    if (found < 0) {
        if (empty >= 0 && atomicCAS(&f.cache.meta[empty].x, 0u, key) == 0u) {
            found = empty;
        } else if (oldest >= 0 && oldest_age > 64u) {             // replace a cell unused for > 64 frames
            const uint32_t old = f.cache.meta[oldest].x;
            if (old != key && atomicCAS(&f.cache.meta[oldest].x, old, key) == old) {
                atomicExch(&f.cache.meta[oldest].y, 0u);
                found = oldest;
            }
        }
        if (found < 0) return;
    }
    const uint32_t my = atomicAdd(&f.cache.meta[found].y, 0u);
    if ((my >> 8) == fr) return;                                   // another pixel updated the cell this frame
    const uint32_t n = my & 255u, nn = n < 255u ? n + 1u : 255u;
    if (atomicCAS(&f.cache.meta[found].y, my, (fr << 8) | nn) != my) return;
    float ev[8];
    gi_unpack8(e[idx], ev);
    if (bad8(ev)) return;
    if (n) {
        float eo[8];
        gi_unpack8(f.cache.e[found], eo);
        const float a = 1.f / fminf((float)nn, 32.f);
#pragma unroll
        for (int b = 0; b < 8; ++b) ev[b] = eo[b] + (ev[b] - eo[b]) * a;
    }
    f.cache.e[found] = gi_pack8(ev);
}

// ---- SVGF: joint-bilateral upsample to the internal resolution ----------------------
__global__ void __launch_bounds__(128) k_gi_upsample(const GiFrame f, const uint4* __restrict__ e) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= f.cur.w || y >= f.cur.h) return;
    const size_t idx = (size_t)y * f.cur.w + x;
    const uint2 g = f.fg[idx];
    uint4 out = make_uint4(0xBC00BC00u, 0u, 0u, 0u);              // band 0 = -1: no GI value
    if (g.y & GG_GI) {
        Gd d;
        gd_unpack(g, d);
        float P[3];
        const float sx = (float)x + 0.5f + f.cur.jx, sy = (float)y + 0.5f + f.cur.jy;
        cam_point(f.cur, sx, sy, d.t, P);
        const float c0 = ((float)x - (float)f.cur.hx) * 0.5f, c1 = ((float)y - (float)f.cur.hy) * 0.5f;
        const int i0 = (int)floorf(c0), j0 = (int)floorf(c1);
        const float fx = c0 - (float)i0, fy = c1 - (float)j0;
        const float foot = 2.f * f.cur.th / (float)f.h2 * fmaxf(d.t, 0.1f);
        float acc[8], wsum = 0.f, best = 0.f;
        int bestq = -1;
#pragma unroll
        for (int b = 0; b < 8; ++b) acc[b] = 0.f;
        // the 2x2 bilinear taps (k < 4); if none of them is this surface, the 4x4 around them (geometric weights only)
        bool ring = false;
        for (int k = 0; k < 20; ++k) {
            int a, b;
            if (k < 4) { a = i0 + (k & 1); b = j0 + (k >> 1); }
            else {
                if (k == 4) ring = wsum <= 1e-3f;
                if (!ring) break;
                const int r = k - 4;                                // the 16 taps of the 4x4 (the inner 2x2 again: harmless)
                a = i0 - 1 + (r & 3); b = j0 - 1 + (r >> 2);
            }
            if (a < 0 || b < 0 || a >= f.w2 || b >= f.h2) continue;
            const size_t q = (size_t)b * f.w2 + a;
            const uint2 gq = f.hg[q];
            if (!(gq.y & GG_GI)) continue;
            Gd dq;
            gd_unpack(gq, dq);
            if (dq.cls != d.cls) continue;
            float Pq[3];
            cam_point(f.cur, half_sx(f.cur, a), half_sy(f.cur, b), dq.t, Pq);
            const float pd = fabsf(d.n[0] * (Pq[0] - P[0]) + d.n[1] * (Pq[1] - P[1]) + d.n[2] * (Pq[2] - P[2]));
            const float nd = fmaxf(0.f, d.n[0] * dq.n[0] + d.n[1] * dq.n[1] + d.n[2] * dq.n[2]);
            float n2 = nd * nd, n4 = n2 * n2, n8 = n4 * n4;
            const float wg = n8 * n8 * __expf(-pd / (2.f * foot + 0.02f));
            const float wb = k < 4 ? ((k & 1) ? fx : 1.f - fx) * ((k >> 1) ? fy : 1.f - fy) + 1e-3f : 1.f;
            const float w = wg * wb;
            if (wg > best) { best = wg; bestq = (int)q; }
            if (w < 1e-5f) continue;
            float eq[8];
            gi_unpack8(e[q], eq);
#pragma unroll
            for (int bb = 0; bb < 8; ++bb) acc[bb] += w * eq[bb];
            wsum += w;
        }
        if (wsum > 1e-5f) {
            const float iw = 1.f / wsum;
#pragma unroll
            for (int bb = 0; bb < 8; ++bb) acc[bb] *= iw;
            out = gi_pack8(acc);
        } else if (bestq >= 0 && best > 1e-3f) {
            out = e[bestq];
        }
    }
    f.e_full[idx] = out;
}

// ---- sun visibility: temporal (alpha 0.2) + 3x3 a-trous ------------------------------
__global__ void __launch_bounds__(128) k_vis_temporal(const GiFrame f) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= f.cur.w || y >= f.cur.h) return;
    const size_t idx = (size_t)y * f.cur.w + x;
    const float v = (float)f.v_smp[idx] * (1.f / 255.f);
    const uint2 g = f.fg[idx];
    Gd d;
    gd_unpack(g, d);
    float vt = v;
    unsigned char age = 1;
    if (d.cls != GC_SKY && f.hist_ok) {
        float X, Y, P[3], tp;
        if (reproject(f, (float)x + 0.5f + f.cur.jx, (float)y + 0.5f + f.cur.jy, d.t, &X, &Y, P, &tp)) {
            const float c0 = X - 0.5f - f.prev.jx, c1 = Y - 0.5f - f.prev.jy;
            const int i0 = (int)floorf(c0), j0 = (int)floorf(c1);
            const float fx = c0 - (float)i0, fy = c1 - (float)j0;
            float hs = 0.f, ws = 0.f, am = 0.f;
            for (int k = 0; k < 4; ++k) {
                const int a = i0 + (k & 1), b = j0 + (k >> 1);
                if (a < 0 || b < 0 || a >= f.prev.w || b >= f.prev.h) continue;
                const size_t q = (size_t)b * f.prev.w + a;
                if (!same_surface(f, f.fg_prev[q], (float)a + 0.5f + f.prev.jx, (float)b + 0.5f + f.prev.jy, P, d.n, d.cls, tp, 0.02f, 0.05f,
                                  0.9f)) continue;
                const float w = ((k & 1) ? fx : 1.f - fx) * ((k >> 1) ? fy : 1.f - fy);
                if (w <= 0.f) continue;
                hs += w * __half2float(f.v_hist_prev[q]);
                am = fmaxf(am, (float)f.v_age_prev[q]);
                ws += w;
            }
            if (ws > 1e-3f) {
                vt = hs / ws + (v - hs / ws) * 0.2f;
                age = (unsigned char)fminf(am + 1.f, 255.f);
            }
        }
    }
    f.v_hist[idx] = __float2half(vt);
    f.v_age[idx] = age;
}
__global__ void __launch_bounds__(128) k_vis_spatial(const GiFrame f) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= f.cur.w || y >= f.cur.h) return;
    const size_t idx = (size_t)y * f.cur.w + x;
    const uint2 g = f.fg[idx];
    Gd d;
    gd_unpack(g, d);
    if (d.cls == GC_SKY) { f.v_out[idx] = __float2half(1.f); return; }
    float P[3];
    cam_point(f.cur, (float)x + 0.5f + f.cur.jx, (float)y + 0.5f + f.cur.jy, d.t, P);
    const float foot = 2.f * f.cur.th / (float)f.cur.h * fmaxf(d.t, 0.1f);
    float s = 0.f, ws = 0.f;
    for (int b = -1; b <= 1; ++b)
        for (int a = -1; a <= 1; ++a) {
            const int xx = x + a, yy = y + b;
            if (xx < 0 || yy < 0 || xx >= f.cur.w || yy >= f.cur.h) continue;
            const size_t q = (size_t)yy * f.cur.w + xx;
            float w = (a ? 0.5f : 1.f) * (b ? 0.5f : 1.f);
            if (a || b) {
                Gd dq;
                gd_unpack(f.fg[q], dq);
                if (dq.cls != d.cls) continue;
                const float nd = d.n[0] * dq.n[0] + d.n[1] * dq.n[1] + d.n[2] * dq.n[2];
                if (nd < 0.8f) continue;
                float Pq[3];
                cam_point(f.cur, (float)xx + 0.5f + f.cur.jx, (float)yy + 0.5f + f.cur.jy, dq.t, Pq);
                const float pd = fabsf(d.n[0] * (Pq[0] - P[0]) + d.n[1] * (Pq[1] - P[1]) + d.n[2] * (Pq[2] - P[2]));
                w *= __expf(-pd / (2.f * foot + 0.01f));
            }
            s += w * __half2float(f.v_hist[q]);
            ws += w;
        }
    f.v_out[idx] = __float2half(s / fmaxf(ws, 1e-6f));
}

// ---- traced reflections: temporal filter ------------------------------------------------
// The reflected radiance of a water / ice / wet pixel follows its surface's
// reprojection (bilinear over the previous grid samples of the same surface);
// a moving camera moves the reflected image differently (parallax), so the
// history is clamped to the range of this frame's 3 x 3 samples of the same
// class. alpha 0.2. Half res (refl_mode 1, hg guides) or full res (2, fg).
__global__ void __launch_bounds__(128) k_refl_temporal(const GiFrame f) {
    const bool half = f.refl_mode == 1;
    const int gw = half ? f.w2 : f.cur.w, gh = half ? f.h2 : f.cur.h;
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= gw || j >= gh) return;
    const size_t idx = (size_t)j * gw + i;
    const uint4 rv = f.r_cur[idx];
    float rc[8];
    gi_unpack8(rv, rc);
    if (!(rc[0] >= 0.f)) { f.r_hist[idx] = rv; return; }
    const uint2 g = half ? f.hg[idx] : f.fg[idx];
    Gd d;
    gd_unpack(g, d);
    float lo[8], hi[8];
#pragma unroll
    for (int b = 0; b < 8; ++b) { lo[b] = rc[b]; hi[b] = rc[b]; }
    for (int b = -1; b <= 1; ++b)
        for (int a = -1; a <= 1; ++a) {
            const int x = i + a, y = j + b;
            if ((!a && !b) || x < 0 || y < 0 || x >= gw || y >= gh) continue;
            const size_t q = (size_t)y * gw + x;
            if ((int)(((half ? f.hg[q] : f.fg[q]).y >> 24) & 15u) != d.cls) continue;
            float rq[8];
            gi_unpack8(f.r_cur[q], rq);
            if (!(rq[0] >= 0.f)) continue;
#pragma unroll
            for (int bb = 0; bb < 8; ++bb) { lo[bb] = fminf(lo[bb], rq[bb]); hi[bb] = fmaxf(hi[bb], rq[bb]); }
        }
    float out[8];
#pragma unroll
    for (int b = 0; b < 8; ++b) out[b] = rc[b];
    if (f.hist_ok && f.r_hist_prev) {
        const float sx = half ? half_sx(f.cur, i) : (float)i + 0.5f + f.cur.jx, sy = half ? half_sy(f.cur, j) : (float)j + 0.5f + f.cur.jy;
        float X, Y, P[3], tp;
        if (reproject(f, sx, sy, d.t, &X, &Y, P, &tp)) {
            const float c0 = half ? (X - (float)f.prev.hx - 0.5f - f.prev.jx) * 0.5f : X - 0.5f - f.prev.jx;
            const float c1 = half ? (Y - (float)f.prev.hy - 0.5f - f.prev.jy) * 0.5f : Y - 0.5f - f.prev.jy;
            const int i0 = (int)floorf(c0), j0 = (int)floorf(c1);
            const float fx = c0 - (float)i0, fy = c1 - (float)j0;
            float h[8], ws = 0.f;
#pragma unroll
            for (int b = 0; b < 8; ++b) h[b] = 0.f;
            for (int k = 0; k < 4; ++k) {
                const int a = i0 + (k & 1), b = j0 + (k >> 1);
                if (a < 0 || b < 0 || a >= gw || b >= gh) continue;
                const size_t q = (size_t)b * gw + a;
                const float psx = half ? half_sx(f.prev, a) : (float)a + 0.5f + f.prev.jx, psy = half ? half_sy(f.prev, b) : (float)b + 0.5f + f.prev.jy;
                if (!same_surface(f, half ? f.hg_prev[q] : f.fg_prev[q], psx, psy, P, d.n, d.cls, tp, 0.02f, 0.05f, 0.9f)) continue;
                float hq[8];
                gi_unpack8(f.r_hist_prev[q], hq);
                if (!(hq[0] >= 0.f)) continue;
                const float w = ((k & 1) ? fx : 1.f - fx) * ((k >> 1) ? fy : 1.f - fy);
                if (w <= 0.f) continue;
#pragma unroll
                for (int bb = 0; bb < 8; ++bb) h[bb] += w * hq[bb];
                ws += w;
            }
            if (ws > 1e-3f) {
#pragma unroll
                for (int b = 0; b < 8; ++b) {
                    const float hv = fminf(fmaxf(h[b] / ws, lo[b]), hi[b]);
                    out[b] = hv + (rc[b] - hv) * 0.2f;
                }
            }
        }
    }
    f.r_hist[idx] = gi_pack8(out);
}

// ---- TAA / TAAU --------------------------------------------------------------------
__device__ __forceinline__ float3 rgb2ycocg(float3 c) {
    return make_float3(0.25f * c.x + 0.5f * c.y + 0.25f * c.z, 0.5f * c.x - 0.5f * c.z, -0.25f * c.x + 0.5f * c.y - 0.25f * c.z);
}
__device__ __forceinline__ float3 ycocg2rgb(float3 c) { return make_float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z); }
__device__ __forceinline__ float luma(float3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }
// Catmull-Rom weights of the 4 taps around a coordinate with fraction t
__device__ __forceinline__ void cr_w(float t, float w[4]) {
    const float t2 = t * t, t3 = t2 * t;
    w[0] = -0.5f * t3 + t2 - 0.5f * t;
    w[1] = 1.5f * t3 - 2.5f * t2 + 1.f;
    w[2] = -1.5f * t3 + 2.f * t2 + 0.5f * t;
    w[3] = 0.5f * t3 - 0.5f * t2;
}

__global__ void __launch_bounds__(128) k_taa(const GiFrame f) {
    const int X = blockIdx.x * blockDim.x + threadIdx.x, Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= f.ow || Y >= f.oh) return;
    const int w = f.cur.w, h = f.cur.h;
    const float rx = (float)w / (float)f.ow, ry = (float)h / (float)f.oh;
    const float sx = ((float)X + 0.5f) * rx, sy = ((float)Y + 0.5f) * ry;    // output pixel centre in internal pixels
    const int ci = min(max((int)floorf(sx), 0), w - 1), cj = min(max((int)floorf(sy), 0), h - 1);
    const size_t oidx = (size_t)Y * f.ow + X;
    const float4 c0 = f.color[(size_t)cj * w + ci];
    if (c0.w < 0.f) { taa_st(f.taa, oidx, make_float4(c0.x, c0.y, c0.z, -1.f)); return; }   // a debug view's display value
    const float k = __ldg(f.k);
    // current samples: neighbourhood statistics (YCoCg, exposed) and the reconstruction
    float3 m1 = make_float3(0.f, 0.f, 0.f), m2 = make_float3(0.f, 0.f, 0.f), cur = make_float3(0.f, 0.f, 0.f);
    float cw = 0.f, conf = 0.f, nw = 0.f;
    // the current frame at this output pixel: a Gaussian of the 3 x 3 jittered samples around its centre
    // (TAAU: sigma 0.47 internal pixels x the upscale; TAA: 0.4 pixels), so the estimate moves smoothly with
    // the jitter instead of jumping between sub-pixel details (flicker of edges and distant detail)
    const float sig = f.taau ? 0.47f * fmaxf(rx, 1.f) : 0.4f;
    for (int b = -1; b <= 1; ++b)
        for (int a = -1; a <= 1; ++a) {
            const int x = min(max(ci + a, 0), w - 1), y = min(max(cj + b, 0), h - 1);
            const float4 c = f.color[(size_t)y * w + x];
            if (c.w < 0.f) continue;
            float3 v = make_float3(c.x * k, c.y * k, c.z * k);
            if (!(fabsf(v.x) + fabsf(v.y) + fabsf(v.z) < 1e30f)) { if (f.nan && !a && !b) atomicAdd(f.nan, 1u); v = make_float3(0.f, 0.f, 0.f); }
            const float3 yc = rgb2ycocg(v);
            m1.x += yc.x; m1.y += yc.y; m1.z += yc.z;
            m2.x += yc.x * yc.x; m2.y += yc.y * yc.y; m2.z += yc.z * yc.z;
            nw += 1.f;
            const float dx = (float)(ci + a) + 0.5f + f.cur.jx - sx, dy = (float)(cj + b) + 0.5f + f.cur.jy - sy;
            const float wt = __expf(-(dx * dx + dy * dy) / (2.f * sig * sig));
            const float lw = wt / (1.f + luma(v));                 // luminance-weighted reconstruction
            cur.x += lw * v.x; cur.y += lw * v.y; cur.z += lw * v.z;
            cw += lw;
            conf = fmaxf(conf, wt);
        }
    if (cw > 0.f) { cur.x /= cw; cur.y /= cw; cur.z /= cw; }
    const float in = 1.f / fmaxf(nw, 1.f);
    m1.x *= in; m1.y *= in; m1.z *= in;
    const float3 sd = make_float3(sqrtf(fmaxf(m2.x * in - m1.x * m1.x, 0.f)), sqrtf(fmaxf(m2.y * in - m1.y * m1.y, 0.f)),
                                  sqrtf(fmaxf(m2.z * in - m1.z * m1.z, 0.f)));
    // history: reproject the visible surface of the nearest internal sample
    float age = 0.f, aw = 0.f;
    float3 hist = cur;
    bool ok = false;
    float mv = 0.f;
    if (f.hist_ok && f.taa_prev) {
        Gd d;
        gd_unpack(f.fg[(size_t)cj * w + ci], d);
        float Xp, Yp, P[3], tp;
        const float ssx = (float)ci + 0.5f + f.cur.jx, ssy = (float)cj + 0.5f + f.cur.jy;
        if (reproject(f, ssx, ssy, d.t, &Xp, &Yp, P, &tp)) {
            // the previous frame's samples around the reprojected point: is any of the 2 x 2 the same surface? (The
            // nearest one alone flips at silhouettes under the jitter, which resets edge pixels and makes them flicker.)
            const int i0 = (int)floorf(Xp - 0.5f - f.prev.jx), j0 = (int)floorf(Yp - 0.5f - f.prev.jy);
            bool seen = false;
            for (int q = 0; q < 4 && !seen; ++q) {
                const int pi = i0 + (q & 1), pj = j0 + (q >> 1);
                if (pi < 0 || pj < 0 || pi >= f.prev.w || pj >= f.prev.h) continue;
                seen = same_surface(f, f.fg_prev[(size_t)pj * f.prev.w + pi], (float)pi + 0.5f + f.prev.jx, (float)pj + 0.5f + f.prev.jy, P, d.n,
                                    d.cls, tp, 0.02f, 0.05f, 0.9f);
            }
            if (seen) {
                // history coordinate of this output pixel: its centre moved by the sample's motion (output pixels)
                const float mx = (Xp - ssx) / rx, my = (Yp - ssy) / ry;
                mv = sqrtf(mx * mx + my * my);
                const float hx = (float)X + 0.5f + mx - 0.5f, hy = (float)Y + 0.5f + my - 0.5f;   // tap space (centre at integer)
                const int hi = (int)floorf(hx), hj = (int)floorf(hy);
                float wx[4], wy[4];
                cr_w(hx - (float)hi, wx);
                cr_w(hy - (float)hj, wy);
                float3 s = make_float3(0.f, 0.f, 0.f);
                float ws = 0.f;
                for (int b = 0; b < 4; ++b)
                    for (int a = 0; a < 4; ++a) {
                        const int x = min(max(hi - 1 + a, 0), f.ow - 1), y = min(max(hj - 1 + b, 0), f.oh - 1);
                        const float4 hv = taa_ld(f.taa_prev, (size_t)y * f.ow + x);
                        if (hv.w < 0.f) continue;
                        const float wt = wx[a] * wy[b];
                        s.x += wt * hv.x; s.y += wt * hv.y; s.z += wt * hv.z;
                        ws += wt;
                    }
                const int ni = min(max((int)floorf(hx + 0.5f), 0), f.ow - 1), nj = min(max((int)floorf(hy + 0.5f), 0), f.oh - 1);
                const float4 hn = taa_ld(f.taa_prev, (size_t)nj * f.ow + ni);
                if (ws > 0.2f && hn.w > 0.f) {
                    const float kh = __ldg(f.k_prev);
                    const float ek = kh > 0.f ? k / kh : 1.f;              // the history's exposure -> this frame's
                    hist = make_float3(fmaxf(s.x / ws, 0.f) * ek, fmaxf(s.y / ws, 0.f) * ek, fmaxf(s.z / ws, 0.f) * ek);
                    age = hn.w;
                    aw = 1.f;
                    ok = true;
                }
            }
        }
    }
    float3 o;
    if (!ok || cw <= 0.f) {
        o = cw > 0.f ? cur : hist;
        age = 1.f;
    } else {
        // clip the history to the current neighbourhood's variance box (YCoCg, gamma 1.25)
        const float g = (mv < 0.05f && age >= 16.f) ? f.gamma_still : 1.25f;
        float3 hy = rgb2ycocg(hist);
        const float3 lo = make_float3(m1.x - g * sd.x, m1.y - g * sd.y, m1.z - g * sd.z), hi = make_float3(m1.x + g * sd.x, m1.y + g * sd.y, m1.z + g * sd.z);
        const float3 cc = make_float3(0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z));
        const float3 ext = make_float3(0.5f * (hi.x - lo.x) + 1e-5f, 0.5f * (hi.y - lo.y) + 1e-5f, 0.5f * (hi.z - lo.z) + 1e-5f);
        const float3 dv = make_float3(hy.x - cc.x, hy.y - cc.y, hy.z - cc.z);
        const float ma = fmaxf(fabsf(dv.x / ext.x), fmaxf(fabsf(dv.y / ext.y), fabsf(dv.z / ext.z)));
        if (ma > 1.f) hy = make_float3(cc.x + dv.x / ma, cc.y + dv.y / ma, cc.z + dv.z / ma);
        hist = ycocg2rgb(hy);
        hist = make_float3(fmaxf(hist.x, 0.f), fmaxf(hist.y, 0.f), fmaxf(hist.z, 0.f));
        // history weight 0.9, 0.8 under fast motion (2 .. 10 output pixels per frame), less while the history is
        // young; 0.95 for a pixel that stands still (< 0.05 px) with a history of 16 frames or more (halves the
        // flicker of sub-pixel detail under the 8-phase jitter); TAAU: the current frame by its samples' coverage
        float ah = 0.9f - 0.1f * gsat((mv - 2.f) * 0.125f);
        if (mv < 0.05f && age >= 16.f) ah = f.ah_still;
        ah = fminf(ah, age / (age + 1.f));
        float ac = (1.f - ah) * (f.taau ? fmaxf(conf, 0.15f) : 1.f);
        const float wc = ac / (1.f + luma(cur)), wh = (1.f - ac) / (1.f + luma(hist));
        const float iw = 1.f / (wc + wh);
        o = make_float3((wc * cur.x + wh * hist.x) * iw, (wc * cur.y + wh * hist.y) * iw, (wc * cur.z + wh * hist.z) * iw);
        age = fminf(age + aw, 255.f);
    }
    if (!(fabsf(o.x) + fabsf(o.y) + fabsf(o.z) < 1e30f)) {
        if (f.nan) atomicAdd(f.nan, 1u);
        o = make_float3(0.f, 0.f, 0.f);
        age = 1.f;
    }
    taa_st(f.taa, oidx, make_float4(o.x, o.y, o.z, age));
}

// reference transport: the mean of the frames since the last reset (internal res == output res)
__global__ void __launch_bounds__(128) k_accum(const GiFrame f) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= f.ow || y >= f.oh) return;
    const int ci = min(x * f.cur.w / f.ow, f.cur.w - 1), cj = min(y * f.cur.h / f.oh, f.cur.h - 1);
    const size_t idx = (size_t)y * f.ow + x;
    const float4 c = f.color[(size_t)cj * f.cur.w + ci];
    if (c.w < 0.f) { taa_st(f.taa, idx, make_float4(c.x, c.y, c.z, -1.f)); return; }
    float3 v = make_float3(c.x, c.y, c.z);
    if (!(fabsf(v.x) + fabsf(v.y) + fabsf(v.z) < 1e30f)) { if (f.nan) atomicAdd(f.nan, 1u); v = make_float3(0.f, 0.f, 0.f); }
    float4 a = f.acc_n > 1 ? f.acc[idx] : make_float4(0.f, 0.f, 0.f, 0.f);
    a.x += v.x; a.y += v.y; a.z += v.z;
    f.acc[idx] = a;
    const float k = __ldg(f.k) / (float)f.acc_n;
    taa_st(f.taa, idx, make_float4(a.x * k, a.y * k, a.z * k, (float)min(f.acc_n, 255u)));
}

// ---- disocclusion check (renderer_disocc) ----------------------------------------------
// truly disoccluded: none of the previous frame's 2x2 samples around the reprojected
// point is the same surface even with a loose test (plane distance 5 % + 0.25 m,
// 45 deg, same class); such a pixel must have reset every history
enum { DC_PIX, DC_DIS, DC_TAA_OLD, DC_VIS_OLD, DC_GI_PIX, DC_GI_DIS, DC_GI_OLD, DC_TAA_MAX, DC_GI_MAX, DC_N };
static_assert(DC_N == GI_DISOCC_N, "render.h GI_DISOCC_N");
__global__ void __launch_bounds__(128) k_disocc(const GiFrame f, const TaaPx* __restrict__ taa, const unsigned char* __restrict__ vage,
                                                unsigned long long* acc) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= f.cur.w || y >= f.cur.h) return;
    const size_t idx = (size_t)y * f.cur.w + x;
    Gd d;
    gd_unpack(f.fg[idx], d);
    if (d.cls == GC_SKY) return;
    atomicAdd(acc + DC_PIX, 1ull);
    float X, Y, P[3], tp;
    bool seen = false;
    if (reproject(f, (float)x + 0.5f + f.cur.jx, (float)y + 0.5f + f.cur.jy, d.t, &X, &Y, P, &tp)) {
        const int i0 = (int)floorf(X - 0.5f - f.prev.jx), j0 = (int)floorf(Y - 0.5f - f.prev.jy);
        for (int k = 0; k < 4 && !seen; ++k) {
            const int a = i0 + (k & 1), b = j0 + (k >> 1);
            if (a < 0 || b < 0 || a >= f.prev.w || b >= f.prev.h) continue;
            seen = same_surface(f, f.fg_prev[(size_t)b * f.prev.w + a], (float)a + 0.5f + f.prev.jx, (float)b + 0.5f + f.prev.jy, P, d.n, d.cls, tp,
                                0.05f, 0.25f, 0.7071f);
        }
    }
    if (seen) return;
    atomicAdd(acc + DC_DIS, 1ull);
    const int ox = min(x * f.ow / f.cur.w, f.ow - 1), oy = min(y * f.oh / f.cur.h, f.oh - 1);
    const float age = taa_ld(taa, (size_t)oy * f.ow + ox).w;
    if (age > 2.f) atomicAdd(acc + DC_TAA_OLD, 1ull);
    atomicMax(acc + DC_TAA_MAX, (unsigned long long)fmaxf(age, 0.f));
    if (vage && vage[idx] > 2) atomicAdd(acc + DC_VIS_OLD, 1ull);
}
__global__ void __launch_bounds__(128) k_disocc_gi(const GiFrame f, const float4* __restrict__ m, unsigned long long* acc) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= f.w2 || j >= f.h2) return;
    const size_t idx = (size_t)j * f.w2 + i;
    const uint2 g = f.hg[idx];
    if (!(g.y & GG_GI)) return;
    Gd d;
    gd_unpack(g, d);
    atomicAdd(acc + DC_GI_PIX, 1ull);
    float X, Y, P[3], tp;
    bool seen = false;
    if (reproject(f, half_sx(f.cur, i), half_sy(f.cur, j), d.t, &X, &Y, P, &tp)) {
        const float c0 = (X - (float)f.prev.hx - 0.5f - f.prev.jx) * 0.5f, c1 = (Y - (float)f.prev.hy - 0.5f - f.prev.jy) * 0.5f;
        const int i0 = (int)floorf(c0), j0 = (int)floorf(c1);
        for (int k = 0; k < 4 && !seen; ++k) {
            const int a = i0 + (k & 1), b = j0 + (k >> 1);
            if (a < 0 || b < 0 || a >= f.w2 || b >= f.h2) continue;
            const uint2 gp = f.hg_prev[(size_t)b * f.w2 + a];
            if (!(gp.y & GG_GI)) continue;
            seen = same_surface(f, gp, half_sx(f.prev, a), half_sy(f.prev, b), P, d.n, d.cls, tp, 0.05f, 0.25f, 0.7071f);
        }
    }
    if (seen) return;
    atomicAdd(acc + DC_GI_DIS, 1ull);
    const float n = m[idx].z;
    if (n > 2.f) atomicAdd(acc + DC_GI_OLD, 1ull);
    atomicMax(acc + DC_GI_MAX, (unsigned long long)fmaxf(n, 0.f));
}

// ---- host API -----------------------------------------------------------------------------
struct GiPipe {
    GiCache cache;
};

GiPipe* gi_create(void) {
    GiPipe* g = (GiPipe*)calloc(1, sizeof(GiPipe));
    if (!g) return NULL;
    if (gck(cudaMalloc(&g->cache.meta, sizeof(uint2) * GI_CACHE_N), "radiance cache") ||
        gck(cudaMalloc(&g->cache.e, sizeof(uint4) * GI_CACHE_N), "radiance cache") ||
        gck(cudaMemset(g->cache.meta, 0, sizeof(uint2) * GI_CACHE_N), "radiance cache")) {
        cudaFree(g->cache.meta);
        cudaFree(g->cache.e);
        g->cache.meta = NULL;
        g->cache.e = NULL;
        fprintf(stderr, "gi: no radiance cache (not enough VRAM): secondary hits use the sky SH fallback\n");
    }
    return g;
}

void gi_destroy(GiPipe* g) {
    if (!g) return;
    cudaFree(g->cache.meta);
    cudaFree(g->cache.e);
    free(g);
}

GiCache gi_cache(GiPipe* g) { return g->cache; }

void gi_cache_clear(GiPipe* g, cudaStream_t s) {
    if (g->cache.meta) cudaMemsetAsync(g->cache.meta, 0, sizeof(uint2) * GI_CACHE_N, s);
}

static dim3 grid2(int w, int h, dim3 b) { return dim3((unsigned)((w + (int)b.x - 1) / (int)b.x), (unsigned)((h + (int)b.y - 1) / (int)b.y)); }

int gi_denoise(GiPipe* g, const GiFrame& f, cudaStream_t s) {
    const dim3 b(16, 8);
    const dim3 hgd = grid2(f.w2, f.h2, b), fgd = grid2(f.cur.w, f.cur.h, b);
    k_gi_temporal<<<hgd, b, 0, s>>>(f);
    const uint4* e = f.e_a;
    uint4* bufs[2] = { f.e_b, f.e_a };
    float* vb[2] = { f.v_b, f.v_a };
    const float* v = f.v_a;
    for (int it = 0; it < f.atrous; ++it) {
        k_gi_atrous<<<hgd, b, 0, s>>>(f, e, v, bufs[it & 1], vb[it & 1], 1 << it, it == 0);
        e = bufs[it & 1];
        v = vb[it & 1];
    }
    if (!f.atrous) cudaMemcpyAsync(f.e_hist, f.e_a, sizeof(uint4) * (size_t)f.w2 * f.h2, cudaMemcpyDeviceToDevice, s);
    if (g->cache.meta) {
        GiFrame fc = f;
        fc.cache = g->cache;
        k_cache_update<<<hgd, b, 0, s>>>(fc, e);
    }
    k_gi_upsample<<<fgd, b, 0, s>>>(f, e);
    return gck(cudaGetLastError(), "denoise");
}

int gi_vis_filter(const GiFrame& f, cudaStream_t s) {
    const dim3 b(16, 8);
    const dim3 fgd = grid2(f.cur.w, f.cur.h, b);
    k_vis_temporal<<<fgd, b, 0, s>>>(f);
    k_vis_spatial<<<fgd, b, 0, s>>>(f);
    return gck(cudaGetLastError(), "sun visibility filter");
}

int gi_refl_filter(const GiFrame& f, cudaStream_t s) {
    const dim3 b(16, 8);
    const int gw = f.refl_mode == 1 ? f.w2 : f.cur.w, gh = f.refl_mode == 1 ? f.h2 : f.cur.h;
    k_refl_temporal<<<grid2(gw, gh, b), b, 0, s>>>(f);
    return gck(cudaGetLastError(), "reflection filter");
}

int gi_resolve(const GiFrame& f, cudaStream_t s) {
    const dim3 b(16, 8);
    const dim3 ogd = grid2(f.ow, f.oh, b);
    if (f.acc) k_accum<<<ogd, b, 0, s>>>(f);
    else k_taa<<<ogd, b, 0, s>>>(f);
    return gck(cudaGetLastError(), f.acc ? "accumulation" : "taa");
}

int gi_disocc(const GiFrame& f, const TaaPx* taa_cur, const unsigned char* v_age, const float4* m_hist_cur, unsigned long long* acc,
              cudaStream_t s) {
    const dim3 b(16, 8);
    cudaMemsetAsync(acc, 0, sizeof(unsigned long long) * DC_N, s);
    if (!f.hist_ok) return gck(cudaGetLastError(), "disocclusion check");
    k_disocc<<<grid2(f.cur.w, f.cur.h, b), b, 0, s>>>(f, taa_cur, v_age, acc);
    if (f.gi_on && m_hist_cur) k_disocc_gi<<<grid2(f.w2, f.h2, b), b, 0, s>>>(f, m_hist_cur, acc);
    return gck(cudaGetLastError(), "disocclusion check");
}

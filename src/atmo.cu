// atmo.cu - device side of the spectral atmosphere (docs/render-upgrade-spec.md
// 4.3 and 4.6, milestone M2).
//
//   path columns   256 x 64, built on the host (spectral.c), uploaded once
//   multiple scat. 32 x 32 per g-point (24), built on the host by iterated orders
//   k_sky_view     192 x 108 sky-view LUT (Hillaire 2020 layout) for the camera's
//                  altitude: single scattering with exact phase functions and
//                  combined sun + view columns per g-point, multiple scattering
//                  from the host LUT, the hazy ground below the horizon. Stored
//                  as XYZ (texture) and per g-point (buffer, for the SH).
//                  Rebuilt when the sun moves more than 0.1 deg (or the camera
//                  altitude by 10 m): about 0.1 ms.
//   k_sky_sh       the sky-view per g-point projected on order-2 SH (sky
//                  ambient of the Low / Medium presets)
//   k_aerial       32^3 froxels of in-scattered light (XYZ) along the frame's
//                  camera rays, quadratic slices to max_dist; the view
//                  transmittance is analytic per pixel (render.cu)
//
// Light per g-point is radiance integrated over the g-point's 5 nm members;
// XYZ = sum_g L_g g_cmf[g] (x 683 for cd/m^2, applied in render.cu).
#include "spectral.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SKY_NSTEP 40           // = spectral.c's host twin
static_assert(SPEC_GPB == 4, "a band's g-points must fill one MS texture (float4)");
static_assert((SPEC_SV_W * SPEC_SV_H) % 8 == 0 && (SPEC_AP_N * SPEC_AP_N) % 8 == 0, "8 bands per texel");
#define AP_SUB    2            // aerial sub-steps per froxel slice

struct AtmoGPU {
    cudaArray_t a_col, a_ms[SPEC_NG / 4], a_sv, a_ap;
    float*  d_svspec;
    float*  d_gdiff;
    float*  d_sh;
    float*  d_probe;
    int     probe_cap;
    AtmoParams P;
    float   sv_sun_y, sv_cam_h;
    int     sv_valid;
    double  sun_disc;
};

static int ack(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return 0;
    fprintf(stderr, "atmo: %s: %s\n", what, cudaGetErrorString(e));
    return 1;
}

static int make_tex(cudaArray_t arr, unsigned long long* tex) {
    cudaResourceDesc rd;
    memset(&rd, 0, sizeof rd);
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;
    cudaTextureDesc td;
    memset(&td, 0, sizeof td);
    td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear;
    td.readMode = cudaReadModeElementType;
    td.normalizedCoords = 1;
    cudaTextureObject_t t = 0;
    if (ack(cudaCreateTextureObject(&t, &rd, &td, NULL), "texture")) return 1;
    *tex = (unsigned long long)t;
    return 0;
}
static int make_surf(cudaArray_t arr, unsigned long long* surf) {
    cudaResourceDesc rd;
    memset(&rd, 0, sizeof rd);
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;
    cudaSurfaceObject_t s = 0;
    if (ack(cudaCreateSurfaceObject(&s, &rd), "surface")) return 1;
    *surf = (unsigned long long)s;
    return 0;
}
static int tex2d(int w, int h, const float* data, int surface, cudaArray_t* arr, unsigned long long* tex, unsigned long long* surf) {
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<float4>();
    if (ack(cudaMallocArray(arr, &ch, w, h, surface ? cudaArraySurfaceLoadStore : 0), "array")) return 1;
    if (data && ack(cudaMemcpy2DToArray(*arr, 0, 0, data, (size_t)w * 16, (size_t)w * 16, h, cudaMemcpyHostToDevice), "upload"))
        return 1;
    if (make_tex(*arr, tex)) return 1;
    return surface ? make_surf(*arr, surf) : 0;
}

__device__ __forceinline__ float at_logmean(float fa, float fb, float ds) {   // exact for exponential f
    if (fa <= 0.f || fb <= 0.f) return 0.5f * (fa + fb) * ds;
    const float r = fa / fb;
    if (fabsf(r - 1.f) < 1e-3f) return 0.5f * (fa + fb) * ds;
    return ds * (fa - fb) / __logf(r);
}

// In-scatter integrand of band b's g-points at one point of a view ray (per
// unit TOA irradiance): single scattering with the combined sun + view
// columns, multiple scattering from the LUT with the view columns. The band's
// g-points are SPEC_GPB = 4 consecutive ones, i.e. one MS texture.
__device__ __forceinline__ void at_source_band(const AtmoParams& P, int b, float hh, float mus, float Av, float Hv, float pr, float ph,
                                               float f[SPEC_GPB]) {
    const float da = __expf(-hh / P.H), dh = __expf(-hh / P.Hh);
    const float2 cs = spec_columns(P, hh, mus);
    float x, y;
    spec_ms_uv(P.R, P.Rt, hh, mus, &x, &y);
    const float4 m = spec_tex2(P.t_ms[b], spec_tc(x, SPEC_MS_W), spec_tc(y, SPEC_MS_H));
    const float fd = spec_ms_fade(mus);
    const float M[4] = { m.x * fd, m.y * fd, m.z * fd, m.w * fd };
#pragma unroll
    for (int j = 0; j < SPEC_GPB; ++j) {
        const int g = b * SPEC_GPB + j;
        const float sr = P.g_sr[g] * da, sh = P.g_sh[g] * dh;
        const float tc = __expf(-(Av + cs.x) * P.g_a[g] - (Hv + cs.y) * P.g_b[g]);
        const float tv = __expf(-Av * P.g_a[g] - Hv * P.g_b[g]);
        f[j] = (sr * pr + sh * ph) * tc + (sr + sh) * M[j] * tv;
    }
}
// sum over 8 consecutive lanes (the bands of one texel)
__device__ __forceinline__ float at_sum8(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    return v;
}

// ---- sky-view LUT: one thread per texel and band (8 lanes per texel) ----------------
__global__ void __launch_bounds__(64) k_sky_view(const __grid_constant__ AtmoParams P, float sy) {
    const int b = threadIdx.x & 7;
    const int texel = blockIdx.x * 8 + (threadIdx.x >> 3);            // SV_W * SV_H is a multiple of 8
    const int i = texel % SPEC_SV_W, j = texel / SPEC_SV_W;
    const float zh = spec_sv_zh(P.R, P.cam_h);
    float zen, phi;
    spec_sv_angles(zh, i / (float)(SPEC_SV_W - 1), j / (float)(SPEC_SV_H - 1), &zen, &phi);
    const float dx = sinf(zen) * cosf(phi), dy = cosf(zen);
    const float sx = sqrtf(fmaxf(0.f, 1.f - sy * sy));
    const float cs = dx * sx + dy * sy;                       // the sun lies in the x-y plane
    const float pr = spec_phase_ray(cs), ph = spec_phase_hg(cs, P.haze_g);
    const float h0 = fmaxf(P.cam_h, 0.f), mu = dy;
    const bool ground = mu < spec_mu_horizon(P.R, h0);
    const float send = ground ? spec_dist_sphere(P.R, h0, mu, 0.f) : spec_dist_sphere(P.R, h0, mu, P.Rt - P.R);
    float acc[SPEC_GPB], fp[SPEC_GPB];
#pragma unroll
    for (int k = 0; k < SPEC_GPB; ++k) { acc[k] = 0.f; fp[k] = 0.f; }
    if (send > 0.f) {
        const float q = powf(1.f + send / 100.f, 1.f / SKY_NSTEP);
        float Av = 0.f, Hv = 0.f, sp = 0.f, dpa = 0.f, dph = 0.f, qk = 1.f, mus = 0.f;
        for (int k = 0; k <= SKY_NSTEP; ++k) {
            const float s = k == SKY_NSTEP ? send : 100.f * (qk - 1.f);
            qk *= q;
            const float hh = fmaxf(spec_alt_at(P.R, h0, mu, s), 0.f);
            mus = (s * cs + (P.R + h0) * sy) / (P.R + hh);
            const float da = __expf(-hh / P.H), dh = __expf(-hh / P.Hh);
            if (k > 0) { Av += at_logmean(dpa, da, s - sp) / P.H; Hv += at_logmean(dph, dh, s - sp) / P.Hh; }
            float f[SPEC_GPB];
            at_source_band(P, b, hh, mus, Av, Hv, pr, ph, f);
#pragma unroll
            for (int g = 0; g < SPEC_GPB; ++g) {
                if (k > 0) acc[g] += at_logmean(fp[g], f[g], s - sp);
                fp[g] = f[g];
            }
            sp = s; dpa = da; dph = dh;
        }
        if (ground) {   // the hazy ground below the horizon: Lambertian, albedo 0.15, sun + sky
            const float2 cg = spec_columns(P, 0.f, mus);
#pragma unroll
            for (int j2 = 0; j2 < SPEC_GPB; ++j2) {
                const int g = b * SPEC_GPB + j2;
                acc[j2] += P.ground_albedo * 0.31830989f *
                           (__expf(-(Av + cg.x) * P.g_a[g] - (Hv + cg.y) * P.g_b[g]) * fmaxf(mus, 0.f) +
                            __expf(-Av * P.g_a[g] - Hv * P.g_b[g]) * spec_gdiff(P, g, mus));
            }
        }
    }
    float X = 0.f, Y = 0.f, Z = 0.f;
    float* out = P.sv_spec + ((size_t)j * SPEC_SV_W + i) * SPEC_NG + b * SPEC_GPB;
#pragma unroll
    for (int j2 = 0; j2 < SPEC_GPB; ++j2) {
        const int g = b * SPEC_GPB + j2;
        const float L = acc[j2] * P.g_e[g];
        out[j2] = L;
        X += L * P.g_cmf[g][0]; Y += L * P.g_cmf[g][1]; Z += L * P.g_cmf[g][2];
    }
    X = at_sum8(X); Y = at_sum8(Y); Z = at_sum8(Z);
    if (b == 0) surf2Dwrite(make_float4(X, Y, Z, 0.f), (cudaSurfaceObject_t)P.s_sv, i * (int)sizeof(float4), j);
}

// bilinear sample of the per-g-point sky-view buffer at a world direction
__device__ __forceinline__ float at_sky_g(const AtmoParams& P, float dx, float dy, float dz, int g) {
    float x, y;
    spec_sky_tc(P, dx, dy, dz, &x, &y);                    // texture coordinates -> node space
    const float fx = fminf(fmaxf(x * SPEC_SV_W - 0.5f, 0.f), (float)(SPEC_SV_W - 1));
    const float fy = fminf(fmaxf(y * SPEC_SV_H - 0.5f, 0.f), (float)(SPEC_SV_H - 1));
    const int i = min((int)fx, SPEC_SV_W - 2), j = min((int)fy, SPEC_SV_H - 2);
    const float u = fx - (float)i, v = fy - (float)j;
    const float* b = P.sv_spec + ((size_t)j * SPEC_SV_W + i) * SPEC_NG + g;
    const float a00 = b[0], a10 = b[SPEC_NG], a01 = b[SPEC_SV_W * SPEC_NG], a11 = b[SPEC_SV_W * SPEC_NG + SPEC_NG];
    return (a00 * (1.f - u) + a10 * u) * (1.f - v) + (a01 * (1.f - u) + a11 * u) * v;
}

// ---- order-2 SH of the sky radiance, per g-point (one block per g-point) ----------------
#define SH_NT 64
#define SH_NP 128
__global__ void __launch_bounds__(256) k_sky_sh(const __grid_constant__ AtmoParams P) {
    const int g = blockIdx.x;
    float c[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) c[k] = 0.f;
    const float dth = 3.14159265f / SH_NT, dph = 6.2831853f / SH_NP;
    for (int n = threadIdx.x; n < SH_NT * SH_NP; n += blockDim.x) {
        const int it = n / SH_NP, ip = n % SH_NP;
        const float th = (it + 0.5f) * dth, ph = (ip + 0.5f) * dph;
        const float st = sinf(th);
        const float x = st * cosf(ph), y = cosf(th), z = st * sinf(ph);
        const float L = at_sky_g(P, x, y, z, g) * st * dth * dph;
        c[0] += L * 0.282095f;
        c[1] += L * 0.488603f * y; c[2] += L * 0.488603f * z; c[3] += L * 0.488603f * x;
        c[4] += L * 1.092548f * x * y; c[5] += L * 1.092548f * y * z; c[6] += L * 0.315392f * (3.f * z * z - 1.f);
        c[7] += L * 1.092548f * x * z; c[8] += L * 0.546274f * (x * x - y * y);
    }
    __shared__ float red[9][256];
#pragma unroll
    for (int k = 0; k < 9; ++k) red[k][threadIdx.x] = c[k];
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w >>= 1) {
        if ((int)threadIdx.x < w)
            for (int k = 0; k < 9; ++k) red[k][threadIdx.x] += red[k][threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x < 9) P.sh[g * 9 + threadIdx.x] = red[threadIdx.x][0];
}

// ---- aerial perspective: in-scatter (XYZ) per froxel; one thread per froxel
// column and band (8 lanes per column) ----------------------------------------
__global__ void __launch_bounds__(64) k_aerial(const __grid_constant__ AtmoParams P, float cam_h) {
    const int b = threadIdx.x & 7;
    const int col = blockIdx.x * 8 + (threadIdx.x >> 3);
    const int ix = col % SPEC_AP_N, iy = col / SPEC_AP_N;
    const float u = (2.f * (ix + 0.5f) / SPEC_AP_N - 1.f) * P.ap_th * P.ap_aspect;
    const float v = (1.f - 2.f * (iy + 0.5f) / SPEC_AP_N) * P.ap_th;
    float d[3];
    for (int k = 0; k < 3; ++k) d[k] = P.ap_fw[k] + P.ap_rt[k] * u + P.ap_up[k] * v;
    const float il = rsqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (int k = 0; k < 3; ++k) d[k] *= il;
    const float cs = d[0] * P.sun[0] + d[1] * P.sun[1] + d[2] * P.sun[2];
    const float pr = spec_phase_ray(cs), ph = spec_phase_hg(cs, P.haze_g);
    const float h0 = fmaxf(cam_h, 0.f), mu = d[1];
    float acc[SPEC_GPB], fp[SPEC_GPB];
    float Av = 0.f, Hv = 0.f, sp = 0.f, dpa = 0.f, dph = 0.f;
    const cudaSurfaceObject_t surf = (cudaSurfaceObject_t)P.s_ap;
    if (b == 0) surf3Dwrite(make_float4(0.f, 0.f, 0.f, 0.f), surf, ix * (int)sizeof(float4), iy, 0);
    for (int k = 0; k < (SPEC_AP_N - 1) * AP_SUB + 1; ++k) {
        const float w = (float)k / (float)((SPEC_AP_N - 1) * AP_SUB);
        const float s = P.max_dist * w * w;
        const float hh = fmaxf(spec_alt_at(P.R, h0, mu, s), 0.f);
        const float mus = (s * cs + (P.R + h0) * P.sun[1]) / (P.R + hh);
        const float da = __expf(-hh / P.H), dh = __expf(-hh / P.Hh);
        if (k > 0) { Av += at_logmean(dpa, da, s - sp) / P.H; Hv += at_logmean(dph, dh, s - sp) / P.Hh; }
        float f[SPEC_GPB];
        at_source_band(P, b, hh, mus, Av, Hv, pr, ph, f);
#pragma unroll
        for (int g = 0; g < SPEC_GPB; ++g) {
            if (k == 0) acc[g] = 0.f;
            else acc[g] += at_logmean(fp[g], f[g], s - sp);
            fp[g] = f[g];
        }
        sp = s; dpa = da; dph = dh;
        if (k > 0 && k % AP_SUB == 0) {
            float X = 0.f, Y = 0.f, Z = 0.f;
#pragma unroll
            for (int j2 = 0; j2 < SPEC_GPB; ++j2) {
                const int g = b * SPEC_GPB + j2;
                const float L = acc[j2] * P.g_e[g];
                X += L * P.g_cmf[g][0]; Y += L * P.g_cmf[g][1]; Z += L * P.g_cmf[g][2];
            }
            X = at_sum8(X); Y = at_sum8(Y); Z = at_sum8(Z);
            if (b == 0) surf3Dwrite(make_float4(X, Y, Z, 0.f), surf, ix * (int)sizeof(float4), iy, k / AP_SUB);
        }
    }
}

// ---- D3: band means of the GPU sky-view at given directions --------------------------------
__global__ void k_sky_probe(const __grid_constant__ AtmoParams P, const float* dirs, int n, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * SPEC_NB) return;
    const int di = i / SPEC_NB, b = i % SPEC_NB;
    float s = 0.f;
    for (int g = b * SPEC_GPB; g < (b + 1) * SPEC_GPB; ++g) s += at_sky_g(P, dirs[3 * di], dirs[3 * di + 1], dirs[3 * di + 2], g);
    out[i] = s / (float)SPEC_BW;
}

// ---- host API ---------------------------------------------------------------------------------
AtmoGPU* atmo_create(const SpectralAtmo* a) {
    AtmoGPU* g = (AtmoGPU*)calloc(1, sizeof(AtmoGPU));
    if (!g) return NULL;
    AtmoParams& P = g->P;
    P.R = (float)a->R; P.Rt = (float)a->top; P.H = (float)a->H; P.Hh = (float)a->haze_H;
    P.Hl = (float)sqrt((a->top - a->R) * (a->top + a->R));
    P.haze_g = (float)a->haze_g;
    P.ground_albedo = (float)SPEC_GROUND_ALBEDO;
    g->sun_disc = a->sun_disc > 0 ? a->sun_disc : 1.0;
    P.sun_theta = (float)(a->theta_sun * g->sun_disc);
    P.sun_scale = (float)(1.0 / (g->sun_disc * g->sun_disc));
    for (int k = 0; k < SPEC_NG; ++k) {
        P.g_e[k] = (float)a->g_e[k]; P.g_a[k] = (float)a->g_a[k]; P.g_b[k] = (float)a->g_b[k];
        P.g_sr[k] = (float)a->g_sr[k]; P.g_sh[k] = (float)a->g_sh[k];
        for (int c = 0; c < 3; ++c) P.g_cmf[k][c] = (float)a->g_cmf[k][c];
    }
    float* buf = (float*)malloc(sizeof(float) * 4 * SPEC_COL_W * SPEC_COL_H);
    int bad = !buf;
    if (!bad) {
        for (int j = 0; j < SPEC_COL_H; ++j)
            for (int i = 0; i < SPEC_COL_W; ++i) {
                float* o = buf + 4 * (j * SPEC_COL_W + i);
                o[0] = a->col[j][i][0]; o[1] = a->col[j][i][1]; o[2] = 0.f; o[3] = 0.f;
            }
        bad = tex2d(SPEC_COL_W, SPEC_COL_H, buf, 0, &g->a_col, &P.t_col, NULL);
    }
    for (int t = 0; t < SPEC_NG / 4 && !bad; ++t) {
        for (int i = 0; i < SPEC_MS_H; ++i)
            for (int j = 0; j < SPEC_MS_W; ++j)
                for (int c = 0; c < 4; ++c) buf[4 * (i * SPEC_MS_W + j) + c] = a->ms[4 * t + c][i][j];
        bad = tex2d(SPEC_MS_W, SPEC_MS_H, buf, 0, &g->a_ms[t], &P.t_ms[t], NULL);
    }
    free(buf);
    if (!bad) bad = tex2d(SPEC_SV_W, SPEC_SV_H, NULL, 1, &g->a_sv, &P.t_sv, &P.s_sv);
    if (!bad) {
        cudaChannelFormatDesc ch = cudaCreateChannelDesc<float4>();
        bad = ack(cudaMalloc3DArray(&g->a_ap, &ch, make_cudaExtent(SPEC_AP_N, SPEC_AP_N, SPEC_AP_N), cudaArraySurfaceLoadStore), "3D array") ||
              make_tex(g->a_ap, &P.t_ap) || make_surf(g->a_ap, &P.s_ap);
    }
    if (!bad) bad = ack(cudaMalloc(&g->d_svspec, sizeof(float) * SPEC_SV_W * SPEC_SV_H * SPEC_NG), "sky-view buffer") ||
                    ack(cudaMalloc(&g->d_gdiff, sizeof(float) * SPEC_NG * SPEC_MS_W), "gdiff") ||
                    ack(cudaMalloc(&g->d_sh, sizeof(float) * SPEC_NG * 9), "sh") ||
                    ack(cudaMemcpy(g->d_gdiff, a->gdiff, sizeof(float) * SPEC_NG * SPEC_MS_W, cudaMemcpyHostToDevice), "gdiff upload") ||
                    ack(cudaMemset(g->d_sh, 0, sizeof(float) * SPEC_NG * 9), "sh");
    P.sv_spec = g->d_svspec;
    P.gdiff = g->d_gdiff;
    P.sh = g->d_sh;
    if (bad) {
        atmo_destroy(g);
        return NULL;
    }
    return g;
}

void atmo_destroy(AtmoGPU* g) {
    if (!g) return;
    AtmoParams& P = g->P;
    if (P.t_col) cudaDestroyTextureObject((cudaTextureObject_t)P.t_col);
    for (int t = 0; t < SPEC_NG / 4; ++t) if (P.t_ms[t]) cudaDestroyTextureObject((cudaTextureObject_t)P.t_ms[t]);
    if (P.t_sv) cudaDestroyTextureObject((cudaTextureObject_t)P.t_sv);
    if (P.t_ap) cudaDestroyTextureObject((cudaTextureObject_t)P.t_ap);
    if (P.s_sv) cudaDestroySurfaceObject((cudaSurfaceObject_t)P.s_sv);
    if (P.s_ap) cudaDestroySurfaceObject((cudaSurfaceObject_t)P.s_ap);
    if (g->a_col) cudaFreeArray(g->a_col);
    for (int t = 0; t < SPEC_NG / 4; ++t) if (g->a_ms[t]) cudaFreeArray(g->a_ms[t]);
    if (g->a_sv) cudaFreeArray(g->a_sv);
    if (g->a_ap) cudaFreeArray(g->a_ap);
    cudaFree(g->d_svspec);
    cudaFree(g->d_gdiff);
    cudaFree(g->d_sh);
    cudaFree(g->d_probe);
    free(g);
}

const AtmoParams* atmo_params(const AtmoGPU* g) { return &g->P; }

static int atmo_sky(AtmoGPU* g, cudaStream_t s) {
    k_sky_view<<<SPEC_SV_W * SPEC_SV_H / 8, 64, 0, s>>>(g->P, g->P.sun[1]);
    k_sky_sh<<<SPEC_NG, 256, 0, s>>>(g->P);
    return ack(cudaGetLastError(), "k_sky_view / k_sky_sh");
}

// Per frame: the sun (world unit vector), camera altitude above the water
// plane, the loaded radius and the camera basis. The sky-view (+ SH) is rebuilt
// when the sun's elevation moved by more than 0.1 deg or the camera altitude by
// more than 10 m; the aerial froxels every frame. Returns 1 if the sky was rebuilt.
int atmo_frame(AtmoGPU* g, const float sun[3], float cam_h, float max_dist, const float fw[3], const float rt[3], const float up[3],
               float th, float aspect, cudaStream_t s, int force_sky) {
    AtmoParams& P = g->P;
    for (int k = 0; k < 3; ++k) { P.sun[k] = sun[k]; P.ap_fw[k] = fw[k]; P.ap_rt[k] = rt[k]; P.ap_up[k] = up[k]; }
    P.ap_th = th; P.ap_aspect = aspect;
    P.max_dist = max_dist > 1.f ? max_dist : 1.f;
    const float ch = fmaxf(cam_h, 0.f);
    int rebuilt = 0;
    if (force_sky || !g->sv_valid || fabsf(sun[1] - g->sv_sun_y) > 1.75e-3f || fabsf(ch - g->sv_cam_h) > 10.f) {
        P.cam_h = ch;
        if (atmo_sky(g, s)) return -1;
        g->sv_sun_y = sun[1];
        g->sv_cam_h = ch;
        g->sv_valid = 1;
        rebuilt = 1;
    }
    k_aerial<<<SPEC_AP_N * SPEC_AP_N / 8, 64, 0, s>>>(P, ch);
    if (ack(cudaGetLastError(), "k_aerial")) return -1;
    return rebuilt;
}

int atmo_sky_probe(AtmoGPU* g, const float sun[3], float cam_h, const float* dirs, int n, float* out, cudaStream_t s) {
    const float fw[3] = { 1, 0, 0 }, rt[3] = { 0, 0, 1 }, up[3] = { 0, 1, 0 };
    if (atmo_frame(g, sun, cam_h, 1000.f, fw, rt, up, 0.5f, 1.f, s, 1) < 0) return -1;
    if (g->probe_cap < n) {
        cudaFree(g->d_probe);
        g->d_probe = NULL;
        if (ack(cudaMalloc(&g->d_probe, sizeof(float) * (size_t)n * (3 + SPEC_NB)), "probe")) return -1;
        g->probe_cap = n;
    }
    float* d_dirs = g->d_probe;
    float* d_out = g->d_probe + 3 * n;
    if (ack(cudaMemcpyAsync(d_dirs, dirs, sizeof(float) * 3 * n, cudaMemcpyHostToDevice, s), "probe dirs")) return -1;
    k_sky_probe<<<(n * SPEC_NB + 63) / 64, 64, 0, s>>>(g->P, d_dirs, n, d_out);
    if (ack(cudaGetLastError(), "k_sky_probe") ||
        ack(cudaMemcpyAsync(out, d_out, sizeof(float) * n * SPEC_NB, cudaMemcpyDeviceToHost, s), "probe out") ||
        ack(cudaStreamSynchronize(s), "probe sync")) return -1;
    return 0;
}

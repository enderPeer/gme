/* spectral.h - spectral light for the smooth-surface renderer
 * (docs/render-upgrade-spec.md section 4, milestone M2; M5: the pigments of
 * section 5.1, the pigment table and the bio's light file, GMESPEC1).
 *
 * Host side (spectral.c, double precision, built once per planet): the star
 * (Planck), CIE 1931 colour matching, per-gas Rayleigh scattering with King
 * factors, CH4 absorption, the tholin haze calibrated to the bio's haze
 * factor, the atmosphere LUTs (path columns, multiple scattering by iterated
 * orders), a Monte Carlo sky reference, material and water spectra, the
 * specular albedo LUT and the D3 self-test.
 *
 * Shared (SPEC_HD, host + device): LUT parametrisations, phase functions, the
 * GGX BRDF with exact Fresnel, refraction, AgX and the sRGB transfer, so the
 * self-test checks the same code the GPU runs. Device only (__CUDACC__): the
 * texture samplers used by atmo.cu and render.cu.
 *
 * Spectral representation. The spec's 8 bands of 45 nm (380-740 nm) are the
 * unit of every surface and water table. Light in the atmosphere is carried in
 * 32 g-points (correlated k): the 9 samples at 5 nm of each band are sorted by
 * their vertical extinction (Rayleigh + CH4) and split into 4 groups, which are
 * grey to a good approximation, so multiple scattering, combined sun + view
 * paths and saturated CH4 bands stay right (D3 checks this against a 5 nm
 * Monte Carlo). Each g-point carries its own colour-matching weights, so
 * XYZ = 683 sum_g L_g g_cmf[g] keeps the colour of narrow absorption bands. */
#ifndef GME_SPECTRAL_H
#define GME_SPECTRAL_H

#include <math.h>
#include <stdint.h>
#include "planetfile.h"

#ifdef __CUDACC__
#define SPEC_HD static __host__ __device__ __forceinline__
#else
#define SPEC_HD static inline
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SPEC_NB    8            /* render bands */
#define SPEC_L0    380.0        /* first band edge, nm */
#define SPEC_BW    45.0         /* band width, nm */
#define SPEC_SUB   9            /* 5 nm sub-intervals per band */
#define SPEC_NF    (SPEC_NB * SPEC_SUB)   /* fine grid: 382.5 .. 737.5 nm */
#define SPEC_GPB   4            /* g-points per band */
#define SPEC_NG    (SPEC_NB * SPEC_GPB)   /* 32 g-points carry the light */
#define SPEC_GMEM  (SPEC_SUB / SPEC_GPB)  /* fine samples per g-point */

/* LUT sizes. Every LUT stores its values at nodes x = i / (N - 1). */
#define SPEC_COL_W 256          /* path-column ("transmittance") LUT: x_mu */
#define SPEC_COL_H 64           /*   x_r */
#define SPEC_MS_W  32           /* multiple scattering: mu_s */
#define SPEC_MS_H  32           /*   altitude */
#define SPEC_SV_W  192          /* sky-view: azimuth from the sun (0..pi) */
#define SPEC_SV_H  108          /*   zenith, non-linear around the horizon */
#define SPEC_AP_N  32           /* aerial-perspective froxels per axis */
#define SPEC_WT_N  64           /* water band transmittance vs depth */
#define SPEC_EL_N  32           /* specular directional albedo E(mu, alpha) */
#define SPEC_NIOR  4            /* dielectric classes */
#define SPEC_NMAT  16           /* material table (MAT_* ids) */

/* M5 pigments (spec 5.1): one hue bin per biomap palette index; index 0 (no hue,
 * e.g. the test pattern's palette 0) takes the default bin, chlorophyll-like */
#define SPEC_NHUE        16
#define SPEC_HUE_DEFAULT 8
#define SPEC_PIG_REF     200.0  /* PIG of a cell without morphology (a version 1 biomap) */
/* version 2 (biomap morphology, spec 5.3): the cell's mean pigment amount in 15 levels (level L = 1..15
 * is PIG 17 L; SPEC_PIG_LEVEL) and its Q band width bin w = 0..15 (sigma = SPEC_QW_NM(w) = 25 + 35 w / 15 nm; version 1: 25 nm) */
#define SPEC_NPIG        15
#define SPEC_NQW         16
#define SPEC_PIG_LEVEL(pig) (((int)(pig) + 8) / 17 < 1 ? 1 : (((int)(pig) + 8) / 17 > SPEC_NPIG ? SPEC_NPIG : ((int)(pig) + 8) / 17))
#define SPEC_QW_NM(w)    (25.0 + 35.0 * (double)(w) / 15.0)   /* bin 0 = the 25 nm band of the pigment table and of the simulation's gene 0 */
/* crowns: leaf area index of an occupied slot (over the crown's footprint), G of
 * random leaves; LAD = LAI pi a^2 / V = 3 LAI / (4 b), sigma = G LAD (spec 4.6) */
#define SPEC_CROWN_LAI   4.0f
#define SPEC_LEAF_G      0.5f

#define SPEC_MS_MU0   (-0.35f)  /* multiple-scattering LUT: mu_s range */
#define SPEC_MS_MU1   1.0f
#define SPEC_WT_D0    0.01f     /* water table: depth range (m), log steps */
#define SPEC_WT_D1    1000.0f
#define SPEC_GROUND_ALBEDO 0.15

enum { SPEC_IOR_ROCK = 0, SPEC_IOR_LEAF, SPEC_IOR_ICE, SPEC_IOR_WATER };

/* Options of spectral_build_atmo_ex (zero = defaults). */
typedef struct SpectralOpts {
    double p_bar;               /* surface pressure, bar (default 1, the bio spec's value) */
    double h2o;                 /* added H2O mole fraction (default 0) */
    double sun_disc;            /* disc radius scale k (radiance / k^2), default 1 */
    int    ms_method;           /* 0 = iterated orders (default), 1 = Hillaire's series */
    int    quiet;
    int    earth;               /* Earth preset: air at 1.01325 bar, g 9.80665, no haze */
    int    haze_q12;            /* M5 export: calibrate the haze to this bio haze factor (0 = the bio's formula from x_CH4) */
    int    ms_j0;               /* M5 export: multiple scattering only for sun cosine nodes >= j0 (0 = all; 8 = mu_s >= 0) */
} SpectralOpts;

/* Everything the renderer needs about light, for one planet. Host struct
 * (large: keep it on the heap). Band quantities are TOA-weighted band means. */
typedef struct SpectralAtmo {
    /* star */
    double teff, lum, a_au, s_top;          /* K, L_sun, AU, W/m^2 (= 1361 L / a^2) */
    double r_star, theta_sun, sun_disc;     /* R_sun units; disc angular radius (rad, true); --sun-disc */
    double par_share, par_wm2;              /* 400-700 nm share of S_top */
    double e_band[SPEC_NB + 1];             /* W/m^2 per band; [8] = NIR 740-900 */
    double e_nm[SPEC_NB];                   /* band-mean spectral irradiance, W/m^2/nm */
    double phot_band[SPEC_NB + 1];          /* umol photons / m^2 / s per band */
    double ev_band[SPEC_NB + 1];            /* mean photon energy (eV) */
    /* planet and gases */
    double p_bar, mbar, g, H, N, R, top, t_surf;   /* g/mol, m/s^2, m, m^-2, m, m, K */
    double ray_factor;                      /* sum x_i sigma_i / sigma_air */
    double x[ATM_N + 1];                    /* mole fractions (ATM_* order) + H2O */
    double ch4_col;                         /* km-amagat, vertical */
    int    ch4_file;                        /* 1 = data/ch4_k98.txt, 0 = built-in coarse band model */
    int    earth;
    /* vertical optical depths from the ground, band means */
    double tau_ray[SPEC_NB], tau_ch4[SPEC_NB], tau_haze[SPEC_NB], omega_haze[SPEC_NB];
    double tau_total[SPEC_NB];              /* -ln of the band-mean vertical transmittance */
    double tau_ray550, tau_ray550_earth;
    /* haze */
    int    haze_q12;                        /* the bio's formula from x_CH4 */
    double haze_target, haze_tau550, haze_achieved, haze_residual, haze_rayleigh_only, haze_ray_nogas;
    double haze_g, haze_H;
    int    haze_iters;
    /* fine grid (5 nm) */
    double lam[SPEC_NF], e_fine[SPEC_NF];   /* nm, W/m^2/nm */
    double a_fine[SPEC_NF];                 /* vertical tau of the air-profile terms (Rayleigh + CH4) */
    double b_fine[SPEC_NF];                 /* vertical haze tau */
    double r_fine[SPEC_NF], c_fine[SPEC_NF], w_fine[SPEC_NF];  /* Rayleigh tau, CH4 tau, haze single-scattering albedo */
    double cmf_fine[SPEC_NF][3];            /* mean xbar, ybar, zbar over each 5 nm interval */
    /* g-points: the 9 fine samples of each band sorted by vertical air-profile
     * extinction (Rayleigh + CH4) and split into 3 groups; each group is grey */
    int    g_member[SPEC_NG][SPEC_SUB];     /* fine sample indices (natural-breaks groups: 1..7 members) */
    int    g_nmem[SPEC_NG];
    int    groups_set;
    double g_e[SPEC_NG];                    /* TOA irradiance of the members, W/m^2 */
    double g_a[SPEC_NG], g_b[SPEC_NG];      /* vertical tau: air profile (Rayleigh + CH4), haze */
    double g_sr[SPEC_NG], g_sh[SPEC_NG];    /* scattering at the ground, 1/m: Rayleigh, haze */
    double g_cmf[SPEC_NG][3];               /* member-averaged xbar, ybar, zbar (TOA-weighted) */
    double g_par[SPEC_NG];                  /* TOA W/m^2 of the members inside 400-700 nm */
    double sig_r[SPEC_NB], sig_hs[SPEC_NB]; /* band means (report) */
    /* LUTs (float, uploaded by renderer_set_atmo) */
    float  col[SPEC_COL_H][SPEC_COL_W][2];  /* path columns to the top: air, haze (vertical-column units) */
    float  ms[SPEC_NG][SPEC_MS_H][SPEC_MS_W];      /* multiple scattering radiance per unit TOA irradiance */
    float  gdiff[SPEC_NG][SPEC_MS_W];       /* diffuse sky irradiance at the ground per unit TOA irradiance vs mu_s */
    int    ms_method;
    double ms_q[SPEC_NG];                   /* convergence ratio of the last orders (ground, high sun) */
    double build_s;                         /* seconds spent in spectral_build_atmo */
    /* colour */
    double cmf[SPEC_NB][3];                 /* band integrals of xbar, ybar, zbar (nm) */
    double cmf_wyman[SPEC_NB][3];           /* the same from the Wyman 2013 fit (check) */
    float  rgb_basis[3][SPEC_NB];           /* smooth reflectance basis: rho = r Br + g Bg + b Bb */
    /* surfaces and water */
    float  mat_rho[SPEC_NMAT][SPEC_NB];     /* spectral albedo per MAT_* id (pigmented tissue: M2 fallback) */
    float  mat_alpha[SPEC_NMAT];            /* GGX roughness */
    int    mat_ior[SPEC_NMAT];              /* SPEC_IOR_* */
    float  pyrite_f0[SPEC_NB];
    float  e_lut[SPEC_NIOR][SPEC_EL_N][SPEC_EL_N];  /* [ior][alpha][mu] */
    float  e_avg[SPEC_NIOR][SPEC_EL_N];
    float  water_a[SPEC_NB], water_b[SPEC_NB];      /* band-mean absorption (incl. CDOM + particles), scattering, 1/m */
    float  water_bm[SPEC_NB];                       /* molecular part of the scattering */
    float  water_T[SPEC_NB][SPEC_WT_N];     /* band-mean e^(-a d), log-spaced d */
    double water_aw[SPEC_NB + 1];           /* pure-water band means (report) */
    /* M5 pigmented tissue (spec 5.1), per hue bin (index 0 = SPEC_HUE_DEFAULT), PIG 200:
     * Soret + Q band from the hue bin, OD = PIG / 255 x 4 x eps, A = band mean of 1 - 10^-OD at 1 nm */
    float  pig_A[SPEC_NHUE][SPEC_NB];       /* absorptance */
    float  leaf_rho[SPEC_NHUE][SPEC_NB];    /* leaf (crowns, tufts, kelp): 0.04 + 0.44 (1 - A) */
    float  leaf_tau[SPEC_NHUE][SPEC_NB];    /* leaf transmittance 0.42 (1 - A) */
    float  mat_tis[SPEC_NHUE][SPEC_NB];     /* microbial mat: 0.03 + rho_sand (1 - A)^2 */
    double pig_A_nir[SPEC_NHUE];            /* absorptance 740-900 nm (export, probe) */
    /* version 2: the band-mean absorptance at pigment level L + 1 (PIG 17 (L + 1)) and Q band width bin w
     * (leaf rho / tau and the mat follow from it as above) */
    float  pig_A2[SPEC_NHUE][SPEC_NPIG][SPEC_NQW][SPEC_NB];
    void*  host;                            /* double-precision copies of the LUTs (spectral.c) */
    int    host_owned;
} SpectralAtmo;

/* M5 --export-spectrum: what the build of each haze bin did (GMESPEC1, spectrum_file.h) */
typedef struct SpectralExportInfo {
    int    haze_q12[8];
    double x_ch4[8], tau_h550[8], target[8], t_par[8];   /* CH4 mole fraction, haze, haze_q12 / 4096, PAR global T (zenith) */
    int    unreachable[8];                               /* Rayleigh + gas already below the target: no haze */
    double ch4_col[8];                                   /* km-amagat */
    double seconds;
} SpectralExportInfo;

/* Parameters of the device-side atmosphere (atmo.cu / render.cu). POD with the
 * same layout in C and CUDA. Texture / surface objects are CUDA handles. Light
 * is carried per g-point as radiance integrated over its members (W m^-2 sr^-1);
 * XYZ = 683 sum_g L_g g_cmf[g]. The sky-view and aerial in-scatter are stored as
 * XYZ without the 683. */
typedef struct AtmoParams {
    float R, Rt, H, Hh;                     /* ground and top radius, scale heights (m) */
    float Hl;                               /* sqrt(Rt^2 - R^2) */
    float haze_g;
    float ground_albedo;
    float sun[3];                           /* world-space unit vector toward the star */
    float sun_theta;                        /* drawn disc radius (rad) */
    float sun_scale;                        /* 1 / k^2 of --sun-disc */
    float cam_h;                            /* camera altitude above the water plane (m) */
    float max_dist;
    float g_e[SPEC_NG];                     /* W/m^2 at the top */
    float g_a[SPEC_NG], g_b[SPEC_NG];       /* vertical tau: air profile, haze */
    float g_sr[SPEC_NG], g_sh[SPEC_NG];     /* scattering at the ground, 1/m */
    float g_cmf[SPEC_NG][3];
    unsigned long long t_col, t_ms[SPEC_NG / 4], t_sv, t_ap;   /* cudaTextureObject_t */
    unsigned long long s_sv, s_ap;                             /* cudaSurfaceObject_t */
    float* sv_spec;                         /* device [SV_H][SV_W][NG]: sky-view per g-point */
    const float* gdiff;                     /* device [NG][MS_W] */
    float* sh;                              /* device [NG][9]: sky radiance in real SH (order 2) */
    float ap_fw[3], ap_rt[3], ap_up[3];     /* aerial froxels: the frame's camera basis */
    float ap_th, ap_aspect;
} AtmoParams;

/* ---- host API (spectral.c) ------------------------------------------------- */
/* Builds every spectral quantity of the planet. p_bar <= 0 means 1 bar. 0 ok.
 * Prints nothing; spectral_print shows the startup summary. */
int  spectral_build_atmo(const HomePlanet* hp, double p_bar, SpectralAtmo* a);
int  spectral_build_atmo_ex(const HomePlanet* hp, const SpectralOpts* o, SpectralAtmo* a);
void spectral_print(const SpectralAtmo* a);
void spectral_free(SpectralAtmo* a);       /* the host-side tables (not the struct) */
/* spectral_build_atmo_ex through a disk cache ($GME_CACHE, else the temp directory; GME_CACHE=0 disables it):
 * the bytes of an earlier build by this binary (the key holds its build stamp, the planet hash, the options and
 * the CH4 table). *cached = 1 when loaded from the cache. 0 ok. */
int  spectral_build_atmo_cached(const HomePlanet* hp, const SpectralOpts* o, uint64_t planet_hash, SpectralAtmo* a, int* cached);
/* D3 physics self-test; gpu_sky: GPU sky-view radiance read back by the
 * renderer for the same sun/camera (NULL = skip that comparison). Returns the
 * number of failed checks. */
typedef int (*SpectralGpuSkyFn)(void* ctx, const SpectralAtmo* a, const float sun[3], float cam_h,
                                const float* dirs, int ndir, float* out /* ndir x SPEC_NB band means */);
int  spectral_selftest_d3(const HomePlanet* hp, const SpectralOpts* o, SpectralGpuSkyFn gpu_sky, void* ctx);
/* Host twin of the sky-view raymarch (k_sky_view): radiance of direction dir
 * (world, y up) seen from altitude cam_h with the sun along sun[]: per g-point
 * (W m^-2 sr^-1, NULL to skip) and per band (band-mean W m^-2 sr^-1 nm^-1). */
void spectral_sky_radiance(const SpectralAtmo* a, const double sun[3], double cam_h, const double dir[3], double* out_g,
                           double* out_band);
/* Transmittance of g-point g on a path with the given air / haze columns
 * (vertical units), and the exact TOA-weighted band average of the 5 nm samples. */
double spectral_g_T(const SpectralAtmo* a, int g, double A, double Hz);
double spectral_band_T(const SpectralAtmo* a, int b, double A, double Hz);
/* Path columns from altitude h (m) along zenith cosine mu to the top. */
void spectral_columns(const SpectralAtmo* a, double h, double mu, double* A, double* Hz);
/* sRGB (linear) of a band spectrum under an equal-energy white, von Kries to D65 */
void spectral_rgb_of(const SpectralAtmo* a, const double* rho, double rgb[3]);

/* ---- M5 pigments and the bio's light file (spec 5.1) ---- */
/* pigment shape eps(lambda) of hue bin 1..15 (0 = SPEC_HUE_DEFAULT): Soret 435 nm, Q band at 383 + 34.13 i nm,
 * background below 690 nm (to 0 at 740) */
double spectral_pigment_eps(double lam_nm, int bin);
/* the same with a Q band of width (sigma) wq_nm (version 2: 25 .. 60 nm, SPEC_QW_NM; 25 = the function above) */
double spectral_pigment_eps_w(double lam_nm, int bin, double wq_nm);
/* leaf colour of every hue bin under an equal-energy white, von Kries (XYZ) to D65, 8-bit sRGB: ref_rgb at 1 nm
 * (the --pigment-table reference), band_rgb from the renderer's 8 band means; de: Delta E2000 between them.
 * Returns the largest |ref_rgb - spec 5.1 table| over bins 1..15 (sRGB levels). */
int  spectral_pigment_table(const SpectralAtmo* a, int ref_rgb[SPEC_NHUE][3], int band_rgb[SPEC_NHUE][3], double de[SPEC_NHUE],
                            int spec_rgb[SPEC_NHUE][3]);
/* the 1 nm leaf reflectance of hue bin `bin` at pigment `pig` (0..255) */
double spectral_leaf_rho_nm(double lam_nm, int bin, double pig);
/* GMESPEC1 (spectrum_file.h; the struct is passed as void* so this header needs no include): the light of the
 * planet for the bio, 8 haze bins x 24 hours x (8 bands + NIR) plus water, photon and pigment tables. Every
 * haze bin is an atmosphere of its own (long: ~8 builds). 0 ok. */
int  spectral_export(const HomePlanet* hp, const SpectralOpts* o, void* spectrum_file, SpectralExportInfo* info);
/* global horizontal irradiance (W/m^2, 380-740 nm: direct + diffuse, the LUT model the renderer uses) at sun
 * elevation sine mu, and its PAR (400-700 nm) part with the TOA PAR; for D5 */
void spectral_lut_global(const SpectralAtmo* a, double mu, double* glob, double* par, double* par_top);
/* D5 export self-test (two exports, file round trip, noon sums, PAR at the planet's haze, pigment table);
 * returns the number of failed checks */
int  spectral_selftest_d5(const HomePlanet* hp, const SpectralOpts* o);

/* ---- shared math (host + device) ---------------------------------------------- */
#define SPEC_PI 3.14159265358979f

SPEC_HD float spec_sat(float x) { return fminf(fmaxf(x, 0.f), 1.f); }

/* path-column LUT (Bruneton 2017): altitude h, zenith cosine mu <-> (x_mu, x_r) */
SPEC_HD void spec_col_uv(float R, float Rt, float Hl, float h, float mu, float* xmu, float* xr) {
    const float hc = fmaxf(h, 0.f), ht = Rt - R;
    const float rho = sqrtf(hc * (2.f * R + hc));
    const float r = R + hc;
    const float disc = r * r * mu * mu + (ht - hc) * (2.f * R + ht + hc);
    const float d = fmaxf(-r * mu + sqrtf(fmaxf(disc, 0.f)), 0.f);
    const float dmin = ht - hc, dmax = rho + Hl;
    *xmu = spec_sat((d - dmin) / fmaxf(dmax - dmin, 1e-3f));
    *xr = spec_sat(rho / Hl);
}
/* the zenith cosine of the geometric horizon at altitude h (rays below it hit the ground) */
SPEC_HD float spec_mu_horizon(float R, float h) {
    const float hc = fmaxf(h, 0.f);
    return -sqrtf(hc * (2.f * R + hc)) / (R + hc);
}
/* distance from altitude h along zenith cosine mu to the sphere of altitude hs
 * (outside crossing for hs >= h, first crossing for hs < h), -1 if missed */
SPEC_HD float spec_dist_sphere(float R, float h, float mu, float hs) {
    const float r = R + h;
    /* r^2 mu^2 - (r^2 - rs^2), with r^2 - rs^2 = (h - hs)(2R + h + hs) */
    const float disc = r * r * mu * mu - (h - hs) * (2.f * R + h + hs);
    if (disc < 0.f) return -1.f;
    const float sq = sqrtf(disc);
    if (hs >= h) return -r * mu + sq;
    const float t = -r * mu - sq;
    return t >= 0.f ? t : -1.f;
}
/* altitude at distance s along zenith cosine mu from altitude h (precise near the ground) */
SPEC_HD float spec_alt_at(float R, float h, float mu, float s) {
    const float r = R + h;
    const float num = h * (2.f * R + h) + 2.f * r * s * mu + s * s;
    const float rr = sqrtf(fmaxf(r * r + 2.f * r * s * mu + s * s, 0.f));
    return num / (rr + R);
}

/* sky-view LUT (Hillaire 2020): zenith angle zen in [0, pi], azimuth from the
 * sun phi in [0, pi] <-> (u, v). v = 0.5 is the geometric horizon. */
SPEC_HD float spec_sv_zh(float R, float h) {
    const float hc = fmaxf(h, 0.f);
    return 0.5f * SPEC_PI + atan2f(sqrtf(hc * (2.f * R + hc)), R);
}
SPEC_HD void spec_sv_uv(float zh, float zen, float phi, float* u, float* v) {
    if (zen < zh) *v = 0.5f * (1.f - sqrtf(spec_sat(1.f - zen / zh)));
    else          *v = 0.5f + 0.5f * sqrtf(spec_sat((zen - zh) / (SPEC_PI - zh)));
    *u = sqrtf(spec_sat(phi / SPEC_PI));
}
SPEC_HD void spec_sv_angles(float zh, float u, float v, float* zen, float* phi) {
    if (v < 0.5f) { const float c = 1.f - 2.f * v; *zen = zh * (1.f - c * c); }
    else          { const float c = 2.f * v - 1.f; *zen = zh + (SPEC_PI - zh) * c * c; }
    *phi = SPEC_PI * u * u;
}

/* multiple-scattering LUT: x = mu_s (range SPEC_MS_MU0..1), y = sqrt(h / (Rt - R)) */
SPEC_HD void spec_ms_uv(float R, float Rt, float h, float mus, float* x, float* y) {
    *x = spec_sat((mus - SPEC_MS_MU0) / (SPEC_MS_MU1 - SPEC_MS_MU0));
    *y = sqrtf(spec_sat(h / (Rt - R)));
}
/* night fade of the multiple-scattering LUT below its lowest sun cosine (the
 * LUT clamps there): e^-60 per unit of mu_s, gone ~6 deg deeper */
SPEC_HD float spec_ms_fade(float mus) { return mus >= SPEC_MS_MU0 ? 1.f : expf(-60.f * (SPEC_MS_MU0 - mus)); }
/* node coordinate -> texture coordinate for hardware filtering */
SPEC_HD float spec_tc(float x, int n) { return (x * (float)(n - 1) + 0.5f) / (float)n; }

/* phase functions (per steradian) */
SPEC_HD float spec_phase_ray(float c) { return 0.0596831f * (1.f + c * c); }     /* 3/(16 pi) */
SPEC_HD float spec_phase_hg(float c, float g) {
    const float d = 1.f + g * g - 2.f * g * c;
    return 0.0795775f * (1.f - g * g) / (d * sqrtf(d));                             /* 1/(4 pi) */
}
/* water: molecular (1 + 0.835 cos^2), normalised */
SPEC_HD float spec_phase_water_mol(float c) { return (1.f + 0.835f * c * c) / (4.f * SPEC_PI * (1.f + 0.835f / 3.f)); }

/* exact unpolarised Fresnel reflectance of a dielectric; ci = cosine on the
 * incident side, eta = n_transmitted / n_incident. 1 under total internal reflection. */
SPEC_HD float spec_fresnel(float ci, float eta) {
    const float c = fabsf(ci);
    const float g2 = eta * eta - 1.f + c * c;
    if (g2 <= 0.f) return 1.f;
    const float g = sqrtf(g2);
    const float A = (g - c) / (g + c);
    const float B = (c * (g + c) - 1.f) / (c * (g - c) + 1.f);
    return 0.5f * A * A * (1.f + B * B);
}
SPEC_HD float spec_schlick(float f0, float c) {
    const float m = 1.f - spec_sat(c), m2 = m * m;
    return f0 + (1.f - f0) * m2 * m2 * m;
}
/* GGX distribution and the height-correlated Smith visibility G2 / (4 nl nv) */
SPEC_HD float spec_ggx_d(float nh, float a) {
    const float a2 = a * a, d = nh * nh * (a2 - 1.f) + 1.f;
    return a2 / (SPEC_PI * d * d);
}
SPEC_HD float spec_ggx_vis(float nl, float nv, float a) {
    const float a2 = a * a;
    const float gv = nl * sqrtf(nv * nv * (1.f - a2) + a2);
    const float gl = nv * sqrtf(nl * nl * (1.f - a2) + a2);
    return 0.5f / fmaxf(gv + gl, 1e-7f);
}
/* refraction of direction d at a surface with unit normal n facing the
 * incident side (dot(n, d) < 0); eta = n_incident / n_transmitted. 0 = total
 * internal reflection (t untouched). */
SPEC_HD int spec_refract(const float d[3], const float n[3], float eta, float t[3]) {
    const float ci = -(d[0] * n[0] + d[1] * n[1] + d[2] * n[2]);
    const float k = 1.f - eta * eta * (1.f - ci * ci);
    if (k < 0.f) return 0;
    const float m = eta * ci - sqrtf(k);
    for (int i = 0; i < 3; ++i) t[i] = eta * d[i] + m * n[i];
    return 1;
}

/* AgX ("minimal AgX", B. Wrensch) on exposed linear sRGB -> display-linear,
 * then the sRGB OETF. Matrices as published (column-major). */
SPEC_HD void spec_agx(float* r, float* g, float* b) {
    const float x = fmaxf(*r, 1e-10f), y = fmaxf(*g, 1e-10f), z = fmaxf(*b, 1e-10f);
    float v[3] = { 0.842479062253094f * x + 0.0784335999999992f * y + 0.0792237451477643f * z,
                   0.0423282422610123f * x + 0.878468636469772f * y + 0.0791661274605434f * z,
                   0.0423756549057051f * x + 0.0784336f * y + 0.879142973793104f * z };
    const float mn = -12.47393f, mx = 4.026069f;
    for (int i = 0; i < 3; ++i) {
        float e = log2f(fmaxf(v[i], 1e-10f));
        e = (fminf(fmaxf(e, mn), mx) - mn) / (mx - mn);
        const float e2 = e * e, e4 = e2 * e2;
        v[i] = 15.5f * e4 * e2 - 40.14f * e4 * e + 31.96f * e4 - 6.868f * e2 * e + 0.4298f * e2 + 0.1191f * e - 0.00232f;
    }
    const float o0 = 1.19687900512017f * v[0] - 0.0980208811401368f * v[1] - 0.0990297440797205f * v[2];
    const float o1 = -0.0528968517574562f * v[0] + 1.15190312990417f * v[1] - 0.0989611768448433f * v[2];
    const float o2 = -0.0529716355144438f * v[0] - 0.0980434501171241f * v[1] + 1.15107367264116f * v[2];
    *r = powf(fmaxf(o0, 0.f), 2.2f);
    *g = powf(fmaxf(o1, 0.f), 2.2f);
    *b = powf(fmaxf(o2, 0.f), 2.2f);
}
SPEC_HD float spec_srgb_oetf(float c) {
    c = spec_sat(c);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.f / 2.4f) - 0.055f;
}

/* ---- device samplers ------------------------------------------------------------- */
#ifdef __CUDACC__
}   /* extern "C" */
#include <cuda_runtime.h>

__device__ __forceinline__ float4 spec_tex2(unsigned long long t, float u, float v) {
    return tex2D<float4>((cudaTextureObject_t)t, u, v);
}
/* path columns (air, haze) from altitude h along mu to the top; a ray that
 * meets the ground gets "infinite" columns (no sunlight) */
__device__ __forceinline__ float2 spec_columns(const AtmoParams& P, float h, float mu) {
    if (mu < spec_mu_horizon(P.R, h)) return make_float2(1e4f, 1e4f);
    float xmu, xr;
    spec_col_uv(P.R, P.Rt, P.Hl, h, mu, &xmu, &xr);
    const float4 c = spec_tex2(P.t_col, spec_tc(xmu, SPEC_COL_W), spec_tc(xr, SPEC_COL_H));
    return make_float2(c.x, c.y);
}
/* multiple-scattering radiance per unit TOA irradiance at (h, mu_s), all g-points */
__device__ __forceinline__ void spec_ms(const AtmoParams& P, float h, float mus, float M[SPEC_NG]) {
    float x, y;
    spec_ms_uv(P.R, P.Rt, h, mus, &x, &y);
    const float tx = spec_tc(x, SPEC_MS_W), ty = spec_tc(y, SPEC_MS_H);
#pragma unroll
    for (int k = 0; k < SPEC_NG / 4; ++k) {
        const float4 v = spec_tex2(P.t_ms[k], tx, ty);
        const float fd = spec_ms_fade(mus);
        M[4 * k] = v.x * fd; M[4 * k + 1] = v.y * fd; M[4 * k + 2] = v.z * fd; M[4 * k + 3] = v.w * fd;
    }
}
/* texture coordinates of a world direction in the sky-view LUT */
__device__ __forceinline__ void spec_sky_tc(const AtmoParams& P, float dx, float dy, float dz, float* x, float* y) {
    const float zh = spec_sv_zh(P.R, P.cam_h);
    const float zen = acosf(fminf(fmaxf(dy, -1.f), 1.f));
    const float hl = sqrtf(dx * dx + dz * dz), sl = sqrtf(P.sun[0] * P.sun[0] + P.sun[2] * P.sun[2]);
    const float c = (hl > 1e-6f && sl > 1e-6f) ? (dx * P.sun[0] + dz * P.sun[2]) / (hl * sl) : 1.f;
    const float phi = acosf(fminf(fmaxf(c, -1.f), 1.f));
    float u, v;
    spec_sv_uv(zh, zen, phi, &u, &v);
    *x = spec_tc(u, SPEC_SV_W);
    *y = spec_tc(v, SPEC_SV_H);
}
/* sky-view radiance as XYZ (no 683; scattered light only, no disc) of a world direction */
__device__ __forceinline__ float3 spec_sky_xyz(const AtmoParams& P, float dx, float dy, float dz) {
    float x, y;
    spec_sky_tc(P, dx, dy, dz, &x, &y);
    const float4 v = spec_tex2(P.t_sv, x, y);
    return make_float3(v.x, v.y, v.z);
}
/* aerial in-scatter (XYZ, no 683) of a camera ray: screen point (sx, sy) in [0,1], distance t */
__device__ __forceinline__ float3 spec_aerial_xyz(const AtmoParams& P, float sx, float sy, float t) {
    const float w = sqrtf(spec_sat(t / P.max_dist));
    const float4 v = tex3D<float4>((cudaTextureObject_t)P.t_ap, sx, sy, spec_tc(w, SPEC_AP_N));
    return make_float3(v.x, v.y, v.z);
}
/* diffuse sky irradiance at the ground per unit TOA irradiance (g-point g) */
__device__ __forceinline__ float spec_gdiff(const AtmoParams& P, int g, float mus) {
    const float x = spec_sat((mus - SPEC_MS_MU0) / (SPEC_MS_MU1 - SPEC_MS_MU0)) * (float)(SPEC_MS_W - 1);
    const int i = min((int)x, SPEC_MS_W - 2);
    const float f = x - (float)i;
    return (__ldg(P.gdiff + g * SPEC_MS_W + i) * (1.f - f) + __ldg(P.gdiff + g * SPEC_MS_W + i + 1) * f) * spec_ms_fade(mus);
}

/* ---- atmo.cu (C++ linkage, used by render.cu) ---- */
struct AtmoGPU;
AtmoGPU* atmo_create(const SpectralAtmo* a);          /* uploads the LUTs; NULL on failure */
void     atmo_destroy(AtmoGPU* g);
const AtmoParams* atmo_params(const AtmoGPU* g);
/* per frame: sky-view + SH when the sun / camera altitude moved (or force_sky),
 * aerial froxels always. -1 error, 1 sky rebuilt, 0 otherwise. */
int      atmo_frame(AtmoGPU* g, const float sun[3], float cam_h, float max_dist, const float fw[3], const float rt[3],
                    const float up[3], float th, float aspect, cudaStream_t s, int force_sky);
/* D3: band-mean radiance of the GPU sky-view at world directions (n x SPEC_NB) */
int      atmo_sky_probe(AtmoGPU* g, const float sun[3], float cam_h, const float* dirs, int n, float* out, cudaStream_t s);
extern "C" {
#endif

#ifdef __cplusplus
}
#endif
#endif

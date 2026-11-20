/* spectral.c - spectral light, host side (docs/render-upgrade-spec.md 4, M2).
 *
 * Everything here runs once per planet in double precision and fills a
 * SpectralAtmo that the renderer uploads (renderer_set_atmo):
 *   star       Planck spectrum scaled to S_top = 1361 L / a^2, disc R / a
 *   colour     CIE 1931 2-degree functions (5 nm table, 1 nm interpolation),
 *              Wyman 2013 fit as a cross-check, band integrals
 *   air        per-gas Rayleigh (Bodhaine 1999 air x refractivity^2 x King),
 *              CH4 absorption (data/ch4_k98.txt or a built-in band model),
 *              tholin haze tau ~ lambda^-2, HG g = 0.65, scale height H/2,
 *              calibrated to the bio's haze_q12 (PAR global transmission)
 *   LUTs       path columns (air, haze) to the top of the atmosphere; band
 *              optical depth vs (air column, haze column) averaged at 5 nm as
 *              transmittance (exact for saturated CH4 bands and for combined
 *              sun + view paths); multiple scattering by iterated orders
 *              (Hillaire's series as an option); diffuse ground irradiance
 *   surfaces   spectral albedo table, a smooth RGB basis (planets' colours),
 *              GGX specular albedo E(mu, alpha) for energy-compensated diffuse
 *   pigments   M5 (spec 5.1): Soret + Q band per hue bin, leaf rho / tau,
 *              mats, the --pigment-table reference at 1 nm
 *   export     M5: GMESPEC1 (spectrum_file.h), the bio's integer light file:
 *              one atmosphere per haze bin, NIR g-points, water, photons
 *   water      Pope & Fry 1997 (+ Kou 1993 beyond 727.5 nm), CDOM, particles,
 *              band transmittance vs depth
 *   D3         the physics self-test, incl. a Monte Carlo sky reference; D5
 *              the export self-test (M5). */
#include "spectral.h"
#include "jobs.h"
#include "platform.h"
#include "worldgen.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#define PI       3.14159265358979323846
#define HPL      6.62607015e-34
#define CLIGHT   2.99792458e8
#define KBOLTZ   1.380649e-23
#define SIGMA_SB 5.670374419e-8
#define NAVO     6.02214076e23
#define RGAS     8.314462618
#define LOSCH    2.6867811e25     /* amagat, m^-3 */
#define G0       9.80665
#define RSUN     6.957e8
#define AU_M     1.495978707e11
#define EVJ      1.602176634e-19

/* ---- CIE 1931 2-degree observer, 380..780 nm at 5 nm ------------------------- */
static const double CIE5[81][3] = {
    {0.001368,0.000039,0.006450},{0.002236,0.000064,0.010550},{0.004243,0.000120,0.020050},{0.007650,0.000217,0.036210},
    {0.014310,0.000396,0.067850},{0.023190,0.000640,0.110200},{0.043510,0.001210,0.207400},{0.077630,0.002180,0.371300},
    {0.134380,0.004000,0.645600},{0.214770,0.007300,1.039050},{0.283900,0.011600,1.385600},{0.328500,0.016840,1.622960},
    {0.348280,0.023000,1.747060},{0.348060,0.029800,1.782600},{0.336200,0.038000,1.772110},{0.318700,0.048000,1.744100},
    {0.290800,0.060000,1.669200},{0.251100,0.073900,1.528100},{0.195360,0.090980,1.287640},{0.142100,0.112600,1.041900},
    {0.095640,0.139020,0.812950},{0.057950,0.169300,0.616200},{0.032010,0.208020,0.465180},{0.014700,0.258600,0.353300},
    {0.004900,0.323000,0.272000},{0.002400,0.407300,0.212300},{0.009300,0.503000,0.158200},{0.029100,0.608200,0.111700},
    {0.063270,0.710000,0.078250},{0.109600,0.793200,0.057250},{0.165500,0.862000,0.042160},{0.225750,0.914850,0.029840},
    {0.290400,0.954000,0.020300},{0.359700,0.980300,0.013400},{0.433450,0.994950,0.008750},{0.512050,1.000000,0.005750},
    {0.594500,0.995000,0.003900},{0.678400,0.978600,0.002750},{0.762100,0.952000,0.002100},{0.842500,0.915400,0.001800},
    {0.916300,0.870000,0.001650},{0.978600,0.816300,0.001400},{1.026300,0.757000,0.001100},{1.056700,0.694900,0.001000},
    {1.062200,0.631000,0.000800},{1.045600,0.566800,0.000600},{1.002600,0.503000,0.000340},{0.938400,0.441200,0.000240},
    {0.854450,0.381000,0.000190},{0.751400,0.321000,0.000100},{0.642400,0.265000,0.000050},{0.541900,0.217000,0.000030},
    {0.447900,0.175000,0.000020},{0.360800,0.138200,0.000010},{0.283500,0.107000,0.000000},{0.218700,0.081600,0.000000},
    {0.164900,0.061000,0.000000},{0.121200,0.044580,0.000000},{0.087400,0.032000,0.000000},{0.063600,0.023200,0.000000},
    {0.046770,0.017000,0.000000},{0.032900,0.011920,0.000000},{0.022700,0.008210,0.000000},{0.015840,0.005723,0.000000},
    {0.011359,0.004102,0.000000},{0.008111,0.002929,0.000000},{0.005790,0.002091,0.000000},{0.004109,0.001484,0.000000},
    {0.002899,0.001047,0.000000},{0.002049,0.000740,0.000000},{0.001440,0.000520,0.000000},{0.001000,0.000361,0.000000},
    {0.000690,0.000249,0.000000},{0.000476,0.000172,0.000000},{0.000332,0.000120,0.000000},{0.000235,0.000085,0.000000},
    {0.000166,0.000060,0.000000},{0.000117,0.000042,0.000000},{0.000083,0.000030,0.000000},{0.000059,0.000021,0.000000},
    {0.000042,0.000015,0.000000}};

static void cie_at(double l, double xyz[3]) {
    double f = (l - 380.0) / 5.0;
    if (f < 0 || f > 80) { xyz[0] = xyz[1] = xyz[2] = 0; return; }
    int i = (int)f;
    if (i > 79) i = 79;
    f -= i;
    for (int c = 0; c < 3; ++c) xyz[c] = CIE5[i][c] * (1 - f) + CIE5[i + 1][c] * f;
}
static double wy_g(double x, double mu, double s1, double s2) {
    const double t = (x - mu) / (x < mu ? s1 : s2);
    return exp(-0.5 * t * t);
}
static void wyman_at(double l, double xyz[3]) {   /* Wyman, Sloan, Shirley 2013, multi-lobe fit */
    xyz[0] = 1.056 * wy_g(l, 599.8, 37.9, 31.0) + 0.362 * wy_g(l, 442.0, 16.0, 26.7) - 0.065 * wy_g(l, 501.1, 20.4, 26.2);
    xyz[1] = 0.821 * wy_g(l, 568.8, 46.9, 40.5) + 0.286 * wy_g(l, 530.9, 16.3, 31.1);
    xyz[2] = 1.217 * wy_g(l, 437.0, 11.8, 36.0) + 0.681 * wy_g(l, 459.0, 26.0, 13.8);
}

/* XYZ (any scale) -> linear sRGB (D65) */
static const double XYZ2RGB[3][3] = { { 3.2406, -1.5372, -0.4986 }, { -0.9689, 1.8758, 0.0415 }, { 0.0557, -0.2040, 1.0570 } };
static const double M16[3][3] = { { 0.401288, 0.650173, -0.051461 }, { -0.250268, 1.204414, 0.045854 }, { -0.002079, 0.048952, 0.953127 } };
static const double D65[3] = { 0.95047, 1.0, 1.08883 };

static void mat3_mul_v(const double m[3][3], const double v[3], double o[3]) {
    for (int i = 0; i < 3; ++i) o[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
}
static int mat3_inv(const double m[3][3], double o[3][3]) {
    const double d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                     m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if (fabs(d) < 1e-300) return -1;
    o[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d; o[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d;
    o[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d; o[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d;
    o[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d; o[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d;
    o[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d; o[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d;
    o[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;
    return 0;
}
/* CAT16 (degree D) from white XYZ w to D65, as a 3x3 on XYZ */
static void cat16(const double w[3], double D, double out[3][3]) {
    double rw[3], rr[3], inv[3][3], g[3];
    const double wn[3] = { w[0] / w[1], 1.0, w[2] / w[1] };
    mat3_mul_v(M16, wn, rw);
    mat3_mul_v(M16, D65, rr);
    for (int i = 0; i < 3; ++i) g[i] = D * rr[i] / rw[i] + 1.0 - D;
    mat3_inv(M16, inv);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += inv[i][k] * g[k] * M16[k][j];
            out[i][j] = s;
        }
}

/* ---- star --------------------------------------------------------------------- */
static double planck_nm(double lam, double T) {       /* W m^-2 sr^-1 nm^-1 */
    const double l = lam * 1e-9;
    const double x = HPL * CLIGHT / (l * KBOLTZ * T);
    if (x > 700) return 0;
    return 2.0 * HPL * CLIGHT * CLIGHT / (l * l * l * l * l) / expm1(x) * 1e-9;
}
static double etop_nm(const SpectralAtmo* a, double lam) {   /* W m^-2 nm^-1 at the top */
    return a->s_top * PI * planck_nm(lam, a->teff) / (SIGMA_SB * pow(a->teff, 4.0));
}

/* ---- gases ------------------------------------------------------------------------ */
/* Bodhaine et al. (1999) eq. 29: Rayleigh cross-section of air, m^2 */
static double sigma_air(double lam) {
    const double l = lam * 1e-3, l2 = l * l, il2 = 1.0 / l2;
    return (1.0455996 - 341.29061 * il2 - 0.90230850 * l2) / (1.0 + 0.0027059889 * il2 - 85.968563 * l2) * 1e-32;
}
static const struct { const char* name; double M, n1, fk; } GAS[ATM_N + 1] = {
    { "H2S", 34.081, 641, 1.0 }, { "CH4", 16.043, 444, 1.0 }, { "NH3", 17.031, 376, 1.0 }, { "CO2", 44.009, 449, 1.15 },
    { "O2", 31.998, 271, 1.096 }, { "N2", 28.014, 298, 1.034 }, { "H2", 2.016, 132, 1.0 }, { "H2O", 18.015, 256, 1.0 } };

/* CH4 absorption, km-amagat^-1. Built-in coarse band model (Gaussian bands at
 * the positions of Karkoschka's room-temperature spectrum; peak and width are
 * rounded estimates): used when data/ch4_k98.txt is missing. */
static const double CH4_BANDS[][3] = {   /* centre nm, peak, FWHM nm */
    { 543, 0.023, 10 }, { 576, 0.040, 14 }, { 598, 0.034, 12 }, { 619, 0.45, 12 }, { 667, 0.15, 12 },
    { 702, 0.35, 10 },  { 727, 2.8, 14 },   { 790, 1.6, 20 },   { 840, 0.6, 20 },  { 862, 3.0, 12 }, { 887, 11.0, 18 } };
static double ch4_model(double lam) {
    double k = 0;
    for (size_t i = 0; i < sizeof CH4_BANDS / sizeof CH4_BANDS[0]; ++i) {
        const double s = CH4_BANDS[i][2] / 2.35482, d = (lam - CH4_BANDS[i][0]) / s;
        k += CH4_BANDS[i][1] * exp(-0.5 * d * d);
    }
    return k;
}
static int    g_ch4_n = -1;
static double g_ch4_l[2048], g_ch4_k[2048];
static int ch4_load(void) {                   /* data/ch4_k98.txt: "lambda_nm k" per line, '#' comments */
    if (g_ch4_n >= 0) return g_ch4_n > 0;
    g_ch4_n = 0;
    FILE* f = fopen("data/ch4_k98.txt", "r");
    if (!f) return 0;
    char line[256];
    while (fgets(line, sizeof line, f) && g_ch4_n < 2048) {
        double l, k;
        if (line[0] == '#') continue;
        if (sscanf(line, "%lf %lf", &l, &k) == 2 && (g_ch4_n == 0 || l > g_ch4_l[g_ch4_n - 1])) {
            g_ch4_l[g_ch4_n] = l; g_ch4_k[g_ch4_n] = k; ++g_ch4_n;
        }
    }
    fclose(f);
    return g_ch4_n > 1;
}
static double ch4_k(double lam) {
    if (g_ch4_n > 1) {
        if (lam <= g_ch4_l[0]) return g_ch4_k[0];
        if (lam >= g_ch4_l[g_ch4_n - 1]) return g_ch4_k[g_ch4_n - 1];
        int lo = 0, hi = g_ch4_n - 1;
        while (hi - lo > 1) { const int m = (lo + hi) / 2; if (g_ch4_l[m] <= lam) lo = m; else hi = m; }
        const double f = (lam - g_ch4_l[lo]) / (g_ch4_l[hi] - g_ch4_l[lo]);
        return g_ch4_k[lo] * (1 - f) + g_ch4_k[hi] * f;
    }
    return ch4_model(lam);
}

/* ---- water (Pope & Fry 1997, 380-727.5 nm at 2.5 nm; Kou 1993 beyond) ------------- */
static const double PF97[] = {
    0.01137, 0.01044, 0.00941, 0.00917, 0.00851, 0.00829, 0.00813, 0.00775, 0.00663, 0.00579, 0.00530, 0.00503, 0.00473,
    0.00452, 0.00444, 0.00442, 0.00454, 0.00474, 0.00478, 0.00482, 0.00495, 0.00504, 0.00530, 0.00580, 0.00635, 0.00696,
    0.00751, 0.00830, 0.00922, 0.00969, 0.00962, 0.00957, 0.00979, 0.01005, 0.01011, 0.0102, 0.0106, 0.0109, 0.0114,
    0.0121, 0.0127, 0.0131, 0.0136, 0.0144, 0.0150, 0.0162, 0.0173, 0.0191, 0.0204, 0.0228, 0.0256, 0.0280, 0.0325,
    0.0372, 0.0396, 0.0399, 0.0409, 0.0416, 0.0417, 0.0428, 0.0434, 0.0447, 0.0452, 0.0466, 0.0474, 0.0489, 0.0511,
    0.0537, 0.0565, 0.0593, 0.0596, 0.0606, 0.0619, 0.0640, 0.0642, 0.0672, 0.0695, 0.0733, 0.0772, 0.0836, 0.0896,
    0.0989, 0.1110, 0.1220, 0.1351, 0.1516, 0.1672, 0.1925, 0.2224, 0.2470, 0.2577, 0.2629, 0.2644, 0.2665, 0.2678,
    0.2707, 0.2755, 0.2810, 0.2834, 0.2904, 0.2916, 0.2995, 0.3012, 0.3077, 0.3108, 0.3220, 0.3250, 0.3350, 0.3400,
    0.3580, 0.3710, 0.3930, 0.4100, 0.4240, 0.4290, 0.4340, 0.4390, 0.4480, 0.4480, 0.4610, 0.4650, 0.4780, 0.4860,
    0.5020, 0.5160, 0.5380, 0.5590, 0.5920, 0.6240, 0.6630, 0.7040, 0.7560, 0.8270, 0.9140, 1.0070, 1.1190, 1.2310,
    1.3560, 1.4890, 1.4690 };
static const double KOU93[][2] = {   /* nm, 1/m (from the imaginary index of Kou, Labrie & Chylek 1993; rounded) */
    { 730, 1.90 }, { 735, 2.23 }, { 740, 2.47 }, { 745, 2.60 }, { 750, 2.62 }, { 760, 2.55 }, { 770, 2.45 }, { 780, 2.33 },
    { 790, 2.15 }, { 800, 2.00 }, { 810, 2.10 }, { 820, 2.50 }, { 830, 2.80 }, { 840, 3.20 }, { 850, 4.10 }, { 860, 4.60 },
    { 870, 5.00 }, { 880, 5.40 }, { 890, 5.90 }, { 900, 6.60 } };
static double water_aw(double lam) {
    const int npf = (int)(sizeof PF97 / sizeof PF97[0]);   /* 380 .. 727.5 */
    if (lam <= 380) return PF97[0];
    const double lpf = 380 + 2.5 * (npf - 1);
    if (lam <= lpf) {
        const double f = (lam - 380) / 2.5;
        int i = (int)f;
        if (i >= npf - 1) i = npf - 2;
        const double t = f - i;
        return PF97[i] * (1 - t) + PF97[i + 1] * t;
    }
    if (lam <= KOU93[0][0]) {
        const double t = (lam - lpf) / (KOU93[0][0] - lpf);
        return PF97[npf - 1] * (1 - t) + KOU93[0][1] * t;
    }
    const int nk = (int)(sizeof KOU93 / sizeof KOU93[0]);
    for (int i = 0; i + 1 < nk; ++i)
        if (lam <= KOU93[i + 1][0]) {
            const double t = (lam - KOU93[i][0]) / (KOU93[i + 1][0] - KOU93[i][0]);
            return KOU93[i][1] * (1 - t) + KOU93[i + 1][1] * t;
        }
    return KOU93[nk - 1][1];
}
static double water_a(double lam) { return water_aw(lam) + 0.05 * exp(-0.014 * (lam - 440)) + 0.02; }
static double water_bmol(double lam) { return 0.0038 * pow(400.0 / lam, 4.32); }
static double water_b(double lam) { return water_bmol(lam) + 0.02 * (550.0 / lam); }

/* ---- small helpers ------------------------------------------------------------------ */
static double clampd(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }
static double logmean_int(double fa, double fb, double ds) {      /* exact for exponential f */
    if (fa <= 0 || fb <= 0) return 0.5 * (fa + fb) * ds;
    const double r = fa / fb;
    if (fabs(r - 1) < 1e-6) return 0.5 * (fa + fb) * ds;
    return ds * (fa - fb) / log(r);
}
static void gauss_legendre(int n, double* x, double* w) {
    for (int i = 0; i < n; ++i) {
        double z = cos(PI * (i + 0.75) / (n + 0.5)), pp = 0;
        for (int it = 0; it < 100; ++it) {
            double p1 = 1, p2 = 0;
            for (int j = 1; j <= n; ++j) { const double p3 = p2; p2 = p1; p1 = ((2.0 * j - 1) * z * p2 - (j - 1.0) * p3) / j; }
            pp = n * (z * p1 - p2) / (z * z - 1);
            const double z1 = z;
            z = z1 - p1 / pp;
            if (fabs(z - z1) < 1e-15) break;
        }
        x[i] = -z;
        w[i] = 2.0 / ((1 - z * z) * pp * pp);
    }
}
static double phase_ray(double c) { return 3.0 / (16.0 * PI) * (1 + c * c); }
static double phase_hg(double c, double g) {
    const double d = 1 + g * g - 2 * g * c;
    return (1 - g * g) / (4 * PI * d * sqrt(d));
}

/* ---- geometry (double) -------------------------------------------------------------- */
static double dist_top(const SpectralAtmo* a, double h, double mu) {
    const double r = a->R + h, ht = a->top - a->R;
    const double disc = r * r * mu * mu + (ht - h) * (2 * a->R + ht + h);
    return fmax(-r * mu + sqrt(fmax(disc, 0)), 0);
}
static double dist_ground(const SpectralAtmo* a, double h, double mu) {   /* -1 = no hit */
    if (mu >= 0) return -1;
    const double r = a->R + h, disc = r * r * mu * mu - h * (2 * a->R + h);
    if (disc < 0) return -1;
    return fmax(-r * mu - sqrt(disc), 0);
}
static double mu_hor(const SpectralAtmo* a, double h) {
    const double hc = fmax(h, 0);
    return -sqrt(hc * (2 * a->R + hc)) / (a->R + hc);
}

/* ---- host tables: double copies for lookups -------------------------------------------- */
typedef struct HostTabs {
    double col[SPEC_COL_H][SPEC_COL_W][2];
} HostTabs;
#define TABS(a) ((HostTabs*)(a)->host)

static void col_node(const SpectralAtmo* a, int i, int j, double* h, double* mu) {
    const double xmu = i / (double)(SPEC_COL_W - 1), xr = j / (double)(SPEC_COL_H - 1);
    const double R = a->R, ht = a->top - R, Hl = sqrt(ht * (2 * R + ht));
    const double rho = Hl * xr, r = sqrt(rho * rho + R * R);
    *h = rho * rho / (r + R);
    const double dmin = ht - *h, dmax = rho + Hl, d = dmin + xmu * (dmax - dmin);
    *mu = d < 1e-6 ? 1.0 : clampd((Hl * Hl - rho * rho - d * d) / (2 * r * d), -1, 1);
}
static void col_uv(const SpectralAtmo* a, double h, double mu, double* xmu, double* xr) {
    const double R = a->R, ht = a->top - R, Hl = sqrt(ht * (2 * R + ht));
    const double hc = fmax(h, 0), rho = sqrt(hc * (2 * R + hc)), r = R + hc;
    const double disc = r * r * mu * mu + (ht - hc) * (2 * R + ht + hc);
    const double d = fmax(-r * mu + sqrt(fmax(disc, 0)), 0);
    const double dmin = ht - hc, dmax = rho + Hl;
    *xmu = clampd((d - dmin) / fmax(dmax - dmin, 1e-3), 0, 1);
    *xr = clampd(rho / Hl, 0, 1);
}
/* exact path columns by Simpson integration (vertical-column units) */
static void columns_exact(const SpectralAtmo* a, double h, double mu, double* A, double* Hz) {
    const double d = dist_top(a, h, mu), r = a->R + h;
    const int n = 1024;
    double sa = 0, sh = 0;
    for (int k = 0; k <= n; ++k) {
        const double s = d * k / n;
        const double hh = sqrt(r * r + 2 * r * mu * s + s * s) - a->R;
        const double wgt = (k == 0 || k == n) ? 1 : ((k & 1) ? 4 : 2);
        sa += wgt * exp(-hh / a->H);
        sh += wgt * exp(-hh / a->haze_H);
    }
    *A = sa * d / (3.0 * n) / a->H;
    *Hz = sh * d / (3.0 * n) / a->haze_H;
}
void spectral_columns(const SpectralAtmo* a, double h, double mu, double* A, double* Hz) {
    if (mu < mu_hor(a, h)) { *A = *Hz = 1e4; return; }
    double xmu, xr;
    col_uv(a, h, mu, &xmu, &xr);
    const double fx = xmu * (SPEC_COL_W - 1), fy = xr * (SPEC_COL_H - 1);
    int i = (int)fx, j = (int)fy;
    if (i > SPEC_COL_W - 2) i = SPEC_COL_W - 2;
    if (j > SPEC_COL_H - 2) j = SPEC_COL_H - 2;
    const double u = fx - i, v = fy - j;
    for (int c = 0; c < 2; ++c) {
        const double val = (TABS(a)->col[j][i][c] * (1 - u) + TABS(a)->col[j][i + 1][c] * u) * (1 - v) +
                           (TABS(a)->col[j + 1][i][c] * (1 - u) + TABS(a)->col[j + 1][i + 1][c] * u) * v;
        if (c == 0) *A = val; else *Hz = val;
    }
}
double spectral_band_T(const SpectralAtmo* a, int b, double A, double Hz) {   /* exact, 5 nm */
    if (A >= 1e3) return 0;
    double t = 0, ws = 0;
    for (int k = b * SPEC_SUB; k < (b + 1) * SPEC_SUB; ++k) {
        t += a->e_fine[k] * exp(-A * a->a_fine[k] - Hz * a->b_fine[k]);
        ws += a->e_fine[k];
    }
    return t / ws;
}
double spectral_g_T(const SpectralAtmo* a, int g, double A, double Hz) {      /* the grey g-point */
    if (A >= 1e3) return 0;
    return exp(-A * a->g_a[g] - Hz * a->g_b[g]);
}
#define GT(a, g, A, Hz) spectral_g_T(a, g, A, Hz)

/* ---- per-planet spectral setup ------------------------------------------------------------- */
/* g-point groups of band b: the 9 fine samples sorted by vertical air-profile
 * extinction, split into SPEC_GPB groups at the natural breaks of its log
 * (smallest total within-group variance, by dynamic programming). Depends on Rayleigh + CH4 only, so
 * the groups stay fixed while the haze is calibrated. */
static void setup_groups(SpectralAtmo* a) {
    for (int b = 0; b < SPEC_NB; ++b) {
        int idx[SPEC_SUB];
        double v[SPEC_SUB];
        for (int i = 0; i < SPEC_SUB; ++i) idx[i] = b * SPEC_SUB + i;
        for (int i = 1; i < SPEC_SUB; ++i)                  /* insertion sort by a_fine */
            for (int j = i; j > 0 && a->a_fine[idx[j]] < a->a_fine[idx[j - 1]]; --j) { const int t = idx[j]; idx[j] = idx[j - 1]; idx[j - 1] = t; }
        double ew[SPEC_SUB], es = 0;
        for (int i = 0; i < SPEC_SUB; ++i) { v[i] = a->a_fine[idx[i]]; es += a->e_fine[idx[i]]; }
        for (int i = 0; i < SPEC_SUB; ++i) ew[i] = a->e_fine[idx[i]] * SPEC_SUB / es;
        /* optimal 1-D clustering (dynamic programming) of the sorted samples: the
         * light-weighted relative spread of transmittance inside each group,
         * sum w (T - m)^2 / m, at 1, 3, 10 and 30 air masses, each relative to the
         * band's light at that air mass (sunset and horizon light travels through
         * the transparent samples), plus a small term that keeps nearly grey
         * bands balanced */
        static const double AM[4] = { 1.0, 3.0, 10.0, 30.0 };
        double tot[4];                       /* the band's light at each air mass: every air mass counts alike */
        for (int m = 0; m < 4; ++m) { tot[m] = 0; for (int t = 0; t < SPEC_SUB; ++t) tot[m] += ew[t] * exp(-v[t] * AM[m]); }
        double D[SPEC_GPB + 1][SPEC_SUB + 1];
        int from[SPEC_GPB + 1][SPEC_SUB + 1];
        for (int k = 0; k <= SPEC_GPB; ++k) for (int j = 0; j <= SPEC_SUB; ++j) { D[k][j] = 1e300; from[k][j] = 0; }
        D[0][0] = 0;
        const double nbal = (double)SPEC_SUB / SPEC_GPB;
        for (int k = 1; k <= SPEC_GPB; ++k)
            for (int j = k; j <= SPEC_SUB; ++j)
                for (int i = k - 1; i < j; ++i) {
                    if (D[k - 1][i] >= 1e299) continue;
                    double c = 1e-6 * (j - i - nbal) * (j - i - nbal);
                    for (int m = 0; m < 4; ++m) {
                        double mt = 0, wt = 0;
                        for (int t = i; t < j; ++t) { mt += ew[t] * exp(-v[t] * AM[m]); wt += ew[t]; }
                        mt /= wt;
                        if (mt < 1e-30) continue;
                        for (int t = i; t < j; ++t) { const double d = exp(-v[t] * AM[m]) - mt; c += ew[t] * d * d / (mt * tot[m]); }
                    }
                    if (D[k - 1][i] + c < D[k][j]) { D[k][j] = D[k - 1][i] + c; from[k][j] = i; }
                }
        int cut[SPEC_GPB + 1];
        cut[SPEC_GPB] = SPEC_SUB;
        for (int k = SPEC_GPB; k > 0; --k) cut[k - 1] = from[k][cut[k]];
        for (int g = 0; g < SPEC_GPB; ++g) {
            const int gg = b * SPEC_GPB + g;
            for (int m = 0; m < SPEC_SUB; ++m) a->g_member[gg][m] = -1;
            a->g_nmem[gg] = cut[g + 1] - cut[g];
            for (int i = cut[g], m = 0; i < cut[g + 1]; ++i, ++m) a->g_member[gg][m] = idx[i];
        }
    }
}
static void setup_fine(SpectralAtmo* a) {
    for (int k = 0; k < SPEC_NF; ++k) {
        const double l = SPEC_L0 + 2.5 + 5.0 * k;
        a->lam[k] = l;
        a->e_fine[k] = etop_nm(a, l);
        a->r_fine[k] = a->N * sigma_air(l) * a->ray_factor;
        a->c_fine[k] = ch4_k(l) * a->ch4_col;
        a->b_fine[k] = a->haze_tau550 * pow(l / 550.0, -2.0);
        a->w_fine[k] = 0.95 - 0.45 * exp(-(l - 380.0) / 90.0);
        a->a_fine[k] = a->r_fine[k] + a->c_fine[k];
        for (int c = 0; c < 3; ++c) a->cmf_fine[k][c] = 0;
        for (int i = 0; i < 5; ++i) {
            double x[3];
            cie_at(l - 2.0 + i, x);
            for (int c = 0; c < 3; ++c) a->cmf_fine[k][c] += x[c] / 5.0;
        }
    }
    if (!a->groups_set) { setup_groups(a); a->groups_set = 1; }
    for (int b = 0; b < SPEC_NB; ++b) {
        double ws = 0, tr = 0, tc = 0, th = 0, ths = 0, tt = 0, srs = 0, shs = 0;
        for (int k = b * SPEC_SUB; k < (b + 1) * SPEC_SUB; ++k) {
            const double w = a->e_fine[k];
            ws += w;
            tr += w * a->r_fine[k];
            tc += w * exp(-a->c_fine[k]);
            th += w * a->b_fine[k];
            ths += w * a->b_fine[k] * a->w_fine[k];
            tt += w * exp(-a->a_fine[k] - a->b_fine[k]);
            srs += w * a->r_fine[k];
            shs += w * a->w_fine[k] * a->b_fine[k];
        }
        a->tau_ray[b] = tr / ws;
        a->tau_ch4[b] = fmax(-log(tc / ws), 0.0);
        a->tau_haze[b] = th / ws;
        a->omega_haze[b] = th > 0 ? ths / th : 0;
        a->tau_total[b] = -log(tt / ws);
        a->sig_r[b] = srs / ws / a->H;
        a->sig_hs[b] = shs / ws / a->haze_H;
    }
    /* g-points: grey groups. The air-profile tau is the effective value at two
     * air masses (exact transmittance there); the rest are TOA-weighted means of
     * the members. */
    for (int g = 0; g < SPEC_NG; ++g) {
        double ws = 0, ta = 0, tb = 0, sr = 0, sh = 0, cm[3] = { 0, 0, 0 }, par = 0;
        for (int m = 0; m < a->g_nmem[g]; ++m) {
            const int k = a->g_member[g][m];
            const double w = a->e_fine[k];
            ws += w;
            ta += w * exp(-2.0 * a->a_fine[k]);
            tb += w * a->b_fine[k];
            sr += w * a->r_fine[k];
            sh += w * a->w_fine[k] * a->b_fine[k];
            for (int c = 0; c < 3; ++c) cm[c] += w * a->cmf_fine[k][c];
            if (a->lam[k] > 400 && a->lam[k] < 700) par += w * 5.0;
        }
        a->g_e[g] = ws * 5.0;
        a->g_a[g] = -0.5 * log(ta / ws);       /* effective at 2 air masses */
        a->g_b[g] = tb / ws;
        a->g_sr[g] = sr / ws / a->H;
        a->g_sh[g] = sh / ws / a->haze_H;
        for (int c = 0; c < 3; ++c) a->g_cmf[g][c] = cm[c] / ws;
        a->g_par[g] = par;
    }
}

typedef struct ColCtx { SpectralAtmo* a; } ColCtx;
static void col_row_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    SpectralAtmo* a = ((ColCtx*)p)->a;
    for (uint32_t j = b; j < e; ++j)
        for (int i = 0; i < SPEC_COL_W; ++i) {
            double h, mu, A, Hz;
            col_node(a, i, (int)j, &h, &mu);
            columns_exact(a, h, mu, &A, &Hz);
            TABS(a)->col[j][i][0] = A; TABS(a)->col[j][i][1] = Hz;
            a->col[j][i][0] = (float)A; a->col[j][i][1] = (float)Hz;
        }
}
static void build_columns(SpectralAtmo* a) {
    ColCtx c = { a };
    jobs_parallel_for(SPEC_COL_H, 1, col_row_fn, &c);
}

/* ---- multiple scattering: iterated orders (or Hillaire's series) ------------------------------ */
#define ORD_NSTEP 24
typedef struct OrdCtx {
    const SpectralAtmo* a;
    int g, order, hillaire, j0;
    const double* J;          /* [MS_H][MS_W] isotropic source of this order (order >= 2) */
    const double* G;          /* [MS_W] ground radiance source of this order (order >= 2) */
    double* Jn;               /* out: source of the next order */
    double* En;               /* out: ground irradiance of this order [MS_W] (row 0) */
    double* F;                /* Hillaire: f_ms [MS_H][MS_W] */
    int nmu_s, nmu_g, nphi;
} OrdCtx;

static double ms_h(const SpectralAtmo* a, int i) { const double y = i / (double)(SPEC_MS_H - 1); return (a->top - a->R) * y * y; }
static double ms_mu(int j) { return SPEC_MS_MU0 + (SPEC_MS_MU1 - SPEC_MS_MU0) * j / (double)(SPEC_MS_W - 1); }
static double tab2_lookup(const SpectralAtmo* a, const double* t, int j0, double h, double mus) {
    const double fy = sqrt(clampd(h / (a->top - a->R), 0, 1)) * (SPEC_MS_H - 1);
    double fx = clampd((mus - SPEC_MS_MU0) / (SPEC_MS_MU1 - SPEC_MS_MU0), 0, 1) * (SPEC_MS_W - 1);
    if (fx < j0) fx = j0;
    int i = (int)fy, j = (int)fx;
    if (i > SPEC_MS_H - 2) i = SPEC_MS_H - 2;
    if (j > SPEC_MS_W - 2) j = SPEC_MS_W - 2;
    const double v = fy - i, u = fx - j;
    return (t[i * SPEC_MS_W + j] * (1 - u) + t[i * SPEC_MS_W + j + 1] * u) * (1 - v) +
           (t[(i + 1) * SPEC_MS_W + j] * (1 - u) + t[(i + 1) * SPEC_MS_W + j + 1] * u) * v;
}
static double tab1_lookup(const double* t, int j0, double mus) {
    double fx = clampd((mus - SPEC_MS_MU0) / (SPEC_MS_MU1 - SPEC_MS_MU0), 0, 1) * (SPEC_MS_W - 1);
    if (fx < j0) fx = j0;
    int j = (int)fx;
    if (j > SPEC_MS_W - 2) j = SPEC_MS_W - 2;
    const double u = fx - j;
    return t[j] * (1 - u) + t[j + 1] * u;
}

/* radiance of one direction (local frame: point at altitude h on +z, sun in the xz plane) */
static void ord_ray(const OrdCtx* c, double h, const double sun[3], const double w[3], int ground, double* L, double* Fms) {
    const SpectralAtmo* a = c->a;
    const int g = c->g;
    const double r = a->R + h, mu = w[2];
    const double send = ground ? dist_ground(a, h, mu) : dist_top(a, h, mu);
    *L = 0; *Fms = 0;
    if (send <= 0) return;
    const double smin = 100.0, q = pow(1 + send / smin, 1.0 / ORD_NSTEP);
    const double cs = w[0] * sun[0] + w[1] * sun[1] + w[2] * sun[2];
    const double pr = c->hillaire ? 1.0 / (4 * PI) : phase_ray(cs), ph = c->hillaire ? 1.0 / (4 * PI) : phase_hg(cs, a->haze_g);
    double Av = 0, Hv = 0, sprev = 0, fprev = 0, gprev = 0, dprev_a = 0, dprev_h = 0, qk = 1;
    for (int k = 0; k <= ORD_NSTEP; ++k) {
        const double s = k == ORD_NSTEP ? send : smin * (qk - 1);
        qk *= q;
        const double px = s * w[0], py = s * w[1], pz = r + s * w[2];
        const double rr = sqrt(px * px + py * py + pz * pz), hh = fmax(rr - a->R, 0);
        const double da = exp(-hh / a->H), dh = exp(-hh / a->haze_H);
        if (k > 0) {
            Av += logmean_int(dprev_a, da, s - sprev) / a->H;
            Hv += logmean_int(dprev_h, dh, s - sprev) / a->haze_H;
        }
        double f;
        const double ss = a->g_sr[g] * da + a->g_sh[g] * dh;
        if (c->order == 1) {
            const double mus = (px * sun[0] + py * sun[1] + pz * sun[2]) / rr;
            double As, Hs;
            spectral_columns(a, hh, mus, &As, &Hs);
            f = (a->g_sr[g] * da * pr + a->g_sh[g] * dh * ph) * GT(a, g, Av + As, Hv + Hs);
        } else {
            const double mus = (px * sun[0] + py * sun[1] + pz * sun[2]) / rr;
            f = tab2_lookup(a, c->J, c->j0, hh, mus) * GT(a, g, Av, Hv);
        }
        const double gfm = c->hillaire ? ss / (4 * PI) * GT(a, g, Av, Hv) : 0;
        if (k > 0) {
            *L += logmean_int(fprev, f, s - sprev);
            *Fms += logmean_int(gprev, gfm, s - sprev);
        }
        sprev = s; fprev = f; gprev = gfm; dprev_a = da; dprev_h = dh;
    }
    if (ground) {   /* the ground at the end of the ray: Lambertian, albedo 0.15 */
        const double px = send * w[0], py = send * w[1], pz = r + send * w[2];
        const double rr = sqrt(px * px + py * py + pz * pz);
        const double mus = (px * sun[0] + py * sun[1] + pz * sun[2]) / rr;
        if (c->order == 1) {
            double As, Hs;
            spectral_columns(a, 0, mus, &As, &Hs);
            *L += GT(a, g, Av + As, Hv + Hs) * SPEC_GROUND_ALBEDO / PI * fmax(mus, 0);
        } else {
            *L += GT(a, g, Av, Hv) * tab1_lookup(c->G, c->j0, mus);
        }
    }
}

static void ord_point(const OrdCtx* c, int i, int j) {
    const SpectralAtmo* a = c->a;
    const double h = ms_h(a, i), mus = ms_mu(j);
    const double sun[3] = { sqrt(fmax(0, 1 - mus * mus)), 0, mus };
    const double mh = mu_hor(a, h);
    double gx[32], gw[32];
    double Ls = 0, Es = 0, Fs = 0;
    for (int seg = 0; seg < 2; ++seg) {      /* 0: ground cone [-1, mh], 1: sky [mh, 1] */
        const int n = seg ? c->nmu_s : c->nmu_g;
        const double lo = seg ? mh : -1.0, hi = seg ? 1.0 : mh;
        if (hi - lo < 1e-9) continue;
        gauss_legendre(n, gx, gw);
        for (int m = 0; m < n; ++m) {
            const double mu = 0.5 * (hi - lo) * gx[m] + 0.5 * (hi + lo), wm = 0.5 * (hi - lo) * gw[m];
            const double st = sqrt(fmax(0, 1 - mu * mu));
            for (int p = 0; p < c->nphi; ++p) {
                const double phi = PI * (p + 0.5) / c->nphi;
                const double w[3] = { st * cos(phi), st * sin(phi), mu };
                double L, F;
                ord_ray(c, h, sun, w, !seg, &L, &F);
                const double wt = wm * 2.0 * PI / c->nphi;   /* phi in [0, pi], doubled by symmetry */
                Ls += wt * L;
                Fs += wt * F;
                if (mu > 0) Es += wt * L * mu;
            }
        }
    }
    const double ss = a->g_sr[c->g] * exp(-h / a->H) + a->g_sh[c->g] * exp(-h / a->haze_H);
    c->Jn[i * SPEC_MS_W + j] = c->hillaire ? Ls / (4 * PI) : ss / (4 * PI) * Ls;
    if (c->F) c->F[i * SPEC_MS_W + j] = Fs;
    if (i == 0 && c->En) c->En[j] = Es;
}
static void ord_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    const OrdCtx* c = (const OrdCtx*)p;
    const int ncol = SPEC_MS_W - c->j0;
    for (uint32_t k = b; k < e; ++k) ord_point(c, (int)(k / (uint32_t)ncol), c->j0 + (int)(k % (uint32_t)ncol));
}

/* Multiple scattering of g-point g into ms (radiance per unit TOA irradiance,
 * divided by the local scattering coefficient: the GPU multiplies by sigma_s)
 * and the ground's diffuse irradiance gd. Columns j < j0 are not computed
 * (haze calibration). Order 1 resolves the haze aureole with more directions;
 * the smoother higher orders use fewer. Orders are iterated until the global
 * ratio of successive orders (the dominant eigenvalue of the scattering
 * operator) has settled; the rest is its geometric tail. */
#define ORD_MIN 5
#define ORD_MAX 16
static void build_ms_g(const SpectralAtmo* a, int g, int method, int j0, double* ms, double* gd, double* qout) {
    const size_t nt = SPEC_MS_H * SPEC_MS_W;
    double* J[ORD_MAX + 2];
    double E[ORD_MAX + 1][SPEC_MS_W], G[SPEC_MS_W];
    for (int k = 0; k <= ORD_MAX + 1; ++k) J[k] = (double*)calloc(nt, sizeof(double));
    memset(E, 0, sizeof E);
    memset(G, 0, sizeof G);
    OrdCtx c;
    memset(&c, 0, sizeof c);
    c.a = a; c.g = g; c.j0 = j0;
    c.nmu_s = 20; c.nmu_g = 10; c.nphi = 16;
    const uint32_t npts = (uint32_t)(SPEC_MS_H * (SPEC_MS_W - j0));
    if (method == 1) {       /* Hillaire 2020: isotropic 2nd order and the series 1 / (1 - f_ms) */
        double* F = (double*)calloc(nt, sizeof(double));
        c.order = 1; c.hillaire = 1; c.Jn = J[0]; c.En = E[0]; c.F = F;
        jobs_parallel_for(npts, 2, ord_fn, &c);
        for (size_t t = 0; t < nt; ++t) ms[t] = J[0][t] / (1.0 - fmin(F[t], 0.99));
        /* ground: the direct-sun-lit hemisphere integrated with the same series */
        for (int j = 0; j < SPEC_MS_W; ++j) gd[j] = E[0][j] / (1.0 - fmin(F[j], 0.99));
        free(F);
        *qout = 0;
    } else {                 /* iterated orders with the geometric tail of the settled ratio */
        c.hillaire = 0;
        int last = 1;
        double Q = 0, Qprev = -1;
        for (int ord = 1; ord <= ORD_MAX; ++ord) {
            c.order = ord;
            if (ord == 1) { c.nmu_s = 20; c.nmu_g = 10; c.nphi = 16; }
            else          { c.nmu_s = 12; c.nmu_g = 6;  c.nphi = 8; }
            c.J = J[ord];            /* source of order ord (unused for ord 1) */
            c.G = G;
            c.Jn = J[ord + 1];
            c.En = E[ord];
            jobs_parallel_for(npts, 2, ord_fn, &c);
            for (int j = 0; j < SPEC_MS_W; ++j) G[j] = SPEC_GROUND_ALBEDO / PI * E[ord][j];
            last = ord;
            if (ord >= 2) {
                double s0 = 0, s1 = 0;
                for (size_t t = 0; t < nt; ++t) {
                    if ((int)(t % SPEC_MS_W) < j0) continue;
                    s0 += J[ord][t]; s1 += J[ord + 1][t];
                }
                Q = s0 > 0 ? s1 / s0 : 0;
                if (ord >= ORD_MIN && fabs(Q - Qprev) < 0.003) break;
                Qprev = Q;
            }
        }
        Q = clampd(Q, 0, 0.97);
        for (size_t t = 0; t < nt; ++t) {
            const int i = (int)(t / SPEC_MS_W);
            double s = 0;
            for (int k = 2; k <= last + 1; ++k) s += J[k][t];
            s += J[last + 1][t] * Q / (1 - Q);
            const double ss = a->g_sr[g] * exp(-ms_h(a, i) / a->H) + a->g_sh[g] * exp(-ms_h(a, i) / a->haze_H);
            ms[t] = ss > 0 ? s / ss : 0;
        }
        double e0 = 0, e1 = 0;
        for (int j = j0; j < SPEC_MS_W; ++j) { e0 += E[last - 1][j]; e1 += E[last][j]; }
        const double QE = clampd(e0 > 0 ? e1 / e0 : 0, 0, 0.97);
        for (int j = 0; j < SPEC_MS_W; ++j) {
            double s = 0;
            for (int k = 1; k <= last; ++k) s += E[k][j];
            gd[j] = s + E[last][j] * QE / (1 - QE);
        }
        *qout = Q + 100.0 * last;      /* report: ratio and the number of orders */
    }
    for (int k = 0; k <= ORD_MAX + 1; ++k) free(J[k]);
}

/* PAR-weighted global horizontal transmission, sun at zenith, ground albedo
 * 0.15: the LUT model (g-points, multiple scattering) */
static double par_global(SpectralAtmo* a, double* msbuf, double* gd) {
    double num = 0, den = 0, q, A, Hz;
    spectral_columns(a, 0, 1.0, &A, &Hz);
    for (int g = 0; g < SPEC_NG; ++g) {
        if (a->g_par[g] <= 0) continue;
        build_ms_g(a, g, a->ms_method, SPEC_MS_W - 4, msbuf, gd, &q);
        num += a->g_par[g] * (GT(a, g, A, Hz) + gd[SPEC_MS_W - 1]);
        den += a->g_par[g];
    }
    return num / den;
}

/* ---- surfaces ---------------------------------------------------------------------------- */
static const float MAT_SPEC[][SPEC_NB] = {
    /* basalt   */ { .050f, .052f, .055f, .058f, .060f, .062f, .065f, .068f },
    /* granite  */ { .20f, .22f, .24f, .26f, .28f, .29f, .30f, .32f },
    /* dirt     */ { .06f, .07f, .09f, .12f, .15f, .18f, .21f, .23f },
    /* sand     */ { .28f, .32f, .37f, .42f, .47f, .50f, .53f, .55f },
    /* regolith */ { .12f, .14f, .16f, .19f, .22f, .24f, .26f, .28f },
    /* hematite */ { .04f, .05f, .06f, .08f, .15f, .28f, .33f, .35f },
    /* ice      */ { .80f, .80f, .79f, .77f, .74f, .70f, .65f, .58f },
    /* bark     */ { .05f, .06f, .07f, .09f, .12f, .15f, .18f, .20f } };
static const float PYRITE_F0[SPEC_NB] = { 0.30f, 0.32f, 0.36f, 0.42f, 0.48f, 0.52f, 0.54f, 0.56f };

/* ---- M5 pigments (spec 5.1) --------------------------------------------------------------- */
/* The hue bin i = 1..15 is the biomap palette index the exporter derives from
 * the gene G_HUE (hue_index in bio_io.c); its Q band sits at 383 + 34.13 i nm
 * (the gene form 400 + 2 G_HUE at the bin centre). Soret band 435 nm, a weak
 * background absorption below 690 nm (to 0 at 740). OD = PIG / 255 x 4 x eps
 * (4 includes the in-leaf path lengthening). */
double spectral_pigment_eps(double l, int bin) {
    if (bin <= 0 || bin >= SPEC_NHUE) bin = SPEC_HUE_DEFAULT;
    const double lq = 383.0 + 34.13 * bin;
    const double s = l < 690 ? 1.0 : (l > 740 ? 0.0 : (740 - l) / 50.0);
    const double ds = (l - 435) / 30.0, dq = (l - lq) / 25.0;
    return 1.2 * exp(-0.5 * ds * ds) + exp(-0.5 * dq * dq) + 0.15 * s;
}
static double pigment_A_nm(double l, int bin, double pig) { return 1 - pow(10.0, -pig / 255.0 * 4.0 * spectral_pigment_eps(l, bin)); }
double spectral_leaf_rho_nm(double l, int bin, double pig) { return 0.04 + 0.44 * (1 - pigment_A_nm(l, bin, pig)); }
/* band means at 1 nm of the absorptance, and the tissue spectra built from it:
 * leaf rho 0.04 + 0.44 (1 - A), tau 0.42 (1 - A) (rho + tau <= 0.9, tau >= 0);
 * mat 0.03 + rho_sand (1 - A)^2 */
static void build_pigments(SpectralAtmo* a) {
    for (int h = 0; h < SPEC_NHUE; ++h) {
        for (int b = 0; b < SPEC_NB; ++b) {
            double s = 0;
            for (int i = 0; i < 45; ++i) s += pigment_A_nm(SPEC_L0 + SPEC_BW * b + i + 0.5, h, SPEC_PIG_REF);
            const double A = s / 45.0;
            a->pig_A[h][b] = (float)A;
            a->leaf_rho[h][b] = (float)(0.04 + 0.44 * (1 - A));
            a->leaf_tau[h][b] = (float)(0.42 * (1 - A));
            a->mat_tis[h][b] = (float)(0.03 + MAT_SPEC[3][b] * (1 - A) * (1 - A));
        }
        double s = 0;
        for (int i = 0; i < 160; ++i) s += pigment_A_nm(740.5 + i, h, SPEC_PIG_REF);
        a->pig_A_nir[h] = s / 160.0;
    }
}

/* smooth band reflectance basis: minimum-slope spectra whose colour under an
 * equal-energy white (CAT16 to D65) is pure linear-sRGB red, green, blue */
static int build_rgb_basis(SpectralAtmo* a) {
    double K[3][SPEC_NB], cat[3][3], E[3] = { 0, 0, 0 };
    for (int b = 0; b < SPEC_NB; ++b) for (int c = 0; c < 3; ++c) E[c] += a->cmf[b][c];
    cat16(E, 1.0, cat);
    for (int b = 0; b < SPEC_NB; ++b) {
        double xyz[3], t[3], rgb[3];
        for (int c = 0; c < 3; ++c) xyz[c] = a->cmf[b][c] / E[1];
        mat3_mul_v(cat, xyz, t);
        mat3_mul_v(XYZ2RGB, t, rgb);
        for (int c = 0; c < 3; ++c) K[c][b] = rgb[c];
    }
    /* Q = D'D + eps I (D: first differences: smooth, flat toward the ends where
     * the eye barely sees), B = Q^-1 K' (K Q^-1 K')^-1 */
    double Q[SPEC_NB][SPEC_NB], Qi[SPEC_NB][2 * SPEC_NB];
    memset(Q, 0, sizeof Q);
    for (int r = 0; r + 1 < SPEC_NB; ++r) {
        const int ix[2] = { r, r + 1 };
        const double cf[2] = { 1, -1 };
        for (int p = 0; p < 2; ++p) for (int q = 0; q < 2; ++q) Q[ix[p]][ix[q]] += cf[p] * cf[q];
    }
    for (int i = 0; i < SPEC_NB; ++i) Q[i][i] += 1e-7;
    for (int i = 0; i < SPEC_NB; ++i) for (int j = 0; j < 2 * SPEC_NB; ++j) Qi[i][j] = j < SPEC_NB ? Q[i][j] : (j - SPEC_NB == i);
    for (int c = 0; c < SPEC_NB; ++c) {          /* Gauss-Jordan */
        int pv = c;
        for (int r = c + 1; r < SPEC_NB; ++r) if (fabs(Qi[r][c]) > fabs(Qi[pv][c])) pv = r;
        for (int j = 0; j < 2 * SPEC_NB; ++j) { const double t = Qi[c][j]; Qi[c][j] = Qi[pv][j]; Qi[pv][j] = t; }
        const double d = Qi[c][c];
        if (fabs(d) < 1e-300) return -1;
        for (int j = 0; j < 2 * SPEC_NB; ++j) Qi[c][j] /= d;
        for (int r = 0; r < SPEC_NB; ++r)
            if (r != c) { const double f = Qi[r][c]; for (int j = 0; j < 2 * SPEC_NB; ++j) Qi[r][j] -= f * Qi[c][j]; }
    }
    double QK[SPEC_NB][3], M[3][3], Mi[3][3];
    for (int i = 0; i < SPEC_NB; ++i)
        for (int c = 0; c < 3; ++c) { double s = 0; for (int k = 0; k < SPEC_NB; ++k) s += Qi[i][SPEC_NB + k] * K[c][k]; QK[i][c] = s; }
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) { double s = 0; for (int k = 0; k < SPEC_NB; ++k) s += K[r][k] * QK[k][c]; M[r][c] = s; }
    if (mat3_inv(M, Mi)) return -1;
    for (int c = 0; c < 3; ++c)
        for (int b = 0; b < SPEC_NB; ++b) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += QK[b][k] * Mi[k][c];
            a->rgb_basis[c][b] = (float)s;
        }
    return 0;
}
static void build_materials(SpectralAtmo* a) {
    memset(a->mat_rho, 0, sizeof a->mat_rho);
    for (int m = 0; m < SPEC_NMAT; ++m) { a->mat_alpha[m] = 0.7f; a->mat_ior[m] = SPEC_IOR_ROCK; }
    const struct { int m, row; float alpha; int ior; } T[] = {
        { MAT_STONE, 0, 0.6f, SPEC_IOR_ROCK }, { MAT_GRANITE, 1, 0.5f, SPEC_IOR_ROCK }, { MAT_DIRT, 2, 0.85f, SPEC_IOR_ROCK },
        { MAT_SAND, 3, 0.7f, SPEC_IOR_ROCK },  { MAT_REGOLITH, 4, 0.8f, SPEC_IOR_ROCK }, { MAT_IRON_ORE, 5, 0.6f, SPEC_IOR_ROCK },
        { MAT_ICE, 6, 0.08f, SPEC_IOR_ICE },   { MAT_WOOD, 7, 0.8f, SPEC_IOR_LEAF } };
    for (size_t i = 0; i < sizeof T / sizeof T[0]; ++i) {
        memcpy(a->mat_rho[T[i].m], MAT_SPEC[T[i].row], sizeof(float) * SPEC_NB);
        a->mat_alpha[T[i].m] = T[i].alpha;
        a->mat_ior[T[i].m] = T[i].ior;
    }
    for (int b = 0; b < SPEC_NB; ++b) { a->mat_rho[MAT_SULFIDE][b] = 0.02f; a->mat_rho[MAT_WATER][b] = 0.02f; }
    a->mat_alpha[MAT_SULFIDE] = 0.35f;
    a->mat_ior[MAT_WATER] = SPEC_IOR_WATER;
    a->mat_alpha[MAT_WATER] = 0.05f;
    memcpy(a->pyrite_f0, PYRITE_F0, sizeof a->pyrite_f0);
    /* pigmented tissue: the default hue bin's spectra (the renderer takes each pixel's own hue bin);
     * grass ground at cover 192: mix(soil, leaf, 0.8 cover / 255) */
    build_pigments(a);
    const int H = SPEC_HUE_DEFAULT;
    for (int b = 0; b < SPEC_NB; ++b) {
        a->mat_rho[MAT_LEAF][b] = a->leaf_rho[H][b];
        a->mat_rho[MAT_KELP][b] = a->leaf_rho[H][b];
        a->mat_rho[MAT_MAT][b] = a->mat_tis[H][b];
        a->mat_rho[MAT_GRASS][b] = MAT_SPEC[2][b] + (a->leaf_rho[H][b] - MAT_SPEC[2][b]) * 0.8f * (192.f / 255.f);
    }
    const int tm[4] = { MAT_GRASS, MAT_MAT, MAT_LEAF, MAT_KELP };
    const float ta[4] = { 0.6f, 0.6f, 0.5f, 0.4f };
    for (int i = 0; i < 4; ++i) {
        a->mat_alpha[tm[i]] = ta[i];
        a->mat_ior[tm[i]] = SPEC_IOR_LEAF;
    }
}

static const double IOR_N[SPEC_NIOR] = { 1.55, 1.45, 1.31, 1.333 };
/* directional albedo of the GGX lobe with exact Fresnel (stratified VNDF sampling) */
static double ggx_albedo(double mu, double alpha, double eta, int n) {
    if (mu < 1e-4) mu = 1e-4;
    if (alpha < 2e-3) alpha = 2e-3;
    const double V[3] = { sqrt(1 - mu * mu), 0, mu };
    const double a2 = alpha * alpha;
    const double lv = (-1 + sqrt(1 + a2 * (1 - mu * mu) / (mu * mu))) * 0.5;
    double sum = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const double u1 = (i + 0.5) / n, u2 = (j + 0.5) / n;
            double Vh[3] = { alpha * V[0], alpha * V[1], V[2] };
            const double vl = sqrt(Vh[0] * Vh[0] + Vh[1] * Vh[1] + Vh[2] * Vh[2]);
            for (int k = 0; k < 3; ++k) Vh[k] /= vl;
            const double lensq = Vh[0] * Vh[0] + Vh[1] * Vh[1];
            double T1[3] = { 1, 0, 0 };
            if (lensq > 0) { const double il = 1 / sqrt(lensq); T1[0] = -Vh[1] * il; T1[1] = Vh[0] * il; T1[2] = 0; }
            const double T2[3] = { Vh[1] * T1[2] - Vh[2] * T1[1], Vh[2] * T1[0] - Vh[0] * T1[2], Vh[0] * T1[1] - Vh[1] * T1[0] };
            const double r = sqrt(u1), ph = 2 * PI * u2;
            const double t1 = r * cos(ph);
            double t2 = r * sin(ph);
            const double s = 0.5 * (1 + Vh[2]);
            t2 = (1 - s) * sqrt(fmax(0, 1 - t1 * t1)) + s * t2;
            const double tz = sqrt(fmax(0, 1 - t1 * t1 - t2 * t2));
            double Nh[3];
            for (int k = 0; k < 3; ++k) Nh[k] = t1 * T1[k] + t2 * T2[k] + tz * Vh[k];
            double Ne[3] = { alpha * Nh[0], alpha * Nh[1], fmax(0, Nh[2]) };
            const double nl = sqrt(Ne[0] * Ne[0] + Ne[1] * Ne[1] + Ne[2] * Ne[2]);
            for (int k = 0; k < 3; ++k) Ne[k] /= nl;
            const double vn = V[0] * Ne[0] + V[1] * Ne[1] + V[2] * Ne[2];
            const double L[3] = { 2 * vn * Ne[0] - V[0], 2 * vn * Ne[1] - V[1], 2 * vn * Ne[2] - V[2] };
            if (L[2] <= 0) continue;
            const double ll = (-1 + sqrt(1 + a2 * (1 - L[2] * L[2]) / (L[2] * L[2]))) * 0.5;
            const double F = spec_fresnel((float)vn, (float)eta);
            sum += F * (1 + lv) / (1 + lv + ll);       /* F G2 / G1(v) */
        }
    return sum / ((double)n * n);
}
typedef struct ElCtx { SpectralAtmo* a; } ElCtx;
static void el_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    SpectralAtmo* a = ((ElCtx*)p)->a;
    for (uint32_t k = b; k < e; ++k) {
        const int ior = (int)(k / SPEC_EL_N), ja = (int)(k % SPEC_EL_N);
        const double alpha = ja / (double)(SPEC_EL_N - 1);
        for (int im = 0; im < SPEC_EL_N; ++im) a->e_lut[ior][ja][im] = (float)ggx_albedo(im / (double)(SPEC_EL_N - 1), alpha, IOR_N[ior], 48);
        double s = 0;
        for (int im = 0; im + 1 < SPEC_EL_N; ++im) {     /* E_avg = 2 int E mu dmu */
            const double m0 = im / (double)(SPEC_EL_N - 1), m1 = (im + 1) / (double)(SPEC_EL_N - 1);
            s += 0.5 * (a->e_lut[ior][ja][im] * m0 + a->e_lut[ior][ja][im + 1] * m1) * (m1 - m0);
        }
        a->e_avg[ior][ja] = (float)(2 * s);
    }
}
static double e_lookup(const SpectralAtmo* a, int ior, double mu, double alpha) {
    const double fx = clampd(mu, 0, 1) * (SPEC_EL_N - 1), fy = clampd(alpha, 0, 1) * (SPEC_EL_N - 1);
    int i = (int)fx, j = (int)fy;
    if (i > SPEC_EL_N - 2) i = SPEC_EL_N - 2;
    if (j > SPEC_EL_N - 2) j = SPEC_EL_N - 2;
    const double u = fx - i, v = fy - j;
    return (a->e_lut[ior][j][i] * (1 - u) + a->e_lut[ior][j][i + 1] * u) * (1 - v) +
           (a->e_lut[ior][j + 1][i] * (1 - u) + a->e_lut[ior][j + 1][i + 1] * u) * v;
}
static double eavg_lookup(const SpectralAtmo* a, int ior, double alpha) {
    const double fy = clampd(alpha, 0, 1) * (SPEC_EL_N - 1);
    int j = (int)fy;
    if (j > SPEC_EL_N - 2) j = SPEC_EL_N - 2;
    const double v = fy - j;
    return a->e_avg[ior][j] * (1 - v) + a->e_avg[ior][j + 1] * v;
}

/* ---- water tables ----------------------------------------------------------------------------- */
static double water_T_1nm(int b, double d) {          /* band mean of e^(-a d) at 1 nm */
    double s = 0;
    for (int i = 0; i < 45; ++i) s += exp(-water_a(SPEC_L0 + SPEC_BW * b + i + 0.5) * d);
    return s / 45.0;
}
static void build_water(SpectralAtmo* a) {
    for (int b = 0; b <= SPEC_NB; ++b) {
        const double l0 = b < SPEC_NB ? SPEC_L0 + SPEC_BW * b : 740, l1 = b < SPEC_NB ? l0 + SPEC_BW : 900;
        double sa = 0, saw = 0, sb = 0, sbm = 0;
        int n = 0;
        for (double l = l0 + 0.5; l < l1; l += 1.0, ++n) { sa += water_a(l); saw += water_aw(l); sb += water_b(l); sbm += water_bmol(l); }
        a->water_aw[b] = saw / n;
        if (b < SPEC_NB) { a->water_a[b] = (float)(sa / n); a->water_b[b] = (float)(sb / n); a->water_bm[b] = (float)(sbm / n); }
    }
    for (int b = 0; b < SPEC_NB; ++b)
        for (int i = 0; i < SPEC_WT_N; ++i) {
            const double d = SPEC_WT_D0 * pow(SPEC_WT_D1 / SPEC_WT_D0, i / (double)(SPEC_WT_N - 1));
            a->water_T[b][i] = (float)water_T_1nm(b, d);
        }
}
/* the table lookup the GPU does: ln T interpolated linearly in d between log-spaced nodes */
static double water_T_lookup(const SpectralAtmo* a, int b, double d) {
    if (d <= 0) return 1;
    if (d <= SPEC_WT_D0) return pow(a->water_T[b][0], d / SPEC_WT_D0);
    const double x = log(d / SPEC_WT_D0) / log(SPEC_WT_D1 / SPEC_WT_D0) * (SPEC_WT_N - 1);
    int i = (int)x;
    if (i > SPEC_WT_N - 2) i = SPEC_WT_N - 2;
    const double d0 = SPEC_WT_D0 * pow(SPEC_WT_D1 / SPEC_WT_D0, i / (double)(SPEC_WT_N - 1));
    const double d1 = SPEC_WT_D0 * pow(SPEC_WT_D1 / SPEC_WT_D0, (i + 1) / (double)(SPEC_WT_N - 1));
    const double f = (d - d0) / (d1 - d0);
    const double l0 = log(fmax(a->water_T[b][i], 1e-30)), l1 = log(fmax(a->water_T[b][i + 1], 1e-30));
    return exp(l0 + (l1 - l0) * f);
}

/* ---- the build ---------------------------------------------------------------------------------- */
int spectral_build_atmo(const HomePlanet* hp, double p_bar, SpectralAtmo* a) {
    SpectralOpts o;
    memset(&o, 0, sizeof o);
    o.p_bar = p_bar;
    return spectral_build_atmo_ex(hp, &o, a);
}

int spectral_build_atmo_ex(const HomePlanet* hp, const SpectralOpts* opt, SpectralAtmo* a) {
    const double t_start = time_now();
    SpectralOpts o;
    memset(&o, 0, sizeof o);
    if (opt) o = *opt;
    void* host = a->host_owned == 0x5EC7 ? a->host : NULL;
    memset(a, 0, sizeof *a);
    a->host = host ? host : calloc(1, sizeof(HostTabs));
    if (!a->host) return -1;
    a->host_owned = 0x5EC7;
    a->earth = o.earth;
    a->ms_method = o.ms_method;
    a->sun_disc = o.sun_disc > 0 ? o.sun_disc : 1.0;
    /* star */
    if (o.earth) { a->teff = 5772; a->lum = 1; a->a_au = 1; }
    else { a->teff = hp->star_teff > 1000 ? hp->star_teff : 5772; a->lum = hp->star_lum > 0 ? hp->star_lum : 1; a->a_au = hp->a_au > 0 ? hp->a_au : 1; }
    a->s_top = 1361.0 * a->lum / (a->a_au * a->a_au);
    a->r_star = sqrt(a->lum) * (5772.0 / a->teff) * (5772.0 / a->teff);
    a->theta_sun = a->r_star * RSUN / (a->a_au * AU_M);
    {
        double par = 0;
        for (int i = 0; i < 300; ++i) par += etop_nm(a, 400.5 + i);
        a->par_wm2 = par;
        a->par_share = par / a->s_top;
        for (int b = 0; b <= SPEC_NB; ++b) {
            const double l0 = b < SPEC_NB ? SPEC_L0 + SPEC_BW * b : 740, l1 = b < SPEC_NB ? l0 + SPEC_BW : 900;
            double e = 0, ph = 0;
            for (double l = l0 + 0.5; l < l1; l += 1.0) { const double el = etop_nm(a, l); e += el; ph += el * l * 1e-9 / (HPL * CLIGHT); }
            a->e_band[b] = e;
            a->phot_band[b] = ph / NAVO * 1e6;
            a->ev_band[b] = ph > 0 ? e / ph / EVJ : 0;
            if (b < SPEC_NB) a->e_nm[b] = e / SPEC_BW;
        }
    }
    /* colour matching: band integrals at 1 nm, and the Wyman fit */
    for (int b = 0; b < SPEC_NB; ++b)
        for (int c = 0; c < 3; ++c) { a->cmf[b][c] = 0; a->cmf_wyman[b][c] = 0; }
    for (int b = 0; b < SPEC_NB; ++b)
        for (int i = 0; i < 45; ++i) {
            double x[3], y[3];
            const double l = SPEC_L0 + SPEC_BW * b + i + 0.5;
            cie_at(l, x);
            wyman_at(l, y);
            for (int c = 0; c < 3; ++c) { a->cmf[b][c] += x[c]; a->cmf_wyman[b][c] += y[c]; }
        }
    /* gases */
    double xs = 0;
    if (o.earth || !hp->atm_known) {
        for (int g = 0; g <= ATM_N; ++g) a->x[g] = 0;
        a->x[ATM_N2] = 0.7808; a->x[ATM_O2] = 0.2095; a->x[ATM_CO2] = 0.0004;
        if (!o.earth) a->x[ATM_N2] += 0.0093;
    } else {
        for (int g = 0; g < ATM_N; ++g) a->x[g] = hp->atm_mix[g];
        a->x[ATM_N] = 0;
    }
    if (o.h2o > 0) a->x[ATM_N] = o.h2o;
    for (int g = 0; g <= ATM_N; ++g) xs += a->x[g];
    for (int g = 0; g <= ATM_N; ++g) a->x[g] /= xs;
    a->mbar = 0;
    a->ray_factor = 0;
    for (int g = 0; g <= ATM_N; ++g) {
        a->mbar += a->x[g] * GAS[g].M;
        const double rn = GAS[g].n1 / 292.0;
        a->ray_factor += a->x[g] * rn * rn * GAS[g].fk / 1.048;
    }
    if (o.earth) { a->mbar = 28.97; a->ray_factor = 1.0; }
    a->p_bar = o.earth ? 1.01325 : (o.p_bar > 0 ? o.p_bar : 1.0);
    a->g = o.earth ? G0 : (hp->gravity_g > 0 ? hp->gravity_g * G0 : G0);
    a->t_surf = o.earth ? 288.15 : (hp->t_surf > 50 ? hp->t_surf : 288.15);
    a->N = a->p_bar * 1e5 / (a->mbar * 1e-3 / NAVO * a->g);
    a->H = RGAS * a->t_surf / (a->mbar * 1e-3 * a->g);
    a->R = (o.earth ? 1.0 : (hp->radius_re > 0 ? hp->radius_re : 1.0)) * 6371e3;
    a->top = a->R + 8.0 * a->H;
    a->ch4_file = ch4_load();
    a->ch4_col = a->x[ATM_CH4] * a->N / LOSCH * 1e-3;          /* km-amagat */
    a->tau_ray550 = a->N * sigma_air(550) * a->ray_factor;
    a->tau_ray550_earth = 1.01325e5 / (28.97e-3 / NAVO * G0) * sigma_air(550);
    /* haze */
    a->haze_g = 0.65;
    a->haze_H = 0.5 * a->H;
    const double xc = a->x[ATM_CH4];
    a->haze_q12 = (int)(4096 - (int64_t)(2048.0 * xc / (xc + 0.2)));
    if (o.earth) a->haze_q12 = 4096;
    else if (o.haze_q12 > 0) a->haze_q12 = o.haze_q12 > 4096 ? 4096 : o.haze_q12;   /* M5 export: a haze bin */
    a->haze_target = a->haze_q12 / 4096.0;
    /* surfaces, water, specular albedo (independent of the haze) */
    build_rgb_basis(a);
    build_materials(a);
    build_water(a);
    {
        ElCtx ec = { a };
        jobs_parallel_for(SPEC_NIOR * SPEC_EL_N, 1, el_fn, &ec);
    }
    /* columns (geometry only), then the haze calibration on the full LUT model */
    a->haze_tau550 = 0;
    setup_fine(a);
    build_columns(a);
    double* msbuf = (double*)malloc(sizeof(double) * SPEC_MS_H * SPEC_MS_W);
    double gd[SPEC_MS_W];
    {   /* Rayleigh only (no gas absorption), for the record */
        const double keep = a->ch4_col;
        a->ch4_col = 0; setup_fine(a);
        a->haze_ray_nogas = par_global(a, msbuf, gd);
        a->ch4_col = keep; setup_fine(a);
    }
    a->haze_rayleigh_only = par_global(a, msbuf, gd);
    double tau_h = 0;
    if (a->haze_target < 0.9999) {
        if (a->haze_rayleigh_only <= a->haze_target + 1e-4) {
            tau_h = 0;
            a->haze_iters = -1;      /* warning: gas + Rayleigh alone are below the target */
        } else {
            double lo = 0, flo = a->haze_rayleigh_only - a->haze_target, hi = 0.5, fhi = 0;
            for (int k = 0; k < 10; ++k) {
                a->haze_tau550 = hi; setup_fine(a);
                fhi = par_global(a, msbuf, gd) - a->haze_target;
                if (fhi < 0) break;
                lo = hi; flo = fhi; hi *= 2;
            }
            int side = 0, it = 0;
            double x = hi, fx = fhi;
            for (it = 0; it < 40 && fabs(fx) > 1e-4; ++it) {   /* Illinois: bracketed, falls back to bisection */
                x = (lo * fhi - hi * flo) / (fhi - flo);
                if (!(x > lo && x < hi)) x = 0.5 * (lo + hi);
                a->haze_tau550 = x; setup_fine(a);
                fx = par_global(a, msbuf, gd) - a->haze_target;
                if (fx * fhi > 0) { hi = x; fhi = fx; if (side == -1) flo *= 0.5; side = -1; }
                else { lo = x; flo = fx; if (side == 1) fhi *= 0.5; side = 1; }
            }
            tau_h = x;
            a->haze_iters = it;
        }
    }
    a->haze_tau550 = tau_h;
    setup_fine(a);
    /* the full multiple-scattering LUT (the export needs the sun above the horizon only: ms_j0) */
    const int j0 = o.ms_j0 > 0 && o.ms_j0 < SPEC_MS_W - 2 ? o.ms_j0 : 0;
    for (int g = 0; g < SPEC_NG; ++g) {
        double q;
        build_ms_g(a, g, a->ms_method, j0, msbuf, gd, &q);
        a->ms_q[g] = q;
        for (int i = 0; i < SPEC_MS_H; ++i) for (int j = 0; j < SPEC_MS_W; ++j) a->ms[g][i][j] = (float)msbuf[i * SPEC_MS_W + j];
        for (int j = 0; j < SPEC_MS_W; ++j) a->gdiff[g][j] = (float)gd[j];
    }
    {   /* achieved PAR transmission with the final tables */
        double num = 0, den = 0, A, Hz;
        spectral_columns(a, 0, 1.0, &A, &Hz);
        for (int g = 0; g < SPEC_NG; ++g) { num += a->g_par[g] * (GT(a, g, A, Hz) + a->gdiff[g][SPEC_MS_W - 1]); den += a->g_par[g]; }
        a->haze_achieved = num / den;
        a->haze_residual = fabs(a->haze_achieved - a->haze_target);
        if (a->haze_target >= 0.9999) a->haze_residual = 0;
    }
    free(msbuf);
    a->build_s = time_now() - t_start;
    return 0;
}

void spectral_free(SpectralAtmo* a) {
    if (a && a->host_owned == 0x5EC7) free(a->host);
    if (a) { a->host = NULL; a->host_owned = 0; }
}

/* ---- a disk cache of the build (the viewer's startup; the build takes 5-18 s) ------------------------------------
 * The key covers the planet hash, the options, this binary's build stamp (spectral.c's compile date and time: every
 * change of the code that fills SpectralAtmo recompiles this file), the struct sizes and the CH4 table if present.
 * The file holds the SpectralAtmo (its host pointer cleared) and the host tables, with an FNV-1a checksum; a
 * mismatch of any of these falls back to the build. Directory: $GME_CACHE, else the temp directory; GME_CACHE=0
 * disables it. */
typedef struct SpecCacheHead { char magic[8]; uint64_t key, size, sum; } SpecCacheHead;
static uint64_t spec_fnv(uint64_t h, const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001B3ull; }
    return h;
}
static uint64_t spec_cache_key(const SpectralOpts* o, uint64_t phash) {
    static const char stamp[] = "GMESPC1 " __DATE__ " " __TIME__;
    uint64_t h = spec_fnv(0xCBF29CE484222325ull, stamp, sizeof stamp);
    const uint64_t sz[3] = { sizeof(SpectralAtmo), sizeof(HostTabs), sizeof(void*) };
    const double od[3] = { o->p_bar, o->h2o, o->sun_disc };
    const int32_t oi[5] = { o->ms_method, o->quiet, o->earth, o->haze_q12, o->ms_j0 };
    h = spec_fnv(h, sz, sizeof sz);
    h = spec_fnv(h, &phash, sizeof phash);
    h = spec_fnv(h, od, sizeof od);
    h = spec_fnv(h, oi, sizeof oi);
    FILE* f = fopen("data/ch4_k98.txt", "rb");
    if (f) {
        unsigned char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0) h = spec_fnv(h, buf, n);
        fclose(f);
    }
    return h;
}
static int spec_cache_path(uint64_t key, char* out, size_t n) {
    const char* d = getenv("GME_CACHE");
    if (d && !strcmp(d, "0")) return 0;
#ifdef _WIN32
    if (!d || !*d) d = getenv("TEMP");
    if (!d || !*d) d = getenv("TMP");
#else
    if (!d || !*d) d = getenv("TMPDIR");
    if (!d || !*d) d = "/tmp";
#endif
    if (!d || !*d) return 0;
    snprintf(out, n, "%s/gme_spectral_%016llx.bin", d, (unsigned long long)key);
    return 1;
}

int spectral_build_atmo_cached(const HomePlanet* hp, const SpectralOpts* o, uint64_t planet_hash, SpectralAtmo* a, int* cached) {
    SpectralOpts oo;
    memset(&oo, 0, sizeof oo);
    if (o) oo = *o;
    *cached = 0;
    char path[1024];
    const uint64_t key = spec_cache_key(&oo, planet_hash);
    const int use = spec_cache_path(key, path, sizeof path);
    const size_t na = sizeof(SpectralAtmo), nh = sizeof(HostTabs), hoff = offsetof(SpectralAtmo, host), ooff = offsetof(SpectralAtmo, host_owned);
    unsigned char* buf = (unsigned char*)malloc(na + nh);
    if (use && buf) {
        FILE* f = fopen(path, "rb");
        if (f) {
            SpecCacheHead hd;
            const int ok = fread(&hd, sizeof hd, 1, f) == 1 && !memcmp(hd.magic, "GMESPC1", 8) && hd.key == key && hd.size == na + nh &&
                           fread(buf, 1, na + nh, f) == na + nh && spec_fnv(0xCBF29CE484222325ull, buf, na + nh) == hd.sum;
            fclose(f);
            if (ok) {
                void* host = a->host_owned == 0x5EC7 ? a->host : NULL;
                memcpy(a, buf, na);
                a->host = host ? host : malloc(nh);
                if (a->host) {
                    memcpy(a->host, buf + na, nh);
                    a->host_owned = 0x5EC7;
                    *cached = 1;
                    free(buf);
                    return 0;
                }
                a->host_owned = 0;
            }
        }
    }
    if (spectral_build_atmo_ex(hp, &oo, a)) { free(buf); return -1; }
    if (use && buf) {                                   /* write a temporary file, then rename it into place */
        memcpy(buf, a, na);
        memset(buf + hoff, 0, sizeof(void*));
        memset(buf + ooff, 0, sizeof(int));
        memcpy(buf + na, a->host, nh);
        SpecCacheHead hd;
        memcpy(hd.magic, "GMESPC1", 8);
        hd.key = key;
        hd.size = na + nh;
        hd.sum = spec_fnv(0xCBF29CE484222325ull, buf, na + nh);
        char tmp[1100];
        snprintf(tmp, sizeof tmp, "%s.%08x.tmp", path, (unsigned)((uint64_t)(time_now() * 1e6) ^ (uint64_t)(uintptr_t)a));
        FILE* f = fopen(tmp, "wb");
        if (f) {
            const int ok = fwrite(&hd, sizeof hd, 1, f) == 1 && fwrite(buf, 1, na + nh, f) == na + nh;
            if (fclose(f) || !ok) remove(tmp);
            else if (rename(tmp, path)) { remove(path); if (rename(tmp, path)) remove(tmp); }
        }
    }
    free(buf);
    return 0;
}

void spectral_print(const SpectralAtmo* a) {
    printf("Light: star %.1f K, L %.4f, a %.4f AU -> S_top %.1f W/m^2 (bio %d); disc R* %.3f R_sun, theta %.2f mrad (%.3f deg)%s\n",
           a->teff, a->lum, a->a_au, a->s_top, (int)floor(a->s_top + 0.5), a->r_star, a->theta_sun * 1e3, a->theta_sun * 180 / PI,
           a->sun_disc != 1.0 ? " (drawn x --sun-disc)" : "");
    printf("       PAR 400-700 nm %.1f%% of S_top = %.0f W/m^2 (the bio assumes 45%% = %.0f W/m^2)\n", 100 * a->par_share,
           a->par_wm2, 0.45 * a->s_top);
    printf("       TOA E per band (W/m^2):");
    for (int b = 0; b < SPEC_NB; ++b) printf(" %.0f", a->e_band[b]);
    printf(" | NIR 740-900 %.0f; photons (umol/m^2/s):", a->e_band[SPEC_NB]);
    for (int b = 0; b <= SPEC_NB; ++b) printf(" %.0f", a->phot_band[b]);
    printf("\nAtmo : P %.3f bar, mbar %.2f g/mol, g %.3f m/s^2, T %.1f K -> H %.2f km, N %.3e m^-2 (Earth %.3e), R_p %.0f km, top +%.1f km\n",
           a->p_bar, a->mbar, a->g, a->t_surf, a->H * 1e-3, a->N, 1.01325e5 / (28.97e-3 / NAVO * G0), a->R * 1e-3,
           (a->top - a->R) * 1e-3);
    printf("       Rayleigh %.3f x air, tau_R(550) %.3f (Earth preset %.4f); tau_R per band", a->ray_factor, a->tau_ray550,
           a->tau_ray550_earth);
    for (int b = 0; b < SPEC_NB; ++b) printf(" %.2f", a->tau_ray[b]);
    printf("\n       CH4 column %.2f km-amagat (%s); tau_CH4 per band", a->ch4_col,
           a->ch4_file ? "data/ch4_k98.txt" : "WARNING: data/ch4_k98.txt not found, built-in coarse band model");
    for (int b = 0; b < SPEC_NB; ++b) printf(" %.3f", a->tau_ch4[b]);
    printf("\n       haze: bio haze_q12 %d (%.3f) -> tau_h550 %.3f, residual %.4f (PAR global transmission %.4f; Rayleigh only %.3f, "
           "Rayleigh + gas %.3f)%s\n",
           a->haze_q12, a->haze_target, a->haze_tau550, a->haze_residual, a->haze_achieved, a->haze_ray_nogas, a->haze_rayleigh_only,
           a->haze_iters < 0 ? "  WARNING: Rayleigh + gas already below the target, no haze" : "");
    printf("       tau_haze per band");
    for (int b = 0; b < SPEC_NB; ++b) printf(" %.2f", a->tau_haze[b]);
    printf("; total vertical tau per band");
    for (int b = 0; b < SPEC_NB; ++b) printf(" %.2f", a->tau_total[b]);
    printf("\n       light: %d g-points (correlated k, %d per band); multiple scattering: %s; sky irradiance / TOA at the ground, "
           "sun at zenith, per band",
           SPEC_NG, SPEC_GPB, a->ms_method ? "Hillaire 2020 series" : "iterated orders until settled + geometric tail");
    for (int b = 0; b < SPEC_NB; ++b) {
        double s = 0, e = 0;
        for (int g = b * SPEC_GPB; g < (b + 1) * SPEC_GPB; ++g) { s += a->g_e[g] * a->gdiff[g][SPEC_MS_W - 1]; e += a->g_e[g]; }
        printf(" %.2f", s / e);
    }
    printf(" (built in %.1f s)\n", a->build_s);
    fflush(stdout);
}

/* ---- host twin of k_sky_view --------------------------------------------------------------------- */
#define SKY_NSTEP 40
static double ms_val(const SpectralAtmo* a, int g, double h, double mus) {
    const double fy = sqrt(clampd(h / (a->top - a->R), 0, 1)) * (SPEC_MS_H - 1);
    const double fx = clampd((mus - SPEC_MS_MU0) / (SPEC_MS_MU1 - SPEC_MS_MU0), 0, 1) * (SPEC_MS_W - 1);
    int i = (int)fy, j = (int)fx;
    if (i > SPEC_MS_H - 2) i = SPEC_MS_H - 2;
    if (j > SPEC_MS_W - 2) j = SPEC_MS_W - 2;
    const double v = fy - i, u = fx - j;
    return ((a->ms[g][i][j] * (1 - u) + a->ms[g][i][j + 1] * u) * (1 - v) + (a->ms[g][i + 1][j] * (1 - u) + a->ms[g][i + 1][j + 1] * u) * v) *
           spec_ms_fade((float)mus);
}
static double gd_val(const SpectralAtmo* a, int g, double mus) {
    const double fx = clampd((mus - SPEC_MS_MU0) / (SPEC_MS_MU1 - SPEC_MS_MU0), 0, 1) * (SPEC_MS_W - 1);
    int j = (int)fx;
    if (j > SPEC_MS_W - 2) j = SPEC_MS_W - 2;
    const double u = fx - j;
    return (a->gdiff[g][j] * (1 - u) + a->gdiff[g][j + 1] * u) * spec_ms_fade((float)mus);
}
void spectral_sky_radiance(const SpectralAtmo* a, const double sun[3], double cam_h, const double dir[3], double* out_g,
                           double* out_band) {
    const double h0 = fmax(cam_h, 0), r = a->R + h0, mu = dir[1];
    const int ground = mu < mu_hor(a, h0);
    const double send = ground ? dist_ground(a, h0, mu) : dist_top(a, h0, mu);
    const double cs = dir[0] * sun[0] + dir[1] * sun[1] + dir[2] * sun[2];
    const double pr = phase_ray(cs), ph = phase_hg(cs, a->haze_g);
    double acc[SPEC_NG], fprev[SPEC_NG];
    for (int g = 0; g < SPEC_NG; ++g) { acc[g] = 0; fprev[g] = 0; }
    if (send > 0) {
        const double smin = 100.0, q = pow(1 + send / smin, 1.0 / SKY_NSTEP);
        double Av = 0, Hv = 0, sprev = 0, dpa = 0, dph = 0, qk = 1, mus = 0;
        for (int k = 0; k <= SKY_NSTEP; ++k) {
            const double s = k == SKY_NSTEP ? send : smin * (qk - 1);
            qk *= q;
            const double px = s * dir[0], py = r + s * dir[1], pz = s * dir[2];
            const double rr = sqrt(px * px + py * py + pz * pz), hh = fmax(rr - a->R, 0);
            mus = (px * sun[0] + py * sun[1] + pz * sun[2]) / rr;
            const double da = exp(-hh / a->H), dh = exp(-hh / a->haze_H);
            if (k > 0) { Av += logmean_int(dpa, da, s - sprev) / a->H; Hv += logmean_int(dph, dh, s - sprev) / a->haze_H; }
            double As, Hs;
            spectral_columns(a, hh, mus, &As, &Hs);
            for (int g = 0; g < SPEC_NG; ++g) {
                const double f = (a->g_sr[g] * da * pr + a->g_sh[g] * dh * ph) * GT(a, g, Av + As, Hv + Hs) +
                                 (a->g_sr[g] * da + a->g_sh[g] * dh) * ms_val(a, g, hh, mus) * GT(a, g, Av, Hv);
                if (k > 0) acc[g] += logmean_int(fprev[g], f, s - sprev);
                fprev[g] = f;
            }
            sprev = s; dpa = da; dph = dh;
        }
        if (ground) {
            double As, Hs;
            spectral_columns(a, 0, mus, &As, &Hs);
            for (int g = 0; g < SPEC_NG; ++g)
                acc[g] += SPEC_GROUND_ALBEDO / PI * (GT(a, g, Av + As, Hv + Hs) * fmax(mus, 0) + GT(a, g, Av, Hv) * gd_val(a, g, mus));
        }
    }
    for (int g = 0; g < SPEC_NG; ++g) {
        acc[g] *= a->g_e[g];
        if (out_g) out_g[g] = acc[g];
    }
    if (out_band)
        for (int b = 0; b < SPEC_NB; ++b) {
            double s = 0;
            for (int g = b * SPEC_GPB; g < (b + 1) * SPEC_GPB; ++g) s += acc[g];
            out_band[b] = s / SPEC_BW;
        }
}

void spectral_rgb_of(const SpectralAtmo* a, const double* rho, double rgb[3]) {
    double E[3] = { 0, 0, 0 }, xyz[3] = { 0, 0, 0 }, cat[3][3], t[3];
    for (int b = 0; b < SPEC_NB; ++b)
        for (int c = 0; c < 3; ++c) { E[c] += a->cmf[b][c]; xyz[c] += rho[b] * a->cmf[b][c]; }
    for (int c = 0; c < 3; ++c) xyz[c] /= E[1];
    cat16(E, 1.0, cat);
    mat3_mul_v(cat, xyz, t);
    mat3_mul_v(XYZ2RGB, t, rgb);
}

/* ==== D3 ============================================================================================ */
/* ---- Monte Carlo sky reference: backward path tracing in the spherical shell,
 * 5 nm spectral (one sample wavelength per path), Rayleigh + HG haze + CH4,
 * Lambertian ground 0.15, sun by next-event estimation with exact transmittance ---- */
typedef struct Rng { uint64_t s; } Rng;
static double rng_u(Rng* r) {
    r->s ^= r->s >> 12; r->s ^= r->s << 25; r->s ^= r->s >> 27;
    return (double)((r->s * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}
typedef struct McWl { double ka, kh, ksr, ksh; } McWl;     /* extinction / scattering per m at the ground */
static double mc_sigt(const SpectralAtmo* a, const McWl* w, double h) { return w->ka * exp(-h / a->H) + w->kh * exp(-h / a->haze_H); }
/* adaptive march along p + s w over [0, send]: returns the integral of sigma_t
 * (mode 0) or stops where it reaches tau_star (mode 1: returns s, > send if not) */
static double mc_march(const SpectralAtmo* a, const McWl* wl, const double p[3], const double w[3], double send, double tau_star,
                       int mode, double* A, double* Hz) {
    double s = 0, tau = 0;
    double q[3] = { p[0], p[1], p[2] };
    double rr = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]), h = rr - a->R;
    double sg0 = mc_sigt(a, wl, h), da0 = exp(-h / a->H), dh0 = exp(-h / a->haze_H);
    double aa = 0, ah = 0;
    while (s < send) {
        const double muloc = (q[0] * w[0] + q[1] * w[1] + q[2] * w[2]) / rr;
        double ds = 0.2 * a->haze_H / fmax(fabs(muloc), 0.01);
        if (ds > 40e3) ds = 40e3;
        if (mode == 1 && sg0 > 0 && ds * sg0 > 0.3) ds = 0.3 / sg0;
        if (ds < 1) ds = 1;
        if (s + ds > send) ds = send - s;
        const double s1 = s + ds;
        const double q1[3] = { p[0] + s1 * w[0], p[1] + s1 * w[1], p[2] + s1 * w[2] };
        const double rr1 = sqrt(q1[0] * q1[0] + q1[1] * q1[1] + q1[2] * q1[2]), h1 = rr1 - a->R;
        const double sg1 = mc_sigt(a, wl, h1), da1 = exp(-h1 / a->H), dh1 = exp(-h1 / a->haze_H);
        const double dt = logmean_int(sg0, sg1, ds);
        if (mode == 1 && tau + dt >= tau_star) {
            const double rem = tau_star - tau;
            const double lam = (sg0 > 0 && sg1 > 0) ? log(sg1 / sg0) / ds : 0;
            double x = fabs(lam) < 1e-12 ? rem / sg0 : log(1 + lam * rem / sg0) / lam;
            if (!(x >= 0 && x <= ds)) x = ds * rem / dt;
            return s + x;
        }
        tau += dt;
        aa += logmean_int(da0, da1, ds);
        ah += logmean_int(dh0, dh1, ds);
        s = s1; q[0] = q1[0]; q[1] = q1[1]; q[2] = q1[2]; rr = rr1; sg0 = sg1; da0 = da1; dh0 = dh1;
    }
    if (A) *A = aa / a->H;
    if (Hz) *Hz = ah / a->haze_H;
    return mode == 1 ? send * 2 + 1 : tau;
}
static double mc_dist(const SpectralAtmo* a, const double p[3], const double w[3], int* ground) {
    const double rr = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    const double h = fmax(rr - a->R, 0), mu = (p[0] * w[0] + p[1] * w[1] + p[2] * w[2]) / rr;
    const double dg = dist_ground(a, h, mu);
    *ground = dg >= 0;
    return dg >= 0 ? dg : dist_top(a, h, mu);
}
static double mc_tsun(const SpectralAtmo* a, const McWl* wl, int k, const double p[3], const double sun[3]) {
    int g;
    const double d = mc_dist(a, p, sun, &g);
    if (g) return 0;
    double A, Hz;
    mc_march(a, wl, p, sun, d, 0, 0, &A, &Hz);
    return exp(-A * a->a_fine[k] - Hz * a->b_fine[k]);
}
static void mc_dir_around(Rng* r, const double w[3], double c, double out[3]) {
    const double s = sqrt(fmax(0, 1 - c * c)), ph = 2 * PI * rng_u(r);
    double t1[3], t2[3];
    if (fabs(w[0]) < 0.9) { t1[0] = 0; t1[1] = -w[2]; t1[2] = w[1]; }
    else { t1[0] = w[2]; t1[1] = 0; t1[2] = -w[0]; }
    const double l = sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
    for (int i = 0; i < 3; ++i) t1[i] /= l;
    t2[0] = w[1] * t1[2] - w[2] * t1[1]; t2[1] = w[2] * t1[0] - w[0] * t1[2]; t2[2] = w[0] * t1[1] - w[1] * t1[0];
    for (int i = 0; i < 3; ++i) out[i] = c * w[i] + s * (cos(ph) * t1[i] + sin(ph) * t2[i]);
}
typedef struct McCtx {
    const SpectralAtmo* a;
    double sun[3], h0;
    const double (*dirs)[3];
    int ndir, npath;
    double* mean;   /* [ndir][NB] */
    double* sem;    /* standard error */
} McCtx;
static void mc_fn(void* p, uint32_t b0, uint32_t e0, uint32_t wk) {
    (void)wk;
    const McCtx* c = (const McCtx*)p;
    const SpectralAtmo* a = c->a;
    for (uint32_t item = b0; item < e0; ++item) {
        const int di = (int)(item / SPEC_NB), band = (int)(item % SPEC_NB);
        Rng rng = { 0x9E3779B97F4A7C15ULL ^ ((uint64_t)item * 0xD1B54A32D192ED03ULL + 12345) };
        double sum = 0, sum2 = 0;
        for (int n = 0; n < c->npath; ++n) {
            const int k = band * SPEC_SUB + (int)(rng_u(&rng) * SPEC_SUB) % SPEC_SUB;
            McWl wl;
            wl.ka = a->a_fine[k] / a->H;
            wl.kh = a->b_fine[k] / a->haze_H;
            wl.ksr = a->r_fine[k] / a->H;
            wl.ksh = a->w_fine[k] * a->b_fine[k] / a->haze_H;
            double pos[3] = { 0, a->R + c->h0, 0 }, w[3] = { c->dirs[di][0], c->dirs[di][1], c->dirs[di][2] };
            double beta = 1, L = 0;
            for (int depth = 0; depth < 200; ++depth) {
                int ground;
                const double send = mc_dist(a, pos, w, &ground);
                const double s = mc_march(a, &wl, pos, w, send, -log(1 - rng_u(&rng)), 1, NULL, NULL);
                if (s >= send) {
                    if (!ground) break;
                    double g[3], nn[3];
                    for (int i = 0; i < 3; ++i) g[i] = pos[i] + send * w[i];
                    const double gl = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
                    for (int i = 0; i < 3; ++i) { nn[i] = g[i] / gl; g[i] = nn[i] * (a->R + 0.01); }
                    const double mus = nn[0] * c->sun[0] + nn[1] * c->sun[1] + nn[2] * c->sun[2];
                    if (mus > 0) L += beta * SPEC_GROUND_ALBEDO / PI * mus * mc_tsun(a, &wl, k, g, c->sun);
                    beta *= SPEC_GROUND_ALBEDO;
                    const double u1 = rng_u(&rng);
                    mc_dir_around(&rng, nn, sqrt(u1), w);
                    for (int i = 0; i < 3; ++i) pos[i] = g[i];
                } else {
                    double y[3];
                    for (int i = 0; i < 3; ++i) y[i] = pos[i] + s * w[i];
                    const double hh = sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]) - a->R;
                    const double sr = wl.ksr * exp(-hh / a->H), sh = wl.ksh * exp(-hh / a->haze_H);
                    const double st = mc_sigt(a, &wl, hh);
                    beta *= (sr + sh) / st;
                    const double cs = w[0] * c->sun[0] + w[1] * c->sun[1] + w[2] * c->sun[2];
                    const double P = (sr * phase_ray(cs) + sh * phase_hg(cs, a->haze_g)) / (sr + sh);
                    L += beta * P * mc_tsun(a, &wl, k, y, c->sun);
                    double cn;                      /* new direction: angle to w from the chosen phase function */
                    if (rng_u(&rng) * (sr + sh) < sr) {
                        const double qq = 4 * rng_u(&rng) - 2, sq = sqrt(qq * qq + 1);
                        cn = cbrt(qq + sq) + cbrt(qq - sq);
                    } else {
                        const double g = a->haze_g, t = (1 - g * g) / (1 - g + 2 * g * rng_u(&rng));
                        cn = (1 + g * g - t * t) / (2 * g);
                    }
                    double nw[3];
                    mc_dir_around(&rng, w, clampd(cn, -1, 1), nw);
                    for (int i = 0; i < 3; ++i) { pos[i] = y[i]; w[i] = nw[i]; }
                }
                if (depth > 3) {                    /* Russian roulette */
                    const double qs = fmin(beta * 4, 1.0);
                    if (rng_u(&rng) >= qs) break;
                    beta /= qs;
                }
            }
            const double v = L * a->e_fine[k];
            sum += v; sum2 += v * v;
        }
        const double m = sum / c->npath;
        c->mean[di * SPEC_NB + band] = m;
        c->sem[di * SPEC_NB + band] = sqrt(fmax(sum2 / c->npath - m * m, 0) / c->npath);
    }
}

/* ---- CIEDE2000 ---- */
static void xyz_to_lab(const double xyz[3], const double wn[3], double lab[3]) {
    double f[3];
    for (int i = 0; i < 3; ++i) {
        const double t = xyz[i] / wn[i];
        f[i] = t > 216.0 / 24389.0 ? cbrt(t) : (24389.0 / 27.0 * t + 16) / 116.0;
    }
    lab[0] = 116 * f[1] - 16; lab[1] = 500 * (f[0] - f[1]); lab[2] = 200 * (f[1] - f[2]);
}
static double de2000(const double l1[3], const double l2[3]) {
    const double C1 = sqrt(l1[1] * l1[1] + l1[2] * l1[2]), C2 = sqrt(l2[1] * l2[1] + l2[2] * l2[2]);
    const double Cb = 0.5 * (C1 + C2), Cb7 = pow(Cb, 7), G = 0.5 * (1 - sqrt(Cb7 / (Cb7 + pow(25, 7))));
    const double a1 = (1 + G) * l1[1], a2 = (1 + G) * l2[1];
    const double c1 = sqrt(a1 * a1 + l1[2] * l1[2]), c2 = sqrt(a2 * a2 + l2[2] * l2[2]);
    double h1 = atan2(l1[2], a1), h2 = atan2(l2[2], a2);
    if (h1 < 0) h1 += 2 * PI;
    if (h2 < 0) h2 += 2 * PI;
    const double dL = l2[0] - l1[0], dC = c2 - c1;
    double dh = 0;
    if (c1 * c2 != 0) { dh = h2 - h1; if (dh > PI) dh -= 2 * PI; else if (dh < -PI) dh += 2 * PI; }
    const double dH = 2 * sqrt(c1 * c2) * sin(dh / 2);
    const double Lb = 0.5 * (l1[0] + l2[0]), cb = 0.5 * (c1 + c2);
    double hb = h1 + h2;
    if (c1 * c2 != 0) { if (fabs(h1 - h2) > PI) hb += (h1 + h2 < 2 * PI) ? 2 * PI : -2 * PI; hb *= 0.5; }
    const double T = 1 - 0.17 * cos(hb - PI / 6) + 0.24 * cos(2 * hb) + 0.32 * cos(3 * hb + PI / 30) - 0.20 * cos(4 * hb - 63 * PI / 180);
    const double dth = 30 * PI / 180 * exp(-pow((hb * 180 / PI - 275) / 25, 2));
    const double cb7 = pow(cb, 7), RC = 2 * sqrt(cb7 / (cb7 + pow(25, 7)));
    const double SL = 1 + 0.015 * (Lb - 50) * (Lb - 50) / sqrt(20 + (Lb - 50) * (Lb - 50));
    const double SC = 1 + 0.045 * cb, SH = 1 + 0.015 * cb * T, RT = -sin(2 * dth) * RC;
    return sqrt(pow(dL / SL, 2) + pow(dC / SC, 2) + pow(dH / SH, 2) + RT * (dC / SC) * (dH / SH));
}
/* leaf of section 5.1 (hue bin i, pigment 0..255): reflectance at 1 nm */
static double leaf_rho(double l, int bin, double pig) { return spectral_leaf_rho_nm(l, bin, pig); }
/* the 1 nm "truth" of a band-table material: linear between band centres */
static double band_interp(const float* v, double l) {
    const double f = (l - (SPEC_L0 + 0.5 * SPEC_BW)) / SPEC_BW;
    if (f <= 0) return v[0];
    if (f >= SPEC_NB - 1) return v[SPEC_NB - 1];
    const int i = (int)f;
    return v[i] + (v[i + 1] - v[i]) * (f - i);
}
static double tau_1nm(const SpectralAtmo* a, double l, double A, double Hz) {
    return A * (a->N * sigma_air(l) * a->ray_factor + ch4_k(l) * a->ch4_col) + Hz * a->haze_tau550 * pow(l / 550.0, -2.0);
}
/* Delta E 2000 between the 1 nm product and the renderer's representation
 * (g-points for light, band tables for surfaces and water); the white is the
 * illuminant (noon direct sun) in each representation */
static double band_error(const SpectralAtmo* a, int kind) {
    double x1[3] = { 0, 0, 0 }, w1[3] = { 0, 0, 0 }, x8[3] = { 0, 0, 0 }, w8[3] = { 0, 0, 0 };
    const float* rtab = kind == 2 ? a->mat_rho[MAT_GRANITE] : a->mat_rho[MAT_SAND];
    double rb[SPEC_NB];
    for (int b = 0; b < SPEC_NB; ++b) {
        rb[b] = 0;
        for (int i = 0; i < 45; ++i) {
            const double l = SPEC_L0 + SPEC_BW * b + i + 0.5;
            const double E = etop_nm(a, l) * exp(-tau_1nm(a, l, 1, 1));
            double rho = kind == 0 ? leaf_rho(l, 8, 200) : band_interp(rtab, l);
            rb[b] += rho / 45.0;
            if (kind == 1) rho *= exp(-water_a(l) * 20.0);                                     /* 10 m down and up */
            if (kind == 2) rho *= exp(-tau_1nm(a, l, 8000.0 / a->H, 8000.0 / a->haze_H));      /* 8 km near the ground */
            double c[3];
            cie_at(l, c);
            for (int k = 0; k < 3; ++k) { x1[k] += E * rho * c[k]; w1[k] += E * c[k]; }
        }
    }
    for (int g = 0; g < SPEC_NG; ++g) {
        const int b = g / SPEC_GPB;
        const double E = a->g_e[g] * GT(a, g, 1, 1);
        double r8 = rb[b];
        if (kind == 1) r8 *= water_T_lookup(a, b, 20.0);
        if (kind == 2) r8 *= GT(a, g, 8000.0 / a->H, 8000.0 / a->haze_H);
        for (int k = 0; k < 3; ++k) { x8[k] += E * r8 * a->g_cmf[g][k]; w8[k] += E * a->g_cmf[g][k]; }
    }
    double n1[3], n8[3], l1[3], l8[3], y1[3], y8[3];
    for (int k = 0; k < 3; ++k) { n1[k] = w1[k] / w1[1]; n8[k] = w8[k] / w8[1]; y1[k] = x1[k] / w1[1]; y8[k] = x8[k] / w8[1]; }
    xyz_to_lab(y1, n1, l1);
    xyz_to_lab(y8, n8, l8);
    return de2000(l1, l8);
}

static int d3_check(const char* name, int ok, const char* fmt, ...);
#include <stdarg.h>
static int d3_check(const char* name, int ok, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("   %-34s %s  ", name, ok ? "ok  " : "FAIL");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    return ok ? 0 : 1;
}

int spectral_selftest_d3(const HomePlanet* hp, const SpectralOpts* opt, SpectralGpuSkyFn gpu_sky, void* ctx) {
    int fails = 0;
    SpectralOpts o;
    memset(&o, 0, sizeof o);
    if (opt) o = *opt;
    SpectralAtmo* a = (SpectralAtmo*)calloc(1, sizeof(SpectralAtmo));
    SpectralAtmo* e = (SpectralAtmo*)calloc(1, sizeof(SpectralAtmo));
    if (!a || !e) { free(a); free(e); return 1; }
    printf("D3 physics:\n");
    /* 1. Planck vs sigma T^4 */
    {
        double worst = 0;
        const double Ts[2] = { hp && hp->star_teff > 1000 ? hp->star_teff : 5772, 5772 };
        for (int t = 0; t < 2; ++t) {
            double s = 0;
            const int n = 200000;
            const double l0 = 50, l1 = 1e6;           /* nm, log steps */
            for (int i = 0; i < n; ++i) {
                const double la = l0 * pow(l1 / l0, i / (double)n), lb = l0 * pow(l1 / l0, (i + 1.0) / n);
                const double lm = sqrt(la * lb);
                s += planck_nm(lm, Ts[t]) * (lb - la);
            }
            const double rel = fabs(PI * s / (SIGMA_SB * pow(Ts[t], 4)) - 1);
            if (rel > worst) worst = rel;
        }
        fails += d3_check("Planck vs sigma T^4", worst <= 0.005, "max deviation %.4f%% (gate 0.5%%) at %.0f K and 5772 K", 100 * worst, Ts[0]);
    }
    /* 2. Earth: tau(550) and a sky bluer at the zenith than at the horizon */
    {
        SpectralOpts eo = o;
        eo.earth = 1;
        eo.quiet = 1;
        spectral_build_atmo_ex(hp, &eo, e);
        const double sun[3] = { cos(45 * PI / 180), sin(45 * PI / 180), 0 };
        const double zen[3] = { 0, 1, 0 }, hor[3] = { 0, sin(3 * PI / 180), cos(3 * PI / 180) };
        double Lz[SPEC_NG], Lh[SPEC_NG], xz[3] = { 0, 0, 0 }, xh[3] = { 0, 0, 0 };
        spectral_sky_radiance(e, sun, 10, zen, Lz, NULL);
        spectral_sky_radiance(e, sun, 10, hor, Lh, NULL);
        for (int g = 0; g < SPEC_NG; ++g) for (int c = 0; c < 3; ++c) { xz[c] += Lz[g] * e->g_cmf[g][c]; xh[c] += Lh[g] * e->g_cmf[g][c]; }
        const double cz = xz[0] / (xz[0] + xz[1] + xz[2]), ch = xh[0] / (xh[0] + xh[1] + xh[2]);
        const double yz = xz[1] / (xz[0] + xz[1] + xz[2]), yh = xh[1] / (xh[0] + xh[1] + xh[2]);
        const int ok = fabs(e->tau_ray550 - 0.097) <= 0.005 && cz < ch && yz < yh;
        fails += d3_check("Earth tau(550), zenith bluer", ok, "tau_R(550) %.4f (gate 0.097 +- 0.005); sky xy zenith (%.3f, %.3f) vs horizon (%.3f, %.3f)",
                          e->tau_ray550, cz, yz, ch, yh);
    }
    /* the planet itself (also checks the build) */
    spectral_build_atmo_ex(hp, &o, a);
    /* 3. equal energy -> x, y = 1/3 */
    {
        double X = 0, Y = 0, Z = 0, Xw = 0, Yw = 0, Zw = 0;
        for (int b = 0; b < SPEC_NB; ++b) { X += a->cmf[b][0]; Y += a->cmf[b][1]; Z += a->cmf[b][2]; Xw += a->cmf_wyman[b][0]; Yw += a->cmf_wyman[b][1]; Zw += a->cmf_wyman[b][2]; }
        const double x = X / (X + Y + Z), y = Y / (X + Y + Z);
        double wdev = 0;
        for (int b = 0; b < SPEC_NB; ++b)
            for (int c = 0; c < 3; ++c)
                if (a->cmf[b][c] > 1.0) { const double d = fabs(a->cmf_wyman[b][c] / a->cmf[b][c] - 1); if (d > wdev) wdev = d; }
        fails += d3_check("equal energy -> x,y = 1/3", fabs(x - 1.0 / 3) <= 0.002 && fabs(y - 1.0 / 3) <= 0.002,
                          "x %.4f y %.4f (gate +-0.002); X Y Z = %.2f %.2f %.2f; Wyman fit %.2f %.2f %.2f, max band deviation %.1f%%",
                          x, y, X, Y, Z, Xw, Yw, Zw, 100 * wdev);
    }
    /* 4. BRDF directional albedo <= 1 */
    {
        double worst = 0, furnace = 0, fmin_ = 2;
        int wm = 0;
        const int mats[] = { MAT_STONE, MAT_DIRT, MAT_GRASS, MAT_SAND, MAT_GRANITE, MAT_IRON_ORE, MAT_SULFIDE, MAT_ICE,
                             MAT_REGOLITH, MAT_MAT, MAT_LEAF, MAT_WOOD, MAT_KELP };
        for (size_t mi = 0; mi < sizeof mats / sizeof mats[0]; ++mi) {
            const int m = mats[mi], ior = a->mat_ior[m];
            const double alpha = a->mat_alpha[m];
            double rmax = 0;
            for (int b = 0; b < SPEC_NB; ++b) if (a->mat_rho[m][b] > rmax) rmax = a->mat_rho[m][b];
            const double ea = eavg_lookup(a, ior, alpha);
            /* cosine-weighted integral of (1 - E_lut(mu_i)) over the hemisphere, by quadrature */
            double ci = 0;
            for (int k = 0; k < 512; ++k) { const double mu = (k + 0.5) / 512; ci += 2 * (1 - e_lookup(a, ior, mu, alpha)) * mu / 512; }
            for (int k = 0; k < 32; ++k) {
                const double mu = (k + 0.5) / 32;
                double Es;
                if (m == MAT_SULFIDE) {      /* conductor: Schlick with the largest band F0, own MC */
                    const double f0 = a->pyrite_f0[SPEC_NB - 1];
                    {   /* VNDF estimator with Schlick */
                        const int n = 64;
                        double s = 0;
                        const double V[3] = { sqrt(1 - mu * mu), 0, mu };
                        const double a2 = alpha * alpha, lv = (-1 + sqrt(1 + a2 * (1 - mu * mu) / (mu * mu))) * 0.5;
                        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
                            const double u1 = (i + 0.5) / n, u2 = (j + 0.5) / n;
                            double Vh[3] = { alpha * V[0], alpha * V[1], V[2] };
                            const double vl = sqrt(Vh[0] * Vh[0] + Vh[1] * Vh[1] + Vh[2] * Vh[2]);
                            for (int q = 0; q < 3; ++q) Vh[q] /= vl;
                            const double ls = Vh[0] * Vh[0] + Vh[1] * Vh[1];
                            double T1[3] = { 1, 0, 0 };
                            if (ls > 0) { const double il = 1 / sqrt(ls); T1[0] = -Vh[1] * il; T1[1] = Vh[0] * il; T1[2] = 0; }
                            const double T2[3] = { Vh[1] * T1[2] - Vh[2] * T1[1], Vh[2] * T1[0] - Vh[0] * T1[2], Vh[0] * T1[1] - Vh[1] * T1[0] };
                            const double r = sqrt(u1), ph = 2 * PI * u2, t1 = r * cos(ph), sw = 0.5 * (1 + Vh[2]);
                            const double t2 = (1 - sw) * sqrt(fmax(0, 1 - t1 * t1)) + sw * r * sin(ph);
                            const double tz = sqrt(fmax(0, 1 - t1 * t1 - t2 * t2));
                            double Ne[3];
                            for (int q = 0; q < 3; ++q) Ne[q] = t1 * T1[q] + t2 * T2[q] + tz * Vh[q];
                            Ne[0] *= alpha; Ne[1] *= alpha; Ne[2] = fmax(Ne[2], 0);
                            const double nl = sqrt(Ne[0] * Ne[0] + Ne[1] * Ne[1] + Ne[2] * Ne[2]);
                            for (int q = 0; q < 3; ++q) Ne[q] /= nl;
                            const double vn = V[0] * Ne[0] + V[1] * Ne[1] + V[2] * Ne[2];
                            const double Lz = 2 * vn * Ne[2] - V[2];
                            if (Lz <= 0) continue;
                            const double ll = (-1 + sqrt(1 + a2 * (1 - Lz * Lz) / (Lz * Lz))) * 0.5;
                            s += spec_schlick((float)f0, (float)vn) * (1 + lv) / (1 + lv + ll);
                        }
                        Es = s / ((double)n * n);
                    }
                } else {
                    Es = ggx_albedo(mu, alpha, IOR_N[ior], 64);
                }
                const double el = e_lookup(a, ior, mu, alpha);
                const double alb = Es + rmax * (1 - el) * ci / (1 - ea);
                const double fur = Es + 1.0 * (1 - el) * ci / (1 - ea);
                if (alb > worst) { worst = alb; wm = m; }
                if (m != MAT_SULFIDE) { if (fur > furnace) furnace = fur; if (fur < fmin_) fmin_ = fur; }
            }
        }
        fails += d3_check("BRDF directional albedo <= 1", worst <= 1.0,
                          "max %.4f (material %d) over 13 materials x 32 angles; furnace (rho = 1) %.4f .. %.4f", worst, wm, fmin_, furnace);
    }
    /* 5. sky-view model vs Monte Carlo */
    {
        const double elev[3] = { 60, 20, 5 };
        const double cam_h = 50;
        double dirs[8][3];
        const double de[8] = { 90, 45, 45, 45, 10, 10, 10, 3 }, da[8] = { 0, 30, 90, 180, 45, 120, 180, 90 };
        for (int i = 0; i < 8; ++i) {
            const double el = de[i] * PI / 180, az = da[i] * PI / 180;
            dirs[i][0] = cos(el) * cos(az); dirs[i][1] = sin(el); dirs[i][2] = cos(el) * sin(az);
        }
        double worst = 0, worst_gpu = 0, mc_sem = 0;
        int wb = 0, we = 0, wd = 0, ncomp = 0, nskip = 0;
        const int npath = 30000;
        double t0 = 0;
        (void)t0;
        double mean[8 * SPEC_NB], sem[8 * SPEC_NB];
        printf("   sky-view vs Monte Carlo (camera 50 m; per band: model / MC - 1 in %%; 5 nm MC, %d paths per value; "
               "'~' = MC std error > 3%%, not compared):\n", npath);
        for (int ie = 0; ie < 3; ++ie) {
            const double sun[3] = { cos(elev[ie] * PI / 180), sin(elev[ie] * PI / 180), 0 };
            McCtx mc = { a, { sun[0], sun[1], sun[2] }, cam_h, (const double(*)[3])dirs, 8, npath, mean, sem };
            jobs_parallel_for(8 * SPEC_NB, 1, mc_fn, &mc);
            float gsun[3] = { (float)sun[0], (float)sun[1], (float)sun[2] }, gdirs[8 * 3], gout[8 * SPEC_NB];
            for (int i = 0; i < 8; ++i) for (int k = 0; k < 3; ++k) gdirs[i * 3 + k] = (float)dirs[i][k];
            const int have_gpu = gpu_sky && !gpu_sky(ctx, a, gsun, (float)cam_h, gdirs, 8, gout);
            for (int i = 0; i < 8; ++i) {
                double L[SPEC_NB];
                spectral_sky_radiance(a, sun, cam_h, dirs[i], NULL, L);
                printf("     sun %2.0f deg, view el %2.0f az %3.0f:", elev[ie], de[i], da[i]);
                for (int b = 0; b < SPEC_NB; ++b) {
                    const double m = mean[i * SPEC_NB + b];
                    if (m <= 1e-12 * a->e_nm[b] || sem[i * SPEC_NB + b] > 0.03 * m) { printf("     ~"); ++nskip; continue; }
                    const double d = L[b] / m - 1;
                    printf(" %+5.1f", 100 * d);
                    ++ncomp;
                    if (fabs(d) > worst) { worst = fabs(d); wb = b; we = ie; wd = i; }
                    if (sem[i * SPEC_NB + b] / m > mc_sem) mc_sem = sem[i * SPEC_NB + b] / m;
                    if (have_gpu && L[b] > 0) { const double g = fabs(gout[i * SPEC_NB + b] / L[b] - 1); if (g > worst_gpu) worst_gpu = g; }
                }
                printf("\n");
            }
        }
        fails += d3_check("sky-view vs Monte Carlo", worst <= 0.10 && nskip * 6 <= ncomp + nskip,
                          "max |model/MC - 1| %.1f%% (gate 10%%) at sun %.0f deg, view %d, band %d; MC std error <= %.1f%%; "
                          "%d values compared, %d too noisy",
                          100 * worst, elev[we], wd, wb, 100 * mc_sem, ncomp, nskip);
        if (gpu_sky)
            fails += d3_check("GPU sky-view LUT vs host twin", worst_gpu <= 0.03, "max |GPU/host - 1| %.2f%% (gate 3%%)", 100 * worst_gpu);
    }
    /* 6. haze calibration residual */
    fails += d3_check("haze calibration residual", a->haze_residual <= 0.005,
                      "|T_PAR - haze_q12/4096| = %.5f (gate 0.005): tau_h550 %.3f, T_PAR %.4f, target %.4f", a->haze_residual,
                      a->haze_tau550, a->haze_achieved, a->haze_target);
    /* 7. water band transmittance at 10 m vs 1 nm integration */
    {
        double worst = 0;
        char buf[256];
        int w = 0;
        for (int b = 0; b < SPEC_NB; ++b) {
            const double t = water_T_lookup(a, b, 10.0), r = water_T_1nm(b, 10.0);
            const double d = r > 1e-6 ? fabs(t / r - 1) : fabs(t - r);
            if (d > worst) worst = d;
            w += snprintf(buf + w, sizeof buf - (size_t)w, " %.3f", r);
        }
        fails += d3_check("water T_b(10 m) vs 1 nm", worst <= 0.01, "max deviation %.3f%% (gate 1%%); T_b(10 m) =%s", 100 * worst, buf);
    }
    /* 8. Snell window from below */
    {
        double crit = 90;
        for (int i = 0; i <= 9000; ++i) {
            const double th = i * 0.01 * PI / 180;
            const float d[3] = { (float)sin(th), (float)cos(th), 0.f }, n[3] = { 0.f, -1.f, 0.f };
            float t[3];
            if (!spec_refract(d, n, 1.333f, t)) { crit = i * 0.01; break; }
        }
        fails += d3_check("Snell window", fabs(crit - 48.6) <= 0.5, "critical angle %.2f deg (gate 48.6 +- 0.5)", crit);
    }
    /* 9. band error */
    {
        const double d0 = band_error(a, 0), d1 = band_error(a, 1), d2 = band_error(a, 2);
        const double m = fmax(d0, fmax(d1, d2));
        fails += d3_check("band error Delta E2000 <= 3", m <= 3.0, "leaf x noon sun %.2f, seabed at 10 m %.2f, terrain at 8 km %.2f",
                          d0, d1, d2);
    }
    printf("D3 %s\n", fails ? "FAIL" : "PASS");
    free(a); free(e);
    return fails;
}

/* ==== M5: pigment table, the bio's light file (GMESPEC1), D5 ======================================== */
#include "spectrum_file.h"

/* spec 5.1 reference: leaf colour per hue bin (equal-energy light, von Kries to D65, sRGB); 0 = the default bin */
static const int PIG_REF[SPEC_NHUE][3] = {
    { 81, 118, 49 },  { 127, 117, 46 }, { 129, 114, 43 }, { 136, 97, 45 },  { 133, 70, 52 },  { 106, 58, 57 },
    { 55, 82, 58 },   { 27, 107, 53 },  { 81, 118, 49 },  { 114, 118, 47 }, { 125, 117, 47 }, { 127, 117, 46 },
    { 127, 117, 46 }, { 127, 117, 46 }, { 127, 117, 46 }, { 127, 117, 46 } };

static double srgb_oetf_d(double c) {
    c = clampd(c, 0, 1);
    return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1 / 2.4) - 0.055;
}
/* XYZ of a reflectance under an equal-energy white (white Y = 1) -> von Kries scaling in XYZ to D65 -> 8-bit sRGB */
static void ee_to_srgb8(const double xyz[3], const double E[3], int rgb[3], double lab[3]) {
    double t[3], lin[3];
    for (int c = 0; c < 3; ++c) t[c] = xyz[c] / (E[c] / E[1]) * D65[c];
    mat3_mul_v(XYZ2RGB, t, lin);
    for (int c = 0; c < 3; ++c) rgb[c] = (int)floor(srgb_oetf_d(lin[c]) * 255.0 + 0.5);
    if (lab) xyz_to_lab(t, D65, lab);
}

int spectral_pigment_table(const SpectralAtmo* a, int ref_rgb[SPEC_NHUE][3], int band_rgb[SPEC_NHUE][3], double de[SPEC_NHUE],
                           int spec_rgb[SPEC_NHUE][3]) {
    double E1[3] = { 0, 0, 0 }, E8[3] = { 0, 0, 0 };
    for (int i = 0; i < 400; ++i) {
        double c[3];
        cie_at(380.5 + i, c);
        for (int k = 0; k < 3; ++k) E1[k] += c[k];
    }
    for (int b = 0; b < SPEC_NB; ++b) for (int k = 0; k < 3; ++k) E8[k] += a->cmf[b][k];
    int worst = 0;
    for (int h = 0; h < SPEC_NHUE; ++h) {
        double x1[3] = { 0, 0, 0 }, x8[3] = { 0, 0, 0 }, l1[3], l8[3];
        for (int i = 0; i < 400; ++i) {
            const double l = 380.5 + i;
            double c[3];
            cie_at(l, c);
            const double r = spectral_leaf_rho_nm(l, h, SPEC_PIG_REF);
            for (int k = 0; k < 3; ++k) x1[k] += r * c[k];
        }
        for (int b = 0; b < SPEC_NB; ++b) for (int k = 0; k < 3; ++k) x8[k] += a->leaf_rho[h][b] * a->cmf[b][k];
        for (int k = 0; k < 3; ++k) { x1[k] /= E1[1]; x8[k] /= E8[1]; }
        ee_to_srgb8(x1, E1, ref_rgb[h], l1);
        ee_to_srgb8(x8, E8, band_rgb[h], l8);
        de[h] = de2000(l1, l8);
        for (int c = 0; c < 3; ++c) {
            spec_rgb[h][c] = PIG_REF[h][c];
            const int d = abs(ref_rgb[h][c] - PIG_REF[h][c]);
            if (h >= 1 && d > worst) worst = d;
        }
    }
    return worst;
}

void spectral_lut_global(const SpectralAtmo* a, double mu, double* glob, double* par, double* par_top) {
    double A, Hz, g = 0, p = 0, pt = 0;
    spectral_columns(a, 0, mu, &A, &Hz);
    for (int k = 0; k < SPEC_NG; ++k) {
        const double e = (mu > 0 ? GT(a, k, A, Hz) * mu : 0) + gd_val(a, k, mu);
        g += a->g_e[k] * e;
        p += a->g_par[k] * e;
        pt += a->g_par[k] * fmax(mu, 0);
    }
    *glob = g;
    *par = p;
    *par_top = pt;
}

/* NIR (740-900 nm) g-points of an atmosphere for the export, written into the g-point
 * arrays of t (a scratch copy of a): 32 samples at 5 nm sorted by their air-profile
 * extinction (Rayleigh + CH4), 8 groups of 4, each grey (as setup_fine's: the air tau
 * is exact at 2 air masses). Rayleigh is weak there; CH4 decides. */
#define NIR_NS 32
#define NIR_NG 8
static void nir_gpoints(const SpectralAtmo* a, SpectralAtmo* t) {
    double e[NIR_NS], r[NIR_NS], hb[NIR_NS], w[NIR_NS], x[NIR_NS];
    int idx[NIR_NS];
    for (int k = 0; k < NIR_NS; ++k) {
        const double l = 742.5 + 5.0 * k;
        e[k] = etop_nm(a, l) * 5.0;
        r[k] = a->N * sigma_air(l) * a->ray_factor;
        hb[k] = a->haze_tau550 * pow(l / 550.0, -2.0);
        w[k] = 0.95 - 0.45 * exp(-(l - 380.0) / 90.0);
        x[k] = r[k] + ch4_k(l) * a->ch4_col;
        idx[k] = k;
    }
    for (int i = 1; i < NIR_NS; ++i)
        for (int j = i; j > 0 && x[idx[j]] < x[idx[j - 1]]; --j) { const int q = idx[j]; idx[j] = idx[j - 1]; idx[j - 1] = q; }
    for (int g = 0; g < SPEC_NG; ++g) {
        t->g_e[g] = 0; t->g_par[g] = 0; t->g_a[g] = 0; t->g_b[g] = 0; t->g_sr[g] = 0; t->g_sh[g] = 0;
        for (int c = 0; c < 3; ++c) t->g_cmf[g][c] = 0;
    }
    const int per = NIR_NS / NIR_NG;
    for (int g = 0; g < NIR_NG; ++g) {
        double ws = 0, ta = 0, tb = 0, sr = 0, sh = 0;
        for (int m = 0; m < per; ++m) {
            const int k = idx[g * per + m];
            ws += e[k]; ta += e[k] * exp(-2.0 * x[k]); tb += e[k] * hb[k]; sr += e[k] * r[k]; sh += e[k] * w[k] * hb[k];
        }
        t->g_e[g] = ws;
        t->g_a[g] = -0.5 * log(fmax(ta / ws, 1e-300));
        t->g_b[g] = tb / ws;
        t->g_sr[g] = sr / ws / a->H;
        t->g_sh[g] = sh / ws / a->haze_H;
    }
}

/* the bio's 24 hours: sun elevation sine bio_sin24(hr) / 4096 (bio_core.h's table) */
static const int SIN24_Q12[24] = { 0, 1060, 2048, 2896, 3547, 3956, 4096, 3956, 3547, 2896, 2048, 1060,
                                   0, -1060, -2048, -2896, -3547, -3956, -4096, -3956, -3547, -2896, -2048, -1060 };
static double sin24(int hr) { return SIN24_Q12[(hr + 18) % 24] / 4096.0; }
static double gd_interp(const double* gd, double mus) {
    const double fx = clampd((mus - SPEC_MS_MU0) / (SPEC_MS_MU1 - SPEC_MS_MU0), 0, 1) * (SPEC_MS_W - 1);
    int j = (int)fx;
    if (j > SPEC_MS_W - 2) j = SPEC_MS_W - 2;
    const double u = fx - j;
    return (gd[j] * (1 - u) + gd[j + 1] * u) * spec_ms_fade((float)mus);
}
static int32_t q16d(double v) { return (int32_t)llround(v * 65536.0); }
static double water_T_nir(double d) {           /* 740-900 nm band mean of e^(-a d) at 1 nm */
    double s = 0;
    for (int i = 0; i < 160; ++i) s += exp(-water_a(740.5 + i) * d);
    return s / 160.0;
}

/* GMESPEC1: one atmosphere per haze bin. The bio's haze factor is a function of
 * the CH4 mole fraction alone (haze_q12 = 4096 - 2048 x / (x + 0.2)), so bin k
 * is the planet's atmosphere with x = 0.2 r / (1 - r), r = (4096 - haze_q12) /
 * 2048 (capped at pure CH4; the other gases rescaled to 1 - x), and the haze
 * calibrated to that bin's PAR transmission haze_q12 / 4096 (spec 4.3; none
 * where Rayleigh + gas already transmit less). Each bin: the LUT model's
 * global horizontal irradiance (direct through the path-column LUT + the
 * multiple-scattering ground irradiance, ground albedo 0.15) at the bio's 24
 * sun elevations, per render band (g-points) and for NIR (8 NIR g-points). */
int spectral_export(const HomePlanet* hp, const SpectralOpts* opt, void* out, SpectralExportInfo* info) {
    SpectrumFile* f = (SpectrumFile*)out;
    SpectralExportInfo dummy;
    if (!info) info = &dummy;
    memset(f, 0, sizeof *f);
    memset(info, 0, sizeof *info);
    const double t0 = time_now();
    SpectralOpts o;
    memset(&o, 0, sizeof o);
    if (opt) o = *opt;
    o.earth = 0;
    o.quiet = 1;
    o.ms_j0 = 8;                                       /* ms_mu(8) = -0.0016: every sun at or above the horizon */
    SpectralAtmo* a = (SpectralAtmo*)calloc(1, sizeof(SpectralAtmo));
    SpectralAtmo* t = (SpectralAtmo*)calloc(1, sizeof(SpectralAtmo));
    double* msbuf = (double*)malloc(sizeof(double) * SPEC_MS_H * SPEC_MS_W);
    double* gdn = (double*)malloc(sizeof(double) * NIR_NG * SPEC_MS_W);
    int rc = (!a || !t || !msbuf || !gdn) ? -1 : 0;
    f->planet_hash = planet_hash(hp);
    for (int b = 0; b < SPECF_NB; ++b) f->band_edge_nm[b] = (int32_t)(SPEC_L0 + SPEC_BW * b);
    f->band_edge_nm[SPECF_NB] = 900;
    const double aa = hp->a_au > 1e-6 ? hp->a_au : 1.0;
    f->s_top_wm2 = (int32_t)llround(1361.0 * hp->star_lum / (aa * aa));   /* as bio_env_planet */
    f->t_surf_ck = (int32_t)llround(hp->t_surf * 100.0);
    const double S = f->s_top_wm2 > 0 ? (double)f->s_top_wm2 : 1361.0;
    double base[ATM_N];
    if (hp->atm_known) for (int g = 0; g < ATM_N; ++g) base[g] = hp->atm_mix[g];
    else {
        for (int g = 0; g < ATM_N; ++g) base[g] = 0;
        base[ATM_N2] = 0.7808 + 0.0093; base[ATM_O2] = 0.2095; base[ATM_CO2] = 0.0004;
    }
    double rest = 0;
    for (int g = 0; g < ATM_N; ++g) if (g != ATM_CH4) rest += base[g];
    for (int k = 0; k < SPECF_NHAZE && !rc; ++k) {
        const int q = 4096 - 256 * k;
        const double rr = (4096.0 - q) / 2048.0;
        double x = rr < 1 ? 0.2 * rr / (1 - rr) : 1.0;
        if (x > 1) x = 1;
        HomePlanet hk = *hp;
        for (int g = 0; g < ATM_N; ++g) hk.atm_mix[g] = g == ATM_CH4 ? x : (rest > 0 ? base[g] * (1 - x) / rest : (g == ATM_N2 ? 1 - x : 0));
        hk.atm_known = 1;
        o.haze_q12 = q;
        if (spectral_build_atmo_ex(&hk, &o, a)) { rc = -1; break; }
        info->haze_q12[k] = q;
        info->x_ch4[k] = a->x[ATM_CH4];
        info->ch4_col[k] = a->ch4_col;
        info->tau_h550[k] = a->haze_tau550;
        info->target[k] = a->haze_target;
        info->t_par[k] = a->haze_achieved;
        info->unreachable[k] = a->haze_iters < 0 || a->haze_achieved < a->haze_target - 0.001;
        memcpy(t, a, sizeof *t);
        t->host_owned = 0;                             /* shares a's column LUT; never freed through t */
        nir_gpoints(a, t);
        for (int g = 0; g < NIR_NG; ++g) {
            double qq;
            build_ms_g(t, g, a->ms_method, o.ms_j0, msbuf, gdn + g * SPEC_MS_W, &qq);
        }
        for (int hr = 0; hr < SPECF_NHOUR; ++hr) {
            const double mu = sin24(hr);
            if (mu < 0) continue;                      /* night: 0 */
            double A, Hz, dir[SPECF_NB], glob[SPECF_NB];
            spectral_columns(a, 0, mu, &A, &Hz);
            for (int b = 0; b < SPECF_NB; ++b) dir[b] = glob[b] = 0;
            for (int g = 0; g < SPEC_NG; ++g) {
                const double dd = mu > 0 ? a->g_e[g] * GT(a, g, A, Hz) * mu : 0, df = a->g_e[g] * gd_val(a, g, mu);
                dir[g / SPEC_GPB] += dd;
                glob[g / SPEC_GPB] += dd + df;
            }
            for (int g = 0; g < NIR_NG; ++g) {
                const double dd = mu > 0 ? t->g_e[g] * GT(t, g, A, Hz) * mu : 0, df = t->g_e[g] * gd_interp(gdn + g * SPEC_MS_W, mu);
                dir[SPEC_NB] += dd;
                glob[SPEC_NB] += dd + df;
            }
            for (int b = 0; b < SPECF_NB; ++b) {
                f->surf_q16[k][hr][b] = q16d(glob[b] / S);
                f->dir_q16[k][hr][b] = q16d(dir[b] / S);
            }
        }
    }
    if (!rc) {
        /* water: band means at 1 nm (pure water + CDOM + particles, as the renderer), 0..32 m; the effective
         * attenuation over 0-20 m is the least-squares slope of -ln T(d) */
        for (int b = 0; b < SPECF_NB; ++b) {
            double sdd = 0, sdl = 0;
            for (int d = 0; d < SPECF_NDEPTH; ++d) {
                const double T = b < SPEC_NB ? water_T_1nm(b, d) : water_T_nir(d);
                f->water_T_q16[b][d] = q16d(T);
                if (d >= 1 && d <= 20) { sdd += (double)d * d; sdl += d * -log(fmax(T, 1e-300)); }
            }
            f->kw_q16[b] = q16d(sdl / sdd);
        }
        /* photons: the star's spectrum per band; uphill: a photon below the reaction centre's energy
         * is used with the Boltzmann factor of the missing energy (kT of the surface) */
        const double kT = 8.617333262e-5 * a->t_surf, erc[2] = { 1239.84198 / 680.0, 1239.84198 / 870.0 };
        for (int b = 0; b < SPECF_NB; ++b) {
            f->phot_q16[b] = q16d(a->e_band[b] > 0 ? a->phot_band[b] / a->e_band[b] : 0);
            f->ev_q16[b] = q16d(a->ev_band[b]);
            for (int p = 0; p < 2; ++p) f->uphill_q16[b][p] = q16d(fmin(1.0, exp(-(erc[p] - a->ev_band[b]) / kT)));
        }
        /* pigments: the effective eps of each band, exact for the band-mean absorptance at PIG 200:
         * A_b = 1 - 10^(-PIG / 255 x 4 x eps); a reader's expn_q16 index is (PIG x eps_q12 x 37) >> 16 = 147.4 OD */
        for (int h = 0; h < SPECF_NHUE; ++h)
            for (int b = 0; b < SPECF_NB; ++b) {
                const double A = b < SPEC_NB ? a->pig_A[h][b] : a->pig_A_nir[h];
                const double e = -log10(fmax(1.0 - A, 1e-12)) / (SPEC_PIG_REF / 255.0 * 4.0);
                f->eps_q12[h][b] = (int32_t)llround(e * 4096.0);
            }
    }
    info->seconds = time_now() - t0;
    if (a) spectral_free(a);
    free(a); free(t); free(msbuf); free(gdn);
    return rc;
}

/* D5 (spec 7.3): the export is reproducible byte for byte; at the planet's own haze (the bio's
 * interpolation of the table) its noon global irradiance equals the renderer's LUT model (+-1 %) and
 * its PAR transmission the bio's haze factor (+-0.005); the pigment table is the spec's (+-3 levels). */
int spectral_selftest_d5(const HomePlanet* hp, const SpectralOpts* opt) {
    int fails = 0;
    printf("D5 export:\n");
    SpectrumFile* f1 = (SpectrumFile*)calloc(1, sizeof(SpectrumFile));
    SpectrumFile* f2 = (SpectrumFile*)calloc(1, sizeof(SpectrumFile));
    SpectrumFile* f3 = (SpectrumFile*)calloc(1, sizeof(SpectrumFile));
    const size_t nb = SPECF_HDR + SPECF_PAYLOAD_BYTES;
    uint8_t* b1 = (uint8_t*)malloc(nb);
    uint8_t* b2 = (uint8_t*)malloc(nb);
    uint8_t* b3 = (uint8_t*)malloc(nb);
    SpectralAtmo* a = (SpectralAtmo*)calloc(1, sizeof(SpectralAtmo));
    if (!f1 || !f2 || !f3 || !b1 || !b2 || !b3 || !a) {
        free(f1); free(f2); free(f3); free(b1); free(b2); free(b3); free(a);
        return 1;
    }
    SpectralExportInfo i1, i2;
    const int r1 = spectral_export(hp, opt, f1, &i1), r2 = spectral_export(hp, opt, f2, &i2);
    spectrum_serialize(f1, b1);
    spectrum_serialize(f2, b2);
    size_t diff = 0;
    for (size_t i = 0; i < nb; ++i) diff += b1[i] != b2[i];
    char path[512];
    const char* tmpdir = getenv("TEMP");
    if (!tmpdir) tmpdir = getenv("TMPDIR");
    snprintf(path, sizeof path, "%s/gme_d5_spectrum.bin", tmpdir ? tmpdir : ".");
    const int sv = spectrum_save(path, f1), ld = sv ? -9 : spectrum_load(path, f3);
    if (!ld) spectrum_serialize(f3, b3);
    const int rt = !sv && !ld && !memcmp(b1, b3, nb);
    remove(path);
    fails += d3_check("spectrum.bin byte-identical x2", !r1 && !r2 && !diff && rt,
                      "%zu bytes, %zu differ between two builds (%.1f s + %.1f s); save + load round trip %s", nb, diff, i1.seconds,
                      i2.seconds, rt ? "identical (CRC ok)" : "FAILED");
    for (int k = 0; k < SPECF_NHAZE; ++k)
        printf("     haze bin %d: haze_q12 %4d -> x_CH4 %.4f (%.2f km-amagat), tau_h550 %.4f, PAR global T %.4f (target %.4f)%s\n", k,
               i1.haze_q12[k], i1.x_ch4[k], i1.ch4_col[k], i1.tau_h550[k], i1.t_par[k], i1.target[k],
               i1.unreachable[k] ? "  [Rayleigh + gas below the target: no haze]" : "");
    SpectralOpts o;
    memset(&o, 0, sizeof o);
    if (opt) o = *opt;
    o.haze_q12 = 0;
    o.ms_j0 = 0;
    o.quiet = 1;
    spectral_build_atmo_ex(hp, &o, a);
    int32_t k, ff;
    spectrum_haze_bin(a->haze_q12, &k, &ff);
    const double S = f1->s_top_wm2;
    double sum = 0, pn = 0, pd = 0, nir = 0;
    for (int b = 0; b < SPECF_NB; ++b) {
        const double v = spectrum_haze_lerp(f1->surf_q16[k][12][b], f1->surf_q16[k + 1][12][b], ff) / 65536.0;
        if (b == SPEC_NB) { nir = v * S; continue; }
        sum += v;
        const double lo = SPEC_L0 + SPEC_BW * b, hi = lo + SPEC_BW;
        const double fr = fmax(0.0, fmin(hi, 700.0) - fmax(lo, 400.0)) / SPEC_BW;   /* the band's share of PAR */
        pn += fr * v * S;
        pd += fr * a->e_band[b];
    }
    double glob, par, part;
    spectral_lut_global(a, 1.0, &glob, &par, &part);
    const double rel = sum * S / glob - 1;
    fails += d3_check("noon global (bands) vs LUT +-1%", fabs(rel) <= 0.01,
                      "haze_q12 %d -> bins %d/%d f %d/256: sum of 8 bands x S_top %.2f W/m^2 vs the LUT model %.2f W/m^2 (%+.3f%%); "
                      "NIR 740-900 %.2f W/m^2", a->haze_q12, k, k + 1, ff, sum * S, glob, 100 * rel, nir);
    const double tpar = pn / pd, tbin = i1.t_par[k] + (i1.t_par[k + 1] - i1.t_par[k]) * ff / 256.0;
    fails += d3_check("PAR transmission at the planet's haze", fabs(tpar - a->haze_q12 / 4096.0) <= 0.005,
                      "from the file's bands %.4f, the bins' 5 nm PAR %.4f, the renderer's LUT %.4f; haze_q12 / 4096 = %.4f (gate +-0.005)",
                      tpar, tbin, par / part, a->haze_q12 / 4096.0);
    int ref[SPEC_NHUE][3], b8[SPEC_NHUE][3], sp[SPEC_NHUE][3];
    double de[SPEC_NHUE];
    const int worst = spectral_pigment_table(a, ref, b8, de, sp);
    double demax = 0;
    for (int h = 1; h < SPEC_NHUE; ++h) demax = fmax(demax, de[h]);
    fails += d3_check("pigment table vs spec 5.1 (+-3)", worst <= 3,
                      "max |sRGB - table| %d levels over bins 1..15; the renderer's 8 bands vs 1 nm: Delta E2000 <= %.1f", worst, demax);
    printf("D5 %s\n", fails ? "FAIL" : "PASS");
    spectral_free(a);
    free(a); free(f1); free(f2); free(f3); free(b1); free(b2); free(b3);
    return fails;
}

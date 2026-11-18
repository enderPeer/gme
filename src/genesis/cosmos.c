/* cosmos.c - a star forms with a disk of dust; dust has already stuck into
 * planetesimals whose make-up depends on how hot the disk is where they
 * formed (rock near the star, ice beyond the frost line). Gravity does the
 * rest: bodies perturb each other, collide and merge, cores past ~10 Earth
 * masses pull in hydrogen/helium gas while the gas disk lasts.
 *
 * Time compression: real accretion takes 10-100 Myr. Bodies merge once they
 * come within a fraction of their Hill radius (the zone where their own
 * gravity beats the star's), which lets a few thousand orbits stand in for
 * that. Everything else is ordinary Newtonian gravity. */
#include "cosmos.h"
#include "../image.h"
#include "../platform.h"
#include "../rtcuda.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cosmos_kernels.inc" /* COSMOS_KERNELS_SRC, generated from cosmos_kernels.cu */

#define G_AU   39.47841760435743   /* 4 pi^2: AU^3 / (Msun yr^2) */
#define PI     3.14159265358979
#define GCM2_TO_MSUN_AU2 (1.0 / 8.888e6)
#define PAIR_CAP 65536

typedef struct { float x, y, z, w; } f4;
typedef struct { int a, b; } i2;

void cosmos_default_config(CosmosConfig* c) {
    memset(c, 0, sizeof *c);
    c->seed = 1;
    c->particles = 8192;
    c->years = 3000;
    c->r_in = 0.4;
    c->r_out = 12.0;
    c->hill_frac = 0.5;
    c->disk_scale = 1.0;
    c->snapshots = 0;
}

/* ---- random -------------------------------------------------------------- */
static uint64_t rng_s;
static double rnd(void) {
    rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17;
    return (double)(rng_s >> 11) * (1.0 / 9007199254740992.0);
}
static double rnd_gauss(void) {
    double u = rnd(), v = rnd();
    if (u < 1e-12) u = 1e-12;
    return sqrt(-2.0 * log(u)) * cos(2.0 * PI * v);
}

/* ---- disk physics ---------------------------------------------------------- */
double disk_temperature(const StarInfo* s, double r) {
    return 280.0 * pow(s->lum, 0.25) / sqrt(r);
}

static double smooth_below(double T, double hi, double lo) { /* 1 when T <= lo, 0 when T >= hi */
    if (T <= lo) return 1.0;
    if (T >= hi) return 0.0;
    const double t = (hi - T) / (hi - lo);
    return t * t * (3.0 - 2.0 * t);
}

/* Solid mass per unit gas mass of each element at disk temperature T. */
static void condensed(double T, double out[E_COUNT]) {
    const double* X = SOLAR_MASS_FRAC;
    double n[E_COUNT];
    for (int e = 0; e < E_COUNT; ++e) { n[e] = X[e] / ELEMENT_MASS[e]; out[e] = 0.0; }
    const double f_rock = smooth_below(T, 1450, 1250);  /* silicates + metal */
    const double f_sulf = smooth_below(T, 750, 600);    /* FeS */
    const double f_org  = smooth_below(T, 450, 350);    /* refractory organics */
    const double f_ice  = smooth_below(T, 190, 150);    /* water ice: the frost line */
    const double f_nh3  = smooth_below(T, 95, 75);
    const double f_co   = smooth_below(T, 45, 30);

    out[E_FE] = X[E_FE] * f_rock;
    out[E_MG] = X[E_MG] * f_rock;
    out[E_SI] = X[E_SI] * f_rock;
    out[E_S]  = X[E_S] * f_sulf;
    double o_mol = (n[E_MG] + 2.0 * n[E_SI] + 0.3 * n[E_FE]) * f_rock;  /* MgO, SiO2, some FeO */
    if (o_mol > n[E_O]) o_mol = n[E_O];
    const double water_mol = (n[E_O] - o_mol) * 0.55 * f_ice;           /* rest of O stays in CO */
    o_mol += water_mol;
    double h_mol = 2.0 * water_mol;
    double c_mol = n[E_C] * (0.15 * f_org + 0.40 * f_co);
    double n_mol = n[E_N] * (0.05 * f_org + 0.30 * f_nh3);
    h_mol += n[E_C] * 0.15 * f_org + 3.0 * n[E_N] * 0.30 * f_nh3;
    out[E_O] = o_mol * ELEMENT_MASS[E_O];
    out[E_H] = h_mol * ELEMENT_MASS[E_H];
    out[E_C] = c_mol * ELEMENT_MASS[E_C];
    out[E_N] = n_mol * ELEMENT_MASS[E_N];
}

static double sigma_gas(double r, double scale) { return 1700.0 * scale * pow(r, -1.5) * GCM2_TO_MSUN_AU2; }

double body_volatile_fraction(const Body* b) { return b->comp[E_H] + b->comp[E_HE]; }

/* Physical radius from composition: rock ~5 g/cc, ices ~1, captured gas
 * envelopes follow a giant-planet mass-radius relation. */
static double body_radius(const Body* b) {
    const double gas = body_volatile_fraction(b);
    const double rock = b->comp[E_FE] + b->comp[E_MG] + b->comp[E_SI] + b->comp[E_S] + 0.5 * b->comp[E_O];
    const double ice = 1.0 - gas - rock > 0 ? 1.0 - gas - rock : 0.0;
    const double mg = b->m * 1.989e33;
    const double vol = mg * (rock / 5.0 + ice / 1.2 + (gas > 0.3 ? 0 : gas / 0.5)) + 1.0;
    double r_cm = cbrt(3.0 * vol / (4.0 * PI));
    double r = r_cm / 1.496e13;
    if (gas > 0.3) { /* giant: ~1 Jupiter radius over a wide mass range */
        const double me = b->m / M_EARTH_SUN;
        const double rg = R_EARTH_AU * 11.0 * pow(fmin(me / 318.0, 3.0), 0.05) * (me < 100 ? pow(me / 100.0, 0.3) : 1.0);
        if (rg > r) r = rg;
    }
    return r;
}

static double collision_radius(const StarInfo* s, const Body* b, double hill_frac) {
    const double r = sqrt(b->x[0] * b->x[0] + b->x[1] * b->x[1] + b->x[2] * b->x[2]);
    const double hill = r * cbrt(b->m / (3.0 * s->mass));
    return fmax(b->radius_au, hill_frac * hill);
}

void orbit_elements(const StarInfo* s, const Body* b, double* a, double* e, double* inc) {
    const double mu = G_AU * (s->mass + b->m);
    const double* x = b->x; const double* v = b->v;
    const double r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    const double v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const double h[3] = { x[1] * v[2] - x[2] * v[1], x[2] * v[0] - x[0] * v[2], x[0] * v[1] - x[1] * v[0] };
    const double hn = sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
    const double ev[3] = { (v[1] * h[2] - v[2] * h[1]) / mu - x[0] / r,
                           (v[2] * h[0] - v[0] * h[2]) / mu - x[1] / r,
                           (v[0] * h[1] - v[1] * h[0]) / mu - x[2] / r };
    *a = 1.0 / (2.0 / r - v2 / mu);
    *e = sqrt(ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2]);
    *inc = hn > 0 ? acos(h[2] / hn) : 0.0;
}

/* ---- snapshot plot ------------------------------------------------------- */
static uint32_t body_color(const Body* b) {
    const double gas = body_volatile_fraction(b);
    const double ice = b->comp[E_O] + b->comp[E_C] + b->comp[E_N] - 0.45 * (b->comp[E_MG] + b->comp[E_SI] + b->comp[E_FE]);
    if (gas > 0.3) return 0xD9B47A;           /* gas giant: tan */
    if (ice > 0.25) return 0x7FD3E6;          /* icy: cyan */
    if (b->comp[E_FE] > 0.25) return 0xB0705A;/* metal-rich: rust */
    return 0xA89A8A;                           /* rocky: grey-brown */
}

static void plot_system(const char* path, const StarInfo* s, const Body* b, int n, double r_view,
                        double frost, int orbits) {
    Image im;
    const int W = 1024;
    if (img_create(&im, W, W, 0x05060A)) return;
    const double sc = (W * 0.48) / r_view;
    const float c = W * 0.5f;
    /* habitable zone (liquid water for an Earth-like planet) */
    const double hz0 = 0.95 * sqrt(s->lum), hz1 = 1.67 * sqrt(s->lum);
    for (double r = hz0; r < hz1; r += 0.5 / sc) img_ring(&im, c, c, (float)(r * sc), 0x2E7D32, 0.25f);
    img_ring(&im, c, c, (float)(frost * sc), 0x4FC3F7, 0.5f); /* frost line */
    for (int i = 0; i < n; ++i) {
        if (b[i].m <= 0) continue;
        const double me = b[i].m / M_EARTH_SUN;
        if (orbits && me > 0.3) {
            double a, e, inc;
            orbit_elements(s, &b[i], &a, &e, &inc);
            if (a > 0 && a < r_view * 1.2) img_ring(&im, c, c, (float)(a * sc), 0x5A5F73, 0.6f);
        }
        float rad = (float)(0.6 + 1.6 * log10(1.0 + me * 30.0));
        if (rad > 14) rad = 14;
        img_disc(&im, (float)(c + b[i].x[0] * sc), (float)(c - b[i].x[1] * sc), rad, body_color(&b[i]), me > 0.05 ? 1.f : 0.7f);
    }
    const float sr = (float)(4.0 + 3.0 * s->radius_au / R_SUN_AU);
    img_disc(&im, c, c, sr * 1.8f, 0xFFD27A, 0.25f);
    img_disc(&im, c, c, sr, 0xFFF2C0, 1.f);
    write_png(path, im.px, im.w, im.h);
    img_free(&im);
}

typedef struct SortKey { double a; int idx; } SortKey;
static int sort_key_cmp(const void* x, const void* y) {
    const double a = ((const SortKey*)x)->a, b = ((const SortKey*)y)->a;
    return (a > b) - (a < b);
}

/* ---- the run --------------------------------------------------------------- */
int cosmos_run(const CosmosConfig* cfg, CosmosResult* res) {
    memset(res, 0, sizeof *res);
    const double t_start = time_now();
    rng_s = 0x9E3779B97F4A7C15ull ^ ((uint64_t)cfg->seed * 0xD1B54A32D192ED03ull);
    for (int i = 0; i < 8; ++i) rnd();

    /* star: mass varies per seed; main-sequence scalings */
    StarInfo s;
    s.mass = 0.8 + 0.4 * rnd();
    s.lum = pow(s.mass, 4.0);
    s.teff = 5778.0 * pow(s.mass, 0.505);
    s.radius_au = R_SUN_AU * pow(s.mass, 0.8);
    res->star = s;
    const double disk_scale = cfg->disk_scale * (0.7 + 0.8 * rnd());
    /* frost line: where water ice condenses */
    res->frost_line_au = pow(280.0 * pow(s.lum, 0.25) / 170.0, 2.0);

    /* --- disk: solid surface density per radius -> planetesimals --- */
    const int N0 = cfg->particles;
    const int NB = 512;
    double cdf[513], comp_r[512][E_COUNT];
    cdf[0] = 0;
    const double dr = (cfg->r_out - cfg->r_in) / NB;
    for (int k = 0; k < NB; ++k) {
        const double r = cfg->r_in + (k + 0.5) * dr;
        double cond[E_COUNT], tot = 0;
        condensed(disk_temperature(&s, r), cond);
        for (int e = 0; e < E_COUNT; ++e) tot += cond[e];
        for (int e = 0; e < E_COUNT; ++e) comp_r[k][e] = tot > 0 ? cond[e] / tot : 0;
        cdf[k + 1] = cdf[k] + 2.0 * PI * r * dr * sigma_gas(r, disk_scale) * tot;
    }
    const double m_solid = cdf[NB];
    res->initial_solid_me = m_solid / M_EARTH_SUN;
    const double m0 = m_solid / N0;

    Body* B = (Body*)calloc((size_t)N0, sizeof(Body));
    f4* hp = (f4*)malloc(sizeof(f4) * N0);
    f4* hv = (f4*)malloc(sizeof(f4) * N0);
    i2* pairs = (i2*)malloc(sizeof(i2) * PAIR_CAP);
    if (!B || !hp || !hv || !pairs) { fprintf(stderr, "cosmos: out of memory\n"); return -1; }
    for (int i = 0; i < N0; ++i) {
        const double u = rnd() * m_solid;
        int k = 0;
        while (k < NB - 1 && cdf[k + 1] < u) ++k;
        const double r = cfg->r_in + (k + rnd()) * dr;
        const double th = rnd() * 2.0 * PI;
        const double vk = sqrt(G_AU * s.mass / r);
        Body* b = &B[i];
        b->m = m0;
        memcpy(b->comp, comp_r[k], sizeof b->comp);
        const double ecc = 0.01, incl = 0.005;
        b->x[0] = r * cos(th); b->x[1] = r * sin(th); b->x[2] = r * incl * rnd_gauss();
        b->v[0] = -vk * sin(th) + vk * ecc * rnd_gauss();
        b->v[1] =  vk * cos(th) + vk * ecc * rnd_gauss();
        b->v[2] = vk * incl * rnd_gauss();
        b->radius_au = body_radius(b);
    }

    /* --- GPU --- */
    RtModule* mod = rt_compile(COSMOS_KERNELS_SRC, "cosmos_kernels.cu");
    if (!mod) { fprintf(stderr, "cosmos: %s\n", rt_error()); return -1; }
    RtFunc f_kd = rt_func(mod, "k_kick_drift"), f_k = rt_func(mod, "k_kick"), f_acc = rt_func(mod, "k_accel");
    rt_dptr d_pos = 0, d_vel = 0, d_acc = 0, d_pairs = 0, d_count = 0;
    if (rt_alloc(&d_pos, sizeof(f4) * N0) || rt_alloc(&d_vel, sizeof(f4) * N0) || rt_alloc(&d_acc, sizeof(f4) * N0) ||
        rt_alloc(&d_pairs, sizeof(i2) * PAIR_CAP) || rt_alloc(&d_count, 4)) {
        fprintf(stderr, "cosmos: %s\n", rt_error());
        return -1;
    }

    int n = N0;
    #define TO_GPU() do { for (int q = 0; q < n; ++q) { \
            hp[q].x = (float)B[q].x[0]; hp[q].y = (float)B[q].x[1]; hp[q].z = (float)B[q].x[2]; hp[q].w = (float)B[q].m; \
            hv[q].x = (float)B[q].v[0]; hv[q].y = (float)B[q].v[1]; hv[q].z = (float)B[q].v[2]; \
            hv[q].w = B[q].m > 0 ? (float)collision_radius(&s, &B[q], cfg->hill_frac) : 0.f; } \
        rt_upload(d_pos, hp, sizeof(f4) * n); rt_upload(d_vel, hv, sizeof(f4) * n); } while (0)
    #define FROM_GPU() do { rt_download(hp, d_pos, sizeof(f4) * n); rt_download(hv, d_vel, sizeof(f4) * n); \
        for (int q = 0; q < n; ++q) { if (B[q].m <= 0) continue; \
            B[q].x[0] = hp[q].x; B[q].x[1] = hp[q].y; B[q].x[2] = hp[q].z; \
            B[q].v[0] = hv[q].x; B[q].v[1] = hv[q].y; B[q].v[2] = hv[q].z; } } while (0)

    TO_GPU();

    const double p_in = 2.0 * PI * sqrt(pow(cfg->r_in, 3) / (G_AU * s.mass));
    const float dt = (float)(p_in / 60.0);
    const long long steps = (long long)(cfg->years / dt);
    const double t_gas = cfg->years * 0.45;           /* gas disk dissipates */
    const float G = (float)G_AU, star_gm = (float)(G_AU * s.mass);
    const float eps2 = 1e-8f;
    const float tau0 = (float)(t_gas * 0.04);
    const float fm0 = (float)m0;
    const long long check_every = 2000;
    const long long snap_every = cfg->snapshots > 0 ? steps / cfg->snapshots : 0;
    int pair_cap = PAIR_CAP;
    int frame = 0;
    long long collisions = 0;
    int ejected = 0;

    printf("cosmos: star %.2f Msun (L %.2f, %.0f K), disk %.1f-%.1f AU, %.1f Earth masses of solids in %d bodies, frost line %.2f AU\n",
           s.mass, s.lum, s.teff, cfg->r_in, cfg->r_out, res->initial_solid_me, N0, res->frost_line_au);
    printf("cosmos: %lld steps of %.4f yr (%.0f years), gas disk for %.0f years\n", steps, dt, cfg->years, t_gas);
    fflush(stdout);

    float gas_on = 1.f;
    {
        int zero = 0;
        rt_upload(d_count, &zero, 4);
        void* a[] = { &n, &d_pos, &d_vel, &d_acc, &G, &star_gm, &eps2, &gas_on, &tau0, &fm0, &d_pairs, &d_count, &pair_cap };
        rt_launch(f_acc, (unsigned)((n + 255) / 256), 256, 0, a);
    }
    double last_print = time_now();
    for (long long step = 0; step < steps; ++step) {
        const double t = (double)step * dt;
        gas_on = t < t_gas ? 1.f : 0.f;
        const unsigned grid = (unsigned)((n + 255) / 256);
        int zero = 0;
        rt_upload(d_count, &zero, 4);
        void* a1[] = { &n, &d_pos, &d_vel, &d_acc, (void*)&dt };
        void* a2[] = { &n, &d_pos, &d_vel, &d_acc, &G, &star_gm, &eps2, &gas_on, &tau0, &fm0, &d_pairs, &d_count, &pair_cap };
        void* a3[] = { &n, &d_vel, &d_acc, (void*)&dt };
        if (rt_launch(f_kd, grid, 256, 0, a1) || rt_launch(f_acc, grid, 256, 0, a2) || rt_launch(f_k, grid, 256, 0, a3)) {
            fprintf(stderr, "cosmos: %s\n", rt_error());
            return -1;
        }
        int count = 0;
        if (rt_download(&count, d_count, 4)) { fprintf(stderr, "cosmos: %s\n", rt_error()); return -1; }

        const int periodic = (step % check_every) == check_every - 1;
        const int snap = snap_every && (step % snap_every) == 0;
        if (count > 0 || periodic || snap) {
            FROM_GPU();
            if (count > pair_cap) count = pair_cap;
            if (count > 0) {
                rt_download(pairs, d_pairs, sizeof(i2) * (size_t)count);
                for (int k = 0; k < count; ++k) {  /* perfect merging, momentum conserved */
                    Body* p = &B[pairs[k].a];
                    Body* q = &B[pairs[k].b];
                    if (p->m <= 0 || q->m <= 0) continue;
                    if (q->m > p->m) { Body* tmp = p; p = q; q = tmp; }
                    const double m = p->m + q->m;
                    for (int d = 0; d < 3; ++d) {
                        p->x[d] = (p->m * p->x[d] + q->m * q->x[d]) / m;
                        p->v[d] = (p->m * p->v[d] + q->m * q->v[d]) / m;
                    }
                    for (int e = 0; e < E_COUNT; ++e) p->comp[e] = (p->m * p->comp[e] + q->m * q->comp[e]) / m;
                    p->m = m;
                    p->merges += q->merges + 1;
                    p->radius_au = body_radius(p);
                    q->m = 0;
                    ++collisions;
                }
            }
            if (periodic) {
                const double dtc = check_every * dt;
                for (int i = 0; i < n; ++i) {
                    Body* b = &B[i];
                    if (b->m <= 0) continue;
                    const double r = sqrt(b->x[0] * b->x[0] + b->x[1] * b->x[1] + b->x[2] * b->x[2]);
                    if (r > 60.0 || r < 2.0 * s.radius_au + 0.02) { b->m = 0; ++ejected; continue; } /* lost */
                    /* runaway gas capture onto massive cores while gas remains */
                    const double me = b->m / M_EARTH_SUN;
                    if (gas_on > 0 && me > 8.0) {
                        const double cap = 320.0 * M_EARTH_SUN * disk_scale * sqrt(r / 5.2);
                        if (b->m < cap) {
                            double dm = b->m * dtc / (t_gas * 0.08);
                            if (b->m + dm > cap) dm = cap - b->m;
                            const double m = b->m + dm;
                            for (int e = 0; e < E_COUNT; ++e)
                                b->comp[e] = (b->comp[e] * b->m + SOLAR_MASS_FRAC[e] * dm) / m;
                            b->m = m;
                            b->radius_au = body_radius(b);
                        }
                    }
                }
                /* compact: drop merged / lost bodies so later steps get cheaper */
                int alive = 0;
                for (int i = 0; i < n; ++i) if (B[i].m > 0) B[alive++] = B[i];
                n = alive;
            }
            if (snap && cfg->snapshot_dir) {
                char path[512];
                snprintf(path, sizeof path, "%s/cosmos_%04d.png", cfg->snapshot_dir, frame++);
                plot_system(path, &s, B, n, cfg->r_out * 1.15, res->frost_line_au, 0);
            }
            /* dead bodies: park far away with zero mass */
            for (int i = 0; i < n; ++i) if (B[i].m <= 0) { B[i].x[0] = 1e6 + i; B[i].x[1] = B[i].x[2] = 0; }
            TO_GPU();
        }

        if (time_now() - last_print > 5.0 || step == steps - 1) {
            int big = 0;
            double mmax = 0;
            for (int i = 0; i < n; ++i) { if (B[i].m > 0.1 * M_EARTH_SUN) ++big; if (B[i].m > mmax) mmax = B[i].m; }
            printf("cosmos: %6.0f yr (%5.1f%%)  bodies %6d  >0.1 Me %3d  largest %7.2f Me  merges %lld\n",
                   t, 100.0 * step / steps, n, big, mmax / M_EARTH_SUN, collisions);
            fflush(stdout);
            last_print = time_now();
        }
    }
    FROM_GPU();

    /* --- survivors sorted by semi-major axis --- */
    int alive = 0;
    for (int i = 0; i < n; ++i) {
        if (B[i].m <= 0) continue;
        double a, e, inc;
        orbit_elements(&s, &B[i], &a, &e, &inc);
        if (a <= 0 || e >= 1) { ++ejected; continue; }
        B[alive++] = B[i];
    }
    {   /* sort by semi-major axis */
        SortKey* keys = (SortKey*)malloc(sizeof(SortKey) * (size_t)(alive + 1));
        Body* tmp = (Body*)malloc(sizeof(Body) * (size_t)(alive + 1));
        for (int i = 0; i < alive; ++i) {
            double e, inc;
            orbit_elements(&s, &B[i], &keys[i].a, &e, &inc);
            keys[i].idx = i;
        }
        qsort(keys, (size_t)alive, sizeof(SortKey), sort_key_cmp);
        for (int i = 0; i < alive; ++i) tmp[i] = B[keys[i].idx];
        memcpy(B, tmp, sizeof(Body) * (size_t)alive);
        free(keys);
        free(tmp);
    }
    if (cfg->snapshot_dir) {
        char path[512];
        snprintf(path, sizeof path, "%s/system.png", cfg->snapshot_dir);
        plot_system(path, &s, B, alive, cfg->r_out * 1.15, res->frost_line_au, 1);
    }
    res->bodies = B;
    res->n = alive;
    res->steps = steps;
    res->collisions = collisions;
    res->ejected = ejected;
    res->wall_seconds = time_now() - t_start;

    rt_free(d_pos); rt_free(d_vel); rt_free(d_acc); rt_free(d_pairs); rt_free(d_count);
    rt_module_free(mod);
    free(hp); free(hv); free(pairs);
    return 0;
}

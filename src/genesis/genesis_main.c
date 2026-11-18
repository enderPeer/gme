/* genesis_main.c - gme_genesis: boots a world from physics, stage by stage.
 *
 *   cosmos  : star + disk -> planets (GPU, one process per GPU / seed)
 *   score   : rank finished systems by habitability (pick the best seed)
 *   chem    : bonding chemistry on the chosen planet (CPU, shardable)
 *   finish  : merge chem shards -> home.planet (what the engine loads)
 *   all     : cosmos + chem + finish in one go on this machine
 *
 * Every stage reads/writes plain files in a run directory, so the stages can
 * run on different machines and be resumed or re-run independently. */
#include "chem.h"
#include "cosmos.h"
#include "../jobs.h"
#include "../planetfile.h"
#include "../platform.h"
#include "../rtcuda.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define MKDIR(p) mkdir(p, 0755)
#endif

typedef struct System { StarInfo star; double frost; int n; Body* b; uint32_t seed; } System;

/* ---- system file --------------------------------------------------------- */
static int system_save(const char* dir, const System* s) {
    char path[512];
    snprintf(path, sizeof path, "%s/system.txt", dir);
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "seed %u\nstar %.10g %.10g %.10g %.10g\nfrost %.10g\n", s->seed, s->star.mass, s->star.lum, s->star.teff,
            s->star.radius_au, s->frost);
    for (int i = 0; i < s->n; ++i) {
        const Body* b = &s->b[i];
        fprintf(f, "body %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %d", b->m, b->x[0], b->x[1], b->x[2], b->v[0],
                b->v[1], b->v[2], b->radius_au, b->merges);
        for (int e = 0; e < E_COUNT; ++e) fprintf(f, " %.8g", b->comp[e]);
        fprintf(f, "\n");
    }
    fclose(f);
    return 0;
}

static int system_load(const char* dir, System* s) {
    char path[512];
    snprintf(path, sizeof path, "%s/system.txt", dir);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    memset(s, 0, sizeof *s);
    int cap = 256;
    s->b = (Body*)calloc((size_t)cap, sizeof(Body));
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "seed %u", &s->seed) == 1) continue;
        if (sscanf(line, "star %lf %lf %lf %lf", &s->star.mass, &s->star.lum, &s->star.teff, &s->star.radius_au) == 4) continue;
        if (sscanf(line, "frost %lf", &s->frost) == 1) continue;
        if (!strncmp(line, "body ", 5)) {
            if (s->n == cap) { cap *= 2; s->b = (Body*)realloc(s->b, sizeof(Body) * (size_t)cap); }
            Body* b = &s->b[s->n];
            memset(b, 0, sizeof *b);
            char* p = line + 5;
            double v[9 + E_COUNT];
            for (int k = 0; k < 9 + E_COUNT; ++k) v[k] = strtod(p, &p);
            b->m = v[0];
            for (int d = 0; d < 3; ++d) { b->x[d] = v[1 + d]; b->v[d] = v[4 + d]; }
            b->radius_au = v[7];
            b->merges = (int)v[8];
            for (int e = 0; e < E_COUNT; ++e) b->comp[e] = v[9 + e];
            s->n++;
        }
    }
    fclose(f);
    return 0;
}

/* ---- habitability --------------------------------------------------------- */
static double body_a(const System* s, const Body* b) { double a, e, i; orbit_elements(&s->star, b, &a, &e, &i); return a; }
static double t_equilibrium(const System* s, double a) { return 278.6 * pow(s->star.lum, 0.25) / sqrt(a) * 0.915; }

static const char* body_type(const Body* b) {
    const double me = b->m / M_EARTH_SUN;
    if (body_volatile_fraction(b) > 0.3) return me > 50 ? "gas giant" : "ice/gas giant";
    if (me < 0.05) return "planetesimal";
    if (me < 0.3) return "dwarf planet";
    return "rocky planet";
}

static double habitability(const System* s, const Body* b) {
    const double me = b->m / M_EARTH_SUN;
    if (me < 0.1 || me > 12 || body_volatile_fraction(b) > 0.2) return -1e9;
    const double T = t_equilibrium(s, body_a(s, b));
    double h = b->comp[E_H];  /* hydrogen in a rocky body = bound water/organics */
    if (h < 1e-7) h = 1e-7;
    double wet = log10(h / 1e-4);
    if (wet > 2) wet = 2;
    if (wet < -3) wet = -3;
    const double small = me < 0.3 ? 1.5 : 0.0;   /* too small to hold an atmosphere */
    return -fabs(T - 262.0) / 35.0 - fabs(log(me)) * 0.7 + 0.6 * wet - small;
}

static int choose_home(const System* s) {
    int best = -1;
    double bs = -1e18;
    for (int i = 0; i < s->n; ++i) {
        const double h = habitability(s, &s->b[i]);
        if (h > bs) { bs = h; best = i; }
    }
    return bs > -1e8 ? best : -1;
}

static void print_system(const System* s) {
    printf("\nstar: %.2f Msun, L %.2f Lsun, %.0f K. frost line %.2f AU. habitable zone %.2f-%.2f AU\n", s->star.mass,
           s->star.lum, s->star.teff, s->frost, 0.95 * sqrt(s->star.lum), 1.67 * sqrt(s->star.lum));
    printf("  #   a [AU]   ecc   mass [Me]  type            T_eq [K]  merged  rock/ice/gas share   habitability\n");
    for (int i = 0; i < s->n; ++i) {
        const Body* b = &s->b[i];
        const double me = b->m / M_EARTH_SUN;
        if (me < 0.05) continue;
        double a, e, inc;
        orbit_elements(&s->star, b, &a, &e, &inc);
        const double gas = body_volatile_fraction(b);
        const double rock = b->comp[E_FE] + b->comp[E_MG] + b->comp[E_SI] + b->comp[E_S] + 0.5 * b->comp[E_O];
        const double hb = habitability(s, b);
        char hs[16] = "-";
        if (hb > -1e8) snprintf(hs, sizeof hs, "%+.2f", hb);
        printf("  %-3d %7.3f  %5.3f  %9.3f  %-14s  %7.0f  %6d   %4.0f%% /%4.0f%% /%4.0f%%   %s\n", i, a, e, me, body_type(b),
               t_equilibrium(s, a), b->merges, 100 * rock, 100 * fmax(0, 1 - rock - gas), 100 * gas, hs);
    }
}

/* ---- planet derivation ------------------------------------------------------ */
static void crust_inventory(const Body* b, double out[E_COUNT]) {
    double t = 0;
    for (int e = 0; e < E_COUNT; ++e) out[e] = b->comp[e];
    out[E_FE] *= 0.3;  /* most iron sank into the core */
    out[E_S] *= 0.5;
    /* impact heating during accretion boils off most volatiles */
    out[E_C] *= 0.08;
    out[E_N] *= 0.2;
    out[E_H] *= 0.5;
    for (int e = 0; e < E_COUNT; ++e) t += out[e];
    for (int e = 0; e < E_COUNT; ++e) out[e] /= t;
}

static void lerp3(float* o, const float* a, const float* b, float t) { for (int i = 0; i < 3; ++i) o[i] = a[i] + (b[i] - a[i]) * t; }

static void star_color(double teff, float* c) {
    static const double T[] = { 3000, 4000, 5000, 5800, 6500, 8000 };
    static const float C[][3] = { {1.0f,0.55f,0.30f}, {1.0f,0.72f,0.48f}, {1.0f,0.86f,0.70f}, {1.0f,0.95f,0.88f}, {0.96f,0.96f,1.0f}, {0.82f,0.88f,1.0f} };
    if (teff <= T[0]) { memcpy(c, C[0], 12); return; }
    for (int i = 0; i < 5; ++i)
        if (teff <= T[i + 1]) { lerp3(c, C[i], C[i + 1], (float)((teff - T[i]) / (T[i + 1] - T[i]))); return; }
    memcpy(c, C[5], 12);
}

static int cmp_int(const void* a, const void* b) { return *(const int*)a - *(const int*)b; }

static void derive_planet(const System* s, int home, const ChemResult* cr, HomePlanet* hp) {
    const Body* b = &s->b[home];
    planet_default(hp);
    hp->system_seed = s->seed;
    hp->planet_index = home;
    hp->star_mass = s->star.mass; hp->star_lum = s->star.lum; hp->star_teff = s->star.teff;
    double a, e, inc;
    orbit_elements(&s->star, b, &a, &e, &inc);
    hp->a_au = a; hp->ecc = e;
    hp->mass_me = b->m / M_EARTH_SUN;
    hp->radius_re = b->radius_au / R_EARTH_AU;
    hp->gravity_g = hp->mass_me / (hp->radius_re * hp->radius_re);
    hp->t_eq = t_equilibrium(s, a);

    const double M = chem_total_mass(cr);
    const double* cm = cr->class_mass;
    const int keeps_light = hp->mass_me > 3.0;      /* light gases escape small planets */
    const int keeps_atm = hp->mass_me > 0.25;
    const double gas_heavy = cm[CC_N2] + cm[CC_O2] + cm[CC_CO2] + cm[CC_CO] + cm[CC_CH4] + cm[CC_NH3] + cm[CC_H2S];
    const double gas = keeps_atm ? gas_heavy + (keeps_light ? cm[CC_H2] + cm[CC_HE] : 0) : 0;
    hp->atm_frac = gas / M;
    hp->water_frac = cm[CC_H2O] / M;
    hp->organic_frac = cm[CC_ORGANIC] / M;

    /* greenhouse warming from the gases that trap heat */
    const double gh = keeps_atm ? (cm[CC_CO2] + 25 * cm[CC_CH4] + 5 * cm[CC_NH3] + 0.05 * cm[CC_H2O]) / M : 0;
    /* the inventory is what *can* outgas; only a small part is airborne at once */
    double dT = 18.0 * log(1.0 + gh * 30.0);
    if (dT > 200) dT = 200;
    hp->t_surf = hp->t_eq + dT;

    GmePlanet* g = &hp->gen;
    *g = gme_planet_default();
    double rel = 1.0 / sqrt(hp->gravity_g > 0.05 ? hp->gravity_g : 0.05);
    if (rel < 0.55) rel = 0.55;
    if (rel > 1.7) rel = 1.7;
    g->relief_q8 = (int32_t)(256 * rel);
    const double fe_mg = cr->sil_mg + cr->sil_fe;
    const double ratio = fe_mg > 0 ? cr->sil_si / fe_mg : 2.0;
    double gr = (ratio - 0.6) / 1.4;
    g->granite_q8 = (int32_t)(256 * (gr < 0.06 ? 0.06 : gr > 0.9 ? 0.9 : gr));
    g->ore_band = (int32_t)fmin(3500.0, cm[CC_IRON_OXIDE] / M * 25000.0 + cm[CC_METAL] / M * 8000.0);
    g->sulfide_band = (int32_t)fmin(2500.0, cm[CC_SULFIDE] / M * 25000.0);

    /* oceans: how much water, and whether it is liquid */
    const int liquid = hp->t_surf >= 268 && hp->t_surf <= 373 && keeps_atm;
    const int frozen = hp->t_surf < 268;
    double cover = pow(hp->water_frac / 0.003, 0.7);
    if (cover > 0.92) cover = 0.92;
    if (!keeps_atm && !frozen) cover = 0;
    if (hp->t_surf > 373) cover = 0;
    hp->ocean_cover = cover;
    if (cover > 0.01) {
        const int NS = 20000;
        int* hs = (int*)malloc(sizeof(int) * NS);
        uint32_t r = s->seed * 2654435761u + 7;
        for (int i = 0; i < NS; ++i) {
            r = r * 1664525u + 1013904223u; const int x = (int)(r >> 16) - 32768;
            r = r * 1664525u + 1013904223u; const int z = (int)(r >> 16) - 32768;
            hs[i] = (gme_height_raw(x, z, s->seed) * g->relief_q8) >> 8;
        }
        qsort(hs, NS, sizeof(int), cmp_int);
        g->sea_level = hs[(int)(cover * (NS - 1))];
        free(hs);
        g->ocean_mat = frozen && hp->t_surf < 240 ? MAT_ICE : MAT_WATER;
        g->ocean_top_mat = frozen ? MAT_ICE : MAT_WATER;
    } else {
        g->sea_level = -100000;
    }
    /* soil needs weathering (liquid water) and organics */
    if (liquid && hp->organic_frac > 1e-5) { g->soil_depth = 3; g->surface_mat = MAT_DIRT; }
    else if (liquid || frozen)             { g->soil_depth = 1; g->surface_mat = MAT_REGOLITH; }
    else                                   { g->soil_depth = 0; g->surface_mat = MAT_REGOLITH; }
    g->beach_mat = cover > 0.01 ? MAT_SAND : MAT_REGOLITH;

    /* light and sky */
    star_color(s->star.teff, hp->sun_color);
    const double flux = s->star.lum / (a * a);
    hp->sun_intensity = (float)fmin(1.6, fmax(0.45, pow(flux, 0.25)));
    hp->sun_size = (float)fmin(0.1, fmax(0.012, s->star.radius_au / a * 8.0));
    double tau = hp->atm_frac * 400.0;
    if (tau > 1.0) tau = 1.0;
    float tint[3] = { 0.24f, 0.46f, 0.85f };                       /* N2/O2 Rayleigh blue */
    if (gas > 0) {
        const float co2 = (float)(cm[CC_CO2] / gas), ch4 = (float)((cm[CC_CH4] + cm[CC_NH3]) / gas);
        const float dusty[3] = { 0.78f, 0.58f, 0.42f }, hazy[3] = { 0.62f, 0.66f, 0.55f };
        lerp3(tint, tint, dusty, co2 > 0.8f ? 0.8f : co2);
        lerp3(tint, tint, hazy, ch4 > 0.8f ? 0.8f : ch4);
    }
    const float space[3] = { 0.01f, 0.012f, 0.03f };
    lerp3(hp->zenith, space, tint, (float)tau);
    float hz[3] = { tint[0] * 0.5f + 0.45f, tint[1] * 0.5f + 0.42f, tint[2] * 0.4f + 0.5f };
    lerp3(hp->horizon, space, hz, (float)tau);

    /* the sky: where the star and the other planets are */
    const double* x0 = b->x;
    const double d0 = sqrt(x0[0] * x0[0] + x0[1] * x0[1] + x0[2] * x0[2]);
    for (int k = 0; k < 3; ++k) hp->sun_ecl[k] = (float)(-x0[k] / d0);
    for (int i = 0; i < s->n && hp->nbodies < PLANET_MAX_SKY_BODIES; ++i) {
        const Body* o = &s->b[i];
        if (i == home || o->m / M_EARTH_SUN < 0.05) continue;
        double d[3], L = 0;
        for (int k = 0; k < 3; ++k) { d[k] = o->x[k] - x0[k]; L += d[k] * d[k]; }
        L = sqrt(L);
        SkyBody* sb = &hp->bodies[hp->nbodies++];
        for (int k = 0; k < 3; ++k) sb->dir[k] = (float)(d[k] / L);
        sb->ang = (float)fmin(0.03, fmax(0.003, o->radius_au / L * 60.0));
        const double gasf = body_volatile_fraction(o);
        const float giant[3] = { 0.85f, 0.72f, 0.52f }, icy[3] = { 0.62f, 0.85f, 0.92f }, rock[3] = { 0.72f, 0.66f, 0.60f };
        memcpy(sb->color, gasf > 0.3 ? (o->m / M_EARTH_SUN > 50 ? giant : icy) : rock, 12);
    }

    /* human-readable summaries */
    struct { int c; double m; } parts[CC_COUNT];
    int np = 0;
    char* out = hp->atmosphere;
    size_t left = sizeof hp->atmosphere;
    out[0] = 0;
    static const int gases[] = { CC_N2, CC_O2, CC_CO2, CC_CO, CC_CH4, CC_NH3, CC_H2S, CC_H2, CC_HE };
    for (int i = 0; i < 9; ++i) {
        const int c = gases[i];
        if ((c == CC_H2 || c == CC_HE) && !keeps_light) continue;
        if (cm[c] > 0 && keeps_atm) { parts[np].c = c; parts[np].m = cm[c]; ++np; }
    }
    for (int i = 0; i < np; ++i) for (int j = i + 1; j < np; ++j) if (parts[j].m > parts[i].m) { int t = parts[i].c; double m = parts[i].m; parts[i] = parts[j]; parts[j].c = t; parts[j].m = m; }
    if (!np) snprintf(out, left, "none (escaped)");
    for (int i = 0; i < np && i < 4; ++i) {
        const int w = snprintf(out, left, "%s%s %.0f%%", i ? ", " : "", CC_NAME[parts[i].c], 100 * parts[i].m / gas);
        if (w < 0 || (size_t)w >= left) break;
        out += w; left -= (size_t)w;
    }
    snprintf(hp->crust, sizeof hp->crust, "silicates %.0f%% (granite share %.0f%%), iron ore %.1f%%, sulfides %.1f%%, water %.2f%%, organics %.3f%%",
             100 * cm[CC_SILICATE] / M, 100.0 * g->granite_q8 / 256, 100 * cm[CC_IRON_OXIDE] / M, 100 * cm[CC_SULFIDE] / M,
             100 * hp->water_frac, 100 * hp->organic_frac);

    /* exact composition for the life simulation (written as atm=): mole fractions
     * (class mass / molar mass) of the gases the planet keeps, in ATM_* order. The
     * atmosphere= text above stays in mass percent. CO and He have no ATM_* slot
     * and are left out. A volcanic CO2 floor is applied below: planet_load only
     * adds its baseline to text-parsed mixes, never to an exact atm= line. */
    if (keeps_atm && gas > 0) {
        static const struct { int cc, atm; double mm; } G[] = {
            { CC_H2S, ATM_H2S, 34.081 }, { CC_CH4, ATM_CH4, 16.043 }, { CC_NH3, ATM_NH3, 17.031 }, { CC_CO2, ATM_CO2, 44.010 },
            { CC_O2, ATM_O2, 31.999 },   { CC_N2, ATM_N2, 28.014 },   { CC_H2, ATM_H2, 2.016 },
        };
        double sum = 0;
        for (size_t i = 0; i < sizeof G / sizeof G[0]; ++i) {
            if (G[i].cc == CC_H2 && !keeps_light) continue;     /* escaped, as in the text summary */
            const double n = cm[G[i].cc] > 0 ? cm[G[i].cc] / G[i].mm : 0;
            hp->atm_mix[G[i].atm] = n;
            sum += n;
        }
        if (sum > 0) {
            for (int i = 0; i < ATM_N; ++i) hp->atm_mix[i] /= sum;
            /* volcanic outgassing keeps some CO2 in any atmosphere, even when the
             * crust chemistry put all its carbon into methane and organics; life
             * needs it as a carbon source (dissolved inorganic carbon). */
            const double co2_floor = 0.005; /* tuned */
            const double co2 = hp->atm_mix[ATM_CO2];
            if (co2 < co2_floor) {
                const double k = (1.0 - co2_floor) / (1.0 - co2);
                for (int i = 0; i < ATM_N; ++i) hp->atm_mix[i] *= k;
                hp->atm_mix[ATM_CO2] = co2_floor;
            }
            hp->atm_known = 1;
        }
    }
}

static void print_chem(const ChemResult* r) {
    const double M = chem_total_mass(r);
    printf("\nchemistry: %d reactors, %.0f atoms\n", r->reactors_run, r->atoms);
    for (int c = 0; c < CC_COUNT; ++c)
        if (r->class_mass[c] > 0) printf("  %-20s %8.4f%% of mass  (%.0f molecules)\n", CC_NAME[c], 100 * r->class_mass[c] / M, r->class_molecules[c]);
    printf("  bonds per element pair (most common):");
    double best[6] = { 0 };
    int ba[6] = { 0 }, bb[6] = { 0 };
    for (int a = 0; a < E_COUNT; ++a) for (int b = a; b < E_COUNT; ++b) {
        const double v = r->bonds[a][b];
        for (int k = 0; k < 6; ++k) if (v > best[k]) {
            for (int j = 5; j > k; --j) { best[j] = best[j - 1]; ba[j] = ba[j - 1]; bb[j] = bb[j - 1]; }
            best[k] = v; ba[k] = a; bb[k] = b; break;
        }
    }
    for (int k = 0; k < 6 && best[k] > 0; ++k) printf(" %s-%s %.0f", ELEMENT_SYMBOL[ba[k]], ELEMENT_SYMBOL[bb[k]], best[k]);
    printf("\n  carbon: %.0f atoms, %.2f different partner elements per atom (silicon: %.2f), %d distinct carbon compounds, largest organic molecule %d atoms\n",
           r->c_atoms, r->c_atoms > 0 ? r->c_partner_kinds / r->c_atoms : 0, r->si_atoms > 0 ? r->si_partner_kinds / r->si_atoms : 0,
           r->n_formulas, r->largest_organic);
}

static void print_planet(const HomePlanet* p) {
    static const char* mats[MAT_COUNT] = { "air", "basalt", "soil", "vegetation", "sand", "water", "granite", "iron ore", "sulfide", "ice", "regolith",
                                           "microbial mat", "leaf", "wood", "kelp" };
    printf("\nhome planet #%d: %.2f Earth masses, %.2f Earth radii, gravity %.2f g, orbit %.3f AU (e %.3f)\n", p->planet_index,
           p->mass_me, p->radius_re, p->gravity_g, p->a_au, p->ecc);
    printf("  temperature: %.0f K equilibrium, %.0f K surface (%.0f C)\n", p->t_eq, p->t_surf, p->t_surf - 273.15);
    printf("  atmosphere : %s\n  crust      : %s\n", p->atmosphere, p->crust);
    if (p->atm_known)
        printf("  atm (mol)  : H2S %.4f CH4 %.4f NH3 %.4f CO2 %.4f O2 %.4f N2 %.4f H2 %.4f, planet hash %016llx\n",
               p->atm_mix[ATM_H2S], p->atm_mix[ATM_CH4], p->atm_mix[ATM_NH3], p->atm_mix[ATM_CO2], p->atm_mix[ATM_O2],
               p->atm_mix[ATM_N2], p->atm_mix[ATM_H2], (unsigned long long)planet_hash(p));
    printf("  surface    : oceans cover %.0f%% (%s), land %s, soil depth %d, relief x%.2f, %d other planets in the sky\n",
           100 * p->ocean_cover, p->ocean_cover > 0.01 ? mats[p->gen.ocean_top_mat] : "none", mats[p->gen.surface_mat],
           p->gen.soil_depth, p->gen.relief_q8 / 256.0, p->nbodies);
}

/* ---- CLI -------------------------------------------------------------- */
typedef struct Args {
    const char* stage;
    const char* dir;
    CosmosConfig cosmos;
    int device, planet, shard, shards, reactors, atoms, sweeps, threads;
} Args;

static void usage(void) {
    printf("usage: gme_genesis <all|cosmos|score|chem|finish> [options]\n"
           "  --dir PATH        run directory (default genesis)\n"
           "  --seed N          system seed (cosmos)\n"
           "  --particles N     planetesimals (default 8192)   --years Y  simulated years (default 3000)\n"
           "  --device K        GPU ordinal                    --snapshots N  PNG frames of the disk\n"
           "  --planet K        override the home planet choice\n"
           "  --reactors N --atoms N --sweeps N   chemistry size (default 64 x 16384 atoms x 300)\n"
           "  --shard I --shards N                run only this slice of the reactors (chem)\n"
           "  score DIR...      rank finished cosmos runs\n"
           "life stages (own options, docs/biosphere-spec.md): env, life, life-test, bioexport, biostat, biomap-test\n"
           "  e.g. gme_genesis env --scout; gme_genesis env --island 5; gme_genesis life --island 5 --resume\n");
}

/* src/bio/bio_main.c: the life stages; returns -1 when argv[1] is not one of them */
int bio_main(int argc, char** argv);

static int run_cosmos(Args* a) {
    RtDevice dev;
    if (rt_init(a->device, &dev)) { fprintf(stderr, "GPU init failed: %s\n", rt_error()); return 1; }
    printf("GPU %d: %s (sm_%d%d, %d SMs, %.1f GiB)\n", a->device, dev.name, dev.cc_major, dev.cc_minor, dev.sm_count,
           dev.total_mem / 1073741824.0);
    MKDIR(a->dir);
    a->cosmos.snapshot_dir = a->dir;
    CosmosResult r;
    if (cosmos_run(&a->cosmos, &r)) return 1;
    System s = { r.star, r.frost_line_au, r.n, r.bodies, a->cosmos.seed };
    system_save(a->dir, &s);
    printf("cosmos done in %.1f s: %lld steps, %lld mergers, %d lost; %d survivors\n", r.wall_seconds, r.steps, r.collisions, r.ejected, r.n);
    print_system(&s);
    const int h = choose_home(&s);
    if (h >= 0) printf("\nbest home candidate: #%d (habitability %+.2f)\n", h, habitability(&s, &s.b[h]));
    else printf("\nno rocky planet suitable for life in this system\n");
    free(r.bodies);
    return 0;
}

static int prepare_chem(Args* a, System* s, int* home, ChemConfig* cc) {
    if (system_load(a->dir, s)) { fprintf(stderr, "no %s/system.txt - run the cosmos stage first\n", a->dir); return 1; }
    *home = a->planet >= 0 ? a->planet : choose_home(s);
    if (*home < 0 || *home >= s->n) { fprintf(stderr, "no habitable rocky planet in %s; try another --seed\n", a->dir); return 1; }
    memset(cc, 0, sizeof *cc);
    crust_inventory(&s->b[*home], cc->comp);
    cc->t_hot = 2500;
    cc->t_cold = t_equilibrium(s, body_a(s, &s->b[*home]));
    cc->atoms = a->atoms; cc->reactors = a->reactors; cc->sweeps = a->sweeps;
    cc->seed = s->seed * 7919u + (uint32_t)*home;
    cc->shard = a->shard; cc->shards = a->shards;
    return 0;
}

static int run_chem(Args* a) {
    System s; int home; ChemConfig cc;
    if (prepare_chem(a, &s, &home, &cc)) return 1;
    printf("chem: planet #%d, cooling %.0f K -> %.0f K, reactors %d..%d of %d (%d atoms, %d sweeps) on %d threads\n", home,
           cc.t_hot, cc.t_cold, cc.reactors * cc.shard / cc.shards, cc.reactors * (cc.shard + 1) / cc.shards - 1, cc.reactors,
           cc.atoms, cc.sweeps, jobs_thread_count());
    const double t0 = time_now();
    ChemResult r;
    chem_run(&cc, &r);
    char path[512];
    snprintf(path, sizeof path, "%s/chem_%d_of_%d.txt", a->dir, cc.shard, cc.shards);
    chem_save(path, &r);
    printf("chem shard done in %.1f s -> %s\n", time_now() - t0, path);
    free(s.b);
    return 0;
}

static int run_finish(Args* a) {
    System s; int home; ChemConfig cc;
    if (prepare_chem(a, &s, &home, &cc)) return 1;
    ChemResult all, part;
    memset(&all, 0, sizeof all);
    for (int i = 0; i < a->shards; ++i) {
        char path[512];
        snprintf(path, sizeof path, "%s/chem_%d_of_%d.txt", a->dir, i, a->shards);
        if (chem_load(path, &part)) { fprintf(stderr, "missing %s\n", path); return 1; }
        chem_merge(&all, &part);
    }
    print_chem(&all);
    HomePlanet hp;
    derive_planet(&s, home, &all, &hp);
    char path[512];
    snprintf(path, sizeof path, "%s/home.planet", a->dir);
    planet_save(path, &hp);
    print_planet(&hp);
    printf("\nwrote %s\n", path);
    free(s.b);
    return 0;
}

static int run_score(int argc, char** argv, int first) {
    printf("%-40s %8s  %s\n", "run", "score", "home candidate");
    for (int i = first; i < argc; ++i) {
        if (argv[i][0] == '-') continue;
        System s;
        if (system_load(argv[i], &s)) { printf("%-40s %8s\n", argv[i], "missing"); continue; }
        const int h = choose_home(&s);
        if (h < 0) printf("%-40s %8s  none\n", argv[i], "-");
        else {
            const Body* b = &s.b[h];
            printf("%-40s %+8.2f  #%d %.2f Me at %.2f AU, T_eq %.0f K\n", argv[i], habitability(&s, b), h, b->m / M_EARTH_SUN,
                   body_a(&s, b), t_equilibrium(&s, body_a(&s, b)));
        }
        free(s.b);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }
    {   /* life stages are dispatched before the option parser below: they take their own options */
        const int brc = bio_main(argc, argv);
        if (brc >= 0) return brc;
    }
    platform_init();
    Args a;
    memset(&a, 0, sizeof a);
    a.stage = argv[1];
    a.dir = "genesis";
    cosmos_default_config(&a.cosmos);
    a.planet = -1; a.shards = 1; a.reactors = 64; a.atoms = 16384; a.sweeps = 300; a.threads = 0;
    for (int i = 2; i < argc; ++i) {
        const char* k = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : "";
        if (k[0] != '-') continue;
        if      (!strcmp(k, "--dir"))       a.dir = v;
        else if (!strcmp(k, "--seed"))      a.cosmos.seed = (uint32_t)strtoul(v, NULL, 10);
        else if (!strcmp(k, "--particles")) a.cosmos.particles = atoi(v);
        else if (!strcmp(k, "--years"))     a.cosmos.years = atof(v);
        else if (!strcmp(k, "--snapshots")) a.cosmos.snapshots = atoi(v);
        else if (!strcmp(k, "--device"))    a.device = atoi(v);
        else if (!strcmp(k, "--planet"))    a.planet = atoi(v);
        else if (!strcmp(k, "--shard"))     a.shard = atoi(v);
        else if (!strcmp(k, "--shards"))    a.shards = atoi(v);
        else if (!strcmp(k, "--reactors"))  a.reactors = atoi(v);
        else if (!strcmp(k, "--atoms"))     a.atoms = atoi(v);
        else if (!strcmp(k, "--sweeps"))    a.sweeps = atoi(v);
        else if (!strcmp(k, "--threads"))   a.threads = atoi(v);
        else { usage(); return 2; }
        ++i;
    }
    SysInfo si;
    sys_query(&si);
    jobs_init(a.threads > 0 ? a.threads - 1 : si.logical_cpus - 1);

    int rc = 2;
    if (!strcmp(a.stage, "cosmos")) rc = run_cosmos(&a);
    else if (!strcmp(a.stage, "chem")) rc = run_chem(&a);
    else if (!strcmp(a.stage, "finish")) rc = run_finish(&a);
    else if (!strcmp(a.stage, "score")) rc = run_score(argc, argv, 2);
    else if (!strcmp(a.stage, "all")) {
        rc = run_cosmos(&a);
        if (!rc) { a.shard = 0; a.shards = 1; rc = run_chem(&a); }
        if (!rc) rc = run_finish(&a);
    } else usage();

    jobs_shutdown();
    platform_shutdown();
    return rc;
}

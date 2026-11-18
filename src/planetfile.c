/* planetfile.c */
#include "planetfile.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ATM_CO2_BASELINE 0.005   /* tuned: genesis rarely lists CO2, life needs some carbon source */

void planet_default(HomePlanet* p) {
    memset(p, 0, sizeof *p);
    p->gen = gme_planet_default();
    p->star_mass = 1; p->star_lum = 1; p->star_teff = 5778;
    p->mass_me = 1; p->radius_re = 1; p->a_au = 1; p->gravity_g = 1;
    p->t_eq = 255; p->t_surf = 288;
    p->sun_color[0] = 1.0f; p->sun_color[1] = 0.95f; p->sun_color[2] = 0.86f;
    p->sun_intensity = 1.f; p->sun_size = 0.03f;
    p->zenith[0] = 0.22f; p->zenith[1] = 0.45f; p->zenith[2] = 0.82f;
    p->horizon[0] = 0.70f; p->horizon[1] = 0.80f; p->horizon[2] = 0.93f;
    p->sun_ecl[0] = 1.f;
    snprintf(p->atmosphere, sizeof p->atmosphere, "default (no genesis run)");
    snprintf(p->crust, sizeof p->crust, "default");
    /* atm_mix all zero, atm_known = 0: composition unknown */
}

/* CO2 baseline + renormalisation. Idempotent: a mix that already has CO2 and
 * sums to 1 is left bit-for-bit unchanged, so load(save(p)) and planet_hash
 * on an in-memory planet agree. */
static void atm_normalize(double m[ATM_N]) {
    double s = 0;
    for (int i = 0; i < ATM_N; ++i) {
        if (!(m[i] > 0)) m[i] = 0;   /* also clears NaN */
        s += m[i];
    }
    if (s <= 0) return;
    if (m[ATM_CO2] <= 0) { m[ATM_CO2] = ATM_CO2_BASELINE; s += ATM_CO2_BASELINE; }
    if (fabs(s - 1.0) > 1e-9)
        for (int i = 0; i < ATM_N; ++i) m[i] /= s;
}

/* gas name -> ATM_* (names as written by genesis, see CC_NAME in genesis/chem.c) */
static int atm_gas(const char* name) {
    static const struct { const char* prefix; int gas; } T[] = {
        { "hydrogen sulfide", ATM_H2S }, { "hydrogen sulphide", ATM_H2S }, { "h2s", ATM_H2S },
        { "methane", ATM_CH4 }, { "ch4", ATM_CH4 },
        { "ammonia", ATM_NH3 }, { "nh3", ATM_NH3 },
        { "carbon dioxide", ATM_CO2 }, { "co2", ATM_CO2 },
        { "oxygen", ATM_O2 }, { "o2", ATM_O2 },
        { "nitrogen", ATM_N2 }, { "n2", ATM_N2 },
        { "hydrogen", ATM_H2 }, { "h2", ATM_H2 },   /* after "hydrogen sulfide" */
    };
    char low[64];
    size_t n = 0;
    for (; name[n] && n + 1 < sizeof low; ++n) low[n] = (char)tolower((unsigned char)name[n]);
    low[n] = 0;
    for (size_t i = 0; i < sizeof T / sizeof T[0]; ++i) {
        const size_t k = strlen(T[i].prefix);
        if (!strncmp(low, T[i].prefix, k) && (low[k] == 0 || low[k] == ' ')) return T[i].gas;
    }
    return -1;
}

/* "hydrogen sulfide 51%, methane 39%, ammonia 11%, oxygen O2 0%" -> fractions.
 * Returns the number of recognised gases (unknown ones, e.g. helium, are skipped). */
static int atm_from_text(const char* s, double m[ATM_N]) {
    int found = 0;
    for (int i = 0; i < ATM_N; ++i) m[i] = 0;
    while (*s) {
        const char* e = strchr(s, ',');
        const size_t len = e ? (size_t)(e - s) : strlen(s);
        char buf[96];
        const size_t n = len < sizeof buf - 1 ? len : sizeof buf - 1;
        memcpy(buf, s, n);
        buf[n] = 0;
        s += len + (e ? 1 : 0);
        char* pct = strrchr(buf, '%');
        if (!pct) continue;
        char* q = pct;
        while (q > buf && (isdigit((unsigned char)q[-1]) || q[-1] == '.')) --q;
        if (q == pct) continue;
        const double v = strtod(q, NULL) / 100.0;
        while (q > buf && q[-1] == ' ') --q;
        *q = 0;
        char* name = buf;
        while (*name == ' ') ++name;
        const int g = atm_gas(name);
        if (g < 0) continue;
        m[g] += v;
        ++found;
    }
    return found;
}

int planet_save(const char* path, const HomePlanet* p) {
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    const GmePlanet* g = &p->gen;
    fprintf(f, "# GME home planet (written by gme_genesis)\n");
    fprintf(f, "system_seed=%u\nplanet_index=%d\n", p->system_seed, p->planet_index);
    fprintf(f, "star=%.6g %.6g %.6g\n", p->star_mass, p->star_lum, p->star_teff);
    fprintf(f, "body=%.6g %.6g %.6g %.6g %.6g\n", p->mass_me, p->radius_re, p->a_au, p->ecc, p->gravity_g);
    fprintf(f, "temperature=%.6g %.6g\n", p->t_eq, p->t_surf);
    fprintf(f, "inventory=%.6g %.6g %.6g %.6g\n", p->water_frac, p->organic_frac, p->ocean_cover, p->atm_frac);
    fprintf(f, "gen=%d %d %d %d %d %d %d %d %d %d %d\n", g->sea_level, g->relief_q8, g->granite_q8, g->ore_band,
            g->sulfide_band, g->soil_depth, g->surface_mat, g->beach_mat, g->ocean_mat, g->ocean_top_mat, g->cave_width);
    fprintf(f, "sun=%.4f %.4f %.4f %.4f %.4f\n", p->sun_color[0], p->sun_color[1], p->sun_color[2], p->sun_intensity, p->sun_size);
    fprintf(f, "sky=%.4f %.4f %.4f %.4f %.4f %.4f\n", p->zenith[0], p->zenith[1], p->zenith[2], p->horizon[0], p->horizon[1], p->horizon[2]);
    fprintf(f, "sun_dir=%.6f %.6f %.6f\n", p->sun_ecl[0], p->sun_ecl[1], p->sun_ecl[2]);
    for (int i = 0; i < p->nbodies; ++i) {
        const SkyBody* b = &p->bodies[i];
        fprintf(f, "sky_body=%.6f %.6f %.6f %.6f %.4f %.4f %.4f\n", b->dir[0], b->dir[1], b->dir[2], b->ang, b->color[0], b->color[1], b->color[2]);
    }
    fprintf(f, "atmosphere=%s\ncrust=%s\n", p->atmosphere, p->crust);
    if (p->atm_known) {   /* exact round trip (%.17g), order ATM_*: H2S CH4 NH3 CO2 O2 N2 H2 */
        fprintf(f, "atm=");
        for (int i = 0; i < ATM_N; ++i) fprintf(f, "%s%.17g", i ? " " : "", p->atm_mix[i]);
        fprintf(f, "\n");
    }
    const int bad = ferror(f);
    if (fclose(f) || bad) return -1;
    return 0;
}

int planet_load(const char* path, HomePlanet* p) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    planet_default(p);
    p->nbodies = 0;
    char line[512];
    GmePlanet* g = &p->gen;
    double atm[ATM_N];
    int have_atm = 0;
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        SkyBody b;
        if (sscanf(line, "system_seed=%u", &p->system_seed) == 1) continue;
        if (sscanf(line, "planet_index=%d", &p->planet_index) == 1) continue;
        if (sscanf(line, "star=%lf %lf %lf", &p->star_mass, &p->star_lum, &p->star_teff) == 3) continue;
        if (sscanf(line, "body=%lf %lf %lf %lf %lf", &p->mass_me, &p->radius_re, &p->a_au, &p->ecc, &p->gravity_g) == 5) continue;
        if (sscanf(line, "temperature=%lf %lf", &p->t_eq, &p->t_surf) == 2) continue;
        if (sscanf(line, "inventory=%lf %lf %lf %lf", &p->water_frac, &p->organic_frac, &p->ocean_cover, &p->atm_frac) == 4) continue;
        if (sscanf(line, "gen=%d %d %d %d %d %d %d %d %d %d %d", &g->sea_level, &g->relief_q8, &g->granite_q8, &g->ore_band,
                   &g->sulfide_band, &g->soil_depth, &g->surface_mat, &g->beach_mat, &g->ocean_mat, &g->ocean_top_mat,
                   &g->cave_width) == 11) continue;
        if (sscanf(line, "sun=%f %f %f %f %f", &p->sun_color[0], &p->sun_color[1], &p->sun_color[2], &p->sun_intensity, &p->sun_size) == 5) continue;
        if (sscanf(line, "sky=%f %f %f %f %f %f", &p->zenith[0], &p->zenith[1], &p->zenith[2], &p->horizon[0], &p->horizon[1], &p->horizon[2]) == 6) continue;
        if (sscanf(line, "sun_dir=%f %f %f", &p->sun_ecl[0], &p->sun_ecl[1], &p->sun_ecl[2]) == 3) continue;
        if (sscanf(line, "sky_body=%f %f %f %f %f %f %f", &b.dir[0], &b.dir[1], &b.dir[2], &b.ang, &b.color[0], &b.color[1], &b.color[2]) == 7) {
            if (p->nbodies < PLANET_MAX_SKY_BODIES) p->bodies[p->nbodies++] = b;
            continue;
        }
        if (!strncmp(line, "atm=", 4)) {
            if (sscanf(line + 4, "%lf %lf %lf %lf %lf %lf %lf", &atm[0], &atm[1], &atm[2], &atm[3], &atm[4], &atm[5],
                       &atm[6]) == ATM_N) have_atm = 1;
            continue;
        }
        if (!strncmp(line, "atmosphere=", 11)) { snprintf(p->atmosphere, sizeof p->atmosphere, "%s", line + 11); continue; }
        if (!strncmp(line, "crust=", 6)) { snprintf(p->crust, sizeof p->crust, "%s", line + 6); continue; }
    }
    fclose(f);

    if (have_atm) {
        memcpy(p->atm_mix, atm, sizeof atm);
        p->atm_known = 1;
    } else {
        p->atm_known = atm_from_text(p->atmosphere, p->atm_mix) > 0;
    }
    if (p->atm_known) atm_normalize(p->atm_mix);
    double s = 0;
    for (int i = 0; i < ATM_N; ++i) s += p->atm_mix[i];
    if (!(s > 0)) { memset(p->atm_mix, 0, sizeof p->atm_mix); p->atm_known = 0; }
    return 0;
}

uint64_t planet_hash(const HomePlanet* p) {
    double m[ATM_N];
    memcpy(m, p->atm_mix, sizeof m);
    if (p->atm_known) atm_normalize(m);
    const GmePlanet* g = &p->gen;
    char s[512];
    int n = snprintf(s, sizeof s, "%u %d %d %d %d %d %d %d %d %d %d %d %d %lld %lld %lld", p->system_seed,
                     p->planet_index, g->sea_level, g->relief_q8, g->granite_q8, g->ore_band, g->sulfide_band,
                     g->soil_depth, g->surface_mat, g->beach_mat, g->ocean_mat, g->ocean_top_mat, g->cave_width,
                     (long long)llround(p->t_surf * 100.0), (long long)llround(p->star_lum * 1e4),
                     (long long)llround(p->a_au * 1e4));
    for (int i = 0; i < ATM_N && n > 0 && (size_t)n < sizeof s; ++i)
        n += snprintf(s + n, sizeof s - (size_t)n, " %lld", (long long)llround(m[i] * 1e4));
    uint64_t h = 0xCBF29CE484222325ull;
    for (int i = 0; i < n && (size_t)i < sizeof s; ++i) { h ^= (uint8_t)s[i]; h *= 0x100000001B3ull; }
    return h;
}

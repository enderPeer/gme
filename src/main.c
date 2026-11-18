/* main.c - GME: detect hardware, size the world to it, then either open the
 * interactive 3D view (default) or run the headless streaming benchmark.
 * Also the render harness (docs/render-upgrade-spec.md 7.1): --view, --perf,
 * --metrics, --raw / --compare, --settle and --selftest. */
#include "biomap.h"
#include "image.h"
#include "jobs.h"
#include "planetfile.h"
#include "platform.h"
#include "window.h"
#include "world.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_BIOMAP  "life/home.biomap"
#define DEFAULT_REGIONS "life/regions.txt"

static const char* QUALITY_NAME[RQ_COUNT] = { "classic", "low", "medium", "high", "ref" };

typedef struct Options {
    double   seconds, speed, vram, ram, scale;
    int      cpu, radius, bench, width, height, fps, frames, seed_set;
    uint32_t seed, batch;
    const char* screenshot;
    const char* planet;
    const char* biomap;            /* --biomap file */
    int      no_life, biomap_test; /* --no-life, --biomap-test */
    int      spawn_set;            /* --spawn X Z [EYE] */
    int32_t  spawn_x, spawn_z, spawn_eye;
    int      look_set;             /* --look YAW PITCH (degrees) */
    float    look_yaw, look_pitch;
    int      region;               /* --region K: start at the coast of biomap region K (default 0) */
    /* renderer + harness */
    int      quality;              /* --quality classic|low|medium|high|ref (RQ_*) */
    int      debug;                /* --debug normals|albedo|dev (RD_*) */
    const char* view;              /* --view coast|peak|forest|shelf|far */
    int      eye_y_set;            /* --eye-y Y: absolute eye height */
    int32_t  eye_y;
    int      settle, raw, metrics, perf, perf_ab, fly, selftest;
    const char* cmp_a;             /* --compare A.raw B.raw */
    const char* cmp_b;
} Options;

static void usage(void) {
    printf("usage: gme [view options] | gme --bench [bench options] | gme --selftest | gme --compare A.raw B.raw\n"
           "  view : [--width N] [--height N] [--scale 0.25..1] [--fps N (0 = uncapped)]\n"
           "         [--frames N] [--screenshot file.png] [--raw (also file.raw, BGRA)]\n"
           "         [--quality classic|low|medium|high|ref] (default medium; V toggles classic, K cycles)\n"
           "         [--debug normals|albedo|dev] [--settle] [--perf] [--perf-ab] [--fly] [--metrics]\n"
           "  bench: [--seconds N] [--speed m/s] [--cpu]\n"
           "  both : [--vram 0..1] [--ram 0..1] [--batch chunks] [--radius chunks] [--seed N]\n"
           "         [--planet file] (default genesis/home.planet if present)\n"
           "         [--spawn X Z [EYE]] (start column, eye EYE m above ground, default 50)\n"
           "         [--look YAW PITCH] (start view direction, degrees) [--eye-y Y] (absolute eye height)\n"
           "         [--view coast|peak|forest|shelf|far] (deterministic spawn + look from the generator)\n"
           "  life : [--biomap file] (default " DEFAULT_BIOMAP " if present) | [--no-life]\n"
           "         | [--biomap-test] (built-in pattern of every vegetation kind around the spawn)\n"
           "         [--region K] (start at the coast of biomap region K, default 0)\n"
           "  test : --selftest (D1 density + D2 flora; exit code = number of failures)\n");
}

static int parse_quality(const char* v) {
    for (int q = 0; q < RQ_COUNT; ++q) if (!strcmp(v, QUALITY_NAME[q])) return q;
    return -1;
}

static int parse(int argc, char** argv, Options* o) {
    memset(o, 0, sizeof *o);
    o->seconds = 20; o->speed = 250; o->vram = 0.6; o->ram = 0.5;
    o->seed = 1337; o->batch = 2048;
    o->width = 1280; o->height = 720; o->scale = 1.0; o->fps = 60;
    o->quality = RQ_MEDIUM;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--cpu"))   { o->cpu = 1; o->bench = 1; continue; }
        if (!strcmp(a, "--bench")) { o->bench = 1; continue; }
        if (!strcmp(a, "--no-life"))     { o->no_life = 1; continue; }
        if (!strcmp(a, "--biomap-test")) { o->biomap_test = 1; continue; }
        if (!strcmp(a, "--settle"))   { o->settle = 1; continue; }
        if (!strcmp(a, "--raw"))      { o->raw = 1; continue; }
        if (!strcmp(a, "--metrics"))  { o->metrics = 1; continue; }
        if (!strcmp(a, "--perf"))     { o->perf = 1; continue; }
        if (!strcmp(a, "--perf-ab"))  { o->perf = o->perf_ab = 1; continue; }
        if (!strcmp(a, "--fly"))      { o->fly = 1; continue; }
        if (!strcmp(a, "--selftest")) { o->selftest = 1; continue; }
        if (!strcmp(a, "--compare")) {
            if (i + 2 >= argc) { usage(); return -1; }
            o->cmp_a = argv[i + 1]; o->cmp_b = argv[i + 2];
            i += 2;
            continue;
        }
        if (!strcmp(a, "--spawn") || !strcmp(a, "--look")) {
            if (i + 2 >= argc) { usage(); return -1; }
            if (a[2] == 's') {
                o->spawn_x = (int32_t)atoi(argv[i + 1]);
                o->spawn_z = (int32_t)atoi(argv[i + 2]);
                o->spawn_set = 1;
                o->spawn_eye = 50;
                if (i + 3 < argc && argv[i + 3][0] != '-') { o->spawn_eye = (int32_t)atoi(argv[i + 3]); ++i; }
            } else {
                o->look_yaw = (float)atof(argv[i + 1]);
                o->look_pitch = (float)atof(argv[i + 2]);
                o->look_set = 1;
            }
            i += 2;
            continue;
        }
        if (!strcmp(a, "--help") || !strcmp(a, "-h") || !v) { usage(); return -1; }
        if      (!strcmp(a, "--width"))      o->width = atoi(v);
        else if (!strcmp(a, "--height"))     o->height = atoi(v);
        else if (!strcmp(a, "--scale"))      o->scale = atof(v);
        else if (!strcmp(a, "--fps"))        o->fps = atoi(v);
        else if (!strcmp(a, "--frames"))     o->frames = atoi(v);
        else if (!strcmp(a, "--screenshot")) o->screenshot = v;
        else if (!strcmp(a, "--planet"))     o->planet = v;
        else if (!strcmp(a, "--biomap"))     o->biomap = v;
        else if (!strcmp(a, "--region"))     o->region = atoi(v);
        else if (!strcmp(a, "--seconds")) o->seconds = atof(v);
        else if (!strcmp(a, "--speed"))   o->speed = atof(v);
        else if (!strcmp(a, "--vram"))    o->vram = atof(v);
        else if (!strcmp(a, "--ram"))     o->ram = atof(v);
        else if (!strcmp(a, "--batch"))   o->batch = (uint32_t)atoi(v);
        else if (!strcmp(a, "--radius"))  o->radius = atoi(v);
        else if (!strcmp(a, "--seed"))    { o->seed = (uint32_t)strtoul(v, NULL, 10); o->seed_set = 1; }
        else if (!strcmp(a, "--view"))    o->view = v;
        else if (!strcmp(a, "--eye-y"))   { o->eye_y = (int32_t)atoi(v); o->eye_y_set = 1; }
        else if (!strcmp(a, "--quality")) {
            o->quality = parse_quality(v);
            if (o->quality < 0) { fprintf(stderr, "--quality: classic, low, medium, high or ref\n"); return -1; }
        } else if (!strcmp(a, "--debug")) {
            if      (!strcmp(v, "normals")) o->debug = RD_NORMALS;
            else if (!strcmp(v, "albedo"))  o->debug = RD_ALBEDO;
            else if (!strcmp(v, "dev"))     o->debug = RD_DEV;
            else if (!strcmp(v, "none"))    o->debug = RD_NONE;
            else { fprintf(stderr, "--debug: normals, albedo or dev in M1 (gi and band come with M2/M4)\n"); return -1; }
        }
        else { usage(); return -1; }
        ++i;
    }
    if (o->no_life + o->biomap_test + (o->biomap != NULL) > 1) {
        fprintf(stderr, "--no-life, --biomap and --biomap-test exclude each other\n");
        return -1;
    }
    if (o->view && strcmp(o->view, "coast") && strcmp(o->view, "peak") && strcmp(o->view, "forest") &&
        strcmp(o->view, "shelf") && strcmp(o->view, "far")) {
        fprintf(stderr, "--view: coast, peak, forest, shelf or far\n");
        return -1;
    }
    return 0;
}

/* ---- spawn point ----------------------------------------------------------- */
typedef struct Spawn { int32_t x, z, y; float yaw, pitch; } Spawn;

/* eye metres above the ground, or above the water surface at sea */
static void spawn_above(int32_t x, int32_t z, int32_t eye, uint32_t seed, const GmePlanet* pl, Spawn* s) {
    const int32_t h = gme_height(x, z, seed, pl);
    s->x = x; s->z = z;
    s->y = (h > pl->sea_level ? h : pl->sea_level) + eye;
    if (s->y > 250) s->y = 250;
    s->yaw = 0.6f; s->pitch = -0.25f;
}

/* without life: spiral out from the origin until the terrain is hilly */
static void spawn_default(uint32_t seed, const GmePlanet* pl, Spawn* s) {
    int32_t sx = 16, sz = 16;
    const int32_t want = (pl->sea_level > -200 ? pl->sea_level : 0) + 40;
    for (int ring = 0, found = 0; ring < 128 && !found; ++ring)
        for (int k = -ring; k <= ring && !found; ++k) {
            const int32_t c[4][2] = { { k, -ring }, { k, ring }, { -ring, k }, { ring, k } };
            for (int e = 0; e < 4 && !found; ++e) {
                const int32_t x = c[e][0] * 64, z = c[e][1] * 64;
                if (gme_height(x, z, seed, pl) > want) { sx = x; sz = z; found = 1; }
            }
        }
    spawn_above(sx, sz, 50, seed, pl, s);
}

/* with life: the low land column nearest to (cx, cz) that looks over open
 * water deep enough for kelp; the camera stands a little inland, facing the
 * water, so shore mats, kelp and land plants are all in view. 0 = found. */
static int spawn_coast(int32_t cx, int32_t cz, uint32_t seed, const GmePlanet* pl, Spawn* s) {
    const int32_t sea = pl->sea_level, step = 16;
    for (int ring = 0; ring <= 320; ++ring) {                /* up to ~5 km */
        for (int k = -ring; k <= ring; ++k) {
            const int32_t c[4][2] = { { k, -ring }, { k, ring }, { -ring, k }, { ring, k } };
            for (int e = 0; e < 4; ++e) {
                const int32_t x = cx + c[e][0] * step, z = cz + c[e][1] * step;
                const int32_t h = gme_height(x, z, seed, pl);
                if (h < sea + 3 || h > sea + 24) continue;  /* low land near a shore */
                float wx = 0.f, wz = 0.f;
                int wet = 0;
                for (int d = 0; d < 16; ++d) {
                    const float a = (float)d * (6.2831853f / 16.f), ca = cosf(a), sa = sinf(a);
                    if (gme_height(x + (int32_t)lrintf(ca * 96.f), z + (int32_t)lrintf(sa * 96.f), seed, pl) >= sea - 5) continue;
                    if (gme_height(x + (int32_t)lrintf(ca * 48.f), z + (int32_t)lrintf(sa * 48.f), seed, pl) >= sea - 2) continue;
                    wx += ca; wz += sa; ++wet;
                }
                const float l = sqrtf(wx * wx + wz * wz);
                if (wet < 4 || l < 1e-3f) continue;
                const float ux = wx / l, uz = wz / l;
                s->x = x - (int32_t)lrintf(ux * 40.f);
                s->z = z - (int32_t)lrintf(uz * 40.f);
                int32_t g = gme_height(s->x, s->z, seed, pl);
                if (g < sea) g = sea;
                s->y = g + 22 > 250 ? 250 : g + 22;
                s->yaw = atan2f(uz, ux);
                s->pitch = -0.30f;
                return 0;
            }
        }
    }
    return -1;
}

/* ---- harness views (spec 7.1): deterministic spawn + look from the generator ---- */
static void look_at(Spawn* s, float x, float y, float z) {
    const float dx = x - (float)s->x, dy = y - (float)s->y, dz = z - (float)s->z;
    s->yaw = atan2f(dz, dx);
    s->pitch = atan2f(dy, sqrtf(dx * dx + dz * dz) + 1e-3f);
}

/* peak: the highest h on a 64 m lattice within 4 km of the coast spawn c, eye 60 m, looking at c */
static void view_peak(const Spawn* c, uint32_t seed, const GmePlanet* pl, Spawn* s) {
    int32_t bx = c->x, bz = c->z, bh = -1 << 30;
    for (int32_t dz = -4032; dz <= 4032; dz += 64)
        for (int32_t dx = -4032; dx <= 4032; dx += 64) {
            if ((int64_t)dx * dx + (int64_t)dz * dz > 4000ll * 4000ll) continue;
            const int32_t h = gme_height(c->x + dx, c->z + dz, seed, pl);
            if (h > bh) { bh = h; bx = c->x + dx; bz = c->z + dz; }
        }
    spawn_above(bx, bz, 60, seed, pl, s);
    look_at(s, (float)c->x, (float)c->y, (float)c->z);
}

/* shelf: the column nearest the coast spawn c with water 8-15 m deep, eye 2 m
 * under the water plane (y_w = sea_level + 1), looking toward the shore */
static int view_shelf(const Spawn* c, uint32_t seed, const GmePlanet* pl, Spawn* s) {
    const int32_t sea = pl->sea_level;
    for (int ring = 0; ring <= 1000; ++ring)
        for (int k = -ring; k <= ring; ++k) {
            const int32_t e4[4][2] = { { k, -ring }, { k, ring }, { -ring, k }, { ring, k } };
            for (int e = 0; e < 4; ++e) {
                const int32_t x = c->x + e4[e][0] * 4, z = c->z + e4[e][1] * 4;
                const int32_t depth = sea - gme_height(x, z, seed, pl);
                if (depth < 8 || depth > 15) continue;
                s->x = x; s->z = z; s->y = sea + 1 - 2;
                look_at(s, (float)c->x, (float)s->y, (float)c->z);
                s->pitch = -0.20f;
                return 0;
            }
        }
    return -1;
}

/* far: the peak view with pitch 0, toward the direction that stays open longest */
static void view_far(const Spawn* peak, uint32_t seed, const GmePlanet* pl, Spawn* s) {
    *s = *peak;
    float best_yaw = peak->yaw, best = -1.f;
    for (int k = 0; k < 64; ++k) {
        const float a = (float)k * (6.2831853f / 64.f), ca = cosf(a), sa = sinf(a);
        float open = 0.f;
        for (float r = 16.f; r <= 8000.f; r += 16.f) {
            if (gme_height(peak->x + (int32_t)lrintf(ca * r), peak->z + (int32_t)lrintf(sa * r), seed, pl) >= peak->y) break;
            open = r;
        }
        if (open > best) { best = open; best_yaw = a; }
    }
    s->yaw = best_yaw;
    s->pitch = 0.f;
}

/* ---- life ------------------------------------------------------------------ */
static int file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

static void print_biomap(const char* what, const Biomap* b) {
    static const char* kinds[6] = { "bare", "mat", "kelp", "cover", "shrub", "tree" };
    uint64_t n[6] = { 0 }, other = 0;
    for (size_t i = 0; i < b->ncells; ++i) {
        const uint32_t k = BIOCELL_KIND(b->cells[i]);
        if (k < 6) n[k]++; else other++;
    }
    char gates[128] = "none";
    for (int g = 0, w = 0; g < 32; ++g)   /* at most 32 x 4 chars: always fits */
        if (b->gates & (1u << g)) w += snprintf(gates + w, sizeof gates - (size_t)w, "%sG%d", w ? " " : "", g);
    printf("Life : %s - sim year %u, gates %s, %d region%s of %d m cells, fallback %s\n", what, b->sim_year, gates,
           b->nreg, b->nreg == 1 ? "" : "s", 1 << b->shift, b->fallback_on ? "on" : "off");
    for (int r = 0; r < b->nreg; ++r)
        printf("       region %d: x %d..%lld, z %d..%lld (%dx%d cells)\n", r, b->reg[r].x0,
               (long long)b->reg[r].x0 + ((long long)b->reg[r].w << b->shift), b->reg[r].z0,
               (long long)b->reg[r].z0 + ((long long)b->reg[r].h << b->shift), b->reg[r].w, b->reg[r].h);
    if (b->ncells) {
        printf("       %.2fM cells:", (double)b->ncells / 1e6);
        for (int k = 0; k < 6; ++k) printf(" %s %.1f%%", kinds[k], 100.0 * (double)n[k] / (double)b->ncells);
        if (other) printf(" unknown %llu", (unsigned long long)other);
        printf("\n");
    }
}

static double gib(double b) { return b / (1024.0 * 1024.0 * 1024.0); }

static void print_line(const char* tag, double t, const WorldPos* p, const WorldStats* s,
                       double rate, double tick_avg, double tick_max) {
    printf("%s t=%6.1fs  pos=(%7d,%4d,%7d)  R=%3d (%5d m)  chunks %8u (%4.1f%% uniform, %u pending)  "
           "slots %4.1f%%  gen %7.0f ch/s  tick %.2f/%.2f ms%s\n",
           tag, t, p->cx * GME_CHUNK_DIM + (int)p->lx, p->cy * GME_CHUNK_DIM + (int)p->ly,
           p->cz * GME_CHUNK_DIM + (int)p->lz, s->radius, s->radius * GME_CHUNK_DIM, s->ready,
           s->ready ? 100.0 * s->uniform_ready / s->ready : 0.0, s->pending,
           100.0 * s->slots_used / (s->slots_total ? s->slots_total : 1), rate, tick_avg, tick_max,
           s->fully_loaded ? "  [loaded]" : "");
    fflush(stdout);
}

/* ---- raw frames: --raw / --compare ------------------------------------------------ */
/* "GMERAW1\0", int32 width, int32 height (little-endian), then width*height
 * uint32 pixels 0xAARRGGBB (BGRA in memory). */
static int write_raw(const char* png_path, const uint32_t* px, int w, int h) {
    char path[1024];
    snprintf(path, sizeof path, "%s", png_path);
    char* dot = strrchr(path, '.');
    if (dot && !strchr(dot, '/') && !strchr(dot, '\\')) *dot = 0;
    if (strlen(path) + 5 >= sizeof path) return -1;
    strcat(path, ".raw");
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    const int32_t dim[2] = { w, h };
    int ok = fwrite("GMERAW1", 1, 8, f) == 8 && fwrite(dim, 4, 2, f) == 2 &&
             fwrite(px, 4, (size_t)w * h, f) == (size_t)w * h;
    ok = (fclose(f) == 0) && ok;
    if (ok) printf("saved %s\n", path);
    return ok ? 0 : -1;
}

static uint32_t* read_raw(const char* path, int* w, int* h) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[8];
    int32_t dim[2];
    uint32_t* px = NULL;
    if (fread(magic, 1, 8, f) == 8 && !memcmp(magic, "GMERAW1", 8) && fread(dim, 4, 2, f) == 2 &&
        dim[0] > 0 && dim[1] > 0 && dim[0] <= 16384 && dim[1] <= 16384) {
        px = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)dim[0] * dim[1]);
        if (px && fread(px, 4, (size_t)dim[0] * dim[1], f) != (size_t)dim[0] * dim[1]) { free(px); px = NULL; }
        *w = dim[0]; *h = dim[1];
    }
    fclose(f);
    return px;
}

/* identical-pixel %, PSNR over RGB and the largest channel difference; exit
 * code 0 identical, 1 different, 2 unreadable / different sizes */
static int compare_raw(const char* a, const char* b) {
    int wa = 0, ha = 0, wb = 0, hb = 0;
    uint32_t* A = read_raw(a, &wa, &ha);
    uint32_t* B = read_raw(b, &wb, &hb);
    if (!A || !B || wa != wb || ha != hb) {
        fprintf(stderr, "compare: %s\n", (!A || !B) ? "cannot read a GMERAW1 file" : "different frame sizes");
        free(A); free(B);
        return 2;
    }
    const size_t n = (size_t)wa * ha;
    size_t same = 0;
    double se = 0;
    int dmax = 0;
    for (size_t i = 0; i < n; ++i) {
        int diff = 0;
        for (int k = 0; k < 3; ++k) {
            const int d = (int)((A[i] >> (8 * k)) & 255u) - (int)((B[i] >> (8 * k)) & 255u);
            se += (double)d * d;
            diff |= d;
            if (abs(d) > dmax) dmax = abs(d);
        }
        same += diff == 0;
    }
    const double mse = se / (3.0 * (double)n);
    printf("compare %s vs %s: %dx%d, identical pixels %.4f%% (%zu of %zu), PSNR %s%.2f dB, max difference %d\n", a, b,
           wa, ha, 100.0 * (double)same / (double)n, same, n, mse > 0 ? "" : "inf ", mse > 0 ? 10.0 * log10(65025.0 / mse) : 0.0,
           dmax);
    free(A); free(B);
    return same == n ? 0 : 1;
}

/* ---- interactive viewer --------------------------------------------------- */
/* Planet sky -> world-space light. The star sits at a fixed afternoon
 * elevation; the ecliptic is rotated so the star lands there, which puts the
 * other planets where they really are relative to it. */
static void build_sky(const HomePlanet* hp, RenderSky* s) {
    memset(s, 0, sizeof *s);
    float w1[3] = { 0.45f, 0.75f, 0.30f };
    float l = sqrtf(w1[0] * w1[0] + w1[1] * w1[1] + w1[2] * w1[2]);
    for (int i = 0; i < 3; ++i) w1[i] /= l;
    /* ecliptic basis: e1 = to star, e3 = orbit normal, e2 = e3 x e1 */
    const float* e1 = hp->sun_ecl;
    const float e3[3] = { 0, 0, 1 };
    float e2[3] = { e3[1] * e1[2] - e3[2] * e1[1], e3[2] * e1[0] - e3[0] * e1[2], e3[0] * e1[1] - e3[1] * e1[0] };
    l = sqrtf(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
    if (l < 1e-6f) { e2[0] = 0; e2[1] = 1; e2[2] = 0; l = 1; }
    for (int i = 0; i < 3; ++i) e2[i] /= l;
    /* world basis: w1 = sun; the ecliptic is tilted ~40 degrees from the horizon */
    float w3[3] = { -0.55f, 0.55f, 0.63f };
    float dp = w3[0] * w1[0] + w3[1] * w1[1] + w3[2] * w1[2];
    for (int i = 0; i < 3; ++i) w3[i] -= dp * w1[i];
    l = sqrtf(w3[0] * w3[0] + w3[1] * w3[1] + w3[2] * w3[2]);
    for (int i = 0; i < 3; ++i) w3[i] /= l;
    const float w2[3] = { w3[1] * w1[2] - w3[2] * w1[1], w3[2] * w1[0] - w3[0] * w1[2], w3[0] * w1[1] - w3[1] * w1[0] };

    memcpy(s->sun_dir, w1, 12);
    memcpy(s->sun_color, hp->sun_color, 12);
    s->sun_intensity = hp->sun_intensity;
    s->sun_size = hp->sun_size;
    memcpy(s->zenith, hp->zenith, 12);
    memcpy(s->horizon, hp->horizon, 12);
    for (int b = 0; b < hp->nbodies && b < RENDER_MAX_BODIES; ++b) {
        const float* d = hp->bodies[b].dir;
        const float c1 = d[0] * e1[0] + d[1] * e1[1] + d[2] * e1[2];
        const float c2 = d[0] * e2[0] + d[1] * e2[1] + d[2] * e2[2];
        const float c3 = d[2];
        for (int i = 0; i < 3; ++i) s->body_dir[s->nbodies][i] = c1 * w1[i] + c2 * w2[i] + c3 * w3[i];
        s->body_ang[s->nbodies] = hp->bodies[b].ang;
        memcpy(s->body_color[s->nbodies], hp->bodies[b].color, 12);
        s->nbodies++;
    }
}

/* p50 / p95 / max of n samples (sorted in place) */
static int cmp_double(const void* a, const void* b) {
    const double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}
static void pct(double* v, uint32_t n, double* p50, double* p95, double* mx) {
    *p50 = *p95 = *mx = 0;
    if (!n) return;
    qsort(v, n, sizeof(double), cmp_double);
    *p50 = v[(n - 1) / 2];
    *p95 = v[(uint32_t)((n - 1) * 0.95 + 0.5)];
    *mx = v[n - 1];
}

/* --perf: perf_<view>_<quality>.csv next to the screenshot (or in the current folder) */
static void perf_csv_path(const Options* o, int quality, char* out, size_t n) {
    char dir[768] = "";
    if (o->screenshot) {
        snprintf(dir, sizeof dir, "%s", o->screenshot);
        char* s1 = strrchr(dir, '/');
        char* s2 = strrchr(dir, '\\');
        char* s = s1 > s2 ? s1 : s2;
        if (s) s[1] = 0; else dir[0] = 0;
    }
    snprintf(out, n, "%sperf_%s%s_%s.csv", dir, o->view ? o->view : "spawn", o->fly ? "_fly" : "", QUALITY_NAME[quality]);
}

static int run_viewer(World* w, const Options* o, WorldPos cam, const HomePlanet* hp, float yaw, float pitch, Renderer* rd) {
    RenderSky sky;
    build_sky(hp, &sky);
    Window* win = window_create("GME", o->width, o->height);
    if (!win || !rd) {
        fprintf(stderr, "viewer: could not create window/renderer\n");
        window_destroy(win);
        return 0;
    }

    printf("Controls: hold right mouse = look | WASD = move | Space/C = up/down | Shift = x8 speed\n"
           "          wheel = base speed | F = shadows | V = classic / smooth | K = quality | P = screenshot | Esc = quit\n");
    printf("Render: quality %s (%s)\n", QUALITY_NAME[o->quality],
           o->quality == RQ_CLASSIC ? "classic voxel ray-marcher" : "smooth surface: column tracer, M1 lighting");
    fflush(stdout);

    float speed = 40.f;
    int shadows = 1, shot = 0, quality = o->quality, last_smooth = o->quality != RQ_CLASSIC ? o->quality : RQ_MEDIUM;
    uint64_t frames = 0, drawn = 0;
    double last = time_now(), title_t = last, acc_world = 0, acc_frame = 0;
    float acc_gpu = 0;
    int acc_n = 0;
    const uint32_t* px = NULL;
    int rw = 0, rh = 0;
    int settled = !o->settle;
    double fly_t = 0;
    /* --perf host samples */
    const uint32_t cap = o->frames > 0 ? (uint32_t)o->frames + 16 : 1u << 16;
    double* s_world = o->perf ? (double*)malloc(sizeof(double) * cap) : NULL;
    double* s_present = o->perf ? (double*)malloc(sizeof(double) * cap) : NULL;
    double* s_frame = o->perf ? (double*)malloc(sizeof(double) * cap) : NULL;
    float* s_gpu = o->perf ? (float*)malloc(sizeof(float) * cap * RENDER_PERF_STAGES) : NULL;
    /* --perf-ab: every frame also draws the classic path (alternating order), same camera and world state */
    float* s_gpu_c = o->perf_ab ? (float*)malloc(sizeof(float) * cap * RENDER_PERF_STAGES) : NULL;
    float last_c[RENDER_PERF_STAGES] = { 0 }, last_new[RENDER_PERF_STAGES] = { 0 };
    uint32_t ns = 0;
    if (!settled) printf("settle: waiting for the world to load before counting frames\n");

    while (window_pump(win)) {
        const double now = time_now();
        double dt = now - last;
        last = now;
        if (dt > 0.1) dt = 0.1;

        /* --- input --- */
        if (window_key_pressed(win, KEY_ESC)) break;
        if (window_key_pressed(win, 'F')) shadows = !shadows;
        if (window_key_pressed(win, 'V')) {          /* A/B: classic vs the smooth renderer */
            quality = quality == RQ_CLASSIC ? last_smooth : RQ_CLASSIC;
            printf("quality %s\n", QUALITY_NAME[quality]); fflush(stdout);
        }
        if (window_key_pressed(win, 'K')) {
            quality = (quality + 1) % RQ_COUNT;
            if (quality != RQ_CLASSIC) last_smooth = quality;
            printf("quality %s\n", QUALITY_NAME[quality]); fflush(stdout);
        }
        const int wheel = window_wheel(win);
        if (wheel) { speed *= powf(1.25f, (float)wheel); if (speed < 1.f) speed = 1.f; if (speed > 5000.f) speed = 5000.f; }
        int mdx, mdy;
        window_mouse_delta(win, &mdx, &mdy);
        yaw += (float)mdx * 0.0025f;
        pitch -= (float)mdy * 0.0025f;
        if (pitch > 1.55f) pitch = 1.55f;
        if (pitch < -1.55f) pitch = -1.55f;

        if (o->fly && settled) {                      /* the bench flight path: fixed 1/60 s steps */
            fly_t += 1.0 / 60.0;
            const double heading = 0.6 * sin(fly_t * 0.05);
            cam.lx += (float)(cos(heading) * o->speed / 60.0);
            cam.lz += (float)(sin(heading) * o->speed / 60.0);
            yaw = (float)heading;
            worldpos_normalize(&cam);
        }
        const float cp = cosf(pitch), sp = sinf(pitch), cy = cosf(yaw), sy = sinf(yaw);
        float mx = 0, my = 0, mz = 0;
        if (window_key_down(win, 'W')) { mx += cp * cy; my += sp; mz += cp * sy; }
        if (window_key_down(win, 'S')) { mx -= cp * cy; my -= sp; mz -= cp * sy; }
        if (window_key_down(win, 'D')) { mx -= sy; mz += cy; }
        if (window_key_down(win, 'A')) { mx += sy; mz -= cy; }
        if (window_key_down(win, KEY_SPACE)) my += 1.f;
        if (window_key_down(win, 'C') || window_key_down(win, KEY_CTRL)) my -= 1.f;
        const float len = sqrtf(mx * mx + my * my + mz * mz);
        if (len > 0.f) {
            const float v = speed * (window_key_down(win, KEY_SHIFT) ? 8.f : 1.f) * (float)dt / len;
            cam.lx += mx * v; cam.ly += my * v; cam.lz += mz * v;
            worldpos_normalize(&cam);
        }

        /* --- stream the world around the camera --- */
        const double tw = time_now();
        if (world_update(w, &cam)) break;
        const double t_world = time_now() - tw;
        acc_world += t_world;
        if (!settled) {
            WorldStats ws;
            world_get_stats(w, &ws);
            if (ws.fully_loaded) {
                settled = 1;
                renderer_perf_reset(rd);
                printf("settle: world loaded after %llu frames (radius %d, %u tiles)\n", (unsigned long long)drawn,
                       ws.radius, ws.tiles_ready);
                fflush(stdout);
            }
        }

        /* --- render straight from the GPU chunk pool --- */
        int cw, ch;
        double t_present = 0;
        window_client_size(win, &cw, &ch);
        if (cw > 0 && ch > 0) {
            rw = (int)(cw * o->scale); rh = (int)(ch * o->scale);
            if (rw < 64) rw = 64;
            if (rh < 36) rh = 36;
            RenderScene sc;
            world_get_scene(w, &sc);
            const RenderCamera rc = {
                .cx = cam.cx, .cy = cam.cy, .cz = cam.cz, .lx = cam.lx, .ly = cam.ly, .lz = cam.lz,
                .yaw = yaw, .pitch = pitch, .fov_y = 1.15f, .shadows = shadows, .sky = &sky,
                .quality = quality, .debug = o->debug, .frame = (uint32_t)drawn, .sun_dir = { 0.f, 0.f, 0.f }, .ev = NAN,
            };
            if (o->perf_ab && quality != RQ_CLASSIC) {
                RenderCamera rcc = rc;
                rcc.quality = RQ_CLASSIC;
                RenderScene sc2 = sc;                     /* the device copies are current after the first draw */
                sc2.page_dirty = 0;
                sc2.tile_dirty = 0;
                const int classic_first = (drawn & 1) == 0;
                if (renderer_draw(rd, &sc, classic_first ? &rcc : &rc, rw, rh, &px)) break;
                float first[RENDER_PERF_STAGES];
                memcpy(first, renderer_perf_last(rd), sizeof first);
                if (renderer_draw(rd, &sc2, classic_first ? &rc : &rcc, rw, rh, &px)) break;
                if (classic_first) {
                    memcpy(last_c, first, sizeof first);
                } else {                                  /* new path drawn first: keep its times, show it */
                    memcpy(last_c, renderer_perf_last(rd), sizeof last_c);
                    float keep[RENDER_PERF_STAGES];
                    memcpy(keep, first, sizeof keep);
                    if (renderer_draw(rd, &sc2, &rc, rw, rh, &px)) break;
                    memcpy(first, keep, sizeof first);
                }
                memcpy(last_new, classic_first ? renderer_perf_last(rd) : first, sizeof last_new);
            } else {
                if (renderer_draw(rd, &sc, &rc, rw, rh, &px)) break;
                memcpy(last_new, renderer_perf_last(rd), sizeof last_new);
            }
            const double tp = time_now();
            window_present(win, px, rw, rh);
            t_present = time_now() - tp;
            acc_gpu += renderer_gpu_ms(rd);
            ++drawn;
            if (window_key_pressed(win, 'P')) {
                char name[64];
                snprintf(name, sizeof name, "screenshot_%03d.png", shot++);
                if (!write_png(name, px, rw, rh)) printf("saved %s\n", name);
            }
        }
        acc_frame += time_now() - now;
        ++acc_n;
        if (settled) {
            ++frames;
            if (o->perf && ns < cap) {
                s_world[ns] = t_world * 1000.0;
                s_present[ns] = t_present * 1000.0;
                s_frame[ns] = (time_now() - now) * 1000.0;
                memcpy(s_gpu + (size_t)ns * RENDER_PERF_STAGES, last_new, sizeof last_new);
                if (s_gpu_c) memcpy(s_gpu_c + (size_t)ns * RENDER_PERF_STAGES, last_c, sizeof last_c);
                ++ns;
            }
        }

        if (now - title_t >= 0.25 && acc_n) {
            WorldStats s;
            world_get_stats(w, &s);
            char t[320];
            snprintf(t, sizeof t,
                     "GME | %.0f fps | gpu render %.1f ms | world %.2f ms | %dx%d | pos %d %d %d | view %d m | %u chunks%s | speed %.0f m/s | shadows %s | %s",
                     acc_n / (now - title_t), acc_gpu / acc_n, acc_world * 1000.0 / acc_n, rw, rh,
                     cam.cx * GME_CHUNK_DIM + (int)cam.lx, cam.cy * GME_CHUNK_DIM + (int)cam.ly,
                     cam.cz * GME_CHUNK_DIM + (int)cam.lz, s.radius * GME_CHUNK_DIM, s.ready,
                     s.fully_loaded ? "" : " (streaming)", speed, shadows ? "on" : "off", QUALITY_NAME[quality]);
            window_set_title(win, t);
            if (o->frames) { printf("%s\n", t); fflush(stdout); }
            title_t = now;
            acc_world = acc_frame = 0;
            acc_gpu = 0;
            acc_n = 0;
        }
        if (o->frames && frames >= (uint64_t)o->frames) break;

        if (o->fps > 0) { /* cap the frame rate so streaming keeps GPU time */
            const double left = 1.0 / o->fps - (time_now() - now);
            if (left > 0.0015) sleep_ms((int)(left * 1000.0));
        }
    }
    if (o->screenshot && px) {
        if (!write_png(o->screenshot, px, rw, rh)) printf("saved %s\n", o->screenshot);
        if (o->raw) write_raw(o->screenshot, px, rw, rh);
    }
    if (o->perf && ns) {
        RenderPerf rp;
        renderer_perf(rd, &rp);
        double p50, p95, mxv;
        double* tmp = (double*)malloc(sizeof(double) * ns);
        printf("perf: %u frames, quality %s, %s%s\n", ns, QUALITY_NAME[quality], o->view ? o->view : "spawn",
               o->fly ? " flight (bench path)" : "");
        for (int k = 0; k < RENDER_PERF_STAGES && tmp; ++k) {
            for (uint32_t i = 0; i < ns; ++i) tmp[i] = s_gpu[(size_t)i * RENDER_PERF_STAGES + k];
            pct(tmp, ns, &p50, &p95, &mxv);
            printf("perf: %-13s p50 %7.3f  p95 %7.3f  max %7.3f ms\n", rp.name[k], p50, p95, mxv);
        }
        if (s_gpu_c && tmp) {                           /* the same frames, classic path */
            double c95[2] = { 0, 0 }, n95[2] = { 0, 0 };
            for (int k = 0; k < RENDER_PERF_STAGES; ++k) {
                if (k == 2 || k == 3 || k == 4) continue; /* classic has no surface / sun / compose */
                for (uint32_t i = 0; i < ns; ++i) tmp[i] = s_gpu_c[(size_t)i * RENDER_PERF_STAGES + k];
                pct(tmp, ns, &p50, &p95, &mxv);
                printf("perf: classic %-7s p50 %7.3f  p95 %7.3f  max %7.3f ms\n", k == 1 ? "k_render" : rp.name[k], p50, p95, mxv);
            }
            for (int pass = 0; pass < 2; ++pass) {      /* frame kernels (stages 1-4), gpu_total */
                const float* src_ = pass ? s_gpu_c : s_gpu;
                for (uint32_t i = 0; i < ns; ++i) {
                    const float* g = src_ + (size_t)i * RENDER_PERF_STAGES;
                    tmp[i] = g[1] + g[2] + g[3] + g[4];
                }
                pct(tmp, ns, &p50, &p95, &mxv);
                (pass ? c95 : n95)[0] = p95;
                for (uint32_t i = 0; i < ns; ++i) tmp[i] = src_[(size_t)i * RENDER_PERF_STAGES + RENDER_PERF_STAGES - 1];
                pct(tmp, ns, &p50, &p95, &mxv);
                (pass ? c95 : n95)[1] = p95;
            }
            printf("perf-ab: p95 frame kernels new %.3f ms vs classic %.3f ms (ratio %.2f); p95 gpu_total new %.3f vs classic %.3f ms (ratio %.2f)\n",
                   n95[0], c95[0], c95[0] > 0 ? n95[0] / c95[0] : 0.0, n95[1], c95[1], c95[1] > 0 ? n95[1] / c95[1] : 0.0);
        }
        if (tmp) {
            memcpy(tmp, s_world, sizeof(double) * ns); pct(tmp, ns, &p50, &p95, &mxv);
            printf("perf: %-13s p50 %7.3f  p95 %7.3f  max %7.3f ms\n", "world_update", p50, p95, mxv);
            memcpy(tmp, s_present, sizeof(double) * ns); pct(tmp, ns, &p50, &p95, &mxv);
            printf("perf: %-13s p50 %7.3f  p95 %7.3f  max %7.3f ms\n", "present", p50, p95, mxv);
            memcpy(tmp, s_frame, sizeof(double) * ns); pct(tmp, ns, &p50, &p95, &mxv);
            printf("perf: %-13s p50 %7.3f  p95 %7.3f  max %7.3f ms\n", "frame (cpu)", p50, p95, mxv);
        }
        free(tmp);
        char path[1024];
        perf_csv_path(o, quality, path, sizeof path);
        FILE* f = fopen(path, "w");
        if (f) {
            fprintf(f, "frame");
            for (int k = 0; k < RENDER_PERF_STAGES; ++k) fprintf(f, ",%s_ms", rp.name[k]);
            if (s_gpu_c) for (int k = 0; k < RENDER_PERF_STAGES; ++k) fprintf(f, ",classic_%s_ms", rp.name[k]);
            fprintf(f, ",world_update_ms,present_ms,frame_ms\n");
            for (uint32_t i = 0; i < ns; ++i) {
                const float* g = s_gpu + (size_t)i * RENDER_PERF_STAGES;
                fprintf(f, "%u", i);
                for (int k = 0; k < RENDER_PERF_STAGES; ++k) fprintf(f, ",%.4f", g[k]);
                if (s_gpu_c) for (int k = 0; k < RENDER_PERF_STAGES; ++k) fprintf(f, ",%.4f", s_gpu_c[(size_t)i * RENDER_PERF_STAGES + k]);
                fprintf(f, ",%.4f,%.4f,%.4f\n", s_world[i], s_present[i], s_frame[i]);
            }
            fclose(f);
            printf("perf: wrote %s\n", path);
        }
        fflush(stdout);
    }
    if (o->metrics && px) {
        RenderMetrics m;
        if (!renderer_metrics(rd, &m)) {
            const double np = (double)(m.pixels ? m.pixels : 1);
            printf("metrics: %llu pixels | smooth: terrain %llu (%.1f%%), axis-aligned normals %llu = %.3f%% of terrain, "
                   "(exactly level %llu = %.3f%%), flora %llu, voxel mode %llu, sky %llu, under water %llu | cracks %llu = %.2e of pixels "
                   "(%llu below the smooth surface) | NaN %llu | classic: terrain %llu, axis-aligned 100%%\n",
                   (unsigned long long)m.pixels, (unsigned long long)m.terrain, 100.0 * (double)m.terrain / np,
                   (unsigned long long)m.axis_aligned, m.terrain ? 100.0 * (double)m.axis_aligned / (double)m.terrain : 0.0,
                   (unsigned long long)m.level, m.terrain ? 100.0 * (double)m.level / (double)m.terrain : 0.0,
                   (unsigned long long)m.flora, (unsigned long long)m.voxel_mode, (unsigned long long)m.sky,
                   (unsigned long long)m.water, (unsigned long long)m.cracks, (double)m.cracks / np,
                   (unsigned long long)m.cracks_below, (unsigned long long)m.nan, (unsigned long long)m.classic_terrain);
            if (o->screenshot && m.image) {      /* <screenshot>_metrics.png */
                char path[1024];
                snprintf(path, sizeof path, "%s", o->screenshot);
                char* dot = strrchr(path, '.');
                if (dot && !strchr(dot, '/') && !strchr(dot, '\\')) *dot = 0;
                if (strlen(path) + 13 < sizeof path) {
                    strcat(path, "_metrics.png");
                    if (!write_png(path, m.image, m.image_w, m.image_h)) printf("saved %s\n", path);
                }
            }
            fflush(stdout);
        }
    }
    free(s_world); free(s_present); free(s_frame); free(s_gpu); free(s_gpu_c);
    window_destroy(win);
    return 0;
}

/* ---- self-tests (spec 7.3): D1 density, D2 flora ------------------------------ */
typedef struct D1Window { int32_t x0, z0; } D1Window;

typedef struct D1Ctx {
    int32_t x0, z0, N;              /* window origin incl. a 4-column margin; N = W + 8 */
    uint32_t seed;
    const GmePlanet* pl;
    int32_t* H;                     /* gme_surface_pair's h */
    int32_t* T;                     /* stored T, q8 */
    double*  raw_row;               /* per row: sum of the unclamped deviation */
    double*  dev_row;               /* per row: sum of the stored deviation */
    uint32_t* hbad_row;             /* per row: h != gme_height */
} D1Ctx;

static void d1_row_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    D1Ctx* c = (D1Ctx*)p;
    for (uint32_t j = b; j < e; ++j) {
        double rs = 0, ds = 0;
        uint32_t hb = 0;
        for (int32_t i = 0; i < c->N; ++i) {
            const int32_t x = c->x0 + i, z = c->z0 + (int32_t)j;
            int32_t h, t;
            gme_surface_pair(x, z, c->seed, c->pl, &h, &t);
            const int32_t dev = gme_tile_dev_q7(t, h, c->pl);
            const size_t k = (size_t)j * c->N + i;
            c->H[k] = h;
            c->T[k] = 256 * (h + 1) + 2 * dev;
            hb += h != gme_height(x, z, c->seed, c->pl);
            if (i >= 4 && i < c->N - 4 && j >= 4 && (int32_t)j < c->N - 4) {
                rs += (double)t / 256.0 - (h + 1);
                ds += (double)dev / GME_DEV_UNIT;
            }
        }
        c->raw_row[j] = rs; c->dev_row[j] = ds; c->hbad_row[j] = hb;
    }
}

typedef struct D1TileCtx {
    const GmeTileJob* jobs;
    GmeColTile* out;
    uint32_t seed;
    const GmePlanet* pl;
    const GmeBioView* bv;
} D1TileCtx;

static void d1_tile_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    D1TileCtx* c = (D1TileCtx*)p;
    for (uint32_t i = b; i < e; ++i) gme_col_tile_build(&c->out[i], c->jobs[i].cx, c->jobs[i].cz, c->seed, c->pl, c->bv);
}

/* the island windows: life/regions.txt when it belongs to this planet, else the
 * biomap's regions, else a ring of 6 windows around the default spawn */
static int d1_windows(uint64_t phash, uint32_t seed, const Biomap* bm, int have_life, int32_t cx, int32_t cz,
                      D1Window* win, char* from, size_t nfrom) {
    int n = 0;
    FILE* f = fopen(DEFAULT_REGIONS, "r");
    if (f) {
        char line[512];
        unsigned long long fh = 0;
        unsigned fs = 0;
        int32_t size = 16384;
        int match = 0;
        while (fgets(line, sizeof line, f)) {
            if (line[0] == '#') {
                if (sscanf(line, "# planet %llx seed %u", &fh, &fs) == 2) match = fh == phash && fs == seed;
                int sz;
                if (sscanf(line, "# window %d m", &sz) == 1 && sz > 0) size = sz;
                continue;
            }
            int k;
            int32_t x0, z0;
            if (match && n < 16 && sscanf(line, "%d %d %d", &k, &x0, &z0) == 3) {
                win[n].x0 = x0 + size / 2 - 1024;
                win[n].z0 = z0 + size / 2 - 1024;
                ++n;
            }
        }
        fclose(f);
        if (n) { snprintf(from, nfrom, "%s (%d islands)", DEFAULT_REGIONS, n); return n; }
    }
    if (have_life && bm->nreg > 0) {
        for (int r = 0; r < bm->nreg && n < 16; ++r) {
            win[n].x0 = (int32_t)(bm->reg[r].x0 + (((int64_t)bm->reg[r].w << bm->shift) >> 1)) - 1024;
            win[n].z0 = (int32_t)(bm->reg[r].z0 + (((int64_t)bm->reg[r].h << bm->shift) >> 1)) - 1024;
            win[n].x0 &= ~31; win[n].z0 &= ~31;
            ++n;
        }
        snprintf(from, nfrom, "biomap regions (%d)", n);
        return n;
    }
    for (int k = 0; k < 6; ++k) {
        const double a = k * 1.0471975512;
        win[n].x0 = ((cx + (int32_t)(20000.0 * cos(a))) & ~31) - 1024;
        win[n].z0 = ((cz + (int32_t)(20000.0 * sin(a))) & ~31) - 1024;
        ++n;
    }
    snprintf(from, nfrom, "6 windows 20 km around the default spawn");
    return n;
}

static int selftest_d1(uint32_t seed, const GmePlanet* pl, uint64_t phash, const Biomap* bm, int have_life,
                       const Spawn* sp, int gpu_ok) {
    D1Window win[16];
    char from[160];
    const int nw = d1_windows(phash, seed, bm, have_life, sp->x, sp->z, win, from, sizeof from);
    const int32_t W = 2048, N = W + 8;
    printf("D1 density: %d windows of %d^2 columns from %s, sea_level %d, relief_q8 %d\n", nw, W, from, pl->sea_level,
           pl->relief_q8);
    D1Ctx c;
    memset(&c, 0, sizeof c);
    c.N = N; c.seed = seed; c.pl = pl;
    c.H = (int32_t*)malloc(sizeof(int32_t) * (size_t)N * N);
    c.T = (int32_t*)malloc(sizeof(int32_t) * (size_t)N * N);
    c.raw_row = (double*)malloc(sizeof(double) * N);
    c.dev_row = (double*)malloc(sizeof(double) * N);
    c.hbad_row = (uint32_t*)malloc(sizeof(uint32_t) * N);
    if (!c.H || !c.T || !c.raw_row || !c.dev_row || !c.hbad_row) {
        fprintf(stderr, "D1: out of memory\n");
        free(c.H); free(c.T); free(c.raw_row); free(c.dev_row); free(c.hbad_row);
        return 1;
    }
    uint64_t cols = 0, hbad = 0, devbad = 0, coast_bad = 0, coast_cols = 0, pairs = 0, ledges = 0, n4 = 0, flat4 = 0, d2n = 0;
    uint64_t tiles = 0, sat = 0, nlevel = 0, nexact = 0, nlevel_dry = 0, ndry = 0;
    double d2sum = 0, wmin = 1e9, wmax = -1e9, rmin = 1e9, rmax = -1e9;
    int32_t hrange_max = 0;
    const int32_t sea_q8 = 256 * (pl->sea_level + 1);
    for (int wi = 0; wi < nw; ++wi) {
        c.x0 = win[wi].x0 - 4; c.z0 = win[wi].z0 - 4;
        jobs_parallel_for((uint32_t)N, 8, d1_row_fn, &c);
        double rs = 0, ds = 0;
        for (int32_t j = 0; j < N; ++j) { rs += c.raw_row[j]; ds += c.dev_row[j]; hbad += c.hbad_row[j]; }
        const double wc = (double)W * W;
        if (ds / wc < wmin) wmin = ds / wc;
        if (ds / wc > wmax) wmax = ds / wc;
        if (rs / wc < rmin) rmin = rs / wc;
        if (rs / wc > rmax) rmax = rs / wc;
        const int32_t* H = c.H;
        const int32_t* T = c.T;
        for (int32_t j = 4; j < N - 4; ++j)
            for (int32_t i = 4; i < N - 4; ++i) {
                const size_t k = (size_t)j * N + i;
                ++cols;
                const int32_t dv = T[k] - 256 * (H[k] + 1);
                devbad += dv < -256 || dv > 256;               /* |T - (h+1)| <= 1 m */
                coast_cols++;
                coast_bad += (T[k] < sea_q8) != (H[k] < pl->sea_level);
                const double gx = (H[k + 4] - H[k - 4]) / 8.0, gz = (H[k + 4 * (size_t)N] - H[k - 4 * (size_t)N]) / 8.0;
                if (sqrt(gx * gx + gz * gz) < 0.3) {           /* gentle: RMS second difference */
                    const double ddx = (T[k + 1] - 2.0 * T[k] + T[k - 1]) / 256.0;
                    const double ddz = (T[k + N] - 2.0 * T[k] + T[k - N]) / 256.0;
                    d2sum += 0.5 * (ddx * ddx + ddz * ddz);
                    ++d2n;
                }
                {                                              /* corner shading normal (spec 3.3), informational */
                    const int32_t gxq = T[k + 1] - T[k - 1], gzq = T[k + N] - T[k - N];
                    const double gx = gxq / 512.0, gz = gzq / 512.0;
                    const int lv = gx * gx + gz * gz < 1.0 / (0.9995 * 0.9995) - 1.0;   /* max |n_i| > 0.9995 */
                    nlevel += lv;
                    nexact += gxq == 0 && gzq == 0;
                    if (H[k] >= pl->sea_level) { ++ndry; nlevel_dry += lv; }
                }
                pairs += 2;                                    /* ledges >= 45 deg between neighbours */
                ledges += abs(T[k + 1] - T[k]) >= 256;
                ledges += abs(T[k + N] - T[k]) >= 256;
                if (((i - 4) & 3) == 0 && ((j - 4) & 3) == 0) { /* exactly flat 4x4 */
                    int32_t mn = T[k], mx = T[k];
                    for (int b = 0; b < 4; ++b)
                        for (int a = 0; a < 4; ++a) {
                            const int32_t v = T[k + (size_t)b * N + a];
                            mn = v < mn ? v : mn; mx = v > mx ? v : mx;
                        }
                    ++n4;
                    flat4 += (mx - mn) < 4;
                }
            }
        /* tile saturation: h range over each tile's 36 x 36 columns */
        for (int32_t tj = 0; tj < W / 32; ++tj)
            for (int32_t ti = 0; ti < W / 32; ++ti) {
                int32_t mn = 1 << 30, mx = -(1 << 30);
                for (int32_t j = 0; j < GME_TILE_N; ++j)
                    for (int32_t i = 0; i < GME_TILE_N; ++i) {
                        const int32_t v = H[(size_t)(4 + tj * 32 - 2 + j) * N + 4 + ti * 32 - 2 + i];
                        mn = v < mn ? v : mn; mx = v > mx ? v : mx;
                    }
                ++tiles;
                sat += (mx - mn) > 255;
                if (mx - mn > hrange_max) hrange_max = mx - mn;
            }
    }
    free(c.H); free(c.T); free(c.raw_row); free(c.dev_row); free(c.hbad_row);

    const double d2 = d2n ? sqrt(d2sum / (double)d2n) : 0;
    const double ledge_share = pairs ? (double)ledges / (double)pairs : 0;
    const double flat_share = n4 ? (double)flat4 / (double)n4 : 0;
    const double wabs = fmax(fmax(fabs(wmin), fabs(wmax)), fmax(fabs(rmin), fabs(rmax)));
    printf("   %llu columns: h != gme_height %llu, |T-(h+1)| > 1 m %llu, coast mismatches %llu of %llu\n",
           (unsigned long long)cols, (unsigned long long)hbad, (unsigned long long)devbad, (unsigned long long)coast_bad,
           (unsigned long long)coast_cols);
    printf("   gentle RMS 2nd difference %.4f m (gate 0.10), ledges >= 45 deg %.4f%% (gate 0.2%%), flat 4x4 %.3f%% (gate 2%%)\n",
           d2, 100.0 * ledge_share, 100.0 * flat_share);
    printf("   window mean dev: stored %+.4f .. %+.4f m, unclamped %+.4f .. %+.4f m (gate |mean| <= 0.10)\n", wmin, wmax, rmin, rmax);
    printf("   tiles %llu: saturated %llu (max h range in a 36x36 tile %d m)\n", (unsigned long long)tiles,
           (unsigned long long)sat, hrange_max);
    printf("   corner normals within 1.8 deg of vertical (max |n_i| > 0.9995): %.2f%% of columns (land %.2f%%), "
           "exactly level %.2f%% [informational]\n", 100.0 * (double)nlevel / (double)cols,
           ndry ? 100.0 * (double)nlevel_dry / (double)ndry : 0.0, 100.0 * (double)nexact / (double)cols);

    /* 4,096 random tiles inside the windows: k_col_tiles == gme_col_tile_build */
    uint64_t gpu_bad = 0;
    int gpu_done = 0;
    if (gpu_ok) {
        const uint32_t nt = 4096;
        GmeTileJob* jobs = (GmeTileJob*)malloc(sizeof(GmeTileJob) * nt);
        GmeColTile* a = (GmeColTile*)malloc(sizeof(GmeColTile) * nt);
        GmeColTile* b = (GmeColTile*)malloc(sizeof(GmeColTile) * nt);
        if (jobs && a && b) {
            uint32_t rng = 0x9E3779B9u ^ seed;
            for (uint32_t k = 0; k < nt; ++k) {
                rng = rng * 1664525u + 1013904223u;
                const D1Window* wd = &win[(rng >> 8) % (uint32_t)nw];
                rng = rng * 1664525u + 1013904223u;
                const int32_t ox = (int32_t)((rng >> 8) % (uint32_t)(W / 32));
                rng = rng * 1664525u + 1013904223u;
                const int32_t oz = (int32_t)((rng >> 8) % (uint32_t)(W / 32));
                jobs[k].cx = (wd->x0 >> 5) + ox;
                jobs[k].cz = (wd->z0 >> 5) + oz;
                jobs[k].index = k;
                jobs[k].pad = 0;
            }
            GmeBioView bv;
            memset(&bv, 0, sizeof bv);
            if (have_life) biomap_view(bm, &bv);
            if (!gpu_build_tiles(jobs, nt, seed, pl, &bv, have_life ? bm->ncells : 0, a)) {
                D1TileCtx tc = { jobs, b, seed, pl, &bv };
                jobs_parallel_for(nt, 4, d1_tile_fn, &tc);
                uint64_t veg = 0;
                for (uint32_t k = 0; k < nt; ++k) {
                    const uint8_t* x = (const uint8_t*)&a[k];
                    const uint8_t* y = (const uint8_t*)&b[k];
                    for (size_t q = 0; q < sizeof(GmeColTile); ++q) gpu_bad += x[q] != y[q];
                    for (int s = 0; s < 16; ++s) veg += b[k].slot[s].cell != 0;
                }
                gpu_done = 1;
                printf("   %u random GPU tiles (k_col_tiles) vs CPU (gme_col_tile_build): %llu differing bytes; %llu vegetated slots in them\n",
                       nt, (unsigned long long)gpu_bad, (unsigned long long)veg);
            } else {
                printf("   GPU tiles: %s\n", gpu_last_error());
            }
        }
        free(jobs); free(a); free(b);
    } else {
        printf("   GPU tiles: skipped (no CUDA device)\n");
    }
    const int ok = !hbad && !devbad && !coast_bad && d2 <= 0.10 && ledge_share <= 0.002 && flat_share <= 0.02 &&
                   wabs <= 0.10 && !sat && (!gpu_ok || (gpu_done && !gpu_bad));
    printf("D1 %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

static int selftest_d2(uint32_t seed, const GmePlanet* plin) {
    GmePlanet pl = *plin;
    if (pl.sea_level < -150 || pl.sea_level > 150) pl.sea_level = 0;  /* room for 45 m of water and 40 m trees */
    uint64_t tvox = 0, tbad = 0, kvox = 0, kbad = 0, strands = 0;
    /* shrubs and trees: every ht 2..32 on flat ground, every voxel centre of the slot */
    const int32_t H0 = pl.sea_level + 20;
    for (int kind = BIO_KIND_SHRUB; kind <= BIO_KIND_TREE; ++kind)
        for (int32_t ht = 2; ht <= GME_FLORA_TREE_MAX; ++ht) {
            int32_t trad = ht / 4;
            trad = trad < 1 ? 1 : (trad > 3 ? 3 : trad);
            int32_t vr = ht / 3;
            vr = vr < trad ? trad : (vr > 8 ? 8 : vr);
            for (int32_t off = 0; off < 8 - 2 * trad; ++off) {      /* every trunk position in the slot */
                GmeSlot s;
                s.cell = BIOCELL(kind, 1, 255, ht, 0);
                s.tx = trad + off; s.tz = 7 - trad - off;
                s.tbase = H0; s.theight = ht; s.trad = trad; s.tvr = vr; s.tcy = H0 + ht - vr;
                for (int32_t z = 0; z < 8; ++z)
                    for (int32_t x = 0; x < 8; ++x) {
                        GmeColumn col;
                        gme_column_in_slot(x, z, H0, seed, &s, &col);
                        for (int32_t y = H0 + 1; y <= H0 + GME_FLORA_TREE_MAX + 4; ++y) {
                            const uint8_t m = gme_voxel(x, y, z, &col, seed, &pl);
                            const float dx = (float)(x - s.tx), dz = (float)(z - s.tz);
                            const uint8_t a = gme_trunk_inside_f(dx, (float)y + 0.5f, dz, &s) ? MAT_WOOD
                                            : gme_crown_inside_f(dx, (float)(y - s.tcy), dz, s.trad, s.tvr) ? MAT_LEAF : MAT_AIR;
                            ++tvox;
                            tbad += a != m;
                        }
                    }
            }
        }
    /* kelp: strand length 1..40 m, deep and shallow water, 64 columns each */
    const int32_t depths[3] = { 45, 12, 3 };
    for (int32_t len = 1; len <= 40; ++len)
        for (int di = 0; di < 3; ++di) {
            const int32_t h = pl.sea_level - depths[di];
            const uint32_t cell = BIOCELL(BIO_KIND_KELP, 2, 255, len, 0);
            for (int32_t x = 0; x < 64; ++x) {
                const int32_t z = 3;
                GmeSlot s;
                memset(&s, 0, sizeof s);
                s.cell = cell;
                GmeColumn col;
                gme_column_in_slot(x, z, h, seed, &s, &col);
                const int32_t L = gme_kelp_len(x, z, h, cell, seed, &pl);
                strands += L > 0;
                for (int32_t y = h + 1; y <= pl.sea_level; ++y) {
                    const uint8_t m = gme_voxel(x, y, z, &col, seed, &pl);
                    const int in = L > 0 && gme_kelp_inside_f(0.f, (float)y + 0.5f, 0.f, (float)(h + 1), (float)(h + L + 1));
                    ++kvox;
                    kbad += in != (m == MAT_KELP);
                }
            }
        }
    const int ok = !tbad && !kbad;
    printf("D2 flora: trunks + crowns (ht 2..%d, every trunk position) %llu voxel centres, %llu mismatches; "
           "kelp (len 1..40, 3 depths, %llu strands) %llu voxel centres, %llu mismatches\nD2 %s\n",
           GME_FLORA_TREE_MAX, (unsigned long long)tvox, (unsigned long long)tbad, (unsigned long long)strands,
           (unsigned long long)kvox, (unsigned long long)kbad, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    Options o;
    if (parse(argc, argv, &o)) return 2;
    if (o.cmp_a) return compare_raw(o.cmp_a, o.cmp_b);
    platform_init();

    /* ---- hardware ---- */
    SysInfo si;
    sys_query(&si);
    GpuInfo gi;
    memset(&gi, 0, sizeof gi);
    if (!o.cpu) gpu_query(&gi);

    printf("== GME engine ==\n");
    printf("CPU  : %s  %d threads, %d cores", si.cpu_name, si.logical_cpus, si.physical_cores);
    if (si.perf_cores) printf(" (%dP + %dE)", si.perf_cores, si.eff_cores);
    printf("\nRAM  : %.1f GiB total, %.1f GiB available\n", gib((double)si.total_ram), gib((double)si.avail_ram));
    if (gi.available)
        printf("GPU  : %s  sm_%d%d, %d SMs, %.1f GiB VRAM (%.1f GiB free)\n", gi.name, gi.cc_major, gi.cc_minor,
               gi.sm_count, gib((double)gi.total_mem), gib((double)gi.free_mem));
    else
        printf("GPU  : none used (%s)\n", o.cpu ? "--cpu" : gpu_last_error());

    const int workers = si.logical_cpus > 1 ? si.logical_cpus - 1 : 0;
    jobs_init(workers);
    printf("Jobs : %d threads (main + %d workers)\n", jobs_thread_count(), workers);

    /* ---- home planet from genesis ---- */
    HomePlanet hp;
    const char* ppath = o.planet ? o.planet : "genesis/home.planet";
    if (planet_load(ppath, &hp) == 0) {
        printf("Planet: %s - system %u planet #%d, %.2f Me, %.2f g, %.0f K surface, oceans %.0f%%\n", ppath,
               hp.system_seed, hp.planet_index, hp.mass_me, hp.gravity_g, hp.t_surf, 100 * hp.ocean_cover);
        printf("        atmosphere: %s\n        crust: %s\n", hp.atmosphere, hp.crust);
        if (!o.seed_set) o.seed = hp.system_seed;
    } else {
        if (o.planet) { fprintf(stderr, "cannot read planet file %s\n", o.planet); return 1; }
        planet_default(&hp);
        printf("Planet: default Earth-like (run gme_genesis to boot a real one)\n");
    }
    const uint64_t phash = planet_hash(&hp);
    if (hp.atm_known)
        printf("        mix H2S %.4f CH4 %.4f NH3 %.4f CO2 %.4f O2 %.4f N2 %.4f H2 %.4f, planet hash %016llx\n",
               hp.atm_mix[ATM_H2S], hp.atm_mix[ATM_CH4], hp.atm_mix[ATM_NH3], hp.atm_mix[ATM_CO2], hp.atm_mix[ATM_O2],
               hp.atm_mix[ATM_N2], hp.atm_mix[ATM_H2], (unsigned long long)phash);
    else
        printf("        atmosphere mix unknown, planet hash %016llx\n", (unsigned long long)phash);

    /* ---- spawn without life (also the centre of --biomap-test) ---- */
    Spawn sp;
    if (o.spawn_set) spawn_above(o.spawn_x, o.spawn_z, o.spawn_eye, o.seed, &hp.gen, &sp);
    else spawn_default(o.seed, &hp.gen, &sp);
    const Spawn sp_default = sp;

    /* ---- life: vegetation frozen from the biosphere simulation ---- */
    Biomap bm;
    memset(&bm, 0, sizeof bm);
    int have_life = 0;
    if (o.no_life) {
        printf("Life : off (--no-life)\n");
    } else if (o.biomap_test) {
        if (biomap_make_test(&bm, phash, o.seed, sp.x, sp.z)) { fprintf(stderr, "cannot build the test biomap\n"); return 1; }
        have_life = 1;
        print_biomap("built-in test pattern (--biomap-test, not saved)", &bm);
    } else {
        const char* bpath = o.biomap ? o.biomap : (file_exists(DEFAULT_BIOMAP) ? DEFAULT_BIOMAP : NULL);
        if (!bpath) {
            printf("Life : none (no %s)\n", DEFAULT_BIOMAP);
        } else {
            const int rc = biomap_load(bpath, &bm);
            char why[192] = "";
            if (rc == -1) snprintf(why, sizeof why, "cannot read it");
            else if (rc == -2) snprintf(why, sizeof why, "not a valid GMEBMAP1 file");
            else if (rc == -3) snprintf(why, sizeof why, "CRC mismatch (corrupt)");
            else if (rc) snprintf(why, sizeof why, "load error %d", rc);
            else if (bm.planet_hash != phash)
                snprintf(why, sizeof why, "made for planet hash %016llx, this planet is %016llx",
                         (unsigned long long)bm.planet_hash, (unsigned long long)phash);
            else if (bm.seed != o.seed)
                snprintf(why, sizeof why, "made for world seed %u, this world is seed %u", bm.seed, o.seed);
            if (why[0]) {
                fprintf(stderr, "biomap %s rejected: %s\n", bpath, why);
                biomap_free(&bm);
                if (o.biomap) return 1;     /* asked for explicitly: do not run without it */
                printf("Life : none (%s rejected)\n", bpath);
            } else {
                have_life = 1;
                print_biomap(bpath, &bm);
            }
        }
    }
    /* with life, start where it lives: the coast nearest the centre of region K (default 0) */
    const char* where = "";
    char where_buf[96];
    if (have_life && !o.spawn_set && bm.nreg > 0) {
        int r = o.region;
        if (r < 0 || r >= bm.nreg) {
            fprintf(stderr, "--region %d: the biomap has regions 0..%d, using 0\n", r, bm.nreg - 1);
            r = 0;
        }
        const int32_t cx = (int32_t)(bm.reg[r].x0 + (((int64_t)bm.reg[r].w << bm.shift) >> 1));
        const int32_t cz = (int32_t)(bm.reg[r].z0 + (((int64_t)bm.reg[r].h << bm.shift) >> 1));
        snprintf(where_buf, sizeof where_buf, " (coast near the centre of region %d)", r);
        where = where_buf;
        if (spawn_coast(cx, cz, o.seed, &hp.gen, &sp)) {
            spawn_above(cx, cz, 50, o.seed, &hp.gen, &sp);
            snprintf(where_buf, sizeof where_buf, " (centre of region %d, no coast within 5 km)", r);
        }
    }
    /* harness views: computed from the generator around the coast spawn */
    if (o.view && strcmp(o.view, "coast")) {
        const Spawn coast = sp;
        if (!strcmp(o.view, "forest")) {
            spawn_above(sp_default.x, sp_default.z, 20, o.seed, &hp.gen, &sp);
        } else if (!strcmp(o.view, "shelf")) {
            if (view_shelf(&coast, o.seed, &hp.gen, &sp)) fprintf(stderr, "--view shelf: no column 8-15 m deep within 4 km\n");
        } else {
            Spawn peak;
            view_peak(&coast, o.seed, &hp.gen, &peak);
            if (!strcmp(o.view, "peak")) sp = peak;
            else view_far(&peak, o.seed, &hp.gen, &sp);
        }
        snprintf(where_buf, sizeof where_buf, " (--view %s)", o.view);
        where = where_buf;
    } else if (o.view) {
        char tmp[128];
        snprintf(tmp, sizeof tmp, "%s (--view coast)", where);
        snprintf(where_buf, sizeof where_buf, "%s", tmp);
        where = where_buf;
    }
    if (o.look_set) { sp.yaw = o.look_yaw / 57.29578f; sp.pitch = o.look_pitch / 57.29578f; }
    if (o.eye_y_set) sp.y = o.eye_y;
    printf("Spawn: (%d, %d, %d), heading %.0f deg%s\n", sp.x, sp.y, sp.z, sp.yaw * 57.29578f, where);
    if (o.view) printf("View : %s at (%d, %d, %d), yaw %.2f deg, pitch %.2f deg\n", o.view, sp.x, sp.y, sp.z,
                       sp.yaw * 57.29578f, sp.pitch * 57.29578f);
    printf("\n");

    /* ---- self-tests (no world needed) ---- */
    if (o.selftest) {
        int fails = 0;
        fails += selftest_d1(o.seed, &hp.gen, phash, &bm, have_life, &sp_default, gi.available);
        fails += selftest_d2(o.seed, &hp.gen);
        printf("D3 physics (M2), D4 render (M4), D5 export (M5): not part of M1, skipped\n");
        printf("SELFTEST: %d failure%s\n", fails, fails == 1 ? "" : "s");
        biomap_free(&bm);
        jobs_shutdown();
        platform_shutdown();
        return fails;
    }

    /* ---- renderer first, so the pool is sized from the VRAM really left ---- */
    Renderer* rd = NULL;
    if (!o.bench && gi.available) {
        const RendererConfig rc = { (int)(o.width * (o.scale > 0 ? o.scale : 1.0)) + 1,
                                    (int)(o.height * (o.scale > 0 ? o.scale : 1.0)) + 1, o.quality };
        rd = renderer_create(&rc);
        if (rd) gpu_query(&gi);
    }

    /* ---- world ---- */
    WorldConfig wc;
    memset(&wc, 0, sizeof wc);
    wc.seed = o.seed;
    wc.planet = hp.gen;
    wc.min_cy = -8;
    wc.max_cy = 8;          /* 512 m of vertical space: y in [-256, 256) */
    wc.vram_fraction = o.vram;
    wc.ram_fraction = o.ram;
    wc.max_batch = o.batch;
    wc.force_cpu = o.cpu;
    wc.radius_override = o.radius;
    if (have_life) {
        biomap_view(&bm, &wc.bio);
        memcpy(wc.bio_palette, bm.palette, sizeof wc.bio_palette);
    }
    World* w = world_create(&wc, &si, &gi);
    if (!w) { renderer_destroy(rd); biomap_free(&bm); jobs_shutdown(); platform_shutdown(); return 1; }

    WorldStats s;
    world_get_stats(w, &s);
    printf("World: %s generation, %u chunk slots (%.2f GiB %s + %.2f GiB host mirror), %u records\n",
           s.use_gpu ? "GPU" : "CPU", s.slots_total, gib((double)s.slots_total * GME_CHUNK_VOXELS),
           s.use_gpu ? "VRAM" : "RAM", gib((double)s.host_reserved_bytes), s.recs_total);
    printf("       %u layers x 32 m tall, max view radius %d chunks (%d m), pinned staging %.0f MiB, index %.0f MiB\n",
           s.layers, s.radius_max, s.radius_max * GME_CHUNK_DIM, (double)s.pinned_bytes / 1048576.0,
           (double)s.host_meta_bytes / 1048576.0);
    if (s.tiles_total)
        printf("       column tiles: %u x %u B = %.0f MiB VRAM (smooth surface, counted in the pool budget)\n",
               s.tiles_total, (unsigned)sizeof(GmeColTile), (double)s.tile_bytes / 1048576.0);
    printf("\n");

    WorldPos cam;
    memset(&cam, 0, sizeof cam);
    cam.lx = (float)sp.x; cam.lz = (float)sp.z; cam.ly = (float)sp.y;
    worldpos_normalize(&cam);

    if (!o.bench) {
        if (!s.use_gpu) { fprintf(stderr, "the 3D view needs a CUDA GPU (use --bench --cpu for CPU-only)\n"); return 1; }
        const int rc = run_viewer(w, &o, cam, &hp, sp.yaw, sp.pitch, rd);
        world_destroy(w);
        renderer_destroy(rd);
        biomap_free(&bm);
        jobs_shutdown();
        platform_shutdown();
        return rc;
    }

    /* ---- phase 1: fill the space from zero to the largest radius that fits ---- */
    printf("-- initial fill --\n");
    const double t0 = time_now();
    double last_print = t0;
    int32_t last_r = -1;
    int stable = 0;
    uint64_t gen0 = 0;
    while (time_now() - t0 < 120.0) {
        if (world_update(w, &cam)) goto fatal;
        world_get_stats(w, &s);
        if (s.fully_loaded && s.radius == last_r) { if (++stable >= 3) break; }
        else stable = 0;
        last_r = s.radius;
        const double now = time_now();
        if (now - last_print >= 1.0) {
            print_line("fill", now - t0, &cam, &s, (double)(s.generated - gen0) / (now - last_print), 0, 0);
            gen0 = s.generated;
            last_print = now;
        }
    }
    const double fill_time = time_now() - t0;
    world_get_stats(w, &s);
    printf("fill done in %.2f s: %llu chunks generated (%.0f ch/s, %.2f Gvoxel/s), radius %d = %d m view distance\n\n",
           fill_time, (unsigned long long)s.generated, (double)s.generated / fill_time,
           (double)s.generated * GME_CHUNK_VOXELS / fill_time * 1e-9, s.radius, s.radius * GME_CHUNK_DIM);

    /* ---- phase 2: fly through the world at 60 Hz, streaming continuously ---- */
    if (o.seconds > 0) {
        printf("-- flight at %.0f m/s for %.0f s --\n", o.speed, o.seconds);
        const double dt = 1.0 / 60.0;
        const double t1 = time_now();
        double next = t1, sec_start = t1, tick_sum = 0, tick_max = 0, worst_tick = 0;
        int ticks = 0;
        uint64_t total_ticks = 0;
        gen0 = s.generated;
        double sim_t = 0;
        while (time_now() - t1 < o.seconds) {
            sim_t += dt;
            const double heading = 0.6 * sin(sim_t * 0.05);
            cam.lx += (float)(cos(heading) * o.speed * dt);
            cam.lz += (float)(sin(heading) * o.speed * dt);
            worldpos_normalize(&cam);

            const double a = time_now();
            if (world_update(w, &cam)) goto fatal;
            const double tick = (time_now() - a) * 1000.0;
            tick_sum += tick;
            if (tick > tick_max) tick_max = tick;
            if (tick > worst_tick) worst_tick = tick;
            ++ticks;
            ++total_ticks;

            const double now = time_now();
            if (now - sec_start >= 1.0) {
                world_get_stats(w, &s);
                print_line("fly ", now - t1, &cam, &s, (double)(s.generated - gen0) / (now - sec_start),
                           tick_sum / ticks, tick_max);
                gen0 = s.generated;
                sec_start = now;
                tick_sum = tick_max = 0;
                ticks = 0;
            }
            next += dt;
            const double wait = next - time_now();
            if (wait > 0.0015) sleep_ms((int)(wait * 1000.0));
            else if (wait < -0.25) next = time_now(); /* fell behind: don't spiral */
        }
        world_get_stats(w, &s);
        printf("flight: %llu ticks, worst tick %.2f ms, %llu chunks evicted, %llu radius shrinks\n\n",
               (unsigned long long)total_ticks, worst_tick, (unsigned long long)s.evicted,
               (unsigned long long)s.pressure_events);
    }

    /* ---- verification ---- */
    printf("-- validation --\n");
    ValidateReport rep;
    const double tv = time_now();
    const int bad = world_validate(w, 512, &rep);
    printf("CPU re-generation of %u chunks (%u uniform), %llu voxels: host mismatches %llu, GPU pool mismatches %llu, meta mismatches %llu\n",
           rep.chunks_checked, rep.uniform_checked, (unsigned long long)rep.voxels_checked,
           (unsigned long long)rep.host_mismatch, (unsigned long long)rep.device_mismatch,
           (unsigned long long)rep.meta_mismatch);
    printf("Voxel queries: %u surface columns probed via world_get_voxel, %u mismatches  (%.2f s)\n",
           rep.columns_checked, rep.column_mismatch, time_now() - tv);
    printf("Reference: %u CPU chunks around the spawn column, ref_hash %016llx\n", rep.ref_chunks,
           (unsigned long long)rep.ref_hash);
    printf("Flora: %u vegetated chunk columns and %llu flora voxels in the checked chunks%s\n", rep.flora_checked,
           (unsigned long long)rep.flora_voxels,
           !have_life ? " (no life loaded)"
                      : (rep.flora_checked ? "" : "  <- FAIL: a biomap is loaded but no vegetated column was checked"));
    printf("Density: %u tiles (%u columns)%s: tile mismatches %llu, h %u, coast %u, saturated %u; "
           "dev T-(h+1) mean %+.3f m rms %.2f m max %.2f m, clamp %.1f%%; dens_hash %016llx\n",
           s.use_gpu ? rep.tiles_checked : rep.tiles_cpu, rep.tiles_cpu * GME_TILE_COLS,
           s.use_gpu ? "" : " (CPU mode: CPU tiles only)", (unsigned long long)rep.tile_mismatch, rep.tile_h_bad,
           rep.tile_coast_bad, rep.tile_sat, rep.dev_mean, rep.dev_rms, rep.dev_max, 100.0 * rep.clamp_share,
           (unsigned long long)rep.dens_hash);

    world_get_stats(w, &s);
    printf("\nResident: %u chunks = %.1f km^2 x 512 m (%.2f Gvoxels addressable), %u stored + %u uniform\n",
           s.ready, (double)s.ready / s.layers * 1024.0 / 1e6,
           (double)s.ready * GME_CHUNK_VOXELS * 1e-9, s.ready - s.uniform_ready, s.uniform_ready);
    printf("Memory  : %.2f GiB chunk data in use of %.2f GiB pool (%.1f%%), uniform chunks saved %.2f GiB\n",
           gib((double)s.slots_used * GME_CHUNK_VOXELS), gib((double)s.slots_total * GME_CHUNK_VOXELS),
           100.0 * s.slots_used / s.slots_total, gib((double)s.uniform_ready * GME_CHUNK_VOXELS));
    if (s.tiles_total)
        printf("Tiles   : %u published of %u table cells, %llu built\n", s.tiles_ready, s.tiles_total,
               (unsigned long long)s.tiles_built);
    printf("%s\n", bad ? "RESULT: FAIL" : "RESULT: OK - world space is consistent across GPU, host mirror and CPU reference");

    world_destroy(w);
    biomap_free(&bm);
    jobs_shutdown();
    platform_shutdown();
    return bad ? 1 : 0;

fatal:
    fprintf(stderr, "fatal error, aborting\n");
    world_destroy(w);
    renderer_destroy(rd);
    biomap_free(&bm);
    jobs_shutdown();
    platform_shutdown();
    return 1;
}

/* main.c - GME: detect hardware, size the world to it, then either open the
 * interactive 3D view (default) or run the headless streaming benchmark.
 * Also the render harness (docs/render-upgrade-spec.md 7.1): --view, --perf,
 * --metrics, --raw / --compare, --settle and --selftest; M5: --pigment-table,
 * --export-spectrum (GMESPEC1 for the bio), --sheet and the probe's photons. */
#include "biomap.h"
#include "image.h"
#include "jobs.h"
#include "planetfile.h"
#include "platform.h"
#include "spectrum_file.h"
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
    const char* view;              /* --view coast|peak|forest|shelf|far|mountain */
    int      eye_y_set;            /* --eye-y Y: absolute eye height */
    int32_t  eye_y;
    int      settle, raw, metrics, perf, perf_ab, fly, selftest;
    int      selftest_mask;        /* bit k: run Dk (--selftest [d1,d2,...]; default all) */
    const char* cmp_a;             /* --compare A.raw B.raw */
    const char* cmp_b;
    /* light (M2) */
    int      time_set;             /* --time H: hour of the day (sun at the zenith at 12, set at 18) */
    double   time_h;
    int      ev_set;               /* --ev EV100: fixed exposure */
    float    ev;
    float    wb;                   /* --wb D: CAT16 degree (default 0.8) */
    double   sun_disc, pressure, h2o;   /* --sun-disc k, --pressure bar, --h2o x */
    int      probe;                /* --probe: print the light at the crosshair after the last frame */
    int      ms_hillaire;          /* --ms hillaire: Hillaire's multiple-scattering series instead of iterated orders */
    /* distance (M3) */
    int      far_off;              /* --far 0: no far field */
    float    far_ring;             /* --far-ring M: near / far handover radius (test knob) */
    float    lod_fine;             /* --lod-fine M: fine -> LOD distance */
    int      lumstat;              /* --lumstat N: temporal luminance statistics over N static frames */
    int      ring_test;            /* --ring-test: luminance of the near field vs the far field at the ring */
    /* frame pipeline (M4) */
    int      gi_off;               /* --gi 0: no indirect light rays (key G toggles) */
    int      accum;                /* --accum N: the reference image (4-bounce paths, no denoising), N frames after --settle */
    int      still;                /* --still: waves and kelp hold still (static comparisons) */
    int      warmup;               /* --warmup N: settled frames before --lumstat starts */
    int      disocc_test;          /* --disocc-test: camera moves; disoccluded pixels must carry no history > 2 frames */
    /* life colours (M5) */
    int      pigment_table;        /* --pigment-table: the leaf colour of every hue bin (spec 5.1), then exit */
    const char* export_path;       /* --export-spectrum FILE: the GMESPEC1 light file for the bio, then exit */
    const char* sheet;             /* --sheet FILE: 5 kinds x 15 hues of the test pattern, close-ups by the renderer */
} Options;

static void usage(void) {
    printf("usage: gme [view options] | gme --bench [bench options] | gme --selftest | gme --compare A.raw B.raw\n"
           "  view : [--width N] [--height N] [--scale 0.25..1] [--fps N (0 = uncapped)]\n"
           "         [--frames N] [--screenshot file.png] [--raw (also file.raw, BGRA)]\n"
           "         [--quality classic|low|medium|high|ref] (default medium; V toggles classic, K cycles)\n"
           "         [--debug normals|albedo|dev|ao|band N] [--settle] [--perf] [--perf-ab] [--fly] [--metrics]\n"
           "  light: [--time H] (hour of the day, sun at the zenith at 12; T / Shift+T step 1 h; default: the\n"
           "         planet file's fixed afternoon sun) [--ev EV100] (fixed exposure, default auto) [--wb 0..1]\n"
           "         (CAT16 degree, default 0.8) [--sun-disc k] [--pressure bar] (default 1) [--h2o x]\n"
           "         [--ms hillaire] [--probe] (light at the crosshair after the last frame; key L)\n"
           "  far  : [--far 0|1] (far-field clipmap to 32 km, default 1) [--lod-fine M] (fine -> 4 m LOD distance)\n"
           "         [--far-ring M] (near / far handover radius, test knob) [--lumstat N] (temporal luminance sigma\n"
           "         over N static frames of the resolved image, needs --settle; [--warmup K] skips K frames first)\n"
           "         [--ring-test] (near vs far field luminance at the ring)\n"
           "  M4   : [--gi 0|1] (indirect light rays, key G) [--accum N] (reference image: 4-bounce paths, no denoising,\n"
           "         N frames after --settle; keeps the preset's scene) [--still] (waves and kelp hold still)\n"
           "         [--debug gi|age|vis|refl] [--disocc-test] (camera moves: disoccluded pixels must have history <= 2)\n"
           "         --scale s: the smooth path renders at s and TAAU upsamples (classic: StretchDIBits as before)\n"
           "  bench: [--seconds N] [--speed m/s] [--cpu]\n"
           "  both : [--vram 0..1] [--ram 0..1] [--batch chunks] [--radius chunks] [--seed N]\n"
           "         [--planet file] (default genesis/home.planet if present)\n"
           "         [--spawn X Z [EYE]] (start column, eye EYE m above ground, default 50)\n"
           "         [--look YAW PITCH] (start view direction, degrees) [--eye-y Y] (absolute eye height)\n"
           "         [--view coast|peak|forest|shelf|far|mountain] (deterministic spawn + look from the generator)\n"
           "  life : [--biomap file] (default " DEFAULT_BIOMAP " if present) | [--no-life]\n"
           "         | [--biomap-test] (built-in pattern of every vegetation kind around the spawn)\n"
           "         [--region K] (start at the coast of biomap region K, default 0)\n"
           "  M5   : --pigment-table (leaf colour of the 15 hue bins: Soret + Q band; spec 5.1) | --export-spectrum FILE\n"
           "         (GMESPEC1 integer light file for the bio: 8 haze bins x 24 h x 9 bands, water, photons, pigments)\n"
           "         [--sheet FILE] (with --biomap-test: 5 kinds x 15 hues, one close-up each) [--debug palette]\n"
           "  test : --selftest (D1 density, D2 flora, D3 physics, D4 render (furnace + 1800-frame flight), D5 export,\n"
           "         D6 far field; exit code = number of failures)\n");
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
    o->wb = 0.8f;
    o->sun_disc = 1.0;
    o->pressure = 1.0;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--probe"))  { o->probe = 1; continue; }
        if (!strcmp(a, "--debug") && v && !strcmp(v, "band")) {
            if (i + 2 >= argc) { usage(); return -1; }
            const int b = atoi(argv[i + 2]);
            if (b < 0 || b >= SPEC_NB) { fprintf(stderr, "--debug band: 0..%d\n", SPEC_NB - 1); return -1; }
            o->debug = RD_BAND0 + b;
            i += 2;
            continue;
        }
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
        if (!strcmp(a, "--selftest")) {
            o->selftest = 1;
            o->selftest_mask = 0x7E;                 /* D1 .. D6 */
            if (v && v[0] == 'd') {                  /* a subset: --selftest d2,d5 */
                o->selftest_mask = 0;
                for (const char* q = v; *q; ++q)
                    if ((*q == 'd' || *q == 'D') && q[1] >= '1' && q[1] <= '6') o->selftest_mask |= 1 << (q[1] - '0');
                ++i;
            }
            continue;
        }
        if (!strcmp(a, "--ring-test")) { o->ring_test = 1; continue; }
        if (!strcmp(a, "--still"))    { o->still = 1; continue; }
        if (!strcmp(a, "--disocc-test")) { o->disocc_test = 1; o->settle = 1; continue; }
        if (!strcmp(a, "--pigment-table")) { o->pigment_table = 1; continue; }
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
        else if (!strcmp(a, "--time"))    { o->time_h = atof(v); o->time_set = 1; }
        else if (!strcmp(a, "--ev"))      { o->ev = (float)atof(v); o->ev_set = 1; }
        else if (!strcmp(a, "--wb"))      o->wb = (float)atof(v);
        else if (!strcmp(a, "--sun-disc")) o->sun_disc = atof(v);
        else if (!strcmp(a, "--pressure")) o->pressure = atof(v);
        else if (!strcmp(a, "--h2o"))     o->h2o = atof(v);
        else if (!strcmp(a, "--far"))     o->far_off = atoi(v) == 0;
        else if (!strcmp(a, "--far-ring")) o->far_ring = (float)atof(v);
        else if (!strcmp(a, "--lod-fine")) o->lod_fine = (float)atof(v);
        else if (!strcmp(a, "--lumstat")) o->lumstat = atoi(v);
        else if (!strcmp(a, "--warmup"))  o->warmup = atoi(v);
        else if (!strcmp(a, "--gi"))      o->gi_off = atoi(v) == 0;
        else if (!strcmp(a, "--export-spectrum")) o->export_path = v;
        else if (!strcmp(a, "--sheet"))   { o->sheet = v; o->biomap_test = 1; }
        else if (!strcmp(a, "--accum"))   { o->accum = atoi(v); o->settle = 1; o->still = 1; }
        else if (!strcmp(a, "--ms")) {
            if (!strcmp(v, "hillaire")) o->ms_hillaire = 1;
            else if (strcmp(v, "orders")) { fprintf(stderr, "--ms: orders or hillaire\n"); return -1; }
        }
        else if (!strcmp(a, "--quality")) {
            o->quality = parse_quality(v);
            if (o->quality < 0) { fprintf(stderr, "--quality: classic, low, medium, high or ref\n"); return -1; }
        } else if (!strcmp(a, "--debug")) {
            if      (!strcmp(v, "normals")) o->debug = RD_NORMALS;
            else if (!strcmp(v, "albedo"))  o->debug = RD_ALBEDO;
            else if (!strcmp(v, "dev"))     o->debug = RD_DEV;
            else if (!strcmp(v, "gi"))      o->debug = RD_GI;    /* M4: the indirect irradiance (without GI: the AO) */
            else if (!strcmp(v, "ao"))      o->debug = RD_AO;    /* the heightfield AO of the sky ambient */
            else if (!strcmp(v, "age"))     o->debug = RD_AGE;   /* M4: TAA history age */
            else if (!strcmp(v, "vis"))     o->debug = RD_VIS;   /* M4: filtered sun visibility */
            else if (!strcmp(v, "refl"))    o->debug = RD_REFL;  /* M4: traced reflections */
            else if (!strcmp(v, "palette")) o->debug = RD_PALETTE;   /* M5: the old HSV palette tint (debug only) */
            else if (!strcmp(v, "none"))    o->debug = RD_NONE;
            else { fprintf(stderr, "--debug: normals, albedo, dev, ao, gi, age, vis, refl, palette or band N\n"); return -1; }
        }
        else { usage(); return -1; }
        ++i;
    }
    if (o->no_life + o->biomap_test + (o->biomap != NULL) > 1) {
        fprintf(stderr, "--no-life, --biomap and --biomap-test exclude each other\n");
        return -1;
    }
    if (o->view && strcmp(o->view, "coast") && strcmp(o->view, "peak") && strcmp(o->view, "forest") &&
        strcmp(o->view, "shelf") && strcmp(o->view, "far") && strcmp(o->view, "mountain")) {
        fprintf(stderr, "--view: coast, peak, forest, shelf, far or mountain\n");
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

/* mountain (M3 shadow gate): with the sun 10 deg high at azimuth sun_az
 * (radians, world atan2(z, x)), the dry ground point on a 512 m lattice within
 * 120 km of the coast spawn c whose first occluder toward the sun (16 m steps of
 * gme_height, up to 4 km) is farthest away: it lies deep in a long mountain
 * shadow (24 coarse candidates are refined on a 32 m lattice). The camera
 * stands at y 250, 900 m to the side of that shadow (square to the sun's
 * azimuth), looking at it. */
#define MTN_R    120000
#define MTN_STEP 512
#define MTN_CAND 24                 /* coarse candidates refined on a 32 m lattice (+-768 m) */
typedef struct MtnCtx {
    int32_t cx, cz;
    float sx, sz, slope;
    uint32_t seed;
    const GmePlanet* pl;
    int32_t* best;              /* per row (coarse) / per candidate x row (fine): occluder distance, x, z, h */
    const int32_t* cand;        /* fine pass: candidate centres x, z */
} MtnCtx;
/* distance to the first terrain point above the line toward the 10 deg sun (0 = none within 4 km) */
static int32_t mtn_occ(const MtnCtx* m, int32_t x, int32_t z, int32_t h) {
    for (int32_t d = 16; d <= 4000; d += 16) {
        const int32_t ho = gme_height(x + (int32_t)lrintf(m->sx * (float)d), z + (int32_t)lrintf(m->sz * (float)d), m->seed, m->pl);
        if ((float)(ho - h) > m->slope * (float)d + 1.f) return d;
    }
    return 0;
}
static void mtn_try(const MtnCtx* m, int32_t x, int32_t z, int32_t* o) {
    const int32_t h = gme_height(x, z, m->seed, m->pl);
    if (h <= m->pl->sea_level + 2) return;
    const int32_t d = mtn_occ(m, x, z, h);
    if (d > o[0]) { o[0] = d; o[1] = x; o[2] = z; o[3] = h; }
}
static void mtn_row_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    const MtnCtx* m = (const MtnCtx*)p;
    for (uint32_t row = b; row < e; ++row) {
        int32_t* o = m->best + 4 * row;
        o[0] = 0;
        const int32_t dz = -MTN_R + (int32_t)row * MTN_STEP;
        for (int32_t dx = -MTN_R; dx <= MTN_R; dx += MTN_STEP)
            if ((int64_t)dx * dx + (int64_t)dz * dz <= (int64_t)MTN_R * MTN_R) mtn_try(m, m->cx + dx, m->cz + dz, o);
    }
}
static void mtn_fine_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {   /* item = candidate * 49 + row */
    (void)wk;
    const MtnCtx* m = (const MtnCtx*)p;
    for (uint32_t it = b; it < e; ++it) {
        int32_t* o = m->best + 4 * it;
        o[0] = 0;
        const int32_t k = (int32_t)(it / 49), dz = -768 + 32 * (int32_t)(it % 49);
        for (int32_t dx = -768; dx <= 768; dx += 32) mtn_try(m, m->cand[2 * k] + dx, m->cand[2 * k + 1] + dz, o);
    }
}
static void view_mountain(const Spawn* c, float sun_az, uint32_t seed, const GmePlanet* pl, Spawn* s) {
    const uint32_t rows = 2 * MTN_R / MTN_STEP + 1;
    MtnCtx m = { c->x, c->z, cosf(sun_az), sinf(sun_az), tanf(10.f / 57.29578f), seed, pl, NULL, NULL };
    m.best = (int32_t*)calloc(4 * (size_t)(rows > MTN_CAND * 49 ? rows : MTN_CAND * 49), sizeof(int32_t));
    int32_t cand[2 * MTN_CAND], nc = 0, coarse = 0;
    int32_t bx = c->x, bz = c->z, bd = 0, bh = 0;
    if (m.best) {
        jobs_parallel_for(rows, 1, mtn_row_fn, &m);
        for (; nc < MTN_CAND; ++nc) {                  /* the best rows' points, best first */
            uint32_t br = 0;
            for (uint32_t r = 1; r < rows; ++r) if (m.best[4 * r] > m.best[4 * br]) br = r;
            if (m.best[4 * br] <= 0) break;
            if (!nc) coarse = m.best[4 * br];
            cand[2 * nc] = m.best[4 * br + 1]; cand[2 * nc + 1] = m.best[4 * br + 2];
            m.best[4 * br] = -1;
        }
        m.cand = cand;
        jobs_parallel_for((uint32_t)nc * 49u, 1, mtn_fine_fn, &m);
        for (uint32_t i = 0; i < (uint32_t)nc * 49u; ++i)
            if (m.best[4 * i] > bd) { bd = m.best[4 * i]; bx = m.best[4 * i + 1]; bz = m.best[4 * i + 2]; bh = m.best[4 * i + 3]; }
        free(m.best);
    }
    printf("View : mountain search: %d candidates, longest 10 deg shadow on the 512 m lattice %d m, refined (32 m) %d m\n", nc, coarse, bd);
    const float sx = m.sx, sz = m.sz;
    const int32_t mx = bx + (int32_t)lrintf(sx * 0.3f * (float)bd), mz = bz + (int32_t)lrintf(sz * 0.3f * (float)bd);   /* in the shadow */
    spawn_above(mx + (int32_t)lrintf(-sz * 900.f), mz + (int32_t)lrintf(sx * 900.f), 250, seed, pl, s);   /* high, beside it */
    look_at(s, (float)mx, (float)bh, (float)mz);
    printf("View : mountain shadow: ground (%d, %d) h %d, first occluder toward a 10 deg sun %d m away; camera looks at (%d, %d)\n",
           bx, bz, bh, bd, mx, mz);
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
    double se = 0, sb = 0;
    int dmax = 0;
    for (size_t i = 0; i < n; ++i) {
        int diff = 0;
        for (int k = 0; k < 3; ++k) {
            const int vb = (int)((B[i] >> (8 * k)) & 255u);
            const int d = (int)((A[i] >> (8 * k)) & 255u) - vb;
            se += (double)d * d;
            sb += (double)vb * vb;
            diff |= d;
            if (abs(d) > dmax) dmax = abs(d);
        }
        same += diff == 0;
    }
    const double mse = se / (3.0 * (double)n);
    /* M4: relative MSE = sum of squared differences / sum of squares of B (B = the reference) */
    printf("compare %s vs %s: %dx%d, identical pixels %.4f%% (%zu of %zu), PSNR %s%.2f dB, max difference %d, relative MSE %.4f%%\n", a, b,
           wa, ha, 100.0 * (double)same / (double)n, same, n, mse > 0 ? "" : "inf ", mse > 0 ? 10.0 * log10(65025.0 / mse) : 0.0,
           dmax, sb > 0 ? 100.0 * se / sb : 0.0);
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

/* ---- time of day (M2) -----------------------------------------------------------
 * The bio's day (bio_sin24): the star rises at 06:00, stands at the zenith at
 * 12:00 and sets at 18:00. Its path is the vertical plane through build_sky's
 * fixed afternoon direction, which is the hour h0 = 18 - elevation / 15 deg, so
 * without --time nothing changes. An hour H turns the whole sky (star and
 * planets) about the horizontal axis normal to that plane by (H - h0) 15 deg. */
static double sky_base_hour(const RenderSky* s) {
    const double l = sqrt((double)s->sun_dir[0] * s->sun_dir[0] + (double)s->sun_dir[1] * s->sun_dir[1] +
                          (double)s->sun_dir[2] * s->sun_dir[2]);
    return 18.0 - asin(s->sun_dir[1] / l) * (180.0 / 3.14159265358979) / 15.0;
}
static void rot_vec(float v[3], const double k[3], double c, double sn) {
    const double x = v[0], y = v[1], z = v[2];
    const double kv = k[0] * x + k[1] * y + k[2] * z;
    const double cx = k[1] * z - k[2] * y, cy = k[2] * x - k[0] * z, cz = k[0] * y - k[1] * x;
    v[0] = (float)(x * c + cx * sn + k[0] * kv * (1 - c));
    v[1] = (float)(y * c + cy * sn + k[1] * kv * (1 - c));
    v[2] = (float)(z * c + cz * sn + k[2] * kv * (1 - c));
}
static void sky_at_time(const RenderSky* base, double hour, RenderSky* out) {
    *out = *base;
    double ax = base->sun_dir[0], az = base->sun_dir[2];
    double l = sqrt(ax * ax + az * az);
    if (l < 1e-6) { ax = 1; az = 0; l = 1; }
    const double k[3] = { az / l, 0.0, -ax / l };      /* up x A: turns the zenith toward the setting side */
    const double ang = (hour - sky_base_hour(base)) * 15.0 * 3.14159265358979 / 180.0;
    const double c = cos(ang), sn = sin(ang);
    rot_vec(out->sun_dir, k, c, sn);
    for (int b = 0; b < out->nbodies; ++b) rot_vec(out->body_dir[b], k, c, sn);
}
static double wrap24(double h) { return h - 24.0 * floor(h / 24.0); }
static void print_sun(const RenderSky* s, double hour) {
    const float* d = s->sun_dir;
    const double l = sqrt((double)d[0] * d[0] + (double)d[1] * d[1] + (double)d[2] * d[2]);
    printf("Sun  : time %.2f h, elevation %.1f deg, azimuth %.1f deg\n", wrap24(hour), asin(d[1] / l) * 57.29578,
           atan2(d[2], d[0]) * 57.29578);
    fflush(stdout);
}

/* Probe (key L, --probe): the light at the crosshair hit, per band, and the
 * ratio of its PAR to the bio's scalar i0 at the same sun elevation */
static void print_probe(Renderer* rd, const SpectralAtmo* a) {
    RenderProbe pr;
    if (!a || renderer_probe(rd, &pr)) { printf("probe: not available (classic path or no frame yet)\n"); fflush(stdout); return; }
    if (!pr.valid) { printf("probe: the crosshair sees the sky (sun elevation %.1f deg)\n", pr.sun_elev_deg); fflush(stdout); return; }
    static const char* kinds[10] = { "?", "terrain", "crown", "trunk", "kelp", "voxel", "ice", "terrain (4 m LOD)", "far field",
                                      "distant ground" };
    printf("probe: %s (material %d", kinds[pr.kind >= 0 && pr.kind < 10 ? pr.kind : 0], pr.material);
    if (pr.w1 > 0.01f) printf(" + %.0f%% material %d", 100.0 * pr.w1, pr.material1);
    printf(") at (%.1f, %.1f, %.1f), normal (%.2f %.2f %.2f), %s, sun visibility %.2f, AO %.2f, sun elevation %.1f deg\n", pr.pos[0],
           pr.pos[1], pr.pos[2], pr.normal[0], pr.normal[1], pr.normal[2], pr.depth > 0 ? "under water" : "in air", pr.sun_vis, pr.ao,
           pr.sun_elev_deg);
    if (pr.depth > 0) printf("       depth %.2f m below the water plane\n", pr.depth);
    if (pr.kind == RK_CROWN) printf("       the sun through the own crown (leaf-area medium): %.3f (in the sun row)\n", pr.tsun);
    /* M5: photons per band from each band's mean photon energy (the star's spectrum, spectral_build_atmo) */
    double phj[SPEC_NB];
    for (int b = 0; b < SPEC_NB; ++b) phj[b] = a->e_band[b] > 0 ? a->phot_band[b] / a->e_band[b] : 0;   /* umol per J */
    printf("       band (nm)          ");
    for (int b = 0; b < SPEC_NB; ++b) printf(" %7.0f", SPEC_L0 + SPEC_BW * (b + 0.5));
    printf("\n       sun  W/m^2/nm      ");
    for (int b = 0; b < SPEC_NB; ++b) printf(" %7.4f", pr.e_sun[b]);
    printf("\n       sky  W/m^2/nm      ");
    for (int b = 0; b < SPEC_NB; ++b) printf(" %7.4f", pr.e_sky[b]);
    printf("\n       sun  umol/m^2/s    ");
    for (int b = 0; b < SPEC_NB; ++b) printf(" %7.1f", pr.e_sun[b] * SPEC_BW * phj[b]);
    printf("\n       sky  umol/m^2/s    ");
    for (int b = 0; b < SPEC_NB; ++b) printf(" %7.1f", pr.e_sky[b] * SPEC_BW * phj[b]);
    double par = 0, par_ph = 0, hor_par = 0, ph_tot[SPEC_NB];
    printf("\n       total umol/m^2/s   ");
    for (int b = 0; b < SPEC_NB; ++b) {
        const double e = (pr.e_sun[b] + pr.e_sky[b]) * SPEC_BW;              /* W/m^2 in the band */
        const double ph = e * phj[b];
        ph_tot[b] = ph;
        printf(" %7.1f", ph);
        const double lo = SPEC_L0 + SPEC_BW * b, hi = lo + SPEC_BW;            /* the band's share inside 400-700 nm */
        const double f = fmax(0.0, fmin(hi, 700.0) - fmax(lo, 400.0)) / SPEC_BW;
        par += f * e; par_ph += f * ph; hor_par += f * pr.e_hor[b] * SPEC_BW;
    }
    double sun_e = 0, sky_e = 0, sun_p = 0, sky_p = 0;
    for (int b = 0; b < SPEC_NB; ++b) {
        sun_e += pr.e_sun[b] * SPEC_BW; sky_e += pr.e_sky[b] * SPEC_BW;
        sun_p += pr.e_sun[b] * SPEC_BW * phj[b]; sky_p += pr.e_sky[b] * SPEC_BW * phj[b];
    }
    const double i0 = a->s_top * 0.45 * (a->haze_q12 / 4096.0) * fmax(0.0, sin(pr.sun_elev_deg / 57.29578));
    printf("\n       380-740 nm: sun %.1f W/m^2 = %.0f umol/m^2/s, sky %.1f W/m^2 = %.0f umol/m^2/s (sky share %.0f%%); "
           "PAR %.1f W/m^2 = %.0f umol/m^2/s\n", sun_e, sun_p, sky_e, sky_p, sun_e + sky_e > 0 ? 100.0 * sky_e / (sun_e + sky_e) : 0.0, par,
           par_ph);
    printf("       bio i0 (S_top x 0.45 x haze %.3f x sin elevation) %.1f W/m^2; render/bio: this surface %.3f, "
           "unshadowed horizontal ground %.3f\n",
           a->haze_q12 / 4096.0, i0, i0 > 0 ? par / i0 : 0.0, i0 > 0 ? hor_par / i0 : 0.0);
    const int tissue = pr.material == MAT_GRASS || pr.material == MAT_MAT || pr.material == MAT_LEAF || pr.material == MAT_KELP ||
                       (pr.w1 > 0.01f && (pr.material1 == MAT_GRASS || pr.material1 == MAT_MAT || pr.material1 == MAT_LEAF ||
                                          pr.material1 == MAT_KELP));
    if (tissue) {                              /* the pigment of the hit's hue bin: photons it absorbs, and what each pathway can use */
        const int h = pr.pal >= 1 && pr.pal < SPEC_NHUE ? pr.pal : SPEC_HUE_DEFAULT;
        const double kT = 8.617333262e-5 * a->t_surf, erc[2] = { 1239.84198 / 680.0, 1239.84198 / 870.0 };
        double abs_ph = 0, use[2] = { 0, 0 };
        printf("       pigment: hue bin %d%s (Q band %.0f nm, PIG %.0f), cover %.2f; absorbed umol/m^2/s", h, pr.pal ? "" : " (palette 0: the default)",
               383 + 34.13 * h, SPEC_PIG_REF, pr.cover);
        for (int b = 0; b < SPEC_NB; ++b) {
            const double q = ph_tot[b] * a->pig_A[h][b];
            abs_ph += q;
            for (int p = 0; p < 2; ++p) use[p] += q * fmin(1.0, exp(-(erc[p] - a->ev_band[b]) / kT));
            printf(" %.1f", q);
        }
        printf("\n       absorbed 380-740 nm %.0f umol/m^2/s; usable uphill (kT %.1f meV): oxygenic (680 nm RC) %.0f, anoxygenic "
               "(870 nm RC) %.0f (NIR is not rendered; the export carries it)\n", abs_ph, 1000 * kT, use[0], use[1]);
    }
    fflush(stdout);
}

/* ---- M5: pigment table and the bio's light file (spec 5.1) ------------------------------ */
/* XYZ of a band reflectance under an equal-energy white, von Kries (XYZ scaling) to D65 -> 8-bit sRGB */
static void band_srgb8(const SpectralAtmo* a, const float* rho, int rgb[3]) {
    double E[3] = { 0, 0, 0 }, x[3] = { 0, 0, 0 };
    for (int b = 0; b < SPEC_NB; ++b)
        for (int c = 0; c < 3; ++c) { E[c] += a->cmf[b][c]; x[c] += rho[b] * a->cmf[b][c]; }
    static const double D65[3] = { 0.95047, 1.0, 1.08883 };
    double t[3];
    for (int c = 0; c < 3; ++c) t[c] = x[c] / E[c] * D65[c];
    const double M[3][3] = { { 3.2406, -1.5372, -0.4986 }, { -0.9689, 1.8758, 0.0415 }, { 0.0557, -0.2040, 1.0570 } };
    for (int r = 0; r < 3; ++r) {
        double v = M[r][0] * t[0] + M[r][1] * t[1] + M[r][2] * t[2];
        v = v < 0 ? 0 : (v > 1 ? 1 : v);
        v = v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1 / 2.4) - 0.055;
        rgb[r] = (int)floor(v * 255.0 + 0.5);
    }
}
static int print_pigment_table(const SpectralAtmo* a) {
    int ref[SPEC_NHUE][3], b8[SPEC_NHUE][3], sp[SPEC_NHUE][3];
    double de[SPEC_NHUE];
    const int worst = spectral_pigment_table(a, ref, b8, de, sp);
    printf("\nPigment table (spec 5.1): eps = 1.2 G(435, 30) + G(lambda_Q, 25) + 0.15 s(lambda), OD = PIG / 255 x 4 x eps, PIG %.0f;\n"
           "leaf rho = 0.04 + 0.44 (1 - A), tau = 0.42 (1 - A); colour under an equal-energy white, von Kries (XYZ) to D65, sRGB\n",
           SPEC_PIG_REF);
    printf(" bin lambda_Q | leaf 1 nm     spec 5.1      |d| | leaf 8 bands  dE2000 | grass (cover 255)  mat         | absorptance per band "
           "(380 ... 740 nm) | NIR\n");
    for (int h = 0; h < SPEC_NHUE; ++h) {
        float grass[SPEC_NB];
        for (int b = 0; b < SPEC_NB; ++b) grass[b] = a->mat_rho[MAT_DIRT][b] + (a->leaf_rho[h][b] - a->mat_rho[MAT_DIRT][b]) * 0.8f;
        int g8[3], m8[3];
        band_srgb8(a, grass, g8);
        band_srgb8(a, a->mat_tis[h], m8);
        int d = 0;
        for (int c = 0; c < 3; ++c) d = abs(ref[h][c] - sp[h][c]) > d ? abs(ref[h][c] - sp[h][c]) : d;
        if (h == 0) printf("   0 default  |");
        else printf("  %2d %4.0f nm  |", h, 383 + 34.13 * h);
        printf(" %3d %3d %3d   %3d %3d %3d  %3d | %3d %3d %3d  %5.2f  | %3d %3d %3d        %3d %3d %3d | ", ref[h][0], ref[h][1], ref[h][2],
               sp[h][0], sp[h][1], sp[h][2], d, b8[h][0], b8[h][1], b8[h][2], de[h], g8[0], g8[1], g8[2], m8[0], m8[1], m8[2]);
        for (int b = 0; b < SPEC_NB; ++b) printf("%.3f ", a->pig_A[h][b]);
        printf("| %.3f%s\n", a->pig_A_nir[h], h == 0 ? "  (palette 0 = bin 8)" : "");
    }
    printf("max |sRGB - spec 5.1| over bins 1..15: %d levels (gate 3): %s\n", worst, worst <= 3 ? "PASS" : "FAIL");
    fflush(stdout);
    return worst <= 3 ? 0 : 1;
}
static int export_spectrum(const HomePlanet* hp, const SpectralOpts* so, const char* path) {
    SpectrumFile* f = (SpectrumFile*)calloc(1, sizeof(SpectrumFile));
    SpectralExportInfo info;
    if (!f) return 1;
    printf("export: GMESPEC1 for planet hash %016llx: 8 haze bins, each its own atmosphere (CH4 fraction from the bio's haze "
           "factor, haze calibrated) ...\n", (unsigned long long)planet_hash(hp));
    fflush(stdout);
    if (spectral_export(hp, so, f, &info)) { fprintf(stderr, "export: the spectral build failed\n"); free(f); return 1; }
    for (int k = 0; k < SPECF_NHAZE; ++k)
        printf("   bin %d: haze_q12 %4d -> x_CH4 %.4f (%.2f km-amagat), tau_h550 %.4f, PAR global T %.4f (target %.4f)%s\n", k, info.haze_q12[k],
               info.x_ch4[k], info.ch4_col[k], info.tau_h550[k], info.t_par[k], info.target[k],
               info.unreachable[k] ? "  [Rayleigh + gas below the target: no haze]" : "");
    printf("   S_top %d W/m^2, T_surf %.2f K; built in %.1f s\n", f->s_top_wm2, f->t_surf_ck / 100.0, info.seconds);
    printf("   band (nm)         ");
    for (int b = 0; b < SPECF_NB; ++b) printf(" %3d-%-3d", f->band_edge_nm[b], f->band_edge_nm[b + 1]);
    for (int k = 0; k < SPECF_NHAZE; k += 7) {
        printf("\n   noon W/m^2 bin %d  ", k);
        for (int b = 0; b < SPECF_NB; ++b) printf(" %7.2f", f->surf_q16[k][12][b] / 65536.0 * f->s_top_wm2);
    }
    printf("\n   umol / J          ");
    for (int b = 0; b < SPECF_NB; ++b) printf(" %7.3f", f->phot_q16[b] / 65536.0);
    printf("\n   photon eV         ");
    for (int b = 0; b < SPECF_NB; ++b) printf(" %7.3f", f->ev_q16[b] / 65536.0);
    printf("\n   uphill OXY        ");
    for (int b = 0; b < SPECF_NB; ++b) printf(" %7.4f", f->uphill_q16[b][0] / 65536.0);
    printf("\n   uphill ANOX       ");
    for (int b = 0; b < SPECF_NB; ++b) printf(" %7.4f", f->uphill_q16[b][1] / 65536.0);
    printf("\n   water T(10 m)     ");
    for (int b = 0; b < SPECF_NB; ++b) printf(" %7.4f", f->water_T_q16[b][10] / 65536.0);
    printf("\n   kw 0-20 m (1/m)   ");
    for (int b = 0; b < SPECF_NB; ++b) printf(" %7.4f", f->kw_q16[b] / 65536.0);
    printf("\n   eps bin 8 (q12)   ");
    for (int b = 0; b < SPECF_NB; ++b) printf(" %7d", f->eps_q12[8][b]);
    printf("\n");
    const int rc = spectrum_save(path, f);
    if (rc) fprintf(stderr, "export: cannot write %s\n", path);
    else printf("saved %s (%u bytes, GMESPEC1, CRC32 header + payload)\n", path, (unsigned)(SPECF_HDR + SPECF_PAYLOAD_BYTES));
    fflush(stdout);
    free(f);
    return rc ? 1 : 0;
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

/* ---- M3 tests: temporal luminance statistics, the near / far ring ------------------------------ */
typedef struct LumStat {
    int      w, h, n;           /* frame size, frames accumulated */
    double*  s1;                /* per pixel: sum of the linear luminance */
    double*  s2;                /* ... of its square */
    float*   lin;               /* renderer_read_linear scratch */
    float*   first;             /* first frame: path length and code per pixel */
    uint64_t changed;           /* pixel-frames whose hit kind differs from the first frame */
} LumStat;

static int lumstat_add(LumStat* L, Renderer* rd, int maxw, int maxh) {
    const size_t cap = (size_t)maxw * maxh;
    if (!L->lin) {
        L->lin = (float*)malloc(sizeof(float) * 4 * cap);
        L->first = (float*)malloc(sizeof(float) * 2 * cap);
        L->s1 = (double*)calloc(cap, sizeof(double));
        L->s2 = (double*)calloc(cap, sizeof(double));
        if (!L->lin || !L->first || !L->s1 || !L->s2) return -1;
    }
    int w, h;
    if (renderer_read_linear(rd, L->lin, &w, &h) || (size_t)w * h > cap) return -1;
    if (!L->n) { L->w = w; L->h = h; }
    else if (w != L->w || h != L->h) return -1;
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        const double y = L->lin[4 * i];
        L->s1[i] += y;
        L->s2[i] += y * y;
        if (!L->n) { L->first[2 * i] = L->lin[4 * i + 1]; L->first[2 * i + 1] = L->lin[4 * i + 3]; }
        else L->changed += L->lin[4 * i + 3] != L->first[2 * i + 1];
    }
    L->n++;
    return 0;
}

static int cmp_float_asc(const void* a, const void* b) {
    const float x = *(const float*)a, y = *(const float*)b;
    return (x > y) - (x < y);
}

/* relative temporal sigma (std / mean of the linear luminance) of the pixels
 * whose path is longer than 1 km, by what they show */
static void lumstat_report(const LumStat* L) {
    if (L->n < 2) { printf("lumstat: fewer than 2 frames\n"); return; }
    static const char* cname[6] = { "all t > 1 km", "terrain (fine, LOD, far)", "water surface", "distant ground",
                                    "check: all hits, any t", "all pixels (M4 flicker)" };
    const size_t np = (size_t)L->w * L->h;
    float* v = (float*)malloc(sizeof(float) * np);
    if (!v) return;
    printf("lumstat: %d static frames, %dx%d, hit kinds changed in %llu pixel-frames\n", L->n, L->w, L->h,
           (unsigned long long)L->changed);
    for (int c = 0; c < 6; ++c) {
        size_t n = 0, over = 0;
        double sum = 0;
        for (size_t i = 0; i < np; ++i) {
            const float t = L->first[2 * i];
            const int code = (int)L->first[2 * i + 1], kind = code & 15, wm = code >> 4;
            if (c < 5 && !(t > (c == 4 ? 0.f : 1000.f))) continue;
            const int terr = !wm && (kind == RK_TERRAIN || kind == RK_LOD || kind == RK_FAR);
            if ((c == 1 && !terr) || (c == 2 && !wm) || (c == 3 && kind != RK_DIST)) continue;
            const double m = L->s1[i] / L->n, var = L->s2[i] / L->n - m * m;
            const float rs = m > 0 ? (float)(sqrt(var > 0 ? var : 0) / m) : 0.f;
            v[n++] = rs;
            sum += rs;
            over += rs > 0.01f;
        }
        if (!n) { printf("   %-26s 0 pixels\n", cname[c]); continue; }
        qsort(v, n, sizeof(float), cmp_float_asc);
        printf("   %-26s %8zu px: relative sigma mean %.4f%%, p95 %.4f%%, p99 %.4f%%, max %.4f%%; above 1%%: %zu (%.3f%%)\n", cname[c], n,
               100.0 * sum / (double)n, 100.0 * v[(size_t)((n - 1) * 0.95)], 100.0 * v[(size_t)((n - 1) * 0.99)], 100.0 * v[n - 1], over,
               100.0 * (double)over / (double)n);
    }
    free(v);
    fflush(stdout);
}

/* M4: <screenshot>_flicker.png: the relative temporal sigma per pixel (black 0, red 1 %, yellow >= 4 %) */
static void lumstat_image(const LumStat* L, const char* shot) {
    if (L->n < 2 || !shot) return;
    char path[1024];
    snprintf(path, sizeof path, "%s", shot);
    char* dot = strrchr(path, '.');
    if (dot && !strchr(dot, '/') && !strchr(dot, '\\')) *dot = 0;
    if (strlen(path) + 13 >= sizeof path) return;
    strcat(path, "_flicker.png");
    const size_t np = (size_t)L->w * L->h;
    uint32_t* img = (uint32_t*)malloc(sizeof(uint32_t) * np);
    if (!img) return;
    for (size_t i = 0; i < np; ++i) {
        const double m = L->s1[i] / L->n, var = L->s2[i] / L->n - m * m;
        const double rs = m > 0 ? sqrt(var > 0 ? var : 0) / m : 0.0;
        const double r = rs / 0.01, g = (rs - 0.01) / 0.03;
        const uint32_t R = (uint32_t)(255.0 * (r < 0 ? 0 : (r > 1 ? 1 : r))), G = (uint32_t)(255.0 * (g < 0 ? 0 : (g > 1 ? 1 : g)));
        img[i] = 0xFF000000u | (R << 16) | (G << 8);
    }
    if (!write_png(path, img, L->w, L->h)) printf("saved %s\n", path);
    free(img);
}

/* --ring-test: the same frame drawn with the far field from the loaded radius
 * (A, as always) and from 0.6 of it (B). Pixels whose straight hit lies just
 * inside the ring are the near field (4 m lattice, morphed to L1) in A and the
 * far field in B: their mean luminance ratio is the step at the ring. Also: the
 * unmorphed lattice vs the far field at 0.70-0.80 of the ring, and A's own
 * bands just inside / outside the ring. */
/* M4: one configuration of a test drawn from a fresh history until TAA, GI and
 * the visibility filter have converged (48 still frames), then read back */
static int draw_converged(Renderer* rd, RenderScene* sc, RenderCamera* rc, int rw, int rh, float* out, int* w, int* h) {
    const uint32_t* px;
    renderer_reset_history(rd);
    for (int k = 0; k < 48; ++k) {
        rc->frame += 1;
        if (renderer_draw(rd, sc, rc, rw, rh, &px)) return -1;
        sc->page_dirty = 0;
        sc->tile_dirty = 0;
    }
    return renderer_finish(rd, &px) || renderer_read_linear(rd, out, w, h);
}

static void ring_test(Renderer* rd, World* w, const RenderCamera* rc0, int rw, int rh) {
    RenderScene sc;
    world_get_scene(w, &sc);
    const float ring = sc.max_dist;
    float* A = (float*)malloc(sizeof(float) * 4 * (size_t)rw * rh);
    float* B = (float*)malloc(sizeof(float) * 4 * (size_t)rw * rh);
    RenderCamera rc = *rc0;
    int wa = 0, ha = 0, wb = 0, hb = 0;
    if (!A || !B) goto done;
    rc.far_ring = 0.f;
    if (draw_converged(rd, &sc, &rc, rw, rh, A, &wa, &ha)) goto done;
    rc.far_ring = 0.6f * ring;
    if (draw_converged(rd, &sc, &rc, rw, rh, B, &wb, &hb) || wa != wb || ha != hb) goto done;
    {
        double sa[3] = { 0, 0, 0 }, sb[3] = { 0, 0, 0 }, sd[3] = { 0, 0, 0 };
        size_t n[3] = { 0, 0, 0 };
        double in_s = 0, out_s = 0;
        size_t in_n = 0, out_n = 0;
        for (size_t i = 0; i < (size_t)wa * ha; ++i) {
            const int ca = (int)A[4 * i + 3], cb = (int)B[4 * i + 3];
            const float r = A[4 * i + 2], ya = A[4 * i], yb = B[4 * i];
            if ((ca >> 4) || r < 0.f) continue;                      /* straight hits only (no water) */
            const int ka = ca & 15, kb = cb & 15;
            if (r >= ring * 0.97f && r < ring && ka == RK_LOD) in_s += ya, ++in_n;
            if (r >= ring && r < ring * 1.03f && ka == RK_FAR) out_s += ya, ++out_n;
            if ((cb >> 4) || kb != RK_FAR || !(yb > 0.f)) continue;
            const int k = (r >= ring * 0.97f && r < ring && ka == RK_LOD) ? 0
                        : ((r >= ring * 0.70f && r < ring * 0.80f && (ka == RK_LOD || ka == RK_TERRAIN)) ? 1 : -1);
            if (k < 0) continue;
            sa[k] += ya; sb[k] += yb; sd[k] += fabs(ya - yb) / yb; ++n[k];
        }
        printf("ring-test: ring %.0f m (the loaded radius); near field = A, far field from %.0f m = B, linear luminance\n", ring, 0.6f * ring);
        printf("   at the ring (0.97-1.00 x ring, morphed lattice vs far field) %zu px: mean near %.4g, far %.4g, step %+.2f%% "
               "(per-pixel mean |difference| %.2f%%)\n", n[0], n[0] ? sa[0] / n[0] : 0.0, n[0] ? sb[0] / n[0] : 0.0,
               sb[0] > 0 ? 100.0 * (sa[0] / sb[0] - 1.0) : 0.0, n[0] ? 100.0 * sd[0] / n[0] : 0.0);
        printf("   unmorphed (0.70-0.80 x ring, fine / 4 m lattice vs far field) %zu px: mean near %.4g, far %.4g, difference %+.2f%%\n",
               n[1], n[1] ? sa[1] / n[1] : 0.0, n[1] ? sb[1] / n[1] : 0.0, sb[1] > 0 ? 100.0 * (sa[1] / sb[1] - 1.0) : 0.0);
        printf("   frame A: band inside the ring (0.97-1.00) %zu px mean %.4g, outside (1.00-1.03) %zu px mean %.4g: %+.2f%%\n", in_n,
               in_n ? in_s / in_n : 0.0, out_n, out_n ? out_s / out_n : 0.0,
               (in_n && out_n && in_s > 0) ? 100.0 * ((out_s / out_n) / (in_s / in_n) - 1.0) : 0.0);
        /* frame A's mean luminance in 3 % bands from 0.85 to 1.15 x ring, and each band's change against the one
         * before it: the change across the ring (1.00) next to the changes between bands of one representation */
        double bs[10] = { 0 };
        size_t bn[10] = { 0 };
        for (size_t i = 0; i < (size_t)wa * ha; ++i) {
            const int ca = (int)A[4 * i + 3];
            const float r = A[4 * i + 2];
            if ((ca >> 4) || r < 0.f) continue;
            const int k = (int)floorf((r / ring - 0.85f) / 0.03f);
            if (k >= 0 && k < 10) { bs[k] += A[4 * i]; ++bn[k]; }
        }
        printf("   frame A bands (x ring: mean, change vs the band before):");
        for (int k = 0; k < 10; ++k) {
            printf(" %.2f: %.4g", 0.85 + 0.03 * k, bn[k] ? bs[k] / bn[k] : 0.0);
            if (k && bn[k] && bn[k - 1]) printf(" (%+.2f%%)", 100.0 * ((bs[k] / bn[k]) / (bs[k - 1] / bn[k - 1]) - 1.0));
            printf(k == 4 ? " |ring|" : ",");
        }
        printf("\n");
        fflush(stdout);
    }
done:
    free(A);
    free(B);
}

/* --fps pacing (M4): waits until time t, sleeping while more than 2 ms remain and spinning the rest */
static void wait_until(double t) {
    for (;;) {
        const double left = t - time_now();
        if (left <= 0.0) return;
        if (left > 0.002) sleep_ms((int)((left - 0.0015) * 1000.0));
    }
}

static const char* preset_desc(int q) {
    switch (q) {
    case RQ_CLASSIC: return "classic voxel ray-marcher";
    case RQ_LOW:     return "smooth surface, spectral light; Low: 0.67 render scale + TAAU, hard shadows, sky SH x AO ambient, sky reflections";
    case RQ_MEDIUM:  return "smooth surface, spectral light; Medium: half-res GI 1 spp + radiance cache + SVGF, half-res traced reflections, "
                            "filtered disc shadows, TAA";
    case RQ_HIGH:    return "smooth surface, spectral light; High: half-res GI 2 spp + 4 a-trous passes, full-res GGX reflections, caustics, TAA";
    default:         return "smooth surface, spectral light; Ref: reference transport (full-res 4-bounce paths, no denoising), "
                            "frames averaged while the camera stands still";
    }
}

static int run_viewer(World* w, const Options* o, WorldPos cam, const HomePlanet* hp, float yaw, float pitch, Renderer* rd,
                      const SpectralAtmo* atmo) {
    RenderSky sky_base, sky;
    build_sky(hp, &sky_base);
    double hour = o->time_set ? o->time_h : sky_base_hour(&sky_base);
    if (o->time_set) sky_at_time(&sky_base, hour, &sky);
    else sky = sky_base;
    print_sun(&sky, hour);
    Window* win = window_create("GME", o->width, o->height);
    if (!win || !rd) {
        fprintf(stderr, "viewer: could not create window/renderer\n");
        window_destroy(win);
        return 0;
    }

    printf("Controls: hold right mouse = look | WASD = move | Space/C = up/down | Shift = x8 speed\n"
           "          wheel = base speed | F = shadows | V = classic / smooth | K = quality | P = screenshot | Esc = quit\n"
           "          T / Shift+T = time of day +1 h / -1 h | L = light probe at the crosshair | G = GI on / off\n");
    printf("Render: quality %s (%s)%s%s\n", QUALITY_NAME[o->quality], preset_desc(o->quality), o->accum ? "; --accum: reference transport" : "",
           o->gi_off ? "; --gi 0" : "");
    fflush(stdout);

    float speed = 40.f;
    int shadows = 1, shot = 0, quality = o->quality, last_smooth = o->quality != RQ_CLASSIC ? o->quality : RQ_MEDIUM, gi_off = o->gi_off;
    uint64_t frames = 0, drawn = 0;
    double last = time_now(), title_t = last, acc_world = 0, acc_frame = 0;
    float acc_gpu = 0;
    int acc_n = 0;
    const uint32_t* px = NULL;
    int rw = 0, rh = 0;
    int settled = !o->settle;
    double fly_t = 0;
    double next_present = 0, last_present = 0;   /* --fps: deadline of the next present; the last one */
    /* --perf host samples */
    const uint32_t cap = (o->frames > 0 ? (uint32_t)o->frames : 0) + (o->accum > 0 ? (uint32_t)o->accum : 0) + (uint32_t)o->lumstat +
                         (uint32_t)o->warmup + 1024u + (o->frames <= 0 && !o->accum ? (1u << 16) : 0u);
    double* s_world = o->perf ? (double*)malloc(sizeof(double) * cap) : NULL;
    double* s_present = o->perf ? (double*)malloc(sizeof(double) * cap) : NULL;
    double* s_frame = o->perf ? (double*)malloc(sizeof(double) * cap) : NULL;
    double* s_pp = o->perf ? (double*)malloc(sizeof(double) * cap) : NULL;      /* present-to-present intervals */
    float* s_gpu = o->perf ? (float*)malloc(sizeof(float) * cap * RENDER_PERF_STAGES) : NULL;
    /* --perf-ab: every frame also draws the classic path (alternating order), same camera and world state */
    float* s_gpu_c = o->perf_ab ? (float*)malloc(sizeof(float) * cap * RENDER_PERF_STAGES) : NULL;
    uint32_t* s_farn = o->perf ? (uint32_t*)malloc(sizeof(uint32_t) * cap) : NULL;   /* far-field samples rebuilt per frame */
    float last_c[RENDER_PERF_STAGES] = { 0 }, last_new[RENDER_PERF_STAGES] = { 0 };
    uint32_t ns = 0, npp = 0;
    RenderCamera last_rc;
    memset(&last_rc, 0, sizeof last_rc);
    if (o->perf_ab) renderer_set_sync(rd, 1);           /* both paths of a frame are timed before the next */
    /* --lumstat: per-pixel sums of the linear luminance over static frames (M3 gate: sigma < 1 % for t > 1 km;
     * M4: the resolved image, flicker) */
    LumStat ls;
    memset(&ls, 0, sizeof ls);
    /* --disocc-test: after --settle 60 still frames, 90 frames strafing 0.5 m and turning 0.3 deg per frame, 60 still */
    RenderDisocc dsum;
    memset(&dsum, 0, sizeof dsum);
    uint32_t dframes = 0;
    uint64_t dviol = 0;
    const uint64_t want_lum = o->lumstat ? (uint64_t)(o->lumstat + o->warmup) : 0;
    uint64_t want_frames = (uint64_t)(o->frames > 0 ? o->frames : 0);
    if (want_lum > want_frames) want_frames = want_lum;
    if ((uint64_t)o->accum > want_frames) want_frames = (uint64_t)o->accum;
    if (o->disocc_test && want_frames < 210) want_frames = 210;
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
            printf("quality %s (%s)\n", QUALITY_NAME[quality], preset_desc(quality)); fflush(stdout);
        }
        if (window_key_pressed(win, 'G')) {
            gi_off = !gi_off;
            printf("GI %s\n", gi_off ? "off (sky SH x AO)" : "on"); fflush(stdout);
        }
        if (window_key_pressed(win, 'T')) {          /* time of day */
            hour = wrap24(hour + (window_key_down(win, KEY_SHIFT) ? -1.0 : 1.0));
            sky_at_time(&sky_base, hour, &sky);
            print_sun(&sky, hour);
        }
        if (window_key_pressed(win, 'L')) print_probe(rd, atmo);
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
        if (o->disocc_test && settled && frames >= 60 && frames < 150) {   /* strafe right, turn left: parallax everywhere */
            cam.lx += -sinf(yaw) * 0.5f;
            cam.lz += cosf(yaw) * 0.5f;
            yaw -= 0.3f / 57.29578f;
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
        int reset = 0;
        if (!settled) {
            WorldStats ws;
            world_get_stats(w, &ws);
            if (ws.fully_loaded) {
                settled = 1;
                renderer_perf_reset(rd);
                reset = 1;                                /* M4: every history (and --accum) starts with the loaded world */
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
            /* classic renders at --scale and the window stretches it; the smooth path renders at the window size
             * (TAAU from its internal --scale, M4) */
            const double rs = quality == RQ_CLASSIC ? o->scale : 1.0;
            rw = (int)(cw * rs); rh = (int)(ch * rs);
            if (rw < 64) rw = 64;
            if (rh < 36) rh = 36;
            RenderScene sc;
            world_get_scene(w, &sc);
            const RenderCamera rc = {
                .cx = cam.cx, .cy = cam.cy, .cz = cam.cz, .lx = cam.lx, .ly = cam.ly, .lz = cam.lz,
                .yaw = yaw, .pitch = pitch, .fov_y = 1.15f, .shadows = shadows, .sky = &sky,
                .quality = quality, .debug = o->debug, .frame = (uint32_t)drawn, .sun_dir = { 0.f, 0.f, 0.f },
                .ev = o->ev_set ? o->ev : NAN, .dt = drawn ? (float)dt : 0.f, .wb = o->wb,
                .far_off = o->far_off, .far_ring = o->far_ring, .lod_fine = o->lod_fine,
                .scale = (float)o->scale, .gi_off = gi_off, .ref = o->accum > 0, .anim = o->still ? 0.f : (float)((double)drawn / 60.0),
                .reset = reset,
            };
            last_rc = rc;
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
            world_set_render_fence(w, renderer_fence(rd));   /* generation waits for the frame in flight (slots, tiles) */
            if (!drawn && quality != RQ_CLASSIC && renderer_far_rebuilt(rd)) {   /* M3: the far field's first (full) build */
                renderer_finish(rd, &px);
                printf("far field: first build %u samples in %.2f ms (GPU, far stage of frame 0)\n", renderer_far_rebuilt(rd),
                       renderer_perf_last(rd)[RS_FAR]);
                fflush(stdout);
            }
            if (o->disocc_test && settled && quality != RQ_CLASSIC && frames >= 60 && frames < 150) {
                RenderDisocc d;
                if (!renderer_disocc(rd, &d)) {
                    dsum.pixels += d.pixels; dsum.disoccluded += d.disoccluded; dsum.taa_old += d.taa_old; dsum.vis_old += d.vis_old;
                    dsum.gi_pixels += d.gi_pixels; dsum.gi_disoccluded += d.gi_disoccluded; dsum.gi_old += d.gi_old;
                    if (d.taa_age_max > dsum.taa_age_max) dsum.taa_age_max = d.taa_age_max;
                    if (d.gi_age_max > dsum.gi_age_max) dsum.gi_age_max = d.gi_age_max;
                    dviol += d.taa_old + d.vis_old + d.gi_old;
                    ++dframes;
                }
                if (frames == 120 && o->screenshot) {     /* <screenshot>_moving.png: a frame in the middle of the move */
                    char path[1024];
                    snprintf(path, sizeof path, "%s", o->screenshot);
                    char* dot = strrchr(path, '.');
                    if (dot && !strchr(dot, '/') && !strchr(dot, '\\')) *dot = 0;
                    if (strlen(path) + 12 < sizeof path) {
                        strcat(path, "_moving.png");
                        const uint32_t* mp;
                        if (!renderer_finish(rd, &mp) && !write_png(path, mp, rw, rh)) printf("saved %s\n", path);
                    }
                }
            }
            /* --fps: present at fixed intervals (spec 7.5: present-to-present at 60 fps) */
            if (o->fps > 0) {
                const double period = 1.0 / o->fps;
                if (next_present <= 0.0) next_present = time_now();
                wait_until(next_present);
                next_present += period;
                const double tn = time_now();
                if (tn > next_present) next_present = tn;   /* behind by more than a frame: present at once next time */
            }
            const double tp = time_now();
            window_present(win, px, rw, rh);
            t_present = time_now() - tp;
            if (settled && s_pp && last_present > 0.0 && npp < cap) s_pp[npp++] = (tp - last_present) * 1000.0;
            last_present = tp;
            acc_gpu += renderer_gpu_ms(rd);
            ++drawn;
            if (window_key_pressed(win, 'P')) {
                char name[64];
                snprintf(name, sizeof name, "screenshot_%03d.png", shot++);
                const uint32_t* sp_px;
                if (!renderer_finish(rd, &sp_px) && !write_png(name, sp_px, rw, rh)) printf("saved %s\n", name);
            }
        } else if (o->fps > 0) {
            sleep_ms(1000 / o->fps);                      /* minimised: nothing to present */
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
                if (s_farn) s_farn[ns] = renderer_far_rebuilt(rd);
                ++ns;
            }
            if (o->lumstat && quality != RQ_CLASSIC && frames > (uint64_t)o->warmup && ls.n < o->lumstat && px) {
                const uint32_t* fp;
                if (renderer_finish(rd, &fp) || lumstat_add(&ls, rd, rw, rh)) fprintf(stderr, "lumstat: linear readback failed\n");
            }
        }

        if (now - title_t >= 0.25 && acc_n) {
            WorldStats s;
            world_get_stats(w, &s);
            char t[320];
            snprintf(t, sizeof t,
                     "GME | %.0f fps | gpu render %.1f ms | world %.2f ms | %dx%d | pos %d %d %d | view %d m | %u chunks%s | speed %.0f m/s | shadows %s | %s%s",
                     acc_n / (now - title_t), acc_gpu / acc_n, acc_world * 1000.0 / acc_n, rw, rh,
                     cam.cx * GME_CHUNK_DIM + (int)cam.lx, cam.cy * GME_CHUNK_DIM + (int)cam.ly,
                     cam.cz * GME_CHUNK_DIM + (int)cam.lz, s.radius * GME_CHUNK_DIM, s.ready,
                     s.fully_loaded ? "" : " (streaming)", speed, shadows ? "on" : "off", QUALITY_NAME[quality],
                     gi_off && quality > RQ_LOW ? " (GI off)" : "");
            window_set_title(win, t);
            if (o->frames || o->accum) { printf("%s\n", t); fflush(stdout); }
            title_t = now;
            acc_world = acc_frame = 0;
            acc_gpu = 0;
            acc_n = 0;
        }
        if (want_frames && frames >= want_frames) break;
    }
    if (px) renderer_finish(rd, &px);                   /* the last frame queued (the readback is asynchronous) */
    if (o->screenshot && px) {
        if (!write_png(o->screenshot, px, rw, rh)) printf("saved %s\n", o->screenshot);
        if (o->raw) write_raw(o->screenshot, px, rw, rh);
    }
    if (quality != RQ_CLASSIC && atmo) {
        printf("exposure: EV100 %.2f (%s)\n", renderer_ev100(rd), o->ev_set ? "fixed by --ev" : "auto");
        printf("nan: %llu NaN pixels (shaded colour, GI, TAA) in %llu frames drawn\n", (unsigned long long)renderer_nan_count(rd),
               (unsigned long long)drawn);
        uint64_t cl = 0, chit = 0;
        renderer_cache_stats(rd, &cl, &chit);
        if (cl) printf("radiance cache: %llu lookups at secondary hits, %.1f%% found a cell (the rest: 0.5 x the sky SH)\n",
                       (unsigned long long)cl, 100.0 * (double)chit / (double)cl);
        if (o->accum) printf("accum: the reference averaged %u frames (4-bounce paths, no denoising)\n", renderer_accum_frames(rd));
        fflush(stdout);
    }
    if (o->disocc_test) {
        printf("disocc-test: %u moving frames (0.5 m strafe + 0.3 deg turn per frame): %llu pixels, %llu disoccluded (%.2f%%); "
               "of these with history older than 2 frames: TAA %llu, sun visibility %llu (largest TAA age %u)\n",
               dframes, (unsigned long long)dsum.pixels, (unsigned long long)dsum.disoccluded,
               dsum.pixels ? 100.0 * (double)dsum.disoccluded / (double)dsum.pixels : 0.0, (unsigned long long)dsum.taa_old,
               (unsigned long long)dsum.vis_old, dsum.taa_age_max);
        printf("disocc-test: GI (half res) %llu samples, %llu disoccluded, history older than 2 frames %llu (largest %u); %s\n",
               (unsigned long long)dsum.gi_pixels, (unsigned long long)dsum.gi_disoccluded, (unsigned long long)dsum.gi_old, dsum.gi_age_max,
               dviol == 0 && dsum.disoccluded > 0 ? "PASS" : "FAIL");
        fflush(stdout);
    }
    if (o->lumstat) { lumstat_report(&ls); lumstat_image(&ls, o->screenshot); }
    free(ls.s1); free(ls.s2); free(ls.lin); free(ls.first);
    if (o->probe && px) print_probe(rd, atmo);
    if (o->perf && ns) {
        RenderPerf rp;
        renderer_perf(rd, &rp);
        double p50, p95, mxv;
        double* tmp = (double*)malloc(sizeof(double) * (ns > npp ? ns : npp));
        printf("perf: %u frames, quality %s, %s%s\n", ns, QUALITY_NAME[quality], o->view ? o->view : "spawn",
               o->fly ? " flight (bench path)" : "");
        for (int k = 0; k < RENDER_PERF_STAGES && tmp; ++k) {
            for (uint32_t i = 0; i < ns; ++i) tmp[i] = s_gpu[(size_t)i * RENDER_PERF_STAGES + k];
            pct(tmp, ns, &p50, &p95, &mxv);
            printf("perf: %-13s p50 %7.3f  p95 %7.3f  max %7.3f ms\n", rp.name[k], p50, p95, mxv);
        }
        if (tmp) {                                      /* frame kernels: stages far .. taa */
            for (uint32_t i = 0; i < ns; ++i) {
                const float* g = s_gpu + (size_t)i * RENDER_PERF_STAGES;
                double s = 0;
                for (int k = RS_FAR; k <= RS_TAA; ++k) s += g[k];
                tmp[i] = s;
            }
            pct(tmp, ns, &p50, &p95, &mxv);
            printf("perf: %-13s p50 %7.3f  p95 %7.3f  max %7.3f ms\n", "frame kernels", p50, p95, mxv);
        }
        if (tmp && s_farn) {                            /* M3: the far-field update of the frames that moved a window */
            uint32_t nm = 0, nfull = 0;
            double nsum = 0;
            for (uint32_t i = 0; i < ns; ++i) {
                if (!s_farn[i]) continue;
                if (s_farn[i] >= (uint32_t)(GME_FAR_N * GME_FAR_N)) { ++nfull; continue; }   /* a full level rebuild */
                tmp[nm++] = s_gpu[(size_t)i * RENDER_PERF_STAGES + RS_FAR];
                nsum += s_farn[i];
            }
            pct(tmp, nm, &p50, &p95, &mxv);
            printf("perf: far update: %u of %u frames moved a window (%.0f samples each), p50 %.3f  p95 %.3f  max %.3f ms; "
                   "%u full rebuilds\n", nm, ns, nm ? nsum / nm : 0.0, p50, p95, mxv, nfull);
        }
        if (s_gpu_c && tmp) {                           /* the same frames, classic path */
            double c95[2] = { 0, 0 }, n95[2] = { 0, 0 };
            for (int k = 0; k < RENDER_PERF_STAGES; ++k) {
                if (k != RS_UPLOAD && k != RS_PRIMARY && k != RS_READBACK && k != RS_TOTAL) continue;   /* classic: k_render only */
                for (uint32_t i = 0; i < ns; ++i) tmp[i] = s_gpu_c[(size_t)i * RENDER_PERF_STAGES + k];
                pct(tmp, ns, &p50, &p95, &mxv);
                printf("perf: classic %-7s p50 %7.3f  p95 %7.3f  max %7.3f ms\n", k == RS_PRIMARY ? "k_render" : rp.name[k], p50, p95, mxv);
            }
            for (int pass = 0; pass < 2; ++pass) {      /* frame kernels (far .. taa), gpu_total */
                const float* src_ = pass ? s_gpu_c : s_gpu;
                for (uint32_t i = 0; i < ns; ++i) {
                    const float* g = src_ + (size_t)i * RENDER_PERF_STAGES;
                    double s = 0;
                    for (int k = RS_FAR; k <= RS_TAA; ++k) s += g[k];
                    tmp[i] = s;
                }
                pct(tmp, ns, &p50, &p95, &mxv);
                (pass ? c95 : n95)[0] = p95;
                for (uint32_t i = 0; i < ns; ++i) tmp[i] = src_[(size_t)i * RENDER_PERF_STAGES + RS_TOTAL];
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
            if (npp) {
                memcpy(tmp, s_pp, sizeof(double) * npp); pct(tmp, npp, &p50, &p95, &mxv);
                printf("perf: %-13s p50 %7.3f  p95 %7.3f  max %7.3f ms (%u intervals%s)\n", "present-to-present", p50, p95, mxv, npp,
                       o->fps > 0 ? "" : ", uncapped");
            }
        }
        free(tmp);
        char path[1024];
        perf_csv_path(o, quality, path, sizeof path);
        FILE* f = fopen(path, "w");
        if (f) {
            fprintf(f, "frame");
            for (int k = 0; k < RENDER_PERF_STAGES; ++k) fprintf(f, ",%s_ms", rp.name[k]);
            if (s_gpu_c) for (int k = 0; k < RENDER_PERF_STAGES; ++k) fprintf(f, ",classic_%s_ms", rp.name[k]);
            fprintf(f, ",world_update_ms,present_ms,frame_ms,far_samples,present_interval_ms\n");
            for (uint32_t i = 0; i < ns; ++i) {
                const float* g = s_gpu + (size_t)i * RENDER_PERF_STAGES;
                fprintf(f, "%u", i);
                for (int k = 0; k < RENDER_PERF_STAGES; ++k) fprintf(f, ",%.4f", g[k]);
                if (s_gpu_c) for (int k = 0; k < RENDER_PERF_STAGES; ++k) fprintf(f, ",%.4f", s_gpu_c[(size_t)i * RENDER_PERF_STAGES + k]);
                fprintf(f, ",%.4f,%.4f,%.4f,%u,%.4f\n", s_world[i], s_present[i], s_frame[i], s_farn ? s_farn[i] : 0u, i < npp ? s_pp[i] : 0.0);
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
            const unsigned long long shb = m.shadow_near + m.shadow_mid + m.shadow_far;
            printf("metrics M3: LOD %llu (%.1f%%), far field %llu (%.1f%%), distant ground %llu | below the geometric horizon %llu px, "
                   "of them sky %llu | shadow rays: lit %llu, blocked %llu: < 512 m %llu, 512 m - 2 km %llu, >= 2 km %llu "
                   "(%.1f%% of blocked rays beyond 512 m)\n",
                   (unsigned long long)m.lod, 100.0 * (double)m.lod / np, (unsigned long long)m.far, 100.0 * (double)m.far / np,
                   (unsigned long long)m.distant, (unsigned long long)m.below_horizon, (unsigned long long)m.sky_below_horizon,
                   (unsigned long long)m.sun_lit, shb, (unsigned long long)m.shadow_near, (unsigned long long)m.shadow_mid,
                   (unsigned long long)m.shadow_far, shb ? 100.0 * (double)(m.shadow_mid + m.shadow_far) / (double)shb : 0.0);
            if (o->screenshot && m.image) {      /* <screenshot>_metrics.png */
                char path[1024];
                snprintf(path, sizeof path, "%s", o->screenshot);
                char* dot = strrchr(path, '.');
                if (dot && !strchr(dot, '/') && !strchr(dot, '\\')) *dot = 0;
                if (strlen(path) + 13 < sizeof path) {
                    const size_t base = strlen(path);
                    strcat(path, "_metrics.png");
                    if (!write_png(path, m.image, m.image_w, m.image_h)) printf("saved %s\n", path);
                    if (m.shadow_image) {              /* <screenshot>_shadow.png: shadows by occluder distance */
                        path[base] = 0;
                        strcat(path, "_shadow.png");
                        if (!write_png(path, m.shadow_image, m.image_w, m.image_h)) printf("saved %s\n", path);
                    }
                }
            }
            fflush(stdout);
        }
    }
    if (o->ring_test && px && quality != RQ_CLASSIC) ring_test(rd, w, &last_rc, rw, rh);
    free(s_world); free(s_present); free(s_frame); free(s_pp); free(s_gpu); free(s_gpu_c); free(s_farn);
    window_destroy(win);
    return 0;
}

/* ---- M5 --sheet: the test pattern's 5 kinds x 15 hues, one close-up each ----------------
 * For every (kind, hue bin) the nearest 32 m block of the --biomap-test pattern with that kind
 * and palette index on terrain where the kind grows (mats on the beach or in shallow water, kelp
 * 5-16 m deep, cover on soil, shrubs / trees with at least 4 plants in the block), preferring
 * the densest cover; the camera looks at the block from 16 m (kelp: from 1.5 m under the
 * water). Rows: mat, kelp, cover, shrub, tree; columns: hue bins 1..15. */
typedef struct SheetPick { int found, score; int32_t x, z, h; double d2; } SheetPick;

/* is (x, y, z) inside (or within ~3 m of) the crown of a shrub / tree of the 3 x 3 slots around it? */
static int sheet_in_crown(double x, double y, double z, uint32_t seed, const GmePlanet* pl, const GmeBioView* bv) {
    const int32_t sx = (int32_t)floor(x) >> GME_FLORA_SLOT_SHIFT, sz = (int32_t)floor(z) >> GME_FLORA_SLOT_SHIFT;
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i) {
            GmeSlot s;
            gme_slot(sx + i, sz + j, seed, pl, bv, &s);
            if (s.trad <= 0) continue;
            if (gme_crown_inside_f((float)(x - (s.tx + 0.5)), (float)(y - (s.tcy + 1.5)), (float)(z - (s.tz + 0.5)), s.trad + 3, s.tvr + 3)) return 1;
        }
    return 0;
}

/* 0: unsuitable; else a score (higher is better): mats on the dry beach beat mats under water */
static int sheet_suits(int kind, int32_t xc, int32_t zc, int32_t hc, uint32_t seed, const GmePlanet* pl, const GmeBioView* bv) {
    const int32_t sea = pl->sea_level;
    if (kind == BIO_KIND_MAT) return hc >= sea && hc <= sea + 2 ? 2 : (hc >= sea - 6 && hc < sea ? 1 : 0);
    if (kind == BIO_KIND_KELP) {
        if (hc > sea - 5 || hc < sea - 16) return 0;
        int strands = 0;
        for (int k = 0; k < 16; ++k) {
            const int32_t x = xc - 8 + (k & 3) * 4, z = zc - 8 + (k >> 2) * 4;
            GmeSlot s;
            gme_slot(x >> GME_FLORA_SLOT_SHIFT, z >> GME_FLORA_SLOT_SHIFT, seed, pl, bv, &s);
            strands += gme_kelp_len(x, z, gme_height(x, z, seed, pl), s.cell, seed, pl) > 2;
        }
        return strands >= 4;
    }
    if (hc <= sea + 3 || gme_ground_mat(hc, pl) != pl->surface_mat) return 0;
    const int32_t dx = gme_height(xc + 12, zc, seed, pl) - gme_height(xc - 12, zc, seed, pl);
    const int32_t dz = gme_height(xc, zc + 12, seed, pl) - gme_height(xc, zc - 12, seed, pl);
    if (dx * dx + dz * dz > 100) return 0;                  /* gentle ground: the plants stand in view */
    if (kind == BIO_KIND_COVER) return 1;
    int plants = 0;
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i) {
            GmeSlot s;
            gme_slot(((xc - 16) >> GME_FLORA_SLOT_SHIFT) + i, ((zc - 16) >> GME_FLORA_SLOT_SHIFT) + j, seed, pl, bv, &s);
            plants += s.trad > 0;
        }
    return plants >= 4;
}

static int run_sheet(World* w, const Options* o, const HomePlanet* hp, Renderer* rd, const Biomap* bm, uint32_t seed) {
    const GmePlanet* pl = &hp->gen;
    enum { TW = 192, TH = 144, NK = 5, NH = 15 };
    const int SW = NH * TW, SHT = NK * TH;
    if (!bm->cells || bm->nreg < 1) { fprintf(stderr, "--sheet: needs the test pattern\n"); return 1; }
    GmeBioView bv;
    biomap_view(bm, &bv);
    SheetPick pick[NK][NH];
    memset(pick, 0, sizeof pick);
    const int32_t nb = bm->reg[0].w / 2, x0 = bm->reg[0].x0, z0 = bm->reg[0].z0;
    const double t0 = time_now();
    for (int32_t bj = 0; bj < nb; ++bj)
        for (int32_t bi = 0; bi < nb; ++bi) {
            const uint32_t cell = bm->cells[(size_t)(2 * bj) * (size_t)bm->reg[0].w + (size_t)(2 * bi)];
            const int kind = (int)BIOCELL_KIND(cell), hue = (int)BIOCELL_PAL(cell);
            if (kind < BIO_KIND_MAT || kind > BIO_KIND_TREE || hue < 1 || hue > NH) continue;
            SheetPick* p = &pick[kind - 1][hue - 1];
            const int cov = (int)BIOCELL_COVER(cell);
            const double d2 = (double)(bi - nb / 2) * (bi - nb / 2) + (double)(bj - nb / 2) * (bj - nb / 2);
            if (p->found && (cov * 4 + 3 < p->score || (cov * 4 + 3 == p->score && d2 >= p->d2))) continue;   /* cannot win */
            const int32_t xc = x0 + 32 * bi + 16, zc = z0 + 32 * bj + 16, hc = gme_height(xc, zc, seed, pl);
            const int su = sheet_suits(kind, xc, zc, hc, seed, pl, &bv);
            if (!su) continue;
            const int score = cov * 4 + su;                 /* the densest cover first, then the better terrain */
            if (p->found && (score < p->score || (score == p->score && d2 >= p->d2))) continue;
            p->found = 1; p->score = score; p->x = xc; p->z = zc; p->h = hc; p->d2 = d2;
        }
    printf("sheet: blocks chosen in %.1f s\n", time_now() - t0);
    uint32_t* sheet = (uint32_t*)calloc((size_t)SW * SHT, sizeof(uint32_t));
    if (!sheet) return 1;
    RenderSky sky_base, sky;
    build_sky(hp, &sky_base);
    sky_at_time(&sky_base, o->time_set ? o->time_h : 14.0, &sky);
    static const char* kn[NK] = { "mat", "kelp", "cover", "shrub", "tree" };
    const int32_t sea = pl->sea_level;
    int shots = 0;
    for (int k = 0; k < NK; ++k)
        for (int h = 0; h < NH; ++h) {
            const SheetPick* p = &pick[k][h];
            if (!p->found) { printf("sheet: %s hue %d: no block on suitable terrain\n", kn[k], h + 1); continue; }
            const int kind = k + 1;
            const double az = 0.785398;                     /* the camera looks north-east at the block */
            double dist = 16, eye = 0;
            float cy;
            if (kind == BIO_KIND_KELP) { dist = 14; cy = (float)sea + 1 - 1.5f; }
            else if (kind == BIO_KIND_MAT && p->h < sea) { dist = 12; cy = (float)sea + 1 + 9; }
            else {
                eye = kind == BIO_KIND_MAT || kind == BIO_KIND_COVER ? 7 : (kind == BIO_KIND_SHRUB ? 6 : 16);
                if (kind == BIO_KIND_TREE) dist = 30;
                cy = (float)p->h + 1 + (float)eye;
            }
            const double cxw = p->x - dist * cos(az), czw = p->z - dist * sin(az);
            if (kind != BIO_KIND_KELP) {                    /* never inside the ground or a crown at the camera */
                const float hg = (float)gme_height((int32_t)floor(cxw), (int32_t)floor(czw), seed, pl) + 3.f;
                if (cy < hg) cy = hg;
                for (int up = 0; up < 24 && sheet_in_crown(cxw, cy, czw, seed, pl, &bv); ++up) cy += 1.5f;
            }
            const float ty = kind == BIO_KIND_KELP ? (float)p->h + 1 + 0.5f * (float)(sea - p->h)
                           : (kind >= BIO_KIND_SHRUB ? (float)p->h + 2.5f : (float)p->h + 1);
            WorldPos cam;
            memset(&cam, 0, sizeof cam);
            cam.lx = (float)cxw; cam.lz = (float)czw; cam.ly = cy;
            worldpos_normalize(&cam);
            const float yaw = (float)az, pitch = (float)-atan2(cy - ty, dist);
            int loaded = 0;
            const double tl = time_now();
            while (!loaded && time_now() - tl < 20.0) {     /* stream the block's surroundings in (batches finish asynchronously) */
                if (world_update(w, &cam)) { free(sheet); return 1; }
                WorldStats ws;
                world_get_stats(w, &ws);
                loaded = ws.fully_loaded;
                if (!loaded) sleep_ms(2);
            }
            const uint32_t* px = NULL;
            for (int f = 0; f < 48; ++f) {
                RenderScene sc;
                world_get_scene(w, &sc);
                const RenderCamera rc = { .cx = cam.cx, .cy = cam.cy, .cz = cam.cz, .lx = cam.lx, .ly = cam.ly, .lz = cam.lz, .yaw = yaw,
                                          .pitch = pitch, .fov_y = 1.0f, .shadows = 1, .sky = &sky, .quality = o->quality, .frame = (uint32_t)f,
                                          .ev = o->ev_set ? o->ev : 14.5f, .dt = 1.f / 60.f, .wb = o->wb, .anim = 0.f, .reset = f == 0 };
                if (renderer_draw(rd, &sc, &rc, TW, TH, &px)) { free(sheet); return 1; }
                world_set_render_fence(w, renderer_fence(rd));
            }
            if (renderer_finish(rd, &px)) { free(sheet); return 1; }
            for (int y = 0; y < TH; ++y)
                for (int x = 0; x < TW; ++x) {
                    const int edge = x == TW - 1 || y == TH - 1;   /* dark grid lines between the tiles */
                    sheet[(size_t)(k * TH + y) * SW + (size_t)(h * TW + x)] = edge ? 0xFF101010u : px[(size_t)y * TW + x];
                }
            ++shots;
            printf("sheet: %-5s hue %2d: block centre (%d, %d), ground %d (sea %d), cover %d, camera (%.0f, %.1f, %.0f)%s\n", kn[k], h + 1,
                   p->x, p->z, p->h, sea, p->score / 4, cxw, cy, czw, loaded ? "" : " (not fully loaded)");
            fflush(stdout);
        }
    const int rc = write_png(o->sheet, sheet, SW, SHT);
    if (!rc) printf("saved %s (%dx%d: rows mat, kelp, cover, shrub, tree; columns hue bins 1..15; %d of %d tiles)\n", o->sheet, SW, SHT,
                    shots, NK * NH);
    free(sheet);
    return rc ? 1 : 0;
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
    /* M5 drawn shapes: a kelp strand (stipe + blades, at either extreme of its sway) stays inside its 1 m
     * column (the walk's nearest-hit order relies on it), and the clump noise lies in [0, 1] (the clumped crown
     * is the ellipsoid displaced inward by 0 .. 0.35 trad) */
    double kmax = 0, nmin = 1, nmax = 0;
    uint64_t blades = 0;
    for (int32_t len = 1; len <= 40; ++len)
        for (int32_t x = 0; x < 64; ++x) {
            const uint32_t hb = gme_hash3(x, 0, 3, seed ^ 0xF10Au);
            const float y0 = 0.f, y1 = (float)len + 1.f;
            const int n = gme_kelp_nblades(y0, y1);
            for (int sw = 0; sw < 4; ++sw) {
                const float swx = (sw & 1) ? 0.11f : -0.11f, swz = (sw & 2) ? 0.08f : -0.08f;
                kmax = fmax(kmax, sqrt(swx * swx + swz * swz) + GME_KELP_STIPE_R);
                for (int i = 0; i < n; ++i) {
                    GmeKelpBlade b;
                    gme_kelp_blade(i, n, 0.f, y0, 0.f, swx, y1, swz, hb, &b);
                    ++blades;
                    for (int a = 0; a < 64; ++a) {
                        const float t = 6.2831853f * a / 64.f, cu = 0.5f * GME_KELP_BLADE_L * cosf(t), sw2 = 0.5f * GME_KELP_BLADE_W * sinf(t);
                        const double px = b.cx + b.ux * cu + b.wx * sw2, pz = b.cz + b.uz * cu + b.wz * sw2;
                        kmax = fmax(kmax, sqrt(px * px + pz * pz));
                    }
                }
            }
        }
    for (uint32_t k = 0; k < 200000; ++k) {
        const float v = gme_clump_noise((float)(k % 97) * 0.173f, (float)(k % 89) * 0.211f - 3.f, (float)(k / 9973) * 0.37f, 0xC1u + k / 50000, NULL);
        nmin = fmin(nmin, v);
        nmax = fmax(nmax, v);
    }
    const int shapes_ok = kmax < 0.49 + 1e-6 && nmin >= 0.0 && nmax <= 1.0;
    const int ok = !tbad && !kbad && shapes_ok;
    printf("D2 flora: trunks + crowns (ht 2..%d, every trunk position) %llu voxel centres, %llu mismatches; "
           "kelp (len 1..40, 3 depths, %llu strands) %llu voxel centres, %llu mismatches\n",
           GME_FLORA_TREE_MAX, (unsigned long long)tvox, (unsigned long long)tbad, (unsigned long long)strands,
           (unsigned long long)kvox, (unsigned long long)kbad);
    printf("   M5 drawn shapes: kelp stipe + %llu blades at full sway reach %.4f m from the column centre (gate < 0.49: inside "
           "its 1 m column); clump noise in [%.4f, %.4f] (inward only)\nD2 %s\n",
           (unsigned long long)blades, kmax, nmin, nmax, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* D4 (M4) render: (1) the furnace: a flat plane of albedo 0.5 under a uniform
 * sky of radiance L, no sun, must give 0.5 L (+-1 %) after 256 frames of the
 * Medium pipeline (GI rays, radiance cache, SVGF, TAA), and the reference
 * transport the same; (2) an 1800-frame flight with streaming (the bench path,
 * 250 m/s, Medium, headless) without a NaN pixel. */
static int selftest_d4(const HomePlanet* hp, uint32_t seed, const Biomap* bm, int have_life, const Spawn* sp, const SysInfo* si,
                       const SpectralAtmo* ta) {
    if (!ta) { printf("D4 render: skipped (no CUDA device or no spectral light)\n"); return 0; }
    int fails = 0;
    {
        const RendererConfig rc = { 640, 360, RQ_MEDIUM };
        Renderer* r = renderer_create(&rc);
        double m = 0, mn = 0, mx = 0, m2 = 0, mn2 = 0, mx2 = 0;
        uint64_t np = 0, nan = 0, np2 = 0, nan2 = 0;
        const double t0 = time_now();
        const int bad = !r || renderer_set_atmo(r, ta) || renderer_d4_furnace(r, seed, &hp->gen, 256, 0, &m, &mn, &mx, &np, &nan) ||
                        renderer_d4_furnace(r, seed, &hp->gen, 64, 1, &m2, &mn2, &mx2, &np2, &nan2);
        renderer_destroy(r);
        const int ok = !bad && np > 0 && fabs(m - 1.0) <= 0.01 && mn >= 0.99 && mx <= 1.01 && !nan && np2 > 0 && fabs(m2 - 1.0) <= 0.01 &&
                       mn2 >= 0.99 && mx2 <= 1.01 && !nan2;
        printf("D4 furnace (flat plane, albedo 0.5, uniform sky L, no sun): Medium after 256 frames %llu terrain px, L_out / (0.5 L) mean %.5f "
               "min %.5f max %.5f; reference after 64 frames %llu px mean %.5f min %.5f max %.5f; NaN %llu / %llu (gate 1 +- 0.01, %.1f s)\n",
               (unsigned long long)np, m, mn, mx, (unsigned long long)np2, m2, mn2, mx2, (unsigned long long)nan, (unsigned long long)nan2,
               time_now() - t0);
        fails += !ok;
    }
    {
        GpuInfo gi;
        memset(&gi, 0, sizeof gi);
        gpu_query(&gi);
        const RendererConfig rc = { 1280, 720, RQ_MEDIUM };
        Renderer* r = renderer_create(&rc);
        World* w = NULL;
        uint64_t frames = 0, nan = 0;
        int bad = !r || renderer_set_atmo(r, ta);
        if (!bad) {
            gpu_query(&gi);
            WorldConfig wc;
            memset(&wc, 0, sizeof wc);
            wc.seed = seed; wc.planet = hp->gen; wc.min_cy = -8; wc.max_cy = 8;
            wc.vram_fraction = 0.6; wc.ram_fraction = 0.5; wc.max_batch = 2048;
            if (have_life) { biomap_view(bm, &wc.bio); memcpy(wc.bio_palette, bm->palette, sizeof wc.bio_palette); }
            w = world_create(&wc, si, &gi);
            bad = !w;
        }
        const double t0 = time_now();
        if (!bad) {
            RenderSky sky;
            build_sky(hp, &sky);
            WorldPos cam;
            memset(&cam, 0, sizeof cam);
            cam.lx = (float)sp->x; cam.lz = (float)sp->z; cam.ly = (float)sp->y;
            worldpos_normalize(&cam);
            const uint32_t* px;
            int settled = 0;
            double fly_t = 0;
            float yaw = sp->yaw;
            for (uint64_t k = 0; !bad && frames < 1800 && k < 20000; ++k) {
                if (settled) {
                    fly_t += 1.0 / 60.0;
                    const double heading = 0.6 * sin(fly_t * 0.05);
                    cam.lx += (float)(cos(heading) * 250.0 / 60.0);
                    cam.lz += (float)(sin(heading) * 250.0 / 60.0);
                    yaw = (float)heading;
                    worldpos_normalize(&cam);
                }
                if (world_update(w, &cam)) { bad = 1; break; }
                if (!settled) { WorldStats ws; world_get_stats(w, &ws); settled = ws.fully_loaded; }
                RenderScene sc;
                world_get_scene(w, &sc);
                const RenderCamera c = { .cx = cam.cx, .cy = cam.cy, .cz = cam.cz, .lx = cam.lx, .ly = cam.ly, .lz = cam.lz, .yaw = yaw,
                                         .pitch = -0.15f, .fov_y = 1.15f, .shadows = 1, .sky = &sky, .quality = RQ_MEDIUM, .frame = (uint32_t)k,
                                         .ev = NAN, .dt = 1.f / 60.f, .wb = 0.8f, .anim = (float)k / 60.f };
                if (renderer_draw(r, &sc, &c, 1280, 720, &px)) { bad = 1; break; }
                world_set_render_fence(w, renderer_fence(r));
                frames += settled;
            }
            if (!bad && renderer_finish(r, &px)) bad = 1;
            nan = renderer_nan_count(r);
        }
        world_destroy(w);
        renderer_destroy(r);
        const int ok = !bad && frames >= 1800 && nan == 0;
        printf("D4 flight (bench path, 250 m/s, Medium, 1280x720, streaming): %llu frames, NaN pixels %llu (%.1f s)\nD4 %s\n",
               (unsigned long long)frames, (unsigned long long)nan, time_now() - t0, ok && !fails ? "PASS" : "FAIL");
        fails += !ok;
    }
    return fails ? 1 : 0;
}

/* D6 (M3): the far-field clipmap. k_far_update's samples after a full build and
 * after a camera move (toroidal update of both levels), read back and compared
 * with the CPU's gme_far_sample at random slots (half on the rows / columns the
 * move rebuilt); every 8 x 8 mip cell against its samples. */
static int selftest_d6(uint32_t seed, const GmePlanet* pl, const Biomap* bm, int have_life, const Spawn* sp, int gpu_ok) {
    if (!gpu_ok) { printf("D6 far field: skipped (no CUDA device)\n"); return 0; }
    const RendererConfig tc = { 64, 64, RQ_MEDIUM };
    Renderer* r = renderer_create(&tc);
    GmeBioView bv;
    memset(&bv, 0, sizeof bv);
    if (have_life) biomap_view(bm, &bv);
    const size_t nc = have_life ? bm->ncells : 0;
    uint64_t chk = 0, bad = 0, mbad = 0, c2 = 0, b2 = 0, m2 = 0;
    float st[4] = { 0, 0, 0, 0 }, st2[4] = { 0, 0, 0, 0 };
    const int rc = !r || renderer_far_selftest(r, seed, pl, &bv, nc, sp->x, sp->z, 1005, -3003, 4096, &chk, &bad, &mbad, st) ||
                   renderer_far_selftest(r, seed, pl, &bv, nc, sp->x + 50000, sp->z - 20000, -700, 2500, 4096, &c2, &b2, &m2, st2);
    chk += c2; bad += b2; mbad += m2;
    renderer_destroy(r);
    const int ok = !rc && chk > 0 && !bad && !mbad;
    printf("D6 far field: %llu samples (2 levels, 2 builds + camera moves) vs CPU gme_far_sample: %llu mismatches; "
           "8x8 mip cells: %llu mismatches%s\n", (unsigned long long)chk, (unsigned long long)bad, (unsigned long long)mbad,
           rc ? " (GPU error)" : "");
    printf("   update per 16 m camera step (GPU, no streaming): one L1 column avg %.3f max %.3f ms, L1 + L2 column avg %.3f max %.3f ms; "
           "second site %.3f / %.3f, %.3f / %.3f ms (gate 0.3 ms)\nD6 %s\n", st[0], st[1], st[2], st[3], st2[0], st2[1], st2[2], st2[3],
           ok ? "PASS" : "FAIL");
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

    /* ---- spectral light of this planet (spec 4; not needed by the bench) ---- */
    SpectralOpts sopt;
    memset(&sopt, 0, sizeof sopt);
    sopt.p_bar = o.pressure;
    sopt.h2o = o.h2o;
    sopt.sun_disc = o.sun_disc;
    sopt.ms_method = o.ms_hillaire;
    SpectralAtmo* atmo = NULL;
    if (o.export_path) {                    /* M5: the bio's light file, then exit */
        const int rc = export_spectrum(&hp, &sopt, o.export_path);
        jobs_shutdown();
        platform_shutdown();
        return rc;
    }
    if (!o.bench && !o.selftest) {
        atmo = (SpectralAtmo*)calloc(1, sizeof(SpectralAtmo));
        if (!atmo || spectral_build_atmo_ex(&hp, &sopt, atmo)) {
            fprintf(stderr, "spectral light: setup failed, the smooth path is not available\n");
            if (atmo) spectral_free(atmo);
            free(atmo);
            atmo = NULL;
        } else {
            spectral_print(atmo);
        }
    }
    if (o.pigment_table) {                  /* M5: spec 5.1's table, then exit */
        const int rc = atmo ? print_pigment_table(atmo) : 1;
        if (atmo) spectral_free(atmo);
        free(atmo);
        jobs_shutdown();
        platform_shutdown();
        return rc;
    }

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
        } else if (!strcmp(o.view, "mountain")) {
            RenderSky sk;
            build_sky(&hp, &sk);
            view_mountain(&coast, atan2f(sk.sun_dir[2], sk.sun_dir[0]), o.seed, &hp.gen, &sp);
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
        const int M = o.selftest_mask;
        if (M & (1 << 1)) fails += selftest_d1(o.seed, &hp.gen, phash, &bm, have_life, &sp_default, gi.available);
        if (M & (1 << 2)) fails += selftest_d2(o.seed, &hp.gen);
        if (M & ((1 << 3) | (1 << 4))) {   /* D3: physics, and the GPU sky-view LUT against its host twin; D4 render */
            Renderer* trd = NULL;
            SpectralAtmo* ta = NULL;
            int have_gpu = 0;
            if (gi.available) {
                const RendererConfig tc = { 64, 64, RQ_MEDIUM };
                trd = renderer_create(&tc);
                ta = (SpectralAtmo*)calloc(1, sizeof(SpectralAtmo));
                if (!trd || !ta || spectral_build_atmo_ex(&hp, &sopt, ta) || renderer_set_atmo(trd, ta)) {
                    renderer_destroy(trd);
                    trd = NULL;
                }
                have_gpu = trd != NULL;
            }
            if (M & (1 << 3)) fails += spectral_selftest_d3(&hp, &sopt, trd ? renderer_d3_sky : NULL, trd);
            renderer_destroy(trd);
            if (M & (1 << 4)) fails += selftest_d4(&hp, o.seed, &bm, have_life, &sp, &si, have_gpu ? ta : NULL);
            if (ta) spectral_free(ta);
            free(ta);
        }
        if (M & (1 << 5)) fails += spectral_selftest_d5(&hp, &sopt);
        if (M & (1 << 6)) fails += selftest_d6(o.seed, &hp.gen, &bm, have_life, &sp, gi.available);
        printf("SELFTEST: %d failure%s\n", fails, fails == 1 ? "" : "s");
        biomap_free(&bm);
        jobs_shutdown();
        platform_shutdown();
        return fails;
    }

    /* ---- renderer first, so the pool is sized from the VRAM really left ---- */
    Renderer* rd = NULL;
    if (!o.bench && gi.available) {
        /* the smooth path renders at the window size (TAAU from its internal --scale), classic at --scale */
        const double sc = o.scale > 1.0 ? o.scale : 1.0;
        const RendererConfig rc = { (int)(o.width * sc) + 1, (int)(o.height * sc) + 1, o.quality };
        rd = renderer_create(&rc);
        if (rd && atmo && renderer_set_atmo(rd, atmo)) fprintf(stderr, "renderer: the spectral LUTs could not be uploaded\n");
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
    wc.radius_override = o.radius ? o.radius : (o.sheet ? 12 : 0);   /* --sheet: a small world around each close-up */
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
        const int rc = o.sheet ? run_sheet(w, &o, &hp, rd, &bm, o.seed) : run_viewer(w, &o, cam, &hp, sp.yaw, sp.pitch, rd, atmo);
        world_destroy(w);
        renderer_destroy(rd);
        if (atmo) spectral_free(atmo);
        free(atmo);
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

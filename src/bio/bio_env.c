/* bio_env.c - island environments for the life simulation (see bio_env.h,
 * docs/biosphere-spec.md 2.2-2.4).
 *
 * Integer-only and independent of thread scheduling (every cell is computed
 * on its own; the distance transforms are sequential), so the PC and the rigs
 * produce the same env.bin byte for byte. Doubles appear only for the printed
 * statistics and the pictures, never in a stored value. */
#include "bio_env.h"
#include "bio_types.h"      /* bio_hash (copy of gme_hash3) */
#include "../flora.h"       /* gme_env_bin, worldgen.h */
#include "../image.h"
#include "../jobs.h"
#include "../platform.h"
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

#define ENV_MAGIC   "GMEENV01"
#define ENV_VERSION 1u
#define ENV_HDR     128u
#define CHAMFER_INF (1 << 28)

/* ---- small helpers ------------------------------------------------------------- */
static int32_t fdiv(int32_t a, int32_t b) {          /* floor division, b > 0 */
    int32_t q = a / b;
    if ((a % b) != 0 && a < 0) --q;
    return q;
}
static int32_t imin(int32_t a, int32_t b) { return a < b ? a : b; }
static int32_t iclamp(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

static uint32_t g_crc_tab[256];
uint32_t bio_file_crc32(uint32_t crc, const void* p, size_t n) {
    if (!g_crc_tab[1])
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            g_crc_tab[i] = c;
        }
    const uint8_t* b = (const uint8_t*)p;
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = g_crc_tab[(crc ^ b[i]) & 255u] ^ (crc >> 8);
    return ~crc;
}

int bio_mkdirs(const char* path) {
    char buf[1024];
    const size_t n = strlen(path);
    if (n == 0 || n >= sizeof buf) return -1;
    memcpy(buf, path, n + 1);
    for (size_t i = 1; i <= n; ++i) {
        if (buf[i] == '/' || buf[i] == '\\' || buf[i] == 0) {
            const char c = buf[i];
            buf[i] = 0;
            if (!(i == 2 && buf[1] == ':')) MKDIR(buf);   /* skip "C:" */
            buf[i] = c;
        }
    }
    FILE* f;
    char probe[1100];
    snprintf(probe, sizeof probe, "%s/.gme_probe", path);
    f = fopen(probe, "wb");
    if (!f) return -1;
    fclose(f);
    remove(probe);
    return 0;
}

static void put32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
static void put64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
static uint32_t get32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t get64(const uint8_t* p) { return (uint64_t)get32(p) | ((uint64_t)get32(p + 4) << 32); }

/* ---- planet scalars ---------------------------------------------------------------- */
void bio_env_planet(BioEnv* e, const HomePlanet* hp) {
    e->sea_level = hp->gen.sea_level;
    e->t_surf_ck = (int32_t)llround(hp->t_surf * 100.0);
    const double a = hp->a_au > 1e-6 ? hp->a_au : 1.0;
    e->s_top_wm2 = (int32_t)llround(1361.0 * hp->star_lum / (a * a));      /* S = 1361 L / a^2 */
    e->gravity_q8 = (int32_t)llround(hp->gravity_g * 256.0);
    e->organic_ppm = (int32_t)llround(hp->organic_frac * 1e6);
    for (int g = 0; g < BIO_ATM_N; ++g)
        e->atm_ppm[g] = hp->atm_known ? (int32_t)llround(hp->atm_mix[g] * 1e6) : 0;  /* 0 = core default */
    e->seed = hp->system_seed;
    e->planet_hash = planet_hash(hp);
}

/* ---- one cell: 4 x 4 column samples ---------------------------------------------------- */
static void cell_heights(int32_t wx, int32_t wz, int32_t shift, uint32_t seed, const GmePlanet* p, int32_t hs[16]) {
    const int32_t step = (1 << shift) / 4, off = step / 2;           /* 16 m cells: 4 m apart, first at +2 */
    for (int b = 0; b < 4; ++b)
        for (int a = 0; a < 4; ++a) hs[b * 4 + a] = gme_height(wx + off + step * a, wz + off + step * b, seed, p);
}

typedef struct CellMat { int32_t soil, ore, sulf, gran; } CellMat;

/* gme_material 1..8 voxels below each sample (128 voxels) */
static void cell_materials(int32_t wx, int32_t wz, int32_t shift, const int32_t hs[16], uint32_t seed,
                           const GmePlanet* p, CellMat* m) {
    const int32_t step = (1 << shift) / 4, off = step / 2;
    m->soil = m->ore = m->sulf = m->gran = 0;
    for (int k = 0; k < 16; ++k) {
        const int32_t x = wx + off + step * (k & 3), z = wz + off + step * (k >> 2);
        for (int32_t d = 1; d <= 8; ++d) {
            switch (gme_material(x, hs[k] - d, z, hs[k], seed, p)) {
            case MAT_DIRT:     m->soil++; break;
            case MAT_IRON_ORE: m->ore++; break;
            case MAT_SULFIDE:  m->sulf++; break;
            case MAT_GRANITE:  m->gran++; break;
            default: break;
            }
        }
    }
}

/* Same test as cell_materials' ore + sulf >= BIO_VENT_ORE_MIN, stopping as
 * soon as the answer is known (the scout runs it on every deep cell). */
static int cell_ore_rich(int32_t wx, int32_t wz, int32_t shift, const int32_t hs[16], uint32_t seed, const GmePlanet* p) {
    const int32_t step = (1 << shift) / 4, off = step / 2;
    int32_t n = 0;
    for (int k = 0; k < 16; ++k) {
        if (n + (16 - k) * 8 < BIO_VENT_ORE_MIN) return 0;
        const int32_t x = wx + off + step * (k & 3), z = wz + off + step * (k >> 2);
        for (int32_t d = 1; d <= 8; ++d) {
            const uint8_t m = gme_material(x, hs[k] - d, z, hs[k], seed, p);
            if (m == MAT_IRON_ORE || m == MAT_SULFIDE) ++n;
        }
        if (n >= BIO_VENT_ORE_MIN) return 1;
    }
    return 0;
}

static int32_t cell_elev(int32_t sum16) { return fdiv(sum16 + 8, 16); }   /* rounded mean of 16 samples */
static int32_t f255(int32_t count) { return (count * 255 + 64) / 128; }    /* of 128 voxels -> 0..255 */

/* Vent rule (spec 2.4): depth > 15, ore- and sulfide-rich crust (f_ore + f_sulf > 96,
 * evaluated on the voxel counts) and (bio_hash(i, j, 0, seed) & 1023) < vent_q10,
 * with (i, j) the global cell index so vents are a property of the planet. */
static uint32_t vent_hash(int32_t wx, int32_t wz, int32_t shift, uint32_t seed) {
    return bio_hash(fdiv(wx, 1 << shift), fdiv(wz, 1 << shift), 0, seed);
}
static uint8_t vent_strength(uint32_t h) { return (uint8_t)(192u + ((h >> 10) & 63u)); }   /* 192..255, tuned */

uint32_t bio_env_bin_at(int32_t x, int32_t z, int32_t shift, uint32_t seed, const GmePlanet* p) {
    const int32_t cx = x + (1 << shift) / 2, cz = z + (1 << shift) / 2;
    return gme_env_bin(cx, cz, gme_height(cx, cz, seed, p), seed, p) & 63u;
}

/* 3-4 chamfer distance, two passes, in place (0 = source) */
static void chamfer34(int32_t* d, int32_t w, int32_t h) {
    for (int32_t z = 0; z < h; ++z)
        for (int32_t x = 0; x < w; ++x) {
            const int64_t i = (int64_t)z * w + x;
            int32_t v = d[i];
            if (x > 0) v = imin(v, d[i - 1] + 3);
            if (z > 0) {
                v = imin(v, d[i - w] + 3);
                if (x > 0) v = imin(v, d[i - w - 1] + 4);
                if (x < w - 1) v = imin(v, d[i - w + 1] + 4);
            }
            d[i] = v;
        }
    for (int32_t z = h - 1; z >= 0; --z)
        for (int32_t x = w - 1; x >= 0; --x) {
            const int64_t i = (int64_t)z * w + x;
            int32_t v = d[i];
            if (x < w - 1) v = imin(v, d[i + 1] + 3);
            if (z < h - 1) {
                v = imin(v, d[i + w] + 3);
                if (x < w - 1) v = imin(v, d[i + w + 1] + 4);
                if (x > 0) v = imin(v, d[i + w - 1] + 4);
            }
            d[i] = v;
        }
}

/* ======================================================================================
 * Scout (2.2)
 * ====================================================================================== */
#define SC_TILE    BIO_SCOUT_LATTICE                       /* m: windows are 8 x 8 tiles */
#define SC_TS      (SC_TILE / BIO_SCOUT_SAMPLE)            /* 32 samples per tile edge */
#define SC_WT      ((BIO_ENV_CELLS << BIO_ENV_SHIFT) / SC_TILE)   /* 8 tiles per window edge */
#define SC_CELLS_T (SC_TILE >> BIO_ENV_SHIFT)              /* 128 cells per tile edge */
#define SC_MARGIN  30                                      /* m: tiles this far above sea cannot hold vents */

typedef struct VentRec { int32_t gi, gj; uint32_t h; } VentRec;
typedef struct VentList { VentRec* v; uint32_t n, cap; } VentList;

typedef struct Scout {
    const GmePlanet* p;
    uint32_t seed;
    int32_t  imax, nt, ns;          /* centre index range, tiles and samples per axis */
    int32_t  x_lo;                  /* world x (= z) of sample 0 / tile 0 */
    int16_t* hs;                    /* ns * ns sample heights */
    int32_t* dist;                  /* chamfer distance to ocean, 64 m units x 3 */
    int32_t* t_ocean; int32_t* t_shelf; int32_t* t_coast; int32_t* t_minh; int32_t* t_vents;
    int64_t* t_deep;                /* deep (> 15 m) 16 m cells per tile (vent-scanned tiles only) */
    int32_t* vtiles; int32_t nvt;   /* tiles scanned for vents */
    VentList* vl;                   /* per worker */
} Scout;

static void scout_heights(void* ctx, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    Scout* S = (Scout*)ctx;
    for (uint32_t row = b; row < e; ++row) {
        const int32_t z = S->x_lo + BIO_SCOUT_SAMPLE * (int32_t)row + BIO_SCOUT_SAMPLE / 2;
        for (int32_t c = 0; c < S->ns; ++c) {
            const int32_t x = S->x_lo + BIO_SCOUT_SAMPLE * c + BIO_SCOUT_SAMPLE / 2;
            S->hs[(int64_t)row * S->ns + c] = (int16_t)gme_height(x, z, S->seed, S->p);
        }
    }
}

static void scout_vents(void* ctx, uint32_t b, uint32_t e, uint32_t wk) {
    Scout* S = (Scout*)ctx;
    const int32_t sea = S->p->sea_level, cell = 1 << BIO_ENV_SHIFT;
    VentList* L = &S->vl[wk];
    for (uint32_t q = b; q < e; ++q) {
        const int32_t t = S->vtiles[q];
        const int32_t tx = t % S->nt, tz = t / S->nt;
        const int32_t wx0 = S->x_lo + SC_TILE * tx, wz0 = S->x_lo + SC_TILE * tz;
        int64_t deep = 0;
        for (int32_t j = 0; j < SC_CELLS_T; ++j)
            for (int32_t i = 0; i < SC_CELLS_T; ++i) {
                const int32_t wx = wx0 + cell * i, wz = wz0 + cell * j;
                int32_t hs[16], sum = 0;
                cell_heights(wx, wz, BIO_ENV_SHIFT, S->seed, S->p, hs);
                for (int k = 0; k < 16; ++k) sum += hs[k];
                if (sea - cell_elev(sum) <= BIO_VENT_DEPTH) continue;
                ++deep;
                if (!cell_ore_rich(wx, wz, BIO_ENV_SHIFT, hs, S->seed, S->p)) continue;
                if (L->n == L->cap) {
                    L->cap = L->cap ? 2 * L->cap : 256;
                    VentRec* nv = (VentRec*)realloc(L->v, sizeof(VentRec) * L->cap);
                    if (!nv) { L->cap = L->n; continue; }   /* out of memory: drop (reported as fewer vents) */
                    L->v = nv;
                }
                L->v[L->n].gi = fdiv(wx, cell); L->v[L->n].gj = fdiv(wz, cell);
                L->v[L->n].h = vent_hash(wx, wz, BIO_ENV_SHIFT, S->seed);
                L->n++;
            }
        S->t_deep[t] = deep;
    }
}

typedef struct Cand { int32_t i, j; int64_t score; int32_t ocean, shelf, coast, vents; } Cand;
static int cand_cmp(const void* a, const void* b) {
    const Cand* x = (const Cand*)a; const Cand* y = (const Cand*)b;
    if (x->score != y->score) return x->score > y->score ? -1 : 1;
    const int32_t dx = abs(x->i) + abs(x->j), dy = abs(y->i) + abs(y->j);
    if (dx != dy) return dx < dy ? -1 : 1;
    if (x->i != y->i) return x->i < y->i ? -1 : 1;
    return x->j < y->j ? -1 : (x->j > y->j ? 1 : 0);
}
static int vent_cmp(const void* a, const void* b) {
    const VentRec* x = (const VentRec*)a; const VentRec* y = (const VentRec*)b;
    if (x->gj != y->gj) return x->gj < y->gj ? -1 : 1;
    return x->gi < y->gi ? -1 : (x->gi > y->gi ? 1 : 0);
}

/* overview picture: one pixel per 4 x 4 samples (256 m), chosen windows outlined */
static void scout_png(const Scout* S, const BioRegions* out, const VentRec* vents, uint32_t nv, uint32_t q, const char* path) {
    const int32_t W = S->ns / 4, sea = S->p->sea_level;
    uint32_t* px = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)W * W);
    if (!px) return;
    for (int32_t y = 0; y < W; ++y)
        for (int32_t x = 0; x < W; ++x) {
            const int32_t h = S->hs[(int64_t)(4 * y + 2) * S->ns + 4 * x + 2];
            int r, g, b;
            if (h < sea) {
                const int d = iclamp(sea - h, 0, 120);
                r = 60 - d / 3; g = 140 - d; b = 200 - d / 2;
            } else {
                const int u = iclamp(h - sea, 0, 220);
                r = 90 + u / 2; g = 130 + u / 4; b = 70 + u / 2;
            }
            px[(size_t)y * W + x] = 0xFF000000u | ((uint32_t)iclamp(r, 0, 255) << 16) | ((uint32_t)iclamp(g, 0, 255) << 8) |
                                    (uint32_t)iclamp(b, 0, 255);
        }
    for (uint32_t k = 0; k < nv; ++k) {
        if ((vents[k].h & 1023u) >= q) continue;
        const int32_t x = (vents[k].gi * (1 << BIO_ENV_SHIFT) - S->x_lo) / 256, y = (vents[k].gj * (1 << BIO_ENV_SHIFT) - S->x_lo) / 256;
        if (x >= 0 && y >= 0 && x < W && y < W) px[(size_t)y * W + x] = 0xFFFF3020u;
    }
    static const uint32_t col[BIO_ISLANDS] = { 0xFFFFFF00u, 0xFFFF8000u, 0xFFFF00FFu, 0xFF00FFFFu, 0xFFFFFFFFu, 0xFFFF2020u };
    for (int r = 0; r < out->n; ++r) {
        const int32_t x0 = (out->r[r].x0 - S->x_lo) / 256, z0 = (out->r[r].z0 - S->x_lo) / 256;
        const int32_t side = (BIO_ENV_CELLS << BIO_ENV_SHIFT) / 256;
        for (int t = 0; t < 2; ++t)
            for (int32_t s = 0; s < side; ++s) {
                const int32_t pts[4][2] = { { x0 + s, z0 + t }, { x0 + s, z0 + side - 1 - t }, { x0 + t, z0 + s }, { x0 + side - 1 - t, z0 + s } };
                for (int k = 0; k < 4; ++k)
                    if (pts[k][0] >= 0 && pts[k][1] >= 0 && pts[k][0] < W && pts[k][1] < W)
                        px[(size_t)pts[k][1] * W + pts[k][0]] = col[r % BIO_ISLANDS];
            }
        for (int d = 0; d <= out->r[r].k; ++d)          /* island number: k + 1 dots in the corner */
            for (int yy = 0; yy < 3; ++yy)
                for (int xx = 0; xx < 3; ++xx) {
                    const int32_t x = x0 + 4 + 5 * d + xx, y = z0 + 4 + yy;
                    if (x >= 0 && y >= 0 && x < W && y < W) px[(size_t)y * W + x] = col[r % BIO_ISLANDS];
                }
    }
    if (write_png(path, px, W, W)) fprintf(stderr, "scout: cannot write %s\n", path);
    else printf("scout: wrote %s (%dx%d, 256 m per pixel; islands outlined, k+1 dots, vents red)\n", path, W, W);
    free(px);
}

int bio_env_scout(const HomePlanet* hp, BioRegions* out, const char* png) {
    memset(out, 0, sizeof *out);
    Scout S;
    memset(&S, 0, sizeof S);
    S.p = &hp->gen;
    S.seed = hp->system_seed;
    S.imax = BIO_SCOUT_RANGE / BIO_SCOUT_LATTICE;                 /* 62 */
    S.nt = 2 * S.imax + 2 * (SC_WT / 2);                          /* 132 tiles */
    S.ns = S.nt * SC_TS;                                          /* 4224 samples */
    S.x_lo = -SC_TILE * (S.imax + SC_WT / 2);
    const int32_t sea = S.p->sea_level, ns = S.ns, nt = S.nt;
    const int64_t nsamp = (int64_t)ns * ns, ntile = (int64_t)nt * nt;
    const double t0 = time_now();
    printf("scout: planet %016llx, world seed %u, sea level %d (from the planet file)\n",
           (unsigned long long)planet_hash(hp), S.seed, sea);
    printf("scout: %dx%d window centres on a %d m lattice within +-%d m; %d m windows sampled every %d m\n",
           2 * S.imax + 1, 2 * S.imax + 1, BIO_SCOUT_LATTICE, BIO_SCOUT_RANGE, BIO_ENV_CELLS << BIO_ENV_SHIFT, BIO_SCOUT_SAMPLE);
    S.hs = (int16_t*)malloc(sizeof(int16_t) * (size_t)nsamp);
    S.dist = (int32_t*)malloc(sizeof(int32_t) * (size_t)nsamp);
    S.t_ocean = (int32_t*)calloc((size_t)ntile, 4); S.t_shelf = (int32_t*)calloc((size_t)ntile, 4);
    S.t_coast = (int32_t*)calloc((size_t)ntile, 4); S.t_minh = (int32_t*)calloc((size_t)ntile, 4);
    S.t_vents = (int32_t*)calloc((size_t)ntile, 4); S.t_deep = (int64_t*)calloc((size_t)ntile, 8);
    S.vtiles = (int32_t*)malloc(sizeof(int32_t) * (size_t)ntile);
    S.vl = (VentList*)calloc((size_t)jobs_thread_count(), sizeof(VentList));
    int rc = -1;
    Cand* cand = NULL;
    VentRec* vents = NULL;
    if (!S.hs || !S.dist || !S.t_ocean || !S.t_shelf || !S.t_coast || !S.t_minh || !S.t_vents || !S.t_deep || !S.vtiles || !S.vl) {
        fprintf(stderr, "scout: out of memory\n");
        goto done;
    }
    /* 1. heights on the 64 m lattice */
    jobs_parallel_for((uint32_t)ns, 4, scout_heights, &S);
    /* 2. distance of every sample to the ocean; per-tile counts */
    int64_t n_ocean = 0;
    for (int64_t i = 0; i < nsamp; ++i) { const int o = S.hs[i] < sea; S.dist[i] = o ? 0 : CHAMFER_INF; n_ocean += o; }
    chamfer34(S.dist, ns, ns);
    for (int64_t i = 0; i < ntile; ++i) S.t_minh[i] = 1 << 20;
    for (int32_t z = 0; z < ns; ++z)
        for (int32_t x = 0; x < ns; ++x) {
            const int64_t i = (int64_t)z * ns + x, t = (int64_t)(z / SC_TS) * nt + x / SC_TS;
            const int32_t h = S.hs[i], depth = sea - h;
            if (depth >= 1) { S.t_ocean[t]++; if (depth <= 60) S.t_shelf[t]++; }
            else if ((int64_t)S.dist[i] * BIO_SCOUT_SAMPLE <= 3 * 1000) S.t_coast[t]++;   /* land within 1 km of the coast */
            if (h < S.t_minh[t]) S.t_minh[t] = h;
        }
    printf("scout: %lld height samples in %.1f s, ocean %.1f%% of the scanned area\n", (long long)nsamp, time_now() - t0,
           100.0 * (double)n_ocean / (double)nsamp);
    /* 3. vents: every deep 16 m cell of the tiles that reach below sea + margin */
    for (int64_t t = 0; t < ntile; ++t) if (S.t_minh[t] < sea + SC_MARGIN) S.vtiles[S.nvt++] = (int32_t)t;
    const double t1 = time_now();
    jobs_parallel_for((uint32_t)S.nvt, 1, scout_vents, &S);
    uint32_t nv = 0;
    for (int w = 0; w < jobs_thread_count(); ++w) nv += S.vl[w].n;
    vents = (VentRec*)malloc(sizeof(VentRec) * (nv ? nv : 1));
    if (!vents) { fprintf(stderr, "scout: out of memory\n"); goto done; }
    nv = 0;
    for (int w = 0; w < jobs_thread_count(); ++w) { memcpy(vents + nv, S.vl[w].v, sizeof(VentRec) * S.vl[w].n); nv += S.vl[w].n; }
    qsort(vents, nv, sizeof(VentRec), vent_cmp);
    int64_t deep = 0;
    for (int64_t t = 0; t < ntile; ++t) deep += S.t_deep[t];
    /* thinning so that about 1 vent per km^2 of deep seafloor remains (all eligible cells if they are rarer) */
    uint32_t q = 1024;
    if (nv) {
        const int64_t num = (int64_t)1024 * deep * (1 << (2 * BIO_ENV_SHIFT)), den = (int64_t)1000000 * nv;
        const int64_t qq = (num + den / 2) / den;
        q = (uint32_t)(qq < 1 ? 1 : (qq > 1024 ? 1024 : qq));
    }
    out->vent_q10 = q;
    uint32_t nvent = 0;
    for (uint32_t k = 0; k < nv; ++k) {
        if ((vents[k].h & 1023u) >= q) continue;
        const int32_t tx = fdiv(vents[k].gi * (1 << BIO_ENV_SHIFT) - S.x_lo, SC_TILE), tz = fdiv(vents[k].gj * (1 << BIO_ENV_SHIFT) - S.x_lo, SC_TILE);
        if (tx >= 0 && tz >= 0 && tx < nt && tz < nt) { S.t_vents[(int64_t)tz * nt + tx]++; ++nvent; }
    }
    const double deep_km2 = (double)deep * (1 << (2 * BIO_ENV_SHIFT)) / 1e6;
    printf("scout: %d tiles of %d m scanned for vents in %.1f s: %.0f km^2 of seafloor deeper than %d m, %u ore-rich cells "
           "(%.2f per km^2) -> hash threshold %u/1024, %u vents\n",
           S.nvt, SC_TILE, time_now() - t1, deep_km2, BIO_VENT_DEPTH, nv, deep_km2 > 0 ? nv / deep_km2 : 0.0, q, nvent);
    /* 4. windows */
    const int32_t nc = 2 * S.imax + 1;
    cand = (Cand*)malloc(sizeof(Cand) * (size_t)nc * nc);
    if (!cand) { fprintf(stderr, "scout: out of memory\n"); goto done; }
    const int32_t tot = (SC_WT * SC_TS) * (SC_WT * SC_TS);     /* samples per window */
    int32_t ncand = 0, n_ocean_ok = 0, n_vent_ok = 0, n_vent20 = 0, max_ocean = 0, max_vents = 0;
    for (int32_t j = -S.imax; j <= S.imax; ++j)
        for (int32_t i = -S.imax; i <= S.imax; ++i) {
            Cand c;
            memset(&c, 0, sizeof c);
            c.i = i; c.j = j;
            const int32_t ti0 = i + S.imax, tj0 = j + S.imax;   /* first tile of the window */
            for (int32_t b = 0; b < SC_WT; ++b)
                for (int32_t a = 0; a < SC_WT; ++a) {
                    const int64_t t = (int64_t)(tj0 + b) * nt + ti0 + a;
                    c.ocean += S.t_ocean[t]; c.shelf += S.t_shelf[t]; c.coast += S.t_coast[t]; c.vents += S.t_vents[t];
                }
            c.score = 2 * (int64_t)c.shelf + c.coast + 8 * (int64_t)c.vents;
            cand[ncand++] = c;
            if (4 * c.ocean >= tot && 2 * c.ocean <= tot) { ++n_ocean_ok; if (c.vents >= BIO_VENTS_MIN) ++n_vent_ok; }
            n_vent20 += c.vents >= BIO_VENTS_MIN;
            if (c.ocean > max_ocean) max_ocean = c.ocean;
            if (c.vents > max_vents) max_vents = c.vents;
        }
    qsort(cand, (size_t)ncand, sizeof(Cand), cand_cmp);
    printf("scout: %d windows: %d with 25-50%% ocean (%d of them with >= %d vents); %d with >= %d vents at any ocean share; "
           "largest ocean share %.1f%%, most vents %d\n", ncand, n_ocean_ok, n_vent_ok, BIO_VENTS_MIN, n_vent20, BIO_VENTS_MIN,
           100.0 * max_ocean / tot, max_vents);
    /* 5. greedy pick by score, >= 20 km apart. The spec's rule first; when it cannot fill all
     * islands, the ocean share is relaxed before the vent count, each step logged. */
    static const struct { int lo, hi, vmin; const char* why; } LV[] = {     /* ocean share in per mille */
        { 250, 500, BIO_VENTS_MIN, "" },
        { 150, 600, BIO_VENTS_MIN, "ocean share 15-60% instead of 25-50%" },
        { 100, 600, BIO_VENTS_MIN, "ocean share 10-60% instead of 25-50%" },
        { 150, 600, 1, "ocean share 15-60% and fewer than 20 vents" },
        { 100, 600, 1, "ocean share 10-60% and fewer than 20 vents" },
        { 1, 1000, 0, "any ocean share, possibly no vents" },
    };
    for (int level = 0; level < (int)(sizeof LV / sizeof LV[0]) && out->n < BIO_ISLANDS; ++level) {
        int added = 0;
        for (int32_t k = 0; k < ncand && out->n < BIO_ISLANDS; ++k) {
            const Cand* c = &cand[k];
            const int ocean_ok = 1000 * (int64_t)c->ocean >= (int64_t)LV[level].lo * tot && 1000 * (int64_t)c->ocean <= (int64_t)LV[level].hi * tot;
            if (!ocean_ok || c->shelf <= 0 || c->coast <= 0 || c->vents < LV[level].vmin) continue;
            int far = 1;
            for (int r = 0; r < out->n && far; ++r) {
                const int64_t dx = (int64_t)(c->i * BIO_SCOUT_LATTICE) - (out->r[r].x0 + (BIO_ENV_CELLS << BIO_ENV_SHIFT) / 2);
                const int64_t dz = (int64_t)(c->j * BIO_SCOUT_LATTICE) - (out->r[r].z0 + (BIO_ENV_CELLS << BIO_ENV_SHIFT) / 2);
                far = dx * dx + dz * dz >= (int64_t)BIO_SCOUT_SEP * BIO_SCOUT_SEP;
            }
            if (!far) continue;
            BioRegion* R = &out->r[out->n];
            R->k = out->n;
            R->x0 = c->i * BIO_SCOUT_LATTICE - (BIO_ENV_CELLS << BIO_ENV_SHIFT) / 2;
            R->z0 = c->j * BIO_SCOUT_LATTICE - (BIO_ENV_CELLS << BIO_ENV_SHIFT) / 2;
            R->score = c->score; R->ocean = c->ocean; R->shelf = c->shelf; R->coastland = c->coast; R->vents = c->vents;
            out->n++;
            ++added;
        }
        if (level > 0 && added)
            printf("scout: WARNING: relaxed the rules (%s) for %d island%s\n", LV[level].why, added, added == 1 ? "" : "s");
    }
    out->planet_hash = planet_hash(hp);
    out->seed = S.seed;
    printf("\n  k   centre x   centre z         x0         z0     score  ocean   shelf  coast<1km  vents\n");
    for (int r = 0; r < out->n; ++r) {
        const BioRegion* R = &out->r[r];
        const int32_t half = (BIO_ENV_CELLS << BIO_ENV_SHIFT) / 2;
        printf("  %d %10d %10d %10d %10d %9lld %5.1f%% %7d %10d %6d\n", R->k, R->x0 + half, R->z0 + half, R->x0, R->z0,
               (long long)R->score, 100.0 * R->ocean / tot, R->shelf, R->coastland, R->vents);
    }
    printf("\nscout: %d islands (0-4 for the rig GPUs, 5 = PC lab) in %.1f s\n", out->n, time_now() - t0);
    if (png) scout_png(&S, out, vents, nv, q, png);
    rc = out->n == BIO_ISLANDS ? 0 : -1;
    if (rc) fprintf(stderr, "scout: only %d islands found\n", out->n);
done:
    if (S.vl) for (int w = 0; w < jobs_thread_count(); ++w) free(S.vl[w].v);
    free(S.vl); free(S.hs); free(S.dist); free(S.t_ocean); free(S.t_shelf); free(S.t_coast); free(S.t_minh);
    free(S.t_vents); free(S.t_deep); free(S.vtiles); free(cand); free(vents);
    return rc;
}

int bio_regions_save(const char* path, const BioRegions* r, const HomePlanet* hp) {
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE* f = fopen(tmp, "w");
    if (!f) return -1;
    const int32_t tot = (BIO_ENV_CELLS << BIO_ENV_SHIFT) / BIO_SCOUT_SAMPLE;
    fprintf(f, "# GME life regions, written by gme_genesis env --scout (docs/biosphere-spec.md 2.2)\n");
    fprintf(f, "# planet %016llx seed %u sea_level %d\n", (unsigned long long)r->planet_hash, r->seed, hp->gen.sea_level);
    fprintf(f, "# window %d m = %d x %d cells of %d m; centres on a %d m lattice within +-%d m, >= %d m apart\n",
            BIO_ENV_CELLS << BIO_ENV_SHIFT, BIO_ENV_CELLS, BIO_ENV_CELLS, 1 << BIO_ENV_SHIFT, BIO_SCOUT_LATTICE,
            BIO_SCOUT_RANGE, BIO_SCOUT_SEP);
    fprintf(f, "# vent_q10 %u\n", r->vent_q10);
    fprintf(f, "# islands 0-4: rig GPUs, 5: PC lab. Data lines: k x0 z0 (world corner of cell 0,0)\n");
    for (int k = 0; k < r->n; ++k) {
        const BioRegion* R = &r->r[k];
        fprintf(f, "%d %d %d   # score %lld ocean %.1f%% shelf %d coast %d vents %d\n", R->k, R->x0, R->z0,
                (long long)R->score, 100.0 * R->ocean / ((double)tot * tot), R->shelf, R->coastland, R->vents);
    }
    const int bad = ferror(f);
    if (fclose(f) || bad) { remove(tmp); return -1; }
    remove(path);
    return rename(tmp, path) ? -1 : 0;
}

int bio_regions_load(const char* path, BioRegions* r) {
    memset(r, 0, sizeof *r);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    char line[512];
    int have_hash = 0;
    r->vent_q10 = 1024;
    while (fgets(line, sizeof line, f)) {
        unsigned long long h;
        unsigned s, q;
        int k, x0, z0;
        if (sscanf(line, "# planet %llx seed %u", &h, &s) == 2) { r->planet_hash = h; r->seed = s; have_hash = 1; continue; }
        if (sscanf(line, "# vent_q10 %u", &q) == 1) { r->vent_q10 = q > 1024 ? 1024 : q; continue; }
        if (line[0] == '#') continue;
        if (sscanf(line, "%d %d %d", &k, &x0, &z0) == 3 && r->n < BIO_ISLANDS) {
            BioRegion* R = &r->r[r->n++];
            memset(R, 0, sizeof *R);
            R->k = k; R->x0 = x0; R->z0 = z0;
        }
    }
    fclose(f);
    return have_hash && r->n > 0 ? 0 : -2;
}

/* ======================================================================================
 * Island environment (2.3)
 * ====================================================================================== */
typedef struct Build {
    BioEnv* e;
    const GmePlanet* p;
    uint32_t seed, vent_q10;
    int32_t  pw;              /* padded width = w + 2 * pad */
    int32_t* sum;             /* padded: sum of the 16 sample heights */
} Build;

static void build_rows(void* ctx, uint32_t b, uint32_t e_, uint32_t wk) {
    (void)wk;
    Build* B = (Build*)ctx;
    BioEnv* e = B->e;
    const int32_t cell = 1 << e->shift, sea = e->sea_level, pad = BIO_ENV_PAD;
    for (uint32_t row = b; row < e_; ++row) {
        const int32_t jz = (int32_t)row - pad;                 /* grid row (may be in the padding) */
        for (int32_t px = 0; px < B->pw; ++px) {
            const int32_t ix = px - pad;
            const int32_t wx = e->x0 + cell * ix, wz = e->z0 + cell * jz;
            int32_t hs[16], sum = 0;
            cell_heights(wx, wz, e->shift, B->seed, B->p, hs);
            for (int k = 0; k < 16; ++k) sum += hs[k];
            B->sum[(int64_t)row * B->pw + px] = sum;
            if (ix < 0 || jz < 0 || ix >= e->w || jz >= e->h) continue;
            const int64_t c = (int64_t)jz * e->w + ix;
            const int32_t elev = cell_elev(sum), depth = sea - elev;
            e->elev[c] = (int16_t)elev;
            e->depth[c] = (int16_t)depth;
            for (int t = 0; t < 3; ++t) {                      /* low, mid, high tide: sea - 1, sea, sea + 1 */
                int32_t n = 0;
                for (int k = 0; k < 16; ++k) n += hs[k] <= sea + t - 1;
                e->wet[t][c] = (uint8_t)imin(255, 16 * n);
            }
            CellMat m;
            cell_materials(wx, wz, e->shift, hs, B->seed, B->p, &m);
            e->f_soil[c] = (uint8_t)f255(m.soil); e->f_ore[c] = (uint8_t)f255(m.ore);
            e->f_sulf[c] = (uint8_t)f255(m.sulf); e->f_gran[c] = (uint8_t)f255(m.gran);
            uint8_t vent = 0;
            if (depth > BIO_VENT_DEPTH && m.ore + m.sulf >= BIO_VENT_ORE_MIN) {
                const uint32_t h = vent_hash(wx, wz, e->shift, B->seed);
                if ((h & 1023u) < B->vent_q10) vent = vent_strength(h);
            }
            e->vent[c] = vent;
            e->env_bin[c] = (uint8_t)bio_env_bin_at(wx, wz, e->shift, B->seed, B->p);
        }
    }
}

int bio_env_build(BioEnv* e, const HomePlanet* hp, const BioRegions* rg, int island) {
    const BioRegion* R = NULL;
    for (int k = 0; k < rg->n; ++k) if (rg->r[k].k == island) R = &rg->r[k];
    if (!R) { fprintf(stderr, "env: island %d is not in the regions file\n", island); return -1; }
    memset(e, 0, sizeof *e);
    if (bio_env_alloc(e, BIO_ENV_CELLS, BIO_ENV_CELLS)) { fprintf(stderr, "env: %s\n", bio_error()); return -1; }
    bio_env_planet(e, hp);
    e->shift = BIO_ENV_SHIFT;
    e->x0 = R->x0; e->z0 = R->z0;
    e->island = island;
    Build B;
    memset(&B, 0, sizeof B);
    B.e = e; B.p = &hp->gen; B.seed = hp->system_seed; B.vent_q10 = rg->vent_q10;
    B.pw = e->w + 2 * BIO_ENV_PAD;
    const int32_t ph = e->h + 2 * BIO_ENV_PAD;
    B.sum = (int32_t*)malloc(sizeof(int32_t) * (size_t)B.pw * ph);
    int32_t* dist = (int32_t*)malloc(sizeof(int32_t) * (size_t)B.pw * ph);
    if (!B.sum || !dist) { free(B.sum); free(dist); bio_env_free(e); fprintf(stderr, "env: out of memory\n"); return -1; }
    jobs_parallel_for((uint32_t)ph, 2, build_rows, &B);
    const int32_t sea = e->sea_level, W = e->w, H = e->h, pw = B.pw, pad = BIO_ENV_PAD;
    /* coast distance: 3-4 chamfer over the padded grid (ocean = mean depth >= 1 m), in cells */
    for (int64_t i = 0; i < (int64_t)pw * ph; ++i) dist[i] = sea - cell_elev(B.sum[i]) >= 1 ? 0 : CHAMFER_INF;
    chamfer34(dist, pw, ph);
    for (int32_t j = 0; j < H; ++j)
        for (int32_t i = 0; i < W; ++i) {
            const int32_t d = dist[(int64_t)(j + pad) * pw + i + pad];
            e->coast[(int64_t)j * W + i] = (uint16_t)(d >= CHAMFER_INF ? 65535 : imin(65535, (d + 1) / 3));
        }
    /* D8 drainage of land cells to the steepest lower neighbour inside the grid
     * (slopes from the 16-sample sums, 1/16 m resolution); 255 = sea or sink */
    static const int32_t dx8[8] = { 1, 1, 0, -1, -1, -1, 0, 1 }, dz8[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    for (int32_t j = 0; j < H; ++j)
        for (int32_t i = 0; i < W; ++i) {
            const int64_t c = (int64_t)j * W + i;
            uint8_t best = 255;
            if (e->depth[c] < 1) {
                const int64_t here = B.sum[(int64_t)(j + pad) * pw + i + pad];
                int64_t bs = 0;
                for (int k = 0; k < 8; ++k) {
                    const int32_t ni = i + dx8[k], nj = j + dz8[k];
                    if (ni < 0 || nj < 0 || ni >= W || nj >= H) continue;
                    const int64_t drop = here - B.sum[(int64_t)(nj + pad) * pw + ni + pad];
                    const int64_t s = drop * ((k & 1) ? 7071 : 10000);     /* per unit distance */
                    if (s > bs) { bs = s; best = (uint8_t)k; }
                }
            }
            e->flow_dir[c] = best;
        }
    free(B.sum);
    free(dist);
    e->crc = bio_env_crc(e);
    return 0;
}

uint32_t bio_env_vents(const BioEnv* e) {
    uint32_t n = 0;
    for (int64_t c = 0; c < (int64_t)e->w * e->h; ++c) n += e->vent[c] > 0;
    return n;
}

/* ---- env.bin ------------------------------------------------------------------------- */
typedef struct Arr { void* p; int bytes; } Arr;
static int env_arrays(const BioEnv* e, Arr a[13]) {
    const Arr t[13] = { { e->elev, 2 }, { e->depth, 2 }, { e->wet[0], 1 }, { e->wet[1], 1 }, { e->wet[2], 1 },
                        { e->f_soil, 1 }, { e->f_ore, 1 }, { e->f_sulf, 1 }, { e->f_gran, 1 }, { e->vent, 1 },
                        { e->flow_dir, 1 }, { e->env_bin, 1 }, { e->coast, 2 } };
    memcpy(a, t, sizeof t);
    return 13;
}

int bio_env_save(const char* path, const BioEnv* e, uint32_t vent_q10, uint32_t* file_crc) {
    const size_t n = (size_t)e->w * e->h;
    uint8_t hd[ENV_HDR];
    memset(hd, 0, sizeof hd);
    memcpy(hd, ENV_MAGIC, 8);
    put32(hd + 8, ENV_VERSION); put32(hd + 12, ENV_HDR);
    put64(hd + 16, e->planet_hash); put32(hd + 24, e->seed); put32(hd + 28, (uint32_t)e->island);
    put32(hd + 32, (uint32_t)e->x0); put32(hd + 36, (uint32_t)e->z0); put32(hd + 40, (uint32_t)e->shift);
    put32(hd + 44, (uint32_t)e->w); put32(hd + 48, (uint32_t)e->h); put32(hd + 52, (uint32_t)e->sea_level);
    put32(hd + 56, (uint32_t)e->s_top_wm2); put32(hd + 60, (uint32_t)e->t_surf_ck); put32(hd + 64, (uint32_t)e->gravity_q8);
    put32(hd + 68, (uint32_t)e->organic_ppm);
    for (int g = 0; g < BIO_ATM_N; ++g) put32(hd + 72 + 4 * g, (uint32_t)e->atm_ppm[g]);
    put32(hd + 100, vent_q10); put32(hd + 104, bio_env_vents(e)); put32(hd + 108, (uint32_t)(n * 16));
    put32(hd + 120, bio_env_crc(e));
    put32(hd + 124, bio_file_crc32(0, hd, 124));
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f) return -1;
    uint8_t* buf = (uint8_t*)malloc(n * 2);
    int ok = buf != NULL && fwrite(hd, 1, ENV_HDR, f) == ENV_HDR;
    uint32_t fc = bio_file_crc32(0, hd, ENV_HDR);
    Arr a[13];
    env_arrays(e, a);
    for (int k = 0; ok && k < 13; ++k) {
        if (a[k].bytes == 1) memcpy(buf, a[k].p, n);
        else for (size_t i = 0; i < n; ++i) { const uint16_t v = ((const uint16_t*)a[k].p)[i]; buf[2 * i] = (uint8_t)v; buf[2 * i + 1] = (uint8_t)(v >> 8); }
        ok = fwrite(buf, (size_t)a[k].bytes, n, f) == n;
        fc = bio_file_crc32(fc, buf, n * (size_t)a[k].bytes);
    }
    free(buf);
    if (fclose(f)) ok = 0;
    if (ok) { remove(path); ok = rename(tmp, path) == 0; }
    if (!ok) { remove(tmp); return -1; }
    if (file_crc) *file_crc = fc;
    return 0;
}

int bio_env_load(const char* path, BioEnv* e, uint32_t* vent_q10, uint32_t* file_crc) {
    memset(e, 0, sizeof *e);
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    uint8_t hd[ENV_HDR];
    if (fread(hd, 1, ENV_HDR, f) != ENV_HDR || memcmp(hd, ENV_MAGIC, 8) || get32(hd + 8) != ENV_VERSION ||
        get32(hd + 12) != ENV_HDR || get32(hd + 124) != bio_file_crc32(0, hd, 124)) { fclose(f); return -2; }
    const int32_t w = (int32_t)get32(hd + 44), h = (int32_t)get32(hd + 48), shift = (int32_t)get32(hd + 40);
    if (w <= 0 || h <= 0 || w > 8192 || h > 8192 || shift < 2 || shift > 8 || get32(hd + 108) != (uint32_t)w * (uint32_t)h * 16u) {
        fclose(f); return -2;
    }
    if (bio_env_alloc(e, w, h)) { fclose(f); return -1; }
    e->shift = shift;
    e->planet_hash = get64(hd + 16); e->seed = get32(hd + 24); e->island = (int32_t)get32(hd + 28);
    e->x0 = (int32_t)get32(hd + 32); e->z0 = (int32_t)get32(hd + 36); e->sea_level = (int32_t)get32(hd + 52);
    e->s_top_wm2 = (int32_t)get32(hd + 56); e->t_surf_ck = (int32_t)get32(hd + 60); e->gravity_q8 = (int32_t)get32(hd + 64);
    e->organic_ppm = (int32_t)get32(hd + 68);
    for (int g = 0; g < BIO_ATM_N; ++g) e->atm_ppm[g] = (int32_t)get32(hd + 72 + 4 * g);
    if (vent_q10) *vent_q10 = get32(hd + 100);
    const size_t n = (size_t)w * h;
    uint8_t* buf = (uint8_t*)malloc(n * 2);
    int rc = buf ? 0 : -1;
    uint32_t fc = bio_file_crc32(0, hd, ENV_HDR);
    Arr a[13];
    env_arrays(e, a);
    for (int k = 0; !rc && k < 13; ++k) {
        if (fread(buf, (size_t)a[k].bytes, n, f) != n) { rc = -2; break; }
        fc = bio_file_crc32(fc, buf, n * (size_t)a[k].bytes);
        if (a[k].bytes == 1) memcpy(a[k].p, buf, n);
        else for (size_t i = 0; i < n; ++i) ((uint16_t*)a[k].p)[i] = (uint16_t)(buf[2 * i] | (buf[2 * i + 1] << 8));
    }
    if (!rc && fgetc(f) != EOF) rc = -2;
    fclose(f);
    free(buf);
    if (!rc) {
        e->crc = bio_env_crc(e);
        if (e->crc != get32(hd + 120)) rc = -3;
    }
    if (rc) { bio_env_free(e); return rc; }
    if (file_crc) *file_crc = fc;
    return 0;
}

/* ---- picture ------------------------------------------------------------------------- */
int bio_env_png(const char* path, const BioEnv* e) {
    const int32_t W = e->w, H = e->h, sea = e->sea_level;
    uint32_t* px = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)W * H);
    if (!px) return -1;
    for (int32_t j = 0; j < H; ++j)
        for (int32_t i = 0; i < W; ++i) {
            const int64_t c = (int64_t)j * W + i;
            const int32_t d = e->depth[c];
            double r, g, b;
            if (e->wet[0][c] >= 255 && d >= 1) {                   /* always under water */
                const double t = iclamp(d, 1, 120) / 120.0, s = sqrt(t);
                r = 70 - 60 * s; g = 160 - 125 * s; b = 205 - 120 * s;
            } else if (e->wet[2][c] > 0) {                         /* tidal */
                r = 205; g = 190; b = 140;
            } else {                                               /* land, hill-shaded */
                const double u = iclamp(e->elev[c] - sea, 0, 220) / 220.0;
                const double soil = e->f_soil[c] / 255.0;
                r = 120 + 90 * u - 20 * soil; g = 135 + 60 * u + 10 * soil; b = 80 + 110 * u - 20 * soil;
                const int32_t l = e->elev[(int64_t)j * W + (i > 0 ? i - 1 : i)], rr = e->elev[(int64_t)j * W + (i < W - 1 ? i + 1 : i)];
                const int32_t u0 = e->elev[(int64_t)(j > 0 ? j - 1 : j) * W + i], d0 = e->elev[(int64_t)(j < H - 1 ? j + 1 : j) * W + i];
                double sh = 1.0 + 0.04 * (double)((l - rr) + (u0 - d0));
                sh = sh < 0.55 ? 0.55 : (sh > 1.35 ? 1.35 : sh);
                r *= sh; g *= sh; b *= sh;
            }
            px[c] = 0xFF000000u | ((uint32_t)iclamp((int32_t)r, 0, 255) << 16) | ((uint32_t)iclamp((int32_t)g, 0, 255) << 8) |
                    (uint32_t)iclamp((int32_t)b, 0, 255);
        }
    for (int32_t j = 0; j < H; ++j)                                 /* vents: red 5 x 5 marks */
        for (int32_t i = 0; i < W; ++i) {
            if (!e->vent[(int64_t)j * W + i]) continue;
            for (int32_t b = -2; b <= 2; ++b)
                for (int32_t a = -2; a <= 2; ++a) {
                    const int32_t x = i + a, y = j + b;
                    if (x >= 0 && y >= 0 && x < W && y < H) px[(int64_t)y * W + x] = 0xFFFF2010u;
                }
        }
    const int rc = write_png(path, px, W, H);
    free(px);
    return rc;
}

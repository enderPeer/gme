/* bio_test.c - life-test suite (spec 8.5): T1-T9 and T12, plus a synthetic
 * environment builder (ocean shelf, vents, land with a coast). */
#include "bio_internal.h"
#include "../platform.h"
#include "../biomap.h"
#include "../fauna.h"
#include "../spectrum_file.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bio_kernels.inc"

static int g_verbose = 1;
static int64_t env_i64(const char* name, int64_t def);
static int residual_zero(BioSim* s, const char* tag);
#define LOG(...) do { if (g_verbose) { printf(__VA_ARGS__); fflush(stdout); } } while (0)

/* ---- synthetic island ------------------------------------------------------------
 * West: shelf from `deep` m down to the coast at x = coast_x; east: land rising
 * 1 m per `slope` cells. Vents at a few deep cells. */
typedef struct SynthOpts {
    int32_t w, h, deep, coast_x, slope, vents, vent_strength;
    int32_t flat_depth;      /* > 0: whole grid is water of this depth (no land) */
} SynthOpts;

static int synth_env(BioEnv* e, const SynthOpts* o) {
    memset(e, 0, sizeof *e);
    if (bio_env_alloc(e, o->w, o->h)) return -1;
    e->shift = 4; e->x0 = 0; e->z0 = 0; e->sea_level = -35;
    e->t_surf_ck = 30400; e->s_top_wm2 = 1701; e->gravity_q8 = 154; e->organic_ppm = 3800;
    e->atm_ppm[0] = 505000; e->atm_ppm[1] = 386000; e->atm_ppm[2] = 104000; e->atm_ppm[3] = 5000;
    e->seed = 2; e->planet_hash = 0x5EEDB10C0FFEEull; e->island = 5;
    for (int32_t z = 0; z < o->h; ++z)
        for (int32_t x = 0; x < o->w; ++x) {
            const int32_t c = z * o->w + x;
            int32_t d;
            if (o->flat_depth > 0) d = o->flat_depth;
            else if (x < o->coast_x) d = 1 + (o->deep - 1) * (o->coast_x - x) / o->coast_x;
            else d = -((x - o->coast_x) / o->slope);
            /* a little relief along z */
            if (o->flat_depth <= 0 && ((bio_hash(x / 4, z / 4, 7, 99) & 7u) == 0) && d > 3) d -= 2;
            e->depth[c] = (int16_t)d;
            e->elev[c] = (int16_t)(e->sea_level - d);
            for (int t = 0; t < 3; ++t) e->wet[t][c] = (uint8_t)bio_clampi(128 + 128 * (d + t - 1), 0, 255);
            const uint32_t h = bio_hash(x, z, 11, 99);
            e->f_soil[c] = (uint8_t)(d <= 0 ? 160 + (h & 63) : 20);
            e->f_ore[c] = (uint8_t)((h >> 8) & 63);
            e->f_sulf[c] = (uint8_t)((h >> 16) & 31);
            e->f_gran[c] = (uint8_t)((h >> 24) & 127);
            e->flow_dir[c] = d <= 0 && x > 0 ? 4 : 255;          /* land drains west */
            e->coast[c] = (uint16_t)(d <= 0 ? (x >= o->coast_x ? x - o->coast_x + 1 : 1) : 0);
            e->env_bin[c] = (uint8_t)((d <= 0 ? 32 : 0) + bio_clampi(d < 0 ? -d / 4 : d / 16, 0, 3) * 4 + (h & 3));
        }
    /* vents: deep cells, deterministic positions */
    int placed = 0;
    for (int k = 0; k < 100000 && placed < o->vents; ++k) {
        const uint32_t h = bio_hash(k, 0, 13, 77);
        const int32_t c = (int32_t)(h % (uint32_t)(o->w * o->h));
        if (e->depth[c] > 15 && !e->vent[c]) { e->vent[c] = (uint8_t)o->vent_strength; placed++; }
    }
    e->crc = bio_env_crc(e);
    return 0;
}

/* GME_BIOFEAT=n (environment) overrides the features of every test run, e.g. 0
 * to run the whole suite as the first release */
static BioConfig test_cfg(int device, uint32_t cap) {
    BioConfig c;
    bio_config_default(&c);
    c.device = device;
    c.cap = cap;
    c.duty = 1.0;
    c.salt = 7;
    c.features = (uint32_t)env_i64("GME_BIOFEAT", (int64_t)c.features) & BIO_F_ALL;
    return c;
}

static void print_stats(const char* tag, BioSim* s) {
    BioStats st;
    if (bio_sim_stats(s, &st)) { LOG("  %s: stats failed: %s\n", tag, bio_error()); return; }
    LOG("  %s step %llu live %llu C[mar %lld coast %lld land %lld] flux METH %lld ANOX %lld OXY %lld FERM %lld SULF %lld AERO %lld\n",
        tag, (unsigned long long)st.step, (unsigned long long)st.live, (long long)st.carbon_marine,
        (long long)st.carbon_coastal, (long long)st.carbon_land, (long long)st.path_flux[0], (long long)st.path_flux[1],
        (long long)st.path_flux[2], (long long)st.path_flux[3], (long long)st.path_flux[4], (long long)st.path_flux[5]);
    LOG("     resid C %lld N %lld S %lld Fe %lld | genes METH %u ANOX %u OXY %u FERM %u PIG %u HOLD %u | births %llu deaths %llu\n",
        (long long)(st.total[0] - st.ledger[0]), (long long)(st.total[1] - st.ledger[1]),
        (long long)(st.total[2] - st.ledger[2]), (long long)(st.total[3] - st.ledger[3]),
        st.mean_gene[0], st.mean_gene[1], st.mean_gene[2], st.mean_gene[3], st.mean_gene[17], st.mean_gene[10],
        (unsigned long long)st.births, (unsigned long long)st.deaths);
}

/* ---- T1: lint ------------------------------------------------------------------- */
static int t1_lint(void) {
    const char* src = BIO_KERNELS_SRC;
    int bad = 0;
    const size_t n = strlen(src);
    for (size_t i = 0; i < n;) {
        if ((src[i] >= 'a' && src[i] <= 'z') || (src[i] >= 'A' && src[i] <= 'Z') || src[i] == '_') {
            size_t j = i;
            while (j < n && ((src[j] >= 'a' && src[j] <= 'z') || (src[j] >= 'A' && src[j] <= 'Z') || (src[j] >= '0' && src[j] <= '9') || src[j] == '_')) ++j;
            if ((j - i == 5 && !strncmp(src + i, "float", 5)) || (j - i == 6 && !strncmp(src + i, "double", 6))) {
                ++bad;
                if (bad < 5) LOG("  T1: real-number token at offset %zu\n", i);
            }
            i = j;
        } else ++i;
    }
    LOG("T1 lint (bio_types.h + bio_core.h + bio_kernels.cu, %zu bytes): %s\n", n, bad ? "FAIL" : "pass");
    return bad ? 1 : 0;
}

/* ---- T2: CPU == GPU ----------------------------------------------------------------- */
static int g_legacy;     /* 1: the run uses no feature (T14: the first release, bit for bit) */
static uint64_t run_hash(int device, uint32_t steps, uint64_t* mid_hash) {
    BioEnv e;
    SynthOpts o = { 64, 64, 40, 48, 4, 6, 220, 0 };
    if (synth_env(&e, &o)) return 0;
    BioConfig c = test_cfg(device, 65536);
    if (g_legacy) c.features = 0;
    BioSim* s = bio_sim_create(&e, &c);
    uint64_t h = 0;
    if (!s) { LOG("  create failed: %s\n", bio_error()); bio_env_free(&e); return 0; }
    if (bio_sim_seed(s, 2000)) { LOG("  seed failed: %s\n", bio_error()); goto out; }
    if (mid_hash) {
        if (bio_sim_run(s, steps / 2)) { LOG("  run failed: %s\n", bio_error()); goto out; }
        *mid_hash = bio_sim_hash(s);
        if (bio_sim_run(s, steps - steps / 2)) { LOG("  run failed: %s\n", bio_error()); goto out; }
    } else if (bio_sim_run(s, steps)) { LOG("  run failed: %s\n", bio_error()); goto out; }
    h = bio_sim_hash(s);
    print_stats(device < 0 ? "cpu" : "gpu", s);
out:
    bio_sim_destroy(s);
    bio_env_free(&e);
    return h;
}

static int t2_cpu_gpu(int device) {
    if (device < 0) { LOG("T2 CPU==GPU: skipped (CPU only)\n"); return 0; }
    uint64_t mc = 0, mg = 0;
    const double t0 = time_now();
    const uint64_t hg = run_hash(device, 500, &mg);
    const double t1 = time_now();
    const uint64_t hc = run_hash(-1, 500, &mc);
    const double t2 = time_now();
    const int ok = hg && hg == hc && mg == mc;
    LOG("T2 CPU==GPU (64^2, 2k agents, 500 steps): gpu %016llx cpu %016llx (step 250: %016llx / %016llx) gpu %.2fs cpu %.2fs: %s\n",
        (unsigned long long)hg, (unsigned long long)hc, (unsigned long long)mg, (unsigned long long)mc, t1 - t0, t2 - t1,
        ok ? "pass" : "FAIL");
    return ok ? 0 : 1;
}

/* ---- T3: conservation in a closed box --------------------------------------------- */
static int t3_conservation(int device) {
    BioEnv e;
    SynthOpts o = { 64, 64, 40, 48, 4, 6, 220, 0 };
    if (synth_env(&e, &o)) return 1;
    BioConfig c = test_cfg(device, 65536);
    c.closed_box = 1;
    BioSim* s = bio_sim_create(&e, &c);
    int fail = 0;
    if (!s || bio_sim_seed(s, 2000)) { LOG("T3: setup failed: %s\n", bio_error()); fail = 1; goto out; }
    BioStats st0;
    bio_sim_stats(s, &st0);
    for (int k = 0; k < 10 && !fail; ++k) {
        if (bio_sim_run(s, 1000)) { LOG("T3: run failed: %s\n", bio_error()); fail = 1; break; }
        BioStats st;
        bio_sim_stats(s, &st);
        for (int q = 0; q < 4; ++q)
            if (st.total[q] != st0.total[q] || st.total[q] != st.ledger[q]) {
                LOG("  T3 step %llu element %d: total %lld start %lld ledger %lld\n", (unsigned long long)st.step, q,
                    (long long)st.total[q], (long long)st0.total[q], (long long)st.ledger[q]);
                fail = 1;
            }
        if (k == 9) {
            print_stats("T3", s);
            LOG("  T3 totals C %lld N %lld S %lld Fe %lld (live %llu)\n", (long long)st.total[0], (long long)st.total[1],
                (long long)st.total[2], (long long)st.total[3], (unsigned long long)st.live);
        }
    }
    LOG("T3 conservation (closed box, 10k steps, %s): %s\n", device < 0 ? "cpu" : "gpu", fail ? "FAIL" : "pass");
out:
    bio_sim_destroy(s);
    bio_env_free(&e);
    return fail;
}

/* ---- T4: resume ------------------------------------------------------------------------ */
static int t4_resume(int device) {
    BioEnv e;
    SynthOpts o = { 64, 64, 40, 48, 4, 6, 220, 0 };
    if (synth_env(&e, &o)) return 1;
    BioConfig c = test_cfg(device, 65536);
    char path[512];
    snprintf(path, sizeof path, "bio_t4_%d.ckpt", device < 0 ? 99 : device);
    int fail = 1;
    uint64_t h_straight = 0, h_resumed = 0;
    BioSim* a = bio_sim_create(&e, &c);
    BioSim* b = NULL;
    if (!a || bio_sim_seed(a, 2000) || bio_sim_run(a, 500)) { LOG("T4: setup failed: %s\n", bio_error()); goto out; }
    const uint64_t h500 = bio_sim_hash(a);
    if (bio_sim_save(a, path)) { LOG("T4: save failed: %s\n", bio_error()); goto out; }
    if (bio_sim_run(a, 500)) goto out;
    h_straight = bio_sim_hash(a);
    b = bio_sim_load(path, &e, &c);
    if (!b) { LOG("T4: load failed: %s\n", bio_error()); goto out; }
    const uint64_t h500b = bio_sim_hash(b);
    if (bio_sim_run(b, 500)) goto out;
    h_resumed = bio_sim_hash(b);
    /* a wrong planet hash and a changed env must be refused */
    BioEnv e2 = e;
    e2.planet_hash ^= 1;
    BioSim* bad = bio_sim_load(path, &e2, &c);
    const int refused_hash = bad == NULL;
    bio_sim_destroy(bad);
    e.vent[0] ^= 1;
    bad = bio_sim_load(path, &e, &c);
    const int refused_env = bad == NULL;
    bio_sim_destroy(bad);
    e.vent[0] ^= 1;
    fail = !(h_straight && h_straight == h_resumed && h500 == h500b && refused_hash && refused_env);
    LOG("T4 resume (500 + ckpt + 500 vs 1000): straight %016llx resumed %016llx, at save %016llx/%016llx, refuses wrong planet %d env %d: %s\n",
        (unsigned long long)h_straight, (unsigned long long)h_resumed, (unsigned long long)h500, (unsigned long long)h500b,
        refused_hash, refused_env, fail ? "FAIL" : "pass");
out:
    remove(path);
    bio_sim_destroy(a);
    bio_sim_destroy(b);
    bio_env_free(&e);
    return fail;
}

/* ---- helpers for the ecological tests -------------------------------------------- */
/* flat water grid; depth from a callback-free table (row-major), no vents */
static int custom_env(BioEnv* e, int32_t w, int32_t h, const int16_t* depth) {
    memset(e, 0, sizeof *e);
    if (bio_env_alloc(e, w, h)) return -1;
    e->shift = 4; e->sea_level = -35; e->t_surf_ck = 30400; e->s_top_wm2 = 1701; e->gravity_q8 = 154;
    e->organic_ppm = 3800; e->seed = 2; e->planet_hash = 0x5EEDB10C0FFEEull; e->island = 9;
    e->atm_ppm[0] = 505000; e->atm_ppm[1] = 386000; e->atm_ppm[2] = 104000; e->atm_ppm[3] = 5000;
    for (int32_t c = 0; c < w * h; ++c) {
        const int32_t d = depth[c];
        e->depth[c] = (int16_t)d;
        e->elev[c] = (int16_t)(e->sea_level - d);
        for (int t = 0; t < 3; ++t) e->wet[t][c] = (uint8_t)bio_clampi(128 + 128 * (d + t - 1), 0, 255);
        e->flow_dir[c] = 255;
        e->coast[c] = (uint16_t)(d <= 0 ? 1 : 0);
    }
    e->crc = bio_env_crc(e);
    return 0;
}

/* a plain genome: everything zero except what the caller sets; T_opt = 304 K, wide tolerance */
static void plain_genome(BioGenome* g) {
    memset(g, 0, sizeof *g);
    g->body[BG_SIZE] = 20; g->body[BG_REPRO_E] = 64; g->body[BG_OFFS] = 128;
    g->body[BG_MUT] = 64; g->body[BG_THINK] = 124; g->body[BG_TOPT] = 124; g->body[BG_TTOL] = 255;
    g->body[BG_LIFESPAN] = 255;
    g->b_o[3] = 16;
}

/* n agents of genome g on cell c (or spread over cells [c0, c1) when c1 > c0);
 * g_place_s: structural carbon they start with (0 unless a test sets it) */
static int32_t g_place_s;
static const SpectrumFile* g_spectrum;      /* run_hash_eco, crown_gain: light in bands when set */
static int g_cg_pig = 255, g_cg_hue = 0;    /* crown_gain: pigment and hue of the plant */
static int g_cg_dim = 0;                    /* crown_gain: the star dimmed to 1 / g_cg_dim (0: as it is) */
static double crown_gain(int device, uint32_t features, int32_t C, int32_t S, int crown, int aspect, int leafang, int porosity);
static int place(BioSim* s, const BioGenome* g, uint32_t n, int32_t c0, int32_t c1, int32_t C, int32_t E) {
    BioWorld* w = &s->w;
    BioAgentRec* a = (BioAgentRec*)calloc(n, sizeof(BioAgentRec));
    if (!a) return -1;
    BioWorld hw = *w;
    hw.T = &s->tables;
    const int32_t cq = 1 << (w->shift + 8);
    for (uint32_t k = 0; k < n; ++k) {
        const int32_t c = c1 > c0 ? c0 + (int32_t)(k % (uint32_t)(c1 - c0)) : c0;
        const uint32_t h = bio_hash((int32_t)k, 5, 5, 5);
        a[k].x = (c % w->w) * cq + (int32_t)(h & (uint32_t)(cq - 1));
        a[k].z = (c / w->w) * cq + (int32_t)((h >> 16) & (uint32_t)(cq - 1));
        a[k].C = C; a[k].S = g_place_s; a[k].E = E; a[k].N = (C + 5) / 6 + 1; a[k].P = C / 106 + 1; a[k].W = 256;
        a[k].flags = BA_ALIVE | (s->env->depth[c] >= 1 ? BA_WET : BA_LAND);
        a[k].elast = E;
        a[k].out = bio_out_bias(&hw, g);
        a[k].g = *g;
    }
    const int r = bio_append_agents(s, a, n, 1);
    free(a);
    return r;
}

static int cmp_u64(const void* a, const void* b) {
    const uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}
static uint32_t distinct_genomes(BioSim* s) {
    BioAgentRec* a = NULL; uint32_t n = 0;
    if (bio_download_agents(s, &a, &n) || !n) { free(a); return 0; }
    uint64_t* h = (uint64_t*)malloc(8 * (size_t)n);
    if (!h) { free(a); return 0; }
    for (uint32_t i = 0; i < n; ++i) h[i] = bio_fnv64(BIO_FNV_INIT, &a[i].g, sizeof(BioGenome));
    qsort(h, n, 8, cmp_u64);
    uint32_t u = 1;
    for (uint32_t i = 1; i < n; ++i) u += h[i] != h[i - 1];
    free(h); free(a);
    return u;
}
/* carbon of phototroph agents (ANOX or OXY >= 32), and mean of one gene */
static int64_t photo_carbon(BioSim* s, int gene, double* mean_gene) {
    BioAgentRec* a = NULL; uint32_t n = 0;
    if (bio_download_agents(s, &a, &n)) return -1;
    int64_t c = 0; double gs = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (a[i].g.body[BG_E_ANOX] >= 32 || a[i].g.body[BG_E_OXY] >= 32) c += (int64_t)a[i].C + a[i].S;
        gs += a[i].g.body[gene];
    }
    if (mean_gene) *mean_gene = n ? gs / n : 0.0;
    free(a);
    return c;
}

/* ---- T5: chemostat ------------------------------------------------------------------ */
static int t5_chemostat(int device) {
    const int16_t depth[1] = { 10 };
    BioEnv e;
    if (custom_env(&e, 1, 1, depth)) return 1;
    e.vent[0] = 255;
    e.crc = bio_env_crc(&e);
    BioConfig c = test_cfg(device, 4096);
    BioSim* s = bio_sim_create(&e, &c);
    int fail = 1;
    if (!s) { LOG("T5: %s\n", bio_error()); goto out; }
    BioParams* P = &s->w.P;
    P->kp_water_q16 = 0; P->hazard_q32 = 0; P->kinpath_q32 = 0; P->mut_off = 1; P->serp_h2_q8 = 0;
    P->vent_h2s = 0; P->vent_fe2 = 0;
    BioGenome g;
    plain_genome(&g);
    g.body[BG_E_METH] = 160; g.body[BG_HOLDFAST] = 255; g.body[BG_SIZE] = 70;   /* m_r = 128 mmol */
    g.body[BG_TOPT] = 184;                                                       /* 319 K at the vent */
    if (place(s, &g, 4, 0, 0, 128, 64)) { LOG("T5: %s\n", bio_error()); goto out; }
    double sumc = 0, sumn = 0; int nsamp = 0;
    for (int day = 0; day < 400; ++day) {
        if (bio_sim_run(s, 24)) { LOG("T5: %s\n", bio_error()); goto out; }
        if (day >= 300) {
            BioStats st; bio_sim_stats(s, &st);
            sumc += (double)(st.carbon_marine + st.carbon_coastal + st.carbon_land); sumn += (double)st.live; ++nsamp;
        }
    }
    {
        const double cbar = sumc / nsamp, nbar = sumn / nsamp;
        /* analytic: energy in = H2 flux / 4 eu per step; cost per mmol C per step =
         * maintenance x Q10(T) x (1 + n/16) + enzyme cost */
        BioWorld hw = s->w;
        hw.T = &s->tables; hw.G = &s->Gh; hw.elev = e.elev; hw.depth = e.depth; hw.vent = e.vent;
        const int32_t T = bio_temp(&hw, 0, 0);
        const double q10 = (double)s->tables.q10_q16[bio_clampi(64 + (T - 30300) / 100, 0, 127)] / 65536.0;
        const double m = (double)P->maint_q16 / 65536.0 * q10 * (1.0 + nbar / 16.0) + (double)P->enz_q16 / 65536.0 * (160.0 * 160.0) / 65025.0;
        const double flux = (double)P->vent_h2 * 255 / 255;
        const double K = flux / 4.0 * ((double)P->yield_q16[BP_METH] / 65536.0) / m;
        const double err = (cbar - K) / K;
        fail = err > 0.2 || err < -0.2;
        LOG("T5 chemostat (1 vent cell, H2 %.0f mmol/h): plateau C %.0f mmol (%.1f agents) vs analytic K %.0f: %+.1f%%: %s\n",
            flux, cbar, nbar, K, 100.0 * err, fail ? "FAIL" : "pass");
    }
out:
    bio_sim_destroy(s);
    bio_env_free(&e);
    return fail;
}

/* ---- T6: light profile ------------------------------------------------------------------ */
static int t6_light(int device) {
    enum { ND = 20 };
    int16_t depth[ND];
    for (int k = 0; k < ND; ++k) depth[k] = (int16_t)(1 + k);
    BioEnv e;
    if (custom_env(&e, ND, 1, depth)) return 1;
    BioConfig c = test_cfg(device, 4096);
    BioSim* s = bio_sim_create(&e, &c);
    int fail = 1;
    BioAgentRec* a0 = NULL; BioAgentRec* a1 = NULL; uint32_t n0 = 0, n1 = 0;
    if (!s) { LOG("T6: %s\n", bio_error()); goto out; }
    BioParams* P = &s->w.P;
    P->grow_off = 1; P->maint_q16 = 0; P->enz_q16 = 0; P->trait_q16 = 0; P->uv_q16 = 0; P->hazard_q32 = 0;
    P->kinpath_q32 = 0; P->mut_off = 1; P->o2tox_q16 = 0; P->struct_q16 = 0; P->e_cap_mul = 1000;
    BioGenome g;
    plain_genome(&g);
    g.body[BG_E_ANOX] = 255; g.body[BG_PIGMENT] = 255; g.body[BG_HOLDFAST] = 255;   /* benthic: z = depth */
    if (place(s, &g, ND * 4, 0, ND, 2000, 0)) { LOG("T6: %s\n", bio_error()); goto out; }
    if (bio_download_agents(s, &a0, &n0) || bio_sim_run(s, 24) || bio_download_agents(s, &a1, &n1) || n0 != n1) {
        LOG("T6: %s\n", bio_error()); goto out;
    }
    double gain[ND]; int cntd[ND];
    memset(gain, 0, sizeof gain); memset(cntd, 0, sizeof cntd);
    for (uint32_t i = 0; i < n1; ++i) {
        const int32_t cx = a1[i].x >> (8 + 4);
        gain[cx] += (double)(a1[i].E - a0[i].E); cntd[cx]++;
    }
    double worst = 0;
    /* from 2 m: the 1 m cell falls dry at low tide (tides are +-1 m), and a leaf in the air is not under water */
    LOG("T6 light profile (benthic ANOX phototrophs, 1 day): depth: gain/gain(2 m) vs exp(-0.06 z)/exp(-0.12):");
    for (int k = 1; k < ND; ++k) {
        const double obs = (gain[k] / cntd[k]) / (gain[1] / cntd[1]);
        const double idx_k = (double)(((int64_t)P->kw_q16 * (depth[k] * 256)) >> 18);
        const double idx_0 = (double)(((int64_t)P->kw_q16 * (depth[1] * 256)) >> 18);
        const double exp_ratio = (double)s->tables.expn_q16[(int)idx_k] / (double)s->tables.expn_q16[(int)idx_0];
        const double err = obs / exp_ratio - 1.0;
        if (err > worst || -err > worst) worst = err < 0 ? -err : err;
        if (k % 3 == 1 || k == ND - 1) LOG(" %dm %.3f/%.3f", depth[k], obs, exp_ratio);
    }
    fail = worst > 0.05;
    LOG("; worst deviation %.1f%%: %s\n", 100.0 * worst, fail ? "FAIL" : "pass");
out:
    free(a0); free(a1);
    bio_sim_destroy(s);
    bio_env_free(&e);
    return fail;
}

/* ---- T7: null controls ---------------------------------------------------------------------- */
static int t7_null(int device) {
    int fails = 0;
    SynthOpts o = { 64, 64, 40, 48, 4, 6, 220, 0 };
    /* (a) no light: phototroph carbon reaches 0 within 2 sim-years */
    {
        BioEnv e; synth_env(&e, &o);
        BioConfig c = test_cfg(device, 262144);
        BioSim* s = bio_sim_create(&e, &c);
        BioGenome g; plain_genome(&g);
        g.body[BG_E_ANOX] = 160; g.body[BG_PIGMENT] = 200; g.body[BG_HOLDFAST] = 200;
        int64_t c0 = -1, c1 = -1;
        if (s) {
            s->w.P.light_off = 1;
            s->w.P.mut_off = 1;          /* no escape into chemotrophy */
            place(s, &g, 4000, 0, 64 * 40, 32, 64);
            c0 = photo_carbon(s, 0, NULL);
            int64_t when = -1;
            for (int d = 0; d < 730 && when < 0; d += 10) {
                if (bio_sim_run(s, 240)) break;
                if (photo_carbon(s, 0, NULL) == 0) when = d + 10;
            }
            c1 = photo_carbon(s, 0, NULL);
            const int ok = c1 == 0;
            fails += !ok;
            LOG("T7a no light: phototroph C %lld -> %lld (zero by day %lld): %s\n", (long long)c0, (long long)c1, (long long)when, ok ? "pass" : "FAIL");
        }
        bio_sim_destroy(s); bio_env_free(&e);
    }
    /* (b) vents off, no weathering, no evolution: the protocells die out */
    {
        SynthOpts o2 = o; o2.vents = 0;
        BioEnv e; synth_env(&e, &o2);
        BioConfig c = test_cfg(device, 262144);
        BioSim* s = bio_sim_create(&e, &c);
        if (s) {
            s->w.P.weath_fe_q16 = s->w.P.weath_s_q16 = s->w.P.weath_p_q16 = s->w.P.weath_doc_q16 = 0;
            s->w.P.serp_h2_q8 = 0; s->w.P.mut_off = 1;
            /* no abiotic organic stock either: DOC removed (and taken out of the ledger) */
            int64_t* f = (int64_t*)malloc(sizeof(int64_t) * 64 * 64 * BF_N);
            if (f && !bio_fields_get(s, f)) {
                int64_t removed = 0;
                for (int k = 0; k < 64 * 64; ++k) { removed += f[BF_DOC * 64 * 64 + k]; f[BF_DOC * 64 * 64 + k] = 0; }
                bio_fields_put(s, f);
                s->init_total[0] -= removed;
            }
            free(f);
            bio_sim_seed(s, 2000);
            int64_t day = -1;
            for (int d = 0; d < 730 && day < 0; d += 10) {
                if (bio_sim_run(s, 240)) break;
                if (s->n_live == 0) day = d + 10;
            }
            const int ok = s->n_live == 0;
            fails += !ok;
            LOG("T7b vents off, no weathering: live %u after %s day %lld: %s\n", s->n_live, ok ? "extinction on" : "730 days, last", (long long)(day < 0 ? 730 : day), ok ? "pass" : "FAIL");
        }
        bio_sim_destroy(s); bio_env_free(&e);
    }
    /* (c) mutation off: the number of distinct genomes does not rise (control: mutation on) */
    {
        uint32_t u[2] = { 0, 0 };
        for (int k = 0; k < 2; ++k) {
            BioEnv e; synth_env(&e, &o);
            BioConfig c = test_cfg(device, 262144);
            BioSim* s = bio_sim_create(&e, &c);
            if (s) {
                s->w.P.mut_off = k == 0;
                bio_sim_seed(s, 2000);
                bio_sim_run(s, 2000);
                u[k] = distinct_genomes(s);
            }
            bio_sim_destroy(s); bio_env_free(&e);
        }
        const int ok = u[0] == 1 && u[1] > 1;
        fails += !ok;
        LOG("T7c mutation off: distinct genomes after 2000 steps %u (control with mutation %u): %s\n", u[0], u[1], ok ? "pass" : "FAIL");
    }
    /* (d) atm_scale -> 1: the box is 10^4 x larger, no G5 (O2 > 1%) */
    {
        BioEnv e; synth_env(&e, &o);
        BioConfig c = test_cfg(device, 262144);
        c.atm_scale_ppm = 1000000;
        BioSim* s = bio_sim_create(&e, &c);
        if (s) {
            bio_sim_seed(s, 2000);
            const int years = (int)env_i64("GME_BIOT7_YEARS", 10);
            bio_sim_run(s, (uint64_t)years * 8760);
            BioStats st; bio_sim_stats(s, &st);
            int64_t tot = 0;
            for (int g2 = 0; g2 < BIO_ATM_N; ++g2) tot += st.atm[g2];
            const int ok = !(st.gates & (1u << BIO_G5_OXIDATION));
            fails += !ok;
            LOG("T7d atm_scale 1: after %d sim-years O2 = %.3g of the box, gates %02x, no G5: %s\n", years,
                tot ? (double)st.atm[BIO_ATM_O2] / (double)tot : 0.0, st.gates, ok ? "pass" : "FAIL");
        }
        bio_sim_destroy(s); bio_env_free(&e);
    }
    return fails;
}

/* ---- T8: selection on pigment ------------------------------------------------------------------ */
static int t8_selection(int device) {
    enum { W8 = 32 };
    int16_t depth[W8 * W8];
    for (int k = 0; k < W8 * W8; ++k) depth[k] = 4;
    double mean[2] = { 0, 0 };
    for (int k = 0; k < 2; ++k) {
        BioEnv e; custom_env(&e, W8, W8, depth);
        BioConfig c = test_cfg(device, 262144);
        BioSim* s = bio_sim_create(&e, &c);
        if (s) {
            s->w.P.mut_off = k == 1;
            BioGenome g; plain_genome(&g);
            g.body[BG_E_ANOX] = 160; g.body[BG_PIGMENT] = 40; g.body[BG_HOLDFAST] = 200; g.body[BG_MUT] = 0;   /* 1/64 per byte */
            place(s, &g, 2000, 0, W8 * W8, 8, 32);
            bio_sim_run(s, 5000);
            photo_carbon(s, BG_PIGMENT, &mean[k]);
        }
        bio_sim_destroy(s); bio_env_free(&e);
    }
    const int ok = mean[0] > mean[1] + 10.0;
    LOG("T8 selection (shallow shelf, 5000 steps, start pigment 40): mean pigment %.1f with mutation vs %.1f control: %s\n",
        mean[0], mean[1], ok ? "pass" : "FAIL");
    return !ok;
}

/* ---- T9: learning ----------------------------------------------------------------------------------- */
/* motile fermenters chase a DOC patch that jumps every 2 days; intake = FERM flux per day */
static double t9_run(int device, int mode) {
    enum { W9 = 32 };
    int16_t depth[W9 * W9];
    for (int k = 0; k < W9 * W9; ++k) depth[k] = 5;
    BioEnv e; custom_env(&e, W9, W9, depth);
    BioConfig c = test_cfg(device, 65536);
    c.salt = (uint32_t)env_i64("GME_BIOT9_SALT", 7);
    BioSim* s = bio_sim_create(&e, &c);
    double intake = 0;
    int64_t* f = (int64_t*)malloc(sizeof(int64_t) * W9 * W9 * BF_N);
    if (!s || !f) goto out;
    BioParams* P = &s->w.P;
    P->mut_off = 1; P->grow_off = 1; P->hazard_q32 = 0; P->kinpath_q32 = 0; P->kp_water_q16 = 0;
    P->d_ww = 3277;                         /* slower diffusion keeps a local gradient */
    P->learn_off = mode == 1 ? 1 : (mode == 3 ? 2 : 0); P->brain_off = mode == 2;
    if (getenv("GME_BIOT9_EXPL")) P->explore_q7 = atoi(getenv("GME_BIOT9_EXPL"));
    if (getenv("GME_BIOT9_SHIFT")) P->learn_shift = atoi(getenv("GME_BIOT9_SHIFT"));
    BioGenome g; plain_genome(&g);
    g.body[BG_E_FERM] = 255; g.body[BG_MOTOR] = 160; g.body[BG_SENSE] = 255; g.body[BG_THINK] = 8;   /* think every 3 steps */
    g.body[BG_NEURONS] = 0xFF; g.body[BG_PLASTIC] = 0x03; g.body[BG_ETA] = 64; g.body[BG_DECAY] = 8;
    /* small random weights plus a weak innate chemotaxis: gradient x -> h0 -> move x,
     * gradient z -> h1 -> move z (an evolved genome would carry some such wiring) */
    for (int j = 0; j < BIO_H; ++j)
        for (int k = 0; k < BIO_IN; ++k) g.w_ih[j][k] = (int8_t)((int32_t)(bio_hash(j, k, 91, 3) % 13u) - 6);
    for (int o2 = 0; o2 < BIO_OUT; ++o2)
        for (int j = 0; j < BIO_H; ++j) g.w_ho[o2][j] = (int8_t)((int32_t)(bio_hash(o2, j, 92, 3) % 13u) - 6);
    if (!getenv("GME_BIOT9_RANDOM")) { g.w_ih[0][7] = 48; g.w_ih[1][8] = 48; g.w_ho[0][0] = 32; g.w_ho[1][1] = 32; }
    else for (int j = 0; j < BIO_H; ++j) { for (int k = 0; k < BIO_IN; ++k) g.w_ih[j][k] *= 3; for (int o2 = 0; o2 < BIO_OUT; ++o2) g.w_ho[o2][j] *= 3; }
    if (getenv("GME_BIOT9_ETA")) g.body[BG_ETA] = (uint8_t)atoi(getenv("GME_BIOT9_ETA"));
    g.b_o[3] = -16;
    place(s, &g, 64, 0, W9 * W9, 400, 2000);
    /* the patch is the only food: background DOC removed (and taken out of the ledger) */
    if (!bio_fields_get(s, f)) {
        int64_t removed = 0;
        for (int k = 0; k < W9 * W9; ++k) { removed += f[BF_DOC * W9 * W9 + k]; f[BF_DOC * W9 * W9 + k] = 0; }
        bio_fields_put(s, f);
        s->init_total[0] -= removed;
    }
    int64_t f0 = 0;
    for (int day = 0; day < 40; ++day) {
        if (day % 2 == 0) {     /* move the food: a 2 x 2 block of DOC at a new place (scarce) */
            bio_fields_get(s, f);
            const uint32_t h = bio_hash(day, 0, 93, 1);
            const int32_t px = 4 + (int32_t)(h % 24u), pz = 4 + (int32_t)((h >> 8) % 24u);
            for (int dz = 0; dz < 2; ++dz)
                for (int dx = 0; dx < 2; ++dx) f[BF_DOC * W9 * W9 + (pz + dz) * W9 + px + dx] += 25000;
            bio_fields_put(s, f);
            /* the added DOC enters the ledger as an import so conservation stays exact */
            bio_g_get(s); s->Gh.acc[BACC_IMPORT_C] += 4 * 25000; bio_g_put(s);
        }
        if (day == 10) { bio_g_get(s); f0 = s->Gh.acc[BACC_PATH + BP_FERM]; }
        if (bio_sim_run(s, 24)) break;
    }
    bio_g_get(s);
    intake = (double)(s->Gh.acc[BACC_PATH + BP_FERM] - f0) / 30.0;
out:
    free(f);
    bio_sim_destroy(s);
    bio_env_free(&e);
    return intake;
}
static int t9_learning(int device) {
    /* x: the control that explores exactly as a learner does but never changes a
     * weight. Learning has to beat it too, or the gain would be random wandering. */
    const double a = t9_run(device, 0), b = t9_run(device, 1), z = t9_run(device, 2), x = t9_run(device, 3);
    const int ok_learn = a > 1.10 * b, ok_brain = z < b && z < a, ok_ctrl = a > 1.05 * x;
    const int ok = ok_learn && ok_brain && ok_ctrl;
    LOG("T9 learning (moving DOC patch, FERM intake per day, days 10-40): eta>0 %.0f, eta=0 %.0f (%+.1f%%, needs > +10%%: %s), "
        "exploring without learning %.0f (%+.1f%%, needs > +5%%: %s), brain zeroed %.0f (worse than both: %s): %s\n",
        a, b, b > 0 ? 100.0 * (a / b - 1.0) : 0.0, ok_learn ? "yes" : "no", x, x > 0 ? 100.0 * (a / x - 1.0) : 0.0,
        ok_ctrl ? "yes" : "no", z, ok_brain ? "yes" : "no", ok ? "pass" : "FAIL");
    return !ok;
}

/* ---- T2b: CPU == GPU with phototrophs, aerobes and learning foragers -------------------
 * Covers the code paths the protocell run of T2 does not reach in 500 steps:
 * canopy light, ANOX/OXY fixation split, O2 production, aerobic respiration,
 * O2 toxicity, steering, gradient sensing and Hebbian learning. */
static uint64_t run_hash_photo(int device, uint32_t steps, uint64_t* mid, BioStats* out) {
    BioEnv e;
    SynthOpts o = { 64, 64, 40, 40, 4, 6, 220, 0 };
    if (synth_env(&e, &o)) return 0;
    BioConfig c = test_cfg(device, 131072);
    if (g_legacy) c.features = 0;
    BioSim* s = bio_sim_create(&e, &c);
    uint64_t h = 0;
    if (!s || bio_sim_seed(s, 1000)) goto out;
    {
        BioGenome g; plain_genome(&g);
        g.body[BG_E_ANOX] = 120; g.body[BG_E_OXY] = 120; g.body[BG_E_AERO] = 60; g.body[BG_E_FERM] = 40;
        g.body[BG_PIGMENT] = 200; g.body[BG_HOLDFAST] = 160; g.body[BG_HUE] = 90; g.body[BG_STRUCT] = 40;
        g.body[BG_ROOT] = 60; g.body[BG_CUTICLE] = 60; g.body[BG_O2TOL] = 200;
        if (place(s, &g, 1500, 0, 64 * 64, 64, 64)) goto out;
        BioGenome m; plain_genome(&m);
        m.body[BG_E_FERM] = 120; m.body[BG_E_AERO] = 160; m.body[BG_MOTOR] = 128; m.body[BG_SENSE] = 200;
        m.body[BG_NEURONS] = 0xFF; m.body[BG_PLASTIC] = 0x03; m.body[BG_ETA] = 32; m.body[BG_DECAY] = 8; m.body[BG_THINK] = 8;
        for (int j = 0; j < BIO_H; ++j) for (int k = 0; k < BIO_IN; ++k) m.w_ih[j][k] = (int8_t)((int32_t)(bio_hash(j, k, 7, 7) % 41u) - 20);
        for (int q = 0; q < BIO_OUT; ++q) for (int j = 0; j < BIO_H; ++j) m.w_ho[q][j] = (int8_t)((int32_t)(bio_hash(q, j, 8, 7) % 41u) - 20);
        if (place(s, &m, 300, 0, 64 * 40, 200, 400)) goto out;
    }
    if (bio_sim_run(s, steps / 2)) goto out;
    *mid = bio_sim_hash(s);
    if (bio_sim_run(s, steps - steps / 2)) goto out;
    h = bio_sim_hash(s);
    bio_sim_stats(s, out);
out:
    if (!h) LOG("  T2b run failed: %s\n", bio_error());
    bio_sim_destroy(s);
    bio_env_free(&e);
    return h;
}
static int t2b_cpu_gpu_photo(int device) {
    if (device < 0) { LOG("T2b CPU==GPU (phototrophs): skipped (CPU only)\n"); return 0; }
    uint64_t mg = 0, mc = 0;
    BioStats sg, sc;
    memset(&sg, 0, sizeof sg); memset(&sc, 0, sizeof sc);
    const uint64_t hg = run_hash_photo(device, 1500, &mg, &sg);
    const uint64_t hc = run_hash_photo(-1, 1500, &mc, &sc);
    const int ok = hg && hg == hc && mg == mc;
    LOG("T2b CPU==GPU (phototrophs + aerobes + learning foragers, 1500 steps): gpu %016llx cpu %016llx (step 750 %016llx / %016llx); "
        "live %llu, C fixed ANOX %lld OXY %lld, AERO %lld, mean |dW| %.2f: %s\n",
        (unsigned long long)hg, (unsigned long long)hc, (unsigned long long)mg, (unsigned long long)mc, (unsigned long long)sg.live,
        (long long)sg.path_flux[BP_ANOX], (long long)sg.path_flux[BP_OXY], (long long)sg.path_flux[BP_AERO], sg.mean_abs_dw,
        ok ? "pass" : "FAIL");
    return ok ? 0 : 1;
}

/* ---- T14: all features off = the first release, bit for bit ---------------------------
 * The hashes below were printed by the binary built from commit 1222849 (the
 * release before the features existed) for the T2, T2b and T4 runs. A run
 * with BioConfig.features = 0 must still produce them, on any device. */
static int t14_legacy(int device) {
    static const uint64_t H500 = 0x2e597966d641f4cfull, H250 = 0x4df4979f7d1ab15cull, H1000 = 0xcd49c6f754c8f80aull,
                          HB1500 = 0xf6976ded4b30df53ull, HB750 = 0x1e47eb8e0b8ae0ffull;
    const int verbose = g_verbose;
    g_legacy = 1;
    g_verbose = 0;
    uint64_t m1 = 0, m2 = 0;
    BioStats st;
    memset(&st, 0, sizeof st);
    const uint64_t h1 = run_hash(device, 500, &m1);
    const uint64_t h2 = run_hash(device, 1000, NULL);
    const uint64_t h3 = run_hash_photo(device, 1500, &m2, &st);
    g_legacy = 0;
    g_verbose = verbose;
    const int ok = h1 == H500 && m1 == H250 && h2 == H1000 && h3 == HB1500 && m2 == HB750 && st.features == 0;
    LOG("T14 features off = first release (%s): protocells 250/500/1000 steps %016llx %016llx %016llx, phototrophs 750/1500 %016llx %016llx: %s\n",
        device < 0 ? "cpu" : "gpu", (unsigned long long)m1, (unsigned long long)h1, (unsigned long long)h2,
        (unsigned long long)m2, (unsigned long long)h3, ok ? "pass" : "FAIL (differs from the recorded hashes)");
    return !ok;
}

/* ---- a small ecosystem that exercises every feature ------------------------------------
 * Two sexual plant lineages with crowns (compatible mate tags), grazers and
 * hunters that learn, on the synthetic island. */
static void random_brain(BioGenome* g, int salt, int scale) {
    for (int j = 0; j < BIO_H; ++j)
        for (int k = 0; k < BIO_IN; ++k) g->w_ih[j][k] = (int8_t)((int32_t)(bio_hash(j, k, 7, (uint32_t)salt) % (uint32_t)(2 * scale + 1)) - scale);
    for (int q = 0; q < BIO_OUT; ++q)
        for (int j = 0; j < BIO_H; ++j) g->w_ho[q][j] = (int8_t)((int32_t)(bio_hash(q, j, 8, (uint32_t)salt) % (uint32_t)(2 * scale + 1)) - scale);
}
static int seed_ecosystem(BioSim* s, int32_t w, int32_t coast_x) {
    BioGenome g; plain_genome(&g);
    g.body[BG_E_ANOX] = 120; g.body[BG_E_OXY] = 120; g.body[BG_E_AERO] = 60; g.body[BG_E_FERM] = 40;
    g.body[BG_PIGMENT] = 200; g.body[BG_HOLDFAST] = 160; g.body[BG_HUE] = 90; g.body[BG_STRUCT] = 40;
    g.body[BG_ROOT] = 60; g.body[BG_CUTICLE] = 60; g.body[BG_O2TOL] = 200;
    g.body[BG_SEX] = 200; g.mate_tag = 0x00F0;
    g.body[BG_CROWN] = 120; g.body[BG_ASPECT] = 90; g.body[BG_LEAFANG] = 60; g.body[BG_POROSITY] = 100; g.body[BG_QWIDTH] = 80;
    if (place(s, &g, 1500, 0, w * w, 64, 64)) return -1;
    g.body[BG_HUE] = 30; g.mate_tag = 0x00F1;                 /* one bit away: compatible */
    g.body[BG_CROWN] = 220; g.body[BG_ASPECT] = 200; g.body[BG_LEAFANG] = 230; g.body[BG_STRUCT] = 90;
    g_place_s = 40000;                                        /* 2 m and up: crowns from the first step */
    const int pl = place(s, &g, 500, 0, w * w, 64, 64);
    g_place_s = 0;
    if (pl) return -1;
    BioGenome m; plain_genome(&m);                            /* grazers */
    m.body[BG_E_FERM] = 200; m.body[BG_E_AERO] = 160; m.body[BG_MOTOR] = 128; m.body[BG_MOUTH] = 200; m.body[BG_SENSE] = 200;
    m.body[BG_NEURONS] = 0xFF; m.body[BG_PLASTIC] = 0x03; m.body[BG_ETA] = 32; m.body[BG_DECAY] = 8; m.body[BG_THINK] = 8;
    m.body[BG_SEX] = 255; m.mate_tag = 0x1234; m.body[BG_O2TOL] = 255; m.body[BG_CUTICLE] = 120;
    random_brain(&m, 7, 20);
    m.b_o[2] = 64;                                            /* attack */
    if (place(s, &m, 300, 0, w * coast_x, 200, 400)) return -1;
    BioGenome h = m;                                          /* hunters: bigger, faster, armoured */
    h.body[BG_MOTOR] = 220; h.body[BG_MOUTH] = 255; h.body[BG_SIZE] = 80; h.body[BG_ARMOR] = 60; h.mate_tag = 0x1235;
    random_brain(&h, 11, 20);
    h.b_o[2] = 96;
    return place(s, &h, 100, 0, w * coast_x, 2000, 4000);
}

/* ---- T2c: CPU == GPU with every feature active ------------------------------------------- */
static uint64_t run_hash_eco(int device, uint32_t steps, uint64_t* mid, BioStats* out, int closed, int* conserved) {
    BioEnv e;
    SynthOpts o = { 64, 64, 40, 40, 4, 6, 220, 0 };
    if (synth_env(&e, &o)) return 0;
    BioConfig c = test_cfg(device, 131072);
    c.closed_box = closed;
    c.spectrum = g_spectrum;
    BioSim* s = bio_sim_create(&e, &c);
    uint64_t h = 0;
    if (!s || bio_sim_seed(s, 1000) || seed_ecosystem(s, 64, 40)) goto out;
    BioStats st0;
    bio_sim_stats(s, &st0);
    if (bio_sim_run(s, steps / 2)) goto out;
    *mid = bio_sim_hash(s);
    if (bio_sim_run(s, steps - steps / 2)) goto out;
    h = bio_sim_hash(s);
    bio_sim_stats(s, out);
    if (conserved) {
        *conserved = 1;
        for (int q = 0; q < 4; ++q)
            if (out->total[q] != out->ledger[q] || (closed && out->total[q] != st0.total[q])) {
                LOG("  element %d: total %lld start %lld ledger %lld\n", q, (long long)out->total[q], (long long)st0.total[q], (long long)out->ledger[q]);
                *conserved = 0;
            }
    }
out:
    if (!h) LOG("  ecosystem run failed: %s\n", bio_error());
    bio_sim_destroy(s);
    bio_env_free(&e);
    return h;
}
static int t2c_cpu_gpu_eco(int device) {
    if (device < 0) { LOG("T2c CPU==GPU (all features): skipped (CPU only)\n"); return 0; }
    uint64_t mg = 0, mc = 0;
    BioStats sg, sc;
    memset(&sg, 0, sizeof sg); memset(&sc, 0, sizeof sc);
    int cg = 0, cc = 0;
    const uint64_t hg = run_hash_eco(device, 1500, &mg, &sg, 0, &cg);
    const uint64_t hc = run_hash_eco(-1, 1500, &mc, &sc, 0, &cc);
    const int covered = sg.kills > 0 && sg.bites > 0 && sg.sexual_births > 0 && sg.pred_carbon > 0;
    const int ok = hg && hg == hc && mg == mc && cg && cc && covered;
    LOG("T2c CPU==GPU (plants with crowns + grazers + hunters, sex, 1500 steps): gpu %016llx cpu %016llx (step 750 %016llx / %016llx); "
        "live %llu, kills %lld, bites %lld, carbon eaten %lld, births with two parents %lld, consumers %llu (mean trophic level %.2f), "
        "ledger %s: %s%s\n",
        (unsigned long long)hg, (unsigned long long)hc, (unsigned long long)mg, (unsigned long long)mc, (unsigned long long)sg.live,
        (long long)sg.kills, (long long)sg.bites, (long long)sg.pred_carbon, (long long)sg.sexual_births,
        (unsigned long long)sg.consumers, sg.mean_troph, cg && cc ? "exact" : "OFF", ok ? "pass" : "FAIL",
        covered ? "" : " (a feature did nothing: the test does not cover it)");
    return ok ? 0 : 1;
}

/* ---- T3b: conservation in a closed box with predation and sex ---------------------------- */
static int t3b_conservation_eco(int device) {
    uint64_t mid = 0;
    BioStats st;
    memset(&st, 0, sizeof st);
    int cons = 0;
    const uint64_t h = run_hash_eco(device, 4000, &mid, &st, 1, &cons);
    const int ok = h && cons && st.kills > 0 && st.bites > 0;
    LOG("T3b conservation with animals (closed box, 4000 steps, %s): C N S Fe totals exactly constant, kills %lld, bites %lld, live %llu: %s\n",
        device < 0 ? "cpu" : "gpu", (long long)st.kills, (long long)st.bites, (unsigned long long)st.live, ok ? "pass" : "FAIL");
    return !ok;
}

/* ---- T4b: resume with every feature ------------------------------------------------------------
 * A checkpoint must resume exactly with animals and sex in the world too. The
 * save is at a step that is not a multiple of 64 and soon after a compaction,
 * so the running island has holes in its agent arrays that the loaded one has
 * not: anything decided by an agent's absolute position would show here. */
static int t4b_resume_eco(int device) {
    BioEnv e;
    SynthOpts o = { 64, 64, 40, 40, 4, 6, 220, 0 };
    if (synth_env(&e, &o)) return 1;
    BioConfig c = test_cfg(device, 131072);
    char path[512];
    snprintf(path, sizeof path, "bio_t4b_%d.ckpt", device < 0 ? 99 : device);
    int fail = 1;
    uint64_t hs = 0, hr = 0, h0 = 0, h0b = 0;
    BioStats st;
    memset(&st, 0, sizeof st);
    BioSim* a = bio_sim_create(&e, &c);
    BioSim* b = NULL;
    if (!a || bio_sim_seed(a, 1000) || seed_ecosystem(a, 64, 40) || bio_sim_run(a, 517)) { LOG("T4b: setup failed: %s\n", bio_error()); goto out; }
    h0 = bio_sim_hash(a);
    if (bio_sim_save(a, path) || bio_sim_run(a, 150)) { LOG("T4b: %s\n", bio_error()); goto out; }
    hs = bio_sim_hash(a);
    bio_sim_stats(a, &st);
    b = bio_sim_load(path, &e, &c);
    if (!b) { LOG("T4b: load failed: %s\n", bio_error()); goto out; }
    h0b = bio_sim_hash(b);
    if (bio_sim_run(b, 150)) goto out;
    hr = bio_sim_hash(b);
    fail = !(hs && hs == hr && h0 == h0b && st.kills > 0 && st.bites > 0 && st.sexual_births > 0);
    LOG("T4b resume with animals and sex (517 + ckpt + 150 vs 667 steps): straight %016llx resumed %016llx, at save %016llx/%016llx; "
        "kills %lld, bites %lld, births with two parents %lld: %s\n", (unsigned long long)hs, (unsigned long long)hr, (unsigned long long)h0,
        (unsigned long long)h0b, (long long)st.kills, (long long)st.bites, (long long)st.sexual_births, fail ? "FAIL" : "pass");
out:
    remove(path);
    bio_sim_destroy(a);
    bio_sim_destroy(b);
    bio_env_free(&e);
    return fail;
}

/* ---- T10: predator and prey ---------------------------------------------------------------
 * Plankton phototrophs and hunters with fixed genomes on an oxygenated shelf.
 * The hunters' biomass must follow the prey's: the cross-correlation of the two
 * series peaks at a positive lag (spec 8.5). */
static void biomass_split(BioSim* s, double* prey, double* pred) {
    BioAgentRec* a = NULL; uint32_t n = 0;
    *prey = 0; *pred = 0;
    if (bio_download_agents(s, &a, &n)) return;
    for (uint32_t i = 0; i < n; ++i) {
        const double b = (double)a[i].C + a[i].S + a[i].gut;
        if (a[i].g.body[BG_MOUTH] > 0) *pred += b; else *prey += b;
    }
    free(a);
}
static int t10_predator_prey(int device) {
    enum { W10 = 32, NS = 400 };
    int16_t depth[W10 * W10];
    for (int k = 0; k < W10 * W10; ++k) depth[k] = (int16_t)env_i64("GME_BIOT10_DEPTH", 4);
    BioEnv e;
    if (custom_env(&e, W10, W10, depth)) return 1;
    /* an oxidised atmosphere: the hunters respire their prey aerobically */
    memset(e.atm_ppm, 0, sizeof e.atm_ppm);
    e.atm_ppm[BIO_ATM_NH3] = 50000; e.atm_ppm[BIO_ATM_CO2] = 10000; e.atm_ppm[BIO_ATM_O2] = 200000; e.atm_ppm[BIO_ATM_N2] = 740000;
    BioConfig c = test_cfg(device, 262144);
    c.features = BIO_F_ANIMALS;
    BioSim* s = bio_sim_create(&e, &c);
    int fail = 1;
    static double prey[NS], pred[NS];
    if (!s) { LOG("T10: %s\n", bio_error()); goto out;  }
    s->w.P.mut_off = 1;
    s->w.P.weath_doc_q16 = 0;
    {   /* no abiotic organic stock: the hunters live on prey (removed from the ledger too) */
        int64_t* f = (int64_t*)malloc(sizeof(int64_t) * W10 * W10 * BF_N);
        if (f && !bio_fields_get(s, f)) {
            int64_t removed = 0;
            for (int k = 0; k < W10 * W10; ++k) { removed += f[BF_DOC * W10 * W10 + k]; f[BF_DOC * W10 * W10 + k] = 0; }
            bio_fields_put(s, f);
            s->init_total[0] -= removed;
        }
        free(f);
    }
    BioGenome g; plain_genome(&g);                            /* plankton */
    g.body[BG_E_OXY] = 200; g.body[BG_PIGMENT] = (uint8_t)env_i64("GME_BIOT10_PIG", 140); g.body[BG_BUOY] = 255; g.body[BG_O2TOL] = 255;
    g.body[BG_DISPERSE] = 64;
    g.body[BG_SIZE] = (uint8_t)env_i64("GME_BIOT10_PSIZE", 20);
    BioGenome h; plain_genome(&h);                            /* hunters: follow the crowd, always attack */
    h.body[BG_E_AERO] = 255; h.body[BG_MOTOR] = 160; h.body[BG_MOUTH] = 255; h.body[BG_SENSE] = 255;
    h.body[BG_SIZE] = (uint8_t)env_i64("GME_BIOT10_HSIZE", 60);
    h.body[BG_O2TOL] = 255; h.body[BG_NEURONS] = 0xFF; h.body[BG_THINK] = 8; h.body[BG_DISPERSE] = 32;
    h.w_ih[0][13] = 96; h.w_ih[1][14] = 96; h.w_ho[0][0] = 96; h.w_ho[1][1] = 96;
    h.b_o[2] = 64;
    if (place(s, &g, (uint32_t)env_i64("GME_BIOT10_NPREY", 6000), 0, W10 * W10, (int32_t)env_i64("GME_BIOT10_PC", 6), 32) ||
        place(s, &h, (uint32_t)env_i64("GME_BIOT10_NH", 60), 0, W10 * W10, (int32_t)env_i64("GME_BIOT10_HC", 100), 400)) { LOG("T10: %s\n", bio_error()); goto out; }
    const int every = (int)env_i64("GME_BIOT10_EVERY", 12);   /* steps per sample */
    int n = 0;
    for (; n < NS; ++n) {
        if (bio_sim_run(s, (uint64_t)every)) { LOG("T10: %s\n", bio_error()); goto out; }
        biomass_split(s, &prey[n], &pred[n]);
        if (pred[n] <= 0 || prey[n] <= 0) { ++n; break; }
    }
    {
        /* cross-correlation of the two series (first quarter dropped): r(lag) = corr(prey(t), pred(t + lag)) */
        const int n0 = n / 4, m = n - n0, maxlag = m / 4;
        double best = -2, r0 = 0;
        int bestlag = 0;
        for (int lag = -maxlag; lag <= maxlag; ++lag) {
            double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
            int cnt = 0;
            for (int t = n0; t < n; ++t) {
                const int u = t + lag;
                if (u < n0 || u >= n) continue;
                sx += prey[t]; sy += pred[u]; sxx += prey[t] * prey[t]; syy += pred[u] * pred[u]; sxy += prey[t] * pred[u];
                ++cnt;
            }
            const double vx = sxx - sx * sx / cnt, vy = syy - sy * sy / cnt;
            const double r = vx > 0 && vy > 0 ? (sxy - sx * sy / cnt) / sqrt(vx * vy) : 0;
            if (lag == 0) r0 = r;
            if (r > best) { best = r; bestlag = lag; }
        }
        BioStats st; bio_sim_stats(s, &st);
        double pmin = 1e300, pmax = 0, qmin = 1e300, qmax = 0;
        for (int t = n0; t < n; ++t) { if (prey[t] < pmin) pmin = prey[t]; if (prey[t] > pmax) pmax = prey[t]; if (pred[t] < qmin) qmin = pred[t]; if (pred[t] > qmax) qmax = pred[t]; }
        fail = !(n == NS && bestlag > 0 && best > 0.3 && st.kills > 0 && residual_zero(s, "T10"));
        LOG("T10 predator and prey (%d samples of %d h): prey C %.3g..%.3g, hunters C %.3g..%.3g mmol, kills %lld; cross-correlation peaks at "
            "lag %+d samples (r %.2f; r at lag 0 %.2f): %s\n", n, every, pmin, pmax, qmin, qmax, (long long)st.kills, bestlag, best, r0,
            fail ? "FAIL (the hunters must follow the prey: peak at a lag > 0 with r > 0.3, both alive, ledger exact)" : "pass");
        if (env_i64("GME_BIOT10_DUMP", 0)) {
            for (int t = 0; t < n; t += (int)env_i64("GME_BIOT10_DUMP", 4)) LOG("  %4d prey %.4g pred %.4g\n", t, prey[t], pred[t]);
            LOG("  live %llu consumers %llu troph %.2f bites %lld kills %lld eaten %lld\n", (unsigned long long)st.live,
                (unsigned long long)st.consumers, st.mean_troph, (long long)st.bites, (long long)st.kills, (long long)st.pred_carbon);
        }
    }
out:
    bio_sim_destroy(s);
    bio_env_free(&e);
    return fail;
}

/* ---- T15: sexual reproduction ---------------------------------------------------------------
 * Two clones that differ at two marker bytes in different 16-byte blocks of
 * the genome, no mutation. Recombinants (one marker of each) can only come from
 * a cross: many with compatible mate tags, none with tags more than 3 bits
 * apart, none with the feature off. */
static int sex_run(int device, uint32_t features, uint16_t tag_b, uint32_t* recomb, uint32_t* total, int64_t* sexb) {
    enum { W15 = 16 };
    int16_t depth[W15 * W15];
    for (int k = 0; k < W15 * W15; ++k) depth[k] = 4;
    BioEnv e;
    if (custom_env(&e, W15, W15, depth)) return -1;
    BioConfig c = test_cfg(device, 131072);
    c.features = features;
    BioSim* s = bio_sim_create(&e, &c);
    int rc = -1;
    if (!s) goto out;
    s->w.P.mut_off = 1;
    BioGenome a; plain_genome(&a);
    a.body[BG_E_ANOX] = 160; a.body[BG_PIGMENT] = 160; a.body[BG_HOLDFAST] = 200; a.body[BG_SEX] = 200;
    a.body[BG_NEUTRAL0] = 10; a.pad[4] = 1; a.mate_tag = 0x0F0F;
    BioGenome b = a;
    b.body[BG_NEUTRAL0] = 200; b.pad[4] = 2; b.mate_tag = tag_b;
    if (place(s, &a, 500, 0, W15 * W15, 8, 32) || place(s, &b, 500, 0, W15 * W15, 8, 32) || bio_sim_run(s, 1500)) goto out;
    BioAgentRec* r = NULL; uint32_t n = 0;
    if (bio_download_agents(s, &r, &n)) goto out;
    *recomb = 0; *total = n;
    for (uint32_t i = 0; i < n; ++i) {
        const int m1 = r[i].g.body[BG_NEUTRAL0] == 10, m2 = r[i].g.pad[4] == 1;
        if (m1 != m2) (*recomb)++;
    }
    free(r);
    BioStats st; bio_sim_stats(s, &st);
    *sexb = st.sexual_births;
    rc = 0;
out:
    bio_sim_destroy(s);
    bio_env_free(&e);
    return rc;
}
static int t15_sex(int device) {
    uint32_t r1 = 0, n1 = 0, r2 = 0, n2 = 0, r3 = 0, n3 = 0;
    int64_t b1 = 0, b2 = 0, b3 = 0;
    const int e1 = sex_run(device, BIO_F_SEX, 0x0F0E, &r1, &n1, &b1);      /* tags 1 bit apart */
    const int e2 = sex_run(device, BIO_F_SEX, 0xF0F0, &r2, &n2, &b2);      /* 16 bits apart: two species */
    const int e3 = sex_run(device, 0, 0x0F0E, &r3, &n3, &b3);              /* feature off */
    const int ok = !e1 && !e2 && !e3 && r1 * 10 > n1 && b1 > 0 && r2 == 0 && b2 > 0 && r3 == 0 && b3 == 0;
    LOG("T15 sex (two clones, markers in two genome blocks, no mutation, 1500 steps): compatible tags %u recombinants of %u (%lld births "
        "with two parents); tags 16 bits apart %u of %u (%lld, within each clone only); feature off %u of %u (%lld): %s\n",
        r1, n1, (long long)b1, r2, n2, (long long)b2, r3, n3, (long long)b3, ok ? "pass" : "FAIL");
    return !ok;
}

/* ---- light in bands: a synthetic GMESPEC1 ---------------------------------------------------
 * Shaped like the home planet's export (band shares of the stellar flux at
 * noon, water attenuation per band, photons per joule, the pigment shapes of
 * render-upgrade-spec 5.1), so the suite needs no file. Hazier bins lose red. */
static void synth_spectrum(SpectrumFile* f, const BioEnv* e) {
    static const double noon[SPECF_NB] = { 33.31, 46.07, 56.60, 63.70, 67.28, 67.98, 66.60, 63.85, 195.84 };   /* W/m^2 of 1485 */
    static const double kw[SPECF_NB] = { 0.1103, 0.0733, 0.0635, 0.0814, 0.1299, 0.3118, 0.4491, 0.7376, 2.2129 };
    static const double phot[SPECF_NB] = { 3.371, 3.744, 4.119, 4.493, 4.869, 5.244, 5.620, 5.996, 6.822 };
    static const double centre[SPECF_NB] = { 402.5, 447.5, 492.5, 537.5, 582.5, 627.5, 672.5, 717.5, 820.0 };
    memset(f, 0, sizeof *f);
    f->planet_hash = e->planet_hash;
    for (int b = 0; b <= SPECF_NB; ++b) f->band_edge_nm[b] = b < SPECF_NB ? 380 + 45 * b : 900;
    f->s_top_wm2 = e->s_top_wm2; f->t_surf_ck = e->t_surf_ck;
    for (int k = 0; k < SPECF_NHAZE; ++k)
        for (int hr = 0; hr < SPECF_NHOUR; ++hr) {
            const double sn = bio_sin24(hr) > 0 ? bio_sin24(hr) / 4096.0 : 0.0;
            for (int b = 0; b < SPECF_NB; ++b) {
                const double red = b >= 4 ? 1.0 - 0.09 * k * (b - 3) / 5.0 : 1.0;      /* CH4 and haze take the red end */
                f->surf_q16[k][hr][b] = (int32_t)lround(65536.0 * noon[b] / 1485.0 * sn * (1.0 - 0.03 * k) * (red > 0 ? red : 0));
                f->dir_q16[k][hr][b] = f->surf_q16[k][hr][b] / 2;
            }
        }
    for (int b = 0; b < SPECF_NB; ++b) {
        for (int d = 0; d < SPECF_NDEPTH; ++d) f->water_T_q16[b][d] = (int32_t)lround(65536.0 * exp(-kw[b] * d));
        f->kw_q16[b] = (int32_t)lround(65536.0 * kw[b]);
        f->phot_q16[b] = (int32_t)lround(65536.0 * phot[b]);
        f->ev_q16[b] = (int32_t)lround(65536.0 * 1239.84 / centre[b]);
        f->uphill_q16[b][0] = b < 7 ? 65536 : (b == 7 ? 1586 : 0);
        f->uphill_q16[b][1] = 65536;
    }
    for (int h = 0; h < SPECF_NHUE; ++h) {
        const double lq = 383.0 + 34.13 * (h ? h : 8);
        for (int b = 0; b < SPECF_NB; ++b) {
            /* band mean of 1.2 G(435, 30) + G(lq, 25) + 0.15 s(lambda) */
            double sum = 0;
            const double lo = f->band_edge_nm[b], hi = f->band_edge_nm[b + 1];
            int n = 0;
            for (double l = lo + 2.5; l < hi; l += 5.0, ++n) {
                const double s0 = l < 690 ? 1.0 : (l < 740 ? (740 - l) / 50.0 : 0.0);
                sum += 1.2 * exp(-0.5 * (l - 435) * (l - 435) / 900.0) + exp(-0.5 * (l - lq) * (l - lq) / 625.0) + 0.15 * s0;
            }
            f->eps_q12[h][b] = (int32_t)lround(4096.0 * sum / n);
        }
    }
}

/* ---- T13: light in bands ----------------------------------------------------------------------
 * (a) calibration: a reference leaf at the surface gains over a day what the
 *     scalar model gives it (+-5%);
 * (b) water: the gain of a benthic phototroph at 2..17 m follows the bands'
 *     own attenuation and the pigment's absorption (+-5% of the double-precision sum);
 * (c) selection: two clones that differ only in pigment colour (Q band at 656 nm,
 *     red, and at 485 nm, blue-green) start in equal numbers on a shelf with a
 *     shallow and a deep half. Water takes the red first: the mean Q band of the
 *     deep half must end >= 15 nm bluer than the shallow half's and bluer than it
 *     started (the depth - colour relation of spec 5.2's T13); with scalar light
 *     the two clones are the same organism and nothing may separate them. */
static double day_gain(int device, const SpectrumFile* sf, const int16_t* depth, int nd, double* gain, int hue) {
    BioEnv e;
    if (custom_env(&e, nd, 1, depth)) return -1;
    BioConfig c = test_cfg(device, 4096);
    c.features = 0;
    c.spectrum = sf;
    BioSim* s = bio_sim_create(&e, &c);
    BioAgentRec* a0 = NULL; BioAgentRec* a1 = NULL; uint32_t n0 = 0, n1 = 0;
    double tot = -1;
    if (!s) { LOG("  T13: %s\n", bio_error()); goto out; }
    BioParams* P = &s->w.P;
    P->grow_off = 1; P->maint_q16 = 0; P->enz_q16 = 0; P->trait_q16 = 0; P->uv_q16 = 0; P->hazard_q32 = 0;
    P->kinpath_q32 = 0; P->mut_off = 1; P->o2tox_q16 = 0; P->struct_q16 = 0; P->e_cap_mul = 1000;
    BioGenome g; plain_genome(&g);
    g.body[BG_E_ANOX] = 255; g.body[BG_PIGMENT] = 200; g.body[BG_HUE] = (uint8_t)hue; g.body[BG_HOLDFAST] = 255;
    if (place(s, &g, (uint32_t)nd * 4, 0, nd, 2000, 0) || bio_download_agents(s, &a0, &n0) || bio_sim_run(s, 24) ||
        bio_download_agents(s, &a1, &n1) || n0 != n1) { LOG("  T13: %s\n", bio_error()); goto out; }
    tot = 0;
    for (int k = 0; k < nd; ++k) gain[k] = 0;
    for (uint32_t i = 0; i < n1; ++i) { gain[a1[i].x >> 12] += (double)(a1[i].E - a0[i].E) / 4.0; tot += (double)(a1[i].E - a0[i].E); }
out:
    free(a0); free(a1);
    bio_sim_destroy(s);
    bio_env_free(&e);
    return tot;
}
static double mean_lq(BioSim* s, int deep, int32_t w) {      /* biomass-weighted Q band of the phototrophs of one half, nm */
    BioAgentRec* a = NULL; uint32_t n = 0;
    if (bio_download_agents(s, &a, &n)) return 0;
    double sw = 0, sl = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (a[i].g.body[BG_E_ANOX] < 32 && a[i].g.body[BG_E_OXY] < 32) continue;
        const int east = (a[i].x >> 12) >= w / 2;
        if (east != deep) continue;
        const double b = (double)a[i].C + a[i].S;
        sw += b; sl += b * (383.0 + 34.13 * bio_hue_bin(a[i].g.body[BG_HUE]));
    }
    free(a);
    return sw > 0 ? sl / sw : 0;
}
static int t13_spectral(int device) {
    int fails = 0;
    BioEnv e0;
    const int16_t d1[1] = { 1 };
    custom_env(&e0, 1, 1, d1);
    static SpectrumFile sf;
    synth_spectrum(&sf, &e0);
    bio_env_free(&e0);
    /* (a) calibration against the scalar model: the reference leaf on land, and another pigment for contrast */
    {
        g_cg_pig = 200; g_cg_hue = 128;
        const double es = crown_gain(device, 0, 4000, 0, 0, 0, 0, 0);
        g_spectrum = &sf;
        const double eb = crown_gain(device, 0, 4000, 0, 0, 0, 0, 0);
        g_cg_hue = 250;                                      /* Q band in the near infrared: an oxygenic leaf cannot use it */
        const double en = crown_gain(device, 0, 4000, 0, 0, 0, 0, 0);
        g_spectrum = NULL; g_cg_pig = 255; g_cg_hue = 0;
        const int ok = es > 0 && fabs(eb / es - 1.0) <= 0.05 && en < 0.9 * eb;
        LOG("T13a light in bands, calibration: the reference leaf (pigment 200, Q band 656 nm) gains %.0f eu/day with bands vs %.0f with the "
            "scalar model (%+.1f%%, +-5%%); the same leaf with its Q band at 895 nm %.0f (%+.1f%%: light it cannot use): %s\n",
            eb, es, es > 0 ? 100.0 * (eb / es - 1.0) : 0.0, en, eb > 0 ? 100.0 * (en / eb - 1.0) : 0.0, ok ? "pass" : "FAIL");
        fails += !ok;
    }
    /* (b) water profile per band */
    {
        enum { ND = 16 };
        int16_t depth[ND];
        for (int k = 0; k < ND; ++k) depth[k] = (int16_t)(2 + k);    /* from 2 m: a 1 m cell falls partly dry at low tide */
        double g[ND];
        const double tot = day_gain(device, &sf, depth, ND, g, 128);
        double worst = 0;
        LOG("T13b light in bands, water (benthic phototrophs, hue bin 8, 1 day): depth: gain / gain(2 m) vs the band sum:");
        for (int k = 0; k < ND && tot > 0; ++k) {
            /* expected: sum over bands of light x transmittance x absorptance x photons (ANOX uses every band). The
             * hour only scales it, so the noon column is enough, at the test atmosphere's haze: CH4 38.6% of the
             * mix gives haze_q12 2747, i.e. bin 5 + 69/256 of the way to bin 6 */
            double ex[2] = { 0, 0 };
            for (int q = 0; q < 2; ++q) {
                const int z = q ? depth[k] : depth[0];
                for (int b = 0; b < SPECF_NB; ++b) {
                    const double od = 200.0 / 255.0 * 4.0 * sf.eps_q12[8][b] / 4096.0;
                    const double lt = sf.surf_q16[5][12][b] + (sf.surf_q16[6][12][b] - sf.surf_q16[5][12][b]) * 69.0 / 256.0;
                    ex[q] += lt * (sf.water_T_q16[b][z] / 65536.0) * (1.0 - pow(10.0, -od)) * (sf.phot_q16[b] / 65536.0);
                }
            }
            const double obs = g[k] / g[0], want2 = ex[1] / ex[0], err = obs / want2 - 1.0;
            if (fabs(err) > worst) worst = fabs(err);
            if (k % 3 == 0 || k == ND - 1) LOG(" %dm %.3f/%.3f", depth[k], obs, want2);
        }
        const int ok = tot > 0 && worst <= 0.05;
        LOG("; worst deviation %.1f%%: %s\n", 100.0 * worst, ok ? "pass" : "FAIL");
        fails += !ok;
    }
    /* (c) selection of the pigment's colour by depth */
    {
        enum { W13 = 32 };
        int16_t depth[W13 * W13];
        for (int k = 0; k < W13 * W13; ++k) depth[k] = (int16_t)((k % W13) < W13 / 2 ? 2 : 12);
        double lq[2][2] = { { 0, 0 }, { 0, 0 } };       /* [bands, scalar][shallow, deep] */
        uint32_t live[2] = { 0, 0 };
        for (int k = 0; k < 2; ++k) {
            BioEnv e; custom_env(&e, W13, W13, depth);
            BioConfig c = test_cfg(device, 262144);
            c.features = 0;
            c.spectrum = k == 0 ? &sf : NULL;
            BioSim* s = bio_sim_create(&e, &c);
            if (s) {
                s->w.P.mut_off = 1;
                BioGenome g; plain_genome(&g);
                g.body[BG_E_ANOX] = 160; g.body[BG_PIGMENT] = 120; g.body[BG_HOLDFAST] = 200; g.body[BG_DISPERSE] = 16;
                g.body[BG_HUE] = 128;                                    /* bin 8: Q band 656 nm */
                place(s, &g, (uint32_t)env_i64("GME_BIOT13_N", 1000), 0, W13 * W13, 8, 32);
                g.body[BG_HUE] = 40;                                     /* bin 3: Q band 485 nm */
                place(s, &g, (uint32_t)env_i64("GME_BIOT13_N", 1000), 0, W13 * W13, 8, 32);
                bio_sim_run(s, (uint64_t)env_i64("GME_BIOT13_STEPS", 5000));
                lq[k][0] = mean_lq(s, 0, W13); lq[k][1] = mean_lq(s, 1, W13);
                live[k] = s->n_live;
            } else LOG("  T13c: %s\n", bio_error());
            bio_sim_destroy(s); bio_env_free(&e);
        }
        const double start = (656.0 + 485.0) / 2;
        const int ok = lq[0][1] > 0 && lq[0][0] > 0 && lq[0][0] - lq[0][1] >= 15.0 && lq[0][1] < start - 3.0 &&
                       fabs(lq[1][0] - lq[1][1]) < 8.0;
        LOG("T13c light in bands, selection (shelf 2 m | 12 m, clones with Q band 656 and 485 nm, mean %.0f nm at the start): after 5000 steps "
            "%.0f nm shallow, %.0f nm deep (scalar light: %.0f / %.0f nm; live %u / %u): %s\n", start, lq[0][0], lq[0][1], lq[1][0], lq[1][1],
            live[0], live[1], ok ? "pass" : "FAIL (deep must end >= 15 nm bluer than shallow and bluer than the start; scalar light must not select)");
        fails += !ok;
    }
    /* (d) CPU == GPU with bands and every feature, and a resume with another light file is refused */
    {
        uint64_t hg = 0, hc = 0, mg = 0, mc = 0;
        BioStats sg, sc;
        memset(&sg, 0, sizeof sg); memset(&sc, 0, sizeof sc);
        g_spectrum = &sf;
        if (device >= 0) { hg = run_hash_eco(device, 1000, &mg, &sg, 0, NULL); hc = run_hash_eco(-1, 1000, &mc, &sc, 0, NULL); }
        else { hg = hc = run_hash_eco(-1, 1000, &mg, &sg, 0, NULL); mc = mg; }
        g_spectrum = NULL;
        BioEnv e;
        SynthOpts o = { 64, 64, 40, 40, 4, 6, 220, 0 };
        synth_env(&e, &o);
        BioConfig c = test_cfg(device, 65536);
        c.spectrum = &sf;
        BioSim* s = bio_sim_create(&e, &c);
        const char* path = "bio_t13.ckpt";
        int refused = 0, resumed = 0, needs = 0;
        if (s && !bio_sim_seed(s, 500) && !bio_sim_run(s, 100) && !bio_sim_save(s, path)) {
            const uint64_t h0 = bio_sim_hash(s);
            static SpectrumFile other;
            other = sf;
            other.kw_q16[2] += 1;
            c.spectrum = &other;
            BioSim* b = bio_sim_load(path, &e, &c);
            refused = b == NULL;
            bio_sim_destroy(b);
            c.spectrum = NULL;
            b = bio_sim_load(path, &e, &c);
            needs = b == NULL;
            bio_sim_destroy(b);
            c.spectrum = &sf;
            b = bio_sim_load(path, &e, &c);
            resumed = b && bio_sim_hash(b) == h0 && (bio_sim_features(b) & BIO_F_SPECTRAL);
            bio_sim_destroy(b);
        }
        remove(path);
        bio_sim_destroy(s); bio_env_free(&e);
        const int ok = hg && hg == hc && mg == mc && (sg.features & BIO_F_SPECTRAL) && refused && needs && resumed;
        LOG("T13d light in bands: all features, 1000 steps: %s %016llx cpu %016llx (step 500 %016llx / %016llx); resume keeps the state %d, "
            "refuses another light file %d, refuses none %d: %s\n", device < 0 ? "cpu" : "gpu", (unsigned long long)hg,
            (unsigned long long)hc, (unsigned long long)mg, (unsigned long long)mc, resumed, refused, needs, ok ? "pass" : "FAIL");
        fails += !ok;
    }
    return fails;
}

/* ---- T16: crown shapes ---------------------------------------------------------------------------
 * The crown model of BIO_FEAT_MORPH, checked through the simulation: land plants
 * of set shape, one day, energy gained.
 * (a) in light too weak to saturate a leaf, a thin crown with every shape gene
 *     at 0 (ellipsoid, 1 : 1, mid leaf angle: what every plant of a world evolved
 *     before the feature has) gains what its leaf area alone gained without the
 *     feature, and so does ground cover; in full sun both use the light less
 *     well, by about the same share (no step at the 0.5 m where a crown begins);
 * (b) a crown cannot catch more light than falls on its top: the gain per leaf
 *     area falls as the crown fills up, and the gain of a full crown stays under
 *     what its top area alone would gain as bare leaf without the feature;
 * (c) leaf angle: flat leaves beat erect ones in a thin crown, erect ones win in
 *     a dense one (their sunlit leaves are not saturated at noon);
 * (d) aspect: the same structure carried tall and narrow is higher but its crown
 *     is smaller, so it gains less when nothing shades it. */
static double crown_gain(int device, uint32_t features, int32_t C, int32_t S, int crown, int aspect, int leafang, int porosity) {
    const int16_t depth[4] = { -3, -3, -3, -3 };
    BioEnv e;
    if (custom_env(&e, 4, 1, depth)) return -1;
    BioConfig c = test_cfg(device, 4096);
    c.features = features;
    c.spectrum = g_spectrum;
    BioSim* s = bio_sim_create(&e, &c);
    BioAgentRec* a0 = NULL; BioAgentRec* a1 = NULL; uint32_t n0 = 0, n1 = 0;
    double g = -1;
    if (!s) goto out;
    BioParams* P = &s->w.P;
    P->grow_off = 1; P->maint_q16 = 0; P->enz_q16 = 0; P->trait_q16 = 0; P->uv_q16 = 0; P->hazard_q32 = 0; P->dry_q8 = 0;
    P->kinpath_q32 = 0; P->mut_off = 1; P->o2tox_q16 = 0; P->struct_q16 = 0; P->e_cap_mul = 100000;
    P->ktox_h2s = 1 << 30;                                    /* no sulfide inhibition: the test is about light */
    if (g_cg_dim > 1) s->w.s_top /= g_cg_dim;
    BioGenome gg; plain_genome(&gg);
    gg.body[BG_E_OXY] = 255; gg.body[BG_PIGMENT] = (uint8_t)g_cg_pig; gg.body[BG_HUE] = (uint8_t)g_cg_hue;
    gg.body[BG_HOLDFAST] = 255; gg.body[BG_CUTICLE] = 255; gg.body[BG_O2TOL] = 255;
    gg.body[BG_CROWN] = (uint8_t)crown; gg.body[BG_ASPECT] = (uint8_t)aspect; gg.body[BG_LEAFANG] = (uint8_t)leafang;
    gg.body[BG_POROSITY] = (uint8_t)porosity;
    g_place_s = S;
    const int pl = place(s, &gg, 1, 1, 1, C, 0);
    g_place_s = 0;
    if (pl || bio_download_agents(s, &a0, &n0) || bio_sim_run(s, 24) || bio_download_agents(s, &a1, &n1) || n0 != 1 || n1 != 1) goto out;
    g = (double)(a1[0].E - a0[0].E);
out:
    free(a0); free(a1);
    bio_sim_destroy(s);
    bio_env_free(&e);
    return g;
}
static int t16_crowns(int device) {
    /* S = 4000 x H^3: 500000 -> 5 m at aspect 1 (top area pi / 4 x 25 = 19.6 m^2), 100 -> 0.29 m (ground cover).
     * Genes: aspect 0 = 1 : 1, 160 = 3 : 1, 192 = 0.5; leaf angle 0 = the mid angle (class 8), 112 = flat (class 15),
     * 128 = erect (class 0). 4000 mmol C of leaf is 1 m^2. */
    const int32_t S = 500000, SC = 100;
    g_cg_dim = 32;                                                                          /* weak light: no leaf saturates */
    const int32_t ST = 32000000;                                                            /* a 20 m tree: 314 m^2 of top over 10 m^2 of leaf */
    const double dim_off = crown_gain(device, 0, 40000, ST, 0, 0, 0, 0), dim_on = crown_gain(device, BIO_F_MORPH, 40000, ST, 0, 0, 0, 0);
    const double dimc_off = crown_gain(device, 0, 40000, SC, 0, 0, 0, 0), dimc_on = crown_gain(device, BIO_F_MORPH, 40000, SC, 0, 0, 0, 0);
    g_cg_dim = 0;
    const double ref_off = crown_gain(device, 0, 4000, S, 0, 0, 0, 0);                     /* leaf area alone: 1 m^2 */
    const double ref_on = crown_gain(device, BIO_F_MORPH, 4000, S, 0, 0, 0, 0);            /* a thin crown, all shape genes 0 */
    const double cov_off = crown_gain(device, 0, 4000, SC, 0, 0, 0, 0), cov_on = crown_gain(device, BIO_F_MORPH, 4000, SC, 0, 0, 0, 0);
    const double thin = crown_gain(device, BIO_F_MORPH, 20000, S, 0, 0, 0, 0);             /* 5 m^2 of leaf under 19.6 m^2 of top */
    const double dense = crown_gain(device, BIO_F_MORPH, 2000000, S, 0, 0, 0, 0);          /* 480 m^2 of leaf */
    const double flat_thin = crown_gain(device, BIO_F_MORPH, 4000, S, 0, 0, 112, 0), erect_thin = crown_gain(device, BIO_F_MORPH, 4000, S, 0, 0, 128, 0);
    const double flat_dense = crown_gain(device, BIO_F_MORPH, 2000000, S, 0, 0, 112, 0), erect_dense = crown_gain(device, BIO_F_MORPH, 2000000, S, 0, 0, 128, 0);
    const double tall = crown_gain(device, BIO_F_MORPH, 2000000, S, 0, 160, 0, 0), squat = crown_gain(device, BIO_F_MORPH, 2000000, S, 0, 192, 0, 0);
    const double open = crown_gain(device, BIO_F_MORPH, 4000, S, 0, 0, 0, 255);
    const double sun_crown = ref_off > 0 ? ref_on / ref_off : 0, sun_cover = cov_off > 0 ? cov_on / cov_off : 0;
    const double top_m2 = 3.14159265 / 4.0 * 25.0;
    const int ok_a = dim_off > 0 && dimc_off > 0 && fabs(dim_on / dim_off - 1.0) <= 0.06 && fabs(dimc_on / dimc_off - 1.0) <= 0.06 &&
                     sun_crown > 0.35 && sun_crown < 0.95 && sun_cover > 0.35 && sun_cover < 0.95 && fabs(sun_crown - sun_cover) <= 0.15;
    const int ok_b = thin > 0 && dense / 2000000.0 < 0.5 * (thin / 20000.0) && dense <= 1.02 * top_m2 * ref_off;
    const int ok_c = flat_thin > erect_thin && erect_dense > flat_dense;
    const int ok_d = squat > tall && open < ref_on;
    const int ok = ok_a && ok_b && ok_c && ok_d;
    LOG("T16 crown shapes (one land plant, one day, eu gained): weak light, thin crown with every shape gene 0: %.0f vs %.0f without the "
        "feature (%+.1f%%), ground cover %.0f vs %.0f (%+.1f%%; both within 6%%); full sun: crown %.0f vs %.0f (%+.1f%%), ground cover "
        "%.0f vs %.0f (%+.1f%%; both saturate, within 15 points of each other): %s; per leaf carbon thin %.4f vs full %.4f, full crown "
        "%.0f <= its top area as bare leaf %.0f (%s); leaf angle thin crown flat %.0f > erect %.0f, full crown erect %.0f > flat %.0f "
        "(%s); full crown squat %.0f > tall narrow %.0f, open crown %.0f < closed %.0f (%s): %s\n",
        dim_on, dim_off, dim_off > 0 ? 100.0 * (dim_on / dim_off - 1.0) : 0.0, dimc_on, dimc_off, dimc_off > 0 ? 100.0 * (dimc_on / dimc_off - 1.0) : 0.0,
        ref_on, ref_off, 100.0 * (sun_crown - 1.0), cov_on, cov_off, 100.0 * (sun_cover - 1.0), ok_a ? "yes" : "no",
        thin / 20000.0, dense / 2000000.0, dense, top_m2 * ref_off, ok_b ? "yes" : "no", flat_thin, erect_thin, erect_dense, flat_dense,
        ok_c ? "yes" : "no", squat, tall, open, ref_on, ok_d ? "yes" : "no", ok ? "pass" : "FAIL");
    return !ok;
}

/* ---- T17: the island hub -----------------------------------------------------------------------
 * Two islands and a hub in temporary folders, a short "epoch" (240 steps).
 * (a) migrants travel both ways and the ledgers stay exact;
 * (b) one island starts with extra O2: the planet-wide atmosphere pulls the two
 *     boxes together (their O2 shares end closer than without the hub);
 * (c) replay: an island resumed from an older checkpoint applies the same files
 *     at the same epochs and reaches the same state, bit for bit. */
static void hub_dirs_clear(const char* root) {
    static const char* const sub[] = { "outbox", "inbox", "imported", "in", "out/0", "out/1", "out", "" };
    char p[300];
    for (size_t k = 0; k < sizeof sub / sizeof sub[0]; ++k) {
        if (sub[k][0]) snprintf(p, sizeof p, "%s/%s", root, sub[k]); else snprintf(p, sizeof p, "%s", root);
        bio_dir_clear(p);
    }
}
static double o2_share(BioSim* s) {
    int64_t a[BIO_ATM_N], t = 0;
    bio_sim_get_atm(s, a);
    for (int g = 0; g < BIO_ATM_N; ++g) t += a[g];
    return t > 0 ? (double)a[BIO_ATM_O2] / (double)t : 0.0;
}
/* one epoch of both islands: run, export, hub pass, deliver; then import at the epoch after */
static int hub_epoch(BioSim* s[2], const char* run[2], const char* hub, uint64_t epoch, int use_hub, int mig[2], int atm[2]) {
    static const int isl[2] = { 0, 1 };
    char a[300], b[300], rep_[2048];
    for (int k = 0; k < 2; ++k) {
        if (bio_sim_run(s[k], 240)) return -1;
        if (!use_hub) continue;
        if (bio_hub_export(s[k], run[k], epoch, 64)) return -1;
        snprintf(a, sizeof a, "%s/outbox", run[k]); snprintf(b, sizeof b, "%s/in", hub);
        if (bio_dir_copy(a, b, "") < 0) return -1;
    }
    if (!use_hub) return 0;
    if (bio_hub_run(hub, isl, 2, 110000, rep_, sizeof rep_) < 0) return -1;
    for (int k = 0; k < 2; ++k) {
        snprintf(a, sizeof a, "%s/out/%d", hub, k); snprintf(b, sizeof b, "%s/inbox", run[k]);
        if (bio_dir_copy(a, b, "") < 0) return -1;
        bio_dir_clear(a);
        if (bio_hub_import(s[k], run[k], epoch, &mig[k], &atm[k])) return -1;
    }
    return 0;
}
/* ---- T17b: the hub's arithmetic on hand-written island files (no simulation) ---------------------
 * (d) the planet's column and clock do not depend on how the islands' reports fall
 *     into passes; (e) an island founded anew (its epoch goes back) is taken in again
 *     from there; (f) a file of another planet is refused. */
static int hubx_atm(const char* hub, int k, long long epoch, unsigned long long planet, long long up_all, long long up_land) {
    char p[300];
    snprintf(p, sizeof p, "%s/in", hub);
    if (bio_mkdir_p(p)) return -1;
    snprintf(p, sizeof p, "%s/in/atm_%d_%08lld.txt", hub, k, epoch);
    FILE* f = fopen(p, "wb");
    if (!f) return -1;
    /* no O2 in the air: the hub's photochemistry has nothing to do, only the uptake moves the column */
    fprintf(f, "GMEATM1\nisland %d\nepoch %lld\nstep %lld\nplanet %016llx\nscale_ppm 100\ncells_ocean 1000\ncells_land 1000\ncell_m2 256\n"
               "box 1000000000 2000000000 1000000000 500000000 0 0 0\nxch_all %lld %lld 0 0 0 0 0\nxch_land %lld %lld 0 0 0 0 0\n",
            k, epoch, epoch * 8760, planet, up_all, 2 * up_all, up_land, 2 * up_land);
    return fclose(f) ? -1 : 0;
}
static int hubx_state(const char* hub, long long* clock_q10, long long g[BIO_ATM_N]) {
    char p[300], line[1024];
    snprintf(p, sizeof p, "%s/hub_state.txt", hub);
    FILE* f = fopen(p, "rb");
    if (!f) return -1;
    int got = 0;
    while (fgets(line, sizeof line, f)) {
        long long a;
        if (sscanf(line, "clock_q10 %lld", &a) == 1) { *clock_q10 = a; got |= 1; }
        else if (!strncmp(line, "global ", 7)) {
            const char* q = line + 7;
            for (int k = 0; k < BIO_ATM_N; ++k) { char* e = NULL; g[k] = strtoll(q, &e, 10); q = e; }
            got |= 2;
        }
    }
    fclose(f);
    return got == 3 ? 0 : -1;
}
static void hubx_clear(const char* hub) {
    char p[300];
    hub_dirs_clear(hub);
    snprintf(p, sizeof p, "%s/hub_state.txt", hub); remove(p);
    snprintf(p, sizeof p, "%s/atm_global.txt", hub); remove(p);
    hub_dirs_clear(hub);
}
static int t17b_hub_arithmetic(void) {
    static const int isl[2] = { 0, 1 };
    const unsigned long long PL = 0x1234abcd5678ef01ull;
    const char* hub[2] = { "bio_t17_hx_a", "bio_t17_hx_b" };
    char rep_[2048];
    long long ck[2] = { 0, 0 }, g[2][BIO_ATM_N], g0[BIO_ATM_N], c0 = 0;
    int ok_d = 0, ok_e = 0, ok_f = 0, bad = 0;
    memset(g, 0, sizeof g); memset(g0, 0, sizeof g0);
    for (int v = 0; v < 2 && !bad; ++v) {
        hubx_clear(hub[v]);
        /* epoch 10: both islands are taken in (the baseline) */
        bad |= hubx_atm(hub[v], 0, 10, PL, 100000, 20000) || hubx_atm(hub[v], 1, 10, PL, 300000, 60000);
        bad |= bio_hub_run(hub[v], isl, 2, 300000, rep_, sizeof rep_) < 0;
        if (v == 0) bad |= hubx_state(hub[v], &c0, g0);
        /* epoch 11 of both: in one pass (v = 0), or island 0 first and island 1 a pass later (v = 1) */
        bad |= hubx_atm(hub[v], 0, 11, PL, 100000 + 512000, 20000 + 128000);
        if (v == 1) bad |= bio_hub_run(hub[v], isl, 2, 300000, rep_, sizeof rep_) < 0;
        bad |= hubx_atm(hub[v], 1, 11, PL, 300000 + 256000, 60000 + 64000);
        bad |= bio_hub_run(hub[v], isl, 2, 300000, rep_, sizeof rep_) < 0;
        bad |= hubx_state(hub[v], &ck[v], g[v]);
    }
    if (!bad) {
        ok_d = ck[0] == c0 + 1024 && ck[1] == ck[0] && g[0][0] < g0[0] && g[0][1] < g0[1];
        for (int k = 0; k < BIO_ATM_N; ++k) if (llabs(g[0][k] - g[1][k]) > 2) ok_d = 0;
        /* (e) island 1 founded anew: its outbox, and so in/, now holds epoch 1 only */
        char p[300];
        for (long long e = 10; e <= 11; ++e) { snprintf(p, sizeof p, "%s/in/atm_1_%08lld.txt", hub[0], e); remove(p); }
        long long c1 = 0, c2 = 0, g1[BIO_ATM_N], g2[BIO_ATM_N];
        bad |= hubx_atm(hub[0], 1, 1, PL, 5000, 1000);
        const int r1 = bio_hub_run(hub[0], isl, 2, 300000, rep_, sizeof rep_);
        const int said = strstr(rep_, "founded anew") != NULL;
        bad |= hubx_state(hub[0], &c1, g1);
        bad |= hubx_atm(hub[0], 1, 2, PL, 5000 + 256000, 1000 + 64000);
        const int r2 = bio_hub_run(hub[0], isl, 2, 300000, rep_, sizeof rep_);
        bad |= hubx_state(hub[0], &c2, g2);
        ok_e = !bad && r1 == 0 && said && c1 == ck[0] && !memcmp(g1, g[0], sizeof g1) && r2 == 1 && c2 == c1 + 512 && g2[0] < g1[0];
        /* (f) a file of another planet stops the pass and changes nothing */
        long long c3 = 0, g3[BIO_ATM_N];
        bad |= hubx_atm(hub[0], 0, 12, PL + 1, 900000, 200000);
        const int r3 = bio_hub_run(hub[0], isl, 2, 300000, rep_, sizeof rep_);
        bad |= hubx_state(hub[0], &c3, g3);
        ok_f = !bad && r3 < 0 && strstr(bio_error(), "planet") != NULL && c3 == c2 && !memcmp(g3, g2, sizeof g3);
    }
    hubx_clear(hub[0]); hubx_clear(hub[1]);
    const int fail = bad || !(ok_d && ok_e && ok_f);
    LOG("T17b hub arithmetic (hand-written island files): one island-year each, reported in one pass or in two: clock +%.3f / +%.3f planet-years, "
        "H2S column %lld / %lld umol/m2 from %lld (the same: %s); an island founded anew is taken in from its new epoch (%s); a file of "
        "another planet is refused and nothing changes (%s): %s\n", (ck[0] - c0) / 1024.0, (ck[1] - c0) / 1024.0, g[0][0], g[1][0], g0[0],
        ok_d ? "yes" : "NO", ok_e ? "yes" : "NO", ok_f ? "yes" : "NO", fail ? "FAIL" : "pass");
    return fail;
}
static int t17_hub(int device) {
    const char* run[2] = { "bio_t17_a", "bio_t17_b" };
    const char* hub = "bio_t17_hub";
    hub_dirs_clear(run[0]); hub_dirs_clear(run[1]); hub_dirs_clear(hub);
    BioEnv e[2];
    SynthOpts o = { 64, 64, 40, 40, 4, 6, 220, 0 };
    double gap[2] = { 0, 0 };
    int ok_m = 0, ok_l = 0, ok_r = 0, migk[2] = { 0, 0 }, atmk[2] = { 0, 0 }, fail = 1;
    uint64_t h_straight = 0, h_replay = 0;
    for (int pass = 0; pass < 2; ++pass) {          /* pass 0: with the hub; pass 1: the same islands on their own */
        BioSim* s[2] = { NULL, NULL };
        for (int k = 0; k < 2; ++k) {
            synth_env(&e[k], &o);
            e[k].island = k; e[k].crc = bio_env_crc(&e[k]);
            BioConfig c = test_cfg(device, 131072);
            c.salt = 7 + (uint32_t)k;
            s[k] = bio_sim_create(&e[k], &c);
        }
        if (s[0] && s[1] && !bio_sim_seed(s[0], 1500) && !bio_sim_seed(s[1], 1500)) {
            int64_t a[BIO_ATM_N];
            bio_sim_get_atm(s[1], a);
            a[BIO_ATM_O2] += a[BIO_ATM_CH4] / 4;                        /* island 1 starts with oxygen in its air */
            bio_sim_set_atm(s[1], a);
            int bad = 0;
            const char* ck = "bio_t17.ckpt";
            for (uint64_t ep = 1; ep <= 6 && !bad; ++ep) {
                int mig[2] = { 0, 0 }, atm[2] = { 0, 0 };
                bad = hub_epoch(s, run, hub, ep, pass == 0, mig, atm);
                if (pass == 0) { migk[0] += mig[0]; migk[1] += mig[1]; atmk[0] += atm[0]; atmk[1] += atm[1]; }
                if (pass == 0 && ep == 2 && !bad) bad = bio_sim_save(s[0], ck);
            }
            if (bad) LOG("  T17: %s\n", bio_error());
            gap[pass] = fabs(o2_share(s[0]) - o2_share(s[1]));
            if (pass == 0 && !bad) {
                ok_m = migk[0] > 0 && migk[1] > 0 && atmk[0] > 0 && atmk[1] > 0;      /* both ways */
                ok_l = residual_zero(s[0], "hub island 0") && residual_zero(s[1], "hub island 1");
                h_straight = bio_sim_hash(s[0]);
                /* (c) island 0 again from its epoch-2 checkpoint: epochs 3..6 replay from imported/ */
                BioConfig c = test_cfg(device, 131072);
                c.salt = 7;
                BioSim* r = bio_sim_load(ck, &e[0], &c);
                if (r && bio_sim_hub_epoch(r) == 2) {
                    int rb = 0;
                    for (uint64_t ep = 3; ep <= 6 && !rb; ++ep) {
                        int m2 = 0, a2 = 0;
                        rb = bio_sim_run(r, 240) || bio_hub_export(r, run[0], ep, 64) || bio_hub_import(r, run[0], ep, &m2, &a2);
                    }
                    if (!rb) h_replay = bio_sim_hash(r);
                }
                ok_r = h_straight && h_straight == h_replay;
                bio_sim_destroy(r);
                remove(ck);
            }
        } else LOG("  T17: setup failed: %s\n", bio_error());
        for (int k = 0; k < 2; ++k) { bio_sim_destroy(s[k]); bio_env_free(&e[k]); }
    }
    hub_dirs_clear(run[0]); hub_dirs_clear(run[1]); hub_dirs_clear(hub);
    const int ok_g = gap[0] < 0.5 * gap[1];
    fail = !(ok_m && ok_l && ok_r && ok_g);
    LOG("T17 island hub (2 islands, 6 epochs of 240 steps): %d + %d migrants imported and %d + %d atmosphere updates applied by island 0 + "
        "island 1 (%s), ledgers exact %d; "
        "O2 share gap between the islands %.4f with the hub vs %.4f without (under half: %s); replay from the epoch-2 checkpoint "
        "%016llx vs %016llx (%s): %s\n", migk[0], migk[1], atmk[0], atmk[1], ok_m ? "both ways" : "NOT both ways", ok_l, gap[0], gap[1],
        ok_g ? "yes" : "no",
        (unsigned long long)h_replay, (unsigned long long)h_straight, ok_r ? "identical" : "DIFFERS", fail ? "FAIL" : "pass");
    return fail + t17b_hub_arithmetic();
}

/* ---- T18: seed bank, rebase, inoculation ---------------------------------------------------------
 * An evolved island is banked (one record per genome and habitat, best per
 * guild x habitat kept, file round trip). Its genomes then found a run on
 * another planet: every founder must sit in a cell of its genome's habitat and
 * the ledger must stay exact. Inoculation is the same with a guild filter. */
static int t18_bank(int device) {
    BioEnv e, e2;
    SynthOpts o = { 64, 64, 40, 40, 4, 6, 220, 0 }, o2 = { 96, 48, 30, 60, 3, 4, 200, 0 };
    synth_env(&e, &o);
    synth_env(&e2, &o2);
    e2.planet_hash ^= 0x9E3779B97F4A7C15ull; e2.seed = 77; e2.island = 2; e2.crc = bio_env_crc(&e2);
    BioConfig c = test_cfg(device, 131072);
    BioSim* a = bio_sim_create(&e, &c);
    BioSim* b = bio_sim_create(&e2, &c);
    BioSim* d = bio_sim_create(&e2, &c);
    BioBankRec* r = NULL; BioBankRec* r2 = NULL;
    uint32_t n = 0, nt = 0, n2 = 0;
    uint64_t ph = 0;
    const char* path = "bio_t18.bank";
    int ok_bank = 0, ok_seed = 0, ok_inoc = 0, placed = 0, inoc = 0, classes = 0;
    if (a && b && d && !bio_sim_seed(a, 1000) && !seed_ecosystem(a, 64, 40) && !bio_sim_run(a, 600) && !bio_sim_bank(a, &r, &n)) {
        nt = bio_bank_top(r, n, 40);
        int cls[BIO_GUILD_N][BIO_HAB_N];
        memset(cls, 0, sizeof cls);
        int sorted = 1;
        for (uint32_t k = 0; k < nt; ++k) {
            cls[r[k].guild][r[k].habitat]++;
            if (k && r[k].guild == r[k - 1].guild && r[k].habitat == r[k - 1].habitat && r[k].mass > r[k - 1].mass) sorted = 0;
        }
        int over = 0;
        for (int g = 0; g < BIO_GUILD_N; ++g) for (int h = 0; h < BIO_HAB_N; ++h) { classes += cls[g][h] > 0; over |= cls[g][h] > 40; }
        ok_bank = nt > 0 && nt <= n && sorted && !over && classes >= 4 && !bio_bank_save(path, r, nt, e.planet_hash) &&
                  !bio_bank_load(path, &r2, &n2, &ph) && n2 == nt && ph == e.planet_hash && !memcmp(r, r2, sizeof(BioBankRec) * nt);
        remove(path);
        /* rebase: founders on the other planet */
        placed = bio_sim_seed_bank(b, r, nt, 3000, -1, -1);
        if (placed > 0) {
            BioAgentRec* ag = NULL; uint32_t na = 0;
            int wrong = 0;
            if (!bio_download_agents(b, &ag, &na)) {
                for (uint32_t i = 0; i < na; ++i) {
                    const int hab = bio_habitat(&e2, bio_cell_at(&b->w, ag[i].x, ag[i].z));
                    int found = 0;
                    for (uint32_t k = 0; k < nt && !found; ++k) found = (int)r[k].habitat == hab && !memcmp(r[k].genome, &ag[i].g, 384);
                    wrong += !found;
                }
                ok_seed = na == (uint32_t)placed && !wrong && residual_zero(b, "rebase") && !bio_sim_run(b, 240) && residual_zero(b, "rebase+run") && b->n_live > 0;
            }
            free(ag);
        }
        /* inoculation: land producers only, into an island that has its own life */
        if (!bio_sim_seed(d, 500)) {
            const uint32_t before = d->n_live;
            inoc = bio_sim_seed_bank(d, r, nt, 400, BIO_GUILD_PRODUCER, BIO_HAB_LAND);
            BioAgentRec* ag = NULL; uint32_t na = 0;
            int wrong = 0, got = 0;
            if (inoc > 0 && !bio_download_agents(d, &ag, &na)) {
                for (uint32_t i = 0; i < na; ++i) {
                    if (ag[i].g.body[BG_E_METH] == 160 && ag[i].g.body[BG_PIGMENT] == 0) continue;      /* the island's own protocells */
                    ++got;
                    wrong += bio_guild(ag[i].g.body) != BIO_GUILD_PRODUCER || bio_habitat(&e2, bio_cell_at(&d->w, ag[i].x, ag[i].z)) != BIO_HAB_LAND;
                }
                ok_inoc = d->n_live == before + (uint32_t)inoc && got == inoc && !wrong && residual_zero(d, "inoculate");
            }
            free(ag);
        }
    } else LOG("  T18: setup failed: %s\n", bio_error());
    LOG("T18 seed bank (ecosystem island, 600 steps): %u genome x habitat records, %u kept (best 40 of %d guild x habitat classes), file round "
        "trip %s; rebase onto another planet: %d founders, each in its habitat, ledger exact: %s; inoculation of land producers: %d placed, "
        "all land producers: %s: %s\n", n, nt, classes, ok_bank ? "ok" : "BAD", placed, ok_seed ? "ok" : "BAD", inoc, ok_inoc ? "ok" : "BAD",
        ok_bank && ok_seed && ok_inoc ? "pass" : "FAIL");
    free(r); free(r2);
    bio_sim_destroy(a); bio_sim_destroy(b); bio_sim_destroy(d);
    bio_env_free(&e); bio_env_free(&e2);
    return !(ok_bank && ok_seed && ok_inoc);
}

/* ---- T19: species count, creature export, vegetation with shapes ---------------------------------- */
static int t19_exports(int device) {
    int fails = 0;
    /* species: two clones far apart in genes and unable to cross -> 2 clusters, half each */
    {
        enum { W19 = 16 };
        int16_t depth[W19 * W19];
        for (int k = 0; k < W19 * W19; ++k) depth[k] = 4;
        BioEnv e; custom_env(&e, W19, W19, depth);
        BioConfig c = test_cfg(device, 65536);
        BioSim* s = bio_sim_create(&e, &c);
        BioSpecies sp, sp1;
        memset(&sp, 0, sizeof sp); memset(&sp1, 0, sizeof sp1);
        int ok = 0;
        if (s) {
            s->w.P.mut_off = 1;
            BioGenome g; plain_genome(&g);
            g.body[BG_E_ANOX] = 160; g.body[BG_PIGMENT] = 160; g.body[BG_HOLDFAST] = 200; g.mate_tag = 0x0F0F;
            place(s, &g, 600, 0, W19 * W19, 8, 32);
            bio_sim_species(s, 0, &sp1);
            BioGenome h = g;
            h.body[BG_E_ANOX] = 0; h.body[BG_E_OXY] = 200; h.body[BG_PIGMENT] = 40; h.body[BG_HUE] = 200; h.body[BG_SIZE] = 60;
            h.body[BG_STRUCT] = 200; h.body[BG_CUTICLE] = 255; h.mate_tag = 0xF0F0;
            place(s, &h, 600, 0, W19 * W19, 8, 32);
            ok = !bio_sim_species(s, 0, &sp) && sp1.richness == 1 && sp.richness == 2 && sp.sample == 1200 &&
                 fabs(sp.shannon - 0.6931) < 0.01 && fabs(sp.top_share - 0.5) < 0.01;
        }
        LOG("T19a species (clusters of similar genomes): one clone -> %u species; two clones far apart -> %u species, Shannon %.3f, largest "
            "%.0f%% of %u sampled: %s\n", sp1.richness, sp.richness, sp.shannon, 100.0 * sp.top_share, sp.sample, ok ? "pass" : "FAIL");
        fails += !ok;
        bio_sim_destroy(s); bio_env_free(&e);
    }
    /* creatures and plant shapes of the ecosystem island */
    {
        BioEnv e;
        SynthOpts o = { 64, 64, 40, 40, 4, 6, 220, 0 };
        synth_env(&e, &o);
        e.x0 = 4096; e.z0 = -8192;
        BioConfig c = test_cfg(device, 131072);
        BioSim* s = bio_sim_create(&e, &c);
        static Fauna fa, fb;
        memset(&fa, 0, sizeof fa); memset(&fb, 0, sizeof fb);
        const char* path = "bio_t19.fauna";
        int ok_f = 0, ok_v = 0;
        uint32_t nveg = 0, nmorph = 0, crowns = 0, want = 0, creepers = 0, motile = 0;
        /* a forest with a motor gene of 2, as evolved islands carry: plants, not creatures */
        BioGenome cp; plain_genome(&cp);
        cp.body[BG_E_OXY] = 160; cp.body[BG_PIGMENT] = 200; cp.body[BG_HOLDFAST] = 200; cp.body[BG_MOTOR] = 2; cp.body[BG_HUE] = 150;
        cp.body[BG_CUTICLE] = 200; cp.body[BG_O2TOL] = 255; cp.body[BG_CROWN] = 120;
        if (s && !bio_sim_seed(s, 500) && !seed_ecosystem(s, 64, 40) && !place(s, &cp, 400, 0, 64 * 64, 64, 64) && !bio_sim_run(s, 300) &&
            !bio_sim_fauna(s, &fa, 4096)) {
            int sorted = 1, inside = 1;
            {
                BioAgentRec* ag = NULL; uint32_t na = 0;
                if (!bio_download_agents(s, &ag, &na))
                    for (uint32_t i = 0; i < na; ++i) {
                        const uint8_t* b = ag[i].g.body;
                        if (!b[BG_MOTOR] || ag[i].C + ag[i].S + ag[i].gut <= 0) continue;
                        ++motile;
                        const int plant = b[BG_HOLDFAST] > 64 && (b[BG_E_ANOX] >= 32 || b[BG_E_OXY] >= 32);
                        creepers += (uint32_t)plant;
                        const int moves = b[BG_MOTOR] >= 16, eats = BE_GUTSHARE(ag[i].eco) > 128u;
                        want += (uint32_t)(plant ? (moves && eats) : (moves || eats));
                    }
                free(ag);
                BioStats st19;
                if (bio_sim_stats(s, &st19) || st19.creatures != want) want = 0xFFFFFFFFu;      /* the statistics count by the same rule */
            }
            for (uint32_t k = 0; k < fa.count; ++k) {
                if (k && fa.rec[k].size_q4 > fa.rec[k - 1].size_q4) sorted = 0;
                const int32_t x = fa.rec[k].x_q4 >> 4, z = fa.rec[k].z_q4 >> 4;
                if (x < e.x0 || x >= e.x0 + (e.w << e.shift) || z < e.z0 || z >= e.z0 + (e.h << e.shift) || !fa.rec[k].motor ||
                    fa.rec[k].pal < 1 || fa.rec[k].pal > 15) inside = 0;
                if ((fa.rec[k].flags & FAUNA_SWIMS) == 0 && fa.rec[k].depth_q4) inside = 0;
            }
            ok_f = fa.count > 0 && fa.count <= fa.total && fa.total == want && creepers > 0 && want < motile && sorted && inside &&
                   fa.planet_hash == e.planet_hash &&
                   !fauna_save(path, &fa) && !fauna_load(path, &fb) && fb.count == fa.count && fb.total == fa.total &&
                   !memcmp(fa.rec, fb.rec, sizeof(FaunaRec) * fa.count) && !memcmp(fa.palette, fb.palette, 48);
            remove(path);
            uint32_t* cells = (uint32_t*)malloc(sizeof(uint32_t) * 64 * 64);
            uint32_t* morph = (uint32_t*)malloc(sizeof(uint32_t) * 64 * 64);
            uint32_t fbk[64], fbm[64];
            uint8_t pal[16][3];
            if (cells && morph && !bio_sim_vegetation2(s, cells, morph, fbk, fbm, pal)) {
                int bad = 0;
                uint32_t seen = 0;
                for (int k = 0; k < 64 * 64; ++k) {
                    if (!BIOCELL_KIND(cells[k])) { bad |= morph[k] != 0; continue; }
                    ++nveg;
                    if (!(morph[k] & BIOMORPH_VALID)) { ++bad; continue; }
                    ++nmorph;
                    if (BIOCELL_KIND(cells[k]) >= BIO_KIND_SHRUB) seen |= 1u << BIOMORPH_CROWN(morph[k]);
                    bad |= BIOMORPH_CROWN(morph[k]) >= BIO_CROWN_N;
                }
                for (uint32_t q = seen; q; q &= q - 1) ++crowns;
                /* the two plant lineages of the ecosystem: crown genes 120 -> class 2 (umbrella) and 220 -> class 4 (dome) */
                ok_v = nveg > 0 && nmorph == nveg && !bad && (crowns == 0 || (seen & ~((1u << 2) | (1u << 4))) == 0);
            }
            free(cells); free(morph);
        } else LOG("  T19: setup failed: %s\n", bio_error());
        LOG("T19b creature export: %u of %u creatures (%u agents have a motor; %u of them are plants with a holdfast, exported only if they walk and graze), "
            "largest first, inside the island, file round trip: %s; vegetation with shapes: "
            "%u vegetated cells, %u with a shape word, %u crown class(es) among shrubs and trees: %s\n", fa.count, fa.total, motile, creepers,
            ok_f ? "pass" : "FAIL", nveg, nmorph, crowns, ok_v ? "pass" : "FAIL");
        fails += !ok_f + !ok_v;
        fauna_free(&fa); fauna_free(&fb);
        bio_sim_destroy(s); bio_env_free(&e);
    }
    return fails;
}

/* ---- API: vegetation, migrants, atmosphere, duty, auto capacity ----------------------- */
static int residual_zero(BioSim* s, const char* tag) {
    BioStats st;
    if (bio_sim_stats(s, &st)) return 0;
    for (int q = 0; q < 4; ++q)
        if (st.total[q] != st.ledger[q]) {
            LOG("  %s: element %d total %lld ledger %lld\n", tag, q, (long long)st.total[q], (long long)st.ledger[q]);
            return 0;
        }
    return 1;
}

static int api_tests(int device) {
    int fails = 0;
    /* vegetation classes: a column of land and water cells with sessile phototrophs of set heights */
    {
        enum { NV = 8 };
        const int16_t depth[NV] = { 5, 5, 6, -3, -3, -3, -3, -3 };
        BioEnv e; custom_env(&e, NV, 1, depth);
        for (int k = 0; k < NV; ++k) e.env_bin[k] = (uint8_t)(k < 3 ? 1 : 40);
        e.crc = bio_env_crc(&e);
        BioConfig c = test_cfg(device, 4096);
        BioSim* s = bio_sim_create(&e, &c);
        int ok = 0;
        if (s) {
            BioGenome g; plain_genome(&g);
            g.body[BG_E_ANOX] = 100; g.body[BG_PIGMENT] = 200; g.body[BG_HOLDFAST] = 200; g.body[BG_HUE] = 128;
            /* cell 0: mat (S = 0); cell 1: kelp (H 2 m: S = 4000 * 8); cell 3: ground cover; 4: shrub (2 m); 5: tree (10 m) */
            const int32_t cells[5] = { 0, 1, 3, 4, 5 };
            const int32_t S[5] = { 0, 32000, 0, 32000, 4000000 };
            for (int k = 0; k < 5; ++k) {
                if (place(s, &g, 3, cells[k], cells[k], 20000, 64)) break;
            }
            /* set S of the agents just placed (host-side edit of the records via a reload of S) */
            BioAgentRec* a = NULL; uint32_t n = 0;
            if (!bio_download_agents(s, &a, &n) && n == 15) {
                int32_t sv[15];
                for (uint32_t i = 0; i < n; ++i) sv[i] = S[(a[i].x >> 12) == 0 ? 0 : ((a[i].x >> 12) == 1 ? 1 : (a[i].x >> 12) - 1)];
                bio_copy_in(s, s->w.S, sv, sizeof sv);
            }
            free(a);
            uint32_t cellv[NV], fb[64];
            uint8_t pal[16][3];
            if (!bio_sim_vegetation(s, cellv, fb, pal)) {
                const uint32_t want[NV] = { BIO_KIND_MAT, BIO_KIND_KELP, BIO_KIND_NONE, BIO_KIND_COVER, BIO_KIND_SHRUB, BIO_KIND_TREE, 0, 0 };
                ok = 1;
                for (int k = 0; k < NV; ++k) if (BIOCELL_KIND(cellv[k]) != want[k]) ok = 0;
                LOG("API vegetation: kinds");
                for (int k = 0; k < NV; ++k) LOG(" %u(h%u,c%u,p%u)", BIOCELL_KIND(cellv[k]), BIOCELL_HEIGHT(cellv[k]), BIOCELL_COVER(cellv[k]), BIOCELL_PAL(cellv[k]));
                LOG("; fallback bin1 kind %u bin40 kind %u; palette[8] = %u %u %u: %s\n", BIOCELL_KIND(fb[1]), BIOCELL_KIND(fb[40]),
                    pal[8][0], pal[8][1], pal[8][2], ok ? "pass" : "FAIL");
            }
        }
        fails += !ok;
        bio_sim_destroy(s); bio_env_free(&e);
    }
    /* migrants, atmosphere set/get, save/load of a grown island, duty */
    {
        BioEnv e;
        SynthOpts o = { 64, 64, 40, 48, 4, 6, 220, 0 };
        synth_env(&e, &o);
        BioConfig c = test_cfg(device, 262144);
        BioSim* a = bio_sim_create(&e, &c);
        c.salt = 99;
        BioSim* b = bio_sim_create(&e, &c);
        int ok_m = 0, ok_a = 0, ok_d = 0;
        if (a && b && !bio_sim_seed(a, 2000) && !bio_sim_seed(b, 500) && !bio_sim_run(a, 2400) && !bio_sim_run(b, 240)) {
            const char* path = "bio_api_mig.bin";
            const int ne = bio_sim_export_migrants(a, path, 64);
            const uint32_t before = b->n_live;
            const int ni = bio_sim_import_migrants(b, path);
            remove(path);
            ok_m = ne > 0 && ni > 0 && b->n_live == before + (uint32_t)ni && residual_zero(b, "migrants") && !bio_sim_run(b, 240) && residual_zero(b, "migrants+run");
            LOG("API migrants: exported %d, imported %d (live %u -> %u), ledger exact: %s\n", ne, ni, before, before + (ni > 0 ? ni : 0), ok_m ? "pass" : "FAIL");
            int64_t atm[BIO_ATM_N], atm2[BIO_ATM_N];
            bio_sim_get_atm(a, atm);
            memcpy(atm2, atm, sizeof atm);
            atm2[BIO_ATM_O2] += 123456; atm2[BIO_ATM_CH4] -= atm2[BIO_ATM_CH4] / 10; atm2[BIO_ATM_CO2] += 5000;
            int64_t atm3[BIO_ATM_N];
            ok_a = !bio_sim_set_atm(a, atm2) && !bio_sim_get_atm(a, atm3) && !memcmp(atm2, atm3, sizeof atm3) &&
                   residual_zero(a, "set_atm") && !bio_sim_run(a, 240) && residual_zero(a, "set_atm+run");
            LOG("API atmosphere: get/set round trip and ledger exact after +O2/-CH4/+CO2 and 240 steps: %s\n", ok_a ? "pass" : "FAIL");
            /* duty 0.5: wall time about twice the busy time */
            a->cfg.duty = 1.0;
            double t0 = time_now(); bio_sim_run(a, 240); const double busy = time_now() - t0;
            a->cfg.duty = 0.5;
            t0 = time_now(); bio_sim_run(a, 240); const double wall = time_now() - t0;
            ok_d = wall > 1.5 * busy;
            LOG("API duty 0.5: 240 steps %.3f s vs %.3f s at duty 1 (ratio %.2f): %s\n", wall, busy, busy > 0 ? wall / busy : 0.0, ok_d ? "pass" : "FAIL");
        } else LOG("API: setup failed: %s\n", bio_error());
        fails += !ok_m + !ok_a + !ok_d;
        bio_sim_destroy(a); bio_sim_destroy(b); bio_env_free(&e);
    }
    /* capacity from free device memory */
    if (device >= 0) {
        BioEnv e;
        SynthOpts o = { 256, 256, 40, 160, 4, 16, 220, 0 };
        synth_env(&e, &o);
        BioConfig c = test_cfg(device, 0);
        BioSim* s = bio_sim_create(&e, &c);
        size_t fr = 0, tt = 0;
        rt_mem_info(&fr, &tt);
        const int ok = s && s->cfg.cap >= 4096 && fr >= ((size_t)1 << 30);
        LOG("API auto capacity: cap %u agents, %.2f GB still free of %.2f GB: %s\n", s ? s->cfg.cap : 0, fr / 1073741824.0, tt / 1073741824.0, ok ? "pass" : "FAIL");
        fails += !ok;
        bio_sim_destroy(s); bio_env_free(&e);
    }
    return fails;
}

/* ---- T12: benchmark ------------------------------------------------------------------ */
static int t12_one(int device, uint32_t n0);
static int t12_bench(int device) {
    int f = t12_one(device, device < 0 ? 65536 : 1000000);
    if (device >= 0 && getenv("GME_BIOT12_ALL")) { t12_one(device, 2000000); t12_one(device, 4000000); }
    return f;
}
static int t12_one(int device, uint32_t n0) {
    const int32_t W = device < 0 ? 256 : 1024;
    BioEnv e;
    SynthOpts o = { W, W, 60, W * 3 / 4, 8, W * W / 4096, 220, 0 };
    if (synth_env(&e, &o)) return 1;
    /* cap so that the 0.95 cap birth guard holds the population at ~1.05 M */
    BioConfig c = test_cfg(device, device < 0 ? 72000 : n0 + n0 / 10);
    BioSim* s = bio_sim_create(&e, &c);
    int fail = 1;
    if (!s) { LOG("T12: create failed: %s\n", bio_error()); goto out; }
    /* fill: protocells plus a phototroph variant so the population holds during the timing */
    if (bio_sim_seed(s, n0)) { LOG("T12: seed failed: %s\n", bio_error()); goto out; }
    if (bio_sim_run(s, 24)) { LOG("T12: run failed: %s\n", bio_error()); goto out; }
    BioStats a, b;
    bio_sim_stats(s, &a);
    const uint32_t steps = device < 0 ? 48 : 480;
    if (bio_sim_run(s, steps)) { LOG("T12: run failed: %s\n", bio_error()); goto out; }
    bio_sim_stats(s, &b);
    const double sps = s->steps_per_s;
    bio_prof_report();
    fail = device >= 0 && n0 <= 1000000 && sps < 150.0;
    LOG("T12 benchmark (%s, %dx%d cells, live %llu -> %llu): %.1f steps/s over %u steps (target >= 150 at 1M on the 3060): %s\n",
        device < 0 ? "cpu" : s->dev.name, W, W, (unsigned long long)a.live, (unsigned long long)b.live, sps, steps,
        device < 0 ? "info" : (fail ? "FAIL" : "pass"));
out:
    bio_sim_destroy(s);
    bio_env_free(&e);
    return fail;
}

static int dbg_run(int device) {
    const int32_t W = 1024;
    BioEnv e;
    SynthOpts o = { W, W, 60, W * 3 / 4, 8, W * W / 4096, 220, 0 };
    if (synth_env(&e, &o)) return 1;
    BioConfig c = test_cfg(device, 2097152);
    BioSim* s = bio_sim_create(&e, &c);
    if (!s || bio_sim_seed(s, 1000000)) { LOG("DBG: %s\n", bio_error()); return 1; }
    print_stats("seeded", s);
    for (int k = 0; k < 6; ++k) {
        bio_sim_run(s, 1);
        LOG("  step %d: scan births %u deaths %u allowed %u live %u slots %u\n", k, s->last_scan_tot[0], s->last_scan_tot[1], s->last_scan_tot[2], s->n_live, s->w.n_slots);
    }
    print_stats("after", s);
    bio_sim_destroy(s);
    bio_env_free(&e);
    return 0;
}

/* ---- ecology run (GME_BIOTEST=ECO; GME_BIOECO_DAYS, GME_BIOECO_W) ------------------
 * 20,000 protocells on a synthetic island: shelf, vents and land with a coast.
 * Prints live counts, carbon by realm, daily pathway fluxes, gene means, mean
 * aqueous chemistry and the conservation residual every 30 sim-days. */
static int64_t env_i64(const char* name, int64_t def) {
    const char* v = getenv(name);
    return v && *v ? (int64_t)strtoll(v, NULL, 10) : def;
}

static int eco_run(int device) {
    const int32_t W = (int32_t)env_i64("GME_BIOECO_W", 256);
    const int64_t days = env_i64("GME_BIOECO_DAYS", 365 * 3);
    const int64_t every = env_i64("GME_BIOECO_EVERY", 30);
    BioEnv e;
    SynthOpts o = { W, W, 60, W * 5 / 8, 6, W * W / 2048, 230, 0 };
    if (env_i64("GME_BIOECO_T3", 0)) { const SynthOpts o3 = { W, W, 40, W * 3 / 4, 4, 6, 220, 0 }; o = o3; }   /* the T2/T3 island */
    if (synth_env(&e, &o)) return 1;
    int32_t nwater = 0, nland = 0, nvent = 0;
    for (int32_t c = 0; c < W * W; ++c) { nwater += e.depth[c] >= 1; nland += e.wet[2][c] == 0; nvent += e.vent[c] > 0; }
    BioConfig c = test_cfg(device, (uint32_t)env_i64("GME_BIOECO_CAP", 2000000));
    c.salt = (uint32_t)env_i64("GME_BIOECO_SALT", 7);
    c.closed_box = (int32_t)env_i64("GME_BIOECO_CLOSED", 0);
    BioSim* s = bio_sim_create(&e, &c);
    int fail = 0;
    if (!s || bio_sim_seed(s, (uint32_t)env_i64("GME_BIOECO_SEED", 20000))) { LOG("ECO: setup failed: %s\n", bio_error()); fail = 1; goto out; }
    LOG("ECO: %dx%d cells (%d water, %d dry land, %d vents), cap %u, %lld days on %s\n", W, W, nwater, nland, nvent,
        s->cfg.cap, (long long)days, device < 0 ? "cpu" : s->dev.name);
    LOG("  day     live  motile  sphoto |   C marine  coastal     land (mmol) | per day: GPP fix  METH  ANOX   OXY  FERM  SULF  AERO METHOX FEOX | genes METH ANOX OXY FERM SULF AERO PIG HOLD STRU CUT ROOT UVS SIZE THINK NEUR | aq mM: H2S DIC DOC O2(uM) H2(uM) | atmO2 ppm | resid | neg | gates\n");
    BioStats prev;
    memset(&prev, 0, sizeof prev);
    int64_t prev_fix = 0;
    const double t0 = time_now();
    for (int64_t d = 0; d <= days; d += every) {
        if (d > 0 && bio_sim_run(s, (uint64_t)every * 24)) { LOG("ECO: run failed: %s\n", bio_error()); fail = 1; break; }
        BioStats st;
        int64_t raw[BIO_STAT_N];
        if (bio_sim_stats(s, &st) || bio_stats_raw(s, raw) < 0) { fail = 1; break; }
        const int64_t fix = s->Gh.acc[BACC_FIX];
        const double div = d > 0 ? (double)every : 1.0;
        int64_t resid = 0;
        for (int q = 0; q < 4; ++q) resid |= st.total[q] - st.ledger[q];
        int64_t vol_w = 0;
        for (int32_t cc = 0; cc < W * W; ++cc) if (e.depth[cc] >= 1) vol_w += 256 * (int64_t)bio_clampi(e.depth[cc], 1, 10);
        const int64_t* F = raw + BS_FLD;
        int64_t atot = 0;
        for (int g = 0; g < BIO_ATM_N; ++g) atot += st.atm[g];
        const unsigned* mg = st.mean_gene;
        LOG("%5lld %8llu %7llu %7llu | %10lld %8lld %8lld | %8.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f %5.0f | %3u %3u %3u %3u %3u %3u %3u %3u %3u %3u %3u %3u %3u %3u %3u | %6.3f %5.2f %5.3f %7.2f %7.2f | %8.1f | %lld | %lld/%lld | %02x\n",
            (long long)d, (unsigned long long)st.live, (unsigned long long)st.motile, (unsigned long long)st.sessile_photo,
            (long long)st.carbon_marine, (long long)st.carbon_coastal, (long long)st.carbon_land,
            (double)(fix - prev_fix) / div,
            (double)(st.path_flux[0] - prev.path_flux[0]) / div, (double)(st.path_flux[1] - prev.path_flux[1]) / div,
            (double)(st.path_flux[2] - prev.path_flux[2]) / div, (double)(st.path_flux[3] - prev.path_flux[3]) / div,
            (double)(st.path_flux[4] - prev.path_flux[4]) / div, (double)(st.path_flux[5] - prev.path_flux[5]) / div,
            (double)(st.path_flux[6] - prev.path_flux[6]) / div, (double)(st.path_flux[7] - prev.path_flux[7]) / div,
            0.0,
            mg[BG_E_METH], mg[BG_E_ANOX], mg[BG_E_OXY], mg[BG_E_FERM], mg[BG_E_SULFRED], mg[BG_E_AERO], mg[BG_PIGMENT],
            mg[BG_HOLDFAST], mg[BG_STRUCT], mg[BG_CUTICLE], mg[BG_ROOT], mg[BG_UVSHIELD], mg[BG_SIZE], mg[BG_THINK], mg[BG_NEURONS],
            vol_w ? (double)F[BF_H2S] / (double)vol_w / 1000.0 : 0.0, vol_w ? (double)F[BF_DIC] / (double)vol_w / 1000.0 : 0.0,
            vol_w ? (double)F[BF_DOC] / (double)vol_w / 1000.0 : 0.0, vol_w ? (double)F[BF_O2] / (double)vol_w : 0.0,
            vol_w ? (double)F[BF_H2] / (double)vol_w : 0.0,
            atot ? 1e6 * (double)st.atm[BIO_ATM_O2] / (double)atot : 0.0, (long long)resid,
            (long long)raw[BS_NEGA], (long long)raw[BS_NEGF], st.gates);
        if (resid || raw[BS_NEGA] || raw[BS_NEGF]) fail = 1;
        prev = st;
        prev_fix = fix;
        if (st.live == 0) { LOG("ECO: extinct at day %lld\n", (long long)d); fail = 1; break; }
    }
    LOG("ECO: %.1f s wall, %.0f steps/s; %s\n", time_now() - t0, (double)days * 24.0 / (time_now() - t0), fail ? "FAIL" : "ok");
out:
    bio_sim_destroy(s);
    bio_env_free(&e);
    return fail;
}

/* GME_BIOTEST=T3,T12 (environment) selects tests; default: all of T1-T9 and T12 */
static int want(const char* name) {
    const char* sel = getenv("GME_BIOTEST");
    if (!sel || !*sel) return strcmp(name, "ECO") && strcmp(name, "DBG");   /* long ecology run and debug only on request */
    const size_t n = strlen(name);
    for (const char* p = sel; (p = strstr(p, name)) != NULL; p += n)
        if ((p == sel || p[-1] == ',') && (p[n] == 0 || p[n] == ',')) return 1;
    return 0;
}

int bio_run_tests(int device, int verbose) {
    g_verbose = verbose;
    int failed = 0;
    if (want("T1")) failed += t1_lint();
    if (want("T2")) failed += t2_cpu_gpu(device);
    if (want("T2")) failed += t2b_cpu_gpu_photo(device);
    if (want("T2")) failed += t2c_cpu_gpu_eco(device);
    if (want("T3")) failed += t3_conservation(device);
    if (want("T3")) failed += t3b_conservation_eco(device);
    if (want("T4")) failed += t4_resume(device);
    if (want("T4")) failed += t4b_resume_eco(device);
    if (want("T5")) failed += t5_chemostat(device);
    if (want("T6")) failed += t6_light(device);
    if (want("T7")) failed += t7_null(device);
    if (want("T8")) failed += t8_selection(device);
    if (want("T9")) failed += t9_learning(device);
    if (want("T10")) failed += t10_predator_prey(device);
    if (want("T12")) failed += t12_bench(device);
    if (want("T13")) failed += t13_spectral(device);
    if (want("T14")) failed += t14_legacy(device);
    if (want("T15")) failed += t15_sex(device);
    if (want("T16")) failed += t16_crowns(device);
    if (want("T17")) failed += t17_hub(device);
    if (want("T18")) failed += t18_bank(device);
    if (want("T19")) failed += t19_exports(device);
    if (want("API")) failed += api_tests(device);
    if (want("DBG")) failed += dbg_run(device);
    if (want("ECO")) failed += eco_run(device);
    return failed;
}

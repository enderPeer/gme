/* bio_test.c - life-test suite (spec 8.5): T1-T9 and T12, plus a synthetic
 * environment builder (ocean shelf, vents, land with a coast). */
#include "bio_internal.h"
#include "../platform.h"
#include "../biomap.h"
#include <stdlib.h>
#include <string.h>

#include "bio_kernels.inc"

static int g_verbose = 1;
static int64_t env_i64(const char* name, int64_t def);
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

static BioConfig test_cfg(int device, uint32_t cap) {
    BioConfig c;
    bio_config_default(&c);
    c.device = device;
    c.cap = cap;
    c.duty = 1.0;
    c.salt = 7;
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
static uint64_t run_hash(int device, uint32_t steps, uint64_t* mid_hash) {
    BioEnv e;
    SynthOpts o = { 64, 64, 40, 48, 4, 6, 220, 0 };
    if (synth_env(&e, &o)) return 0;
    BioConfig c = test_cfg(device, 65536);
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

/* n agents of genome g on cell c (or spread over cells [c0, c1) when c1 > c0) */
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
        a[k].C = C; a[k].E = E; a[k].N = (C + 5) / 6 + 1; a[k].P = C / 106 + 1; a[k].W = 256;
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
    LOG("T6 light profile (benthic ANOX phototrophs, 1 day): depth: gain/gain(1 m) vs exp(-0.06 z)/exp(-0.06):");
    for (int k = 0; k < ND; ++k) {
        const double obs = (gain[k] / cntd[k]) / (gain[0] / cntd[0]);
        const double idx_k = (double)(((int64_t)P->kw_q16 * (depth[k] * 256)) >> 18);
        const double idx_0 = (double)(((int64_t)P->kw_q16 * (depth[0] * 256)) >> 18);
        const double exp_ratio = (double)s->tables.expn_q16[(int)idx_k] / (double)s->tables.expn_q16[(int)idx_0];
        const double err = obs / exp_ratio - 1.0;
        if (err > worst || -err > worst) worst = err < 0 ? -err : err;
        if (k % 3 == 0 || k == ND - 1) LOG(" %dm %.3f/%.3f", depth[k], obs, exp_ratio);
    }
    fail = worst > 0.25;
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
    BioSim* s = bio_sim_create(&e, &c);
    double intake = 0;
    int64_t* f = (int64_t*)malloc(sizeof(int64_t) * W9 * W9 * BF_N);
    if (!s || !f) goto out;
    BioParams* P = &s->w.P;
    P->mut_off = 1; P->grow_off = 1; P->hazard_q32 = 0; P->kinpath_q32 = 0; P->kp_water_q16 = 0;
    P->d_ww = 3277;                         /* slower diffusion keeps a local gradient */
    P->learn_off = mode == 1; P->brain_off = mode == 2;
    BioGenome g; plain_genome(&g);
    g.body[BG_E_FERM] = 255; g.body[BG_MOTOR] = 160; g.body[BG_SENSE] = 255; g.body[BG_THINK] = 8;   /* think every 3 steps */
    g.body[BG_NEURONS] = 0xFF; g.body[BG_PLASTIC] = 0x03; g.body[BG_ETA] = 16; g.body[BG_DECAY] = 8;
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
    const double a = t9_run(device, 0), b = t9_run(device, 1), z = t9_run(device, 2);
    const int ok_learn = a > 1.10 * b, ok_brain = z < b && z < a;
    const int ok = ok_learn && ok_brain;
    LOG("T9 learning (moving DOC patch, FERM intake per day, days 10-40): eta>0 %.0f, eta=0 %.0f (%+.1f%%, needs > +10%%: %s), "
        "brain zeroed %.0f (worse than both: %s): %s\n",
        a, b, b > 0 ? 100.0 * (a / b - 1.0) : 0.0, ok_learn ? "yes" : "no", z, ok_brain ? "yes" : "no", ok ? "pass" : "FAIL");
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
    if (want("T3")) failed += t3_conservation(device);
    if (want("T4")) failed += t4_resume(device);
    if (want("T5")) failed += t5_chemostat(device);
    if (want("T6")) failed += t6_light(device);
    if (want("T7")) failed += t7_null(device);
    if (want("T8")) failed += t8_selection(device);
    if (want("T9")) failed += t9_learning(device);
    if (want("T12")) failed += t12_bench(device);
    if (want("API")) failed += api_tests(device);
    if (want("DBG")) failed += dbg_run(device);
    if (want("ECO")) failed += eco_run(device);
    return failed;
}

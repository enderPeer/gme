/* bio_sim.c - host driver of the life simulation: allocation, seeding, the
 * step loop (GPU kernels through rtcuda/NVRTC, or the CPU reference),
 * compaction, statistics, gates and the state hash. See bio_api.h. */
#include "bio_internal.h"
#include "../platform.h"
#include "../jobs.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include "bio_kernels.inc" /* BIO_KERNELS_SRC = bio_types.h + bio_core.h + bio_kernels.cu */

/* ---- errors, checksums ------------------------------------------------------- */
static char g_err[1024];
const char* bio_error(void) { return g_err; }
void bio_set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
}

static uint32_t crc_tab[256];
uint32_t bio_crc32(uint32_t crc, const void* p, size_t n) {
    if (!crc_tab[1])
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t v = i;
            for (int k = 0; k < 8; ++k) v = (v & 1) ? 0xEDB88320u ^ (v >> 1) : v >> 1;
            crc_tab[i] = v;
        }
    const uint8_t* b = (const uint8_t*)p;
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = crc_tab[(crc ^ b[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}
uint64_t bio_fnv64(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001B3ull; }
    return h;
}

/* ---- environment memory (contract) -------------------------------------------- */
int bio_env_alloc(BioEnv* e, int32_t w, int32_t h) {
    if (!e || w <= 0 || h <= 0) { bio_set_error("bio_env_alloc: bad size %d x %d", w, h); return -1; }
    const size_t n = (size_t)w * (size_t)h;
    e->w = w; e->h = h;
    e->elev = (int16_t*)calloc(n, 2); e->depth = (int16_t*)calloc(n, 2);
    for (int k = 0; k < 3; ++k) e->wet[k] = (uint8_t*)calloc(n, 1);
    e->f_soil = (uint8_t*)calloc(n, 1); e->f_ore = (uint8_t*)calloc(n, 1);
    e->f_sulf = (uint8_t*)calloc(n, 1); e->f_gran = (uint8_t*)calloc(n, 1);
    e->vent = (uint8_t*)calloc(n, 1); e->flow_dir = (uint8_t*)calloc(n, 1);
    e->env_bin = (uint8_t*)calloc(n, 1); e->coast = (uint16_t*)calloc(n, 2);
    if (!e->elev || !e->depth || !e->wet[0] || !e->wet[1] || !e->wet[2] || !e->f_soil || !e->f_ore ||
        !e->f_sulf || !e->f_gran || !e->vent || !e->flow_dir || !e->env_bin || !e->coast) {
        bio_env_free(e);
        bio_set_error("bio_env_alloc: out of memory");
        return -1;
    }
    return 0;
}

void bio_env_free(BioEnv* e) {
    if (!e) return;
    free(e->elev); free(e->depth);
    for (int k = 0; k < 3; ++k) { free(e->wet[k]); e->wet[k] = NULL; }
    free(e->f_soil); free(e->f_ore); free(e->f_sulf); free(e->f_gran);
    free(e->vent); free(e->flow_dir); free(e->env_bin); free(e->coast);
    e->elev = e->depth = NULL;
    e->f_soil = e->f_ore = e->f_sulf = e->f_gran = e->vent = e->flow_dir = e->env_bin = NULL;
    e->coast = NULL;
}

uint32_t bio_env_crc(const BioEnv* e) {
    if (!e || !e->elev) return 0;
    const size_t n = (size_t)e->w * (size_t)e->h;
    uint32_t c = 0;
    c = bio_crc32(c, e->elev, n * 2); c = bio_crc32(c, e->depth, n * 2);
    for (int k = 0; k < 3; ++k) c = bio_crc32(c, e->wet[k], n);
    c = bio_crc32(c, e->f_soil, n); c = bio_crc32(c, e->f_ore, n);
    c = bio_crc32(c, e->f_sulf, n); c = bio_crc32(c, e->f_gran, n);
    c = bio_crc32(c, e->vent, n); c = bio_crc32(c, e->flow_dir, n);
    c = bio_crc32(c, e->env_bin, n); c = bio_crc32(c, e->coast, n * 2);
    return c;
}

void bio_config_default(BioConfig* c) {
    memset(c, 0, sizeof *c);
    c->device = 0;
    c->cap = 0;
    c->salt = 0;              /* 0: 1 for a new run, the checkpoint's salt on load */
    c->duty = 0.5;            /* spec 5.2: 0.5 on the PC, 0.6 on the rigs */
    c->barrier_q8 = 256;
    c->atm_scale_ppm = 100;   /* 1e-4 of the real column */
    c->closed_box = 0;
}

/* ---- sim memory (device or host) --------------------------------------------- */
static void* sim_alloc(BioSim* s, size_t bytes) {
    bytes = (bytes + 255) & ~(size_t)255;
    if (s->n_mem >= (int)(sizeof s->mem / sizeof s->mem[0])) { bio_set_error("too many allocations"); return NULL; }
    void* p = NULL;
    if (s->gpu) {
        rt_dptr d = 0;
        if (rt_alloc(&d, bytes)) { bio_set_error("device alloc of %zu bytes: %s", bytes, rt_error()); return NULL; }
        if (rt_memset32(d, 0, bytes / 4)) { rt_free(d); bio_set_error("memset: %s", rt_error()); return NULL; }
        p = (void*)(uintptr_t)d;
    } else {
        p = calloc(1, bytes);
        if (!p) { bio_set_error("host alloc of %zu bytes failed", bytes); return NULL; }
    }
    s->mem[s->n_mem++] = p;
    return p;
}

int bio_copy_out(BioSim* s, void* dst, const void* src, size_t bytes) {
    if (!bytes) return 0;
    if (s->gpu) {
        if (rt_download(dst, (rt_dptr)(uintptr_t)src, bytes)) { bio_set_error("download: %s", rt_error()); return -1; }
    } else memcpy(dst, src, bytes);
    return 0;
}
int bio_copy_in(BioSim* s, void* dst, const void* src, size_t bytes) {
    if (!bytes) return 0;
    if (s->gpu) {
        if (rt_upload((rt_dptr)(uintptr_t)dst, src, bytes)) { bio_set_error("upload: %s", rt_error()); return -1; }
    } else memcpy(dst, src, bytes);
    return 0;
}
int bio_g_get(BioSim* s) { return bio_copy_out(s, &s->Gh, s->w.G, sizeof(BioGlobals)); }
int bio_g_put(BioSim* s) { return bio_copy_in(s, s->w.G, &s->Gh, sizeof(BioGlobals)); }
int bio_fields_get(BioSim* s, int64_t* dst) { return bio_copy_out(s, dst, s->w.fld, (size_t)BF_N * s->w.n_cells * 8); }
int bio_fields_put(BioSim* s, const int64_t* src) { return bio_copy_in(s, s->w.fld, src, (size_t)BF_N * s->w.n_cells * 8); }

void bio_agent_array(const BioWorld* w, int k, void** p, int* elem) {
    void* const a4[20] = { w->x, w->z, w->cell, w->C, w->S, w->E, w->N, w->Pp, w->gut, w->W,
                           w->age, w->flags, w->rbase, w->elast, w->gslot, w->uid, w->tr,
                           w->lineage, w->parent, w->bstep };
    if (k < 20) { *p = a4[k]; *elem = 4; return; }
    *elem = 8;
    if (k == 20) { *p = w->hid; return; }
    if (k == 21) { *p = w->out; return; }
    *p = w->dW + (size_t)(k - 22) * w->cap;
}

/* ---- kernels ----------------------------------------------------------------- */
static RtModule* g_mod;
static RtFunc    g_k[BK_N];
static int       g_prof = -1;
static const char* const g_kname[BK_N] = { "k_globals", "k_env", "k_summary", "k_pre", "k_supply", "k_metab",
    "k_scan_a", "k_scan_b", "k_birth", "k_free", "k_gather32", "k_gather64", "k_stats",
    "k_sort_count", "k_cscan_a", "k_cscan_b", "k_cscan_c", "k_sort_scatter", "k_sort_fix", "k_permute32", "k_permute64",
    "k_think_list", "k_think" };

static int gpu_init(BioSim* s) {
    if (rt_init(s->cfg.device, &s->dev)) { bio_set_error("CUDA init (device %d): %s", s->cfg.device, rt_error()); return -1; }
    if (!g_mod) {
        g_mod = rt_compile(BIO_KERNELS_SRC, "bio_kernels.cu");
        if (!g_mod) { bio_set_error("%s", rt_error()); return -1; }
        for (int k = 0; k < BK_N; ++k) {
            g_k[k] = rt_func(g_mod, g_kname[k]);
            if (!g_k[k]) { bio_set_error("kernel %s: %s", g_kname[k], rt_error()); return -1; }
        }
    }
    s->mod = g_mod;
    memcpy(s->k, g_k, sizeof g_k);
    return 0;
}

static unsigned blocks(uint64_t n, unsigned b) { return (unsigned)((n + b - 1) / b); }
static int s_prof_on(void) {
    if (g_prof < 0) { const char* e = getenv("GME_BIOPROF"); g_prof = (e && *e == '1') ? 1 : 0; }
    return g_prof;
}

/* optional per-stage profile (GME_BIOPROF=1): synchronises after every stage */
static double g_prof_t[BK_N + 4];
static long   g_prof_n[BK_N + 4];
static void prof_mark(int id, double t0) {
    if (s_prof_on()) { rt_sync(); g_prof_t[id] += time_now() - t0; g_prof_n[id]++; }
}
void bio_prof_report(void) {
    if (g_prof != 1) return;
    const char* nm[BK_N + 4];
    for (int k = 0; k < BK_N; ++k) nm[k] = g_kname[k] + 2;
    nm[BK_N] = "compact(all)"; nm[BK_N + 1] = "scan+dl"; nm[BK_N + 2] = "stats(cells)"; nm[BK_N + 3] = "";
    double tot = 0;
    for (int k = 0; k < BK_N + 3; ++k) tot += g_prof_t[k];
    printf("  kernels (registers / local bytes):");
    for (int k = 0; k < BK_N; ++k) {
        int regs = 0, loc = 0;
        if (g_k[k] && !rt_func_attr(g_k[k], 4, &regs) && !rt_func_attr(g_k[k], 3, &loc)) printf(" %s %d/%d", g_kname[k] + 2, regs, loc);
    }
    printf("\n  profile (ms per call, share):");
    for (int k = 0; k < BK_N + 3; ++k)
        if (g_prof_n[k]) printf(" %s %.3f (%.0f%%)", nm[k], 1e3 * g_prof_t[k] / g_prof_n[k], 100.0 * g_prof_t[k] / tot);
    printf("\n");
    memset(g_prof_t, 0, sizeof g_prof_t); memset(g_prof_n, 0, sizeof g_prof_n);
}

/* run one stage on the GPU or the CPU */
static int stage_raw(BioSim* s, int id, int arg, int arg2);
static int stage(BioSim* s, int id, int arg, int arg2) {
    const double t0 = s_prof_on() ? time_now() : 0.0;
    const int r = stage_raw(s, id, arg, arg2);
    prof_mark(id == BK_STATS && arg ? BK_N + 2 : id, t0);
    return r;
}
static int stage_raw(BioSim* s, int id, int arg, int arg2) {
    BioWorld* w = &s->w;
    if (!s->gpu) { bio_cpu_stage(w, id, arg); return 0; }
    void* a1[1] = { w };
    void* a2[2] = { w, &arg };
    void* a3[3] = { w, &arg, &arg2 };
    int r = 0;
    switch (id) {
    case BK_GLOBALS: r = rt_launch(s->k[id], 1, 32, 0, a1); break;
    case BK_ENV: case BK_SUPPLY: r = rt_launch(s->k[id], blocks((uint64_t)w->n_cells, BIO_BLOCK), BIO_BLOCK, 0, a1); break;
    case BK_SUMMARY: case BK_PRE: case BK_METAB: case BK_BIRTH: case BK_FREE:
        if (!w->n_slots) return 0;
        r = rt_launch(s->k[id], blocks(w->n_slots, BIO_BLOCK), BIO_BLOCK, 0, a1);
        break;
    case BK_SCAN_A: r = rt_launch(s->k[id], blocks(w->n_slots ? w->n_slots : 1, BIO_SCAN_BLOCK), BIO_SCAN_BLOCK, 0, a2); break;
    case BK_SCAN_B: r = rt_launch(s->k[id], 1, BIO_SCAN_BLOCK, 0, a3); break;
    case BK_THINK_LIST:
        if (!w->n_slots) return 0;
        r = rt_launch(s->k[id], blocks(w->n_slots, BIO_BLOCK), BIO_BLOCK, 0, a1);
        break;
    case BK_THINK: {   /* grid-stride over the due list (its length stays on the device) */
        if (!w->n_slots) return 0;
        unsigned g = blocks(w->n_slots, BIO_BLOCK);
        const unsigned gmax = (unsigned)(s->dev.sm_count > 0 ? s->dev.sm_count : 16) * 8u;
        if (g > gmax) g = gmax;
        r = rt_launch(s->k[id], g, BIO_BLOCK, 0, a1);
        break;
    }
    case BK_SORT_COUNT: case BK_SORT_SCATTER:
        if (!w->n_slots) return 0;
        r = rt_launch(s->k[id], blocks(w->n_slots, BIO_BLOCK), BIO_BLOCK, 0, a1);
        break;
    case BK_CSCAN_A: r = rt_launch(s->k[id], blocks((uint64_t)w->n_cells, BIO_SCAN_BLOCK), BIO_SCAN_BLOCK, 0, a1); break;
    case BK_CSCAN_B: r = rt_launch(s->k[id], 1, BIO_SCAN_BLOCK, 0, a2); break;
    case BK_CSCAN_C: case BK_SORT_FIX: r = rt_launch(s->k[id], blocks((uint64_t)w->n_cells, BIO_BLOCK), BIO_BLOCK, 0, a1); break;
    case BK_STATS: {
        const uint64_t n = arg ? (uint64_t)w->n_cells : w->n_slots;
        if (!n) return 0;
        r = rt_launch(s->k[id], blocks(n, BIO_BLOCK), BIO_BLOCK, 0, a2);
        break;
    }
    default: break;
    }
    if (r) { bio_set_error("launch %s: %s", g_kname[id], rt_error()); return -1; }
    return 0;
}

/* exclusive scan of births/deaths (mode 0) or alive (mode 1); totals out */
static int scan(BioSim* s, int mode, uint32_t tot[3]) {
    BioWorld* w = &s->w;
    const uint32_t nb = blocks(w->n_slots ? w->n_slots : 1, BIO_SCAN_BLOCK);
    if (s->gpu) {
        if (stage(s, BK_SCAN_A, mode, 0) || stage(s, BK_SCAN_B, mode, (int)nb)) return -1;
    } else bio_cpu_scan(w, mode);
    return bio_copy_out(s, tot, w->boff + (size_t)nb * 3, 12);
}

/* ---- creation --------------------------------------------------------------- */
/* island atmosphere box: column = P / (g M) (P = 1 bar, tuned) x area x scale */
static void atm_init(const BioEnv* e, int32_t scale_ppm, int64_t atm[BIO_ATMG]) {
    static const int64_t mmg[BIO_ATMG] = { 34081, 16043, 17031, 44010, 31999, 28014, 2016 };  /* mg/mol */
    int64_t ppm[BIO_ATMG], sum = 0;
    for (int g = 0; g < BIO_ATMG; ++g) { ppm[g] = e->atm_ppm[g] > 0 ? e->atm_ppm[g] : 0; sum += ppm[g]; }
    if (sum <= 0) {   /* composition unknown: the home planet's text (spec facts) plus the CO2 baseline */
        ppm[0] = 505000; ppm[1] = 386000; ppm[2] = 104000; ppm[3] = 5000; ppm[4] = 0; ppm[5] = 0; ppm[6] = 0;
        sum = 1000000;
    }
    int64_t m = 0;
    for (int g = 0; g < BIO_ATMG; ++g) m += ppm[g] * mmg[g];
    m /= sum;                                                 /* mean molar mass, mg/mol */
    if (m < 1000) m = 1000;
    const int64_t gmm = e->gravity_q8 > 0 ? (int64_t)9807 * e->gravity_q8 / 256 : 9807;   /* mm/s^2 */
    const int64_t column = (int64_t)100000000000000000LL / (gmm * m);                    /* mmol per m^2 */
    const int64_t cell_m2 = (int64_t)1 << (2 * e->shift);
    const int64_t area_scaled = (int64_t)e->w * e->h * cell_m2 * (scale_ppm > 0 ? scale_ppm : 1);   /* m^2 x ppm */
    const int64_t total = (column / 1000) * area_scaled / 1000;                          /* mmol */
    for (int g = 0; g < BIO_ATMG; ++g) atm[g] = (total / sum) * ppm[g] + ((total % sum) * ppm[g]) / sum;
}

static void totals_from(const BioSim* s, const int64_t st[BIO_STAT_N], const BioGlobals* G, int64_t tot[5]) {
    int64_t a[BIO_ATMG];
    for (int g = 0; g < BIO_ATMG; ++g) a[g] = G->atm[g] + G->acc[BACC_ATMD + g];
    (void)s;
    tot[0] = st[BS_FLD + BF_DIC] + st[BS_FLD + BF_DOC] + st[BS_FLD + BF_CH4] + st[BS_FLD + BF_DET] + st[BS_C] +
             a[3] + a[1] + G->acc[BACC_SED_C];
    tot[1] = st[BS_FLD + BF_NH4] + st[BS_N] + a[2] + 2 * a[5];
    tot[2] = st[BS_FLD + BF_H2S] + st[BS_FLD + BF_SO4] + a[0] + G->acc[BACC_SED_S];
    tot[3] = st[BS_FLD + BF_FE2] + G->acc[BACC_SED_FE];
    tot[4] = st[BS_FLD + BF_PO4] + st[BS_P];
}
static void ledger_from(const BioSim* s, const BioGlobals* G, int64_t led[5]) {
    const int64_t* a = G->acc;
    led[0] = s->init_total[0] + a[BACC_VENT_C] + a[BACC_WEATH_C] + a[BACC_IMPORT_C] + a[BACC_ATMSET_C];
    led[1] = s->init_total[1] + a[BACC_IMPORT_N] + a[BACC_ATMSET_N];
    led[2] = s->init_total[2] + a[BACC_VENT_S] + a[BACC_WEATH_S] + a[BACC_ATMSET_S];
    led[3] = s->init_total[3] + a[BACC_VENT_FE] + a[BACC_WEATH_FE];
    led[4] = s->init_total[4] + a[BACC_WEATH_P] + a[BACC_IMPORT_P];
}

static int alloc_all(BioSim* s) {
    BioWorld* w = &s->w;
    const size_t n = (size_t)w->n_cells, sl = s->slots, cap = s->cfg.cap;
#define A(field, type, count) do { w->field = (type*)sim_alloc(s, sizeof(type) * (size_t)(count)); if (!w->field) return -1; } while (0)
    A(T, BioTables, 1); A(G, BioGlobals, 1);
    A(elev, int16_t, n); A(depth, int16_t, n);
    A(wet0, uint8_t, n); A(wet1, uint8_t, n); A(wet2, uint8_t, n);
    A(f_soil, uint8_t, n); A(f_ore, uint8_t, n); A(f_sulf, uint8_t, n); A(f_gran, uint8_t, n);
    A(vent, uint8_t, n); A(flow_dir, uint8_t, n); A(coast, uint16_t, n);
    A(fld, int64_t, n * BF_N); A(fld_new, int64_t, n * BF_N);
    A(demand, uint64_t, n * BIO_POOLS); A(ratio, uint32_t, n * BIO_POOLS); A(dflag, uint32_t, n);
    A(cnt, uint32_t, n); A(lai, uint32_t, n * BIO_LAI_BANDS); A(kin, uint32_t, n * (BIO_KIN_BUCKETS / 2));
    A(x, int32_t, sl); A(z, int32_t, sl); A(cell, uint32_t, sl);
    A(C, int32_t, sl); A(S, int32_t, sl); A(E, int32_t, sl); A(N, int32_t, sl); A(Pp, int32_t, sl);
    A(gut, int32_t, sl); A(W, int32_t, sl); A(age, uint32_t, sl); A(flags, uint32_t, sl);
    A(rbase, int32_t, sl); A(elast, int32_t, sl); A(gslot, uint32_t, sl); A(uid, uint32_t, sl); A(tr, uint32_t, sl);
    A(lineage, uint32_t, sl); A(parent, uint32_t, sl); A(bstep, uint32_t, sl);
    A(hid, uint64_t, sl); A(out, uint64_t, sl); A(dW, uint64_t, sl * BIO_OUT);
    A(req, int32_t, sl * BR_N); A(pre_bd, uint32_t, sl); A(pre_a, uint32_t, sl);
    A(boff, uint32_t, (sl / BIO_SCAN_BLOCK + 2) * 3);
    A(cscan, uint32_t, n / BIO_SCAN_BLOCK + 2);
    A(gen, BioGenome, cap); A(fstack, uint32_t, cap);
    A(part, int64_t, BIO_STAT_N);
#undef A
    s->scratch = sim_alloc(s, 8 * sl);
    return s->scratch ? 0 : -1;
}

static int upload_env(BioSim* s) {
    const BioEnv* e = s->env;
    BioWorld* w = &s->w;
    const size_t n = (size_t)w->n_cells;
    if (bio_copy_in(s, (void*)w->elev, e->elev, n * 2) || bio_copy_in(s, (void*)w->depth, e->depth, n * 2) ||
        bio_copy_in(s, (void*)w->wet0, e->wet[0], n) || bio_copy_in(s, (void*)w->wet1, e->wet[1], n) ||
        bio_copy_in(s, (void*)w->wet2, e->wet[2], n) || bio_copy_in(s, (void*)w->f_soil, e->f_soil, n) ||
        bio_copy_in(s, (void*)w->f_ore, e->f_ore, n) || bio_copy_in(s, (void*)w->f_sulf, e->f_sulf, n) ||
        bio_copy_in(s, (void*)w->f_gran, e->f_gran, n) || bio_copy_in(s, (void*)w->vent, e->vent, n) ||
        bio_copy_in(s, (void*)w->flow_dir, e->flow_dir, n) || bio_copy_in(s, (void*)w->coast, e->coast, n * 2))
        return -1;
    return 0;
}

/* a host-side world view for host computations that only need params, tables
 * and globals (never agent or field pointers) */
static BioWorld host_view(BioSim* s) {
    BioWorld hw = s->w;
    hw.T = &s->tables;
    hw.G = &s->Gh;
    return hw;
}

static int init_fields(BioSim* s) {
    const BioEnv* e = s->env;
    BioWorld* w = &s->w;
    const int64_t n = w->n_cells;
    int64_t* f = (int64_t*)calloc((size_t)n * BF_N, 8);
    if (!f) { bio_set_error("out of memory"); return -1; }
    const BioGlobals* G = &s->Gh;
    for (int64_t c = 0; c < n; ++c) {
        const int water = e->depth[c] >= 1;
        const int64_t V = water ? 256 * (int64_t)bio_clampi(e->depth[c], 1, 10) : 128;
#define SAT(g) ((G->sat_q16[g] * V) >> 16)
        f[BF_DIC * n + c] = SAT(3);
        f[BF_DOC * n + c] = (water ? 20 : 20 + 200 * (int64_t)e->f_soil[c] / 255) * V;   /* tuned, mmol/m^3 */
        f[BF_NH4 * n + c] = SAT(2);
        f[BF_PO4 * n + c] = (water ? 2 : 5) * V;
        f[BF_FE2 * n + c] = (water ? 50 : 10) * V;
        f[BF_H2S * n + c] = SAT(0);
        f[BF_SO4 * n + c] = (water ? 100 : 20) * V;
        f[BF_O2 * n + c] = SAT(4);
        f[BF_CH4 * n + c] = SAT(1);
        f[BF_H2 * n + c] = SAT(6);
        f[BF_SOILW * n + c] = water ? 0 : w->P.soil_cap_l / 2;
        f[BF_DET * n + c] = 0;
#undef SAT
    }
    const int r = bio_fields_put(s, f);
    free(f);
    return r;
}

static int compute_totals(BioSim* s, int64_t tot[5], int64_t st[BIO_STAT_N]) {
    if (bio_stats_raw(s, st) < 0) return -1;
    totals_from(s, st, &s->Gh, tot);
    return 0;
}

BioSim* bio_sim_create(const BioEnv* env, const BioConfig* cfg) {
    if (!env || !cfg || env->w <= 0 || env->h <= 0 || !env->elev || !env->depth || !env->wet[0] || !env->wet[1] ||
        !env->wet[2] || !env->f_soil || !env->f_ore || !env->f_sulf || !env->f_gran || !env->vent || !env->flow_dir || !env->coast) {
        bio_set_error("bio_sim_create: incomplete environment");
        return NULL;
    }
    if (env->shift < 0 || env->shift > 8 || (int64_t)env->w * env->h > ((int64_t)1 << 24) ||
        ((int64_t)env->w << (env->shift + 8)) >= ((int64_t)1 << 31) || ((int64_t)env->h << (env->shift + 8)) >= ((int64_t)1 << 31)) {
        bio_set_error("bio_sim_create: grid %d x %d (shift %d) out of range", env->w, env->h, env->shift);
        return NULL;
    }
    BioSim* s = (BioSim*)calloc(1, sizeof *s);
    if (!s) { bio_set_error("out of memory"); return NULL; }
    s->cfg = *cfg;
    s->env = env;
    s->env_crc = bio_env_crc(env);
    s->planet_hash = env->planet_hash;
    s->gpu = cfg->device >= 0;
    if (s->gpu && gpu_init(s)) { free(s); return NULL; }
    BioWorld* w = &s->w;
    w->w = env->w; w->h = env->h; w->shift = env->shift; w->n_cells = env->w * env->h;
    w->salt = cfg->salt ? cfg->salt : 1u; w->seed = env->seed;
    s->cfg.salt = w->salt;
    w->sea_level = env->sea_level; w->t_surf_ck = env->t_surf_ck > 0 ? env->t_surf_ck : 30400;
    w->s_top = env->s_top_wm2 > 0 ? env->s_top_wm2 : 1361;
    bio_params_default(&w->P);
    w->P.barrier_q8 = cfg->barrier_q8 > 0 ? cfg->barrier_q8 : 256;
    w->P.atm_scale_ppm = cfg->atm_scale_ppm > 0 ? cfg->atm_scale_ppm : 100;
    w->P.closed_box = cfg->closed_box ? 1 : 0;
    if (env->organic_ppm > 0) w->P.weath_doc_q16 = (int32_t)((int64_t)w->P.weath_doc_q16 * env->organic_ppm / 3800);
    /* capacity: configured, or from free device memory leaving >= 1.5 GB to others */
    uint32_t cap = cfg->cap;
    if (!cap) {
        if (s->gpu) {
            size_t fr = 0, tt = 0;
            rt_mem_info(&fr, &tt);
            const int64_t cells = (int64_t)w->n_cells * 420;
            const int64_t avail = (int64_t)fr - ((int64_t)1536 << 20) - ((int64_t)128 << 20) - cells;
            int64_t c = avail > 0 ? avail / 860 : 0;
            if (c > 4194304) c = 4194304;
            cap = (uint32_t)c;
        } else cap = 262144;
        if (cap < 4096) { bio_set_error("not enough free memory for the life simulation"); free(s); return NULL; }
    }
    if (cap > 16u << 20) cap = 16u << 20;
    s->cfg.cap = cap;
    s->slots = 2 * cap;
    w->cap = s->slots;
    if (alloc_all(s) || upload_env(s)) { bio_sim_destroy(s); return NULL; }
    bio_build_tables(&s->tables);
    if (bio_copy_in(s, (void*)w->T, &s->tables, sizeof s->tables)) { bio_sim_destroy(s); return NULL; }
    /* free genome stack: pops return 0, 1, 2, ... */
    {
        uint32_t* fs = (uint32_t*)malloc((size_t)cap * 4);
        if (!fs) { bio_set_error("out of memory"); bio_sim_destroy(s); return NULL; }
        for (uint32_t i = 0; i < cap; ++i) fs[i] = cap - 1 - i;
        const int r = bio_copy_in(s, w->fstack, fs, (size_t)cap * 4);
        free(fs);
        if (r) { bio_sim_destroy(s); return NULL; }
    }
    w->top = cap; w->uid_next = 1; w->n_slots = 0; w->step = 0;
    /* atmosphere and this step's saturation values */
    memset(&s->Gh, 0, sizeof s->Gh);
    atm_init(env, w->P.atm_scale_ppm, s->Gh.atm);
    w->cap_agents = cap;
    { BioWorld hw = host_view(s); bio_globals_update(&hw); }
    if (bio_g_put(s) || init_fields(s)) { bio_sim_destroy(s); return NULL; }
    for (int32_t c = 0; c < w->n_cells; ++c) {
        if (env->wet[2][c] == 0) s->dry_cells++;
        if (env->depth[c] >= 1) s->water_cells++;
    }
    int64_t st[BIO_STAT_N];
    if (compute_totals(s, s->init_total, st)) { bio_sim_destroy(s); return NULL; }
    s->g0_since = s->g1_since = UINT64_MAX;
    return s;
}

void bio_sim_destroy(BioSim* s) {
    if (!s) return;
    for (int i = 0; i < s->n_mem; ++i) {
        if (s->gpu) rt_free((rt_dptr)(uintptr_t)s->mem[i]);
        else free(s->mem[i]);
    }
    free(s->part_host);
    free(s);
}

/* ---- agents: append / download ------------------------------------------------ */
static uint64_t rec_get(const BioAgentRec* r, int k, uint32_t cell, uint32_t gslot, uint32_t tr) {
    switch (k) {
    case 0: return (uint32_t)r->x;   case 1: return (uint32_t)r->z;   case 2: return cell;
    case 3: return (uint32_t)r->C;   case 4: return (uint32_t)r->S;   case 5: return (uint32_t)r->E;
    case 6: return (uint32_t)r->N;   case 7: return (uint32_t)r->P;   case 8: return (uint32_t)r->gut;
    case 9: return (uint32_t)r->W;   case 10: return r->age;          case 11: return r->flags;
    case 12: return (uint32_t)r->rbase; case 13: return (uint32_t)r->elast; case 14: return gslot;
    case 15: return r->uid;          case 16: return tr;              case 17: return r->lineage;
    case 18: return r->parent;       case 19: return r->bstep;        case 20: return r->hid;
    case 21: return r->out;          default: return r->dW[k - 22];
    }
}
static void rec_set(BioAgentRec* r, int k, uint64_t v) {
    const uint32_t u = (uint32_t)v;
    switch (k) {
    case 0: r->x = (int32_t)u; break;   case 1: r->z = (int32_t)u; break;   case 2: break;
    case 3: r->C = (int32_t)u; break;   case 4: r->S = (int32_t)u; break;   case 5: r->E = (int32_t)u; break;
    case 6: r->N = (int32_t)u; break;   case 7: r->P = (int32_t)u; break;   case 8: r->gut = (int32_t)u; break;
    case 9: r->W = (int32_t)u; break;   case 10: r->age = u; break;         case 11: r->flags = u; break;
    case 12: r->rbase = (int32_t)u; break; case 13: r->elast = (int32_t)u; break; case 14: break;
    case 15: r->uid = u; break;         case 16: break;                     case 17: r->lineage = u; break;
    case 18: r->parent = u; break;      case 19: r->bstep = u; break;       case 20: r->hid = v; break;
    case 21: r->out = v; break;         default: r->dW[k - 22] = v; break;
    }
}

int bio_append_agents(BioSim* s, BioAgentRec* a, uint32_t n, int fresh) {
    BioWorld* w = &s->w;
    if (!n) return 0;
    if (s->n_live + n > s->cfg.cap) { bio_set_error("agent capacity %u exceeded", s->cfg.cap); return -1; }
    if (w->n_slots + n > s->slots && bio_compact(s, 0)) return -1;
    uint32_t* gs = (uint32_t*)malloc((size_t)n * 4);
    uint32_t* cells = (uint32_t*)malloc((size_t)n * 4);
    uint32_t* trs = (uint32_t*)malloc((size_t)n * 4);
    uint8_t* tmp = (uint8_t*)malloc((size_t)n * 8);
    if (!gs || !cells || !trs || !tmp) { free(gs); free(cells); free(trs); free(tmp); bio_set_error("out of memory"); return -1; }
    int rc = -1;
    if (bio_copy_out(s, tmp, w->fstack + (w->top - n), (size_t)n * 4)) goto done;
    for (uint32_t k = 0; k < n; ++k) gs[k] = ((uint32_t*)tmp)[n - 1 - k];
    if (fresh) {
        int64_t sc = 0, sn = 0, sp = 0;
        for (uint32_t k = 0; k < n; ++k) {
            a[k].uid = w->uid_next + k;
            if (!a[k].lineage) a[k].lineage = a[k].uid;
            sc += (int64_t)a[k].C + a[k].S + a[k].gut; sn += a[k].N; sp += a[k].P;
        }
        w->uid_next += n;
        if (bio_g_get(s)) goto done;
        s->Gh.acc[BACC_IMPORT_C] += sc; s->Gh.acc[BACC_IMPORT_N] += sn; s->Gh.acc[BACC_IMPORT_P] += sp;
        if (bio_g_put(s)) goto done;
    }
    for (uint32_t k = 0; k < n; ++k) {
        a[k].x = bio_clampi(a[k].x, 0, (w->w << (w->shift + 8)) - 1);
        a[k].z = bio_clampi(a[k].z, 0, (w->h << (w->shift + 8)) - 1);
        cells[k] = (uint32_t)bio_cell_at(w, a[k].x, a[k].z);
        trs[k] = bio_traits(&a[k].g);
    }
    for (int k = 0; k < BIO_AGENT_ARRAYS; ++k) {
        void* p; int el;
        bio_agent_array(w, k, &p, &el);
        for (uint32_t i = 0; i < n; ++i) {
            const uint64_t v = rec_get(&a[i], k, cells[i], gs[i], trs[i]);
            if (el == 4) ((uint32_t*)tmp)[i] = (uint32_t)v; else ((uint64_t*)tmp)[i] = v;
        }
        if (bio_copy_in(s, (uint8_t*)p + (size_t)w->n_slots * el, tmp, (size_t)n * el)) goto done;
    }
    /* genomes, uploaded in runs of consecutive slots */
    for (uint32_t k = 0; k < n;) {
        uint32_t e = k + 1;
        while (e < n && gs[e] == gs[e - 1] + 1) ++e;
        BioGenome* buf = (BioGenome*)malloc(sizeof(BioGenome) * (e - k));
        if (!buf) { bio_set_error("out of memory"); goto done; }
        for (uint32_t i = k; i < e; ++i) buf[i - k] = a[i].g;
        const int r = bio_copy_in(s, &w->gen[gs[k]], buf, sizeof(BioGenome) * (e - k));
        free(buf);
        if (r) goto done;
        k = e;
    }
    w->n_slots += n;
    w->top -= n;
    s->n_live += n;
    rc = 0;
done:
    free(gs); free(cells); free(trs); free(tmp);
    return rc;
}

int bio_download_agents(BioSim* s, BioAgentRec** out, uint32_t* nout) {
    BioWorld* w = &s->w;
    *out = NULL; *nout = 0;
    const uint32_t ns = w->n_slots;
    uint32_t* flags = (uint32_t*)malloc((size_t)(ns ? ns : 1) * 4);
    uint32_t* gsl = (uint32_t*)malloc((size_t)(ns ? ns : 1) * 4);
    uint8_t* tmp = (uint8_t*)malloc((size_t)(ns ? ns : 1) * 8);
    BioAgentRec* r = (BioAgentRec*)calloc((size_t)(s->n_live ? s->n_live : 1), sizeof(BioAgentRec));
    BioGenome* pool = NULL;
    int rc = -1;
    if (!flags || !gsl || !tmp || !r) { bio_set_error("out of memory"); goto done; }
    if (bio_copy_out(s, flags, w->flags, (size_t)ns * 4) || bio_copy_out(s, gsl, w->gslot, (size_t)ns * 4)) goto done;
    uint32_t nl = 0;
    for (uint32_t i = 0; i < ns; ++i) if (flags[i] & BA_ALIVE) ++nl;
    if (nl != s->n_live) { bio_set_error("live count mismatch (%u vs %u)", nl, s->n_live); goto done; }
    for (int k = 0; k < BIO_AGENT_ARRAYS; ++k) {
        void* p; int el;
        bio_agent_array(w, k, &p, &el);
        if (bio_copy_out(s, tmp, p, (size_t)ns * el)) goto done;
        uint32_t j = 0;
        for (uint32_t i = 0; i < ns; ++i) {
            if (!(flags[i] & BA_ALIVE)) continue;
            rec_set(&r[j++], k, el == 4 ? ((uint32_t*)tmp)[i] : ((uint64_t*)tmp)[i]);
        }
    }
    /* genomes: stream the slot pool in chunks (host memory stays bounded) */
    {
        const uint32_t chunk = 1u << 16;                       /* 65536 genomes = 24 MB */
        pool = (BioGenome*)malloc(sizeof(BioGenome) * chunk);
        uint32_t* own = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)(nl ? nl : 1));
        if (!pool || !own) { free(own); bio_set_error("out of memory"); goto done; }
        uint32_t j = 0, gmax = 0;
        for (uint32_t i = 0; i < ns; ++i) if (flags[i] & BA_ALIVE) { own[j++] = gsl[i]; if (gsl[i] > gmax) gmax = gsl[i]; }
        for (uint32_t base = 0; nl && base <= gmax; base += chunk) {
            const uint32_t cnt = gmax - base + 1 < chunk ? gmax - base + 1 : chunk;
            if (bio_copy_out(s, pool, w->gen + base, sizeof(BioGenome) * cnt)) { free(own); goto done; }
            for (uint32_t k = 0; k < nl; ++k)
                if (own[k] >= base && own[k] < base + cnt) r[k].g = pool[own[k] - base];
        }
        free(own);
    }
    *out = r; *nout = nl; r = NULL;
    rc = 0;
done:
    free(flags); free(gsl); free(tmp); free(r); free(pool);
    return rc;
}

/* ---- compaction: alive agents to the front, order kept ------------------------- */
static int compact_sorted(BioSim* s) {
    BioWorld* w = &s->w;
    uint32_t* perm = (uint32_t*)w->req;           /* req is per-step scratch */
    const uint32_t n = s->n_live;
    if (s->gpu) {
        const int nb = (int)blocks((uint64_t)w->n_cells, BIO_SCAN_BLOCK);
        if (rt_memset32((rt_dptr)(uintptr_t)w->cnt, 0, (size_t)w->n_cells)) { bio_set_error("memset: %s", rt_error()); return -1; }
        if (stage(s, BK_SORT_COUNT, 0, 0) || stage(s, BK_CSCAN_A, 0, 0) || stage(s, BK_CSCAN_B, nb, 0) ||
            stage(s, BK_CSCAN_C, 0, 0) || stage(s, BK_SORT_SCATTER, 0, 0) || stage(s, BK_SORT_FIX, 0, 0)) return -1;
    } else bio_cpu_sort_perm(w, perm);
    for (int k = 0; k < BIO_AGENT_ARRAYS; ++k) {
        void* p; int el;
        bio_agent_array(w, k, &p, &el);
        if (s->gpu) {
            void* src = p; void* dst = s->scratch; void* pm = perm; uint32_t nn = n;
            void* args[4] = { &src, &dst, &pm, &nn };
            if (n && rt_launch(s->k[el == 4 ? BK_PERMUTE32 : BK_PERMUTE64], blocks(n, BIO_BLOCK), BIO_BLOCK, 0, args)) {
                bio_set_error("permute: %s", rt_error()); return -1;
            }
            if (n && rt_copy((rt_dptr)(uintptr_t)p, (rt_dptr)(uintptr_t)s->scratch, (size_t)n * el)) {
                bio_set_error("copy: %s", rt_error()); return -1;
            }
        } else {
            for (uint32_t j = 0; j < n; ++j) {
                if (el == 4) ((uint32_t*)s->scratch)[j] = ((const uint32_t*)p)[perm[j]];
                else ((uint64_t*)s->scratch)[j] = ((const uint64_t*)p)[perm[j]];
            }
            memcpy(p, s->scratch, (size_t)n * el);
        }
    }
    w->n_slots = n;
    return 0;
}

int bio_compact(BioSim* s, int sort) {
    BioWorld* w = &s->w;
    if (sort) return compact_sorted(s);
    uint32_t tot[3];
    if (scan(s, 1, tot)) return -1;
    if (tot[2] != s->n_live) { bio_set_error("compact: live %u vs %u", tot[2], s->n_live); return -1; }
    static const int order[BIO_AGENT_ARRAYS] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19,
                                                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 11 };   /* flags last */
    for (int oi = 0; oi < BIO_AGENT_ARRAYS; ++oi) {
        void* p; int el;
        bio_agent_array(w, order[oi], &p, &el);
        if (s->gpu) {
            uint32_t n = w->n_slots;
            void* src = p; void* dst = s->scratch;
            void* fl = w->flags; void* pa = w->pre_a; void* bo = w->boff;
            void* args[6] = { &src, &dst, &fl, &pa, &bo, &n };
            if (n && rt_launch(s->k[el == 4 ? BK_GATHER32 : BK_GATHER64], blocks(n, BIO_BLOCK), BIO_BLOCK, 0, args)) {
                bio_set_error("gather: %s", rt_error()); return -1;
            }
            if (s->n_live && rt_copy((rt_dptr)(uintptr_t)p, (rt_dptr)(uintptr_t)s->scratch, (size_t)s->n_live * el)) {
                bio_set_error("copy: %s", rt_error()); return -1;
            }
        } else {
            bio_cpu_gather(w, p, s->scratch, el, w->n_slots);
            memcpy(p, s->scratch, (size_t)s->n_live * el);
        }
    }
    w->n_slots = s->n_live;
    return 0;
}

/* ---- one step (spec 5.2, fused: env, summary, pre, supply, metab, scan, free, birth) */
static int sim_step(BioSim* s) {
    BioWorld* w = &s->w;
    const uint32_t cap = s->cfg.cap;
    const uint32_t holes = w->n_slots - s->n_live;
    /* every 64 steps: compact and order by cell (deterministic schedule, so the
     * agent order never depends on when holes appeared); otherwise a stable
     * compaction when holes pile up (relative order unchanged) */
    const int sort = (w->step % 64u) == 0 && s->n_live > 0;
    if (sort || w->n_slots > cap || (uint64_t)holes * 10 > w->n_slots) {
        const double t0 = s_prof_on() ? time_now() : 0.0;
        if (bio_compact(s, sort)) return -1;
        prof_mark(BK_N, t0);
    }
    w->n_live = s->n_live;
    w->cap_agents = cap;
    if (stage(s, BK_GLOBALS, 0, 0) || stage(s, BK_ENV, 0, 0)) return -1;
    { int64_t* t = w->fld; w->fld = w->fld_new; w->fld_new = t; }
    if (stage(s, BK_SUMMARY, 0, 0) || stage(s, BK_THINK_LIST, 0, 0) || stage(s, BK_THINK, 0, 0) ||
        stage(s, BK_PRE, 0, 0) || stage(s, BK_SUPPLY, 0, 0) || stage(s, BK_METAB, 0, 0)) return -1;
    uint32_t tot[3];
    const double t0 = s_prof_on() ? time_now() : 0.0;
    if (scan(s, 0, tot)) return -1;
    prof_mark(BK_N + 1, t0);
    const uint32_t B = tot[0], D = tot[1];
    if (D > s->n_live) { bio_set_error("step: %u deaths of %u live", D, s->n_live); return -1; }
    /* capacity guard: births refused above 0.95 cap (depends on live counts only) */
    const int64_t room = (int64_t)cap * 95 / 100 - (int64_t)(s->n_live - D);
    const uint32_t K = (uint32_t)(room <= 0 ? 0 : (B < room ? B : room));
    if (stage(s, BK_FREE, 0, 0)) return -1;
    w->top += D;
    w->k_births = K;
    if (stage(s, BK_BIRTH, 0, 0)) return -1;
    w->top -= K;
    w->n_slots += K;
    w->uid_next += K;
    w->k_births = 0;
    s->n_live = s->n_live - D + K;
    s->births += K; s->deaths += D;
    s->last_scan_tot[0] = B; s->last_scan_tot[1] = D; s->last_scan_tot[2] = K;
    w->step++;
    return 0;
}

/* ---- statistics --------------------------------------------------------------- */
int64_t bio_stats_raw(BioSim* s, int64_t st[BIO_STAT_N]) {
    BioWorld* w = &s->w;
    memset(st, 0, sizeof(int64_t) * BIO_STAT_N);
    if (s->gpu) {
        if (rt_memset32((rt_dptr)(uintptr_t)w->part, 0, BIO_STAT_N * 2)) { bio_set_error("memset: %s", rt_error()); return -1; }
        if (stage(s, BK_STATS, 0, 0) || stage(s, BK_STATS, 1, 0)) return -1;
        if (bio_copy_out(s, st, w->part, sizeof(int64_t) * BIO_STAT_N)) return -1;
    } else bio_cpu_stats(w, st);
    if (bio_g_get(s)) return -1;
    return st[BS_LIVE];
}

static void update_gates(BioSim* s, const int64_t st[BIO_STAT_N]) {
    const uint64_t step = s->w.step;
    const BioGlobals* G = &s->Gh;
    /* G0: live > 1e5 for one sim-year */
    if (st[BS_LIVE] > 100000) {
        if (s->g0_since == UINT64_MAX) s->g0_since = step;
        if (step - s->g0_since >= 8760) s->gates |= 1u << BIO_G0_ESTABLISHED;
    } else s->g0_since = UINT64_MAX;
    /* G1: light pathways >= 50% of fixation for 5 sim-years; G2: OXY >= 10% and O2 > 1 uM somewhere */
    const int64_t photo = G->acc[BACC_PATH + BP_ANOX] + G->acc[BACC_PATH + BP_OXY];
    const int64_t fix = G->acc[BACC_FIX];
    if (step > s->g1_prev_step) {
        const int64_t dp = photo - s->g1_prev_photo, df = fix - s->g1_prev_fix;
        if (df > 0 && 2 * dp >= df) {
            if (s->g1_since == UINT64_MAX) s->g1_since = s->g1_prev_step;
            if (step - s->g1_since >= 5 * 8760) s->gates |= 1u << BIO_G1_MARINE_PHOTO;
        } else s->g1_since = UINT64_MAX;
        const int64_t doxy = G->acc[BACC_PATH + BP_OXY] - s->g2_prev_oxy;
        if (df > 0 && doxy > 0 && 10 * doxy >= df && st[BS_O2CELLS] > 0) s->gates |= 1u << BIO_G2_OXYGENIC;
    }
    s->g1_prev_photo = photo; s->g1_prev_fix = fix; s->g1_prev_step = step;
    s->g2_prev_oxy = G->acc[BACC_PATH + BP_OXY];
    /* G3: > 1% of dry land cells vegetated */
    if (s->dry_cells && (uint64_t)st[BS_VEG] * 100 > s->dry_cells) s->gates |= 1u << BIO_G3_LAND_PLANTS;
    /* G5: O2 > 1% of the island box */
    int64_t tot = 0;
    for (int g = 0; g < BIO_ATMG; ++g) tot += G->atm[g] > 0 ? G->atm[g] : 0;
    if (tot > 0 && G->atm[4] * 100 > tot) s->gates |= 1u << BIO_G5_OXIDATION;
}

int bio_sim_run(BioSim* s, uint64_t steps) {
    if (!s) return -1;
    const double t0 = time_now();
    double tb = t0;
    const double duty = s->cfg.duty > 0.0 && s->cfg.duty < 1.0 ? s->cfg.duty : 1.0;
    for (uint64_t k = 0; k < steps; ++k) {
        if (sim_step(s)) return -1;
        if (s->w.step % 24u == 0) {
            int64_t st[BIO_STAT_N];
            if (bio_stats_raw(s, st) < 0) return -1;
            update_gates(s, st);
            if (duty < 1.0) {
                const double busy = time_now() - tb;
                const int ms = (int)(busy * (1.0 - duty) / duty * 1000.0);
                if (ms > 0) sleep_ms(ms);
            }
            tb = time_now();
        }
    }
    const double dt = time_now() - t0;
    s->steps_per_s = dt > 0 ? (double)steps / dt : 0.0;
    return 0;
}

int bio_sim_stats(BioSim* s, BioStats* o) {
    if (!s || !o) return -1;
    int64_t st[BIO_STAT_N];
    if (bio_stats_raw(s, st) < 0) return -1;
    memset(o, 0, sizeof *o);
    const BioGlobals* G = &s->Gh;
    o->step = s->w.step;
    o->sim_years = (double)s->w.step / 8760.0;
    o->live = (uint64_t)st[BS_LIVE];
    o->motile = (uint64_t)st[BS_MOTILE];
    o->sessile_photo = (uint64_t)st[BS_SPHOTO];
    o->births = s->births; o->deaths = s->deaths;
    o->carbon_marine = st[BS_CMAR]; o->carbon_coastal = st[BS_CCOAST]; o->carbon_land = st[BS_CLAND];
    for (int k = 0; k < BIO_PATHWAYS; ++k) o->path_flux[k] = G->acc[BACC_PATH + k];
    for (int g = 0; g < BIO_ATM_N; ++g) o->atm[g] = G->atm[g] + G->acc[BACC_ATMD + g];
    int64_t tot[5], led[5];
    totals_from(s, st, G, tot);
    ledger_from(s, G, led);
    for (int k = 0; k < 4; ++k) { o->total[k] = tot[k]; o->ledger[k] = led[k]; }
    for (int k = 0; k < 64; ++k) o->mean_gene[k] = st[BS_LIVE] ? (uint32_t)(st[BS_GENE + k] / st[BS_LIVE]) : 0;
    o->mean_abs_dw = st[BS_LIVE] ? (double)st[BS_ABSDW] / (64.0 * (double)st[BS_LIVE]) : 0.0;
    o->think_share = st[BS_LIVE] ? (double)st[BS_THINK] / (double)st[BS_LIVE] : 0.0;
    o->land_cells_vegetated = (uint32_t)st[BS_VEG];
    o->gates = s->gates;
    o->steps_per_s = s->steps_per_s;
    return 0;
}

/* ---- seeding (spec 4) ---------------------------------------------------------- */
void bio_protocell(BioGenome* g, uint32_t salt) {
    memset(g, 0, sizeof *g);
    uint8_t* b = g->body;
    b[BG_E_METH] = 160; b[BG_E_FERM] = 40;
    b[BG_SIZE] = 20;          /* m_r = 4 mmol C */
    b[BG_HOLDFAST] = 96;      /* drifts (< 128) */
    b[BG_HUE] = 200;
    b[BG_REPRO_E] = 64; b[BG_OFFS] = 128; b[BG_DISPERSE] = 16;
    b[BG_MUT] = 64;           /* ~1/256 per byte */
    b[BG_DECAY] = 16;
    b[BG_NEURONS] = 0x0F;
    b[BG_THINK] = 124;        /* period 32 */
    b[BG_TOPT] = 120;         /* 303 K */
    b[BG_TTOL] = 160;         /* 11 K */
    b[BG_LIFESPAN] = 112;     /* 128 days */
    for (int k = BG_NEUTRAL0; k < 64; ++k) b[k] = (uint8_t)bio_hash(k, 0, BH_SEED, salt);
    /* random brain within +-4 */
    int8_t* br = &g->w_ih[0][0];
    for (int k = 0; k < BIO_MUT_BYTES - 64; ++k) br[k] = (int8_t)((int32_t)(bio_hash(k, 1, BH_SEED, salt) % 9u) - 4);
    g->b_o[3] = 16;           /* reproduce gate open by default */
    g->b_o[4] = 0;
    g->mate_tag = (uint16_t)bio_hash(0, 2, BH_SEED, salt);
}

static int cmp_i64(const void* a, const void* b) {
    const int64_t x = *(const int64_t*)a, y = *(const int64_t*)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

int bio_sim_seed(BioSim* s, uint32_t n) {
    if (!s) return -1;
    if (!n) return 0;
    BioWorld* w = &s->w;
    const BioEnv* e = s->env;
    const int64_t nc = w->n_cells;
    int64_t* f = (int64_t*)malloc((size_t)nc * BF_N * 8);
    int32_t* cand = (int32_t*)malloc((size_t)nc * 4);
    int64_t* val = (int64_t*)malloc((size_t)nc * 8);
    BioAgentRec* a = (BioAgentRec*)calloc(n, sizeof(BioAgentRec));
    int rc = -1;
    if (!f || !cand || !val || !a) { bio_set_error("out of memory"); goto done; }
    if (bio_fields_get(s, f)) goto done;
    {
        BioWorld hw = host_view(s);
        hw.elev = e->elev; hw.depth = e->depth; hw.vent = e->vent;
        int32_t nv = 0, nshelf = 0;
        for (int32_t c = 0; c < nc; ++c) {
            if (e->vent[c] > 0) { cand[nv++] = c; continue; }
        }
        int32_t m = 0;
        for (int32_t c = 0; c < nc; ++c) {
            if (e->vent[c] > 0 || e->depth[c] < 2 || e->depth[c] > 60) continue;
            const int32_t T = bio_temp(&hw, c, 0);
            if (T < 27300 || T > 33000) continue;
            val[m++] = (f[BF_H2 * nc + c] + f[BF_H2S * nc + c]) / bio_volume(&hw, c);   /* concentration */
        }
        int64_t med = 0;
        if (m) { qsort(val, (size_t)m, 8, cmp_i64); med = val[m / 2]; }
        for (int32_t c = 0; c < nc; ++c) {
            if (e->vent[c] > 0 || e->depth[c] < 2 || e->depth[c] > 60) continue;
            const int32_t T = bio_temp(&hw, c, 0);
            if (T < 27300 || T > 33000) continue;
            const int64_t v = (f[BF_H2 * nc + c] + f[BF_H2S * nc + c]) / bio_volume(&hw, c);
            if (v >= med) cand[nv + nshelf++] = c;     /* at or above the median (ties included) */
        }
        const int32_t ncand = nv + nshelf;
        if (!ncand) { bio_set_error("seed: no habitable cells (no vents and no 2-60 m shelf)"); goto done; }
        BioGenome g;
        bio_protocell(&g, w->salt);
        const uint64_t ob = bio_out_bias(&hw, &g);
        const int32_t mr = bio_mr(&hw, g.body[BG_SIZE]);
        const int32_t cell_q8 = 1 << (w->shift + 8);
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t h = bio_hash((int32_t)k, (int32_t)w->step, BH_SEED, w->salt);
            const uint32_t h2 = bio_hash((int32_t)k, (int32_t)w->step, BH_SEED + 32, w->salt);
            const int32_t c = cand[h % (uint32_t)ncand];
            BioAgentRec* r = &a[k];
            r->x = (c % w->w) * cell_q8 + (int32_t)(h2 & (uint32_t)(cell_q8 - 1));
            r->z = (c / w->w) * cell_q8 + (int32_t)((h2 >> 16) & (uint32_t)(cell_q8 - 1));
            r->C = mr; r->S = 0; r->E = 16; r->N = (mr + w->P.cn - 1) / w->P.cn; r->P = (mr + w->P.cp - 1) / w->P.cp;
            r->gut = 0; r->W = 256; r->age = 0;
            r->flags = BA_ALIVE | (e->depth[c] >= 1 ? BA_WET : BA_LAND);
            r->rbase = 0; r->elast = r->E;
            r->lineage = 0; r->parent = 0; r->bstep = (uint32_t)w->step;
            r->hid = 0; r->out = ob;
            r->g = g;
        }
    }
    rc = bio_append_agents(s, a, n, 1);
done:
    free(f); free(cand); free(val); free(a);
    return rc;
}

/* ---- state hash (canonical: independent of holes and genome slots) -------------- */
static uint64_t hash_rec(uint64_t h, const BioAgentRec* r) {
    const int32_t v[17] = { r->x, r->z, r->C, r->S, r->E, r->N, r->P, r->gut, r->W, (int32_t)r->age,
                            (int32_t)(r->flags & (BA_ALIVE | BA_WET | BA_LAND)), r->rbase, r->elast,
                            (int32_t)r->uid, (int32_t)r->lineage, (int32_t)r->parent, (int32_t)r->bstep };
    h = bio_fnv64(h, v, sizeof v);
    h = bio_fnv64(h, &r->hid, 8);
    h = bio_fnv64(h, &r->out, 8);
    h = bio_fnv64(h, r->dW, sizeof r->dW);
    return bio_fnv64(h, &r->g, sizeof r->g);
}

uint64_t bio_sim_hash(BioSim* s) {
    if (!s) return 0;
    BioWorld* w = &s->w;
    uint64_t h = BIO_FNV_INIT;
    if (bio_g_get(s)) return 0;
    const uint64_t hdr[4] = { w->step, s->n_live, w->uid_next, (uint64_t)w->salt };
    h = bio_fnv64(h, hdr, sizeof hdr);
    h = bio_fnv64(h, s->Gh.atm, sizeof s->Gh.atm);
    h = bio_fnv64(h, s->Gh.acc, sizeof s->Gh.acc);
    int64_t* f = (int64_t*)malloc((size_t)w->n_cells * BF_N * 8);
    if (!f) return 0;
    if (bio_fields_get(s, f)) { free(f); return 0; }
    h = bio_fnv64(h, f, (size_t)w->n_cells * BF_N * 8);
    free(f);
    BioAgentRec* a = NULL;
    uint32_t n = 0;
    if (bio_download_agents(s, &a, &n)) return 0;
    for (uint32_t i = 0; i < n; ++i) h = hash_rec(h, &a[i]);
    free(a);
    return h;
}

/* ---- atmosphere exchange with the hub ------------------------------------------ */
int bio_sim_get_atm(BioSim* s, int64_t atm[BIO_ATM_N]) {
    if (!s || bio_g_get(s)) return -1;
    for (int g = 0; g < BIO_ATM_N; ++g) atm[g] = s->Gh.atm[g] + s->Gh.acc[BACC_ATMD + g];
    return 0;
}

int bio_sim_set_atm(BioSim* s, const int64_t atm[BIO_ATM_N]) {
    if (!s) return -1;
    for (int g = 0; g < BIO_ATM_N; ++g) if (atm[g] < 0) { bio_set_error("set_atm: negative gas %d", g); return -1; }
    if (bio_g_get(s)) return -1;
    int64_t d[BIO_ATM_N];
    for (int g = 0; g < BIO_ATM_N; ++g) {
        d[g] = atm[g] - (s->Gh.atm[g] + s->Gh.acc[BACC_ATMD + g]);
        s->Gh.atm[g] += d[g];
    }
    s->Gh.acc[BACC_ATMSET_C] += d[BIO_ATM_CO2] + d[BIO_ATM_CH4];
    s->Gh.acc[BACC_ATMSET_N] += d[BIO_ATM_NH3] + 2 * d[BIO_ATM_N2];
    s->Gh.acc[BACC_ATMSET_S] += d[BIO_ATM_H2S];
    return bio_g_put(s);
}

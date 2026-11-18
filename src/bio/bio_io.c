/* bio_io.c - checkpoints (GMEBIO02, spec 6.3), vegetation export for the
 * biomap (spec 7.2) and migrant files (spec 6.4). Host side only. */
#include "bio_internal.h"
#include "../biomap.h"
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#define CKPT_MAGIC "GMEBIO02"
#define CKPT_VERSION 2u
#define CKPT_HDR 512

/* ---- little-endian helpers ---------------------------------------------------- */
static void put32(uint8_t* p, uint32_t v) { for (int k = 0; k < 4; ++k) p[k] = (uint8_t)(v >> (8 * k)); }
static void put64(uint8_t* p, uint64_t v) { for (int k = 0; k < 8; ++k) p[k] = (uint8_t)(v >> (8 * k)); }
static uint32_t get32(const uint8_t* p) { uint32_t v = 0; for (int k = 0; k < 4; ++k) v |= (uint32_t)p[k] << (8 * k); return v; }
static uint64_t get64(const uint8_t* p) { uint64_t v = 0; for (int k = 0; k < 8; ++k) v |= (uint64_t)p[k] << (8 * k); return v; }

/* atomic replace: write tmp, flush to disk, rename over the target */
static int commit_file(FILE* f, const char* tmp, const char* path) {
    if (fflush(f)) { fclose(f); remove(tmp); bio_set_error("write %s failed", tmp); return -1; }
#ifdef _WIN32
    _commit(_fileno(f));
#else
    fsync(fileno(f));
#endif
    if (fclose(f)) { remove(tmp); bio_set_error("close %s failed", tmp); return -1; }
#ifdef _WIN32
    if (!MoveFileExA(tmp, path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        remove(tmp); bio_set_error("rename %s -> %s failed", tmp, path); return -1;
    }
#else
    if (rename(tmp, path)) { remove(tmp); bio_set_error("rename %s -> %s failed", tmp, path); return -1; }
#endif
    return 0;
}

static int write_section(FILE* f, const char tag[4], const void* p, uint64_t n) {
    uint8_t h[16];
    memcpy(h, tag, 4);
    put64(h + 4, n);
    put32(h + 12, bio_crc32(0, p, (size_t)n));
    if (fwrite(h, 1, 16, f) != 16) return -1;
    if (n && fwrite(p, 1, (size_t)n, f) != (size_t)n) return -1;
    return 0;
}

/* reads the next section; returns malloc'd payload (caller frees) */
static void* read_section(FILE* f, char tag[5], uint64_t* n) {
    uint8_t h[16];
    if (fread(h, 1, 16, f) != 16) return NULL;
    memcpy(tag, h, 4); tag[4] = 0;
    *n = get64(h + 4);
    if (*n > ((uint64_t)1 << 40)) return NULL;
    void* p = malloc(*n ? (size_t)*n : 1);
    if (!p) return NULL;
    if (*n && fread(p, 1, (size_t)*n, f) != (size_t)*n) { free(p); return NULL; }
    if (bio_crc32(0, p, (size_t)*n) != get32(h + 12)) { free(p); return NULL; }
    return p;
}

/* ---- checkpoint ----------------------------------------------------------------- */
typedef struct { uint64_t h; uint32_t idx; } GKey;
static int gkey_cmp(const void* a, const void* b) {
    const GKey* x = (const GKey*)a; const GKey* y = (const GKey*)b;
    if (x->h != y->h) return x->h < y->h ? -1 : 1;
    return x->idx < y->idx ? -1 : (x->idx > y->idx ? 1 : 0);
}

int bio_sim_save(BioSim* s, const char* path) {
    if (!s || !path) return -1;
    BioWorld* w = &s->w;
    BioAgentRec* a = NULL;
    uint32_t n = 0;
    int rc = -1;
    int64_t* fl = NULL;
    uint8_t* hot = NULL; uint8_t* cold = NULL;
    BioGenome* gt = NULL; uint32_t* gidx = NULL; GKey* keys = NULL;
    FILE* f = NULL;
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    if (bio_download_agents(s, &a, &n) || bio_g_get(s)) goto done;
    fl = (int64_t*)malloc((size_t)w->n_cells * BF_N * 8);
    hot = (uint8_t*)malloc((size_t)(n ? n : 1) * 72);
    cold = (uint8_t*)malloc((size_t)(n ? n : 1) * 76);
    gt = (BioGenome*)malloc(sizeof(BioGenome) * (n ? n : 1));
    gidx = (uint32_t*)malloc((size_t)(n ? n : 1) * 4);
    keys = (GKey*)malloc(sizeof(GKey) * (n ? n : 1));
    if (!fl || !hot || !cold || !gt || !gidx || !keys) { bio_set_error("out of memory"); goto done; }
    if (bio_fields_get(s, fl)) goto done;
    /* agents */
    for (uint32_t i = 0; i < n; ++i) {
        const BioAgentRec* r = &a[i];
        uint8_t* p = hot + (size_t)i * 72;
        const uint32_t v[14] = { (uint32_t)r->x, (uint32_t)r->z, (uint32_t)r->C, (uint32_t)r->S, (uint32_t)r->E,
                                 (uint32_t)r->N, (uint32_t)r->P, (uint32_t)r->gut, (uint32_t)r->W, r->age, r->flags,
                                 (uint32_t)r->rbase, (uint32_t)r->elast, r->uid };
        for (int k = 0; k < 14; ++k) put32(p + 4 * k, v[k]);
        put64(p + 56, r->hid); put64(p + 64, r->out);
        uint8_t* q = cold + (size_t)i * 76;
        for (int k = 0; k < BIO_OUT; ++k) put64(q + 8 * k, r->dW[k]);
        put32(q + 64, r->lineage); put32(q + 68, r->parent); put32(q + 72, r->bstep);
        keys[i].h = bio_fnv64(BIO_FNV_INIT, &r->g, sizeof r->g);
        keys[i].idx = i;
    }
    /* genome table: deduplicated by FNV-64 + memcmp, first occurrence order */
    qsort(keys, n, sizeof(GKey), gkey_cmp);
    uint32_t* first = (uint32_t*)malloc(sizeof(uint32_t) * (n ? n : 1));     /* agent -> representative agent */
    if (!first) { bio_set_error("out of memory"); goto done; }
    for (uint32_t k = 0; k < n;) {
        uint32_t e = k;
        while (e < n && keys[e].h == keys[k].h) ++e;
        for (uint32_t i = k; i < e; ++i) {
            first[keys[i].idx] = keys[i].idx;
            for (uint32_t j = k; j < i; ++j)
                if (first[keys[j].idx] == keys[j].idx && !memcmp(&a[keys[j].idx].g, &a[keys[i].idx].g, sizeof(BioGenome))) {
                    first[keys[i].idx] = keys[j].idx; break;
                }
        }
        k = e;
    }
    uint32_t ng = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (first[i] == i) { gidx[i] = ng; gt[ng++] = a[i].g; }
        else gidx[i] = gidx[first[i]];
    }
    free(first);
    /* header */
    uint8_t hdr[CKPT_HDR];
    memset(hdr, 0, sizeof hdr);
    memcpy(hdr, CKPT_MAGIC, 8);
    put32(hdr + 8, CKPT_VERSION); put32(hdr + 12, CKPT_HDR);
    put64(hdr + 16, s->planet_hash); put32(hdr + 24, s->env_crc); put32(hdr + 28, (uint32_t)s->env->island);
    put32(hdr + 32, w->salt); put32(hdr + 36, s->cfg.cap); put64(hdr + 40, w->step);
    put32(hdr + 48, n); put32(hdr + 52, w->uid_next);
    put32(hdr + 56, (uint32_t)w->w); put32(hdr + 60, (uint32_t)w->h); put32(hdr + 64, (uint32_t)w->shift);
    put32(hdr + 68, (uint32_t)w->P.atm_scale_ppm); put32(hdr + 72, (uint32_t)w->P.barrier_q8); put32(hdr + 76, (uint32_t)w->P.closed_box);
    for (int g = 0; g < BIO_ATMG; ++g) put64(hdr + 80 + 8 * g, (uint64_t)(s->Gh.atm[g] + s->Gh.acc[BACC_ATMD + g]));
    put64(hdr + 136, (uint64_t)s->Gh.acc[BACC_SED_C]); put64(hdr + 144, (uint64_t)s->Gh.acc[BACC_SED_S]);
    put64(hdr + 152, (uint64_t)s->Gh.acc[BACC_SED_FE]); put64(hdr + 160, (uint64_t)s->Gh.acc[BACC_IMPORT_C]);
    put64(hdr + 168, (uint64_t)s->Gh.acc[BACC_VENT_C]);
    put64(hdr + 176, s->births); put64(hdr + 184, s->deaths);
    put32(hdr + 192, s->gates); put32(hdr + 196, 0);   /* last hub epoch applied (hub: M3) */
    for (int k = 0; k < 5; ++k) put64(hdr + 200 + 8 * k, (uint64_t)s->init_total[k]);
    put32(hdr + 240, s->env->seed);
    put32(hdr + 508, bio_crc32(0, hdr, 508));
    /* sim state */
    int64_t sims[16];
    memset(sims, 0, sizeof sims);
    sims[0] = (int64_t)s->g0_since; sims[1] = (int64_t)s->g1_since; sims[2] = s->g1_prev_photo; sims[3] = s->g1_prev_fix;
    sims[4] = (int64_t)s->g1_prev_step; sims[5] = s->g2_prev_oxy; sims[6] = s->gates;
    sims[7] = (int64_t)s->births; sims[8] = (int64_t)s->deaths;
    for (int k = 0; k < 5; ++k) sims[9 + k] = s->init_total[k];
    f = fopen(tmp, "wb");
    if (!f) { bio_set_error("cannot write %s", tmp); goto done; }
    if (fwrite(hdr, 1, CKPT_HDR, f) != CKPT_HDR ||
        write_section(f, "PARM", &w->P, sizeof w->P) ||
        write_section(f, "GLOB", &s->Gh, sizeof s->Gh) ||
        write_section(f, "SIMS", sims, sizeof sims) ||
        write_section(f, "FLDS", fl, (uint64_t)w->n_cells * BF_N * 8) ||
        write_section(f, "HOT_", hot, (uint64_t)n * 72) ||
        write_section(f, "COLD", cold, (uint64_t)n * 76) ||
        write_section(f, "GTAB", gt, (uint64_t)ng * sizeof(BioGenome)) ||
        write_section(f, "GIDX", gidx, (uint64_t)n * 4) ||
        write_section(f, "END_", NULL, 0)) {
        fclose(f); f = NULL; remove(tmp);
        bio_set_error("write %s failed", tmp);
        goto done;
    }
    rc = commit_file(f, tmp, path);
    f = NULL;
done:
    free(a); free(fl); free(hot); free(cold); free(gt); free(gidx); free(keys);
    return rc;
}

BioSim* bio_sim_load(const char* path, const BioEnv* env, const BioConfig* cfg) {
    if (!path || !env || !cfg) { bio_set_error("bio_sim_load: bad arguments"); return NULL; }
    FILE* f = fopen(path, "rb");
    if (!f) { bio_set_error("cannot open %s", path); return NULL; }
    uint8_t hdr[CKPT_HDR];
    BioSim* s = NULL;
    void* sec[16] = { 0 };
    uint64_t secn[16] = { 0 };
    char tags[16][5];
    int nsec = 0;
    BioAgentRec* a = NULL;
    if (fread(hdr, 1, CKPT_HDR, f) != CKPT_HDR || memcmp(hdr, CKPT_MAGIC, 8) || get32(hdr + 508) != bio_crc32(0, hdr, 508)) {
        bio_set_error("%s: not a valid GMEBIO02 checkpoint", path); goto fail;
    }
    if (get64(hdr + 16) != env->planet_hash) {
        bio_set_error("%s: planet hash %016llx does not match the environment (%016llx)", path,
                      (unsigned long long)get64(hdr + 16), (unsigned long long)env->planet_hash);
        goto fail;
    }
    if (get32(hdr + 24) != bio_env_crc(env)) {
        bio_set_error("%s: env crc %08x does not match the environment (%08x)", path, get32(hdr + 24), bio_env_crc(env));
        goto fail;
    }
    if ((int32_t)get32(hdr + 56) != env->w || (int32_t)get32(hdr + 60) != env->h || (int32_t)get32(hdr + 64) != env->shift) {
        bio_set_error("%s: grid size does not match the environment", path); goto fail;
    }
    while (nsec < 16) {
        uint64_t n = 0;
        void* p = read_section(f, tags[nsec], &n);
        if (!p) { bio_set_error("%s: truncated or corrupt section", path); goto fail; }
        sec[nsec] = p; secn[nsec] = n;
        if (!strcmp(tags[nsec++], "END_")) break;
    }
    fclose(f); f = NULL;
#define SEC(name, var, sz) do { var = NULL; for (int _k = 0; _k < nsec; ++_k) if (!strcmp(tags[_k], name)) { var = sec[_k]; sz = secn[_k]; } } while (0)
    void *parm, *glob, *sims, *flds, *hot, *cold, *gtab, *gidx;
    uint64_t nparm = 0, nglob = 0, nsims = 0, nflds = 0, nhot = 0, ncold = 0, ngtab = 0, ngidx = 0;
    SEC("PARM", parm, nparm); SEC("GLOB", glob, nglob); SEC("SIMS", sims, nsims); SEC("FLDS", flds, nflds);
    SEC("HOT_", hot, nhot); SEC("COLD", cold, ncold); SEC("GTAB", gtab, ngtab); SEC("GIDX", gidx, ngidx);
#undef SEC
    const uint32_t n = get32(hdr + 48);
    if (!glob || nglob != sizeof(BioGlobals) || !sims || nsims != 16 * 8 || !flds ||
        nflds != (uint64_t)env->w * env->h * BF_N * 8 || !hot || nhot != (uint64_t)n * 72 || !cold || ncold != (uint64_t)n * 76 ||
        !gtab || ngtab % sizeof(BioGenome) || !gidx || ngidx != (uint64_t)n * 4) {
        bio_set_error("%s: missing or malformed sections", path); goto fail;
    }
    BioConfig c2 = *cfg;
    if (!c2.salt) c2.salt = get32(hdr + 32);
    if (!c2.cap) c2.cap = get32(hdr + 36);
    if (c2.cap < n) { bio_set_error("%s: capacity %u below %u live agents", path, c2.cap, n); goto fail; }
    c2.atm_scale_ppm = (int32_t)get32(hdr + 68);
    c2.barrier_q8 = (int32_t)get32(hdr + 72);
    c2.closed_box = (int32_t)get32(hdr + 76);
    s = bio_sim_create(env, &c2);
    if (!s) goto fail;
    BioWorld* w = &s->w;
    if (parm && nparm == sizeof(BioParams)) memcpy(&w->P, parm, sizeof(BioParams));
    w->salt = c2.salt;
    w->step = get64(hdr + 40);
    memcpy(&s->Gh, glob, sizeof(BioGlobals));
    if (bio_g_put(s) || bio_fields_put(s, (const int64_t*)flds)) goto fail;
    {
        const int64_t* sm = (const int64_t*)sims;
        s->g0_since = (uint64_t)sm[0]; s->g1_since = (uint64_t)sm[1]; s->g1_prev_photo = sm[2]; s->g1_prev_fix = sm[3];
        s->g1_prev_step = (uint64_t)sm[4]; s->g2_prev_oxy = sm[5]; s->gates = (uint32_t)sm[6];
        s->births = (uint64_t)sm[7]; s->deaths = (uint64_t)sm[8];
        for (int k = 0; k < 5; ++k) s->init_total[k] = sm[9 + k];
    }
    const uint32_t ng = (uint32_t)(ngtab / sizeof(BioGenome));
    a = (BioAgentRec*)calloc(n ? n : 1, sizeof(BioAgentRec));
    if (!a) { bio_set_error("out of memory"); goto fail; }
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* p = (const uint8_t*)hot + (size_t)i * 72;
        BioAgentRec* r = &a[i];
        r->x = (int32_t)get32(p); r->z = (int32_t)get32(p + 4); r->C = (int32_t)get32(p + 8); r->S = (int32_t)get32(p + 12);
        r->E = (int32_t)get32(p + 16); r->N = (int32_t)get32(p + 20); r->P = (int32_t)get32(p + 24); r->gut = (int32_t)get32(p + 28);
        r->W = (int32_t)get32(p + 32); r->age = get32(p + 36); r->flags = get32(p + 40); r->rbase = (int32_t)get32(p + 44);
        r->elast = (int32_t)get32(p + 48); r->uid = get32(p + 52); r->hid = get64(p + 56); r->out = get64(p + 64);
        const uint8_t* q = (const uint8_t*)cold + (size_t)i * 76;
        for (int k = 0; k < BIO_OUT; ++k) r->dW[k] = get64(q + 8 * k);
        r->lineage = get32(q + 64); r->parent = get32(q + 68); r->bstep = get32(q + 72);
        const uint32_t gi = ((const uint32_t*)gidx)[i];
        if (gi >= ng) { bio_set_error("%s: genome index out of range", path); goto fail; }
        r->g = ((const BioGenome*)gtab)[gi];
    }
    if (bio_append_agents(s, a, n, 0)) goto fail;
    w->uid_next = get32(hdr + 52);
    free(a);
    for (int k = 0; k < nsec; ++k) free(sec[k]);
    return s;
fail:
    if (f) fclose(f);
    free(a);
    for (int k = 0; k < nsec; ++k) free(sec[k]);
    bio_sim_destroy(s);
    return NULL;
}

/* ---- vegetation (spec 7.2) -------------------------------------------------------- */
/* integer HSV wheel, identical to biomap_default_palette() in biomap.c */
static void veg_palette(uint8_t pal[16][3]) {
    pal[0][0] = pal[0][1] = pal[0][2] = 0;
    const int v = 176, sat = 150;
    for (int i = 1; i < 16; ++i) {
        const int hue = (i - 1) * 1536 / 15;
        const int sec = hue >> 8, fr = hue & 255;
        const int lo = v * (255 - sat) / 255;
        const int dn = v * (255 - sat * fr / 255) / 255;
        const int up = v * (255 - sat * (255 - fr) / 255) / 255;
        int r, g, b;
        switch (sec) {
        case 0:  r = v;  g = up; b = lo; break;
        case 1:  r = dn; g = v;  b = lo; break;
        case 2:  r = lo; g = v;  b = up; break;
        case 3:  r = lo; g = dn; b = v;  break;
        case 4:  r = up; g = lo; b = v;  break;
        default: r = v;  g = lo; b = dn; break;
        }
        pal[i][0] = (uint8_t)r; pal[i][1] = (uint8_t)g; pal[i][2] = (uint8_t)b;
    }
}
/* pigment hue (0..255) -> palette index 1..15 */
static uint32_t hue_index(uint32_t hue) { return 1u + hue * 15u / 256u; }

typedef struct { uint32_t cell, idx; } CellKey;
static int ckey_cmp(const void* a, const void* b) {
    const CellKey* x = (const CellKey*)a; const CellKey* y = (const CellKey*)b;
    if (x->cell != y->cell) return x->cell < y->cell ? -1 : 1;
    return x->idx < y->idx ? -1 : (x->idx > y->idx ? 1 : 0);
}

int bio_sim_vegetation(BioSim* s, uint32_t* cells_out, uint32_t fallback_out[64], uint8_t palette[16][3]) {
    if (!s || !cells_out) return -1;
    const BioEnv* e = s->env;
    const BioWorld* w = &s->w;
    const int32_t nc = w->n_cells;
    BioAgentRec* a = NULL;
    uint32_t n = 0;
    if (bio_download_agents(s, &a, &n)) return -1;
    CellKey* keys = (CellKey*)malloc(sizeof(CellKey) * (n ? n : 1));
    if (!keys) { free(a); bio_set_error("out of memory"); return -1; }
    uint32_t m = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* g = a[i].g.body;
        if (g[BG_HOLDFAST] <= 64 || (g[BG_E_ANOX] < 32 && g[BG_E_OXY] < 32)) continue;
        if (a[i].C + a[i].S <= 0) continue;
        keys[m].cell = (uint32_t)bio_cell_at(w, a[i].x, a[i].z);
        keys[m].idx = i;
        ++m;
    }
    qsort(keys, m, sizeof(CellKey), ckey_cmp);
    memset(cells_out, 0, sizeof(uint32_t) * (size_t)nc);
    const int64_t cell_m2 = (int64_t)1 << (2 * w->shift);
    const int64_t ref = 4167;   /* 50 g C per m^2 in mmol */
    for (uint32_t k = 0; k < m;) {
        const uint32_t c = keys[k].cell;
        int64_t cls[5] = { 0, 0, 0, 0, 0 }, hsum[5] = { 0, 0, 0, 0, 0 }, hue[16];
        memset(hue, 0, sizeof hue);
        int64_t tot = 0;
        const int water = e->depth[c] >= 1 || e->wet[2][c] > 0;
        uint32_t j = k;
        for (; j < m && keys[j].cell == c; ++j) {
            const BioAgentRec* r = &a[keys[j].idx];
            const int64_t b = (int64_t)r->C + r->S;
            const int32_t hq4 = bio_height_q4(r->S);     /* 1/16 m */
            int cl;
            if (water) cl = hq4 >= 16 ? 1 : 0;          /* kelp from 1 m, else mat */
            else cl = hq4 < 8 ? 2 : (hq4 < 64 ? 3 : 4); /* cover < 0.5 m, shrub < 4 m, tree */
            cls[cl] += b; hsum[cl] += b * hq4; tot += b;
            hue[hue_index(r->g.body[BG_HUE])] += b;
        }
        int best = 0;
        for (int q = 1; q < 5; ++q) if (cls[q] > cls[best]) best = q;
        static const uint32_t kind_of[5] = { BIO_KIND_MAT, BIO_KIND_KELP, BIO_KIND_COVER, BIO_KIND_SHRUB, BIO_KIND_TREE };
        int64_t cover = tot * 255 / (ref * cell_m2);
        if (cover > 255) cover = 255;
        if (cover < 1) cover = 1;
        int64_t hm = cls[best] > 0 ? hsum[best] / cls[best] / 16 : 0;
        if (hm > 255) hm = 255;
        uint32_t pi = 1;
        for (uint32_t q = 2; q < 16; ++q) if (hue[q] > hue[pi]) pi = q;
        cells_out[c] = BIOCELL(kind_of[best], pi, (uint32_t)cover, (uint32_t)hm, 0u);
        k = j;
    }
    free(keys);
    free(a);
    /* fallback per env_bin: the most common kind, its statistical cover, mean height, commonest palette */
    if (fallback_out) {
        for (int b = 0; b < 64; ++b) {
            int64_t ncell = 0, kcnt[6] = { 0 }, kcov[6] = { 0 }, khgt[6] = { 0 }, kpal[6][16];
            memset(kpal, 0, sizeof kpal);
            for (int32_t c = 0; c < nc; ++c) {
                if ((e->env_bin[c] & 63) != b) continue;
                ++ncell;
                const uint32_t v = cells_out[c], kd = BIOCELL_KIND(v);
                if (!kd || kd > 5) continue;
                kcnt[kd]++; kcov[kd] += BIOCELL_COVER(v); khgt[kd] += BIOCELL_HEIGHT(v); kpal[kd][BIOCELL_PAL(v)]++;
            }
            int best = 0;
            for (int q = 1; q <= 5; ++q) if (kcnt[q] > kcnt[best]) best = q;
            if (!best || !ncell) { fallback_out[b] = 0; continue; }
            uint32_t pi = 0;
            for (uint32_t q = 1; q < 16; ++q) if (kpal[best][q] > kpal[best][pi]) pi = q;
            const int64_t cov = kcov[best] / ncell;      /* mean cover x share of cells with this kind */
            const int64_t hgt = khgt[best] / kcnt[best];
            fallback_out[b] = cov > 0 ? BIOCELL((uint32_t)best, pi, (uint32_t)cov, (uint32_t)hgt, 0u) : 0u;
        }
    }
    if (palette) veg_palette(palette);
    return 0;
}

/* ---- migrants (spec 6.4) -------------------------------------------------------------- */
#define MIG_MAGIC "GMEMIG01"
#define MIG_REC (4 + 4 + 8 + (int)sizeof(BioGenome))

typedef struct { int32_t cls; uint32_t lineage; int64_t mass; uint32_t rep; int64_t repmass; } LinAgg;
typedef struct { uint64_t key; uint32_t idx; } LKey;
static int lkey_cmp(const void* a, const void* b) {
    const LKey* x = (const LKey*)a; const LKey* y = (const LKey*)b;
    if (x->key != y->key) return x->key < y->key ? -1 : 1;
    return x->idx < y->idx ? -1 : (x->idx > y->idx ? 1 : 0);
}
static int lin_cmp(const void* a, const void* b) {
    const LinAgg* x = (const LinAgg*)a; const LinAgg* y = (const LinAgg*)b;
    if (x->cls != y->cls) return x->cls < y->cls ? -1 : 1;
    if (x->mass != y->mass) return x->mass > y->mass ? -1 : 1;
    return x->lineage < y->lineage ? -1 : (x->lineage > y->lineage ? 1 : 0);
}

int bio_sim_export_migrants(BioSim* s, const char* path, int n_per_class) {
    if (!s || !path || n_per_class <= 0) return -1;
    const BioWorld* w = &s->w;
    BioAgentRec* a = NULL;
    uint32_t n = 0;
    if (bio_download_agents(s, &a, &n)) return -1;
    LKey* keys = (LKey*)malloc(sizeof(LKey) * (n ? n : 1));
    LinAgg* lin = (LinAgg*)malloc(sizeof(LinAgg) * (n ? n : 1));
    uint8_t* buf = NULL;
    int rc = -1;
    if (!keys || !lin) { bio_set_error("out of memory"); goto done; }
    /* habitat class from the host env arrays (sim pointers may be device memory) */
    for (uint32_t i = 0; i < n; ++i) {
        keys[i].idx = i;
        const int32_t c = bio_cell_at(w, a[i].x, a[i].z);
        const BioEnv* e = s->env;
        const int32_t cls = e->wet[2][c] == 0 ? 2 : ((e->wet[0][c] >= 255 && e->depth[c] >= 1) ? 0 : 1);
        keys[i].key = ((uint64_t)(uint32_t)cls << 32) | a[i].lineage;
    }
    qsort(keys, n, sizeof(LKey), lkey_cmp);
    uint32_t nl = 0;
    for (uint32_t k = 0; k < n;) {
        uint32_t j = k;
        LinAgg L = { (int32_t)(keys[k].key >> 32), (uint32_t)keys[k].key, 0, keys[k].idx, -1 };
        for (; j < n && keys[j].key == keys[k].key; ++j) {
            const BioAgentRec* r = &a[keys[j].idx];
            const int64_t b = (int64_t)r->C + r->S;
            L.mass += b;
            if (b > L.repmass) { L.repmass = b; L.rep = keys[j].idx; }
        }
        lin[nl++] = L;
        k = j;
    }
    qsort(lin, nl, sizeof(LinAgg), lin_cmp);
    uint32_t cnt = 0;
    {
        int32_t per[3] = { 0, 0, 0 };
        for (uint32_t k = 0; k < nl; ++k) if (per[lin[k].cls] < n_per_class) { per[lin[k].cls]++; cnt++; }
    }
    const size_t bytes = 32 + (size_t)cnt * MIG_REC + 4;
    buf = (uint8_t*)calloc(1, bytes);
    if (!buf) { bio_set_error("out of memory"); goto done; }
    memcpy(buf, MIG_MAGIC, 8);
    put32(buf + 8, 1); put32(buf + 12, cnt); put64(buf + 16, s->planet_hash); put64(buf + 24, w->step);
    {
        int32_t per[3] = { 0, 0, 0 };
        uint8_t* p = buf + 32;
        for (uint32_t k = 0; k < nl; ++k) {
            if (per[lin[k].cls] >= n_per_class) continue;
            per[lin[k].cls]++;
            put32(p, (uint32_t)lin[k].cls); put32(p + 4, lin[k].lineage); put64(p + 8, (uint64_t)lin[k].mass);
            memcpy(p + 16, &a[lin[k].rep].g, sizeof(BioGenome));
            p += MIG_REC;
        }
    }
    put32(buf + bytes - 4, bio_crc32(0, buf, bytes - 4));
    {
        char tmp[1024];
        snprintf(tmp, sizeof tmp, "%s.tmp", path);
        FILE* f = fopen(tmp, "wb");
        if (!f) { bio_set_error("cannot write %s", tmp); goto done; }
        if (fwrite(buf, 1, bytes, f) != bytes) { fclose(f); remove(tmp); bio_set_error("write failed"); goto done; }
        if (commit_file(f, tmp, path)) goto done;
    }
    rc = (int)cnt;
done:
    free(a); free(keys); free(lin); free(buf);
    return rc;
}

int bio_sim_import_migrants(BioSim* s, const char* path) {
    if (!s || !path) return -1;
    BioWorld* w = &s->w;
    const BioEnv* e = s->env;
    FILE* f = fopen(path, "rb");
    if (!f) { bio_set_error("cannot open %s", path); return -1; }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 36) { fclose(f); bio_set_error("%s: too short", path); return -1; }
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (!buf) { fclose(f); bio_set_error("out of memory"); return -1; }
    const size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    BioAgentRec* a = NULL;
    int rc = -1;
    if (got != (size_t)sz || memcmp(buf, MIG_MAGIC, 8) || get32(buf + (size_t)sz - 4) != bio_crc32(0, buf, (size_t)sz - 4)) {
        bio_set_error("%s: not a valid migrant file", path); goto done;
    }
    const uint32_t cnt = get32(buf + 12);
    if ((size_t)sz != 32 + (size_t)cnt * MIG_REC + 4) { bio_set_error("%s: bad size", path); goto done; }
    if (get64(buf + 16) && s->planet_hash && get64(buf + 16) != s->planet_hash) { bio_set_error("%s: other planet", path); goto done; }
    a = (BioAgentRec*)calloc(cnt ? cnt : 1, sizeof(BioAgentRec));
    if (!a) { bio_set_error("out of memory"); goto done; }
    BioWorld hw = *w;
    hw.T = &s->tables;
    uint32_t m = 0;
    const int32_t cell_q8 = 1 << (w->shift + 8);
    for (uint32_t k = 0; k < cnt; ++k) {
        const uint8_t* p = buf + 32 + (size_t)k * MIG_REC;
        const int32_t cls = (int32_t)get32(p);
        BioAgentRec* r = &a[m];
        memcpy(&r->g, p + 16, sizeof(BioGenome));
        int32_t c = -1;
        for (int t = 0; t < 256 && c < 0; ++t) {
            const uint32_t h = bio_hash((int32_t)k, (int32_t)w->step, BH_IMPORT + 16 * t, w->salt);
            const int32_t cc = (int32_t)(h % (uint32_t)w->n_cells);
            const int32_t rc2 = e->wet[2][cc] == 0 ? 2 : ((e->wet[0][cc] >= 255 && e->depth[cc] >= 1) ? 0 : 1);
            if (rc2 == cls) c = cc;
        }
        if (c < 0) continue;
        const uint32_t h2 = bio_hash((int32_t)k, (int32_t)w->step, BH_IMPORT + 7, w->salt);
        r->x = (c % w->w) * cell_q8 + (int32_t)(h2 & (uint32_t)(cell_q8 - 1));
        r->z = (c / w->w) * cell_q8 + (int32_t)((h2 >> 16) & (uint32_t)(cell_q8 - 1));
        const int32_t mr = bio_mr(&hw, r->g.body[BG_SIZE]);
        r->C = mr; r->S = 0;
        r->E = bio_repro_e(&hw, r->g.body) + w->P.copy_cost;
        r->N = (mr + w->P.cn - 1) / w->P.cn; r->P = (mr + w->P.cp - 1) / w->P.cp;
        r->W = 256;
        r->flags = BA_ALIVE | (e->depth[c] >= 1 ? BA_WET : BA_LAND);
        r->elast = r->E;
        r->lineage = get32(p + 4);
        r->bstep = (uint32_t)w->step;
        r->out = bio_out_bias(&hw, &r->g);
        ++m;
    }
    if (s->n_live + m > s->cfg.cap) m = s->cfg.cap > s->n_live ? s->cfg.cap - s->n_live : 0;
    if (bio_append_agents(s, a, m, 1)) goto done;
    rc = (int)m;
done:
    free(buf); free(a);
    return rc;
}

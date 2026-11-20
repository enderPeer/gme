/* bio_bank.c - evolved genomes as cargo (docs/biosphere-spec.md 6.6, 7.5):
 *
 *   the seed bank   one record per distinct genome of a run, with its guild,
 *                   habitat and living biomass (GMESEED1), ranked and cut to
 *                   the best per guild x habitat;
 *   seeding         founders of banked genomes into cells of the matching
 *                   habitat of any island (rebase onto another planet,
 *                   inoculation of one island from another);
 *   emigrants       what an island sends to the hub each epoch: its most
 *                   successful genomes per guild x habitat (GMEMIG01);
 *   fauna           the creatures of a run (what moves or lives on prey and is
 *                   not vegetation) as a GMEFAUN1 file for the engine.
 *
 * Genomes carry no planet state: a founder starts as a fresh agent with its
 * reproduction mass, exactly like a migrant. Host side only. */
#include "bio_internal.h"
#include "../fauna.h"
#include <stdlib.h>
#include <string.h>

/* ---- guild and habitat ----------------------------------------------------------- */
/* Guilds by capability (genes), not by realised flux: producers can
 * photosynthesise (pigment and ANOX or OXY >= 32); chemotrophs lean to
 * lithotrophy (METH / FEOX / METHOX at least as strong as FERM / SULFRED /
 * AERO); the other heterotrophs are decomposers, or consumers when motile. */
int bio_guild(const uint8_t* g) {
    if (g[BG_PIGMENT] > 0 && (g[BG_E_ANOX] >= 32 || g[BG_E_OXY] >= 32)) return BIO_GUILD_PRODUCER;
    int chemo = g[BG_E_METH];
    if (g[BG_E_FEOX] > chemo) chemo = g[BG_E_FEOX];
    if (g[BG_E_METHOX] > chemo) chemo = g[BG_E_METHOX];
    int het = g[BG_E_FERM];
    if (g[BG_E_SULFRED] > het) het = g[BG_E_SULFRED];
    if (g[BG_E_AERO] > het) het = g[BG_E_AERO];
    if (chemo >= het && chemo > 0) return BIO_GUILD_CHEMOTROPH;
    return g[BG_MOTOR] > 0 ? BIO_GUILD_CONSUMER : BIO_GUILD_DECOMPOSER;
}

/* habitat of a cell: marine (always under water), coastal (tidal), land (never wet) */
int bio_habitat(const BioEnv* e, int64_t c) {
    if (e->wet[2][c] == 0) return BIO_HAB_LAND;
    if (e->wet[0][c] >= 255 && e->depth[c] >= 1) return BIO_HAB_MARINE;
    return BIO_HAB_COASTAL;
}

const char* bio_guild_name(int g) {
    static const char* const n[BIO_GUILD_N] = { "producer", "chemotroph", "decomposer", "consumer" };
    return g >= 0 && g < BIO_GUILD_N ? n[g] : "any";
}
const char* bio_habitat_name(int h) {
    static const char* const n[BIO_HAB_N] = { "marine", "coastal", "land" };
    return h >= 0 && h < BIO_HAB_N ? n[h] : "any";
}

/* ---- little-endian helpers --------------------------------------------------------- */
static void put32(uint8_t* p, uint32_t v) { for (int k = 0; k < 4; ++k) p[k] = (uint8_t)(v >> (8 * k)); }
static void put64(uint8_t* p, uint64_t v) { for (int k = 0; k < 8; ++k) p[k] = (uint8_t)(v >> (8 * k)); }
static uint32_t get32(const uint8_t* p) { uint32_t v = 0; for (int k = 0; k < 4; ++k) v |= (uint32_t)p[k] << (8 * k); return v; }
static uint64_t get64(const uint8_t* p) { uint64_t v = 0; for (int k = 0; k < 8; ++k) v |= (uint64_t)p[k] << (8 * k); return v; }

/* ---- the bank of one run ------------------------------------------------------------- */
typedef struct { uint64_t h; uint32_t cls; uint32_t idx; } BKey;
static int bkey_cmp(const void* a, const void* b) {
    const BKey* x = (const BKey*)a; const BKey* y = (const BKey*)b;
    if (x->cls != y->cls) return x->cls < y->cls ? -1 : 1;
    if (x->h != y->h) return x->h < y->h ? -1 : 1;
    return x->idx < y->idx ? -1 : (x->idx > y->idx ? 1 : 0);
}

/* One record per distinct genome and habitat: its living biomass there, the
 * lineage of its largest carrier. Order: by class, then genome hash (stable). */
int bio_sim_bank(BioSim* s, BioBankRec** out, uint32_t* nout) {
    if (!s || !out || !nout) return -1;
    *out = NULL; *nout = 0;
    BioAgentRec* a = NULL;
    uint32_t n = 0;
    if (bio_download_agents(s, &a, &n)) return -1;
    BKey* keys = (BKey*)malloc(sizeof(BKey) * (n ? n : 1));
    BioBankRec* r = (BioBankRec*)malloc(sizeof(BioBankRec) * (n ? n : 1));
    int rc = -1;
    if (!keys || !r) { bio_set_error("out of memory"); goto done; }
    for (uint32_t i = 0; i < n; ++i) {
        keys[i].h = bio_fnv64(BIO_FNV_INIT, &a[i].g, sizeof(BioGenome));
        keys[i].cls = (uint32_t)bio_habitat(s->env, bio_cell_at(&s->w, a[i].x, a[i].z));
        keys[i].idx = i;
    }
    qsort(keys, n, sizeof(BKey), bkey_cmp);
    uint32_t m = 0;
    for (uint32_t k = 0; k < n;) {
        uint32_t j = k, rep = keys[k].idx;
        int64_t mass = 0, repmass = -1;
        for (; j < n && keys[j].cls == keys[k].cls && keys[j].h == keys[k].h; ++j) {
            const BioAgentRec* x = &a[keys[j].idx];
            const int64_t b = (int64_t)x->C + x->S + x->gut;
            mass += b;
            if (b > repmass) { repmass = b; rep = keys[j].idx; }
        }
        BioBankRec* q = &r[m++];
        memset(q, 0, sizeof *q);
        q->habitat = keys[k].cls;
        q->guild = (uint32_t)bio_guild(a[rep].g.body);
        q->lineage = a[rep].lineage;
        q->island = (uint32_t)s->env->island;
        q->mass = mass;
        memcpy(q->genome, &a[rep].g, sizeof(BioGenome));
        k = j;
    }
    *out = r; *nout = m; r = NULL;
    rc = 0;
done:
    free(a); free(keys); free(r);
    return rc;
}

static int bank_rank(const void* a, const void* b) {
    const BioBankRec* x = (const BioBankRec*)a; const BioBankRec* y = (const BioBankRec*)b;
    if (x->guild != y->guild) return x->guild < y->guild ? -1 : 1;
    if (x->habitat != y->habitat) return x->habitat < y->habitat ? -1 : 1;
    if (x->mass != y->mass) return x->mass > y->mass ? -1 : 1;
    const int c = memcmp(x->genome, y->genome, sizeof x->genome);
    if (c) return c;
    return x->island < y->island ? -1 : (x->island > y->island ? 1 : 0);
}
/* class, then genome: equal genomes of a class end up next to each other */
static int bank_same(const void* a, const void* b) {
    const BioBankRec* x = (const BioBankRec*)a; const BioBankRec* y = (const BioBankRec*)b;
    if (x->guild != y->guild) return x->guild < y->guild ? -1 : 1;
    if (x->habitat != y->habitat) return x->habitat < y->habitat ? -1 : 1;
    const int c = memcmp(x->genome, y->genome, sizeof x->genome);
    if (c) return c;
    if (x->mass != y->mass) return x->mass > y->mass ? -1 : 1;
    return x->island < y->island ? -1 : (x->island > y->island ? 1 : 0);
}

/* Merges records with the same genome in the same guild x habitat (several
 * islands carry it: the masses add, the largest carrier's island and lineage
 * stay), ranks each class by biomass and keeps its best per_class. Returns
 * the new count. Sorting only: fine for the millions of records of a full island. */
uint32_t bio_bank_top(BioBankRec* r, uint32_t n, uint32_t per_class) {
    if (!r || !n) return 0;
    qsort(r, n, sizeof(BioBankRec), bank_same);
    uint32_t m = 0;
    for (uint32_t k = 0; k < n; ++k) {
        if (m && r[m - 1].guild == r[k].guild && r[m - 1].habitat == r[k].habitat && !memcmp(r[m - 1].genome, r[k].genome, sizeof r[k].genome))
            r[m - 1].mass += r[k].mass;          /* r[m - 1] is the heaviest carrier: bank_same puts it first */
        else r[m++] = r[k];
    }
    qsort(r, m, sizeof(BioBankRec), bank_rank);
    uint32_t w = 0, run = 0;
    for (uint32_t k = 0; k < m; ++k) {
        if (k && (r[k].guild != r[k - 1].guild || r[k].habitat != r[k - 1].habitat)) run = 0;
        if (run++ < per_class) r[w++] = r[k];
    }
    return w;
}

/* File GMESEED1, little-endian: 64-byte header, then count x 408-byte records
 *    0  char[8] "GMESEED1"   8  u32 version (1)   12  u32 header bytes (64)
 *   16  u64 planet_hash (where the genomes evolved)   24  u32 count   28  u32 record bytes (408)
 *   32  u32 reserved[6]     56  u32 crc32 of the records   60  u32 crc32 of bytes 0..59
 * record: u32 habitat, u32 guild, u32 lineage, u32 island, i64 mass, u8 genome[384] */
#define BANK_MAGIC "GMESEED1"
#define BANK_HDR 64
#define BANK_REC 408

int bio_bank_save(const char* path, const BioBankRec* r, uint32_t n, uint64_t planet_hash) {
    if (!path || (n && !r)) return -1;
    const size_t bytes = BANK_HDR + (size_t)n * BANK_REC;
    uint8_t* buf = (uint8_t*)calloc(1, bytes);
    if (!buf) { bio_set_error("out of memory"); return -1; }
    for (uint32_t k = 0; k < n; ++k) {
        uint8_t* p = buf + BANK_HDR + (size_t)k * BANK_REC;
        put32(p, r[k].habitat); put32(p + 4, r[k].guild); put32(p + 8, r[k].lineage); put32(p + 12, r[k].island);
        put64(p + 16, (uint64_t)r[k].mass);
        memcpy(p + 24, r[k].genome, 384);
    }
    memcpy(buf, BANK_MAGIC, 8);
    put32(buf + 8, 1); put32(buf + 12, BANK_HDR); put64(buf + 16, planet_hash); put32(buf + 24, n); put32(buf + 28, BANK_REC);
    put32(buf + 56, bio_crc32(0, buf + BANK_HDR, bytes - BANK_HDR));
    put32(buf + 60, bio_crc32(0, buf, 60));
    char tmp[1100];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    int ok = f != NULL;
    if (ok) ok = fwrite(buf, 1, bytes, f) == bytes;
    if (f && fclose(f)) ok = 0;
    free(buf);
    if (ok) { remove(path); ok = rename(tmp, path) == 0; }
    if (!ok) { remove(tmp); bio_set_error("cannot write %s", path); return -1; }
    return 0;
}

int bio_bank_load(const char* path, BioBankRec** out, uint32_t* nout, uint64_t* planet_hash) {
    if (!path || !out || !nout) return -1;
    *out = NULL; *nout = 0;
    FILE* f = fopen(path, "rb");
    if (!f) { bio_set_error("cannot open %s", path); return -1; }
    uint8_t hd[BANK_HDR];
    uint8_t* buf = NULL;
    BioBankRec* r = NULL;
    int rc = -1;
    if (fread(hd, 1, BANK_HDR, f) != BANK_HDR || memcmp(hd, BANK_MAGIC, 8) || get32(hd + 8) != 1 || get32(hd + 12) != BANK_HDR ||
        get32(hd + 28) != BANK_REC || get32(hd + 60) != bio_crc32(0, hd, 60)) {
        bio_set_error("%s: not a valid GMESEED1 seed bank", path); goto done;
    }
    const uint32_t n = get32(hd + 24);
    if (n > (1u << 22)) { bio_set_error("%s: too many records", path); goto done; }
    const size_t bytes = (size_t)n * BANK_REC;
    buf = (uint8_t*)malloc(bytes ? bytes : 1);
    r = (BioBankRec*)malloc(sizeof(BioBankRec) * (n ? n : 1));
    if (!buf || !r) { bio_set_error("out of memory"); goto done; }
    if ((bytes && fread(buf, 1, bytes, f) != bytes) || fgetc(f) != EOF || bio_crc32(0, buf, bytes) != get32(hd + 56)) {
        bio_set_error("%s: truncated or corrupt", path); goto done;
    }
    for (uint32_t k = 0; k < n; ++k) {
        const uint8_t* p = buf + (size_t)k * BANK_REC;
        r[k].habitat = get32(p); r[k].guild = get32(p + 4); r[k].lineage = get32(p + 8); r[k].island = get32(p + 12);
        r[k].mass = (int64_t)get64(p + 16);
        memcpy(r[k].genome, p + 24, 384);
        if (r[k].habitat >= BIO_HAB_N || r[k].guild >= BIO_GUILD_N) { bio_set_error("%s: record %u has an unknown class", path, k); goto done; }
    }
    if (planet_hash) *planet_hash = get64(hd + 16);
    *out = r; *nout = n; r = NULL;
    rc = 0;
done:
    fclose(f);
    free(buf); free(r);
    return rc;
}

/* ---- seeding ---------------------------------------------------------------------------- */
/* Places `founders` agents of the banked genomes that match guild / habitat
 * (-1 = any) into cells of each genome's habitat on this island. With at
 * least as many founders as matching records, founder k is record k mod
 * (records): every genome gets its share and the best-ranked the remainder.
 * With fewer, the founders are spread evenly over the records (which are in
 * guild, habitat, biomass order), so every guild and habitat gets its part. Each starts with its reproduction mass, the
 * energy to reproduce once and the nutrients for its body; all of it enters
 * the import ledger, so the totals stay exact. Records whose habitat this
 * island does not have are skipped. Returns the number placed, -1 on error. */
int bio_sim_seed_bank(BioSim* s, const BioBankRec* r, uint32_t n, uint32_t founders, int guild, int habitat) {
    if (!s || (n && !r)) return -1;
    BioWorld* w = &s->w;
    const BioEnv* e = s->env;
    /* cells of each habitat (deterministic order) */
    uint32_t* cells[BIO_HAB_N] = { NULL, NULL, NULL };
    uint32_t ncell[BIO_HAB_N] = { 0, 0, 0 };
    for (int32_t c = 0; c < w->n_cells; ++c) ncell[bio_habitat(e, c)]++;
    int rc = -1;
    uint32_t* pick = NULL;
    BioAgentRec* a = NULL;
    for (int h = 0; h < BIO_HAB_N; ++h) {
        cells[h] = (uint32_t*)malloc(sizeof(uint32_t) * (ncell[h] ? ncell[h] : 1));
        if (!cells[h]) { bio_set_error("out of memory"); goto done; }
        ncell[h] = 0;
    }
    for (int32_t c = 0; c < w->n_cells; ++c) { const int h = bio_habitat(e, c); cells[h][ncell[h]++] = (uint32_t)c; }
    pick = (uint32_t*)malloc(sizeof(uint32_t) * (n ? n : 1));
    if (!pick) { bio_set_error("out of memory"); goto done; }
    uint32_t np = 0;
    for (uint32_t k = 0; k < n; ++k) {
        if (guild >= 0 && (int)r[k].guild != guild) continue;
        if (habitat >= 0 && (int)r[k].habitat != habitat) continue;
        if (!ncell[r[k].habitat]) continue;
        pick[np++] = k;
    }
    if (!np || !founders) { rc = 0; goto done; }
    if (s->n_live + founders > s->cfg.cap) founders = s->cfg.cap > s->n_live ? s->cfg.cap - s->n_live : 0;
    a = (BioAgentRec*)calloc(founders ? founders : 1, sizeof(BioAgentRec));
    if (!a) { bio_set_error("out of memory"); goto done; }
    BioWorld hw = *w;
    hw.T = &s->tables;
    const int32_t cell_q8 = 1 << (w->shift + 8);
    for (uint32_t k = 0; k < founders; ++k) {
        const BioBankRec* q = &r[pick[founders < np ? (uint32_t)(((uint64_t)k * np) / founders) : k % np]];
        BioAgentRec* x = &a[k];
        memcpy(&x->g, q->genome, sizeof(BioGenome));
        const uint32_t h = bio_hash((int32_t)k, (int32_t)w->step, BH_IMPORT + 3, w->salt ^ 0xBA2Cu);
        const uint32_t h2 = bio_hash((int32_t)k, (int32_t)w->step, BH_IMPORT + 5, w->salt ^ 0xBA2Cu);
        const int32_t c = (int32_t)cells[q->habitat][h % ncell[q->habitat]];
        x->x = (c % w->w) * cell_q8 + (int32_t)(h2 & (uint32_t)(cell_q8 - 1));
        x->z = (c / w->w) * cell_q8 + (int32_t)((h2 >> 16) & (uint32_t)(cell_q8 - 1));
        const int32_t mr = bio_mr(&hw, x->g.body[BG_SIZE]);
        x->C = mr; x->S = 0;
        x->E = bio_repro_e(&hw, x->g.body) + w->P.copy_cost;
        x->N = (mr + w->P.cn - 1) / w->P.cn; x->P = (mr + w->P.cp - 1) / w->P.cp;
        x->W = 256;
        x->flags = BA_ALIVE | (e->depth[c] >= 1 ? BA_WET : BA_LAND);
        x->elast = x->E;
        x->lineage = q->lineage;
        x->bstep = (uint32_t)w->step;
        x->out = bio_out_bias(&hw, &x->g);
    }
    if (bio_append_agents(s, a, founders, 1)) goto done;
    rc = (int)founders;
done:
    for (int h = 0; h < BIO_HAB_N; ++h) free(cells[h]);
    free(pick); free(a);
    return rc;
}

/* ---- emigrants (spec 6.4, 10.7) ------------------------------------------------------------ */
/* The genomes an island sends out each epoch, as the GMEMIG01 file that
 * bio_sim_import_migrants reads. Per habitat and guild: the distinct genomes
 * with the most living biomass, one per species-like cluster first (no genome
 * within the species distance of one already chosen; the EMIG_SCAN heaviest of
 * the class are looked at), then the next heaviest, until a quarter of
 * per_habitat are chosen. Founder lineages are no measure of what an island
 * has to offer: an old world descends from a handful of them, and choosing by
 * lineage sent three genomes per habitat from a 2.8 M-agent island.
 * Returns the number of genomes written, -1 on error. */
#define EMIG_SCAN 4096u
typedef struct { uint64_t h; int64_t mass; uint32_t cls, rep; } EAgg;       /* cls = habitat * BIO_GUILD_N + guild */
static int eagg_cmp(const void* a, const void* b) {
    const EAgg* x = (const EAgg*)a; const EAgg* y = (const EAgg*)b;
    if (x->cls != y->cls) return x->cls < y->cls ? -1 : 1;
    if (x->mass != y->mass) return x->mass > y->mass ? -1 : 1;
    return x->h < y->h ? -1 : (x->h > y->h ? 1 : 0);
}
int bio_sim_export_emigrants(BioSim* s, const char* path, int per_habitat) {
    if (!s || !path || per_habitat <= 0) return -1;
    const uint32_t quota = ((uint32_t)per_habitat + BIO_GUILD_N - 1) / BIO_GUILD_N;
    BioAgentRec* a = NULL;
    uint32_t n = 0;
    if (bio_download_agents(s, &a, &n)) return -1;
    BKey* keys = (BKey*)malloc(sizeof(BKey) * (n ? n : 1));
    EAgg* g = (EAgg*)malloc(sizeof(EAgg) * (n ? n : 1));
    uint32_t* pick = (uint32_t*)malloc(sizeof(uint32_t) * BIO_HAB_N * BIO_GUILD_N * quota);
    uint8_t* taken = NULL;
    uint8_t* buf = NULL;
    int rc = -1;
    if (!keys || !g || !pick) { bio_set_error("out of memory"); goto done; }
    for (uint32_t i = 0; i < n; ++i) {
        keys[i].h = bio_fnv64(BIO_FNV_INIT, &a[i].g, sizeof(BioGenome));
        keys[i].cls = (uint32_t)bio_habitat(s->env, bio_cell_at(&s->w, a[i].x, a[i].z));
        keys[i].idx = i;
    }
    qsort(keys, n, sizeof(BKey), bkey_cmp);
    uint32_t m = 0;
    for (uint32_t k = 0; k < n;) {                       /* one entry per genome and habitat */
        uint32_t j = k, best = keys[k].idx;
        int64_t mass = 0, bestmass = -1;
        for (; j < n && keys[j].cls == keys[k].cls && keys[j].h == keys[k].h; ++j) {
            const BioAgentRec* x = &a[keys[j].idx];
            const int64_t b = (int64_t)x->C + x->S + x->gut;
            mass += b;
            if (b > bestmass) { bestmass = b; best = keys[j].idx; }
        }
        g[m].h = keys[k].h; g[m].mass = mass; g[m].rep = best;
        g[m].cls = keys[k].cls * BIO_GUILD_N + (uint32_t)bio_guild(a[best].g.body);
        ++m;
        k = j;
    }
    qsort(g, m, sizeof(EAgg), eagg_cmp);
    taken = (uint8_t*)calloc(m ? m : 1, 1);
    if (!taken) { bio_set_error("out of memory"); goto done; }
    uint32_t np = 0;
    for (uint32_t k = 0; k < m;) {
        uint32_t j = k;
        while (j < m && g[j].cls == g[k].cls) ++j;
        const uint32_t first = np, scan = j - k < EMIG_SCAN ? j - k : EMIG_SCAN;
        for (uint32_t i = k; i < k + scan && np - first < quota; ++i) {       /* one per cluster */
            int near = 0;
            for (uint32_t q = first; q < np && !near; ++q) near = bio_genome_near(&a[g[i].rep].g, &a[g[pick[q]].rep].g);
            if (!near) { pick[np++] = i; taken[i] = 1; }
        }
        for (uint32_t i = k; i < j && np - first < quota; ++i)                /* then the next heaviest */
            if (!taken[i]) { pick[np++] = i; taken[i] = 1; }
        k = j;
    }
    const size_t bytes = 32 + (size_t)np * MIG_REC + 4;
    buf = (uint8_t*)calloc(1, bytes);
    if (!buf) { bio_set_error("out of memory"); goto done; }
    memcpy(buf, MIG_MAGIC, 8);
    put32(buf + 8, 1); put32(buf + 12, np); put64(buf + 16, s->planet_hash); put64(buf + 24, s->w.step);
    for (uint32_t q = 0; q < np; ++q) {
        uint8_t* p = buf + 32 + (size_t)q * MIG_REC;
        const EAgg* e = &g[pick[q]];
        put32(p, e->cls / BIO_GUILD_N); put32(p + 4, a[e->rep].lineage); put64(p + 8, (uint64_t)e->mass);
        memcpy(p + 16, &a[e->rep].g, sizeof(BioGenome));
    }
    put32(buf + bytes - 4, bio_crc32(0, buf, bytes - 4));
    {
        char tmp[1100];
        snprintf(tmp, sizeof tmp, "%s.tmp", path);
        FILE* f = fopen(tmp, "wb");
        int ok = f != NULL;
        if (ok) ok = fwrite(buf, 1, bytes, f) == bytes;
        if (f && fclose(f)) ok = 0;
        if (ok) { remove(path); ok = rename(tmp, path) == 0; }
        if (!ok) { remove(tmp); bio_set_error("cannot write %s", path); goto done; }
    }
    rc = (int)np;
done:
    free(a); free(keys); free(g); free(pick); free(taken); free(buf);
    return rc;
}

/* ---- fauna (spec 7.5) --------------------------------------------------------------------- */
typedef struct { int64_t body; uint32_t idx; } FKey;
static int fkey_cmp(const void* a, const void* b) {
    const FKey* x = (const FKey*)a; const FKey* y = (const FKey*)b;
    if (x->body != y->body) return x->body > y->body ? -1 : 1;
    return x->idx < y->idx ? -1 : (x->idx > y->idx ? 1 : 0);
}
/* integer HSV wheel: the same palette as the vegetation export (biomap.c) */
static void fauna_palette(uint8_t pal[16][3]) {
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

/* 1 if the agent is a creature for the engine (bio_creature in bio_core.h, the
 * rule the statistics count by) and has a body to draw */
int bio_is_creature(const BioAgentRec* x) {
    return (int64_t)x->C + x->S + x->gut > 0 && bio_creature(x->g.body, x->eco);
}

int bio_sim_motor_census(BioSim* s, uint32_t out[6]) {
    if (!s || !out) return -1;
    memset(out, 0, sizeof(uint32_t) * 6);
    BioAgentRec* a = NULL;
    uint32_t n = 0;
    if (bio_download_agents(s, &a, &n)) return -1;
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* g = a[i].g.body;
        if (g[BG_MOTOR] == 0 || (int64_t)a[i].C + a[i].S + a[i].gut <= 0) continue;
        out[0]++;
        if (bio_is_creature(&a[i])) { out[1]++; continue; }
        if (g[BG_HOLDFAST] > 64 && (g[BG_E_ANOX] >= 32 || g[BG_E_OXY] >= 32)) {
            out[2]++;
            if (BE_GUTSHARE(a[i].eco) > 128u) out[3]++;
            if (g[BG_MOTOR] >= BIO_CREATURE_MOTOR) out[4]++;
        } else out[5]++;
    }
    free(a);
    return 0;
}

/* The creatures of this island, the largest bodies first, at most max of them
 * (<= FAUNA_MAX). One record per agent: a herd, a swarm or one large animal
 * (fauna.h). out->rec is malloc'd (fauna_free). */
int bio_sim_fauna(BioSim* s, struct Fauna* out, uint32_t max) {
    if (!s || !out) return -1;
    memset(out, 0, sizeof *out);
    if (max > FAUNA_MAX) max = FAUNA_MAX;
    const BioEnv* e = s->env;
    const BioWorld* w = &s->w;
    BioAgentRec* a = NULL;
    uint32_t n = 0;
    if (bio_download_agents(s, &a, &n)) return -1;
    FKey* keys = (FKey*)malloc(sizeof(FKey) * (n ? n : 1));
    if (!keys) { free(a); bio_set_error("out of memory"); return -1; }
    uint32_t m = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (!bio_is_creature(&a[i])) continue;
        const int64_t body = (int64_t)a[i].C + a[i].S + a[i].gut;
        if (body <= 0) continue;
        keys[m].body = body; keys[m].idx = i; ++m;
    }
    qsort(keys, m, sizeof(FKey), fkey_cmp);
    const uint32_t cnt = m < max ? m : max;
    out->rec = (FaunaRec*)calloc(cnt ? cnt : 1, sizeof(FaunaRec));
    if (!out->rec) { free(a); free(keys); bio_set_error("out of memory"); return -1; }
    BioWorld hw = *w;
    hw.T = &s->tables;
    for (uint32_t k = 0; k < cnt; ++k) {
        const BioAgentRec* x = &a[keys[k].idx];
        const uint8_t* g = x->g.body;
        FaunaRec* r = &out->rec[k];
        const int32_t c = bio_cell_at(w, x->x, x->z);
        r->x_q4 = e->x0 * 16 + (x->x >> 4);
        r->z_q4 = e->z0 * 16 + (x->z >> 4);
        const int swims = e->depth[c] >= 1;
        uint32_t fl = swims ? FAUNA_SWIMS : 0u;
        if (swims) {      /* its depth, as bio_zq8: the buoyancy gene and output 5 */
            const int32_t b = bio_clampi(g[BG_BUOY] + bio_lane(x->out, 5) / 2, 0, 255);
            const int32_t zq4 = e->depth[c] * 16 * (255 - b) / 255;
            r->depth_q4 = (uint16_t)(zq4 > 65535 ? 65535 : zq4);
        }
        /* diameter of a sphere of the body's volume, 0.12 cm^3 per mmol C: d^3 = 6 V / pi */
        const uint32_t d = bio_icbrt((uint64_t)keys[k].body * 939u / 1000000u);
        r->size_q4 = (uint16_t)(d < 1u ? 1u : (d > 65535u ? 65535u : d));
        if (BE_TROPH(x->eco) >= 24u) fl |= FAUNA_HUNTER;                  /* trophic level 2.5 and up */
        if (g[BG_ARMOR] >= 128) fl |= FAUNA_ARMORED;
        if (keys[k].body >= 64 * (int64_t)bio_mr(&hw, g[BG_SIZE])) fl |= FAUNA_SWARM;
        r->flags = (uint8_t)fl;
        r->pal = (uint8_t)bio_hue_bin(g[BG_HUE]);
        const uint32_t tl = 16u + BE_TROPH(x->eco);
        r->troph_q4 = (uint8_t)(tl > 255u ? 255u : tl);
        r->motor = g[BG_MOTOR];
    }
    out->planet_hash = s->planet_hash;
    out->seed = e->seed;
    out->sim_year = (uint32_t)(w->step / 8760u);
    out->count = cnt;
    out->total = m;
    fauna_palette(out->palette);
    free(a); free(keys);
    return 0;
}

/* bio_stats.c - counting species (docs/biosphere-spec.md 8.3).
 *
 * Nothing in the simulation assigns a species; they are read off afterwards
 * as clusters of similar genomes in a sample of the population:
 *
 *   distance = L1(body genes 0..41) / 255
 *            + 0.25 x L1(expressed brain bytes) / 127
 *            + 0.5 if the mate tags differ in more than 3 bits (cannot cross)
 *
 * joined by single linkage at 0.08 x 42 genes. "Expressed" brain bytes are the
 * weights of hidden neurons switched on in both genomes (and the output
 * biases): the weights of a neuron that is off drift freely and say nothing
 * about kinship. Host side; the pair distances run on the job pool. */
#include "bio_internal.h"
#include "../jobs.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SP_GENES 42          /* body genes that mean something: enzymes .. Q band width */
#define SP_THRESH_Q8 860     /* 0.08 x 42 = 3.36, in 1/256 */

typedef struct { uint32_t h, idx; } SKey;
static int skey_cmp(const void* a, const void* b) {
    const SKey* x = (const SKey*)a; const SKey* y = (const SKey*)b;
    if (x->h != y->h) return x->h < y->h ? -1 : 1;
    return x->idx < y->idx ? -1 : (x->idx > y->idx ? 1 : 0);
}

static int popc16(uint32_t v) { int n = 0; for (v &= 0xFFFFu; v; v &= v - 1) ++n; return n; }

/* 1 if the two genomes are within the threshold. Sums in 1/256 units, leaves early. */
int bio_genome_near(const BioGenome* a, const BioGenome* b) {
    int64_t d = 0;                                   /* q8 */
    for (int k = 0; k < SP_GENES; ++k) {
        const int v = (int)a->body[k] - (int)b->body[k];
        d += (int64_t)(v < 0 ? -v : v) * 256 / 255;
    }
    if (popc16((uint32_t)(a->mate_tag ^ b->mate_tag)) > 3) d += 128;
    if (d > SP_THRESH_Q8) return 0;
    const uint32_t on = (uint32_t)a->body[BG_NEURONS] & (uint32_t)b->body[BG_NEURONS];
    int64_t l1 = 0;
    for (int j = 0; j < BIO_H; ++j) {
        if (!((on >> j) & 1u)) continue;
        for (int k = 0; k < BIO_IN; ++k) { const int v = a->w_ih[j][k] - b->w_ih[j][k]; l1 += v < 0 ? -v : v; }
        for (int k = 0; k < BIO_H; ++k) { const int v = a->w_hh[j][k] - b->w_hh[j][k]; l1 += v < 0 ? -v : v; }
        { const int v = a->b_h[j] - b->b_h[j]; l1 += v < 0 ? -v : v; }
        for (int o = 0; o < BIO_OUT; ++o) { const int v = a->w_ho[o][j] - b->w_ho[o][j]; l1 += v < 0 ? -v : v; }
    }
    for (int o = 0; o < BIO_OUT; ++o) { const int v = a->b_o[o] - b->b_o[o]; l1 += v < 0 ? -v : v; }
    d += l1 * 64 / 127;                              /* 0.25 x L1 / 127, q8 */
    return d <= SP_THRESH_Q8;
}

typedef struct { const BioGenome* g; uint32_t n; uint32_t* bits; uint32_t words; } SpCtx;
static void sp_rows(void* ctx, uint32_t b, uint32_t e, uint32_t worker) {
    (void)worker;
    SpCtx* x = (SpCtx*)ctx;
    for (uint32_t i = b; i < e; ++i) {
        uint32_t* row = x->bits + (size_t)i * x->words;
        for (uint32_t j = i + 1; j < x->n; ++j)
            if (bio_genome_near(&x->g[i], &x->g[j])) row[j >> 5] |= 1u << (j & 31u);
    }
}

static uint32_t uf_find(uint32_t* p, uint32_t i) {
    while (p[i] != i) { p[i] = p[p[i]]; i = p[i]; }
    return i;
}

int bio_sim_species(BioSim* s, uint32_t sample, BioSpecies* out) {
    if (!s || !out) return -1;
    memset(out, 0, sizeof *out);
    if (sample < 2) sample = 4096;
    if (sample > 16384) sample = 16384;
    BioAgentRec* a = NULL;
    uint32_t n = 0;
    if (bio_download_agents(s, &a, &n)) return -1;
    int rc = -1;
    SKey* keys = (SKey*)malloc(sizeof(SKey) * (n ? n : 1));
    BioGenome* g = NULL;
    uint32_t* bits = NULL;
    uint32_t* par = NULL;
    uint32_t* size = NULL;
    if (!keys) { bio_set_error("out of memory"); goto done; }
    /* the sample: the agents with the smallest hash of their uid (the same agents on every device) */
    for (uint32_t i = 0; i < n; ++i) { keys[i].h = bio_hash((int32_t)a[i].uid, 0, BH_SEED + 77, 0x5BEC1E5u); keys[i].idx = i; }
    qsort(keys, n, sizeof(SKey), skey_cmp);
    const uint32_t m = n < sample ? n : sample;
    out->sample = m;
    if (m == 0) { rc = 0; goto done; }
    const uint32_t words = (m + 31) / 32;
    g = (BioGenome*)malloc(sizeof(BioGenome) * m);
    bits = (uint32_t*)calloc((size_t)m * words, 4);
    par = (uint32_t*)malloc(sizeof(uint32_t) * m);
    size = (uint32_t*)calloc(m, 4);
    if (!g || !bits || !par || !size) { bio_set_error("out of memory"); goto done; }
    for (uint32_t i = 0; i < m; ++i) { g[i] = a[keys[i].idx].g; par[i] = i; }
    SpCtx x = { g, m, bits, words };
    jobs_parallel_for(m, 8, sp_rows, &x);
    for (uint32_t i = 0; i < m; ++i) {
        const uint32_t* row = bits + (size_t)i * words;
        for (uint32_t wd = i >> 5; wd < words; ++wd) {
            uint32_t v = row[wd];
            while (v) {
                uint32_t bit = 0;
                while (!((v >> bit) & 1u)) ++bit;
                v &= v - 1;
                const uint32_t ra = uf_find(par, i), rb = uf_find(par, wd * 32 + bit);
                if (ra != rb) par[ra > rb ? ra : rb] = ra < rb ? ra : rb;
            }
        }
    }
    for (uint32_t i = 0; i < m; ++i) size[uf_find(par, i)]++;
    double sh = 0, si = 0;
    uint32_t top = 0, rich = 0, big = 0;
    for (uint32_t i = 0; i < m; ++i) {
        if (!size[i]) continue;
        ++rich;
        if (size[i] >= 2) ++big;
        if (size[i] > top) top = size[i];
        const double p = (double)size[i] / (double)m;
        sh -= p * log(p);
        si += p * p;
    }
    out->richness = rich;
    out->clusters2 = big;
    out->shannon = sh;
    out->simpson = 1.0 - si;
    out->top_share = (double)top / (double)m;
    rc = 0;
done:
    free(a); free(keys); free(g); free(bits); free(par); free(size);
    return rc;
}

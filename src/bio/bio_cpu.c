/* bio_cpu.c - CPU reference of every GPU stage. It calls the same bio_core.h
 * functions; per-cell and per-agent stages run on the job pool with the same
 * commutative atomics, scans run serially, so the result is bit-identical to
 * the GPU (test T2). */
#include "bio_internal.h"
#include "../jobs.h"
#include <stdlib.h>
#include <string.h>

typedef struct CpuCtx { BioWorld* w; int stage; int arg; int64_t* out; } CpuCtx;

static void flush_acc(int64_t* dst, const int64_t* acc, int n) {
    for (int k = 0; k < n; ++k) if (acc[k]) BIO_ADD64(&dst[k], acc[k]);
}
/* both accumulator blocks of the globals (kernels index them as one: BACC_X) */
static void flush_globals(BioWorld* w, const int64_t* acc) {
    flush_acc(w->G->acc, acc, BACC_N);
    flush_acc(w->G->acc2, acc + BACC_N, BACC2_N);
}

static void job_stage(void* vctx, uint32_t b, uint32_t e, uint32_t worker) {
    (void)worker;
    CpuCtx* x = (CpuCtx*)vctx;
    BioWorld* w = x->w;
    int64_t acc[BIO_STAT_N + BACC_TOT];   /* large enough for either use */
    memset(acc, 0, sizeof acc);
    switch (x->stage) {
    case BK_ENV:
        for (uint32_t c = b; c < e; ++c) bio_cell_env(w, (int32_t)c, acc);
        flush_globals(w, acc);
        break;
    case BK_SUMMARY: for (uint32_t i = b; i < e; ++i) bio_agent_summary(w, (int32_t)i); break;
    case BK_THINK:   for (uint32_t i = b; i < e; ++i) if (bio_think_due(w, (int32_t)i)) bio_think_run(w, (int32_t)i); break;
    case BK_PRE:     for (uint32_t i = b; i < e; ++i) bio_agent_pre(w, (int32_t)i); break;
    case BK_SUPPLY:  for (uint32_t c = b; c < e; ++c) bio_cell_supply(w, (int32_t)c); break;
    case BK_METAB:
        for (uint32_t i = b; i < e; ++i) bio_agent_metab(w, (int32_t)i, acc);
        flush_globals(w, acc);
        break;
    case BK_FEED:
        for (uint32_t i = b; i < e; ++i) bio_agent_feed(w, (int32_t)i, acc);
        flush_globals(w, acc);
        break;
    case BK_BIRTH:   for (uint32_t i = b; i < e; ++i) bio_agent_birth(w, (int32_t)i); break;
    case BK_FREE:    for (uint32_t i = b; i < e; ++i) bio_agent_free(w, (int32_t)i); break;
    case BK_STATS:
        if (x->arg) {
            for (uint32_t c = b; c < e; ++c) bio_cell_stats(w, (int32_t)c, acc + BS_FLD);
        } else {
            for (uint32_t i = b; i < e; ++i) {
                bio_agent_stats(w, (int32_t)i, acc);
                if (w->flags[i] & BA_ALIVE)
                    for (int k = 0; k < 64; ++k) acc[BS_GENE + k] += bio_agent_gene(w, (int32_t)i, k);
            }
        }
        flush_acc(x->out, acc, BIO_STAT_N);
        break;
    default: break;
    }
}

void bio_cpu_scan(BioWorld* w, int mode) {
    const uint32_t n = w->n_slots, nb = n ? (n + BIO_SCAN_BLOCK - 1) / BIO_SCAN_BLOCK : 1;
    uint32_t run0 = 0, run1 = 0;
    for (uint32_t blk = 0; blk < nb; ++blk) {
        uint32_t loc = 0;
        const uint32_t e = (blk + 1) * BIO_SCAN_BLOCK < n ? (blk + 1) * BIO_SCAN_BLOCK : n;
        for (uint32_t i = blk * BIO_SCAN_BLOCK; i < e; ++i) {
            const uint32_t v = mode == 0 ? bio_scan_bits(w->flags[i]) : ((w->flags[i] & BA_ALIVE) ? 1u : 0u);
            if (mode == 0) w->pre_bd[i] = loc; else w->pre_a[i] = loc;
            loc += v;
        }
        if (mode == 0) {
            w->boff[blk * 3 + 0] = run0; w->boff[blk * 3 + 1] = run1;
            run0 += loc & 0xFFFFu; run1 += loc >> 16;
        } else {
            w->boff[blk * 3 + 2] = run0;
            run0 += loc;
        }
    }
    if (mode == 0) { w->boff[nb * 3 + 0] = run0; w->boff[nb * 3 + 1] = run1; }
    else w->boff[nb * 3 + 2] = run0;
}

void bio_cpu_gather(BioWorld* w, void* src, void* dst, int elem, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        if (!(w->flags[i] & BA_ALIVE)) continue;
        const uint32_t j = w->pre_a[i] + w->boff[(i / BIO_SCAN_BLOCK) * 3 + 2];
        if (elem == 4) ((uint32_t*)dst)[j] = ((const uint32_t*)src)[i];
        else ((uint64_t*)dst)[j] = ((const uint64_t*)src)[i];
    }
}

void bio_cpu_stats(BioWorld* w, int64_t st[BIO_STAT_N]) {
    CpuCtx x = { w, BK_STATS, 0, st };
    jobs_parallel_for(w->n_slots, 0, job_stage, &x);
    x.arg = 1;
    jobs_parallel_for((uint32_t)w->n_cells, 0, job_stage, &x);
}

void bio_cpu_stage(BioWorld* w, int stage, int arg) {
    CpuCtx x = { w, stage, arg, NULL };
    switch (stage) {
    case BK_GLOBALS: bio_globals_update(w); break;
    case BK_ENV: case BK_SUPPLY:
        jobs_parallel_for((uint32_t)w->n_cells, 0, job_stage, &x);
        break;
    case BK_SUMMARY: case BK_THINK: case BK_PRE: case BK_METAB: case BK_FEED: case BK_BIRTH: case BK_FREE:
        jobs_parallel_for(w->n_slots, 0, job_stage, &x);
        break;
    case BK_THINK_LIST: break;    /* the CPU tests due agents inside BK_THINK */
    case BK_SCAN_A: bio_cpu_scan(w, arg); break;
    default: break;
    }
}

void bio_cpu_sort_perm(BioWorld* w, uint32_t* perm) {
    const int32_t nc = w->n_cells;
    uint32_t* off = (uint32_t*)calloc((size_t)nc + 1, 4);
    if (!off) return;
    for (uint32_t i = 0; i < w->n_slots; ++i) if (w->flags[i] & BA_ALIVE) off[w->cell[i] + 1]++;
    for (int32_t c = 0; c < nc; ++c) off[c + 1] += off[c];
    for (uint32_t i = 0; i < w->n_slots; ++i) if (w->flags[i] & BA_ALIVE) perm[off[w->cell[i]]++] = i;
    free(off);
}

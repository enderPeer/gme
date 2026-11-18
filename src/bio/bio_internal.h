/* bio_internal.h - private declarations shared by the bio core sources
 * (bio_sim.c, bio_cpu.c, bio_io.c, bio_test.c). Not part of the contract. */
#ifndef GME_BIO_INTERNAL_H
#define GME_BIO_INTERNAL_H

#include "bio_api.h"
#include "bio_core.h"
#include "../rtcuda.h"
#include <stdio.h>

/* kernels / CPU stages */
enum { BK_GLOBALS = 0, BK_ENV, BK_SUMMARY, BK_PRE, BK_SUPPLY, BK_METAB, BK_SCAN_A, BK_SCAN_B,
       BK_BIRTH, BK_FREE, BK_GATHER32, BK_GATHER64, BK_STATS,
       BK_SORT_COUNT, BK_CSCAN_A, BK_CSCAN_B, BK_CSCAN_C, BK_SORT_SCATTER, BK_SORT_FIX, BK_PERMUTE32, BK_PERMUTE64,
       BK_THINK_LIST, BK_THINK, BK_N };

/* one agent as a plain record (seeding, migrants, checkpoints) */
typedef struct BioAgentRec {
    int32_t  x, z, C, S, E, N, P, gut, W;
    uint32_t age, flags;
    int32_t  rbase, elast;
    uint32_t uid, lineage, parent, bstep;
    uint64_t hid, out, dW[BIO_OUT];
    BioGenome g;
} BioAgentRec;

/* agent arrays that persist across steps (compacted together), in a fixed order */
#define BIO_AGENT_ARRAYS 30      /* 20 x 4 B, hid, out, 8 dW rows */

struct BioSim {
    BioConfig  cfg;
    const BioEnv* env;
    uint32_t   env_crc;
    uint64_t   planet_hash;
    int        gpu;              /* 1 = CUDA device, 0 = CPU reference */
    RtDevice   dev;
    RtModule*  mod;
    RtFunc     k[BK_N];
    BioWorld   w;                /* device pointers (GPU) or host pointers (CPU) */
    BioGlobals Gh;               /* GPU: last downloaded copy */
    BioTables  tables;
    uint32_t   slots;            /* agent array length = 2 * cap */
    uint32_t   n_live;
    uint64_t   births, deaths;   /* cumulative */
    int64_t    init_total[5];    /* C N S Fe P at creation */
    uint32_t   dry_cells, water_cells;
    /* gates (observed only) */
    uint32_t   gates;
    uint64_t   g0_since, g1_since;
    int64_t    g1_prev_photo, g1_prev_fix;
    uint64_t   g1_prev_step;
    int64_t    g2_prev_oxy;
    double     steps_per_s;
    /* allocations */
    void*      mem[96];
    int        n_mem;
    void*      scratch;          /* gather scratch, 8 B x slots */
    int64_t*   part_host;
    uint32_t   last_scan_tot[3];
};

/* bio_sim.c */
void   bio_set_error(const char* fmt, ...);
int    bio_copy_out(BioSim* s, void* dst, const void* src, size_t bytes);   /* sim memory -> host */
int    bio_copy_in(BioSim* s, void* dst, const void* src, size_t bytes);    /* host -> sim memory */
void   bio_agent_array(const BioWorld* w, int k, void** p, int* elem);      /* k < BIO_AGENT_ARRAYS */
int    bio_append_agents(BioSim* s, BioAgentRec* a, uint32_t n, int fresh); /* fresh: new uids + import ledger */
int    bio_download_agents(BioSim* s, BioAgentRec** out, uint32_t* n);    /* live agents, index order */
int    bio_compact(BioSim* s, int sort);   /* sort: order by cell (stable) */
void   bio_prof_report(void);       /* GME_BIOPROF=1: per-stage times since the last report */
int    bio_g_get(BioSim* s);            /* device globals -> s->Gh */
int    bio_g_put(BioSim* s);            /* s->Gh -> device globals */
int    bio_fields_get(BioSim* s, int64_t* dst);   /* BF_N x n_cells */
int    bio_fields_put(BioSim* s, const int64_t* src);
void   bio_protocell(BioGenome* g, uint32_t salt);
int64_t bio_stats_raw(BioSim* s, int64_t st[BIO_STAT_N]);                 /* fills stats partials */
uint32_t bio_crc32(uint32_t crc, const void* p, size_t n);
uint64_t bio_fnv64(uint64_t h, const void* p, size_t n);
#define BIO_FNV_INIT 0xCBF29CE484222325ull

/* bio_cpu.c: CPU reference for each stage (same bio_core.h functions) */
void   bio_cpu_stage(BioWorld* w, int stage, int arg);
void   bio_cpu_scan(BioWorld* w, int mode);
void   bio_cpu_gather(BioWorld* w, void* src, void* dst, int elem, uint32_t n);
void   bio_cpu_stats(BioWorld* w, int64_t st[BIO_STAT_N]);
void   bio_cpu_sort_perm(BioWorld* w, uint32_t* perm);   /* stable order of live agents by cell */

#endif

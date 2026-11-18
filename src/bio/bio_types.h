/* bio_types.h - shared types of the life simulation (docs/biosphere-spec.md).
 *
 * Compiled three ways: as C by the host compiler (CPU reference + host driver),
 * and, concatenated with bio_core.h and bio_kernels.cu, as C++ by NVRTC (no
 * project includes there). Integer-only by design: the lint test T1 rejects
 * real-number type tokens in bio_core.h and bio_kernels.cu, and this header
 * keeps to the same rule. */
#ifndef GME_BIO_TYPES_H
#define GME_BIO_TYPES_H

/* ---- integer shim ----------------------------------------------------------- */
#ifdef __CUDACC_RTC__
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef short              int16_t;
typedef unsigned short     uint16_t;
typedef int                int32_t;
typedef unsigned int       uint32_t;
typedef long long          int64_t;
typedef unsigned long long uint64_t;
#define BIO_FN static __device__ __forceinline__
#define BIO_GPU 1
/* atomics on global or shared memory; all are commutative, so results do not
 * depend on thread order */
#define BIO_ADD64(p, v) atomicAdd((unsigned long long*)(p), (unsigned long long)(int64_t)(v))
#define BIO_ADD32(p, v) atomicAdd((unsigned int*)(p), (unsigned int)(v))
#define BIO_DP4A(a, b, c) __dp4a((int)(a), (int)(b), (int)(c))
#define BIO_UNROLL _Pragma("unroll")
#else
#include <stdint.h>
#include <stddef.h>
#define BIO_FN static inline
#define BIO_GPU 0
#if defined(_MSC_VER)
#include <intrin.h>
#define BIO_ADD64(p, v) ((void)_InterlockedExchangeAdd64((volatile long long*)(p), (long long)(v)))
#define BIO_ADD32(p, v) ((void)_InterlockedExchangeAdd((volatile long*)(p), (long)(v)))
#else
#define BIO_ADD64(p, v) ((void)__atomic_fetch_add((int64_t*)(p), (int64_t)(v), __ATOMIC_RELAXED))
#define BIO_ADD32(p, v) ((void)__atomic_fetch_add((uint32_t*)(p), (uint32_t)(v), __ATOMIC_RELAXED))
#endif
#define BIO_UNROLL
static inline int bio_dp4a_c(int a, int b, int c) {
    for (int k = 0; k < 4; ++k) c += (int)(int8_t)(a >> (8 * k)) * (int)(int8_t)(b >> (8 * k));
    return c;
}
#define BIO_DP4A(a, b, c) bio_dp4a_c((int)(a), (int)(b), (int)(c))
#endif

/* accumulators private to one thread (GPU, reduced by warp shuffles in the
 * kernels) or to one CPU job chunk: plain adds, exact and order-free */
#define BIO_ACC(p, v) (*(int64_t*)(p) += (int64_t)(v))

/* ---- sizes -------------------------------------------------------------------- */
#define BIO_IN  16
#define BIO_H   8
#define BIO_OUT 8
#define BIO_GENOME_BYTES 384
#define BIO_MUT_BYTES    336          /* body 64 + brain 272 */
#define BIO_CELL_HARDCAP 64
#define BIO_BLOCK        256          /* threads per block for most kernels */
#define BIO_SCAN_BLOCK   1024         /* items per scan block */
#define BIO_ATMG         7            /* = BIO_ATM_N of bio_api.h */
#define BIO_LAI_BANDS    8
#define BIO_KIN_BUCKETS  16

/* genome body genes (spec 3.2) */
enum {
    BG_E_METH = 0, BG_E_ANOX, BG_E_OXY, BG_E_FERM, BG_E_SULFRED, BG_E_AERO, BG_E_METHOX, BG_E_FEOX,
    BG_SIZE = 8, BG_STRUCT, BG_HOLDFAST, BG_CUTICLE, BG_ROOT, BG_BUOY, BG_MOTOR, BG_MOUTH,
    BG_ARMOR = 16, BG_PIGMENT, BG_HUE, BG_UVSHIELD, BG_SENSE, BG_REPRO_E, BG_OFFS, BG_DISPERSE,
    BG_MUT = 24, BG_ETA, BG_DECAY, BG_PLASTIC, BG_NEURONS, BG_THINK, BG_TOPT, BG_TTOL,
    BG_O2TOL = 32, BG_SEX, BG_LIFESPAN, BG_KINSHARE, BG_STORE,
    BG_NEUTRAL0 = 48
};

/* metabolic pathways (same order as BioStats.path_flux) */
enum { BP_METH = 0, BP_ANOX, BP_OXY, BP_FERM, BP_SULFRED, BP_AERO, BP_METHOX, BP_FEOX, BP_N };

/* dynamic fields, int64 mmol per cell (SOILW in litres). The first BIO_POOLS
 * are pools agents can draw from (fair share); the first BIO_DIFFUSING diffuse. */
enum { BF_DIC = 0, BF_DOC, BF_NH4, BF_PO4, BF_FE2, BF_H2S, BF_SO4, BF_O2, BF_CH4, BF_H2,
       BF_SOILW, BF_DET, BF_N };
#define BIO_POOLS     11
#define BIO_DIFFUSING 10

/* per-agent request scratch (written by k_pre, read by k_metab) */
/* BR_FIXA / BR_FIXO: the parts of BR_FIX reduced with H2S (ANOX) or water (OXY) */
enum { BR_PATH = 0, BR_FIX = 8, BR_ASSIM, BR_NH4, BR_PO4, BR_WATER, BR_AUX, BR_FIXA, BR_FIXO, BR_N };

/* agent flags */
#define BA_ALIVE   1u
#define BA_WET     2u
#define BA_LAND    4u
#define BA_DEAD    8u     /* died this step (genome slot not yet freed) */
#define BA_BIRTH   16u    /* reproduces this step */
#define BA_THOUGHT 32u    /* ran its brain this step */

/* cached traits (derived from the genome at birth): bits 0-3 kin bucket,
 * 4-11 pigment, 12 holdfast >= 128, 13 motile, 14 phototroph-capable,
 * 16-21 think period - 1 */
#define BT_KIN(t)    ((t) & 15u)
#define BT_PIG(t)    (((t) >> 4) & 255u)
#define BT_HOLD      (1u << 12)
#define BT_MOTILE    (1u << 13)
#define BT_PHOTO     (1u << 14)
#define BT_PERIOD(t) ((((t) >> 16) & 63u) + 1u)

/* 64-bit accumulators in BioGlobals.acc (all exact integer sums) */
enum {
    BACC_PATH = 0,                    /* 8 pathway fluxes (cumulative) */
    BACC_FIX = 8,                     /* C fixed from DIC (GPP) */
    BACC_ASSIM,                       /* C assimilated from DOC */
    BACC_RESP,                        /* C returned to DIC by agents (pathways + catabolism) */
    BACC_DEATH_C,                     /* C of dead agents to DET */
    BACC_SED_C, BACC_SED_S, BACC_SED_FE,          /* sediments (part of the totals) */
    BACC_VENT_C, BACC_VENT_S, BACC_VENT_FE,       /* sources (ledger) */
    BACC_WEATH_C, BACC_WEATH_S, BACC_WEATH_FE, BACC_WEATH_P,
    BACC_IMPORT_C, BACC_IMPORT_N, BACC_IMPORT_P,  /* seeding, migrants */
    BACC_ATMSET_C, BACC_ATMSET_N, BACC_ATMSET_S,  /* hub atmosphere edits */
    BACC_WANT,                        /* parents eligible to reproduce this step (birth lottery) */
    BACC_ATMD = 32,                   /* 7 pending atmosphere deltas (env -> box) */
    BACC_N = 40
};

/* ---- genome (spec 3.2), 384 B, immutable for the agent's life ----------------- */
typedef struct BioGenome {
    uint8_t  body[64];
    int8_t   w_ih[BIO_H][BIO_IN];
    int8_t   w_hh[BIO_H][BIO_H];
    int8_t   b_h[BIO_H];
    int8_t   w_ho[BIO_OUT][BIO_H];
    int8_t   b_o[BIO_OUT];
    uint16_t mate_tag;
    uint8_t  pad[46];
} BioGenome;

/* ---- parameters: all integer, fixed point where noted. Defaults in
 * bio_params_default() (bio_core.h); values marked tuned there. ----------------- */
typedef struct BioParams {
    /* sources and sinks, per step per cell (mmol) unless noted */
    int32_t vent_h2, vent_h2s, vent_fe2, vent_dic;      /* times vent/255 */
    int32_t serp_h2_q8;                                 /* seafloor ore: H2 += serp*f_ore/255, q8 */
    int32_t weath_fe_q16, weath_s_q16, weath_p_q16, weath_doc_q16; /* per (f/255) per step, q16 */
    int32_t rain_l;                                     /* litres per land cell per day at the coast */
    int32_t soil_cap_l;                                 /* soil water above this runs off */
    int32_t runoff_q16;                                 /* share of excess water + solutes moved per day */
    int32_t kp_water_q16, kp_soil_q16;                  /* gas exchange per step, q16 */
    int32_t henry[BIO_ATMG];                            /* mmol per m^3 at 1 atm partial pressure */
    int32_t aq_cap[BIO_ATMG];                           /* cap on saturation, mmol per m^3 */
    int32_t k_py;                                       /* pyrite: (H2S*FE2*k_py) >> 40 per m^3 */
    int32_t k_ox_h2s, k_ox_ch4, k_ox_fe;                /* abiotic oxidation, q16 per (mmol/m^3) */
    int32_t burial_shift, mineral_shift;                /* daily DET burial / mineralisation */
    int32_t burial_depth;                               /* m */
    int32_t d_ww, d_ss, d_wl;                           /* diffusion, q16 per neighbour */
    int32_t photo_ch4_q16, photo_h2s_q16;               /* island-box photochemistry per step */
    /* light */
    int32_t par_q12;                                    /* PAR share of the stellar flux */
    int32_t kw_q16;                                     /* water attenuation per m */
    int32_t sla_q20;                                    /* leaf area per mmol C at full pigment, m^2 q20 */
    int32_t eff_anox_q16, eff_oxy_q16;                  /* eu per step per (W/m^2 * m^2 q10), q16 */
    int32_t ktox_h2s;                                   /* OXY sulfide inhibition, mmol/m^3 */
    /* metabolism: vmax per mmol C per step (q16); half saturation (mmol/m^3) */
    int32_t vmax_q16[BP_N];
    int32_t km[BIO_POOLS];
    int32_t yield_q16[BP_N];                            /* eu per unit flux, q16 */
    int32_t mu_q16;                                     /* max growth per step per mmol C */
    int32_t fix_eu, assim_eu;                           /* eu per mmol C */
    int32_t cn, cp;                                     /* C:N and C:P of biomass */
    int32_t water_q16;                                  /* litres of water per mmol C (q16) */
    /* costs (eu per mmol C per step unless noted, q16) */
    int32_t maint_q16, struct_q16, enz_q16, trait_q16;
    int32_t brain_q16, sense_q16, learn_q16;            /* eu per think */
    int32_t o2tox_q16, k_o2;                            /* k_o2: mmol/m^3 */
    int32_t uv_q16;
    int32_t dry_q8;                                     /* W loss per step on dry ground at cuticle 0 */
    int32_t move_q16, move_land_q16;                    /* locomotion */
    int32_t starve_aero_q8, starve_anaer_q8;            /* eu per mmol C catabolised, q8 */
    int32_t copy_cost;                                  /* eu per birth (0.02 eu x 384 B) */
    int32_t hazard_q32;                                 /* background death per step, q32 */
    int32_t kinpath_q32;                                /* kin pathogen max per step, q32 */
    int32_t e_cap_mul;                                  /* E <= e_cap_mul * C + 4096 */
    int32_t drift_q8;                                   /* plankton random walk, m q8 per step */
    int32_t speed_q8;                                   /* motile top speed, m q8 per step */
    int32_t barrier_q8;                                 /* island barrier multiplier (land costs) */
    int32_t atm_scale_ppm;
    int32_t closed_box;                                 /* 1: no vents, weathering, burial, exchange */
    int32_t mut_off;                                    /* 1: no mutation (null control) */
    int32_t learn_off;                                  /* 1: eta forced to 0 (test) */
    int32_t brain_off;                                  /* 1: outputs fixed at the biases (test) */
    int32_t light_off;                                  /* 1: no light (null control) */
    int32_t grow_off;                                   /* 1: no growth or birth (light profile test) */
    int32_t reserved[16];
} BioParams;

/* ---- lookup tables (built by integer routines, uploaded) ------------------------ */
typedef struct BioTables {
    int32_t expn_q16[1024];     /* 65536 * exp(-i/64) */
    int32_t gauss_q16[256];     /* 65536 * exp(-(i/64)^2 / 2) */
    int32_t q10_q16[128];       /* 65536 * 2^((i-64)/10) */
    int32_t sigmoid_q16[256];   /* 65536 / (1 + exp(-(i-128)/32)) */
    int8_t  tanh_q7[256];       /* 127 * tanh((i-128)/64) */
} BioTables;

/* ---- per-step globals, device resident --------------------------------------- */
typedef struct BioGlobals {
    int64_t atm[BIO_ATMG];          /* island box, mmol */
    int64_t acc[BACC_N];            /* exact accumulators (cumulative) */
    int64_t sat_q16[BIO_ATMG];      /* saturation, mmol per m^3 (q16) for this step */
    int64_t dis_cap[BIO_ATMG];      /* max dissolution per cell per step */
    int32_t i0;                     /* surface PAR this step, W/m^2 */
    int32_t uv_q8;                  /* UV dose at the surface, 0..255 */
    int32_t haze_q12;
    int32_t tide;                   /* 0 low, 1 mid, 2 high */
    int32_t hr;                     /* hour of day */
    int32_t day_sin_q12;            /* signed diurnal sine */
    int32_t birth_q16;              /* share of eligible parents allowed to reproduce (capacity guard) */
    uint32_t think_n;               /* agents due to think this step (list in BioWorld.pre_a) */
} BioGlobals;

/* ---- the world as kernels see it (passed BY VALUE to every kernel) ------------
 * Pointers are device pointers on the GPU path and host pointers on the CPU
 * path; both are 8 bytes, so the layout is the same for the host compiler and
 * NVRTC. */
typedef struct BioWorld {
    int32_t  w, h, shift, n_cells;
    uint32_t cap, n_slots;           /* agent array capacity / used slots (live + holes) */
    uint32_t salt, seed;
    uint64_t step;
    int32_t  sea_level, t_surf_ck, s_top, pad0;
    uint32_t k_births, top;          /* births allowed this step; free stack top */
    uint32_t uid_next, n_live;       /* next uid; live agents at the start of the step */
    uint32_t cap_agents, pad1;       /* agent capacity (births refused above 0.95 cap) */
    BioParams P;
    const BioTables* T;
    BioGlobals* G;
    /* static environment */
    const int16_t* elev; const int16_t* depth;
    const uint8_t* wet0; const uint8_t* wet1; const uint8_t* wet2;
    const uint8_t* f_soil; const uint8_t* f_ore; const uint8_t* f_sulf; const uint8_t* f_gran;
    const uint8_t* vent; const uint8_t* flow_dir; const uint16_t* coast;
    /* fields: fld[f * n_cells + c]; fld_new is the other buffer */
    int64_t* fld; int64_t* fld_new;
    /* fair share */
    uint64_t* demand;                /* [pool * n_cells + c] */
    uint32_t* ratio;                 /* [pool * n_cells + c], q16 */
    uint32_t* dflag;                 /* cell has demand this step */
    /* per-cell summaries (cleared by k_env each step) */
    uint32_t* cnt;                   /* agents in the cell */
    uint32_t* lai;                   /* [band * n_cells + c] leaf area, m^2 q10 */
    uint32_t* kin;                   /* [k * n_cells + c], two u16 counters per word */
    /* agents (SoA, index < n_slots) */
    int32_t* x; int32_t* z;          /* metres q8 inside the region */
    uint32_t* cell;
    int32_t* C; int32_t* S; int32_t* E; int32_t* N; int32_t* Pp; int32_t* gut; int32_t* W;
    uint32_t* age; uint32_t* flags; int32_t* rbase; int32_t* elast;
    uint32_t* gslot; uint32_t* uid; uint32_t* tr;
    uint64_t* hid; uint64_t* out;    /* 8 x int8 packed */
    uint64_t* dW;                    /* [row * cap + i], 8 x int8 per row */
    uint32_t* lineage; uint32_t* parent; uint32_t* bstep;
    int32_t* req;                    /* [k * cap + i] */
    uint32_t* pre_bd;                /* scan: birth (low 16) / dead (high 16) prefix within block */
    uint32_t* pre_a;                 /* scan: alive prefix within block */
    uint32_t* boff;                  /* scan: per block exclusive offsets [blk * 3 + {b,d,a}] */
    uint32_t* cscan;                 /* scan over cells: per block sums (sort by cell) */
    /* genome pool */
    BioGenome* gen;
    uint32_t* fstack;
    /* stats partials [block * BIO_STAT_N + k] */
    int64_t* part;
} BioWorld;

/* stats partial layout (per block, int64) */
enum {
    BS_LIVE = 0, BS_MOTILE, BS_SPHOTO, BS_CMAR, BS_CCOAST, BS_CLAND, BS_C, BS_N, BS_P,
    BS_THINK, BS_ABSDW, BS_NEGA, BS_AGENT_N, BS_GENE = 16, BS_FLD = 80, BS_VEG = 92, BS_O2CELLS, BS_NEGF, BIO_STAT_N = 96
};

/* ---- RNG: stateless hash, a copy of gme_hash3 (worldgen.h) -------------------- */
BIO_FN uint32_t bio_hash(int32_t x, int32_t y, int32_t z, uint32_t seed) {
    uint32_t h = seed;
    h ^= (uint32_t)x * 0x8DA6B343u;
    h ^= (uint32_t)y * 0xD8163841u;
    h ^= (uint32_t)z * 0xCB1AB31Fu;
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

/* hash purposes (third argument) */
enum { BH_TIDE = 1, BH_DRIFT, BH_MOVE, BH_SR, BH_DEATH, BH_KINP, BH_BIRTH, BH_MUT, BH_MACRO,
       BH_LEARN, BH_SEED, BH_COST, BH_IMPORT, BH_DISP, BH_TAG };

#endif

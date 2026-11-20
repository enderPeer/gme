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
#define BIO_MAX32(p, v) atomicMax((unsigned int*)(p), (unsigned int)(v))
#define BIO_MIN64(p, v) atomicMin((unsigned long long*)(p), (unsigned long long)(v))
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
/* atomic max / min: commutative, so the result does not depend on thread order */
static inline void bio_max32_c(uint32_t* p, uint32_t v) {
#if defined(_MSC_VER)
    long cur = *(volatile long*)p;
    while ((uint32_t)cur < v) {
        const long seen = _InterlockedCompareExchange((volatile long*)p, (long)v, cur);
        if (seen == cur) break;
        cur = seen;
    }
#else
    uint32_t cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (cur < v && !__atomic_compare_exchange_n(p, &cur, v, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
#endif
}
static inline void bio_min64_c(uint64_t* p, uint64_t v) {
#if defined(_MSC_VER)
    long long cur = *(volatile long long*)p;
    while ((uint64_t)cur > v) {
        const long long seen = _InterlockedCompareExchange64((volatile long long*)p, (long long)v, cur);
        if (seen == cur) break;
        cur = seen;
    }
#else
    uint64_t cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (cur > v && !__atomic_compare_exchange_n(p, &cur, v, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
#endif
}
#define BIO_MAX32(p, v) bio_max32_c((uint32_t*)(p), (uint32_t)(v))
#define BIO_MIN64(p, v) bio_min64_c((uint64_t*)(p), (uint64_t)(v))
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
#define BIO_PSLOTS       4            /* predator offers per cell per step (power of two) */
#define BIO_MSLOTS       2            /* mate offers per cell per step (power of two) */
#define BIO_NB           9            /* light bands: 8 of 45 nm (380-740 nm) + NIR 740-900 nm (spectrum_file.h) */
#define BIO_NHAZE        8
#define BIO_NHUE         16
#define BIO_NDEPTH       33

/* ---- features (BioParams.feat). All off = the simulation of the first public
 * release, bit for bit: every checkpoint written before these existed carries 0
 * and resumes exactly as it would have with the binary that wrote it. ------------- */
#define BIO_FEAT_PRED   1u    /* animals: predation and grazing, gut, trophic level, predator / prey senses */
#define BIO_FEAT_SEX    2u    /* sexual reproduction: crossover with a compatible cellmate (mate tags) */
#define BIO_FEAT_LEARN  4u    /* learning by exploration (replaces the plain Hebbian rule) */
#define BIO_FEAT_MORPH  8u    /* plant shapes: crown class, aspect, leaf angle, porosity, Q band width */
#define BIO_FEAT_HUB    16u   /* per-realm gas exchange ledgers for the planet-wide atmosphere (island hub) */
#define BIO_FEAT_ALL    31u
/* light model (BioParams.light_mode) */
#define BIO_LIGHT_SCALAR   0   /* one PAR number: S_top x par_q12 x haze */
#define BIO_LIGHT_SPECTRAL 1   /* 9 wavelength bands from a GMESPEC1 file (spectrum_crc names it) */

/* genome body genes (spec 3.2) */
enum {
    BG_E_METH = 0, BG_E_ANOX, BG_E_OXY, BG_E_FERM, BG_E_SULFRED, BG_E_AERO, BG_E_METHOX, BG_E_FEOX,
    BG_SIZE = 8, BG_STRUCT, BG_HOLDFAST, BG_CUTICLE, BG_ROOT, BG_BUOY, BG_MOTOR, BG_MOUTH,
    BG_ARMOR = 16, BG_PIGMENT, BG_HUE, BG_UVSHIELD, BG_SENSE, BG_REPRO_E, BG_OFFS, BG_DISPERSE,
    BG_MUT = 24, BG_ETA, BG_DECAY, BG_PLASTIC, BG_NEURONS, BG_THINK, BG_TOPT, BG_TTOL,
    BG_O2TOL = 32, BG_SEX, BG_LIFESPAN, BG_KINSHARE, BG_STORE,
    /* plant morphology (active with BIO_FEAT_MORPH, neutral before): crown class,
     * height : width, leaf angle, crown porosity, width of the pigment's Q band */
    BG_CROWN = 37, BG_ASPECT, BG_LEAFANG, BG_POROSITY, BG_QWIDTH,
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
/* BR_PRED / BR_BITE (predation): written by the prey in k_pre (the predator it
 * offers itself to, the carbon at stake) and again in k_metab (what it tells the
 * predator about itself, the carbon actually lost); read by the predator in k_feed */
enum { BR_PATH = 0, BR_FIX = 8, BR_ASSIM, BR_NH4, BR_PO4, BR_WATER, BR_AUX, BR_FIXA, BR_FIXO, BR_PRED, BR_BITE, BR_N };

/* agent flags */
#define BA_ALIVE   1u
#define BA_WET     2u
#define BA_LAND    4u
#define BA_DEAD    8u     /* died this step (genome slot not yet freed) */
#define BA_BIRTH   16u    /* reproduces this step */
#define BA_THOUGHT 32u    /* ran its brain this step */
#define BA_HUNT    64u    /* offered to attack this step (k_summary .. k_feed) */

/* cached traits (derived from the genome at birth): bits 0-3 kin bucket,
 * 4-11 pigment, 12 holdfast >= 128, 13 motile, 14 phototroph-capable,
 * 16-21 think period - 1 */
#define BT_KIN(t)    ((t) & 15u)
#define BT_PIG(t)    (((t) >> 4) & 255u)
#define BT_HOLD      (1u << 12)
#define BT_MOTILE    (1u << 13)
#define BT_PHOTO     (1u << 14)
#define BT_PERIOD(t) ((((t) >> 16) & 63u) + 1u)
/* second cached trait word (also from the genome at birth): bits 0-7 motor,
 * 8-15 mouth, 16-20 size class (G_SIZE / 10: an individual weighs about 2^class
 * mmol C), 21-23 crown class, 24-26 aspect bin, 27-30 leaf angle class
 * (16 steps), 31 sexual (G_SEX >= 128) */
#define BT2_MOTOR(t)   ((t) & 255u)
#define BT2_MOUTH(t)   (((t) >> 8) & 255u)
#define BT2_SIZEC(t)   (((t) >> 16) & 31u)
#define BT2_CROWN(t)   (((t) >> 21) & 7u)
#define BT2_ASPECT(t)  (((t) >> 24) & 7u)
#define BT2_LEAFANG(t) (((t) >> 27) & 15u)
#define BT2_SEXUAL     (1u << 31)
/* gape: the largest individual (mmol C) an attacker can swallow: its own size x MOUTH / 32 */
#define BT2_GAPE(t)    ((((uint64_t)1 << BT2_SIZEC(t)) * BT2_MOUTH(t)) >> 5)

/* ecology word of an agent (BioWorld.eco; all zero for a newly seeded agent):
 * bits 0-7   trophic level above 1, in 1/16 (0 = a producer's level 1)
 * bits 8-15  share of its organic carbon intake that came from its gut (prey), q8, moving average
 * bits 16-23 share of its meals that were producers on dry land, q8, moving average
 * bits 24-31 generations in a row born to a land animal (saturating) */
#define BE_TROPH(e)    ((e) & 255u)
#define BE_GUTSHARE(e) (((e) >> 8) & 255u)
#define BE_LANDPROD(e) (((e) >> 16) & 255u)
#define BE_LANDGEN(e)  (((e) >> 24) & 255u)
#define BE_MAKE(troph, gut, land, gen) ((uint32_t)(troph) | ((uint32_t)(gut) << 8) | ((uint32_t)(land) << 16) | ((uint32_t)(gen) << 24))

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
/* second accumulator block (BioGlobals.acc2), written only by the features
 * above. Kernels address both blocks through one index space: k < BACC_N is
 * acc[k], BACC_N + j is acc2[j]. */
enum {
    BACC2_XLAND = 0,                  /* 7: cumulative gas flux atmosphere -> land cells (BIO_FEAT_HUB) */
    BACC2_XALL = 7,                   /* 7: cumulative gas flux atmosphere -> all cells */
    BACC2_PRED_C = 14,                /* carbon moved from prey into guts */
    BACC2_KILLS,                      /* prey eaten whole */
    BACC2_BITES,                      /* bites out of colonies (grazing) */
    BACC2_SEXB,                       /* births with a second parent */
    BACC2_GUT_C,                      /* gut carbon respired or assimilated */
    BACC2_N = 24
};
#define BACC_TOT (BACC_N + BACC2_N)
#define BACC_X(j) (BACC_N + (j))      /* combined index of acc2[j] */

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
    int32_t learn_off;                                  /* 1: eta forced to 0 (test); 2: explore but never change a weight (control) */
    int32_t brain_off;                                  /* 1: outputs fixed at the biases (test) */
    int32_t light_off;                                  /* 1: no light (null control) */
    int32_t grow_off;                                   /* 1: no growth or birth (light profile test) */
    /* ---- what used to be reserved[16]; all zero = the first release's behaviour ---- */
    int32_t light_mode;                                 /* BIO_LIGHT_* */
    int32_t spectrum_crc;                               /* crc32 of the GMESPEC1 payload the run uses (bits of a u32) */
    int32_t feat;                                       /* BIO_FEAT_* */
    int32_t eff_sp_anox_q16, eff_sp_oxy_q16;            /* spectral harvest: eu per step per (umol photons/s x eV), q16, set at creation */
    int32_t pred_bite_q10;                              /* bite out of a colony: C_att x MOUTH/255 x this >> 10 */
    int32_t pred_kill_q8;                               /* share of a prey eaten whole that reaches the gut (rest -> DOC) */
    int32_t gut_k_q8;                                   /* gut half saturation: this x C >> 8 */
    int32_t explore_q7;                                 /* exploration: output noise amplitude at ETA 255, q7 */
    int32_t morph_pmax;                                 /* light saturation of a crown: eu per step per m^2 of sunlit leaf */
    int32_t morph_gain_q8;                              /* harvest gain of the crown model (keeps a reference plant's yield) */
    int32_t qwidth_cost_q8;                             /* pigment cost of the widest Q band, relative to the pigment cost, q8 */
    int32_t pred_refuge1;                               /* 1 + hiding places per cell: that many of its non-motile agents cannot be found */
    int32_t learn_shift;                                /* exploration learning: weight step = ETA x reward x nudge x hidden >> this */
    int32_t reserved[2];
} BioParams;

/* ---- lookup tables (built by integer routines, uploaded) ------------------------ */
typedef struct BioTables {
    int32_t expn_q16[1024];     /* 65536 * exp(-i/64) */
    int32_t gauss_q16[256];     /* 65536 * exp(-(i/64)^2 / 2) */
    int32_t q10_q16[128];       /* 65536 * 2^((i-64)/10) */
    int32_t sigmoid_q16[256];   /* 65536 / (1 + exp(-(i-128)/32)) */
    int8_t  tanh_q7[256];       /* 127 * tanh((i-128)/64) */
    /* leaf projection (Goudriaan): k = G(theta) / cos(theta) for 16 leaf angle
     * classes (erect .. flat) and the 24 hours, q12, capped at 8.0; 0 at night.
     * 4096 = flat leaves (every hour). Built by bio_build_tables. */
    int32_t kleaf_q12[16][24];
} BioTables;

/* ---- spectral light tables (GMESPEC1, spectrum_file.h): loaded by the host,
 * uploaded, read-only. Integer only. --------------------------------------------- */
typedef struct BioSpectral {
    int32_t surf_q16[BIO_NHAZE][24][BIO_NB];   /* band irradiance on a horizontal surface / S_top */
    int32_t water_q16[BIO_NB][BIO_NDEPTH];     /* band transmittance of 0..32 m of water */
    int32_t kw_q16[BIO_NB];                    /* attenuation per m below 32 m (bio_light's kw scale) */
    int32_t phot_q16[BIO_NB];                  /* umol photons per J */
    int32_t ev_q16[BIO_NB];                    /* mean photon energy, eV */
    int32_t uphill_q16[BIO_NB][2];             /* usable share of a band's photons: [0] OXY, [1] ANOX */
    int32_t eps_q12[BIO_NHUE][BIO_NB];         /* pigment shape per hue bin */
} BioSpectral;

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
    /* ---- appended after the first release (a checkpoint of that release stops
     * above this line and loads with zeros below) ---- */
    int64_t acc2[BACC2_N];          /* second accumulator block (BACC2_*) */
    int32_t i0b_q8[BIO_NB];         /* spectral: band irradiance at the surface this step, W/m^2 q8 */
    int32_t pad_g[3];
} BioGlobals;
#define BIO_GLOBALS_V2_BYTES 520   /* sizeof(BioGlobals) of the first release */

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
    /* ---- added with the features (bio_types.h BIO_FEAT_*) ---- */
    const BioSpectral* SP;           /* spectral light tables; valid when P.light_mode == BIO_LIGHT_SPECTRAL */
    uint32_t* pslot;                 /* [k * n_cells + c], k < BIO_PSLOTS: predator offers of this step */
    uint32_t* mslot;                 /* [k * n_cells + c], k < BIO_MSLOTS: mate offers of this step */
    uint32_t* thr;                   /* per cell: reach (largest prey body) of its strongest attacker this step */
    uint32_t* nmot;                  /* per cell: movers (a real motor and no photosynthesis, bio_mover) this step */
    uint64_t* claim;                 /* per agent: the prey that offered itself to this attacker (minimum wins) */
    uint32_t* tr2;                   /* per agent: second cached trait word (BT2_*) */
    uint32_t* eco;                   /* per agent: ecology word (BE_*) */
    uint64_t* expl;                  /* per agent: exploration of the last think, 8 x int8 (BIO_FEAT_LEARN) */
} BioWorld;

/* stats partial layout (per block, int64) */
enum {
    BS_LIVE = 0, BS_MOTILE, BS_SPHOTO, BS_CMAR, BS_CCOAST, BS_CLAND, BS_C, BS_N, BS_P,
    BS_THINK, BS_ABSDW, BS_NEGA,
    BS_CONSUMER,          /* agents living mostly on prey (gut share > 1/2) */
    BS_TROPH,             /* sum of their trophic level above 1, in 1/16 */
    BS_LANDANIM,          /* gate G4: creatures that are fast consumers of land producers on dry land, 5+ generations */
    BS_SEXUAL,            /* agents with G_SEX >= 128 */
    BS_CREATURE,          /* creatures (bio_creature): what the engine is given to draw as animals */
    BS_AGENT_N, BS_GENE = 24, BS_FLD = 88, BS_VEG = 100, BS_O2CELLS, BS_NEGF, BIO_STAT_N = 104
};
/* A creature moves at least this fast at full effort (MOTOR 16: 1 m per step) unless it lives on prey. */
#define BIO_CREATURE_MOTOR 16

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
       BH_LEARN, BH_SEED, BH_COST, BH_IMPORT, BH_DISP, BH_TAG,
       /* purposes of the features; far from the ranges the first release derives
        * from the ones above (BH_LEARN * 64 + ..., BH_MUT + 64, BH_SR + 16, ...) */
       BH_PSLOT = 4096, BH_PMASK, BH_PREY, BH_CLAIM, BH_ROLL, BH_MSLOT, BH_MMASK, BH_MATE, BH_CROSS, BH_EXPL,
       BH_HIDE };

#endif

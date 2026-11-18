/* bio_api.h - CONTRACT for the life simulation (docs/biosphere-spec.md).
 *
 *   bio core  (bio_types.h, bio_core.h, bio_kernels.cu, bio_sim.c, bio_cpu.c)
 *             implements everything declared here except the BioEnv builders.
 *   bio glue  (bio_env.c, bio_main.c, cluster scripts) builds BioEnv from the
 *             planet + terrain, drives runs, exports biomaps.
 *
 * All simulation state is integer, so a run is bit-identical on any GPU and on
 * the CPU reference (device = -1), and checkpoints resume exactly. */
#ifndef GME_BIO_API_H
#define GME_BIO_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* atmosphere gases (island box + planet file) */
enum { BIO_ATM_H2S = 0, BIO_ATM_CH4, BIO_ATM_NH3, BIO_ATM_CO2, BIO_ATM_O2, BIO_ATM_N2, BIO_ATM_H2, BIO_ATM_N };

/* stage gates (observed only, never trigger behaviour) - bit k of BioStats.gates */
enum { BIO_G0_ESTABLISHED = 0, BIO_G1_MARINE_PHOTO, BIO_G2_OXYGENIC, BIO_G3_LAND_PLANTS, BIO_G4_LAND_ANIMALS, BIO_G5_OXIDATION };

/* ---- environment of one island (static per run) ---------------------------- */
typedef struct BioEnv {
    int32_t  w, h, shift;          /* grid cells; one cell = 2^shift m square (spec: 1024 x 1024, shift 4) */
    int32_t  x0, z0;               /* world voxel coords of the corner of cell (0, 0) */
    int32_t  sea_level;            /* voxel y, from home.planet */
    int32_t  t_surf_ck;            /* mean surface temperature, centi-kelvin */
    int32_t  s_top_wm2;            /* stellar flux at the top of the atmosphere, W/m^2 */
    int32_t  gravity_q8;           /* surface gravity in g, q8 (256 = 1 g) */
    int32_t  organic_ppm;          /* crust organics, ppm by mass */
    int32_t  atm_ppm[BIO_ATM_N];   /* initial atmosphere mole fractions, ppm (sum ~ 1e6) */
    uint32_t seed;                 /* world (terrain) seed */
    uint64_t planet_hash;
    int32_t  island;
    uint32_t crc;                  /* crc32 of the per-cell payload, see bio_env_crc() */
    /* per cell, row-major [z * w + x]; allocated by bio_env_alloc */
    int16_t*  elev;      /* mean surface height (voxel y) */
    int16_t*  depth;     /* sea_level - elev (<= 0 on land) */
    uint8_t*  wet[3];    /* low / mid / high tide: share of the cell under water, 0..255 */
    uint8_t*  f_soil;    /* 0..255 share of soil in the top layers */
    uint8_t*  f_ore;     /* iron ore */
    uint8_t*  f_sulf;    /* sulfides */
    uint8_t*  f_gran;    /* granite (vs basalt) */
    uint8_t*  vent;      /* hydrothermal source strength 0..255 (0 = none) */
    uint8_t*  flow_dir;  /* D8 downhill neighbour 0..7, 255 = sink or sea */
    uint8_t*  env_bin;   /* gme_env_bin() at the cell centre (engine fallback key) */
    uint16_t* coast;     /* distance to the nearest ocean cell, in cells */
} BioEnv;

int      bio_env_alloc(BioEnv* e, int32_t w, int32_t h);   /* zeroed arrays; 0 ok   (bio core) */
void     bio_env_free(BioEnv* e);                           /*                       (bio core) */
uint32_t bio_env_crc(const BioEnv* e);                      /* crc32 of all arrays   (bio core) */

/* ---- run configuration ------------------------------------------------------ */
typedef struct BioConfig {
    int      device;          /* CUDA ordinal, or -1 = CPU reference implementation */
    uint32_t cap;             /* max agents; 0 = size from free memory */
    uint32_t salt;            /* RNG salt (a rollback after extinction changes it) */
    double   duty;            /* fraction of wall time the GPU is busy (share with other jobs), 0..1 */
    int32_t  barrier_q8;      /* island land-barrier multiplier for desiccation/UV/locomotion, 256 = 1.0 */
    int32_t  atm_scale_ppm;   /* island atmosphere box scale in ppm of the real column (spec 1e-4 = 100) */
    int32_t  closed_box;      /* 1 = no vents, weathering, burial or gas exchange (conservation test) */
} BioConfig;

void bio_config_default(BioConfig* c);                      /* (bio core) */

/* ---- statistics (every call computes a fresh snapshot) ---------------------- */
#define BIO_PATHWAYS 8   /* METH, ANOX, OXY, FERM, SULFRED, AERO, METHOX, FEOX */
typedef struct BioStats {
    uint64_t step;               /* 1 step = 1 simulated hour */
    double   sim_years;
    uint64_t live, motile, sessile_photo;
    uint64_t births, deaths;     /* cumulative */
    int64_t  carbon_marine, carbon_coastal, carbon_land;  /* mmol C held by agents */
    int64_t  path_flux[BIO_PATHWAYS];                     /* cumulative mmol through each pathway */
    int64_t  atm[BIO_ATM_N];                              /* island box, mmol */
    int64_t  total[4];           /* whole-system C, N, S, Fe (mmol) incl. fields, agents, sediments, atm */
    int64_t  ledger[4];          /* expected totals from the source/sink ledgers; total - ledger must be 0 */
    uint32_t mean_gene[64];      /* mean body gene values across live agents (0..255) */
    double   mean_abs_dw;        /* mean |learned weight delta| */
    double   think_share;        /* share of live agents that thought this step */
    uint32_t land_cells_vegetated;   /* dry cells with sessile phototroph cover */
    uint32_t gates;              /* bit k = gate k reached */
    double   steps_per_s;        /* wall-clock speed of the last bio_sim_run */
} BioStats;

/* ---- simulation ---------------------------------------------------------------- */
typedef struct BioSim BioSim;

const char* bio_error(void);
BioSim*  bio_sim_create(const BioEnv* env, const BioConfig* cfg);   /* NULL on error */
int      bio_sim_seed(BioSim* s, uint32_t n_protocells);            /* founders in habitable cells */
int      bio_sim_run(BioSim* s, uint64_t steps);                    /* advance; honours cfg.duty */
int      bio_sim_stats(BioSim* s, BioStats* out);
uint64_t bio_sim_hash(BioSim* s);                                   /* FNV-64 of the whole state */
int      bio_sim_save(BioSim* s, const char* path);                 /* checkpoint, atomic (tmp + rename) */
BioSim*  bio_sim_load(const char* path, const BioEnv* env, const BioConfig* cfg); /* refuses planet/env mismatch */
/* Vegetation of this island as packed biomap cells (biomap.h): cells_out has
 * env->w * env->h entries; fallback_out[64] gets the dominant cell per env_bin. */
int      bio_sim_vegetation(BioSim* s, uint32_t* cells_out, uint32_t fallback_out[64], uint8_t palette[16][3]);
/* Migration: write up to n_per_class best genomes per habitat class / import a
 * file written by another island (placed into matching habitat cells). */
int      bio_sim_export_migrants(BioSim* s, const char* path, int n_per_class);
int      bio_sim_import_migrants(BioSim* s, const char* path);
/* Atmosphere exchange with the hub (mmol in this island's box). */
int      bio_sim_get_atm(BioSim* s, int64_t atm[BIO_ATM_N]);
int      bio_sim_set_atm(BioSim* s, const int64_t atm[BIO_ATM_N]);
void     bio_sim_destroy(BioSim* s);

/* Test suite (bio_test.c): returns number of failed tests; device -1 = CPU only. */
int      bio_run_tests(int device, int verbose);

#ifdef __cplusplus
}
#endif
#endif

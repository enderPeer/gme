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

struct SpectrumFile;   /* spectrum_file.h */

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
    /* Features (BIO_F_*). A new run gets `features`; a loaded run keeps the ones in its
     * checkpoint and adds `enable` (an operator intervention: log it). All off is the
     * first release, bit for bit, and that is how a checkpoint of that release resumes. */
    uint32_t features;
    uint32_t enable;
    /* Light in 9 wavelength bands (GMESPEC1 of the same planet; spectrum_file.h). New
     * run: used when set. Loaded run: required when the checkpoint used one, and its
     * CRC must match; a scalar-light checkpoint switches only with BIO_F_SPECTRAL in
     * `enable`. NULL = the scalar PAR model. */
    const struct SpectrumFile* spectrum;
} BioConfig;

/* feature bits of BioConfig.features / .enable and BioStats.features */
#define BIO_F_ANIMALS  1u    /* predation and grazing, gut, trophic levels, predator / prey senses */
#define BIO_F_SEX      2u    /* crossover with a compatible cellmate (mate tags) */
#define BIO_F_LEARN    4u    /* learning by exploration */
#define BIO_F_MORPH    8u    /* plant shapes: crown class, aspect, leaf angle, porosity, Q band width */
#define BIO_F_HUB      16u   /* gas exchange ledgers by realm, for the planet-wide atmosphere */
#define BIO_F_ALL      31u
#define BIO_F_SPECTRAL 256u  /* light in wavelength bands (needs BioConfig.spectrum) */

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
    /* features (zero while they are off) */
    uint32_t features;           /* BIO_F_* in force */
    uint64_t consumers;          /* agents living mostly on prey (gut share of their organic carbon > 1/2) */
    uint64_t creatures;          /* a motor and 1 m per hour at full effort, or living on prey; a phototroph with a holdfast only if both */
    uint64_t land_animals;       /* gate G4 candidates: motile consumers of land producers on dry land, 5+ generations */
    uint64_t sexual;             /* agents with G_SEX >= 128 */
    double   mean_troph;         /* mean trophic level of the consumers (1 = producer, 2 = grazer, ...) */
    int64_t  kills, bites;       /* cumulative: prey eaten whole / bites out of colonies */
    int64_t  pred_carbon;        /* cumulative mmol C moved from prey into guts */
    int64_t  sexual_births;      /* cumulative births with two parents */
    int64_t  xch_all[BIO_ATM_N]; /* cumulative gas uptake by all cells from the atmosphere, mmol (BIO_F_HUB) */
    int64_t  xch_land[BIO_ATM_N];/* the part taken up by land cells */
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
uint32_t bio_sim_features(const BioSim* s);                         /* BIO_F_* in force (incl. BIO_F_SPECTRAL) */
void     bio_sim_destroy(BioSim* s);

/* ---- what the features export ------------------------------------------------- */
/* Vegetation with morphology (biomap.h version 2): morph_out (n cells) and
 * fallback_morph_out[64] get BIOMORPH words where the plant-shape feature is on,
 * zeros otherwise; either may be NULL. bio_sim_vegetation is this without them. */
int      bio_sim_vegetation2(BioSim* s, uint32_t* cells_out, uint32_t* morph_out, uint32_t fallback_out[64],
                             uint32_t fallback_morph_out[64], uint8_t palette[16][3]);
/* Creatures: agents with a motor that move 1 m per hour or more at full effort
 * or live mostly on prey (a phototroph with a holdfast, which the vegetation
 * export draws, only if it does both); largest bodies first, at most max
 * (fauna.h; out->rec is malloc'd). */
struct Fauna;
int      bio_sim_fauna(BioSim* s, struct Fauna* out, uint32_t max);
/* Who carries a motor gene: out[0] all of them, [1] creatures (as above),
 * [2] phototrophs with a holdfast (plants), [3] plants living mostly on prey,
 * [4] plants that could move 1 m per hour, [5] free but slower and not on prey. */
int      bio_sim_motor_census(BioSim* s, uint32_t out[6]);

/* Species: clusters of similar genomes in a sample of the population (spec 8.3). */
typedef struct BioSpecies {
    uint32_t sample;          /* genomes compared */
    uint32_t richness;        /* clusters */
    uint32_t clusters2;       /* clusters of at least two sampled genomes */
    double   shannon, simpson;/* H = -sum p ln p; 1 - sum p^2 */
    double   top_share;       /* share of the sample in the largest cluster */
} BioSpecies;
int      bio_sim_species(BioSim* s, uint32_t sample, BioSpecies* out);   /* sample 0 = 4096 */

/* Guilds (by genes) and habitats (of a cell). */
enum { BIO_GUILD_PRODUCER = 0, BIO_GUILD_CHEMOTROPH, BIO_GUILD_DECOMPOSER, BIO_GUILD_CONSUMER, BIO_GUILD_N };
enum { BIO_HAB_MARINE = 0, BIO_HAB_COASTAL, BIO_HAB_LAND, BIO_HAB_N };
const char* bio_guild_name(int guild);
const char* bio_habitat_name(int habitat);

/* Seed bank (GMESEED1): evolved genomes with their guild, habitat and biomass.
 * Genomes carry no planet state, so a bank seeds any island of any planet. */
typedef struct BioBankRec {
    uint32_t habitat, guild;  /* BIO_HAB_*, BIO_GUILD_* */
    uint32_t lineage, island;
    int64_t  mass;            /* living mmol C carrying this genome in that habitat */
    uint8_t  genome[384];
} BioBankRec;
int      bio_sim_bank(BioSim* s, BioBankRec** out, uint32_t* n);     /* one record per distinct genome x habitat (free()) */
uint32_t bio_bank_top(BioBankRec* r, uint32_t n, uint32_t per_class);/* merge equal genomes, keep the best per guild x habitat */
int      bio_bank_save(const char* path, const BioBankRec* r, uint32_t n, uint64_t planet_hash);
int      bio_bank_load(const char* path, BioBankRec** out, uint32_t* n, uint64_t* planet_hash);
/* founders of the banked genomes matching guild / habitat (-1 = any) into cells of their habitat; returns the number placed */
int      bio_sim_seed_bank(BioSim* s, const BioBankRec* r, uint32_t n, uint32_t founders, int guild, int habitat);

/* Island hub (bio_hub.c): files of one epoch (a sim-year) in the run folder's
 * outbox/ and inbox/; the hub's pass over its own folder. n_migrants is the
 * number of emigrant genomes per habitat: a quarter of it per guild, the
 * distinct genomes with the most living biomass, one per species-like cluster
 * first. */
int      bio_hub_export(BioSim* s, const char* run_dir, uint64_t epoch, int n_migrants);
int      bio_hub_import(BioSim* s, const char* run_dir, uint64_t epoch, int* n_migrants, int* atm_applied);
uint64_t bio_sim_hub_epoch(const BioSim* s);                         /* last epoch whose inputs were applied */
int      bio_hub_run(const char* hub_dir, const int* islands, int n_islands, int32_t ocean_cover_ppm, char* report, size_t report_n);

/* Test suite (bio_test.c): returns number of failed tests; device -1 = CPU only. */
int      bio_run_tests(int device, int verbose);

#ifdef __cplusplus
}
#endif
#endif

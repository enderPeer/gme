/* bio_env.h - static environment of the life-simulation islands
 * (docs/biosphere-spec.md 2.2-2.4): the region scout, the per-island BioEnv
 * builder, env.bin read/write and the env picture.
 *
 * Everything here derives from the planet file and the integer terrain
 * functions of worldgen.h / flora.h, using integer arithmetic only, so the PC
 * and every rig build bit-identical environments (same crc) from the same
 * home.planet and regions.txt. */
#ifndef GME_BIO_ENV_H
#define GME_BIO_ENV_H

#include "bio_api.h"
#include "../planetfile.h"

#define BIO_ENV_CELLS      1024            /* island grid: 1024 x 1024 cells */
#define BIO_ENV_SHIFT      4               /* of 16 m: 16.4 km per island */
#define BIO_ENV_PAD        128             /* cells around the grid seen by the coast distance */
#define BIO_SCOUT_SAMPLE   64              /* m between height samples */
#define BIO_SCOUT_LATTICE  2048            /* m between candidate window centres ("2 km lattice") */
#define BIO_SCOUT_RANGE    128000          /* centres within +-128 km of (0, 0) */
#define BIO_SCOUT_SEP      20000           /* m, minimum distance between chosen centres */
#define BIO_ISLANDS        6               /* 0-4 rig GPUs, 5 = PC lab */
#define BIO_VENTS_MIN      20
#define BIO_VENT_DEPTH     15              /* vents only below this depth (m) */
#define BIO_VENT_ORE_MIN   49              /* ore + sulfide voxels of the 128 sampled (f_ore + f_sulf > 96) */

/* one scouted window (island); x0, z0 = world corner, side BIO_ENV_CELLS << BIO_ENV_SHIFT */
typedef struct BioRegion {
    int32_t  k, x0, z0;
    int64_t  score;
    int32_t  ocean, shelf, coastland, vents;   /* 64 m samples (of 65536), vents = 16 m cells */
} BioRegion;

/* regions file (life/regions.txt): the islands plus what they were made for */
typedef struct BioRegions {
    uint64_t  planet_hash;
    uint32_t  seed;
    uint32_t  vent_q10;                        /* vent hash threshold of 1024 (planet specific) */
    int       n;
    BioRegion r[BIO_ISLANDS];
} BioRegions;

/* Planet-derived scalars of a BioEnv (sea level, flux, temperature, gravity,
 * organics, atmosphere, seed, planet hash). Arrays untouched. */
void bio_env_planet(BioEnv* e, const HomePlanet* hp);

/* Region scout (2.2): scans the planet and picks BIO_ISLANDS windows.
 * Returns 0 ok; prints progress and the table. png may be NULL. */
int  bio_env_scout(const HomePlanet* hp, BioRegions* out, const char* png);
int  bio_regions_save(const char* path, const BioRegions* r, const HomePlanet* hp);
int  bio_regions_load(const char* path, BioRegions* r);          /* 0 ok */

/* Builds island k's BioEnv (allocates; free with bio_env_free). 0 ok. */
int  bio_env_build(BioEnv* e, const HomePlanet* hp, const BioRegions* rg, int island);
uint32_t bio_env_vents(const BioEnv* e);                         /* vent cells */

/* env.bin: header (GMEENV01) + little-endian payload, crc-checked. load
 * returns 0 ok, -1 cannot read, -2 not a valid env file, -3 crc mismatch. */
int  bio_env_save(const char* path, const BioEnv* e, uint32_t vent_q10, uint32_t* file_crc);
int  bio_env_load(const char* path, BioEnv* e, uint32_t* vent_q10, uint32_t* file_crc);
int  bio_env_png(const char* path, const BioEnv* e);

/* env_bin (gme_env_bin) of the cell of world corner (x, z): evaluated at the
 * cell centre with the height there, as bio_env_build stores it. */
uint32_t bio_env_bin_at(int32_t x, int32_t z, int32_t shift, uint32_t seed, const GmePlanet* p);

/* shared helpers for bio_main.c */
uint32_t bio_file_crc32(uint32_t crc, const void* p, size_t n);
int      bio_mkdirs(const char* path);                           /* mkdir -p; 0 ok */

#endif

/* planetfile.h - the home planet produced by genesis and loaded by the engine
 * (plain key=value text, so it is easy to inspect and diff). */
#ifndef GME_PLANETFILE_H
#define GME_PLANETFILE_H

#include <stdint.h>
#include "worldgen.h"

#define PLANET_MAX_SKY_BODIES 16

/* Atmosphere gases of atm_mix[]: same order as BIO_ATM_* in bio/bio_api.h. */
enum { ATM_H2S = 0, ATM_CH4, ATM_NH3, ATM_CO2, ATM_O2, ATM_N2, ATM_H2, ATM_N };

typedef struct SkyBody {
    float dir[3];     /* direction from the home planet, ecliptic frame */
    float ang;        /* apparent angular radius (exaggerated for visibility) */
    float color[3];
} SkyBody;

typedef struct HomePlanet {
    uint32_t  system_seed;
    int       planet_index;
    GmePlanet gen;                     /* drives voxel generation */
    double    star_mass, star_lum, star_teff;
    double    mass_me, radius_re, a_au, ecc, gravity_g;
    double    t_eq, t_surf;            /* K */
    double    water_frac, organic_frac, ocean_cover, atm_frac;
    float     sun_color[3], sun_intensity, sun_size;
    float     zenith[3], horizon[3];
    float     sun_ecl[3];              /* direction to the star, ecliptic frame */
    int       nbodies;
    SkyBody   bodies[PLANET_MAX_SKY_BODIES];
    char      atmosphere[160];
    char      crust[160];
    /* Gas mole fractions in ATM_* order, summing to 1. planet_load reads the
     * "atm=" key when present, otherwise it parses the atmosphere= text by gas
     * name; a missing CO2 gets a 0.005 baseline before renormalising.
     * atm_known = 1 when the file gave a composition (planet_save then writes
     * atm=), 0 when unknown (all zeros). */
    double    atm_mix[ATM_N];
    int       atm_known;
} HomePlanet;

void planet_default(HomePlanet* p);
int  planet_save(const char* path, const HomePlanet* p);
int  planet_load(const char* path, HomePlanet* p);   /* 0 ok */

/* FNV-1a 64 over a canonical text of the values that define the world and its
 * life: system seed, planet index, the 11 gen ints, round(t_surf*100),
 * round(star_lum*1e4), round(a_au*1e4) and round(atm_mix[i]*1e4). Independent
 * of file formatting and line endings, so the PC and the rigs agree. */
uint64_t planet_hash(const HomePlanet* p);

#endif

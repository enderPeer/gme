/* cosmos.h - star + protoplanetary disk -> planets (GPU N-body accretion). */
#ifndef GME_COSMOS_H
#define GME_COSMOS_H

#include <stdint.h>
#include "elements.h"

#ifdef __cplusplus
extern "C" {
#endif

#define M_EARTH_SUN 3.0035e-6    /* Earth mass in solar masses */
#define R_EARTH_AU  4.2635e-5    /* Earth radius in AU */
#define R_SUN_AU    4.6505e-3

typedef struct StarInfo { double mass, lum, teff, radius_au; } StarInfo;

typedef struct Body {
    double m;                 /* solar masses */
    double x[3], v[3];        /* AU, AU/yr */
    double comp[E_COUNT];     /* element mass fractions */
    double radius_au;         /* physical radius */
    int    merges;
} Body;

typedef struct CosmosConfig {
    uint32_t    seed;
    int         particles;    /* initial planetesimals */
    double      years;        /* simulated time */
    double      r_in, r_out;  /* disk extent, AU */
    double      hill_frac;    /* bodies merge inside this fraction of their Hill radius */
    double      disk_scale;   /* solid surface density multiplier */
    const char* snapshot_dir; /* PNG frames, or NULL */
    int         snapshots;    /* number of frames over the run */
} CosmosConfig;

typedef struct CosmosResult {
    StarInfo star;
    Body*    bodies;          /* survivors, sorted by orbit; caller frees */
    int      n;
    double   initial_solid_me;
    double   frost_line_au;
    long long steps, collisions;
    int      ejected;
    double   wall_seconds;
} CosmosResult;

void   cosmos_default_config(CosmosConfig* c);
int    cosmos_run(const CosmosConfig* c, CosmosResult* out);   /* 0 ok */

/* helpers shared with planet.c */
double disk_temperature(const StarInfo* s, double r_au);
void   orbit_elements(const StarInfo* s, const Body* b, double* a, double* e, double* inc);
double body_volatile_fraction(const Body* b);   /* H + He mass fraction */

#ifdef __cplusplus
}
#endif
#endif

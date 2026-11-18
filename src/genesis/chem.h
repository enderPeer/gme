/* chem.h - bonding chemistry: atoms with valences form, break and swap bonds
 * (Metropolis Monte Carlo on bond enthalpies) while the planet cools. The
 * molecules that survive are classified into the materials the world is
 * built from. Reactors are independent, so they shard across machines. */
#ifndef GME_CHEM_H
#define GME_CHEM_H

#include <stdint.h>
#include "elements.h"

enum {
    CC_H2 = 0, CC_HE, CC_H2O, CC_CO2, CC_CO, CC_CH4, CC_NH3, CC_N2, CC_O2, CC_H2S,
    CC_SILICATE, CC_MG_OXIDE, CC_IRON_OXIDE, CC_SULFIDE, CC_METAL, CC_CARBONATE,
    CC_ORGANIC, CC_GRAPHITE, CC_OTHER, CC_COUNT
};
extern const char* const CC_NAME[CC_COUNT];

#define CHEM_MAX_FORMULAS 4096

typedef struct ChemConfig {
    double   comp[E_COUNT];   /* element mass fractions of the inventory */
    double   t_hot, t_cold;   /* K: start of cooling, final surface temperature */
    int      atoms;           /* per reactor */
    int      reactors;        /* total over all shards */
    int      sweeps;          /* MC attempts per atom */
    uint32_t seed;
    int      shard, shards;
} ChemConfig;

typedef struct ChemResult {
    int      reactors_run;
    double   atoms;
    double   class_mass[CC_COUNT];      /* amu */
    double   class_molecules[CC_COUNT];
    double   sil_si, sil_mg, sil_fe;    /* atoms inside silicate networks */
    double   bonds[E_COUNT][E_COUNT];   /* bond units by element pair (a <= b) */
    double   c_partner_kinds;           /* sum over C atoms of distinct partner elements */
    double   c_atoms, si_partner_kinds, si_atoms;
    int      largest_organic;           /* atoms */
    int      n_formulas;                /* distinct carbon compounds seen */
    uint64_t formulas[CHEM_MAX_FORMULAS];
} ChemResult;

void chem_run(const ChemConfig* c, ChemResult* out);    /* runs this shard's reactors on all cores */
void chem_merge(ChemResult* into, const ChemResult* add);
int  chem_save(const char* path, const ChemResult* r);
int  chem_load(const char* path, ChemResult* r);
double chem_total_mass(const ChemResult* r);

#endif

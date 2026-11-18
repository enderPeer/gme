/* elements.h - the element set genesis works with: abundances, condensation
 * and bonding data. Values are rounded literature numbers (solar abundances,
 * 50% condensation temperatures, mean bond enthalpies in kJ/mol). */
#ifndef GME_ELEMENTS_H
#define GME_ELEMENTS_H

enum { E_H = 0, E_HE, E_C, E_N, E_O, E_MG, E_SI, E_S, E_FE, E_COUNT };

static const char* const ELEMENT_SYMBOL[E_COUNT] = { "H", "He", "C", "N", "O", "Mg", "Si", "S", "Fe" };
static const double ELEMENT_MASS[E_COUNT]   = { 1.008, 4.003, 12.011, 14.007, 15.999, 24.305, 28.086, 32.06, 55.845 };
/* bonding capacity: H 1, He 0 (noble), C 4 (the "life atom"), N 3, O 2, Mg 2, Si 4, S 2, Fe 3 */
static const int    ELEMENT_VALENCE[E_COUNT] = { 1, 0, 4, 3, 2, 2, 4, 2, 3 };
/* solar mass fractions (Asplund 2009, normalized over this set) */
static const double SOLAR_MASS_FRAC[E_COUNT] = { 0.7381, 0.2485, 0.00237, 0.000693, 0.00573, 0.000708, 0.000665, 0.000310, 0.00129 };

/* Single-bond enthalpies (kJ/mol), symmetric. 0 = no bond possible. */
static const double BOND_E1[E_COUNT][E_COUNT] = {
    /*        H    He   C    N    O    Mg   Si   S    Fe */
    /* H  */ {436,  0, 413, 391, 463, 126, 318, 363, 150},
    /* He */ {  0,  0,   0,   0,   0,   0,   0,   0,   0},
    /* C  */ {413,  0, 348, 293, 358, 140, 318, 272, 120},
    /* N  */ {391,  0, 293, 163, 201, 100, 355, 155, 110},
    /* O  */ {463,  0, 358, 201, 146, 394, 452, 265, 409},
    /* Mg */ {126,  0, 140, 100, 394,  50, 100, 230,  60},
    /* Si */ {318,  0, 318, 355, 452, 100, 222, 293, 110},
    /* S  */ {363,  0, 272, 155, 265, 230, 293, 266, 300},
    /* Fe */ {150,  0, 120, 110, 409,  60, 110, 300, 100},
};

/* Extra enthalpy for the 2nd and 3rd bond between the same pair
 * (double/triple bonds); 0 = not allowed. */
static double bond_extra(int a, int b, int order /* 2 or 3 */) {
    if (a > b) { int t = a; a = b; b = t; }
    if (a == E_C && b == E_C) return order == 2 ? 266 : 225;  /* C=C 614, C#C 839 */
    if (a == E_C && b == E_N) return order == 2 ? 322 : 276;  /* C=N 615, C#N 891 */
    if (a == E_C && b == E_O) return order == 2 ? 441 : 0;    /* C=O 799 */
    if (a == E_N && b == E_N) return order == 2 ? 255 : 527;  /* N=N 418, N#N 945 */
    if (a == E_N && b == E_O) return order == 2 ? 406 : 0;    /* N=O 607 */
    if (a == E_O && b == E_O) return order == 2 ? 352 : 0;    /* O=O 498 */
    if (a == E_O && b == E_SI) return order == 2 ? 190 : 0;   /* Si=O */
    if (a == E_O && b == E_S) return order == 2 ? 257 : 0;    /* S=O 522 */
    if (a == E_C && b == E_S) return order == 2 ? 301 : 0;    /* C=S 573 */
    return 0;
}

/* Energy of the k-th bond unit (k = 1, 2, 3) between elements a and b. */
static double bond_unit_energy(int a, int b, int k) {
    return k == 1 ? BOND_E1[a][b] : bond_extra(a, b, k);
}

#endif

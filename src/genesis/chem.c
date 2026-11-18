/* chem.c - see chem.h. */
#include "chem.h"
#include "../jobs.h"
#include "../platform.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char* const CC_NAME[CC_COUNT] = {
    "hydrogen H2", "helium", "water H2O", "carbon dioxide", "carbon monoxide", "methane", "ammonia",
    "nitrogen N2", "oxygen O2", "hydrogen sulfide", "silicate rock", "magnesium oxide", "iron oxide (ore)",
    "sulfides", "metal (Fe/Mg)", "carbonates", "organic molecules", "graphite", "other"
};

#define SMALL_MOLECULE 64  /* bigger components are network solids: classify per atom */

typedef struct Atom { uint8_t el, val, used, pad; int32_t nb[4]; } Atom;

typedef struct Rng { uint64_t s; } Rng;
static inline uint64_t rng_next(Rng* r) { r->s ^= r->s << 13; r->s ^= r->s >> 7; r->s ^= r->s << 17; return r->s; }
static inline double rng_u(Rng* r) { return (double)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0); }
static inline int rng_i(Rng* r, int n) { return (int)((rng_next(r) >> 33) % (uint64_t)n); }

static int order_of(const Atom* a, int b) {
    int k = 0;
    for (int i = 0; i < a->used; ++i) k += a->nb[i] == b;
    return k;
}
static void add_unit(Atom* A, int a, int b) { A[a].nb[A[a].used++] = b; A[b].nb[A[b].used++] = a; }
static void del_one(Atom* x, int b) {
    for (int i = 0; i < x->used; ++i)
        if (x->nb[i] == b) { x->nb[i] = x->nb[--x->used]; return; }
}
static void del_unit(Atom* A, int a, int b) { del_one(&A[a], b); del_one(&A[b], a); }

/* energy of the bond unit that would be added between a and b (0 = impossible) */
static double add_energy(const Atom* A, int a, int b) {
    if (A[a].used >= A[a].val || A[b].used >= A[b].val) return 0;
    const int k = order_of(&A[a], b) + 1;
    return k > 3 ? 0 : bond_unit_energy(A[a].el, A[b].el, k);
}

static int pick_unit(Rng* r, const Atom* A, int n, int* a, int* b) {
    for (int t = 0; t < 16; ++t) {
        const int i = rng_i(r, n);
        if (!A[i].used) continue;
        *a = i;
        *b = A[i].nb[rng_i(r, A[i].used)];
        return 1;
    }
    return 0;
}

static int uf_find(int* p, int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }

static uint64_t formula_hash(const int* cnt) {
    uint64_t h = 1469598103934665603ull;
    for (int e = 0; e < E_COUNT; ++e) { h ^= (uint64_t)cnt[e] + 0x9E37u * (uint64_t)e; h *= 1099511628211ull; }
    return h | 1;
}

static void add_formula(ChemResult* r, uint64_t h) {
    for (int i = 0; i < r->n_formulas; ++i) if (r->formulas[i] == h) return;
    if (r->n_formulas < CHEM_MAX_FORMULAS) r->formulas[r->n_formulas++] = h;
}

static int classify(const int* c, int k) {
    const int H = c[E_H], He = c[E_HE], C = c[E_C], N = c[E_N], O = c[E_O];
    const int Mg = c[E_MG], Si = c[E_SI], S = c[E_S], Fe = c[E_FE];
    if (He == k) return CC_HE;
    if (k == 1) {
        if (H) return CC_H2;
        if (O) return CC_O2;
        if (N) return CC_N2;
        if (Fe || Mg) return CC_METAL;
        return CC_OTHER;
    }
    if (Si && O) return CC_SILICATE;
    if (C && O && (Mg || Fe)) return CC_CARBONATE;
    if (C >= 2 && H) return CC_ORGANIC;
    if (C == 1) {
        if (k == 3 && O == 2) return CC_CO2;
        if (k == 2 && O == 1) return CC_CO;
        if (k == 5 && H == 4) return CC_CH4;
        if (H) return CC_ORGANIC;
        return CC_OTHER;
    }
    if (k == 3 && H == 2 && O == 1) return CC_H2O;
    if (k == 4 && N == 1 && H == 3) return CC_NH3;
    if (k == 2 && N == 2) return CC_N2;
    if (k == 2 && O == 2) return CC_O2;
    if (k == 2 && H == 2) return CC_H2;
    if (k == 3 && H == 2 && S == 1) return CC_H2S;
    if ((Fe || Mg) && S && !O) return CC_SULFIDE;
    if ((Fe || Mg) && O && !S && !C && !N && !H) return Fe >= Mg ? CC_IRON_OXIDE : CC_MG_OXIDE;
    if ((Fe || Mg) && !O && !S && !C && !N && !H && !Si) return CC_METAL;
    return CC_OTHER;
}

/* Network solids (rock, metal, graphite) are one huge bonded component, so
 * each atom is classified by its own bonding environment instead. */
static int atom_class(const Atom* A, int i) {
    int has[E_COUNT] = { 0 };
    int o_si = 0, o_metal = 0;
    for (int s = 0; s < A[i].used; ++s) {
        const int j = A[i].nb[s];
        has[A[j].el] = 1;
        if (A[j].el == E_O)
            for (int t = 0; t < A[j].used; ++t) {
                const int e = A[A[j].nb[t]].el;
                o_si |= e == E_SI;
                o_metal |= e == E_MG || e == E_FE;
            }
    }
    switch (A[i].el) {
    case E_SI: return CC_SILICATE;
    case E_HE: return CC_HE;
    case E_O:
        if (has[E_SI]) return CC_SILICATE;
        if (has[E_C]) return (has[E_MG] || has[E_FE]) ? CC_CARBONATE : CC_ORGANIC;
        if (has[E_FE]) return CC_IRON_OXIDE;
        if (has[E_MG]) return CC_MG_OXIDE;
        if (has[E_H]) return CC_H2O;      /* hydroxyl / bound water */
        return CC_OTHER;
    case E_MG: case E_FE:
        if (o_si) return CC_SILICATE;
        if (has[E_O]) return A[i].el == E_FE ? CC_IRON_OXIDE : CC_MG_OXIDE;
        if (has[E_S]) return CC_SULFIDE;
        if (has[E_C] || has[E_N] || has[E_H] || has[E_SI]) return CC_OTHER;
        return CC_METAL;
    case E_C:
        if (has[E_H] || has[E_N]) return CC_ORGANIC;
        if (has[E_O]) return o_metal ? CC_CARBONATE : CC_ORGANIC;
        if (has[E_C] && !has[E_SI] && !has[E_FE] && !has[E_MG] && !has[E_S]) return CC_GRAPHITE;
        return CC_OTHER;
    case E_H:
        if (has[E_O]) return CC_H2O;
        if (has[E_C]) return CC_ORGANIC;
        if (has[E_N]) return CC_NH3;
        if (has[E_S]) return CC_H2S;
        return CC_OTHER;
    case E_N:
        if (has[E_C]) return CC_ORGANIC;
        if (has[E_H]) return CC_NH3;
        if (has[E_N]) return CC_N2;
        return CC_OTHER;
    case E_S:
        if (has[E_FE] || has[E_MG]) return CC_SULFIDE;
        if (has[E_H]) return CC_H2S;
        if (has[E_C]) return CC_ORGANIC;
        return CC_OTHER;
    }
    return CC_OTHER;
}

typedef struct RunCtx { const ChemConfig* c; ChemResult* per; int first; } RunCtx;

static void run_reactor(const ChemConfig* cfg, int index, ChemResult* res) {
    memset(res, 0, sizeof *res);
    Rng rng = { 0x2545F4914F6CDD1Dull ^ ((uint64_t)cfg->seed << 20) ^ ((uint64_t)(index + 1) * 0x9E3779B97F4A7C15ull) };
    const int n = cfg->atoms;
    Atom* A = (Atom*)calloc((size_t)n, sizeof(Atom));
    int* parent = (int*)malloc(sizeof(int) * (size_t)n);
    if (!A || !parent) { free(A); free(parent); return; }

    /* atoms in proportion to moles */
    double mol[E_COUNT], tot = 0;
    for (int e = 0; e < E_COUNT; ++e) { mol[e] = cfg->comp[e] / ELEMENT_MASS[e]; tot += mol[e]; }
    int k = 0;
    for (int e = 0; e < E_COUNT && k < n; ++e) {
        const double want = mol[e] / tot * n + rng_u(&rng); /* stochastic rounding */
        for (int i = 0; i < (int)want && k < n; ++i, ++k) { A[k].el = (uint8_t)e; A[k].val = (uint8_t)ELEMENT_VALENCE[e]; }
    }
    for (; k < n; ++k) { A[k].el = E_O; A[k].val = 2; }

    /* anneal: hot -> surface temperature */
    const long long attempts = (long long)n * cfg->sweeps;
    double kT = 0.0083145 * cfg->t_hot; /* kJ/mol */
    for (long long t = 0; t < attempts; ++t) {
        if ((t & 1023) == 0) kT = 0.0083145 * cfg->t_hot * pow(cfg->t_cold / cfg->t_hot, (double)t / attempts);
        const double u = rng_u(&rng);
        if (u < 0.40) {                       /* form a bond between two free valences */
            const int a = rng_i(&rng, n), b = rng_i(&rng, n);
            if (a == b) continue;
            if (add_energy(A, a, b) > 0) add_unit(A, a, b);
        } else if (u < 0.88) {                /* swap partners: a-b + c-d -> a-c + b-d */
            int a, b, c, d;
            if (!pick_unit(&rng, A, n, &a, &b) || !pick_unit(&rng, A, n, &c, &d)) continue;
            if (a == c || a == d || b == c || b == d) continue;
            const double old_e = bond_unit_energy(A[a].el, A[b].el, order_of(&A[a], b)) +
                                 bond_unit_energy(A[c].el, A[d].el, order_of(&A[c], d));
            del_unit(A, a, b);
            del_unit(A, c, d);
            const int oac = order_of(&A[a], c) + 1, obd = order_of(&A[b], d) + 1;
            const double eac = oac <= 3 ? bond_unit_energy(A[a].el, A[c].el, oac) : 0;
            const double ebd = obd <= 3 ? bond_unit_energy(A[b].el, A[d].el, obd) : 0;
            const double gain = eac + ebd - old_e;
            if (eac > 0 && ebd > 0 && (gain >= 0 || rng_u(&rng) < exp(gain / kT))) {
                add_unit(A, a, c);
                add_unit(A, b, d);
            } else {
                add_unit(A, a, b);
                add_unit(A, c, d);
            }
        } else {                              /* thermal bond breaking */
            int a, b;
            if (!pick_unit(&rng, A, n, &a, &b)) continue;
            const double e = bond_unit_energy(A[a].el, A[b].el, order_of(&A[a], b));
            if (rng_u(&rng) < exp(-e / kT)) del_unit(A, a, b);
        }
    }

    /* molecules = connected components */
    for (int i = 0; i < n; ++i) parent[i] = i;
    for (int i = 0; i < n; ++i)
        for (int s = 0; s < A[i].used; ++s) {
            const int x = uf_find(parent, i), y = uf_find(parent, A[i].nb[s]);
            if (x != y) parent[x] = y;
            if (i <= A[i].nb[s]) {
                int ea = A[i].el, eb = A[A[i].nb[s]].el;
                if (ea > eb) { int t2 = ea; ea = eb; eb = t2; }
                res->bonds[ea][eb] += 1;
            }
        }
    /* bond diversity: how many different partner elements C and Si use */
    for (int i = 0; i < n; ++i) {
        if (A[i].el != E_C && A[i].el != E_SI) continue;
        int seen = 0, kinds = 0;
        for (int s = 0; s < A[i].used; ++s) {
            const int bit = 1 << A[A[i].nb[s]].el;
            if (!(seen & bit)) { seen |= bit; ++kinds; }
        }
        if (A[i].el == E_C) { res->c_atoms += 1; res->c_partner_kinds += kinds; }
        else { res->si_atoms += 1; res->si_partner_kinds += kinds; }
    }
    int* cnt = (int*)calloc((size_t)n * E_COUNT, sizeof(int));
    int* size = (int*)calloc((size_t)n, sizeof(int));
    if (cnt && size) {
        for (int i = 0; i < n; ++i) {
            const int r = uf_find(parent, i);
            cnt[(size_t)r * E_COUNT + A[i].el]++;
            size[r]++;
        }
        /* small molecules: classified whole (gases, water, small organics) */
        for (int r = 0; r < n; ++r) {
            if (!size[r] || size[r] > SMALL_MOLECULE) continue;
            const int* c = &cnt[(size_t)r * E_COUNT];
            const int cls = classify(c, size[r]);
            double m = 0;
            for (int e = 0; e < E_COUNT; ++e) m += c[e] * ELEMENT_MASS[e];
            res->class_mass[cls] += m;
            res->class_molecules[cls] += 1;
            if (cls == CC_SILICATE) { res->sil_si += c[E_SI]; res->sil_mg += c[E_MG]; res->sil_fe += c[E_FE]; }
            if (cls == CC_ORGANIC && size[r] > res->largest_organic) res->largest_organic = size[r];
            if (c[E_C]) add_formula(res, formula_hash(c));
        }
        /* network solids: per atom, by bonding environment */
        for (int i = 0; i < n; ++i) {
            if (size[uf_find(parent, i)] <= SMALL_MOLECULE) continue;
            const int cls = atom_class(A, i);
            res->class_mass[cls] += ELEMENT_MASS[A[i].el];
            if (cls == CC_SILICATE) {
                if (A[i].el == E_SI) res->sil_si += 1;
                else if (A[i].el == E_MG) res->sil_mg += 1;
                else if (A[i].el == E_FE) res->sil_fe += 1;
            }
        }
    }
    res->atoms = n;
    res->reactors_run = 1;
    free(cnt); free(size); free(A); free(parent);
}

static void reactor_fn(void* p, uint32_t b, uint32_t e, uint32_t wk) {
    (void)wk;
    RunCtx* x = (RunCtx*)p;
    for (uint32_t i = b; i < e; ++i) run_reactor(x->c, x->first + (int)i, &x->per[i]);
}

void chem_merge(ChemResult* d, const ChemResult* s) {
    d->reactors_run += s->reactors_run;
    d->atoms += s->atoms;
    for (int c = 0; c < CC_COUNT; ++c) { d->class_mass[c] += s->class_mass[c]; d->class_molecules[c] += s->class_molecules[c]; }
    d->sil_si += s->sil_si; d->sil_mg += s->sil_mg; d->sil_fe += s->sil_fe;
    for (int a = 0; a < E_COUNT; ++a) for (int b = 0; b < E_COUNT; ++b) d->bonds[a][b] += s->bonds[a][b];
    d->c_partner_kinds += s->c_partner_kinds; d->c_atoms += s->c_atoms;
    d->si_partner_kinds += s->si_partner_kinds; d->si_atoms += s->si_atoms;
    if (s->largest_organic > d->largest_organic) d->largest_organic = s->largest_organic;
    for (int i = 0; i < s->n_formulas; ++i) add_formula(d, s->formulas[i]);
}

void chem_run(const ChemConfig* c, ChemResult* out) {
    memset(out, 0, sizeof *out);
    const int shards = c->shards > 0 ? c->shards : 1;
    const int first = (int)((long long)c->reactors * c->shard / shards);
    const int last = (int)((long long)c->reactors * (c->shard + 1) / shards);
    const int count = last - first;
    if (count <= 0) return;
    ChemResult* per = (ChemResult*)calloc((size_t)count, sizeof(ChemResult));
    if (!per) return;
    RunCtx x = { c, per, first };
    jobs_parallel_for((uint32_t)count, 1, reactor_fn, &x);
    for (int i = 0; i < count; ++i) chem_merge(out, &per[i]);
    free(per);
}

double chem_total_mass(const ChemResult* r) {
    double m = 0;
    for (int c = 0; c < CC_COUNT; ++c) m += r->class_mass[c];
    return m;
}

int chem_save(const char* path, const ChemResult* r) {
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "reactors %d\natoms %.17g\n", r->reactors_run, r->atoms);
    for (int c = 0; c < CC_COUNT; ++c) fprintf(f, "class %d %.17g %.17g\n", c, r->class_mass[c], r->class_molecules[c]);
    fprintf(f, "silicate %.17g %.17g %.17g\n", r->sil_si, r->sil_mg, r->sil_fe);
    for (int a = 0; a < E_COUNT; ++a) for (int b = 0; b < E_COUNT; ++b)
        if (r->bonds[a][b] > 0) fprintf(f, "bond %d %d %.17g\n", a, b, r->bonds[a][b]);
    fprintf(f, "cdiv %.17g %.17g %.17g %.17g\n", r->c_partner_kinds, r->c_atoms, r->si_partner_kinds, r->si_atoms);
    fprintf(f, "largest_organic %d\n", r->largest_organic);
    for (int i = 0; i < r->n_formulas; ++i) fprintf(f, "formula %llu\n", (unsigned long long)r->formulas[i]);
    fclose(f);
    return 0;
}

int chem_load(const char* path, ChemResult* r) {
    memset(r, 0, sizeof *r);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        int a, b;
        double x, y, z, w;
        unsigned long long u;
        if (sscanf(line, "reactors %d", &a) == 1) r->reactors_run = a;
        else if (sscanf(line, "atoms %lf", &x) == 1) r->atoms = x;
        else if (sscanf(line, "class %d %lf %lf", &a, &x, &y) == 3 && a >= 0 && a < CC_COUNT) { r->class_mass[a] = x; r->class_molecules[a] = y; }
        else if (sscanf(line, "silicate %lf %lf %lf", &x, &y, &z) == 3) { r->sil_si = x; r->sil_mg = y; r->sil_fe = z; }
        else if (sscanf(line, "bond %d %d %lf", &a, &b, &x) == 3 && a >= 0 && b >= 0 && a < E_COUNT && b < E_COUNT) r->bonds[a][b] = x;
        else if (sscanf(line, "cdiv %lf %lf %lf %lf", &x, &y, &z, &w) == 4) { r->c_partner_kinds = x; r->c_atoms = y; r->si_partner_kinds = z; r->si_atoms = w; }
        else if (sscanf(line, "largest_organic %d", &a) == 1) r->largest_organic = a;
        else if (sscanf(line, "formula %llu", &u) == 1) add_formula(r, (uint64_t)u);
    }
    fclose(f);
    return 0;
}

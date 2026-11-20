/* fauna_scene.c - places the creatures of a GMEFAUN1 snapshot on the drawn
 * world and bins them into the renderer's hash grid (see fauna_scene.h). */
#include "fauna_scene.h"
#include "density.h"
#include <stdlib.h>
#include <string.h>

#define COORD_MAX 33000000.0          /* |x|, |z| inside the world's chunk coordinate range (2^20 chunks) */

static uint32_t mix32(uint32_t h) {
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

/* The drawn ground at (x, z): the bilinear patch of the smooth surface T over
 * the 4 column centres around the point (the column tracer's dual cell), and
 * k = sqrt(1 + |grad T|^2), the lift per unit radius of a sphere touching it. */
static double ground_at(double x, double z, uint32_t seed, const GmePlanet* pl, double* k) {
    const double fx = floor(x - 0.5), fz = floor(z - 0.5);
    const int32_t DX = (int32_t)fx, DZ = (int32_t)fz;
    const double u = x - (fx + 0.5), v = z - (fz + 0.5);
    /* the 4 columns as one 2 x 2 region: the T of gme_surface_q8 at each (a column's T depends only on the
     * columns within GME_SMOOTH_HALO, and every region computes that halo), for a third of the noise lookups */
    int32_t h4[4], d4[4], scratch[4 * (2 + 2 * GME_SMOOTH_HALO) * (2 + 2 * GME_SMOOTH_HALO)];
    gme_surface_region(DX, DZ, 2, seed, pl, h4, d4, scratch);
    const double T00 = (256 * (h4[0] + 1) + 2 * d4[0]) / 256.0, T10 = (256 * (h4[1] + 1) + 2 * d4[1]) / 256.0;
    const double T01 = (256 * (h4[2] + 1) + 2 * d4[2]) / 256.0, T11 = (256 * (h4[3] + 1) + 2 * d4[3]) / 256.0;
    const double gx = (T10 - T00) * (1 - v) + (T11 - T01) * v, gz = (T01 - T00) * (1 - u) + (T11 - T10) * u;
    if (k) *k = sqrt(1.0 + gx * gx + gz * gz);
    return (T00 + (T10 - T00) * u) * (1 - v) + (T01 + (T11 - T01) * u) * v;
}

/* One sphere of radius r at (x, z): its rest height, motion amplitude and rest
 * height in the classic (voxel) view by the habitat rules. 0 placed, 1 a
 * swimmer over land, 2 a walker over water. */
static int place(double x, double z, double r, int swims, double depth, double amp_want, uint32_t seed, const GmePlanet* pl,
                 double* y_out, double* amp_out, double* yv_out) {
    double k = 1.0;
    const double T = ground_at(x, z, seed, pl, &k);
    const double yw = (double)pl->sea_level + 1.0;
    const int ice = pl->ocean_top_mat == MAT_ICE || pl->ocean_mat == MAT_ICE;
    /* the voxel ground under the point: the top face of its column's ground voxel */
    const double Tv = (double)gme_height((int32_t)floor(x), (int32_t)floor(z), seed, pl) + 1.0;
    double y, yv, amp = amp_want;
    if (swims) {
        if (!(T < yw)) return 1;
        const double lo = T + r, hi = yw - r - 0.05;       /* resting on the bed .. just under the surface */
        if (hi <= lo) {                                    /* shallower than the body: on the bed, still */
            y = lo;
            amp = 0;
        } else {
            if (amp > hi - lo) amp = hi - lo;              /* the drift is +-amp / 2 vertically */
            y = yw - depth;
            if (y > hi - 0.5 * amp) y = hi - 0.5 * amp;
            if (y < lo + 0.5 * amp) y = lo + 0.5 * amp;
        }
        yv = y > Tv + r + 0.5 * amp ? y : Tv + r + 0.5 * amp;
    } else if (T < yw) {
        if (!ice) return 2;
        y = yw + r;                                        /* a frozen sea: on the ice sheet */
        yv = y;
    } else {
        y = T + r * k;                                     /* the sphere touches the slope */
        yv = (Tv > yw ? Tv : yw) + r;
    }
    *y_out = y;
    *amp_out = amp;
    *yv_out = yv;
    return 0;
}

typedef struct Ref { int32_t cx, cz; uint32_t sph; } Ref;

static int ref_cmp(const void* a, const void* b) {
    const Ref* x = (const Ref*)a;
    const Ref* y = (const Ref*)b;
    if (x->cx != y->cx) return x->cx < y->cx ? -1 : 1;
    if (x->cz != y->cz) return x->cz < y->cz ? -1 : 1;
    return (x->sph > y->sph) - (x->sph < y->sph);
}

void fauna_scene_free(FaunaScene* s) {
    if (!s) return;
    free(s->sph);
    free(s->grid);
    free(s->cells);
    memset(s, 0, sizeof *s);
}

int fauna_scene_build(const Fauna* f, uint32_t seed, const GmePlanet* pl, FaunaScene* out) {
    memset(out, 0, sizeof *out);
    out->records = f->count;
    if (!f->count) return 0;
    FaunaSphere* sph = (FaunaSphere*)malloc(sizeof(FaunaSphere) * (size_t)f->count * FAUNA_SWARM_N);
    if (!sph) return -1;
    uint32_t n = 0;
    for (uint32_t i = 0; i < f->count; ++i) {
        const FaunaRec* rec = &f->rec[i];
        const double x = rec->x_q4 / 16.0, z = rec->z_q4 / 16.0;
        if (fabs(x) > COORD_MAX || fabs(z) > COORD_MAX) { out->drop_range++; continue; }
        const int swims = (rec->flags & FAUNA_SWIMS) != 0;
        double r = rec->size_q4 / 32.0;
        r = r < FAUNA_R_MIN ? FAUNA_R_MIN : (r > FAUNA_R_MAX ? FAUNA_R_MAX : r);
        const double depth = rec->depth_q4 / 16.0, mot = rec->motor / 255.0;
        const uint32_t h0 = mix32((uint32_t)rec->x_q4 * 0x9E3779B1u ^ mix32((uint32_t)rec->z_q4 + i * 0x85EBCA6Bu) ^ seed);
        const uint32_t base = ((uint32_t)rec->pal & 15u) | (swims ? FA_SWIMS : 0u) | ((rec->flags & FAUNA_HUNTER) ? FA_HUNTER : 0u) |
                              ((rec->flags & FAUNA_ARMORED) ? FA_ARMORED : 0u) | ((uint32_t)rec->motor << 8) | ((h0 >> 26) << 26);
        /* idle motion: a walker hops up to min(0.5 r, 0.3) m, a swimmer drifts up to min(1.5 r, 0.6) m, by the motor gene */
        const double amp0 = mot * (swims ? fmin(1.5 * r, 0.6) : fmin(0.5 * r, 0.3));
        double y, amp, yv;
        const int why = place(x, z, r, swims, depth, amp0, seed, pl, &y, &amp, &yv);
        if (why == 1) { out->drop_swim_land++; continue; }
        if (why == 2) { out->drop_walk_water++; continue; }
        const uint32_t n0 = n;
        if (rec->flags & FAUNA_SWARM) {
            /* many small individuals: a jittered ring of spheres around the record's position, each on its own ground */
            double rs = 0.42 * r;
            rs = rs < FAUNA_R_MIN ? FAUNA_R_MIN : rs;
            const double rc = fmax(1.6 * r, 2.6 * rs);
            const double amps = mot * (swims ? fmin(1.5 * rs, 0.6) : fmin(0.5 * rs, 0.3));
            for (int k = 0; k < FAUNA_SWARM_N; ++k) {
                const uint32_t hk = mix32(h0 + 0x9E3779B9u * (uint32_t)(k + 1));
                const double ang = (k + (hk & 255u) / 255.0 * 0.6) * (6.283185307179586 / FAUNA_SWARM_N);
                const double dist = rc * (0.75 + 0.5 * ((hk >> 8) & 255u) / 255.0);
                const double xs = x + dist * cos(ang), zs = z + dist * sin(ang);
                const double ds = depth + (((hk >> 16) & 255u) / 255.0 - 0.5) * rc;
                double ys, as, yvs;
                if (place(xs, zs, rs, swims, ds < 0 ? 0 : ds, amps, seed, pl, &ys, &as, &yvs)) continue;   /* no room for this one */
                const double cxf = floor(xs / 32.0), czf = floor(zs / 32.0);
                FaunaSphere* s = &sph[n++];
                s->cx = (int32_t)cxf; s->cz = (int32_t)czf;
                s->lx = (float)(xs - 32.0 * cxf); s->lz = (float)(zs - 32.0 * czf);
                s->y = (float)ys; s->r = (float)rs; s->amp = (float)as; s->yv = (float)yvs;
                s->attr = (base & 0x03FFFFFFu) | FA_SWARM | (((hk >> 20) & 511u) << 16) | ((((h0 >> 26) + (hk & 7u)) & 63u) << 26);
            }
        }
        if (n == n0) {                                     /* one body (or a swarm none of whose members found room) */
            const double cxf = floor(x / 32.0), czf = floor(z / 32.0);
            FaunaSphere* s = &sph[n++];
            s->cx = (int32_t)cxf; s->cz = (int32_t)czf;
            s->lx = (float)(x - 32.0 * cxf); s->lz = (float)(z - 32.0 * czf);
            s->y = (float)y; s->r = (float)r; s->amp = (float)amp; s->yv = (float)yv;
            s->attr = base | (((h0 >> 8) & 511u) << 16);
        } else {
            out->swarms++;
        }
        sph[n0].attr |= FA_FIRST;
        out->kept++;
        if (swims) out->swimmers++; else out->walkers++;
    }
    if (!n) { free(sph); return 0; }

    /* grid: every cell a sphere can touch (its radius + motion + 5 cm) */
    Ref* refs = (Ref*)malloc(sizeof(Ref) * (size_t)n * 4);
    if (!refs) { free(sph); return -1; }
    size_t nr = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const FaunaSphere* s = &sph[i];
        const double pad = (double)s->r + (double)s->amp + 0.05;
        const int32_t x0 = s->cx + (int32_t)floor(((double)s->lx - pad) / 32.0), x1 = s->cx + (int32_t)floor(((double)s->lx + pad) / 32.0);
        const int32_t z0 = s->cz + (int32_t)floor(((double)s->lz - pad) / 32.0), z1 = s->cz + (int32_t)floor(((double)s->lz + pad) / 32.0);
        for (int32_t cz = z0; cz <= z1; ++cz)
            for (int32_t cx = x0; cx <= x1; ++cx) {        /* pad < 32 m: at most 2 x 2 cells */
                refs[nr].cx = cx; refs[nr].cz = cz; refs[nr].sph = i;
                ++nr;
            }
    }
    qsort(refs, nr, sizeof(Ref), ref_cmp);
    uint32_t used = 0;
    for (size_t i = 0; i < nr; ++i) used += i == 0 || refs[i].cx != refs[i - 1].cx || refs[i].cz != refs[i - 1].cz;
    uint32_t cap = 16;
    while (cap < used * 4u) cap <<= 1;
    FaunaSphere* grid = (FaunaSphere*)malloc(sizeof(FaunaSphere) * nr);   /* the spheres of a cell side by side */
    FaunaCell* cells = NULL;
    if (!grid) { free(refs); free(sph); return -1; }
    for (;;) {
        cells = (FaunaCell*)calloc(cap, sizeof(FaunaCell));
        if (!cells) { free(grid); free(refs); free(sph); return -1; }
        uint32_t longest = 0;
        for (size_t i = 0; i < nr;) {
            size_t j = i;
            float ylo = 3.0e38f, yhi = -3.0e38f;
            for (; j < nr && refs[j].cx == refs[i].cx && refs[j].cz == refs[i].cz; ++j) {
                const FaunaSphere* s = &sph[refs[j].sph];
                grid[j] = *s;
                /* both views: the smooth rest height and the classic one */
                const float lo = (s->y < s->yv ? s->y : s->yv) - s->r - 0.5f * s->amp - 0.05f;
                const float hi = (s->y > s->yv ? s->y : s->yv) + s->r + s->amp + 0.05f;
                ylo = lo < ylo ? lo : ylo;
                yhi = hi > yhi ? hi : yhi;
            }
            uint32_t h = fauna_cell_hash(refs[i].cx, refs[i].cz) & (cap - 1u), probes = 1;
            while (cells[h].count) { h = (h + 1u) & (cap - 1u); ++probes; }
            longest = probes > longest ? probes : longest;
            cells[h].cx = refs[i].cx; cells[h].cz = refs[i].cz;
            cells[h].first = (uint32_t)i; cells[h].count = (uint32_t)(j - i);
            cells[h].ylo = ylo; cells[h].yhi = yhi;
            i = j;
        }
        /* fauna_cell_find looks at 64 slots: a longer probe run (never seen at a quarter full) gets a larger table */
        if (longest <= 48u || cap >= (1u << 28)) break;
        free(cells);
        cap <<= 1;
    }
    free(refs);
    out->sph = sph; out->nsph = n;
    out->grid = grid; out->ngrid = (uint32_t)nr;
    out->cells = cells; out->ncells = cap;
    out->used_cells = used;
    return 0;
}

uint32_t fauna_scene_within(const FaunaScene* s, double x, double z, double radius) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < s->nsph; ++i) {
        const FaunaSphere* p = &s->sph[i];
        if (!(p->attr & FA_FIRST)) continue;
        const double dx = 32.0 * p->cx + p->lx - x, dz = 32.0 * p->cz + p->lz - z;
        n += dx * dx + dz * dz <= radius * radius;
    }
    return n;
}

/* biomap.h - CONTRACT between the life simulation (exporter) and the engine
 * (voxel generation + renderer). See docs/biosphere-spec.md section 7 and
 * docs/render-upgrade-spec.md 5.3 (version 2).
 *
 * A biomap is a frozen snapshot of vegetation: up to 8 rectangular regions
 * of packed cells (one uint32 per 2^shift m square) plus a 64-entry fallback
 * table indexed by gme_env_bin() that vegetates terrain outside the regions.
 * Version 2 (GMEBMAP2) adds one morphology word per cell (the evolved plant
 * shape: crown class, aspect, leaf angle, porosity, pigment amount, Q band
 * width), the same per fallback class, and the atmosphere life has made.
 * A version 1 file loads with morph == NULL, and every consumer then takes
 * the version 1 code path unchanged.
 * Usable from C and CUDA (types only; functions are host-side). */
#ifndef GME_BIOMAP_H
#define GME_BIOMAP_H

#include <stddef.h>
#include <stdint.h>

#define BIOMAP_MAGIC       "GMEBMAP1"
#define BIOMAP_MAGIC2      "GMEBMAP2"
#define BIOMAP_MAX_REGIONS 8
#define BIOMAP_FALLBACK    64
#define BIOMAP_PALETTE     16
#define BIOMAP_ATM         7    /* H2S CH4 NH3 CO2 O2 N2 H2 (planetfile.h ATM_*, bio_api.h BIO_ATM_*) */

/* vegetation kinds (bits 0-3 of a cell) */
enum {
    BIO_KIND_NONE  = 0,
    BIO_KIND_MAT   = 1,   /* microbial / algal mat on the ground (seabed, tidal, beach) */
    BIO_KIND_KELP  = 2,   /* standing algae in the water column (+ mat on the bed) */
    BIO_KIND_COVER = 3,   /* ground cover on land: grass voxel + low tufts */
    BIO_KIND_SHRUB = 4,   /* small woody plants (trunk + crown, <= 4 m) */
    BIO_KIND_TREE  = 5    /* trees (> 4 m) */
};

/* packed cell: bits 0-3 kind, 4-7 palette index, 8-15 cover 0..255,
 * 16-23 height in metres, 24-31 fauna density (reserved) */
#define BIOCELL(kind, pal, cover, height, fauna) \
    ((uint32_t)(kind) | ((uint32_t)(pal) << 4) | ((uint32_t)(cover) << 8) | ((uint32_t)(height) << 16) | ((uint32_t)(fauna) << 24))
#define BIOCELL_KIND(c)   ((uint32_t)(c) & 15u)
#define BIOCELL_PAL(c)    (((uint32_t)(c) >> 4) & 15u)
#define BIOCELL_COVER(c)  (((uint32_t)(c) >> 8) & 255u)
#define BIOCELL_HEIGHT(c) (((uint32_t)(c) >> 16) & 255u)
#define BIOCELL_FAUNA(c)  (((uint32_t)(c) >> 24) & 255u)

/* ---- version 2: morphology of a cell's vegetation ------------------------------
 * The biomass-weighted dominant (classes) or mean (amounts) of the cell's
 * plants of the exported kind, from the body genes BG_CROWN, BG_ASPECT,
 * BG_LEAFANG, BG_POROSITY, BG_PIGMENT, BG_QWIDTH (bio_types.h). The genes
 * were neutral before shapes existed and sit near 0 in every evolved world,
 * so gene 0 is the plant of version 1: an ellipsoid crown, height : width 1,
 * leaves at the mid angle, a closed crown (the *_OF macros below wrap the
 * gene's range around that point).
 *   bits 0-2   crown class (BIO_CROWN_*)
 *   bits 3-5   aspect bin 0..7: height : width 0.5, 0.75, 1, 1.4, 1.8, 2.2, 2.6, 3 (BIOMORPH_ASPECT_Q8)
 *   bits 6-7   leaf angle bin 0..3: erect .. flat (chi = -0.4 + (bin + 0.5) / 4)
 *   bits 8-9   porosity bin 0..3: closed .. open crown
 *   bits 10-17 mean pigment amount 0..255 (colour saturation of the tissue)
 *   bits 18-21 Q band width bin 0..15: 25 + 35 * bin / 15 nm (25 .. 60 nm; bin 0 is the version 1 pigment)
 *   bit  31    valid: the cell carries evolved morphology (0: use the v1 look) */
enum {
    BIO_CROWN_ELLIPSOID = 0,   /* the version 1 crown */
    BIO_CROWN_CONE      = 1,   /* widest at the base of the crown, a point at the top */
    BIO_CROWN_UMBRELLA  = 2,   /* a flat, wide cap on a long bare trunk */
    BIO_CROWN_COLUMN    = 3,   /* a narrow cylinder over the whole crown height */
    BIO_CROWN_DOME      = 4,   /* the upper half of an ellipsoid, flat underneath */
    BIO_CROWN_N         = 5
};
#define BIOMORPH_VALID 0x80000000u
#define BIOMORPH(crown, aspect, leafang, porosity, pig, qwidth)                                              \
    (BIOMORPH_VALID | (uint32_t)(crown) | ((uint32_t)(aspect) << 3) | ((uint32_t)(leafang) << 6) |           \
     ((uint32_t)(porosity) << 8) | ((uint32_t)(pig) << 10) | ((uint32_t)(qwidth) << 18))
#define BIOMORPH_CROWN(m)    ((uint32_t)(m) & 7u)
#define BIOMORPH_ASPECT(m)   (((uint32_t)(m) >> 3) & 7u)
#define BIOMORPH_LEAFANG(m)  (((uint32_t)(m) >> 6) & 3u)
#define BIOMORPH_POROSITY(m) (((uint32_t)(m) >> 8) & 3u)
#define BIOMORPH_PIG(m)      (((uint32_t)(m) >> 10) & 255u)
#define BIOMORPH_QWIDTH(m)   (((uint32_t)(m) >> 18) & 15u)
/* gene (0..255) -> bin: the one mapping shared by the life simulation and the engine.
 * Aspect: gene 0..31 is bin 2 (1 : 1), rising to bin 7 (3 : 1) at 160..191, then
 * 192..223 is bin 0 (0.5) and 224..255 bin 1 (0.75), next to gene 0 again.
 * Leaf angle: gene 0..15 is the mid angle (class 8 of 16), flatter up to gene 127,
 * erect from gene 128, steepening back to the mid angle at 255. */
#define BIOMORPH_CROWN_OF(g)    (((uint32_t)(g) * 5u) >> 8)                 /* 0..4 */
#define BIOMORPH_ASPECT_OF(g)   ((((uint32_t)(g) >> 5) + 2u) & 7u)          /* 0..7 */
#define BIOMORPH_LEAFCLASS_OF(g) ((((uint32_t)(g) >> 4) + 8u) & 15u)        /* 0..15: the simulation's 16 classes, erect .. flat */
#define BIOMORPH_LEAFANG_OF(g)  (BIOMORPH_LEAFCLASS_OF(g) >> 2)             /* 0..3 */
#define BIOMORPH_POROSITY_OF(g) ((uint32_t)(g) >> 6)                        /* 0..3 */
#define BIOMORPH_QWIDTH_OF(g)   ((uint32_t)(g) >> 4)                        /* 0..15 */
/* height : width of an aspect bin, q8: 128 (0.5), 192 (0.75), 256 (1), 358, 460, 563, 665, 768 (3) */
#define BIOMORPH_ASPECT_Q8(bin) ((uint32_t)(bin) == 0u ? 128u : ((uint32_t)(bin) == 1u ? 192u : 256u + (((uint32_t)(bin) - 2u) * 512u) / 5u))

/* region: cells [offset, offset + w*h), row-major (z * w + x); cell (i, j)
 * covers world voxels x in [x0 + i*2^shift, x0 + (i+1)*2^shift), same for z */
typedef struct GmeBioRegion { int32_t x0, z0, w, h; uint32_t offset; } GmeBioRegion;

/* What generation needs. Passed BY VALUE to the CUDA generation kernel.
 * cells == NULL and fallback_on == 0  =>  no vegetation at all (output must
 * equal plain gme_material byte for byte).
 * morph == NULL  =>  version 1 shapes (byte-identical to a version 1 engine);
 * otherwise morph[] is parallel to cells[] and fallback_morph[] to fallback[]. */
typedef struct GmeBioView {
    const uint32_t* cells;            /* host pointer on CPU paths, device pointer on GPU */
    int32_t  nreg, shift;
    uint32_t seed, fallback_on;
    GmeBioRegion reg[BIOMAP_MAX_REGIONS];
    uint32_t fallback[BIOMAP_FALLBACK];
    const uint32_t* morph;            /* v2: one BIOMORPH word per cell; NULL = version 1 */
    uint32_t fallback_morph[BIOMAP_FALLBACK];
} GmeBioView;

/* Host-side biomap (file GMEBMAP1 or GMEBMAP2). */
typedef struct Biomap {
    uint64_t planet_hash;
    uint32_t seed;            /* world seed it was made for */
    uint32_t sim_year;        /* life sim time it shows */
    uint32_t gates;           /* stage gates reached (bit k = G_k) */
    int32_t  nreg, shift;
    uint32_t fallback_on;
    GmeBioRegion reg[BIOMAP_MAX_REGIONS];
    uint32_t fallback[BIOMAP_FALLBACK];
    uint8_t  palette[BIOMAP_PALETTE][3];  /* sRGB per palette index (index 0 = "use material colour") */
    uint32_t* cells;
    size_t    ncells;
    /* version 2 (all zero / NULL for a version 1 biomap) */
    uint32_t* morph;                      /* ncells BIOMORPH words, or NULL */
    uint32_t  fallback_morph[BIOMAP_FALLBACK];
    int32_t   atm_ppm[BIOMAP_ATM];        /* atmosphere at sim_year, mole fractions in ppm */
    uint32_t  atm_known;                  /* 1: atm_ppm is set (the sky may follow it) */
} Biomap;

#ifdef __cplusplus
extern "C" {
#endif

/* Writes GMEBMAP2 when b->morph is set, else GMEBMAP1 (byte-identical to before). 0 ok. */
int  biomap_save(const char* path, const Biomap* b);
int  biomap_load(const char* path, Biomap* b);         /* 0 ok; reads both versions; validates magic + crc32 */
void biomap_free(Biomap* b);
void biomap_view(const Biomap* b, GmeBioView* out);    /* host view (cells, morph = host pointers) */
/* Fill pal[1..15] with the hue wheel used by the exporter (pigment hue bins);
 * pal[0] = {0,0,0} meaning "no tint". */
void biomap_default_palette(uint8_t pal[BIOMAP_PALETTE][3]);
/* Test pattern: one 1024x1024-cell region (shift 4) centred on world (cx, cz)
 * with a checkerboard of every kind x palette x cover, fallback disabled. */
int  biomap_make_test(Biomap* b, uint64_t planet_hash, uint32_t seed, int32_t cx, int32_t cz);
/* The same pattern as version 2: adds morph[] cycling through every crown
 * class x aspect bin (and leaf angle, porosity, pigment, Q width) per block,
 * so every evolved shape shows somewhere. 0 ok. */
int  biomap_make_test2(Biomap* b, uint64_t planet_hash, uint32_t seed, int32_t cx, int32_t cz);

#ifdef __cplusplus
}
#endif
#endif

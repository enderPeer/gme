/* biomap.h - CONTRACT between the life simulation (exporter) and the engine
 * (voxel generation + renderer). See docs/biosphere-spec.md section 7.
 *
 * A biomap is a frozen snapshot of vegetation: up to 8 rectangular regions
 * of packed cells (one uint32 per 2^shift m square) plus a 64-entry fallback
 * table indexed by gme_env_bin() that vegetates terrain outside the regions.
 * Usable from C and CUDA (types only; functions are host-side). */
#ifndef GME_BIOMAP_H
#define GME_BIOMAP_H

#include <stddef.h>
#include <stdint.h>

#define BIOMAP_MAGIC       "GMEBMAP1"
#define BIOMAP_MAX_REGIONS 8
#define BIOMAP_FALLBACK    64
#define BIOMAP_PALETTE     16

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

/* region: cells [offset, offset + w*h), row-major (z * w + x); cell (i, j)
 * covers world voxels x in [x0 + i*2^shift, x0 + (i+1)*2^shift), same for z */
typedef struct GmeBioRegion { int32_t x0, z0, w, h; uint32_t offset; } GmeBioRegion;

/* What generation needs. Passed BY VALUE to the CUDA generation kernel.
 * cells == NULL and fallback_on == 0  =>  no vegetation at all (output must
 * equal plain gme_material byte for byte). */
typedef struct GmeBioView {
    const uint32_t* cells;            /* host pointer on CPU paths, device pointer on GPU */
    int32_t  nreg, shift;
    uint32_t seed, fallback_on;
    GmeBioRegion reg[BIOMAP_MAX_REGIONS];
    uint32_t fallback[BIOMAP_FALLBACK];
} GmeBioView;

/* Host-side biomap (file GMEBMAP1). */
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
} Biomap;

#ifdef __cplusplus
extern "C" {
#endif

int  biomap_save(const char* path, const Biomap* b);   /* 0 ok */
int  biomap_load(const char* path, Biomap* b);         /* 0 ok; validates magic + crc32 */
void biomap_free(Biomap* b);
void biomap_view(const Biomap* b, GmeBioView* out);    /* host view (cells = host pointer) */
/* Fill pal[1..15] with the hue wheel used by the exporter (pigment hue bins);
 * pal[0] = {0,0,0} meaning "no tint". */
void biomap_default_palette(uint8_t pal[BIOMAP_PALETTE][3]);
/* Test pattern: one 1024x1024-cell region (shift 4) centred on world (cx, cz)
 * with a checkerboard of every kind x palette x cover, fallback disabled. */
int  biomap_make_test(Biomap* b, uint64_t planet_hash, uint32_t seed, int32_t cx, int32_t cz);

#ifdef __cplusplus
}
#endif
#endif

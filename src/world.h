/* world.h - the 3D world space: sparse chunk residency around a viewer,
 * streamed by the GPU, mirrored on the host for CPU-side queries. */
#ifndef GME_WORLD_H
#define GME_WORLD_H

#include <stddef.h>
#include <stdint.h>
#include "platform.h"
#include "gpu.h"
#include "render.h"

/* Position = chunk coordinate + local offset in [0, 32). Precise anywhere. */
typedef struct WorldPos { int32_t cx, cy, cz; float lx, ly, lz; } WorldPos;
void worldpos_normalize(WorldPos* p);

typedef struct WorldConfig {
    uint32_t seed;
    GmePlanet planet;          /* geology from genesis (or gme_planet_default) */
    int32_t  min_cy, max_cy;   /* vertical chunk range [min, max), at most 32 layers */
    double   vram_fraction;    /* share of free VRAM for the chunk pool */
    double   ram_fraction;     /* share of available RAM for the host mirror */
    uint32_t max_batch;        /* chunks per GPU batch */
    int      force_cpu;        /* generate on CPU even if a GPU exists */
    int      radius_override;  /* fixed view radius in chunks, 0 = adaptive */
    /* vegetation (biomap_view of a loaded biomap; all zero = no life). The host
     * cells must stay valid until world_destroy; world_create uploads a copy
     * to the GPU. */
    GmeBioView bio;
    uint8_t    bio_palette[BIOMAP_PALETTE][3];   /* sRGB tint per palette index (0 = none) */
} WorldConfig;

typedef struct WorldStats {
    int      use_gpu;
    int32_t  radius, radius_max;      /* horizontal view radius, chunks */
    uint32_t layers;
    uint32_t slots_total, slots_used; /* chunk storage (device + host mirror) */
    uint32_t recs_total, recs_used;   /* chunk records (incl. uniform chunks) */
    uint32_t ready, pending, uniform_ready;
    uint64_t generated, generated_uniform, evicted, pressure_events;
    int      fully_loaded;
    size_t   device_bytes, pinned_bytes, host_reserved_bytes, host_meta_bytes;
    uint32_t tiles_total, tiles_ready;  /* column tiles (GPU mode): table cells, published */
    uint64_t tiles_built;
    size_t   tile_bytes;                /* device tile table */
} WorldStats;

typedef struct ValidateReport {
    uint32_t chunks_checked, uniform_checked;
    uint64_t voxels_checked;
    uint64_t host_mismatch, device_mismatch, meta_mismatch;
    uint32_t columns_checked, column_mismatch;
    /* vegetation in the checked chunks: chunk columns whose gme_column found a
     * biomap kind != 0, and voxels of flora materials (gme_is_flora). With a
     * biomap loaded, flora_checked == 0 fails validation (nothing was tested). */
    uint32_t flora_checked;
    uint64_t flora_voxels;
    /* generator fingerprint: FNV-1a 64 over the CPU reference generation of a
     * fixed block of chunks (WORLD_REF_COLS^2 columns x all layers) centred on
     * the first viewer column. It does not depend on timing or residency, so
     * two runs with the same seed, planet and biomap print the same value. */
    uint64_t ref_hash;
    uint32_t ref_chunks;
    /* smooth-surface tiles (density.h): published tiles read back from the GPU
     * and compared byte for byte with the CPU's gme_col_tile_build; decoded h
     * against gme_height and wet/dry against the voxel truth at every column.
     * In CPU mode the CPU tiles of the same columns are checked (tiles_checked
     * then counts 0 device tiles). dev_* / clamp_share: stored deviation
     * T - (h+1) over the tiles' own 32 x 32 columns. */
    uint32_t tiles_checked;                /* published tiles compared GPU vs CPU */
    uint32_t tiles_cpu;                    /* CPU tiles checked (both modes) */
    uint64_t tile_mismatch;                /* differing bytes */
    uint32_t tile_h_bad, tile_coast_bad, tile_sat;
    double   dev_mean, dev_rms, dev_max, clamp_share;
    /* density fingerprint: FNV-1a 64 over the CPU tiles of the
     * WORLD_REF_COLS^2 reference columns (the ref_hash block), x fastest */
    uint64_t dens_hash;
} ValidateReport;

#define WORLD_REF_COLS 8

typedef struct World World;

World*  world_create(const WorldConfig* cfg, const SysInfo* si, const GpuInfo* gi);
void    world_destroy(World* w);
int     world_update(World* w, const WorldPos* viewer); /* 0 ok, -1 fatal */
int     world_flush(World* w);                          /* wait for in-flight batches */
void    world_get_stats(const World* w, WorldStats* out);

/* Lookup table + pool for the renderer. Clears the dirty flag. */
void    world_get_scene(World* w, RenderScene* out);

/* CPU-side voxel query on the host mirror. *loaded = 0 if not resident. */
uint8_t world_get_voxel(const World* w, int32_t x, int32_t y, int32_t z, int* loaded);

/* Re-generates sampled chunks on the CPU and compares them bit-for-bit
 * with the host mirror and the GPU pool. Returns 0 if everything matches. */
int     world_validate(World* w, uint32_t samples, ValidateReport* rep);

/* The drawn (smooth) surface at column (x, z) in q8 metres, analytic: the
 * same T the renderer draws, within [h, h+2). Walking should use it. 0 ok. */
int     world_surface_q8(const World* w, int32_t x, int32_t z, int32_t* t_q8);

#endif

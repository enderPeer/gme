/* worldgen.h - world space layout + procedural generator shared by C (CPU)
 * and CUDA (GPU). Everything is integer / fixed-point, so CPU and GPU
 * produce bit-identical worlds from the same seed (needed for validation,
 * networking and save/load determinism). */
#ifndef GME_WORLDGEN_H
#define GME_WORLDGEN_H

#include <stdint.h>

#ifdef __CUDACC__
#define GME_HD static __host__ __device__ __forceinline__
#else
#define GME_HD static inline
#endif

/* ---- Space layout ------------------------------------------------------
 * 1 voxel = 1 m. Space is split into 32^3 chunks addressed by int32 chunk
 * coordinates; positions are (chunk, local float) so precision never
 * degrades far from the origin. */
#define GME_CHUNK_SHIFT  5
#define GME_CHUNK_DIM    32
#define GME_CHUNK_VOXELS (GME_CHUNK_DIM * GME_CHUNK_DIM * GME_CHUNK_DIM)
/* x fastest, then z, then y: one XZ layer = 1024 contiguous bytes. */
#define GME_VOXEL_INDEX(x, y, z) ((uint32_t)(x) | ((uint32_t)(z) << 5) | ((uint32_t)(y) << 10))

/* Voxel materials. Geology comes from the planet's chemistry (genesis);
 * GRASS is reserved for vegetation grown by life, never by geology. */
enum {
    MAT_AIR = 0,
    MAT_STONE,     /* basaltic rock: Mg/Fe silicates */
    MAT_DIRT,      /* soil: weathered rock + organics */
    MAT_GRASS,     /* vegetation (life) */
    MAT_SAND,
    MAT_WATER,
    MAT_GRANITE,   /* Si-rich silicate rock */
    MAT_IRON_ORE,  /* iron oxides */
    MAT_SULFIDE,   /* iron/magnesium sulfides */
    MAT_ICE,
    MAT_REGOLITH,  /* bare, organic-free surface dust */
    /* life (flora.h, from a frozen biomap). GRASS and MAT replace the ground
     * voxel (y == h); LEAF, WOOD and KELP stand on it (gme_is_standing_flora). */
    MAT_MAT,       /* 11 microbial / algal mat on seabed and beach */
    MAT_LEAF,      /* 12 leaves, low ground-cover tufts */
    MAT_WOOD,      /* 13 trunks */
    MAT_KELP,      /* 14 standing algae in the water column (replaces water) */
    MAT_COUNT
};

#define GME_CAVE_MIN_Y  (-160)
#define GME_HEIGHT_MAX  224

/* Planet parameters consumed by the generator. Integers only, so CPU and GPU
 * stay bit-identical. Produced by genesis (planet.c) or gme_planet_default. */
typedef struct GmePlanet {
    int32_t sea_level;      /* voxel y of the ocean surface; very low = no ocean */
    int32_t relief_q8;      /* terrain amplitude, 256 = reference */
    int32_t granite_q8;     /* share of granite in the crust, 0..256 */
    int32_t ore_band;       /* half-width of iron-ore vein noise band, 0 = none */
    int32_t sulfide_band;   /* same for sulfides */
    int32_t soil_depth;     /* voxels of soil on land, 0 = bare rock */
    int32_t surface_mat;    /* top of land: MAT_DIRT or MAT_REGOLITH */
    int32_t beach_mat;
    int32_t ocean_mat;      /* MAT_WATER, or MAT_ICE for a frozen world */
    int32_t ocean_top_mat;  /* MAT_ICE for an ice sheet over liquid water */
    int32_t cave_width;
    int32_t pad;
} GmePlanet;

GME_HD GmePlanet gme_planet_default(void) {
    GmePlanet p;
    p.sea_level = 0; p.relief_q8 = 256; p.granite_q8 = 96;
    p.ore_band = 900; p.sulfide_band = 300; p.soil_depth = 3;
    p.surface_mat = MAT_DIRT; p.beach_mat = MAT_SAND;
    p.ocean_mat = MAT_WATER; p.ocean_top_mat = MAT_WATER;
    p.cave_width = 2600; p.pad = 0;
    return p;
}

/* Per-chunk generation request / result (shared with gpu.cu). */
typedef struct GmeGenJob { int32_t cx, cy, cz; uint32_t slot; } GmeGenJob;
typedef struct GmeChunkMeta { uint32_t mat_min, mat_max, non_air, pad; } GmeChunkMeta;

/* GPU-visible chunk lookup: a toroidal (cx mod n, cz mod n, layer) table.
 * val >= 0: storage slot; -1: not loaded; <= -2: uniform chunk of material (-2 - val).
 * tag identifies which chunk currently owns the cell. */
typedef struct GmePageEntry { uint32_t tag; int32_t val; } GmePageEntry;
#define GME_PAGE_UNLOADED (-1)

GME_HD uint32_t gme_page_index(int32_t cx, int32_t layer, int32_t cz, uint32_t n, uint32_t layers) {
    int32_t mx = cx % (int32_t)n, mz = cz % (int32_t)n;
    if (mx < 0) mx += (int32_t)n;
    if (mz < 0) mz += (int32_t)n;
    return ((uint32_t)mx * n + (uint32_t)mz) * layers + (uint32_t)layer;
}

GME_HD uint32_t gme_page_tag(int32_t cx, int32_t cz) {
    return ((uint32_t)cx & 0xFFFFu) | ((uint32_t)cz << 16);
}

/* ---- Fixed-point value noise (16.16) ----------------------------------- */
GME_HD uint32_t gme_hash3(int32_t x, int32_t y, int32_t z, uint32_t seed) {
    uint32_t h = seed;
    h ^= (uint32_t)x * 0x8DA6B343u;
    h ^= (uint32_t)y * 0xD8163841u;
    h ^= (uint32_t)z * 0xCB1AB31Fu;
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

GME_HD int32_t gme_hval(uint32_t h) { return (int32_t)(h & 0xFFFFu) - 32768; }

/* smoothstep on t in [0, 65536] */
GME_HD int32_t gme_fade(int32_t t) {
    int64_t tt = (int64_t)t * t;
    return (int32_t)((tt * (196608 - 2 * (int64_t)t)) >> 32);
}

GME_HD int32_t gme_lerp(int32_t a, int32_t b, int32_t s) {
    return a + (int32_t)(((int64_t)(b - a) * s) >> 16);
}

/* Gradient of one lattice corner dotted with the offset to it (16.16).
 * 8 directions: axes and diagonals. */
GME_HD int32_t gme_grad2(uint32_t h, int32_t dx, int32_t dz) {
    switch (h & 7u) {
    case 0: return  dx + dz;
    case 1: return -dx + dz;
    case 2: return  dx - dz;
    case 3: return -dx - dz;
    case 4: return  dx;
    case 5: return -dx;
    case 6: return  dz;
    default: return -dz;
    }
}

/* 2D gradient (Perlin-style) noise, integer only. Cell size = 2^shift voxels
 * (1 <= shift <= 16). Output roughly [-32768, 32767]. Unlike value noise its
 * slope does not vanish along lattice lines, so no grid seams in terrain. */
GME_HD int32_t gme_noise2(int32_t x, int32_t z, int shift, uint32_t seed) {
    const int32_t m = (1 << shift) - 1;
    const int up = 16 - shift;
    const int32_t cx = x >> shift, cz = z >> shift;
    const int32_t rx = (x & m) << up, rz = (z & m) << up;   /* 0..65535 */
    const int32_t fx = gme_fade(rx), fz = gme_fade(rz);
    const int32_t a = gme_grad2(gme_hash3(cx,     0, cz,     seed), rx,         rz);
    const int32_t b = gme_grad2(gme_hash3(cx + 1, 0, cz,     seed), rx - 65536, rz);
    const int32_t c = gme_grad2(gme_hash3(cx,     0, cz + 1, seed), rx,         rz - 65536);
    const int32_t d = gme_grad2(gme_hash3(cx + 1, 0, cz + 1, seed), rx - 65536, rz - 65536);
    return gme_lerp(gme_lerp(a, b, fx), gme_lerp(c, d, fx), fz) >> 1;
}

GME_HD int32_t gme_noise3(int32_t x, int32_t y, int32_t z, int shift, uint32_t seed) {
    const int32_t m = (1 << shift) - 1;
    const int up = 16 - shift;
    const int32_t cx = x >> shift, cy = y >> shift, cz = z >> shift;
    const int32_t fx = gme_fade((x & m) << up);
    const int32_t fy = gme_fade((y & m) << up);
    const int32_t fz = gme_fade((z & m) << up);
    const int32_t x00 = gme_lerp(gme_hval(gme_hash3(cx, cy,     cz,     seed)), gme_hval(gme_hash3(cx + 1, cy,     cz,     seed)), fx);
    const int32_t x10 = gme_lerp(gme_hval(gme_hash3(cx, cy + 1, cz,     seed)), gme_hval(gme_hash3(cx + 1, cy + 1, cz,     seed)), fx);
    const int32_t x01 = gme_lerp(gme_hval(gme_hash3(cx, cy,     cz + 1, seed)), gme_hval(gme_hash3(cx + 1, cy,     cz + 1, seed)), fx);
    const int32_t x11 = gme_lerp(gme_hval(gme_hash3(cx, cy + 1, cz + 1, seed)), gme_hval(gme_hash3(cx + 1, cy + 1, cz + 1, seed)), fx);
    return gme_lerp(gme_lerp(x00, x10, fy), gme_lerp(x01, x11, fy), fz);
}

GME_HD int32_t gme_fbm2(int32_t x, int32_t z, int shift, int octaves, uint32_t seed) {
    int32_t sum = 0;
    for (int o = 0; o < octaves; ++o)
        sum += gme_noise2(x, z, shift - o, seed + (uint32_t)o * 0x9E3779B9u) >> o;
    return sum;
}

/* ---- Terrain ------------------------------------------------------------ */
/* Surface height (voxel y of the top solid voxel) for column (x, z). */
/* Unscaled terrain shape, roughly [-206, 206]. */
GME_HD int32_t gme_height_raw(int32_t x, int32_t z, uint32_t seed) {
    const int32_t cont   = gme_noise2(x, z, 11, seed ^ 0xA511E9B3u);  /* continents, 2 km cells */
    const int32_t detail = gme_fbm2(x, z, 8, 5, seed);                /* hills, 256 m .. 16 m */
    const int32_t mount  = cont > 0 ? cont : 0;
    const int32_t amp    = 20 + ((mount * 110) >> 15);                /* mountains on high ground */
    return ((cont * 80) >> 15) + (int32_t)(((int64_t)detail * amp) >> 16);
}

/* Surface height (voxel y of the top solid voxel) for column (x, z).
 * Relief scales with the planet (weaker gravity -> taller mountains). */
GME_HD int32_t gme_height(int32_t x, int32_t z, uint32_t seed, const GmePlanet* p) {
    int32_t h = (gme_height_raw(x, z, seed) * p->relief_q8) >> 8;
    if (h >  GME_HEIGHT_MAX) h =  GME_HEIGHT_MAX;
    if (h < -GME_HEIGHT_MAX) h = -GME_HEIGHT_MAX;
    return h;
}

/* Material at voxel (x, y, z) given the column height h = gme_height(x, z). */
GME_HD uint8_t gme_material(int32_t x, int32_t y, int32_t z, int32_t h, uint32_t seed, const GmePlanet* p) {
    const int32_t depth = h - y;
    if (depth < 0) {
        if (y > p->sea_level) return MAT_AIR;
        return (uint8_t)(y == p->sea_level ? p->ocean_top_mat : p->ocean_mat);
    }
    if (depth > 6 && y >= GME_CAVE_MIN_Y) {
        /* tunnels = intersection of two thin noise bands */
        const int32_t a = gme_noise3(x, y, z, 5, seed ^ 0x51ED270Bu);
        if (a > -p->cave_width && a < p->cave_width) {
            const int32_t b = gme_noise3(x, y, z, 5, seed ^ 0x2C1B3C6Du);
            if (b > -p->cave_width && b < p->cave_width) return MAT_AIR;
        }
    }
    const int beach = h <= p->sea_level + 2 && h >= p->sea_level - 3;
    if (depth == 0 && beach) return (uint8_t)p->beach_mat;
    if (depth < p->soil_depth) return (uint8_t)(beach ? p->beach_mat : p->surface_mat);
    if (depth == 0) return MAT_REGOLITH; /* no soil: weathered rock dust on top */

    /* crust: one noise field carves ore veins (band around 0) and sulfide
     * veins (band around +16384) out of granite / basalt provinces */
    if (depth > 2 && depth < 96) { /* veins only in the reachable crust: deep rock stays uniform (cheap) */
        const int32_t v = gme_noise3(x, y, z, 4, seed ^ 0x0BE5EED1u);
        if (v > -p->ore_band && v < p->ore_band) return MAT_IRON_ORE;
        if (v > 16384 - p->sulfide_band && v < 16384 + p->sulfide_band) return MAT_SULFIDE;
    }
    const int32_t prov = gme_noise2(x, z, 9, seed ^ 0x6A417E00u); /* 512 m rock provinces */
    return (uint8_t)((((prov + 32768) >> 8) < p->granite_q8) ? MAT_GRANITE : MAT_STONE);
}

#endif

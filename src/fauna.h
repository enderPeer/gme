/* fauna.h - CONTRACT between the life simulation (writer: gme_genesis
 * bioexport --fauna) and the engine (reader: creatures in the 3D view).
 * docs/biosphere-spec.md 7.5.
 *
 * GMEFAUN1: a frozen snapshot of the creatures of the simulated islands:
 * agents with a motor that move at 1 m per hour or more at full effort or live
 * mostly on prey. What the vegetation is drawn from (a phototroph with a
 * holdfast) is in here only if it does both, and is then left out there.
 * One record per agent (a super-individual: a herd, swarm or one large
 * animal), at most FAUNA_MAX of them, the largest bodies first.
 * Positions are world voxel coordinates in 1/16 m; the engine puts a land
 * animal on its own ground and a swimmer at its depth below the water plane,
 * so the file carries no heights and never disagrees with the terrain.
 * Header-only, integer-only, static inline, C and C++ (as spectrum_file.h).
 *
 * File, little-endian: a 128-byte header, then count x 16-byte records.
 *    0  char[8] "GMEFAUN1"     8  u32 version (1)     12  u32 header bytes (128)
 *   16  u64 planet_hash       24  u32 seed (world)   28  u32 sim_year
 *   32  u32 count             36  u32 record bytes (16)
 *   40  u32 total creatures alive (the file may hold fewer)
 *   44  u32 reserved (0)      48  u8 palette[16][3] (sRGB per palette index, as the biomap's)
 *   96  u32 reserved[6] (0)  120  u32 crc32 of the records   124  u32 crc32 of bytes 0..123
 * fauna_load returns 0 ok, -1 cannot open / short read / out of memory,
 * -2 not a valid GMEFAUN1 header, -3 CRC mismatch.
 * fauna_save writes "<path>.tmp" and renames it over <path>; 0 ok. */
#ifndef GME_FAUNA_H
#define GME_FAUNA_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FAUNA_MAGIC   "GMEFAUN1"
#define FAUNA_VERSION 1u
#define FAUNA_HDR     128u
#define FAUNA_REC     16u
#define FAUNA_MAX     65536u

/* record flags */
#define FAUNA_SWIMS    1u    /* lives in the water column (depth_q4 below the water plane); else walks on the ground */
#define FAUNA_HUNTER   2u    /* eats whole prey (trophic level >= 2.5); else a grazer of anchored producers */
#define FAUNA_ARMORED  4u    /* armour gene >= 128: a shell, plates */
#define FAUNA_SWARM    8u    /* many small individuals rather than one body (reproduction mass far below the body) */

typedef struct FaunaRec {    /* 16 bytes */
    int32_t  x_q4, z_q4;     /* world position, 1/16 m (voxel x = x_q4 >> 4) */
    uint16_t depth_q4;       /* swimmers: depth below the water plane, 1/16 m; walkers: 0 */
    uint16_t size_q4;        /* diameter of a sphere of the body's volume, 1/16 m (0.12 cm^3 per mmol C) */
    uint8_t  flags;          /* FAUNA_* */
    uint8_t  pal;            /* palette index 1..15 (pigment hue bin, as the biomap's) */
    uint8_t  troph_q4;       /* trophic level, 1/16 (16 = a producer's level 1, 32 = a grazer) */
    uint8_t  motor;          /* locomotion gene 1..255: speed of the animation */
} FaunaRec;

typedef struct Fauna {
    uint64_t planet_hash;
    uint32_t seed, sim_year;
    uint32_t count, total;
    uint8_t  palette[16][3];
    FaunaRec* rec;           /* count records (malloc; fauna_free) */
} Fauna;

static inline uint32_t fauna_crc(uint32_t crc, const uint8_t* p, size_t n) {
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) {
        crc ^= p[i];
        for (int k = 0; k < 8; ++k) crc = (crc & 1u) ? 0xEDB88320u ^ (crc >> 1) : crc >> 1;
    }
    return ~crc;
}
static inline void fauna_put32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
static inline void fauna_put16(uint8_t* p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static inline uint32_t fauna_get32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline uint32_t fauna_get16(const uint8_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }

static inline void fauna_free(Fauna* f) {
    if (!f) return;
    free(f->rec);
    memset(f, 0, sizeof *f);
}

static inline int fauna_save(const char* path, const Fauna* f) {
    if (!path || !f || f->count > FAUNA_MAX || (f->count && !f->rec)) return -1;
    const size_t pb = (size_t)f->count * FAUNA_REC;
    uint8_t* buf = (uint8_t*)calloc(1, FAUNA_HDR + (pb ? pb : 1));
    if (!buf) return -1;
    for (uint32_t i = 0; i < f->count; ++i) {
        uint8_t* q = buf + FAUNA_HDR + (size_t)i * FAUNA_REC;
        const FaunaRec* r = &f->rec[i];
        fauna_put32(q, (uint32_t)r->x_q4); fauna_put32(q + 4, (uint32_t)r->z_q4);
        fauna_put16(q + 8, r->depth_q4); fauna_put16(q + 10, r->size_q4);
        q[12] = r->flags; q[13] = r->pal; q[14] = r->troph_q4; q[15] = r->motor;
    }
    memcpy(buf, FAUNA_MAGIC, 8);
    fauna_put32(buf + 8, FAUNA_VERSION);
    fauna_put32(buf + 12, FAUNA_HDR);
    fauna_put32(buf + 16, (uint32_t)f->planet_hash);
    fauna_put32(buf + 20, (uint32_t)(f->planet_hash >> 32));
    fauna_put32(buf + 24, f->seed);
    fauna_put32(buf + 28, f->sim_year);
    fauna_put32(buf + 32, f->count);
    fauna_put32(buf + 36, FAUNA_REC);
    fauna_put32(buf + 40, f->total);
    memcpy(buf + 48, f->palette, 48);
    fauna_put32(buf + 120, fauna_crc(0u, buf + FAUNA_HDR, pb));
    fauna_put32(buf + 124, fauna_crc(0u, buf, 124));
    char tmp[1024];
    const size_t pl = strlen(path);
    if (pl + 5 > sizeof tmp) { free(buf); return -1; }
    memcpy(tmp, path, pl);
    memcpy(tmp + pl, ".tmp", 5);
    FILE* fp = fopen(tmp, "wb");
    int ok = fp != NULL;
    if (ok) ok = fwrite(buf, 1, FAUNA_HDR + pb, fp) == FAUNA_HDR + pb;
    if (fp && fclose(fp)) ok = 0;
    free(buf);
    if (ok) {
        remove(path);
        ok = rename(tmp, path) == 0;
    }
    if (!ok) remove(tmp);
    return ok ? 0 : -1;
}

static inline int fauna_load(const char* path, Fauna* f) {
    memset(f, 0, sizeof *f);
    FILE* fp = path ? fopen(path, "rb") : NULL;
    if (!fp) return -1;
    uint8_t hd[FAUNA_HDR];
    int rc = -2;
    uint8_t* buf = NULL;
    FaunaRec* rec = NULL;
    if (fread(hd, 1, FAUNA_HDR, fp) != FAUNA_HDR) { rc = -1; goto fail; }
    if (memcmp(hd, FAUNA_MAGIC, 8) || fauna_get32(hd + 8) != FAUNA_VERSION || fauna_get32(hd + 12) != FAUNA_HDR ||
        fauna_get32(hd + 36) != FAUNA_REC || fauna_get32(hd + 124) != fauna_crc(0u, hd, 124))
        goto fail;
    {
        const uint32_t count = fauna_get32(hd + 32);
        if (count > FAUNA_MAX) goto fail;
        const size_t pb = (size_t)count * FAUNA_REC;
        rc = -1;
        buf = (uint8_t*)malloc(pb ? pb : 1);
        rec = (FaunaRec*)malloc(sizeof(FaunaRec) * (count ? count : 1));
        if (!buf || !rec) goto fail;
        if (pb && fread(buf, 1, pb, fp) != pb) goto fail;
        rc = -2;
        if (fgetc(fp) != EOF) goto fail;
        rc = -3;
        if (fauna_crc(0u, buf, pb) != fauna_get32(hd + 120)) goto fail;
        for (uint32_t i = 0; i < count; ++i) {
            const uint8_t* q = buf + (size_t)i * FAUNA_REC;
            FaunaRec* r = &rec[i];
            r->x_q4 = (int32_t)fauna_get32(q); r->z_q4 = (int32_t)fauna_get32(q + 4);
            r->depth_q4 = (uint16_t)fauna_get16(q + 8); r->size_q4 = (uint16_t)fauna_get16(q + 10);
            r->flags = q[12]; r->pal = q[13]; r->troph_q4 = q[14]; r->motor = q[15];
        }
        f->planet_hash = (uint64_t)fauna_get32(hd + 16) | ((uint64_t)fauna_get32(hd + 20) << 32);
        f->seed = fauna_get32(hd + 24);
        f->sim_year = fauna_get32(hd + 28);
        f->count = count;
        f->total = fauna_get32(hd + 40);
        memcpy(f->palette, hd + 48, 48);
        f->rec = rec;
    }
    fclose(fp);
    free(buf);
    return 0;
fail:
    fclose(fp);
    free(buf);
    free(rec);
    return rc;
}

/* Deterministic test herd for the engine (no life simulation needed): n
 * creatures of every size, palette, flag combination and trophic level on a
 * grid of `step` metres centred on world (cx, cz). Swimmers get depths of
 * 1..12 m; the engine drops records whose habitat does not match the terrain
 * (a swimmer over land, a walker over water). 0 ok. */
static inline int fauna_make_test(Fauna* f, uint64_t planet_hash, uint32_t seed, int32_t cx, int32_t cz, uint32_t n, int32_t step) {
    memset(f, 0, sizeof *f);
    if (n > FAUNA_MAX) n = FAUNA_MAX;
    f->rec = (FaunaRec*)calloc(n ? n : 1, sizeof(FaunaRec));
    if (!f->rec) return -1;
    uint32_t side = 1;
    while (side * side < n) ++side;
    for (uint32_t i = 0; i < n; ++i) {
        FaunaRec* r = &f->rec[i];
        const int32_t gx = (int32_t)(i % side) - (int32_t)side / 2, gz = (int32_t)(i / side) - (int32_t)side / 2;
        uint32_t h = i * 0x9E3779B1u + seed;
        h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12;
        r->x_q4 = (cx + gx * step) * 16 + (int32_t)(h & 127u);
        r->z_q4 = (cz + gz * step) * 16 + (int32_t)((h >> 7) & 127u);
        r->flags = (uint8_t)(((i & 1u) ? FAUNA_SWIMS : 0u) | ((i & 2u) ? FAUNA_HUNTER : 0u) | (((i >> 2) % 3u == 0u) ? FAUNA_ARMORED : 0u) |
                             (((i >> 3) % 4u == 0u) ? FAUNA_SWARM : 0u));
        r->depth_q4 = (uint16_t)((r->flags & FAUNA_SWIMS) ? 16u * (1u + (h >> 14) % 12u) : 0u);
        r->size_q4 = (uint16_t)(4u + (h >> 18) % 60u);            /* 0.25 .. 4 m */
        r->pal = (uint8_t)(1u + i % 15u);
        r->troph_q4 = (uint8_t)((r->flags & FAUNA_HUNTER) ? 48u : 32u);
        r->motor = (uint8_t)(32u + (h >> 24) % 224u);
    }
    f->planet_hash = planet_hash;
    f->seed = seed;
    f->count = n;
    f->total = n;
    return 0;
}

#endif

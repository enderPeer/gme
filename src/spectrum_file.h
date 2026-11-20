/* spectrum_file.h - CONTRACT between the engine (writer: gme --export-spectrum)
 * and the life simulation (reader, phase B of docs/render-upgrade-spec.md 5.2).
 *
 * GMESPEC1: the planet's light as integer tables, computed once on the PC in
 * double precision (spectral.c, the renderer's own atmosphere and water
 * model) and shipped as integers, so a reader never calls libm and every rig
 * gets the same bits. Header-only, integer-only, static inline, C and C++.
 *
 * Bands: the renderer's 8 bands of 45 nm (380-740 nm) plus NIR 740-900 nm.
 * Haze bins: haze_q12 = 4096 - 256 k (the bio's haze factor, which is a
 * function of the CH4 mole fraction alone: x = 0.2 r / (1 - r), r =
 * (4096 - haze_q12) / 2048). Bin k is the planet's atmosphere with that CH4
 * fraction (the other gases rescaled) and the tholin haze calibrated so the
 * PAR-weighted global transmission equals haze_q12 / 4096 where Rayleigh +
 * gas allow it (else no haze). A reader interpolates: k = clamp((4096 -
 * haze_q12) >> 8, 0, 6), f = (4096 - haze_q12) & 255 (clamped to 255 at k = 6
 * when haze_q12 < 2304), value = lerp(t[k], t[k + 1], f / 256).
 * Hours: the bio's 24 hours, sun elevation sin = bio_sin24(hr) / 4096 (rise
 * 06:00, noon 12:00, set 18:00); 0 where the sun is below the horizon.
 *
 * File: little-endian, a fixed 64-byte header, then the payload as int32:
 *    0  char[8] "GMESPEC1"     8  u32 version (1)     12  u32 header bytes (64)
 *   16  u64 planet_hash       24  u32 payload bytes  28  u32 bands (9)
 *   32  u32 haze bins (8)     36  u32 hours (24)     40  u32 depths (33)
 *   44  u32 hues (16)         48  u32 reserved (0)   52  u32 reserved (0)
 *   56  u32 crc32 of the payload                     60  u32 crc32 of bytes 0..59
 *   64  payload: the SpectrumFile fields after planet_hash, in declaration order
 * spectrum_load returns 0 ok, -1 cannot open / short read, -2 not a valid
 * GMESPEC1 header (or other dimensions), -3 CRC mismatch.
 * spectrum_save writes "<path>.tmp" and renames it over <path>; 0 ok. */
#ifndef GME_SPECTRUM_FILE_H
#define GME_SPECTRUM_FILE_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SPECF_MAGIC   "GMESPEC1"
#define SPECF_VERSION 1u
#define SPECF_HDR     64u
#define SPECF_NB      9      /* 8 render bands + NIR 740-900 nm */
#define SPECF_NHAZE   8      /* haze_q12 bins 4096 - 256 k */
#define SPECF_NHOUR   24
#define SPECF_NDEPTH  33     /* water transmittance at 0..32 m */
#define SPECF_NHUE    16     /* pigment hue bins (biomap palette index; 0 = the default pigment, bin 8) */

typedef struct SpectrumFile {
    uint64_t planet_hash;                                   /* planet_hash() of the planet it was made for */
    int32_t  band_edge_nm[SPECF_NB + 1];                    /* 380 425 ... 740 900 */
    int32_t  s_top_wm2, t_surf_ck;                          /* as the bio env: round(1361 L / a^2); surface T x 100 */
    int32_t  surf_q16[SPECF_NHAZE][SPECF_NHOUR][SPECF_NB];  /* global horizontal irradiance in the band / S_top, q16 */
    int32_t  dir_q16[SPECF_NHAZE][SPECF_NHOUR][SPECF_NB];   /* its direct-beam part */
    int32_t  water_T_q16[SPECF_NB][SPECF_NDEPTH];           /* band-averaged transmittance of water at 0..32 m */
    int32_t  kw_q16[SPECF_NB];                              /* effective attenuation over 0-20 m, per m (bio_light's kw scale) */
    int32_t  phot_q16[SPECF_NB];                            /* umol photons per J of the band's light */
    int32_t  ev_q16[SPECF_NB];                              /* mean photon energy, eV */
    int32_t  uphill_q16[SPECF_NB][2];                       /* min(1, exp(-(E_RC - E_ph) / kT)): [0] OXY (RC 680 nm), [1] ANOX (870 nm) */
    int32_t  eps_q12[SPECF_NHUE][SPECF_NB];                 /* pigment shape per hue bin: effective epsilon, q12 (index 147.4 OD) */
} SpectrumFile;

#define SPECF_PAYLOAD_INTS                                                                                                 \
    ((SPECF_NB + 1) + 2 + 2 * SPECF_NHAZE * SPECF_NHOUR * SPECF_NB + SPECF_NB * SPECF_NDEPTH + 3 * SPECF_NB + 2 * SPECF_NB + \
     SPECF_NHUE * SPECF_NB)
#define SPECF_PAYLOAD_BYTES (4u * (uint32_t)SPECF_PAYLOAD_INTS)

/* ---- crc32 (IEEE 802.3, reflected), as biomap.c ---- */
static inline uint32_t specf_crc(uint32_t crc, const uint8_t* p, size_t n) {
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) {
        crc ^= p[i];
        for (int k = 0; k < 8; ++k) crc = (crc & 1u) ? 0xEDB88320u ^ (crc >> 1) : crc >> 1;
    }
    return ~crc;
}
static inline void specf_put32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
static inline uint32_t specf_get32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* the int32 fields after planet_hash, in declaration order */
static inline const int32_t* specf_field(const SpectrumFile* s, int k, int* n) {
    switch (k) {
    case 0: *n = SPECF_NB + 1; return s->band_edge_nm;
    case 1: *n = 1; return &s->s_top_wm2;
    case 2: *n = 1; return &s->t_surf_ck;
    case 3: *n = SPECF_NHAZE * SPECF_NHOUR * SPECF_NB; return &s->surf_q16[0][0][0];
    case 4: *n = SPECF_NHAZE * SPECF_NHOUR * SPECF_NB; return &s->dir_q16[0][0][0];
    case 5: *n = SPECF_NB * SPECF_NDEPTH; return &s->water_T_q16[0][0];
    case 6: *n = SPECF_NB; return s->kw_q16;
    case 7: *n = SPECF_NB; return s->phot_q16;
    case 8: *n = SPECF_NB; return s->ev_q16;
    case 9: *n = 2 * SPECF_NB; return &s->uphill_q16[0][0];
    case 10: *n = SPECF_NHUE * SPECF_NB; return &s->eps_q12[0][0];
    default: *n = 0; return NULL;
    }
}

/* The whole file in memory (SPECF_HDR + SPECF_PAYLOAD_BYTES bytes at out). */
static inline void spectrum_serialize(const SpectrumFile* s, uint8_t* out) {
    uint8_t* q = out + SPECF_HDR;
    for (int k = 0;; ++k) {
        int n;
        const int32_t* v = specf_field(s, k, &n);
        if (!v) break;
        for (int i = 0; i < n; ++i, q += 4) specf_put32(q, (uint32_t)v[i]);
    }
    memset(out, 0, SPECF_HDR);
    memcpy(out, SPECF_MAGIC, 8);
    specf_put32(out + 8, SPECF_VERSION);
    specf_put32(out + 12, SPECF_HDR);
    specf_put32(out + 16, (uint32_t)s->planet_hash);
    specf_put32(out + 20, (uint32_t)(s->planet_hash >> 32));
    specf_put32(out + 24, SPECF_PAYLOAD_BYTES);
    specf_put32(out + 28, SPECF_NB);
    specf_put32(out + 32, SPECF_NHAZE);
    specf_put32(out + 36, SPECF_NHOUR);
    specf_put32(out + 40, SPECF_NDEPTH);
    specf_put32(out + 44, SPECF_NHUE);
    specf_put32(out + 56, specf_crc(0u, out + SPECF_HDR, SPECF_PAYLOAD_BYTES));
    specf_put32(out + 60, specf_crc(0u, out, 60));
}

/* Parses a file image of n bytes. 0 ok, -2 bad header / dimensions / size, -3 CRC mismatch. */
static inline int spectrum_parse(const uint8_t* in, size_t n, SpectrumFile* s) {
    if (n < SPECF_HDR || memcmp(in, SPECF_MAGIC, 8) || specf_get32(in + 8) != SPECF_VERSION || specf_get32(in + 12) != SPECF_HDR ||
        specf_get32(in + 24) != SPECF_PAYLOAD_BYTES || specf_get32(in + 28) != SPECF_NB || specf_get32(in + 32) != SPECF_NHAZE ||
        specf_get32(in + 36) != SPECF_NHOUR || specf_get32(in + 40) != SPECF_NDEPTH || specf_get32(in + 44) != SPECF_NHUE)
        return -2;
    if (specf_get32(in + 60) != specf_crc(0u, in, 60)) return -2;
    if (n != (size_t)SPECF_HDR + SPECF_PAYLOAD_BYTES) return -2;
    if (specf_get32(in + 56) != specf_crc(0u, in + SPECF_HDR, SPECF_PAYLOAD_BYTES)) return -3;
    memset(s, 0, sizeof *s);
    s->planet_hash = (uint64_t)specf_get32(in + 16) | ((uint64_t)specf_get32(in + 20) << 32);
    const uint8_t* q = in + SPECF_HDR;
    for (int k = 0;; ++k) {
        int m;
        int32_t* v = (int32_t*)specf_field(s, k, &m);
        if (!v) break;
        for (int i = 0; i < m; ++i, q += 4) v[i] = (int32_t)specf_get32(q);
    }
    return 0;
}

static inline int spectrum_save(const char* path, const SpectrumFile* s) {
    static uint8_t buf[SPECF_HDR + SPECF_PAYLOAD_BYTES];
    if (!path || !s) return -1;
    spectrum_serialize(s, buf);
    char tmp[1024];
    const size_t pl = strlen(path);
    if (pl + 5 > sizeof tmp) return -1;
    memcpy(tmp, path, pl);
    memcpy(tmp + pl, ".tmp", 5);
    FILE* f = fopen(tmp, "wb");
    int ok = f != NULL;
    if (ok) ok = fwrite(buf, 1, sizeof buf, f) == sizeof buf;
    if (f && fclose(f)) ok = 0;
    if (ok) {
        remove(path);
        ok = rename(tmp, path) == 0;
    }
    if (!ok) remove(tmp);
    return ok ? 0 : -1;
}

static inline int spectrum_load(const char* path, SpectrumFile* s) {
    static uint8_t buf[SPECF_HDR + SPECF_PAYLOAD_BYTES + 1];
    FILE* f = path ? fopen(path, "rb") : NULL;
    if (!f) return -1;
    const size_t n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    if (n < SPECF_HDR) return -1;
    return spectrum_parse(buf, n, s);
}

/* Reader helper (integer only): table value at haze_q12 by the bio's interpolation. */
static inline int32_t spectrum_haze_lerp(int32_t v_k, int32_t v_k1, int32_t f) {
    return v_k + (int32_t)(((int64_t)(v_k1 - v_k) * f) >> 8);
}
static inline void spectrum_haze_bin(int32_t haze_q12, int32_t* k, int32_t* f) {
    const int32_t d = 4096 - haze_q12;
    int32_t kk = d >> 8, ff = d & 255;
    if (d < 0) { kk = 0; ff = 0; }
    if (kk > SPECF_NHAZE - 2) { kk = SPECF_NHAZE - 2; ff = 255; }
    *k = kk;
    *f = ff;
}

#endif

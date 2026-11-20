/* bio_core.h - all per-cell and per-agent math of the life simulation, shared
 * by the CPU reference (C, host compiler) and the GPU kernels (C++, NVRTC).
 * Integer only. Every element transfer (C, N, S, Fe, P) is a matched pair of
 * integer adds, so the totals are conserved to the bit.
 *
 * Features (BioParams.feat, bio_types.h; all off = the first release, bit for
 * bit): BIO_FEAT_PRED predation and grazing (spec 3.8: offers and claims, gut,
 * trophic level, brain inputs 11-14, output 2), BIO_FEAT_SEX crossover with a
 * compatible cellmate (G_SEX, mate_tag), BIO_FEAT_LEARN learning by
 * exploration, BIO_FEAT_MORPH crown shapes (genes 37-41), BIO_FEAT_HUB gas
 * exchange ledgers per realm; BioParams.light_mode = BIO_LIGHT_SPECTRAL light
 * in 9 wavelength bands.
 * Deferred (genes exist, no behaviour yet): kin sharing (output 7,
 * G_KINSHARE), spore-release timing (output 6), wind-directed spores on land,
 * G_STORE. */
#ifndef GME_BIO_CORE_H
#define GME_BIO_CORE_H
#ifndef __CUDACC_RTC__
#include "bio_types.h"
#endif

/* ==== small integer helpers ===================================================== */
BIO_FN int64_t bio_min64(int64_t a, int64_t b) { return a < b ? a : b; }
BIO_FN int64_t bio_max64(int64_t a, int64_t b) { return a > b ? a : b; }
BIO_FN int32_t bio_clampi(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }
BIO_FN int64_t bio_clamp64(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : (v > hi ? hi : v); }
BIO_FN int32_t bio_popc8(uint32_t v) { v &= 255u; v = v - ((v >> 1) & 0x55u); v = (v & 0x33u) + ((v >> 2) & 0x33u); return (int32_t)((v + (v >> 4)) & 15u); }

/* floor(a * b / 2^s) for unsigned a, b (exact 128-bit product), saturating at 2^62 */
BIO_FN int64_t bio_mulshr(uint64_t a, uint64_t b, int s) {
    const uint64_t al = a & 0xFFFFFFFFull, ah = a >> 32, bl = b & 0xFFFFFFFFull, bh = b >> 32;
    const uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
    const uint64_t mid = (ll >> 32) + (lh & 0xFFFFFFFFull) + (hl & 0xFFFFFFFFull);
    const uint64_t lo = (ll & 0xFFFFFFFFull) | (mid << 32);
    const uint64_t hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
    uint64_t r;
    if (s >= 64) { r = hi >> (s - 64); }
    else if (s == 0) { if (hi) return (int64_t)1 << 62; r = lo; }
    else { if (hi >> s) return (int64_t)1 << 62; r = (lo >> s) | (hi << (64 - s)); }
    return r > ((uint64_t)1 << 62) ? (int64_t)1 << 62 : (int64_t)r;
}

/* stochastic rounding of a q16 value: floor + (frac > h16). Unbiased. */
BIO_FN int64_t bio_sr16(int64_t v_q16, uint32_t h) {
    return (v_q16 >> 16) + (((uint32_t)v_q16 & 0xFFFFu) > (h & 0xFFFFu) ? 1 : 0);
}
/* stochastic rounding of v / 2^s for s in 1..31 */
BIO_FN int64_t bio_srs(int64_t v, int s, uint32_t h) {
    const uint32_t m = (1u << s) - 1u;
    return (v >> s) + (((uint32_t)v & m) > (h & m) ? 1 : 0);
}
/* truncating (toward zero) a*d >> 16; odd in a, so pairwise fluxes are antisymmetric */
BIO_FN int64_t bio_flux16(int64_t a, int32_t d) {
    const int64_t lim = (int64_t)1 << 46;
    a = bio_clamp64(a, -lim, lim);
    return a >= 0 ? (a * d) >> 16 : -((-a * d) >> 16);
}
/* Michaelis-Menten S/(S+K) in q16, S = pool (mmol), K = half saturation (mmol) */
BIO_FN int64_t bio_mm(int64_t pool, int64_t k) {
    if (pool <= 0) return 0;
    if (k <= 0) return 65536;
    const int64_t den = pool + k;
    if (pool < ((int64_t)1 << 46)) return (pool << 16) / den;
    return pool / bio_max64(1, den >> 16);
}
/* integer cube root (floor) */
BIO_FN uint32_t bio_icbrt(uint64_t v) {
    uint32_t r = 0;
    for (int b = 21; b >= 0; --b) {
        const uint64_t t = (uint64_t)(r | (1u << b));
        if (t * t * t <= v) r |= 1u << b;
    }
    return r;
}
/* q7 int8 lanes of a packed u64 */
BIO_FN int32_t bio_lane(uint64_t v, int k) { return (int32_t)(int8_t)(uint8_t)(v >> (8 * k)); }
BIO_FN uint64_t bio_set_lane(uint64_t v, int k, int32_t x) {
    const uint64_t m = (uint64_t)0xFF << (8 * k);
    return (v & ~m) | ((uint64_t)(uint8_t)(int8_t)x << (8 * k));
}
BIO_FN int32_t bio_ld32(const int8_t* p) {
    return (int32_t)((uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) | ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24));
}

/* ==== lookup tables, built by integer series (no libm) =========================== */
/* e^-x for x in q30, x in [0, 64): returns q30 */
BIO_FN int64_t bio_expneg_q30(int64_t x) {
    const int64_t one = (int64_t)1 << 30;
    if (x <= 0) return one;
    /* e^-1 by series */
    int64_t e1 = 0, t = one;
    for (int k = 1; k < 24; ++k) { e1 += t; t = -t / k; }
    int64_t n = x >> 30, f = x & (one - 1);
    int64_t ef = 0; t = one;
    for (int k = 1; k < 24; ++k) { ef += t; t = -((t * f) >> 30) / k; }
    int64_t r = ef;
    for (int64_t i = 0; i < n && r > 0; ++i) r = (r * e1) >> 30;
    return r;
}

/* literal tables */
#ifdef __CUDACC_RTC__
__device__ __constant__
#else
static const
#endif
int32_t BIO_SIN24_Q12[24] = { 0, 1060, 2048, 2896, 3547, 3956, 4096, 3956, 3547, 2896, 2048, 1060,
                               0, -1060, -2048, -2896, -3547, -3956, -4096, -3956, -3547, -2896, -2048, -1060 };

BIO_FN void bio_build_tables(BioTables* T) {
    const int64_t one = (int64_t)1 << 30;
    for (int i = 0; i < 1024; ++i) T->expn_q16[i] = (int32_t)((bio_expneg_q30(((int64_t)i << 30) / 64) + 8192) >> 14);
    for (int i = 0; i < 256; ++i) T->gauss_q16[i] = (int32_t)((bio_expneg_q30(((int64_t)i * i << 30) / 8192) + 8192) >> 14);
    const int64_t ln2_q30 = 744261118;
    for (int i = 0; i < 128; ++i) {
        const int32_t m = i - 64;                       /* 2^(m/10) */
        const int32_t c = m >= 0 ? (m + 9) / 10 : -((-m) / 10);
        const int64_t u = ((int64_t)(10 * c - m) << 30) / 10;     /* in [0, 1) */
        const int64_t f = bio_expneg_q30((u * ln2_q30) >> 30);    /* 2^-u, q30 */
        int64_t v = (f * 65536) >> 30;
        v = c >= 0 ? v << c : v >> (-c);
        T->q10_q16[i] = (int32_t)v;
    }
    for (int i = 0; i < 256; ++i) {
        const int32_t d = i - 128;
        const int64_t e = bio_expneg_q30(((int64_t)(d < 0 ? -d : d) << 30) / 32);
        T->sigmoid_q16[i] = (int32_t)(d >= 0 ? (65536 * one) / (one + e) : (65536 * e) / (one + e));
    }
    for (int i = 0; i < 256; ++i) {
        const int32_t d = i - 128;
        const int64_t e = bio_expneg_q30(((int64_t)(d < 0 ? -d : d) << 31) / 64);   /* e^-2|x| */
        int64_t t = ((one - e) * 127 + ((one + e) >> 1)) / (one + e);
        T->tanh_q7[i] = (int8_t)(d < 0 ? -t : t);
    }
    /* leaf projection k = G(theta) / cos(theta), Goudriaan: G = phi1 + phi2 cos(theta),
     * phi1 = 0.5 - 0.633 chi - 0.33 chi^2, phi2 = 0.877 (1 - 2 phi1); chi from -0.37
     * (erect leaves, class 0) to +0.57 (flat leaves, class 15); cos(theta) is the
     * sine of the sun's elevation at that hour */
    for (int la = 0; la < 16; ++la) {
        const int64_t chi = -26214 + (int64_t)(2 * la + 1) * 2048;            /* q16 */
        const int64_t phi1 = 32768 - (41484 * chi) / 65536 - (21627 * ((chi * chi) / 65536)) / 65536;
        const int64_t phi2 = (57475 * (65536 - 2 * phi1)) / 65536;
        for (int hr = 0; hr < 24; ++hr) {
            const int64_t c = BIO_SIN24_Q12[(hr + 18) % 24];                  /* q12, as bio_sin24 */
            int64_t k = 0;
            if (c > 0) {
                const int64_t g = phi1 + (phi2 * c) / 4096;                   /* q16 */
                k = g > 0 ? g * 256 / c : 0;                                  /* q12 */
                if (k > 32768) k = 32768;
            }
            T->kleaf_q12[la][hr] = (int32_t)k;
        }
    }
}

/* signed diurnal sine, zero at 06:00 and 18:00, peak at noon */
BIO_FN int32_t bio_sin24(int32_t hr) { return BIO_SIN24_Q12[(hr + 18) % 24]; }
/* semidiurnal tide, period 12.4 h: 0 low, 1 mid, 2 high */
BIO_FN int32_t bio_tide(uint64_t step) {
    const int32_t p = (int32_t)((step * 10u) % 124u);
    if (p >= 11 && p <= 51) return 2;
    if (p >= 73 && p <= 113) return 0;
    return 1;
}

/* ==== default parameters (tuned values marked) ================================== */
BIO_FN void bio_params_default(BioParams* p) {
    int32_t* raw = (int32_t*)p;
    for (unsigned i = 0; i < sizeof(BioParams) / 4; ++i) raw[i] = 0;
    p->vent_h2 = 40; p->vent_h2s = 20; p->vent_fe2 = 10; p->vent_dic = 20;     /* spec 2.5.5 */
    p->serp_h2_q8 = 2 * 256;                                                  /* spec */
    p->weath_fe_q16 = 33;   /* tuned: 0.5 mmol per f_ore=255 per day */
    p->weath_s_q16 = 20;    /* tuned */
    p->weath_p_q16 = 4;     /* tuned */
    p->weath_doc_q16 = 14;  /* tuned, times organic_ppm / 3800 at creation */
    p->rain_l = 4000;       /* tuned: litres per land cell per day at the coast */
    p->soil_cap_l = 20000;  /* tuned: 256 m^2 x 0.5 m x 15% */
    p->runoff_q16 = 16384;  /* tuned: a quarter of the excess per day */
    p->kp_water_q16 = 131;  /* 0.02 m/h over a 10 m layer = 0.002/h, tuned */
    p->kp_soil_q16 = 3277;  /* tuned: 0.05/h */
    /* Henry constants at ~304 K, mmol/m^3/atm; CO2 includes carbonate (DIC) */
    p->henry[0] = 80000;    /* H2S */
    p->henry[1] = 1300;     /* CH4 */
    p->henry[2] = 50000;    /* NH3 (capped below) */
    p->henry[3] = 400000;   /* CO2 as DIC, tuned: ~2 mM DIC at 0.5% */
    p->henry[4] = 1100;     /* O2 */
    p->henry[5] = 0;        /* N2 inert */
    p->henry[6] = 700;      /* H2 */
    p->aq_cap[0] = 2000;    /* tuned: H2S kept at 2 mM by pyrite + caps */
    p->aq_cap[1] = 1000; p->aq_cap[2] = 200; p->aq_cap[3] = 4000; p->aq_cap[4] = 400; p->aq_cap[5] = 0; p->aq_cap[6] = 100;
    p->k_py = 16;           /* tuned */
    p->k_ox_h2s = 64; p->k_ox_ch4 = 16; p->k_ox_fe = 256;    /* tuned */
    p->burial_shift = 9; p->mineral_shift = 6; p->burial_depth = 30;   /* spec */
    p->d_ww = 13107; p->d_ss = 655; p->d_wl = 1311;                    /* spec 0.2, 0.01, 0.02 */
    p->photo_ch4_q16 = 1; p->photo_h2s_q16 = 1;                        /* tuned */
    p->par_q12 = 1843;      /* 0.45 */
    p->kw_q16 = 3932;       /* 0.06 per m */
    p->sla_q20 = 252;       /* 2.4e-4 m^2 per mmol C */
    p->eff_anox_q16 = 141558; /* 3600 s x 0.03 / 50 J per eu = 2.16 */
    p->eff_oxy_q16 = 235930;  /* 3600 x 0.05 / 50 = 3.6 */
    p->ktox_h2s = 500;      /* tuned, mmol/m^3 */
    p->vmax_q16[BP_METH] = 32768; p->vmax_q16[BP_ANOX] = 32768; p->vmax_q16[BP_OXY] = 32768;
    p->vmax_q16[BP_FERM] = 19661; p->vmax_q16[BP_SULFRED] = 13107; p->vmax_q16[BP_AERO] = 19661;
    p->vmax_q16[BP_METHOX] = 13107; p->vmax_q16[BP_FEOX] = 65536;
    /* half saturation, mmol/m^3: DIC DOC NH4 PO4 FE2 H2S SO4 O2 CH4 H2 SOILW */
    p->km[BF_DIC] = 100; p->km[BF_DOC] = 50; p->km[BF_NH4] = 2; p->km[BF_PO4] = 1; p->km[BF_FE2] = 20;
    p->km[BF_H2S] = 20; p->km[BF_SO4] = 50; p->km[BF_O2] = 5; p->km[BF_CH4] = 20; p->km[BF_H2] = 5; p->km[BF_SOILW] = 10;
    p->yield_q16[BP_METH] = 65536;      /* +1 eu per CH4 */
    p->yield_q16[BP_ANOX] = 8 * 65536;  /* light energy per C-equivalent */
    p->yield_q16[BP_OXY] = 8 * 65536;
    p->yield_q16[BP_FERM] = 32768;      /* +0.5 eu per mmol C */
    p->yield_q16[BP_SULFRED] = 65536;   /* +1 */
    p->yield_q16[BP_AERO] = 5 * 65536;  /* +5 */
    p->yield_q16[BP_METHOX] = 3 * 65536;/* +3 */
    p->yield_q16[BP_FEOX] = 26214;      /* +0.1 eu per FE2 = 0.4 per unit (4 FE2) */
    p->mu_q16 = 3277;        /* tuned: 5% per step */
    p->fix_eu = 8; p->assim_eu = 2; p->cn = 6; p->cp = 106;
    p->water_q16 = 655;      /* tuned: 0.01 L per mmol C */
    p->maint_q16 = 131;      /* 0.002 */
    p->struct_q16 = 13;      /* 0.0002 */
    p->enz_q16 = 66;         /* 0.001 */
    p->trait_q16 = 33;       /* 0.0005 */
    p->brain_q16 = 3277;     /* 0.05 eu per active neuron per think */
    p->sense_q16 = 655;      /* 0.01 */
    p->learn_q16 = 131;      /* 0.002 */
    p->o2tox_q16 = 655; p->k_o2 = 20;  /* 0.01; K_O2 tuned */
    p->uv_q16 = 655;         /* 0.01 */
    p->dry_q8 = 2048;        /* tuned: W -= 8 per step at cuticle 0 (dry in ~32 h) */
    p->move_q16 = 1;         /* tuned: full speed (16 m/step) in water = 2x maintenance */
    p->move_land_q16 = 33;   /* tuned: + 4x maintenance at full speed on land (0.6 g) */
    p->starve_aero_q8 = 1024; p->starve_anaer_q8 = 128;   /* 4 and 0.5 eu per mmol C */
    p->copy_cost = 8;        /* 0.02 x 384 = 7.7 */
    p->hazard_q32 = 429497;  /* 1e-4 per h */
    p->kinpath_q32 = 357913; /* 0.002 per day */
    p->e_cap_mul = 64;
    p->drift_q8 = 512;       /* tuned: +-2 m per step */
    p->speed_q8 = 4096;      /* tuned: 16 m per step at full motor */
    p->barrier_q8 = 256;
    p->atm_scale_ppm = 100;  /* 1e-4 */
}
/* Parameters of the features (all tuned). A first-release checkpoint carries
 * zeros here; they are filled in whenever a run is created or loaded, so a
 * feature can be switched on later without anything else. They do nothing
 * while their feature is off. */
BIO_FN void bio_params_features(BioParams* p) {
    if (!p->pred_bite_q10) p->pred_bite_q10 = 256;   /* a bite out of a colony: C_att x MOUTH/255 / 4 per step */
    if (!p->pred_kill_q8) p->pred_kill_q8 = 179;     /* spec 3.8: 70% of a prey eaten whole reaches the gut */
    if (!p->gut_k_q8) p->gut_k_q8 = 32;              /* digestion half-saturates at gut = C / 8 */
    if (!p->explore_q7) p->explore_q7 = 40;          /* output noise up to +-0.31 at ETA 255 */
    if (!p->morph_pmax) p->morph_pmax = 720;         /* 200 W/m^2 of sunlit leaf x eff_oxy 3.6: ~25 umol C/m^2/s */
    if (!p->qwidth_cost_q8) p->qwidth_cost_q8 = 256; /* the widest Q band doubles the pigment cost */
    if (!p->pred_refuge1) p->pred_refuge1 = 2;       /* one hiding place per 16 m cell */
    if (!p->learn_shift) p->learn_shift = 21;        /* with ETA 64 the best step on the foraging test (T9) */
}

/* ==== environment ================================================================ */
BIO_FN int32_t bio_is_water(const BioWorld* w, int32_t c) { return w->depth[c] >= 1; }
/* active volume, m^3 */
BIO_FN int64_t bio_volume(const BioWorld* w, int32_t c) {
    const int32_t d = w->depth[c];
    return d >= 1 ? 256 * (int64_t)bio_clampi(d, 1, 10) : 128;
}
BIO_FN int32_t bio_wet_frac(const BioWorld* w, int32_t c, int32_t tide) {
    return tide == 0 ? w->wet0[c] : (tide == 1 ? w->wet1[c] : w->wet2[c]);
}
/* realm for statistics: 0 marine (always submerged), 1 coastal (tidal), 2 land (never wet) */
BIO_FN int32_t bio_realm(const BioWorld* w, int32_t c) {
    if (w->wet2[c] == 0) return 2;
    if (w->wet0[c] >= 255 && w->depth[c] >= 1) return 0;
    return 1;
}
/* temperature in centi-kelvin (spec 2.5.2) */
BIO_FN int32_t bio_temp(const BioWorld* w, int32_t c, int32_t day_sin_q12) {
    int32_t t = w->t_surf_ck;
    const int32_t above = w->elev[c] - w->sea_level;
    const int32_t d = w->depth[c];
    if (above > 0) t -= 65 * above / 10;
    t += (d >= 1 ? 50 : 600) * day_sin_q12 / 4096;
    if (d > 20) t -= bio_clampi(5 * (d - 20), 0, 700);
    if (w->vent[c]) t += 1500 * w->vent[c] / 255;
    return t;
}
BIO_FN int32_t bio_expn(const BioWorld* w, int64_t idx) {
    return idx >= 1023 ? w->T->expn_q16[1023] : (idx <= 0 ? 65536 : w->T->expn_q16[idx]);
}
/* 65536 x exp(-x) for x in q16, interpolated between the table's steps of 1/64
 * (the plain lookup above rounds x down, which is up to 12% of 1 - exp(-x) at
 * x = 1/8) */
BIO_FN int64_t bio_expn_q16(const BioWorld* w, int64_t x) {
    if (x <= 0) return 65536;
    const int64_t i = x >> 10;
    if (i >= 1022) return w->T->expn_q16[1023];
    const int64_t a = w->T->expn_q16[i], b = w->T->expn_q16[i + 1];
    return a - (((a - b) * (x & 1023)) >> 10);
}

/* diffusion coefficient between two cells */
BIO_FN int32_t bio_dcoef(const BioWorld* w, int32_t a, int32_t b) {
    const int32_t wa = bio_is_water(w, a), wb = bio_is_water(w, b);
    return wa && wb ? w->P.d_ww : (!wa && !wb ? w->P.d_ss : w->P.d_wl);
}
/* D8 neighbour offsets; flow_dir k points to (x + dx[k], z + dz[k]) */
BIO_FN int32_t bio_d8x(int32_t k) { k &= 7; return (k == 0 || k == 1 || k == 7) ? 1 : ((k >= 3 && k <= 5) ? -1 : 0); }
BIO_FN int32_t bio_d8z(int32_t k) { k &= 7; return (k >= 1 && k <= 3) ? 1 : (k >= 5 ? -1 : 0); }

/* daily runoff out of a land cell: excess soil water and dissolved solutes move
 * to the D8 downhill neighbour. Returns the amount of field f leaving cell c
 * (computed from the old values, so senders and receivers agree exactly). */
BIO_FN int64_t bio_runoff_out(const BioWorld* w, const int64_t* old, int32_t c, int32_t f) {
    if (bio_is_water(w, c) || w->flow_dir[c] > 7) return 0;
    {   /* a drain that points outside the grid is treated as a sink */
        const int32_t tx = c % w->w + bio_d8x(w->flow_dir[c]), tz = c / w->w + bio_d8z(w->flow_dir[c]);
        if (tx < 0 || tz < 0 || tx >= w->w || tz >= w->h) return 0;
    }
    const int64_t sw = old[(int64_t)BF_SOILW * w->n_cells + c];
    const int64_t ex = sw - w->P.soil_cap_l;
    if (ex <= 0 || sw <= 0) return 0;
    const int64_t ow = (ex * w->P.runoff_q16) >> 16;
    if (f == BF_SOILW) return ow;
    if (f != BF_DOC && f != BF_NH4 && f != BF_PO4) return 0;
    const int64_t frac = bio_mm(ow, sw - ow) ;   /* = ow / sw in q16 */
    return (old[(int64_t)f * w->n_cells + c] * frac) >> 16;
}

/* gas -> aqueous field (-1 = none) */
BIO_FN int32_t bio_gas_field(int32_t g) {
    return g == 0 ? BF_H2S : g == 1 ? BF_CH4 : g == 2 ? BF_NH4 : g == 3 ? BF_DIC : g == 4 ? BF_O2 : g == 6 ? BF_H2 : -1;
}

/* One cell of k_env: diffusion (antisymmetric truncating fluxes), daily runoff
 * and rain, then sources and sinks. Reads w->fld, writes w->fld_new for cell c.
 * Block-level sums of ledger quantities go to acc[] (BIO_ACC). Also clears
 * the cell's summary counters for this step. */
BIO_FN void bio_cell_env(const BioWorld* w, int32_t c, int64_t* acc) {
    const int32_t W = w->w, H = w->h, n = w->n_cells;
    const int32_t cx = c % W, cz = c / W;
    const int64_t* old = w->fld;
    int64_t v[BF_N];
    const int32_t nb[4] = { cx > 0 ? c - 1 : -1, cx < W - 1 ? c + 1 : -1, cz > 0 ? c - W : -1, cz < H - 1 ? c + W : -1 };
    int32_t dc[4];
    BIO_UNROLL
    for (int k = 0; k < 4; ++k) dc[k] = nb[k] >= 0 ? bio_dcoef(w, c, nb[k]) : 0;
    BIO_UNROLL
    for (int f = 0; f < BF_N; ++f) {
        const int64_t a = old[(int64_t)f * n + c];
        int64_t r = a;
        if (f < BIO_DIFFUSING) {
            BIO_UNROLL
            for (int k = 0; k < 4; ++k)
                if (nb[k] >= 0) r -= bio_flux16(a - old[(int64_t)f * n + nb[k]], dc[k]);
        }
        v[f] = r;
    }
    const int daily = (w->step % 24u) == 0;
    const int water = bio_is_water(w, c);
    const BioParams* P = &w->P;
    /* summaries for this step start from zero */
    w->cnt[c] = 0;
    for (int b = 0; b < BIO_LAI_BANDS; ++b) w->lai[(int64_t)b * n + c] = 0;
    for (int k = 0; k < BIO_KIN_BUCKETS / 2; ++k) w->kin[(int64_t)k * n + c] = 0;
    if (P->feat & BIO_FEAT_PRED) {
        for (int k = 0; k < BIO_PSLOTS; ++k) w->pslot[(int64_t)k * n + c] = 0u;
        w->thr[c] = 0u;
        w->nmot[c] = 0u;
    }
    if (P->feat & BIO_FEAT_SEX)
        for (int k = 0; k < BIO_MSLOTS; ++k) w->mslot[(int64_t)k * n + c] = 0u;

    if (daily) {
        /* runoff: gather from upslope neighbours that drain into c */
        BIO_UNROLL
        for (int f = 0; f < BF_N; ++f) {
            if (f != BF_SOILW && f != BF_DOC && f != BF_NH4 && f != BF_PO4) continue;
            v[f] -= bio_runoff_out(w, old, c, f);
            for (int k = 0; k < 8; ++k) {
                const int32_t x2 = cx - bio_d8x(k), z2 = cz - bio_d8z(k);
                if (x2 < 0 || z2 < 0 || x2 >= W || z2 >= H) continue;
                const int32_t j = z2 * W + x2;
                if (w->flow_dir[j] != k) continue;
                const int64_t o = bio_runoff_out(w, old, j, f);
                if (f == BF_SOILW && water) continue;     /* water reaching the sea leaves the soil budget */
                v[f] += o;
            }
        }
        if (!water && !P->closed_box) {
            const int64_t dist_m = (int64_t)w->coast[c] << w->shift;
            v[BF_SOILW] += ((int64_t)P->rain_l * bio_expn(w, dist_m * 64 / 3000)) >> 16;
        }
    }
    const int64_t V = bio_volume(w, c);
    const uint32_t hs = bio_hash(c, (int32_t)w->step, BH_COST, w->salt);
    if (!P->closed_box) {
        const int32_t vent = w->vent[c];
        if (vent) {
            const int64_t h2 = (int64_t)P->vent_h2 * vent / 255, h2s = (int64_t)P->vent_h2s * vent / 255;
            const int64_t fe = (int64_t)P->vent_fe2 * vent / 255, dic = (int64_t)P->vent_dic * vent / 255;
            v[BF_H2] += h2; v[BF_H2S] += h2s; v[BF_FE2] += fe; v[BF_DIC] += dic;
            BIO_ACC(&acc[BACC_VENT_C], dic); BIO_ACC(&acc[BACC_VENT_S], h2s); BIO_ACC(&acc[BACC_VENT_FE], fe);
        }
        const int32_t ore = w->f_ore[c], sulf = w->f_sulf[c], soil = w->f_soil[c], gran = w->f_gran[c];
        if (water && ore) v[BF_H2] += bio_srs((int64_t)P->serp_h2_q8 * ore / 255, 8, hs);
        /* weathering on land and on the seabed */
        const int64_t fe = bio_sr16((int64_t)P->weath_fe_q16 * ore / 255, hs >> 3);
        const int64_t s = bio_sr16((int64_t)P->weath_s_q16 * sulf / 255, hs >> 7);
        const int64_t po = bio_sr16((int64_t)P->weath_p_q16 * (gran + soil) / 255, hs >> 11);
        const int64_t doc = bio_sr16((int64_t)P->weath_doc_q16 * soil / 255, hs >> 15);
        v[BF_FE2] += fe; v[BF_H2S] += s; v[BF_PO4] += po; v[BF_DOC] += doc;
        BIO_ACC(&acc[BACC_WEATH_FE], fe); BIO_ACC(&acc[BACC_WEATH_S], s);
        BIO_ACC(&acc[BACC_WEATH_P], po); BIO_ACC(&acc[BACC_WEATH_C], doc);
        /* gas exchange with the island box (piston velocity) */
        const int32_t kp = water ? P->kp_water_q16 : P->kp_soil_q16;
        BIO_UNROLL
        for (int g = 0; g < BIO_ATMG; ++g) {
            const int32_t f = bio_gas_field(g);
            if (f < 0) continue;
            const int64_t sat = (w->G->sat_q16[g] * V) >> 16;
            int64_t fl = bio_flux16(sat - v[f], kp);
            if (fl > w->G->dis_cap[g]) fl = w->G->dis_cap[g];
            if (fl < -v[f]) fl = -v[f];
            v[f] += fl;
            BIO_ACC(&acc[BACC_ATMD + g], -fl);
            if (P->feat & BIO_FEAT_HUB) {      /* what the planet-wide atmosphere needs: uptake by realm */
                BIO_ACC(&acc[BACC_X(BACC2_XALL + g)], fl);
                if (!water) BIO_ACC(&acc[BACC_X(BACC2_XLAND + g)], fl);
            }
        }
    }
    /* pyrite: FE2 + 2 H2S -> FeS2 (sediment) */
    {
        const int64_t r = bio_mulshr((uint64_t)bio_mulshr((uint64_t)v[BF_H2S], (uint64_t)v[BF_FE2], 16), (uint64_t)P->k_py, 24) / V;
        const int64_t py = bio_min64(bio_min64(v[BF_H2S] / 2, v[BF_FE2]), r);
        if (py > 0) {
            v[BF_FE2] -= py; v[BF_H2S] -= 2 * py;
            BIO_ACC(&acc[BACC_SED_S], 2 * py); BIO_ACC(&acc[BACC_SED_FE], py);
        }
    }
    /* abiotic oxidation (second order in concentrations) */
    if (v[BF_O2] > 0) {
        int64_t r = bio_mulshr((uint64_t)v[BF_O2], (uint64_t)v[BF_H2S], 0) / V;
        r = bio_min64(bio_min64(v[BF_O2], v[BF_H2S]), (r * P->k_ox_h2s) >> 16);
        if (r > 0) { v[BF_O2] -= r; v[BF_H2S] -= r; v[BF_SO4] += r; }
        r = bio_mulshr((uint64_t)v[BF_O2], (uint64_t)v[BF_CH4], 0) / V;
        r = bio_min64(bio_min64(v[BF_O2] / 2, v[BF_CH4]), (r * P->k_ox_ch4) >> 16);
        if (r > 0) { v[BF_O2] -= 2 * r; v[BF_CH4] -= r; v[BF_DIC] += r; }
        r = bio_mulshr((uint64_t)v[BF_O2], (uint64_t)v[BF_FE2], 0) / V;
        r = bio_min64(bio_min64(v[BF_O2], v[BF_FE2] / 4), (r * P->k_ox_fe) >> 16);
        if (r > 0) { v[BF_O2] -= r; v[BF_FE2] -= 4 * r; BIO_ACC(&acc[BACC_SED_FE], 4 * r); }
    }
    if (daily) {
        if (!P->closed_box && water && w->depth[c] > P->burial_depth) {
            const int64_t b = v[BF_DET] >> P->burial_shift;
            v[BF_DET] -= b; BIO_ACC(&acc[BACC_SED_C], b);
        }
        const int64_t m = v[BF_DET] >> P->mineral_shift;
        v[BF_DET] -= m; v[BF_DOC] += m;
    }
    BIO_UNROLL
    for (int f = 0; f < BF_N; ++f) w->fld_new[(int64_t)f * n + c] = v[f];
}

/* Once per step, before k_env (single thread): apply the previous step's
 * exchange deltas to the box, photochemistry, and this step's light, UV,
 * haze, tide and saturation values. */
BIO_FN void bio_globals_update(const BioWorld* w) {
    BioGlobals* G = w->G;
    const BioParams* P = &w->P;
    for (int g = 0; g < BIO_ATMG; ++g) { G->atm[g] += G->acc[BACC_ATMD + g]; G->acc[BACC_ATMD + g] = 0; }
    /* photochemistry in the box: O2 + CH4 -> CO2, O2 + H2S -> sulfate (deposited) */
    if (G->atm[4] > 0) {
        int64_t r = bio_min64(bio_min64(G->atm[4] / 2, G->atm[1]), (bio_mulshr((uint64_t)G->atm[1], (uint64_t)P->photo_ch4_q16, 16)));
        if (r > 0) { G->atm[4] -= 2 * r; G->atm[1] -= r; G->atm[3] += r; }
        r = bio_min64(bio_min64(G->atm[4] / 2, G->atm[0]), (bio_mulshr((uint64_t)G->atm[0], (uint64_t)P->photo_h2s_q16, 16)));
        if (r > 0) { G->atm[4] -= 2 * r; G->atm[0] -= r; G->acc[BACC_SED_S] += r; }
    }
    int64_t tot = 0;
    for (int g = 0; g < BIO_ATMG; ++g) tot += G->atm[g] > 0 ? G->atm[g] : 0;
    int64_t xq[BIO_ATMG];      /* mole fractions, q30 */
    for (int g = 0; g < BIO_ATMG; ++g) {
        const int64_t a = G->atm[g] > 0 ? G->atm[g] : 0;
        if (tot <= 0) xq[g] = 0;
        else if (tot < ((int64_t)1 << 32)) xq[g] = (a << 30) / tot;
        else xq[g] = a / bio_max64(1, tot >> 30);
    }
    const int64_t k_uv = 1073742;   /* 0.001 in q30 */
    G->uv_q8 = (int32_t)(255 * k_uv / (xq[4] + k_uv));
    const int64_t k_hz = 214748365; /* 0.2 in q30 */
    G->haze_q12 = (int32_t)(4096 - 2048 * xq[1] / (xq[1] + k_hz));
    G->hr = (int32_t)(w->step % 24u);
    G->day_sin_q12 = bio_sin24(G->hr);
    const int64_t s = G->day_sin_q12 > 0 ? G->day_sin_q12 : 0;
    G->i0 = P->light_off ? 0 : (int32_t)(((int64_t)w->s_top * P->par_q12 * G->haze_q12 / 4096 * s) >> 24);
    if (P->light_mode == BIO_LIGHT_SPECTRAL) {
        /* the planet's light in 9 bands at this hour and haze (GMESPEC1 tables: the
         * renderer's atmosphere). i0 becomes the 380-740 nm total: the 45% PAR
         * share of the scalar model is gone, and the haze now colours the light. */
        int32_t d = 4096 - G->haze_q12;
        int32_t k = d >> 8, f = d & 255;
        if (d < 0) { k = 0; f = 0; }
        if (k > BIO_NHAZE - 2) { k = BIO_NHAZE - 2; f = 255; }
        int64_t vis = 0;
        for (int b = 0; b < BIO_NB; ++b) {
            const int64_t a = w->SP->surf_q16[k][G->hr][b], c = w->SP->surf_q16[k + 1][G->hr][b];
            const int64_t share = a + (((c - a) * f) >> 8);                 /* of S_top, q16 */
            const int64_t ib = (P->light_off || share <= 0) ? 0 : ((int64_t)w->s_top * share) >> 8;
            G->i0b_q8[b] = (int32_t)ib;
            if (b < 8) vis += ib;
        }
        G->i0 = (int32_t)(vis >> 8);
    }
    G->tide = bio_tide(w->step);
    G->think_n = 0u;
    /* capacity guard as a fair lottery: when last step's eligible parents would
     * overfill 0.95 cap, each parent reproduces with probability room / eligible */
    {
        const int64_t room = (int64_t)w->cap_agents * 95 / 100 - (int64_t)w->n_live;
        const int64_t want = G->acc[BACC_WANT];
        G->birth_q16 = room <= 0 ? 0 : (want <= room ? 65536 : (int32_t)(room * 65536 / want));
        G->acc[BACC_WANT] = 0;
    }
    for (int g = 0; g < BIO_ATMG; ++g) {
        int64_t sq = ((int64_t)P->henry[g] * xq[g]) >> 14;     /* q16 mmol/m^3 */
        const int64_t capq = (int64_t)P->aq_cap[g] << 16;
        G->sat_q16[g] = sq < capq ? sq : capq;
        G->dis_cap[g] = G->atm[g] > 0 ? G->atm[g] / w->n_cells : 0;
    }
}

/* ==== agents: genes and derived quantities ======================================= */
/* reproduction mass m_r = 2^(g/10) mmol C */
BIO_FN int32_t bio_mr(const BioWorld* w, int32_t g) {
    return (int32_t)(((int64_t)w->T->q10_q16[64 + g % 10] << (g / 10)) >> 16);
}
BIO_FN int32_t bio_repro_e(const BioWorld* w, const uint8_t* gb) {
    return (int32_t)((int64_t)gb[BG_REPRO_E] * bio_mr(w, gb[BG_SIZE]) / 64);
}
BIO_FN int32_t bio_topt(const uint8_t* gb) { return 27300 + 25 * gb[BG_TOPT]; }
BIO_FN int32_t bio_ttol(const uint8_t* gb) { return 100 + 25 * gb[BG_TTOL] / 4; }
BIO_FN int64_t bio_lifespan_h(const uint8_t* gb) {
    const int32_t g = gb[BG_LIFESPAN];
    return 24 * ((((int64_t)16 + g % 16) << (g / 16)) >> 4);
}
BIO_FN int32_t bio_period(const uint8_t* gb) { return 1 + (gb[BG_THINK] >> 2); }
/* height of the structure in 1/16 m: H = cbrt(S / 4000) m, capped at 60 m */
BIO_FN int32_t bio_height_q4(int32_t S) {
    if (S <= 0) return 0;
    const int32_t h = (int32_t)bio_icbrt((uint64_t)S * 4096u / 4000u);
    return h > 960 ? 960 : h;
}
BIO_FN int32_t bio_band(int32_t hq4) {
    int32_t b = 0;
    while (b < 7 && hq4 >= (8 << b)) ++b;
    return b;
}
/* leaf area, m^2 q20 */
BIO_FN int64_t bio_leaf_q20(const BioWorld* w, int32_t C, int32_t pig) {
    return C > 0 ? (int64_t)C * pig * w->P.sla_q20 / 255 : 0;
}
/* cached traits from a genome */
BIO_FN uint32_t bio_traits(const BioGenome* g) {
    const uint8_t* b = g->body;
    uint32_t q1 = 0;
    for (int k = 0; k < 8; ++k) q1 |= (uint32_t)(b[k] >> 5) << (3 * k);
    const uint32_t q2 = (uint32_t)(b[BG_HOLDFAST] >> 6) | ((uint32_t)(b[BG_PIGMENT] >> 6) << 2) |
                        ((uint32_t)(b[BG_MOTOR] >> 6) << 4) | ((uint32_t)(b[BG_BUOY] >> 6) << 6);
    uint32_t t = bio_hash((int32_t)q1, (int32_t)q2, 0, 0x5EEDB10Cu) >> 28;
    t |= (uint32_t)b[BG_PIGMENT] << 4;
    if (b[BG_HOLDFAST] >= 128) t |= BT_HOLD;
    if (b[BG_MOTOR] > 0) t |= BT_MOTILE;
    if ((b[BG_E_ANOX] > 0 || b[BG_E_OXY] > 0) && b[BG_PIGMENT] > 0) t |= BT_PHOTO;
    t |= (uint32_t)(b[BG_THINK] >> 2) << 16;          /* period - 1 */
    return t;
}
/* second cached trait word (BT2_*): what other agents' threads and the per-cell
 * summaries need without reading the genome */
BIO_FN uint32_t bio_traits2(const BioGenome* g) {
    const uint8_t* b = g->body;
    uint32_t t = (uint32_t)b[BG_MOTOR] | ((uint32_t)b[BG_MOUTH] << 8);
    t |= ((uint32_t)b[BG_SIZE] / 10u) << 16;
    t |= (((uint32_t)b[BG_CROWN] * 5u) >> 8) << 21;
    t |= ((((uint32_t)b[BG_ASPECT] >> 5) + 2u) & 7u) << 24;      /* gene 0 = height : width 1 (BIOMORPH_ASPECT_OF, biomap.h) */
    t |= ((((uint32_t)b[BG_LEAFANG] >> 4) + 8u) & 15u) << 27;    /* gene 0 = the mid leaf angle (BIOMORPH_LEAFCLASS_OF) */
    if (b[BG_SEX] >= 128) t |= BT2_SEXUAL;
    return t;
}
/* pigment hue (0..255) -> hue bin 1..15 (the biomap palette index and the
 * pigment shape of the spectral tables: Q band at 383 + 34 bin nm) */
BIO_FN int32_t bio_hue_bin(int32_t hue) { return 1 + hue * 15 / 256; }
/* A mover, for the space rule and the refuge of predation: a real motor
 * (BIO_CREATURE_MOTOR) and no photosynthesis. A phototroph needs its place in
 * the light whatever its motor gene, and an unused motor gene costs nothing,
 * so it must not buy a plant room: evolved forests carry such genes.
 * tr: the cached traits (BT_*), motor: the gene. */
BIO_FN int bio_mover(uint32_t tr, int32_t motor) {
    return motor >= BIO_CREATURE_MOTOR && !(tr & BT_PHOTO);
}
/* A creature, for the statistics, the land-animal gate and the engine: an
 * agent with a motor that moves visibly at full effort (BIO_CREATURE_MOTOR) or
 * lives mostly on prey. What the vegetation export draws, a phototroph with a
 * holdfast, is a plant whatever its motor gene (evolved forests carry motor
 * genes that drifted in), unless it does both: walks and lives on prey.
 * eco: the agent's ecology word (0 while animals are off). */
BIO_FN int bio_creature(const uint8_t* gb, uint32_t eco) {
    if (gb[BG_MOTOR] == 0) return 0;
    const int moves = gb[BG_MOTOR] >= BIO_CREATURE_MOTOR, eats = BE_GUTSHARE(eco) > 128u;
    if (gb[BG_HOLDFAST] > 64 && (gb[BG_E_ANOX] >= 32 || gb[BG_E_OXY] >= 32)) return moves && eats;
    return moves || eats;
}

/* ---- crown morphology (BIO_FEAT_MORPH; docs/biosphere-spec.md 3.9) -------------
 * Per crown class: top area relative to a disc of the crown's width, clumping
 * of the foliage (lower = light gets deeper into the crown), structure cost.
 * 0 ellipsoid, 1 cone, 2 umbrella, 3 column, 4 dome. All tuned. */
BIO_FN int32_t bio_crown_area_q8(uint32_t cc)  { return cc == 1u ? 154 : (cc == 2u ? 410 : (cc == 3u ? 102 : (cc == 4u ? 307 : 256))); }
BIO_FN int32_t bio_crown_clump_q8(uint32_t cc) { return cc == 1u ? 154 : (cc == 2u ? 230 : (cc == 3u ? 179 : (cc == 4u ? 256 : 205))); }
BIO_FN int32_t bio_crown_cost_q8(uint32_t cc)  { return cc == 1u ? 218 : (cc == 2u ? 346 : (cc == 3u ? 192 : (cc == 4u ? 282 : 256))); }
/* height : width of an aspect bin, q8: 0.5, 0.75, 1, 1.4, 1.8, 2.2, 2.6, 3 (the same as BIOMORPH_ASPECT_Q8 of biomap.h) */
BIO_FN int32_t bio_aspect_q8(uint32_t bin) { return bin == 0u ? 128 : (bin == 1u ? 192 : 256 + (int32_t)(((bin - 2u) * 512u) / 5u)); }

/* plant height in 1/16 m. Without the morphology genes H = cbrt(S / 4000) m;
 * with them the same structure builds a taller, narrower or a lower, wider
 * plant: H = cbrt(S / 4000 x aspect^2). Aspect gene 0 is 1 : 1, the old height. */
BIO_FN int32_t bio_plant_hq4(const BioWorld* w, int32_t S, uint32_t t2) {
    if (!(w->P.feat & BIO_FEAT_MORPH)) return bio_height_q4(S);
    if (S <= 0) return 0;
    const uint64_t a = (uint64_t)bio_aspect_q8(BT2_ASPECT(t2));
    const int32_t h = (int32_t)bio_icbrt((((uint64_t)S * 4096u / 4000u) * a * a) >> 16);
    return h > 960 ? 960 : h;
}

/* Light interception of a land plant's crown at this hour. In: leaf area aq
 * (m^2 q20), height. The world's light is horizontal flux per unit of leaf
 * area, as if every leaf lay flat: that stays the unit. The crown's top area A
 * is a disc of its width (height / aspect) times the class factor. Its leaves
 * have a projection kappa = k x gain: k from leaf angle, sun elevation,
 * clumping and porosity, and the gain (morph_gain_q8) scales it so that the
 * reference foliage (mid leaf angle, ellipsoid) has a light-weighted mean of 1
 * over the day. With crown leaf area index Lc = leaf area / A the crown
 * intercepts the light falling on A x (1 - exp(-kappa Lc)): kappa x leaf area
 * for a thin crown (what its leaves caught before, redistributed over the
 * day), never more than A however much leaf it holds. Out: that area (m^2
 * q20), and the leaf area the intercepted light is spread over (m^2 q12: the
 * whole leaf of a thin crown, A / kappa of a full one) for the saturation. */
BIO_FN int64_t bio_crown(const BioWorld* w, int64_t aq, int32_t hq4, uint32_t t2, int32_t porosity, int64_t* asun_q12) {
    const uint32_t cc = BT2_CROWN(t2);
    const int64_t wd = (int64_t)hq4 * 256 / bio_aspect_q8(BT2_ASPECT(t2));     /* crown width, 1/16 m */
    int64_t A = (((wd * wd * 201) >> 8) * bio_crown_area_q8(cc)) >> 8;         /* top area, m^2 q8 (pi/4 = 201/256) */
    if (A < 16) A = 16;
    const int64_t la = aq >> 8;                                                /* leaf area, m^2 q12 */
    *asun_q12 = 0;
    if (la <= 0) return 0;
    int64_t k = ((int64_t)w->T->kleaf_q12[BT2_LEAFANG(t2)][w->G->hr] * bio_crown_clump_q8(cc)) >> 8;   /* q12 */
    k = (k * (256 - 43 * (int64_t)(porosity >> 6))) >> 8;        /* an open crown lets up to half the light through */
    if (k < 64) k = 64;
    k = (k * w->P.morph_gain_q8) >> 8;                           /* kappa, q12 */
    if (k < 16) k = 16;
    const int64_t x = (k * la) / A;                                            /* kappa Lc, q16 */
    int64_t aabs;                                                              /* m^2 q12 */
    if (x < 8192) aabs = (k * la * (131072 - x)) >> 29;                        /* thin crown: kappa La (1 - x / 2) */
    else aabs = ((A << 4) * (65536 - bio_expn_q16(w, x))) >> 16;
    *asun_q12 = aabs * 4096 / k;
    return aabs << 8;
}
/* Light saturation of leaves: a harvest L (eu per step, q16) spread over leaf
 * that is good for at most cap (eu per step, q16: its area x morph_pmax) gives
 * cap x (1 - exp(-L / cap)); L itself in weak light. */
BIO_FN int64_t bio_saturate(const BioWorld* w, int64_t L, int64_t cap) {
    if (L <= 0 || cap <= 0) return 0;
    if (L >> 46) return cap;                                                   /* far beyond any harvest */
    const int64_t x = (L << 16) / cap;                                         /* L / cap, q16 */
    if (x < 8192) return L - ((L * x) >> 17);                                  /* weak light: L (1 - x / 2) */
    return bio_mulshr((uint64_t)cap, (uint64_t)(65536 - bio_expn_q16(w, x)), 16);
}

BIO_FN int32_t bio_cell_at(const BioWorld* w, int32_t x, int32_t z) {
    const int32_t cx = bio_clampi(x >> (8 + w->shift), 0, w->w - 1);
    const int32_t cz = bio_clampi(z >> (8 + w->shift), 0, w->h - 1);
    return cz * w->w + cx;
}
/* reflecting edge, v in metres q8, lim = extent in metres q8 */
BIO_FN int32_t bio_reflect(int32_t v, int32_t lim) {
    if (v < 0) v = -v - 1;
    if (v >= lim) v = 2 * lim - 1 - v;
    return bio_clampi(v, 0, lim - 1);
}
/* output of a newborn (no thinking yet): the output biases */
BIO_FN uint64_t bio_out_bias(const BioWorld* w, const BioGenome* g) {
    uint64_t o = 0;
    for (int k = 0; k < BIO_OUT; ++k) {
        const int32_t v = bio_clampi((64 * g->b_o[k]) >> 7, -128, 127);
        o = bio_set_lane(o, k, w->T->tanh_q7[v + 128]);
    }
    return o;
}
/* agent depth below the surface, metres q8 (0 for dry agents) */
BIO_FN int32_t bio_zq8(const BioWorld* w, int32_t c, uint32_t flags, const uint8_t* gb, uint64_t out) {
    if (!(flags & BA_WET)) return 0;
    const int32_t d = w->depth[c];
    if (d <= 0) return 0;
    const int32_t b = bio_clampi(gb[BG_BUOY] + bio_lane(out, 5) / 2, 0, 255);
    return d * 256 * (255 - b) / 255;
}
/* PAR per unit leaf area reaching the agent, W/m^2. Light above the agent's
 * layer is attenuated by water (k_w z, spec 2.5.1) or by the canopy above
 * (exp(-0.5 LAI_above), land bands); inside its own layer (all water
 * phototrophs of the cell, or the land height band) the leaves share what
 * arrives: absorption per unit leaf area (1 - exp(-L)) / L, so a cell never
 * absorbs more light than falls on it. */
/* the two grey factors of that rule for an agent of height hq4: *idx, the
 * attenuation above its layer as an index into expn_q16, and *g (q16), the
 * share per unit leaf area inside the layer */
BIO_FN void bio_shade(const BioWorld* w, int32_t c, uint32_t flags, int32_t zq8, int32_t hq4, int64_t* idx, int64_t* g) {
    const int64_t n = w->n_cells;
    int64_t layer = 0;
    if (flags & BA_WET) {
        for (int b = 0; b < BIO_LAI_BANDS; ++b) layer += w->lai[b * n + c];
        *idx = ((int64_t)w->P.kw_q16 * zq8) >> 18;
    } else {
        const int32_t b = bio_band(hq4);
        int64_t above = 0;
        for (int k = b + 1; k < BIO_LAI_BANDS; ++k) above += w->lai[k * n + c];
        layer = w->lai[b * n + c];
        /* 0.5 LAI of the leaf above, in 1/64 units. With shapes on, the bands above
         * hold what the crowns there intercept, and exactly that is gone below them. */
        *idx = (w->P.feat & BIO_FEAT_MORPH) ? above / 4096 : above / 8192;
    }
    const int64_t li = layer / 4096;              /* LAI of the layer in 1/64 units */
    *g = 65536;
    if (li > 0) *g = li >= 1023 ? ((int64_t)65536 << 6) / li : ((65536 - (int64_t)w->T->expn_q16[li]) << 6) / li;
}
BIO_FN int32_t bio_light_h(const BioWorld* w, int32_t c, uint32_t flags, int32_t zq8, int32_t hq4) {
    const int32_t i0 = w->G->i0;
    if (i0 <= 0) return 0;
    int64_t idx, g;
    bio_shade(w, c, flags, zq8, hq4, &idx, &g);
    return (int32_t)(((((int64_t)i0 * bio_expn(w, idx)) >> 16) * g) >> 16);
}
BIO_FN int32_t bio_light(const BioWorld* w, int32_t c, uint32_t flags, int32_t zq8, int32_t S) {
    return bio_light_h(w, c, flags, zq8, bio_height_q4(S));
}

/* ---- spectral light (BIO_LIGHT_SPECTRAL; docs/render-upgrade-spec.md 5.2) -------
 * The same rule per band: band irradiance at the surface (BioGlobals.i0b_q8)
 * times the water above the agent (the band's transmittance table, then the
 * band's attenuation below 32 m) or the grey canopy above it, times the
 * layer's sharing factor. out: W/m^2 q8 per band. */
BIO_FN void bio_light_bands(const BioWorld* w, uint32_t flags, int32_t zq8, int64_t idx, int64_t g, int32_t* out) {
    const BioSpectral* sp = w->SP;
    const int32_t zi = zq8 >> 8, fr = zq8 & 255;
    const int64_t above = bio_expn(w, idx);
    for (int b = 0; b < BIO_NB; ++b) {
        int64_t t = above;
        if (flags & BA_WET) {
            if (zi >= BIO_NDEPTH - 1)
                t = ((int64_t)sp->water_q16[b][BIO_NDEPTH - 1] *
                     bio_expn(w, ((int64_t)sp->kw_q16[b] * (zq8 - ((BIO_NDEPTH - 1) << 8))) >> 18)) >> 16;
            else {
                const int64_t a = sp->water_q16[b][zi], a2 = sp->water_q16[b][zi + 1];
                t = a + (((a2 - a) * fr) >> 8);
            }
        }
        out[b] = (int32_t)(((((int64_t)w->G->i0b_q8[b] * t) >> 16) * g) >> 16);
    }
}
/* photons a pigment harvests per m^2 of leaf, umol/s q8: absorptance per band
 * 1 - 10^-OD (OD = PIG/255 x 4 x epsilon of the hue's pigment shape; a wider Q
 * band adds part of the neighbouring hues' shapes), photons per joule of the
 * band, and the share a reaction centre can use: q[0] oxygenic (680 nm),
 * q[1] anoxygenic (870 nm, so the near infrared counts). */
BIO_FN void bio_harvest(const BioWorld* w, const int32_t* ib_q8, int32_t pig, int32_t hue, int32_t qw, int64_t* q) {
    const BioSpectral* sp = w->SP;
    const int32_t h = bio_hue_bin(hue);
    q[0] = 0; q[1] = 0;
    for (int b = 0; b < BIO_NB; ++b) {
        if (ib_q8[b] <= 0) continue;
        int64_t e = sp->eps_q12[h][b];
        if (qw) e += (qw * ((int64_t)sp->eps_q12[h > 1 ? h - 1 : h][b] + sp->eps_q12[h < BIO_NHUE - 1 ? h + 1 : h][b])) >> 5;
        const int64_t a = 65536 - (int64_t)bio_expn(w, ((int64_t)pig * e * 37) >> 16);
        const int64_t ph = ((((int64_t)ib_q8[b] * a) >> 16) * sp->phot_q16[b]) >> 16;
        q[0] += (ph * sp->uphill_q16[b][0]) >> 16;
        q[1] += (ph * sp->uphill_q16[b][1]) >> 16;
    }
}

/* Tie-break of a slot key among agents of equal hash priority: their array
 * index, or its complement, by a hash bit of the cell and step. So the first
 * or the last of the tied agents wins, never one picked by the bit pattern of
 * its absolute position: only the relative order of the live agents counts,
 * which a checkpoint keeps while the holes of a running island's arrays are
 * gone after a load (T4b). Ties are rare (about n / 127 for n offers). */
BIO_FN uint32_t bio_slot_mask(const BioWorld* w, int32_t c, int32_t purpose) {
    return (bio_hash(c, (int32_t)w->step, purpose, w->salt) & 1u) ? 0x1FFFFFFu : 0u;
}

/* ==== k_summary: per-cell counts, leaf area per height band, kin buckets ========= */
BIO_FN void bio_agent_summary(const BioWorld* w, int32_t i) {
    const uint32_t fl = w->flags[i];
    if (!(fl & BA_ALIVE)) return;
    const int32_t c = (int32_t)w->cell[i];
    const int64_t n = w->n_cells;
    const uint32_t t = w->tr[i];
    BIO_ADD32(&w->cnt[c], 1u);
    const uint32_t kb = BT_KIN(t);
    BIO_ADD32(&w->kin[(kb >> 1) * n + c], 1u << (16 * (kb & 1u)));
    const uint32_t feat = (uint32_t)w->P.feat;
    if ((t & BT_PHOTO) && w->C[i] > 0) {
        /* The cell's leaf area per height band is what the plants shade each other
         * with, so it must be the area each one harvests with (bio_agent_pre), or a
         * layer could absorb more light than falls on it: the whole leaf with light
         * in bands (the pigment acts through its absorptance there), and for a
         * standing plant with a crown what the crown intercepts this hour. A small
         * dense crown shades little ground, however much leaf it holds. */
        const int spec = w->P.light_mode == BIO_LIGHT_SPECTRAL, morph = (feat & BIO_FEAT_MORPH) != 0;
        const int32_t hq4 = (fl & BA_LAND) ? bio_plant_hq4(w, w->S[i], morph ? w->tr2[i] : 0u) : 0;
        int64_t aq = bio_leaf_q20(w, w->C[i], spec ? 255 : (int32_t)BT_PIG(t));
        /* By night nothing is shared or shaded, and the band holds the plain leaf
         * area: the statistics (vegetated cells, gate G3) are taken at the end of
         * the day and keep their meaning. */
        if (morph && (fl & BA_LAND) && hq4 >= 8 && aq > 0 && w->G->i0 > 0) {
            int64_t asun;
            aq = bio_crown(w, aq, hq4, w->tr2[i], w->gen[w->gslot[i]].body[BG_POROSITY], &asun);
        }
        int64_t a = aq >> 10;
        if (a > ((int64_t)1 << 25)) a = (int64_t)1 << 25;
        if (a > 0) BIO_ADD32(&w->lai[(int64_t)((fl & BA_LAND) ? bio_band(hq4) : 0) * n + c], (uint32_t)a);
    }
    if (!(feat & (BIO_FEAT_PRED | BIO_FEAT_SEX))) return;
    const uint32_t t2 = w->tr2[i];
    const uint32_t uid = w->uid[i];
    if (feat & BIO_FEAT_PRED) {
        /* an attacker (mouth, motor, attack output > 0, gut less than half full: a
         * full animal does not hunt) offers itself in one of the cell's BIO_PSLOTS
         * slots; the highest key of a slot stands this step. Key: a 7-bit hash
         * priority, then the agent index or its complement (bio_slot_mask), so
         * neither end of the array order is favoured. */
        uint32_t nf = fl & ~BA_HUNT;
        if (bio_mover(t, (int32_t)BT2_MOTOR(t2))) BIO_ADD32(&w->nmot[c], 1u);
        if (BT2_MOUTH(t2) && BT2_MOTOR(t2) && w->C[i] > 0 && bio_lane(w->out[i], 2) > 0 && 2 * (int64_t)w->gut[i] <= w->C[i]) {
            const uint32_t h = bio_hash((int32_t)uid, (int32_t)w->step, BH_PSLOT, w->salt);
            const uint32_t key = ((1u + (h >> 8) % 127u) << 25) |
                                 (((uint32_t)i ^ bio_slot_mask(w, c, BH_PMASK)) & 0x1FFFFFFu);
            BIO_MAX32(&w->pslot[(int64_t)(h & (BIO_PSLOTS - 1)) * n + c], key);
            BIO_MAX32(&w->thr[c], (uint32_t)bio_min64((int64_t)BT2_GAPE(t2), 0x7FFFFFFF));
            w->claim[i] = ~(uint64_t)0;
            nf |= BA_HUNT;
        }
        if (nf != fl) w->flags[i] = nf;
    }
    if ((feat & BIO_FEAT_SEX) && (t2 & BT2_SEXUAL)) {   /* a sexual agent offers itself as a mate */
        const uint32_t h = bio_hash((int32_t)uid, (int32_t)w->step, BH_MSLOT, w->salt);
        const uint32_t key = ((1u + (h >> 8) % 127u) << 25) |
                             (((uint32_t)i ^ bio_slot_mask(w, c, BH_MMASK)) & 0x1FFFFFFu);
        BIO_MAX32(&w->mslot[(int64_t)(h & (BIO_MSLOTS - 1)) * n + c], key);
    }
}
/* the agent index behind a slot key of cell c (purpose BH_PMASK or BH_MMASK) */
BIO_FN int32_t bio_slot_agent(const BioWorld* w, uint32_t key, int32_t c, int32_t purpose) {
    return (int32_t)((key ^ bio_slot_mask(w, c, purpose)) & 0x1FFFFFFu);
}
BIO_FN uint32_t bio_kin_count(const BioWorld* w, int32_t c, uint32_t kb) {
    return (w->kin[(int64_t)(kb >> 1) * w->n_cells + c] >> (16 * (kb & 1u))) & 0xFFFFu;
}

/* ==== brain (spec 3.4) =============================================================== */
BIO_FN int32_t bio_q7(int64_t v) { return (int32_t)bio_clamp64(v, -127, 127); }
BIO_FN int32_t bio_sense_mm(int64_t pool, int32_t km, int64_t V) {
    return (int32_t)((bio_mm(pool, (int64_t)km * V) * 127) >> 16);
}
/* field of the agent's dominant substrate (-1: light / none) */
BIO_FN int32_t bio_best_field(const uint8_t* gb) {
    int32_t best = 0, bk = -1;
    for (int k = 0; k < 8; ++k) if (gb[k] > best) { best = gb[k]; bk = k; }
    if (bk < 0 || bk == BP_OXY) return -1;
    if (bk == BP_METH) return BF_H2;
    if (bk == BP_ANOX) return BF_H2S;
    if (bk == BP_METHOX) return BF_CH4;
    if (bk == BP_FEOX) return BF_FE2;
    return BF_DOC;
}
BIO_FN int32_t bio_grad(const BioWorld* w, int32_t c, int32_t f, int32_t dx, int32_t dz, int32_t km) {
    const int32_t cx = c % w->w, cz = c / w->w;
    const int32_t a = (dx ? bio_clampi(cx - 1, 0, w->w - 1) : cx) + (dz ? bio_clampi(cz - 1, 0, w->h - 1) : cz) * w->w;
    const int32_t b = (dx ? bio_clampi(cx + 1, 0, w->w - 1) : cx) + (dz ? bio_clampi(cz + 1, 0, w->h - 1) : cz) * w->w;
    if (f < 0) return bio_q7(((int64_t)w->depth[a] - w->depth[b]) * 16);   /* toward shallower water */
    /* relative gradient (Weber's law): (b - a) / (a + b), q7 */
    const int64_t n = w->n_cells;
    (void)km;
    const int64_t pa = bio_max64(0, w->fld[f * n + a]), pb = bio_max64(0, w->fld[f * n + b]);
    return bio_q7((pb - pa) * 127 / (pa + pb + 1));
}

/* relative gradient across the two neighbours in x or z of a per-cell summary of
 * this step: threat = the gape of the largest attacker where it fits an
 * individual of this size, else the number of agents (BIO_FEAT_PRED) */
BIO_FN int32_t bio_grad_sum(const BioWorld* w, int32_t c, int32_t dx, int32_t dz, int64_t body, int threat) {
    const int32_t cx = c % w->w, cz = c / w->w;
    const int32_t a = (dx ? bio_clampi(cx - 1, 0, w->w - 1) : cx) + (dz ? bio_clampi(cz - 1, 0, w->h - 1) : cz) * w->w;
    const int32_t b = (dx ? bio_clampi(cx + 1, 0, w->w - 1) : cx) + (dz ? bio_clampi(cz + 1, 0, w->h - 1) : cz) * w->w;
    int64_t pa, pb;
    if (threat) {
        pa = (int64_t)w->thr[a] >= body ? (int64_t)w->thr[a] : 0;
        pb = (int64_t)w->thr[b] >= body ? (int64_t)w->thr[b] : 0;
    } else { pa = w->cnt[a]; pb = w->cnt[b]; }
    return bio_q7((pb - pa) * 127 / (pa + pb + 1));
}

/* think: learning from the last action, then a new forward pass. Returns eu cost. */
BIO_FN int32_t bio_think(const BioWorld* w, int32_t i, const BioGenome* g, int32_t c, int32_t light, int32_t T) {
    const uint8_t* gb = g->body;
    const BioParams* P = &w->P;
    const int64_t n = w->n_cells;
    const uint32_t uid = w->uid[i];
    const uint64_t hid = w->hid[i], out = w->out[i];
    const int32_t E = w->E[i], C = w->C[i];
    /* neuromodulated Hebbian update on the previous action (Baldwinian).
     * BIO_FEAT_LEARN: learning by exploration instead. The plain rule reinforces
     * whatever the agent did whenever the reward beat its average, which cannot
     * tell a good action from a bad one. Here each plastic output is nudged by a
     * small random amount at every think (below), and the weights follow reward
     * x that nudge x the hidden activity: a nudge that paid is made more likely
     * in that situation, one that cost less likely (node perturbation). */
    const int learn2 = (P->feat & BIO_FEAT_LEARN) != 0;
    const uint64_t nudge = learn2 ? w->expl[i] : 0;
    const int32_t eta = P->learn_off == 1 ? 0 : gb[BG_ETA];
    const uint32_t plastic = gb[BG_PLASTIC];
    const int32_t r = E - w->elast[i];
    int32_t rb = w->rbase[i];
    const int32_t m = bio_q7(((int64_t)r - rb) * 128 / (1 + C / 64));
    rb += (r - rb) >> 3;
    w->rbase[i] = rb;
    w->elast[i] = E;
    if (eta && plastic && !P->brain_off && P->learn_off != 2) {
        for (int o = 0; o < BIO_OUT; ++o) {
            if (!((plastic >> o) & 1u)) continue;
            uint64_t row = w->dW[(int64_t)o * w->cap + i];
            const int32_t ov = learn2 ? bio_lane(nudge, o) : bio_lane(out, o);
            for (int j = 0; j < BIO_H; ++j) {
                const int32_t hj = bio_lane(hid, j);
                const int32_t dw = bio_lane(row, j);
                if (!hj && !dw) continue;
                const uint32_t h = bio_hash((int32_t)uid, (int32_t)w->step, BH_LEARN * 64 + o * 8 + j, w->salt);
                const int64_t up = bio_srs((int64_t)eta * m * ov * hj, learn2 ? P->learn_shift : 21, h);
                const int64_t dn = bio_srs((int64_t)dw * gb[BG_DECAY], 8, h >> 11);
                row = bio_set_lane(row, j, (int32_t)bio_clamp64(dw + up - dn, -32, 32));
            }
            w->dW[(int64_t)o * w->cap + i] = row;
        }
    }
    /* inputs, q7 */
    int32_t in[BIO_IN];
    const int64_t V = bio_volume(w, c);
    const int32_t ereq = bio_repro_e(w, gb);
    in[0] = bio_q7((int64_t)E * 127 / (ereq > 0 ? ereq : 1));
    in[1] = bio_q7((int64_t)light * 127 / 600);
    in[2] = bio_q7(((int64_t)T - bio_topt(gb)) * 32 / bio_ttol(gb));
    in[3] = (w->flags[i] & BA_WET) ? 127 : bio_q7(w->W[i] / 2);
    in[4] = bio_sense_mm(w->fld[BF_O2 * n + c], P->km[BF_O2], V);
    in[5] = bio_sense_mm(w->fld[BF_H2S * n + c], P->km[BF_H2S], V);
    in[6] = bio_sense_mm(w->fld[BF_DOC * n + c] + w->fld[BF_DET * n + c], P->km[BF_DOC], V);
    if (gb[BG_SENSE] >= 128) {
        const int32_t f = bio_best_field(gb);
        const int32_t km = f >= 0 ? P->km[f] : 0;
        in[7] = bio_grad(w, c, f, 1, 0, km);
        in[8] = bio_grad(w, c, f, 0, 1, km);
    } else { in[7] = 0; in[8] = 0; }
    const uint32_t cn = w->cnt[c];
    in[9] = bio_q7((int64_t)cn * 4);
    in[10] = cn ? bio_q7((int64_t)bio_kin_count(w, c, BT_KIN(w->tr[i])) * 127 / cn) : 0;
    in[11] = 0; in[12] = 0; in[13] = 0; in[14] = 0;
    if ((P->feat & BIO_FEAT_PRED) && gb[BG_SENSE] >= 128) {
        /* where the danger grows (attackers whose gape fits an individual of this size) and where the crowd, i.e. food, grows */
        const int64_t body = bio_mr(w, gb[BG_SIZE]);
        in[11] = bio_grad_sum(w, c, 1, 0, body, 1); in[12] = bio_grad_sum(w, c, 0, 1, body, 1);
        in[13] = bio_grad_sum(w, c, 1, 0, body, 0); in[14] = bio_grad_sum(w, c, 0, 1, body, 0);
    }
    in[15] = bio_q7(w->G->day_sin_q12 >> 5);
    if (P->brain_off) {
        w->hid[i] = 0;
        w->out[i] = bio_out_bias(w, g);
    } else {
        int32_t pin[4], ph[2];
        for (int k = 0; k < 4; ++k)
            pin[k] = (int32_t)((uint32_t)(uint8_t)in[4 * k] | ((uint32_t)(uint8_t)in[4 * k + 1] << 8) |
                               ((uint32_t)(uint8_t)in[4 * k + 2] << 16) | ((uint32_t)(uint8_t)in[4 * k + 3] << 24));
        ph[0] = (int32_t)(uint32_t)hid; ph[1] = (int32_t)(uint32_t)(hid >> 32);
        const uint32_t on = gb[BG_NEURONS];
        uint64_t nh = 0;
        for (int j = 0; j < BIO_H; ++j) {
            if (!((on >> j) & 1u)) continue;
            int32_t s = 64 * g->b_h[j];
            for (int k = 0; k < 4; ++k) s = BIO_DP4A(bio_ld32(&g->w_ih[j][4 * k]), pin[k], s);
            s = BIO_DP4A(bio_ld32(&g->w_hh[j][0]), ph[0], s);
            s = BIO_DP4A(bio_ld32(&g->w_hh[j][4]), ph[1], s);
            nh = bio_set_lane(nh, j, w->T->tanh_q7[bio_clampi(s >> 7, -128, 127) + 128]);
        }
        const int32_t h0 = (int32_t)(uint32_t)nh, h1 = (int32_t)(uint32_t)(nh >> 32);
        uint64_t no = 0;
        for (int o = 0; o < BIO_OUT; ++o) {
            const uint64_t row = w->dW[(int64_t)o * w->cap + i];
            int32_t s = 64 * g->b_o[o];
            s = BIO_DP4A(bio_ld32(&g->w_ho[o][0]), h0, s);
            s = BIO_DP4A(bio_ld32(&g->w_ho[o][4]), h1, s);
            s = BIO_DP4A((int32_t)(uint32_t)row, h0, s);
            s = BIO_DP4A((int32_t)(uint32_t)(row >> 32), h1, s);
            no = bio_set_lane(no, o, w->T->tanh_q7[bio_clampi(s >> 7, -128, 127) + 128]);
        }
        if (learn2) {
            /* exploration: every plastic output of a learner moves by up to +-explore_q7 until its next think */
            uint64_t ne = 0;
            if (eta && plastic) {
                const uint32_t hx = bio_hash((int32_t)uid, (int32_t)w->step, BH_EXPL, w->salt);
                for (int o = 0; o < BIO_OUT; ++o) {
                    if (!((plastic >> o) & 1u)) continue;
                    const int32_t ov = bio_lane(no, o);
                    const int32_t nv = bio_clampi(ov + (2 * (int32_t)((hx >> (4 * o)) & 15u) - 15) * P->explore_q7 / 15, -127, 127);
                    no = bio_set_lane(no, o, nv);
                    ne = bio_set_lane(ne, o, nv - ov);
                }
            }
            w->expl[i] = ne;
        }
        w->hid[i] = nh;
        w->out[i] = no;
    }
    /* cost: neurons, senses, learning */
    int64_t cost = (int64_t)P->brain_q16 * bio_popc8(gb[BG_NEURONS]) + (int64_t)P->sense_q16 * gb[BG_SENSE] / 255;
    if (eta) cost += (int64_t)P->learn_q16 * bio_popc8(plastic) * 8 * eta / 255;
    return (int32_t)bio_sr16(cost, bio_hash((int32_t)uid, (int32_t)w->step, BH_COST, w->salt));
}

/* ==== k_think: agents whose think period is due this step (spec 3.4) ============== */
/* due test; clears last step's THOUGHT flag of every agent */
BIO_FN int bio_think_due(const BioWorld* w, int32_t i) {
    const uint32_t fl = w->flags[i];
    if (!(fl & BA_ALIVE)) return 0;
    if (fl & BA_THOUGHT) w->flags[i] = fl & ~BA_THOUGHT;
    return (uint32_t)((w->step + w->uid[i]) % (uint64_t)BT_PERIOD(w->tr[i])) == 0u;
}
/* the brain of one due agent; touches only that agent's state, so the order in
 * which due agents are processed does not matter */
BIO_FN void bio_think_run(const BioWorld* w, int32_t i) {
    const BioGenome* g = &w->gen[w->gslot[i]];
    const int32_t c = (int32_t)w->cell[i];
    const uint32_t fl = w->flags[i];
    const int32_t lt = bio_light_h(w, c, fl, bio_zq8(w, c, fl, g->body, w->out[i]),
                                   bio_plant_hq4(w, w->S[i], (w->P.feat & BIO_FEAT_MORPH) ? w->tr2[i] : 0u));
    w->E[i] -= bio_think(w, i, g, c, lt, bio_temp(w, c, w->G->day_sin_q12));
    w->flags[i] = fl | BA_THOUGHT;
}

/* ==== demands: requests -> per-pool amounts (shared by k_pre and k_metab) ======== */
/* gut: the agent's private organic carbon (eaten prey); it is used before the
 * cell's DOC, so only the remainder is demanded from the pool */
BIO_FN void bio_demands(const int32_t* rq, int64_t gut, int64_t* d) {
    for (int f = 0; f < BIO_POOLS; ++f) d[f] = 0;
    d[BF_H2] = 4 * (int64_t)rq[BP_METH];
    d[BF_DIC] = (int64_t)rq[BP_METH] + rq[BR_FIX];
    d[BF_H2S] = 2 * (int64_t)rq[BR_FIXA];          /* ANOX: 2 H2S per C fixed */
    d[BF_DOC] = (int64_t)rq[BP_FERM] + rq[BP_SULFRED] + rq[BP_AERO] + rq[BR_ASSIM];
    if (gut > 0) d[BF_DOC] -= bio_min64(gut, d[BF_DOC]);
    d[BF_SO4] = (int64_t)rq[BP_SULFRED] - rq[BP_SULFRED] / 2;
    d[BF_O2] = (int64_t)rq[BP_AERO] + 2 * (int64_t)rq[BP_METHOX] + rq[BP_FEOX];
    d[BF_CH4] = rq[BP_METHOX];
    d[BF_FE2] = 4 * (int64_t)rq[BP_FEOX];
    d[BF_NH4] = rq[BR_NH4];
    d[BF_PO4] = rq[BR_PO4];
    d[BF_SOILW] = rq[BR_WATER];
}

/* rate law for a chemotrophic pathway: C * e/255 * vmax * MM * MM * fT, q16 */
BIO_FN int64_t bio_rate(int32_t C, int32_t e, int32_t vmax_q16, int64_t mm1, int64_t mm2, int32_t ft) {
    int64_t v = (int64_t)C * e * vmax_q16 / 255;
    v = (v * mm1) >> 16;
    v = (v * mm2) >> 16;
    return (v * ft) >> 16;
}

/* ==== k_pre: think, move, then post requests (fields are read-only here) ========= */
BIO_FN void bio_agent_pre(const BioWorld* w, int32_t i) {
    uint32_t fl = w->flags[i];
    if (!(fl & BA_ALIVE)) return;
    const BioParams* P = &w->P;
    const BioGenome* g = &w->gen[w->gslot[i]];
    const uint8_t* gb = g->body;
    const uint32_t uid = w->uid[i];
    const int64_t n = w->n_cells;
    int32_t c = (int32_t)w->cell[i];
    int32_t E = w->E[i];
    const int32_t C = w->C[i], S = w->S[i];
    fl &= ~(BA_BIRTH | BA_DEAD);
    const uint64_t out = w->out[i];
    /* prey side of predation, before anyone moves: offer itself to the attacker
     * standing in one of the cell's slots. An anchored colony can always be
     * grazed; anything else only if one of its individuals (its reproduction
     * mass) fits the attacker's gape (the attacker's own size x MOUTH / 32).
     * Reach: an attacker slower than a metre an hour (MOTOR below
     * BIO_CREATURE_MOTOR) cannot go to its food and takes only what comes to
     * it, prey that walks or swims that fast, or that floats free in water.
     * Grazing what stands still is for those that move: evolved forests carry
     * mouths and motor genes of 1 or 2 that drifted in, and without this rule
     * the plants eat each other where they stand. A small body is then eaten
     * whole, a colony loses a bite. The attacker takes the offer with the
     * lowest key (k_metab, k_feed).
     * Refuge: a cell hides pred_refuge1 - 1 of its agents that are not movers
     * (bio_mover), so of nn of them each is found with probability
     * 1 - refuge / nn, and the last one never: predation thins a cell out but
     * does not empty it. The movers are out in the open. */
    int32_t prey_of = 0, prey_bite = 0;
    if ((P->feat & BIO_FEAT_PRED) && (int64_t)C + S > 0) {
        const uint32_t hp = bio_hash((int32_t)uid, (int32_t)w->step, BH_PREY, w->salt);
        const int64_t nn = (int64_t)w->cnt[c] - (int64_t)w->nmot[c], hide = P->pred_refuge1 - 1;
        const int found = bio_mover(w->tr[i], gb[BG_MOTOR]) ||
                          (nn > hide && (int64_t)(bio_hash((int32_t)uid, (int32_t)w->step, BH_HIDE, w->salt) % (uint32_t)nn) >= hide);
        const uint32_t key = found ? w->pslot[(int64_t)(hp & (BIO_PSLOTS - 1)) * n + c] : 0u;
        if (key) {
            const int32_t a = bio_slot_agent(w, key, c, BH_PMASK);
            if (a != i) {
                const uint32_t ta = w->tr2[a];
                const int64_t mouth = BT2_MOUTH(ta), ca = w->C[a], body = (int64_t)C + S;
                const int64_t b0 = bio_max64(1, ((ca * mouth / 255) * P->pred_bite_q10) >> 10);
                const int hold = (w->tr[i] & BT_HOLD) != 0;
                const int reach = BT2_MOTOR(ta) >= BIO_CREATURE_MOTOR || gb[BG_MOTOR] >= BIO_CREATURE_MOTOR || ((fl & BA_WET) && !hold);
                if (reach && (hold || (uint64_t)bio_mr(w, gb[BG_SIZE]) <= BT2_GAPE(ta))) {
                    BIO_MIN64(&w->claim[a], ((uint64_t)bio_hash((int32_t)uid, (int32_t)w->step, BH_CLAIM, w->salt) << 32) | (uint32_t)i);
                    prey_of = a + 1;
                    prey_bite = (hold || body > 4 * b0) ? (int32_t)bio_min64(b0, 1 << 28) : -1;   /* -1: eaten whole */
                }
            }
        }
    }
    /* move: motile agents steer, plankton drifts */
    {
        int32_t dx = 0, dz = 0;
        if (gb[BG_MOTOR] > 0) {
            const int64_t sp = (int64_t)gb[BG_MOTOR] * P->speed_q8 / 255;
            dx = (int32_t)(sp * bio_lane(out, 0) / 127);
            dz = (int32_t)(sp * bio_lane(out, 1) / 127);
        } else if (gb[BG_HOLDFAST] < 128 && (fl & BA_WET)) {
            const uint32_t h = bio_hash((int32_t)uid, (int32_t)w->step, BH_DRIFT, w->salt);
            dx = (((int32_t)(h & 1023u) - 512) * P->drift_q8) >> 9;
            dz = (((int32_t)((h >> 10) & 1023u) - 512) * P->drift_q8) >> 9;
        }
        if (dx || dz) {
            const int32_t lx = w->w << (w->shift + 8), lz = w->h << (w->shift + 8);
            const int32_t nx = bio_reflect(w->x[i] + dx, lx), nz = bio_reflect(w->z[i] + dz, lz);
            if (gb[BG_MOTOR] > 0) {
                const int64_t v = (int64_t)(dx < 0 ? -dx : dx) + (dz < 0 ? -dz : dz);   /* m q8 */
                int64_t cost = ((int64_t)C * P->move_q16 * ((v * v) >> 8)) >> 8;            /* q16 eu */
                if (fl & BA_LAND) cost += (((int64_t)C * P->move_land_q16 * v) >> 8) * P->barrier_q8 >> 8;
                E -= (int32_t)bio_min64(bio_sr16(cost, bio_hash((int32_t)uid, (int32_t)w->step, BH_MOVE, w->salt)), (int64_t)1 << 28);
            }
            w->x[i] = nx; w->z[i] = nz;
            c = bio_cell_at(w, nx, nz);
            w->cell[i] = (uint32_t)c;
        }
    }
    /* wet or dry this hour: a fixed micro-elevation rank per agent */
    {
        const int32_t wf = bio_wet_frac(w, c, w->G->tide);
        const int wet = wf >= 255 ? 1 : (wf <= 0 ? 0 : (int)((bio_hash((int32_t)uid, 0, BH_TIDE, w->salt) & 255u) < (uint32_t)wf));
        fl &= ~(BA_WET | BA_LAND);
        fl |= wet ? BA_WET : BA_LAND;
    }
    /* requests */
    int32_t rq[BR_N];
    for (int k = 0; k < BR_N; ++k) rq[k] = 0;
    rq[BR_PRED] = prey_of; rq[BR_BITE] = prey_bite;
    const int64_t gutv = (P->feat & BIO_FEAT_PRED) ? w->gut[i] : 0;
    const int64_t V = bio_volume(w, c);
    int64_t pool[BIO_POOLS];
    for (int f = 0; f < BIO_POOLS; ++f) { pool[f] = w->fld[f * n + c]; if (pool[f] < 0) pool[f] = 0; }
    int32_t dt = bio_temp(w, c, w->G->day_sin_q12) - bio_topt(gb);
    if (dt < 0) dt = -dt;
    const int32_t ft = w->T->gauss_q16[bio_clampi(dt * 64 / bio_ttol(gb), 0, 255)];
    const uint32_t hh = bio_hash((int32_t)uid, (int32_t)w->step, BH_SR, w->salt);
#define BIO_MMK(f) bio_mm(pool[f], (int64_t)P->km[f] * V)
#define BIO_RQ(k, e, m1, m2, sh) rq[k] = (int32_t)bio_min64(bio_sr16(bio_rate(C, (e), P->vmax_q16[k], (m1), (m2), ft), hh >> (sh)), 1 << 28)
    if (C > 0) {
        /* organic carbon for the heterotrophic pathways: the cell's DOC, or the gut when that is fuller */
        int64_t mmdoc = 0;
        if (gb[BG_E_FERM] | gb[BG_E_SULFRED] | gb[BG_E_AERO]) {
            mmdoc = BIO_MMK(BF_DOC);
            if (gutv > 0) {
                const int64_t mg = bio_mm(gutv, bio_max64(1, ((int64_t)C * P->gut_k_q8) >> 8));
                if (mg > mmdoc) mmdoc = mg;
            }
        }
        if (gb[BG_E_METH]) BIO_RQ(BP_METH, gb[BG_E_METH], BIO_MMK(BF_H2), BIO_MMK(BF_DIC), 0);
        if (gb[BG_E_FERM]) BIO_RQ(BP_FERM, gb[BG_E_FERM], mmdoc, 65536, 1);
        if (gb[BG_E_SULFRED]) BIO_RQ(BP_SULFRED, gb[BG_E_SULFRED], mmdoc, BIO_MMK(BF_SO4), 2);
        if (gb[BG_E_AERO]) BIO_RQ(BP_AERO, gb[BG_E_AERO], mmdoc, BIO_MMK(BF_O2), 3);
        if (gb[BG_E_METHOX]) BIO_RQ(BP_METHOX, gb[BG_E_METHOX], BIO_MMK(BF_CH4), BIO_MMK(BF_O2), 4);
        if (gb[BG_E_FEOX]) BIO_RQ(BP_FEOX, gb[BG_E_FEOX], BIO_MMK(BF_FE2), BIO_MMK(BF_O2), 5);
        /* light pathways: absorbed light split between ANOX and OXY by enzyme level */
        const int32_t ea = gb[BG_E_ANOX], eo = gb[BG_E_OXY];
        if ((ea || eo) && gb[BG_PIGMENT] && (P->light_mode != BIO_LIGHT_SCALAR || (P->feat & BIO_FEAT_MORPH))) {
            /* bands and / or crown model. Leaf area: the scalar model scales it with the
             * pigment; with bands the pigment acts through its absorptance instead. */
            if (w->G->i0 > 0) {
                const int spec = P->light_mode == BIO_LIGHT_SPECTRAL, morph = (P->feat & BIO_FEAT_MORPH) != 0;
                const int32_t zq8 = bio_zq8(w, c, fl, gb, out);
                const uint32_t t2 = w->tr2[i];
                const int32_t hq4 = bio_plant_hq4(w, S, t2);
                int64_t idx, g2, satcap = 0;
                bio_shade(w, c, fl, zq8, hq4, &idx, &g2);
                int64_t aq = bio_leaf_q20(w, C, spec ? 255 : gb[BG_PIGMENT]);
                /* With shapes on, the leaves of every land plant saturate (satcap: what
                 * its lit leaf can use, eu q16 per step), and a standing plant (0.5 m and
                 * up) has a crown. Ground cover keeps its whole leaf in the light. */
                if (morph && (fl & BA_LAND)) {
                    if (hq4 >= 8) {
                        int64_t asun;
                        aq = bio_crown(w, aq, hq4, t2, gb[BG_POROSITY], &asun);
                        satcap = (asun * P->morph_pmax) << 4;
                    } else satcap = (aq * P->morph_pmax) >> 4;
                }
                int64_t La, Lo;                                               /* eu q16 per step, each pathway alone */
                if (spec) {
                    int32_t ib[BIO_NB];
                    int64_t q[2];
                    bio_light_bands(w, fl, zq8, idx, g2, ib);
                    bio_harvest(w, ib, gb[BG_PIGMENT], gb[BG_HUE], morph ? (gb[BG_QWIDTH] >> 4) : 0, q);
                    Lo = bio_mulshr((uint64_t)q[0] * (uint64_t)aq, (uint64_t)P->eff_sp_oxy_q16, 28);
                    La = bio_mulshr((uint64_t)q[1] * (uint64_t)aq, (uint64_t)P->eff_sp_anox_q16, 28);
                } else {
                    const int64_t lt = ((((int64_t)w->G->i0 * bio_expn(w, idx)) >> 16) * g2) >> 16;
                    La = bio_mulshr((uint64_t)lt * (uint64_t)aq, (uint64_t)P->eff_anox_q16, 20);
                    Lo = bio_mulshr((uint64_t)lt * (uint64_t)aq, (uint64_t)P->eff_oxy_q16, 20);
                }
                La = La * ea / (ea + eo); Lo = Lo * eo / (ea + eo);
                if (satcap > 0 && La + Lo > 0) {                              /* the lit leaves saturate */
                    const int64_t all = La + Lo, r = (bio_saturate(w, all, satcap) << 16) / all;
                    La = (La * r) >> 16; Lo = (Lo * r) >> 16;
                }
                if (ea && La > 0) {
                    int64_t L = (La * BIO_MMK(BF_H2S)) >> 16;
                    L = (L * ft) >> 16;
                    const int64_t cap = (int64_t)C * ea * P->vmax_q16[BP_ANOX] / 255 * P->fix_eu;
                    rq[BP_ANOX] = (int32_t)bio_min64(bio_sr16(bio_min64(L, cap) / P->fix_eu, hh >> 6), 1 << 28);
                }
                if (eo && Lo > 0) {
                    const int64_t ktox = (int64_t)P->ktox_h2s * V;
                    int64_t L = bio_mulshr((uint64_t)Lo, (uint64_t)bio_mm(ktox, pool[BF_H2S]), 16);   /* sulfide inhibition */
                    L = (L * ft) >> 16;
                    const int64_t cap = (int64_t)C * eo * P->vmax_q16[BP_OXY] / 255 * P->fix_eu;
                    rq[BP_OXY] = (int32_t)bio_min64(bio_sr16(bio_min64(L, cap) / P->fix_eu, hh >> 7), 1 << 28);
                }
            }
        } else if ((ea || eo) && gb[BG_PIGMENT]) {
            const int32_t lt = bio_light(w, c, fl, bio_zq8(w, c, fl, gb, out), S);
            if (lt > 0) {
                const int64_t aq = bio_leaf_q20(w, C, gb[BG_PIGMENT]);
                if (ea) {
                    int64_t L = bio_mulshr((uint64_t)lt * (uint64_t)aq, (uint64_t)P->eff_anox_q16, 20) * ea / (ea + eo);
                    L = (L * BIO_MMK(BF_H2S)) >> 16;
                    L = (L * ft) >> 16;
                    const int64_t cap = (int64_t)C * ea * P->vmax_q16[BP_ANOX] / 255 * P->fix_eu;
                    rq[BP_ANOX] = (int32_t)bio_min64(bio_sr16(bio_min64(L, cap) / P->fix_eu, hh >> 6), 1 << 28);
                }
                if (eo) {
                    int64_t L = bio_mulshr((uint64_t)lt * (uint64_t)aq, (uint64_t)P->eff_oxy_q16, 20) * eo / (ea + eo);
                    const int64_t ktox = (int64_t)P->ktox_h2s * V;
                    L = bio_mulshr((uint64_t)L, (uint64_t)bio_mm(ktox, pool[BF_H2S]), 16);   /* sulfide inhibition */
                    L = (L * ft) >> 16;
                    const int64_t cap = (int64_t)C * eo * P->vmax_q16[BP_OXY] / 255 * P->fix_eu;
                    rq[BP_OXY] = (int32_t)bio_min64(bio_sr16(bio_min64(L, cap) / P->fix_eu, hh >> 7), 1 << 28);
                }
            }
        }
        /* growth: energy above a reserve, scaled by the allocation output */
        if (!P->grow_off) {
            const int32_t mr = bio_mr(w, gb[BG_SIZE]);
            const int32_t ereq = bio_repro_e(w, gb);
            const int64_t surplus = (int64_t)E - (C < mr ? 2 * P->copy_cost : ereq + P->copy_cost);
            const int64_t geu = surplus > 0 ? (surplus * (128 + bio_lane(out, 4))) >> 8 : 0;
            const int64_t gcap = bio_sr16((int64_t)C * P->mu_q16, hh >> 8);
            int64_t a = 0, f = 0;
            const int32_t eh = gb[BG_E_FERM] + gb[BG_E_SULFRED] + gb[BG_E_AERO];
            const int32_t eat = gb[BG_E_METH] + gb[BG_E_ANOX] + gb[BG_E_OXY] + gb[BG_E_FEOX] + gb[BG_E_METHOX];
            if (eh > 0 && geu > 0) {
                const int64_t up = bio_sr16(bio_rate(C, eh > 255 ? 255 : eh, P->vmax_q16[BP_FERM], mmdoc, 65536, 65536), hh >> 9);
                a = bio_min64(bio_min64(gcap, geu / P->assim_eu), up);
            }
            /* fixation is requested independently of assimilation: when DOC is contended
             * its grant falls short and fixation fills the growth capacity instead
             * (k_metab caps assimilation + fixation at this request) */
            if (eat > 0 && geu > 0) f = bio_min64(gcap, geu / P->fix_eu);
            if (f < 0) f = 0;
            rq[BR_ASSIM] = (int32_t)bio_min64(a, 1 << 28);
            rq[BR_FIX] = (int32_t)bio_min64(f, 1 << 28);
            /* reductant for fixation: phototrophs use H2S (ANOX) or water (OXY, releases
             * O2) in proportion to their enzymes; chemoautotrophs need no extra donor */
            if ((ea || eo) && gb[BG_PIGMENT]) {
                rq[BR_FIXA] = (int32_t)((int64_t)rq[BR_FIX] * ea / (ea + eo));
                rq[BR_FIXO] = rq[BR_FIX] - rq[BR_FIXA];
            }
        }
    }
#undef BIO_RQ
    /* nutrients for the body plus planned growth (small luxury store) */
    {
        const int64_t body = (int64_t)C + S + rq[BR_FIX] + rq[BR_ASSIM];
        const int64_t needn = (body + P->cn - 1) / P->cn + 1 - w->N[i];
        const int64_t needp = (body + P->cp - 1) / P->cp + 1 - w->Pp[i];
        rq[BR_NH4] = (int32_t)bio_clamp64(needn, 0, 1 << 28);
        rq[BR_PO4] = (int32_t)bio_clamp64(needp, 0, 1 << 28);
    }
    /* water from the soil on dry ground */
    if ((fl & BA_LAND) && gb[BG_ROOT] > 0 && w->W[i] < 256) {
        const int64_t capl = bio_max64(1, ((int64_t)(C + S) * P->water_q16) >> 16);
        rq[BR_WATER] = (int32_t)bio_min64(((256 - w->W[i]) * capl / 256 + 1) * gb[BG_ROOT] / 255, 1 << 28);
    }
    /* aerobic catabolism possible (used when starving) */
    rq[BR_AUX] = (gb[BG_E_AERO] > 0 && pool[BF_O2] > (int64_t)P->k_o2 * V) ? 1 : 0;
    /* O2 toxicity for agents with anaerobic enzymes */
    if (pool[BF_O2] > 0 && C > 0 && (gb[BG_E_METH] | gb[BG_E_ANOX] | gb[BG_E_FERM] | gb[BG_E_SULFRED])) {
        const int64_t mm = BIO_MMK(BF_O2);
        const int64_t cost = (((int64_t)C * P->o2tox_q16 * mm) >> 16) * (255 - gb[BG_O2TOL]) / 255;
        E -= (int32_t)bio_min64(bio_sr16(cost, hh >> 10), (int64_t)1 << 28);
    }
#undef BIO_MMK
    int64_t d[BIO_POOLS];
    bio_demands(rq, gutv, d);
    int any = 0;
    for (int f = 0; f < BIO_POOLS; ++f) if (d[f] > 0) { BIO_ADD64(&w->demand[f * n + c], d[f]); any = 1; }
    if (any) w->dflag[c] = 1u;
    const int nrq = (P->feat & BIO_FEAT_PRED) ? BR_N : BR_PRED;       /* the predation slots only when it is on */
    for (int k = 0; k < nrq; ++k) w->req[(int64_t)k * w->cap + i] = rq[k];
    w->E[i] = E;
    w->flags[i] = fl;
}

/* ==== k_supply: fair-share ratio per pool; clears the demand ===================== */
BIO_FN void bio_cell_supply(const BioWorld* w, int32_t c) {
    if (!w->dflag[c]) return;
    const int64_t n = w->n_cells;
    for (int f = 0; f < BIO_POOLS; ++f) {
        const uint64_t d = w->demand[f * n + c];
        uint32_t r = 65536u;
        if (d) {
            int64_t pool = w->fld[f * n + c];
            if (pool < 0) pool = 0;
            if ((uint64_t)pool < d) {
                if (pool < ((int64_t)1 << 46)) r = (uint32_t)(((uint64_t)pool << 16) / d);
                else r = (uint32_t)((uint64_t)pool / (d >> 16));
            }
            w->demand[f * n + c] = 0;
        }
        w->ratio[f * n + c] = r;
    }
    w->dflag[c] = 0u;
}

/* offspring position: parent + hash offset of up to max(4 m, DISPERSE/16 cells) */
BIO_FN void bio_birth_pos(const BioWorld* w, int32_t i, const uint8_t* gb, int32_t* ox, int32_t* oz) {
    const uint32_t h = bio_hash((int32_t)w->uid[i], (int32_t)w->step, BH_DISP, w->salt);
    int64_t r = (int64_t)(gb[BG_DISPERSE] >> 4) << (w->shift + 8);
    if (r < 4 * 256) r = 4 * 256;
    const int32_t dx = (int32_t)((((int64_t)(h & 0xFFFFu)) * (2 * r + 1)) >> 16) - (int32_t)r;
    const int32_t dz = (int32_t)((((int64_t)(h >> 16)) * (2 * r + 1)) >> 16) - (int32_t)r;
    const int32_t lx = w->w << (w->shift + 8), lz = w->h << (w->shift + 8);
    *ox = bio_reflect(w->x[i] + dx, lx);
    *oz = bio_reflect(w->z[i] + dz, lz);
}

/* what a prey tells the animal that ate from it: bits 0-7 its trophic level above
 * 1, bit 8 a producer, bit 9 on dry land, bits 10-13 its kin bucket */
BIO_FN uint32_t bio_prey_info(uint32_t eco, uint32_t tr, uint32_t fl) {
    return BE_TROPH(eco) | ((tr & BT_PHOTO) ? 256u : 0u) | ((fl & BA_LAND) ? 512u : 0u) | (BT_KIN(tr) << 10);
}

/* ==== k_metab: grants, pathways, growth, costs, death, birth decision ============ */
BIO_FN void bio_agent_metab(const BioWorld* w, int32_t i, int64_t* acc) {
    uint32_t fl = w->flags[i];
    if (!(fl & BA_ALIVE)) return;
    const BioParams* P = &w->P;
    const BioGenome* g = &w->gen[w->gslot[i]];
    const uint8_t* gb = g->body;
    const uint32_t uid = w->uid[i];
    const int64_t n = w->n_cells, cap = w->cap;
    const int32_t c = (int32_t)w->cell[i];
    int64_t C = w->C[i], S = w->S[i], E = w->E[i], N = w->N[i], Pa = w->Pp[i];
    int32_t W = w->W[i];
    const int pred_on = (P->feat & BIO_FEAT_PRED) != 0;
    int32_t rq[BR_N];
    for (int k = 0; k < BR_PRED; ++k) rq[k] = w->req[(int64_t)k * cap + i];
    rq[BR_PRED] = pred_on ? w->req[(int64_t)BR_PRED * cap + i] : 0;
    rq[BR_BITE] = pred_on ? w->req[(int64_t)BR_BITE * cap + i] : 0;
    int64_t gutv = pred_on ? w->gut[i] : 0;
    /* prey side of predation (spec 3.8): did the attacker this agent offered itself
     * to take its offer, and does the attack succeed? Everything written here is the
     * prey's own state; the attacker collects the carbon in k_feed. */
    if (rq[BR_PRED]) {
        const int32_t a = rq[BR_PRED] - 1;
        int64_t given = 0;
        if (w->claim[a] == (((uint64_t)bio_hash((int32_t)uid, (int32_t)w->step, BH_CLAIM, w->salt) << 32) | (uint32_t)i)) {
            /* success: faster than the prey, and through its armour (the spec's sigmoid, steeper: tuned) */
            const int32_t dm = ((int32_t)BT2_MOTOR(w->tr2[a]) - gb[BG_MOTOR]) / 4;
            const uint32_t p = (uint32_t)(((int64_t)w->T->sigmoid_q16[bio_clampi(128 + dm, 0, 255)] * (300 - gb[BG_ARMOR])) / 300);
            if ((bio_hash((int32_t)uid, (int32_t)w->step, BH_ROLL, w->salt) & 0xFFFFu) < p) {
                if (rq[BR_BITE] >= 0) {                     /* a bite out of a colony: at most half of its reserve */
                    given = bio_min64(C / 2, rq[BR_BITE]);
                    C -= given;
                    if (given > 0) BIO_ACC(&acc[BACC_X(BACC2_BITES)], 1);
                } else {                                    /* eaten whole: most to the gut, the rest dissolves */
                    const int64_t body = C + S + gutv;
                    given = bio_min64((body * P->pred_kill_q8) >> 8, 2000000000);
                    BIO_ADD64(&w->fld[BF_DOC * n + c], body - given);
                    if (N) BIO_ADD64(&w->fld[BF_NH4 * n + c], N);
                    if (Pa) BIO_ADD64(&w->fld[BF_PO4 * n + c], Pa);
                    BIO_ACC(&acc[BACC_X(BACC2_KILLS)], 1);
                    w->req[(int64_t)BR_BITE * cap + i] = (int32_t)given;
                    w->req[(int64_t)BR_PRED * cap + i] = (int32_t)bio_prey_info(w->eco[i], w->tr[i], fl);
                    w->C[i] = 0; w->S[i] = 0; w->E[i] = 0; w->N[i] = 0; w->Pp[i] = 0; w->gut[i] = 0;
                    w->age[i] = w->age[i] + 1u;
                    w->flags[i] = BA_DEAD | (fl & BA_HUNT);
                    return;
                }
            }
        }
        w->req[(int64_t)BR_BITE * cap + i] = (int32_t)given;
        if (given > 0)
            w->req[(int64_t)BR_PRED * cap + i] = (int32_t)bio_prey_info(w->eco[i], w->tr[i], fl);
    }
    int64_t d[BIO_POOLS], B[BIO_POOLS], dF[BF_N];
    bio_demands(rq, gutv, d);
    for (int f = 0; f < BIO_POOLS; ++f) B[f] = d[f] > 0 ? (int64_t)(((uint64_t)d[f] * w->ratio[f * n + c]) >> 16) : 0;
    for (int f = 0; f < BF_N; ++f) dF[f] = 0;
    /* the gut stands in for the cell's DOC as far as it reaches (it was left out of the demand) */
    const int64_t gdraw = gutv > 0 ? bio_min64(gutv, (int64_t)rq[BP_FERM] + rq[BP_SULFRED] + rq[BP_AERO] + rq[BR_ASSIM]) : 0;
    B[BF_DOC] += gdraw;
    int64_t a_used = 0, f_used = 0;
    /* pathway grants, limited by this agent's share of every substrate */
    int64_t gp[BP_N];
    gp[BP_METH] = bio_min64(bio_min64(rq[BP_METH], B[BF_H2] / 4), B[BF_DIC]);
    B[BF_H2] -= 4 * gp[BP_METH]; B[BF_DIC] -= gp[BP_METH];
    /* light harvest gives ATP without a chemical signature (cyclic photophosphorylation);
     * the donor is consumed when carbon is fixed (below) */
    gp[BP_ANOX] = rq[BP_ANOX];
    gp[BP_OXY] = rq[BP_OXY];
    gp[BP_FERM] = bio_min64(rq[BP_FERM], B[BF_DOC]); B[BF_DOC] -= gp[BP_FERM];
    gp[BP_SULFRED] = bio_min64(bio_min64(rq[BP_SULFRED], B[BF_DOC]), 2 * B[BF_SO4]);
    B[BF_DOC] -= gp[BP_SULFRED]; B[BF_SO4] -= gp[BP_SULFRED] - gp[BP_SULFRED] / 2;
    gp[BP_AERO] = bio_min64(bio_min64(rq[BP_AERO], B[BF_DOC]), B[BF_O2]);
    B[BF_DOC] -= gp[BP_AERO]; B[BF_O2] -= gp[BP_AERO];
    gp[BP_METHOX] = bio_min64(bio_min64(rq[BP_METHOX], B[BF_CH4]), B[BF_O2] / 2);
    B[BF_CH4] -= gp[BP_METHOX]; B[BF_O2] -= 2 * gp[BP_METHOX];
    gp[BP_FEOX] = bio_min64(bio_min64(rq[BP_FEOX], B[BF_FE2] / 4), B[BF_O2]);
    B[BF_FE2] -= 4 * gp[BP_FEOX]; B[BF_O2] -= gp[BP_FEOX];
    /* products (matched element transfers) */
    { const int64_t q = gp[BP_METH]; dF[BF_H2] -= 4 * q; dF[BF_DIC] -= q; dF[BF_CH4] += q; }
    { const int64_t q = gp[BP_FERM]; dF[BF_DOC] -= q; dF[BF_DIC] += q - q / 2; dF[BF_CH4] += q / 2; }
    { const int64_t q = gp[BP_SULFRED], h = q - q / 2; dF[BF_DOC] -= q; dF[BF_SO4] -= h; dF[BF_DIC] += q; dF[BF_H2S] += h; }
    { const int64_t q = gp[BP_AERO]; dF[BF_DOC] -= q; dF[BF_O2] -= q; dF[BF_DIC] += q; }
    { const int64_t q = gp[BP_METHOX]; dF[BF_CH4] -= q; dF[BF_O2] -= 2 * q; dF[BF_DIC] += q; }
    { const int64_t q = gp[BP_FEOX]; dF[BF_FE2] -= 4 * q; dF[BF_O2] -= q; BIO_ACC(&acc[BACC_SED_FE], 4 * q); }
    int64_t gain = 0;
    BIO_UNROLL
    for (int k = 0; k < BP_N; ++k) {
        gain += gp[k] * P->yield_q16[k];
        if (k != BP_ANOX && k != BP_OXY) BIO_ACC(&acc[BACC_PATH + k], gp[k]);   /* ANOX/OXY: C fixed, below */
    }
    const int64_t resp = (gp[BP_FERM] - gp[BP_FERM] / 2) + gp[BP_SULFRED] + gp[BP_AERO] + gp[BP_METHOX];
    uint32_t hh = bio_hash((int32_t)uid, (int32_t)w->step, BH_SR + 16, w->salt);
    E += bio_sr16(gain, hh);
    /* nutrients into the agent's store */
    N += B[BF_NH4]; dF[BF_NH4] -= B[BF_NH4];
    Pa += B[BF_PO4]; dF[BF_PO4] -= B[BF_PO4];
    /* growth: assimilation of DOC (2 eu/mmol) or fixation of DIC (8 eu/mmol) */
    {
        int64_t a = bio_min64(rq[BR_ASSIM], B[BF_DOC]);
        int64_t f = bio_min64(rq[BR_FIX], B[BF_DIC]);
        const int64_t e0 = E > 0 ? E : 0;
        a = bio_min64(a, e0 / P->assim_eu);
        f = bio_min64(f, bio_max64(0, rq[BR_FIX] - a));      /* growth capacity shared with assimilation */
        f = bio_min64(f, (e0 - a * P->assim_eu) / P->fix_eu);
        const int64_t lim = bio_max64(0, bio_min64(bio_min64(N * P->cn, Pa * P->cp), 1000000000) - (C + S));
        if (a > lim) a = lim;
        if (a + f > lim) f = lim - a;
        if (f < 0) f = 0;
        /* split by reductant: ANOX 2 H2S -> 2 SO4 (S0 lumped) per C, OXY 1 O2 per C */
        int64_t fa = 0, fo = 0;
        if (f > 0 && rq[BR_FIX] > 0 && (rq[BR_FIXA] | rq[BR_FIXO])) {
            fa = f * rq[BR_FIXA] / rq[BR_FIX];
            fo = rq[BR_FIXA] + rq[BR_FIXO] == rq[BR_FIX] ? f - fa : f * rq[BR_FIXO] / rq[BR_FIX];
            if (2 * fa > B[BF_H2S]) { const int64_t cut = fa - B[BF_H2S] / 2; fa -= cut; f -= cut; }
        }
        if (a + f > 0) {
            E -= a * P->assim_eu + f * P->fix_eu;
            dF[BF_DOC] -= a; dF[BF_DIC] -= f;
            dF[BF_H2S] -= 2 * fa; dF[BF_SO4] += 2 * fa; dF[BF_O2] += fo;
            BIO_ACC(&acc[BACC_PATH + BP_ANOX], fa); BIO_ACC(&acc[BACC_PATH + BP_OXY], fo);
            BIO_ACC(&acc[BACC_FIX], f); BIO_ACC(&acc[BACC_ASSIM], a);
            const int64_t ds = ((a + f) * gb[BG_STRUCT]) >> 8;
            S += ds; C += a + f - ds;
            a_used = a; f_used = f;
        }
    }
    if (pred_on) {
        /* the organic carbon used came out of the gut first; the diet record follows */
        const int64_t used = gp[BP_FERM] + gp[BP_SULFRED] + gp[BP_AERO] + a_used;
        const int64_t fromgut = bio_min64(gdraw, used);
        if (fromgut > 0) {
            dF[BF_DOC] += fromgut; gutv -= fromgut;
            BIO_ACC(&acc[BACC_X(BACC2_GUT_C)], fromgut);
        }
        if (used + f_used > 0) {
            const uint32_t e = w->eco[i];
            const uint32_t gs = (BE_GUTSHARE(e) * 15u + (uint32_t)(fromgut * 255 / (used + f_used)) + 8u) >> 4;
            if (gs != BE_GUTSHARE(e)) w->eco[i] = (e & ~0xFF00u) | (gs << 8);
        }
    }
    /* costs */
    const int32_t T = bio_temp(w, c, w->G->day_sin_q12);
    {
        const int32_t q10 = w->T->q10_q16[bio_clampi(64 + (T - 30300) / 100, 0, 127)];
        /* crowding: the anchored and the drifting compete with everyone in the cell for
         * room; with animals in the world, motile agents only with other motile ones */
        const int64_t crowd = (pred_on && bio_mover(w->tr[i], gb[BG_MOTOR])) ? w->nmot[c] : w->cnt[c];
        int64_t cost = (((C * P->maint_q16) * (int64_t)q10) >> 16) * (16 + crowd) / 16;
        int64_t e2 = 0;
        if (P->feat & BIO_FEAT_MORPH) {
            /* a crown class costs structure; a wide Q band costs pigment (only light in bands can use it) */
            const int64_t qw = P->light_mode == BIO_LIGHT_SPECTRAL ? (gb[BG_QWIDTH] >> 4) : 0;
            cost += (S * P->struct_q16 * bio_crown_cost_q8(BT2_CROWN(w->tr2[i]))) >> 8;
            for (int k = 0; k < 8; ++k) e2 += (int64_t)gb[k] * gb[k];
            cost += C * P->enz_q16 * e2 / 65025;
            cost += C * P->trait_q16 * (gb[BG_UVSHIELD] + gb[BG_CUTICLE] + gb[BG_ARMOR] + gb[BG_ROOT] +
                                        gb[BG_PIGMENT] + ((gb[BG_PIGMENT] * qw * P->qwidth_cost_q8 / 15) >> 8)) / 255;
        } else {
        cost += S * P->struct_q16;
        for (int k = 0; k < 8; ++k) e2 += (int64_t)gb[k] * gb[k];
        cost += C * P->enz_q16 * e2 / 65025;
        cost += C * P->trait_q16 * (gb[BG_UVSHIELD] + gb[BG_CUTICLE] + gb[BG_ARMOR] + gb[BG_ROOT] + gb[BG_PIGMENT]) / 255;
        }
        const int32_t zq8 = bio_zq8(w, c, fl, gb, w->out[i]);
        if ((fl & BA_LAND) || zq8 < 3 * 256) {
            const int64_t dose = ((int64_t)w->G->uv_q8 * bio_expn(w, zq8 / 6)) >> 16;
            int64_t u = C * P->uv_q16 * dose / 255 * (255 - gb[BG_UVSHIELD]) / 255;
            if (fl & BA_LAND) u = u * P->barrier_q8 >> 8;
            cost += u;
        }
        E -= bio_sr16(cost, hh >> 1);
    }
    /* water */
    if (fl & BA_WET) W = 256;
    else {
        const int64_t loss = (int64_t)P->dry_q8 * (255 - gb[BG_CUTICLE]) / 255 * P->barrier_q8 >> 8;
        W -= (int32_t)bio_srs(loss, 8, hh >> 2);
        if (B[BF_SOILW] > 0) {
            const int64_t capl = bio_max64(1, ((C + S) * P->water_q16) >> 16);
            W += (int32_t)bio_min64(256, B[BF_SOILW] * 256 / capl);
            dF[BF_SOILW] -= B[BF_SOILW];
            if (W > 256) W = 256;
        }
    }
    /* starvation: catabolise reserve, then structure */
    int64_t cat = 0;
    if (E < 0) {
        const int64_t y = rq[BR_AUX] ? P->starve_aero_q8 : P->starve_anaer_q8;
        int64_t need = ((-E) * 256 + y - 1) / y;
        int64_t k = bio_min64(C, need);
        C -= k; cat += k; need -= k;
        if (need > 0) { k = bio_min64(S, need); S -= k; cat += k; }
        E += (cat * y) >> 8;
        dF[BF_DIC] += cat;
    }
    BIO_ACC(&acc[BACC_RESP], resp + cat);
    /* stores and caps */
    {
        const int64_t mn = 2 * ((C + S + P->cn - 1) / P->cn) + 2, mp = 2 * ((C + S + P->cp - 1) / P->cp) + 2;
        if (N > mn) { dF[BF_NH4] += N - mn; N = mn; }
        if (Pa > mp) { dF[BF_PO4] += Pa - mp; Pa = mp; }
        const int64_t ecap = bio_min64((int64_t)P->e_cap_mul * (C + S) + 4096, 1000000000);
        if (E > ecap) E = ecap;
    }
    /* death */
    const uint32_t age = w->age[i] + 1u;
    const int32_t mr = bio_mr(w, gb[BG_SIZE]);
    int dead = 0;
    if (E < 0 || W < 0) dead = 1;
    if (C + S < (mr >= 64 ? mr / 64 : 1)) dead = 1;
    if ((int64_t)age > bio_lifespan_h(gb)) dead = 1;
    { int32_t dt = T - bio_topt(gb); if (dt < 0) dt = -dt; if (dt > 3 * bio_ttol(gb)) dead = 1; }
    if (bio_hash((int32_t)uid, (int32_t)w->step, BH_DEATH, w->salt) < (uint32_t)P->hazard_q32) dead = 1;
    {
        const int64_t k = (int64_t)bio_kin_count(w, c, BT_KIN(w->tr[i]));
        const int64_t km = k > 0 ? k - 1 : 0;
        const uint32_t p = (uint32_t)((int64_t)P->kinpath_q32 * km / (km + 20));
        if (bio_hash((int32_t)uid, (int32_t)w->step, BH_KINP, w->salt) < p) dead = 1;
    }
    if (dead) {
        const int64_t body = C + S + (pred_on ? gutv : (int64_t)w->gut[i]);
        dF[BF_DET] += body; dF[BF_NH4] += N; dF[BF_PO4] += Pa;
        BIO_ACC(&acc[BACC_DEATH_C], body);
        C = 0; S = 0; N = 0; Pa = 0; E = 0;
        w->gut[i] = 0;
        gutv = 0;
        fl = BA_DEAD | (fl & BA_HUNT);      /* an attacker that dies still has to be fed in k_feed (to the detritus) */
    } else if (!P->grow_off && C >= mr && E >= bio_repro_e(w, gb) && bio_lane(w->out[i], 3) > 0) {
        const int32_t offs = bio_clampi(gb[BG_OFFS], 16, 240);
        if (((E * offs) >> 8) > P->copy_cost) {
            int32_t tx, tz;
            bio_birth_pos(w, i, gb, &tx, &tz);
            const int32_t tc = bio_cell_at(w, tx, tz);
            if (((pred_on && bio_mover(w->tr[i], gb[BG_MOTOR])) ? w->nmot[tc] : w->cnt[tc]) < BIO_CELL_HARDCAP) {
                BIO_ACC(&acc[BACC_WANT], 1);
                if ((bio_hash((int32_t)uid, (int32_t)w->step, BH_BIRTH, w->salt) & 0xFFFFu) < (uint32_t)w->G->birth_q16) fl |= BA_BIRTH;
            }
        }
    }
    for (int f = 0; f < BF_N; ++f) if (dF[f]) BIO_ADD64(&w->fld[f * n + c], dF[f]);
    w->C[i] = (int32_t)C; w->S[i] = (int32_t)S;
    w->E[i] = (int32_t)E; w->N[i] = (int32_t)N; w->Pp[i] = (int32_t)Pa; w->W[i] = W;
    if (pred_on) w->gut[i] = (int32_t)gutv;
    w->age[i] = age;
    w->flags[i] = fl;
}

/* ==== k_feed (BIO_FEAT_PRED): an attacker collects what its prey gave up =========
 * Runs after k_metab. claim[i] names the prey whose offer stood (the lowest
 * key); that prey left the carbon it lost in its BR_BITE slot and a word
 * about itself in its BR_PRED slot. Everything written here is the attacker's
 * own state or a commutative add, so the order of threads does not matter. */
BIO_FN void bio_agent_feed(const BioWorld* w, int32_t i, int64_t* acc) {
    const uint32_t fl = w->flags[i];
    if (!(fl & BA_HUNT)) return;
    w->flags[i] = fl & ~BA_HUNT;
    const uint64_t ck = w->claim[i];
    if (ck == ~(uint64_t)0) return;
    const int64_t cap = w->cap;
    const uint32_t prey = (uint32_t)ck;
    int64_t b = w->req[(int64_t)BR_BITE * cap + prey];
    if (b <= 0) return;
    const int64_t det = (int64_t)BF_DET * w->n_cells + w->cell[i];
    if (!(fl & BA_ALIVE)) {                     /* the attacker died this step: its meal joins its remains */
        BIO_ADD64(&w->fld[det], b);
        return;
    }
    const int64_t room = 2000000000 - (int64_t)w->gut[i];
    if (b > room) { BIO_ADD64(&w->fld[det], b - room); b = room; }
    w->gut[i] += (int32_t)b;
    BIO_ACC(&acc[BACC_X(BACC2_PRED_C)], b);
    /* diet record: trophic level = moving average of (prey's level + 1), where a meal
     * of its own kind (same kin bucket) does not count: cannibals would otherwise
     * climb without end; share of meals that were land producers */
    const uint32_t info = (uint32_t)w->req[(int64_t)BR_PRED * cap + prey];
    const uint32_t e = w->eco[i];
    uint32_t tl = BE_TROPH(e);
    if (((info >> 10) & 15u) != BT_KIN(w->tr[i])) tl = (3u * tl + (info & 255u) + 16u + 2u) >> 2;
    if (tl > 255u) tl = 255u;
    const uint32_t lp = (3u * BE_LANDPROD(e) + ((info & 768u) == 768u ? 255u : 0u) + 2u) >> 2;
    w->eco[i] = (e & 0xFF00FF00u) | tl | (lp << 16);
}

/* ==== mutation (spec 3.6) ========================================================= */
/* mutates a genome in place; cuid is the child's uid */
BIO_FN void bio_mutate_in_place(const BioWorld* w, BioGenome* dst, uint32_t cuid);
BIO_FN void bio_mutate(const BioWorld* w, const BioGenome* src, BioGenome* dst, uint32_t cuid) {
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* d = (uint8_t*)dst;
    for (int k = 0; k < BIO_GENOME_BYTES; ++k) d[k] = s[k];
    bio_mutate_in_place(w, dst, cuid);
}
/* sexual reproduction (BIO_FEAT_SEX): the child takes each 16-byte block of the
 * genome from one parent or the other by hash bits. A block is 16 body genes,
 * one hidden neuron's input weights, two neurons' recurrent weights, ... */
BIO_FN void bio_cross(const BioWorld* w, const BioGenome* a, const BioGenome* b, BioGenome* dst, uint32_t cuid) {
    const uint8_t* pa = (const uint8_t*)a;
    const uint8_t* pb = (const uint8_t*)b;
    uint8_t* d = (uint8_t*)dst;
    const uint32_t h = bio_hash((int32_t)cuid, 0, BH_CROSS, w->salt);
    for (int blk = 0; blk < BIO_GENOME_BYTES / 16; ++blk) {
        const uint8_t* s = ((h >> blk) & 1u) ? pb : pa;
        for (int k = 0; k < 16; ++k) d[16 * blk + k] = s[16 * blk + k];
    }
}
BIO_FN int32_t bio_popc16(uint32_t v) { return bio_popc8(v) + bio_popc8(v >> 8); }
BIO_FN void bio_mutate_in_place(const BioWorld* w, BioGenome* dst, uint32_t cuid) {
    uint8_t* d = (uint8_t*)dst;
    if (w->P.mut_off) return;
    const uint32_t mask = (1u << (6 + dst->body[BG_MUT] / 32)) - 1u;
    for (int k = 0; k < BIO_MUT_BYTES; ++k) {
        if (bio_hash((int32_t)cuid, k, BH_MUT, w->salt) & mask) continue;
        const uint32_t h = bio_hash((int32_t)cuid, k, BH_MUT + 64, w->salt);
        const int32_t dl = 1 + (int32_t)((h >> 4) & 7u);
        if (k < 64) {
            int32_t v = d[k];
            if (h & 15u) v = bio_clampi(v + ((h & 128u) ? -dl : dl), 0, 255);
            else v = (int32_t)((h >> 8) & 255u);
            d[k] = (uint8_t)v;
        } else {
            int32_t v = (int32_t)(int8_t)d[k];
            if (h & 15u) v = bio_clampi(v + ((h & 128u) ? -dl : dl), -127, 127);
            else v = bio_clampi((int32_t)(int8_t)(uint8_t)(h >> 8), -127, 127);
            d[k] = (uint8_t)(int8_t)v;
        }
    }
    const uint32_t ht = bio_hash((int32_t)cuid, 0, BH_TAG, w->salt);
    if ((ht & 63u) == 0u) dst->mate_tag = (uint16_t)(dst->mate_tag ^ (1u << ((ht >> 6) & 15u)));
    const uint32_t hm = bio_hash((int32_t)cuid, 1, BH_MACRO, w->salt);
    if ((hm & 2047u) == 0u) {
        const uint32_t kind = (hm >> 11) % 3u;
        if (kind == 0u) {            /* duplicate a hidden neuron over another */
            const int32_t a = (int32_t)((hm >> 13) & 7u), b = (int32_t)((hm >> 16) & 7u);
            if (a != b) {
                for (int k = 0; k < BIO_IN; ++k) dst->w_ih[b][k] = dst->w_ih[a][k];
                for (int k = 0; k < BIO_H; ++k) dst->w_hh[b][k] = dst->w_hh[a][k];
                dst->b_h[b] = dst->b_h[a];
                for (int o = 0; o < BIO_OUT; ++o) dst->w_ho[o][b] = dst->w_ho[o][a];
            }
        } else if (kind == 1u) {     /* toggle a neuron or a plastic row */
            const int32_t gene = ((hm >> 13) & 1u) ? BG_NEURONS : BG_PLASTIC;
            dst->body[gene] = (uint8_t)(dst->body[gene] ^ (1u << ((hm >> 14) & 7u)));
        } else {                     /* jump one enzyme gene by +-64 */
            const int32_t e = (int32_t)((hm >> 13) & 7u);
            dst->body[e] = (uint8_t)bio_clampi(dst->body[e] + (((hm >> 16) & 1u) ? -64 : 64), 0, 255);
        }
    }
}

/* ==== k_birth: one thread per reproducing parent ================================== */
BIO_FN void bio_agent_birth(const BioWorld* w, int32_t i) {
    const uint32_t fl = w->flags[i];
    if (!(fl & BA_BIRTH)) return;
    w->flags[i] = fl & ~BA_BIRTH;
    const uint32_t b = (w->pre_bd[i] & 0xFFFFu) + w->boff[(i / BIO_SCAN_BLOCK) * 3 + 0];
    if (b >= w->k_births) return;
    const uint32_t j = w->n_slots + b;
    const uint32_t gs = w->fstack[w->top - 1u - b];
    const uint32_t cuid = w->uid_next + b;
    const BioGenome* pg = &w->gen[w->gslot[i]];
    BioGenome* cg = &w->gen[gs];
    const int32_t offs = bio_clampi(pg->body[BG_OFFS], 16, 240);
    int32_t tx, tz;
    bio_birth_pos(w, i, pg->body, &tx, &tz);
    const int32_t cC = (int32_t)(((int64_t)w->C[i] * offs) >> 8);
    const int32_t cN = (int32_t)(((int64_t)w->N[i] * offs) >> 8);
    const int32_t cP = (int32_t)(((int64_t)w->Pp[i] * offs) >> 8);
    const int32_t cE = (int32_t)(((int64_t)w->E[i] * offs) >> 8);
    w->C[i] -= cC; w->N[i] -= cN; w->Pp[i] -= cP; w->E[i] -= cE;
    int sexual = 0;
    if ((w->P.feat & BIO_FEAT_SEX) && pg->body[BG_SEX] >= 128) {
        /* a sexual parent crosses with the first compatible mate among its cell's
         * offers of this step: alive, not itself, mate tags within 3 bits (spec 3.6).
         * Without one it reproduces alone. Offers were posted before this step's
         * births and a live agent's genome never changes, so this reads stable data. */
        const int32_t pc = (int32_t)w->cell[i];
        const uint32_t h0 = bio_hash((int32_t)cuid, 0, BH_MATE, w->salt);
        for (uint32_t k = 0; k < BIO_MSLOTS && !sexual; ++k) {
            const uint32_t key = w->mslot[(int64_t)((h0 + k) & (BIO_MSLOTS - 1)) * w->n_cells + pc];
            if (!key) continue;
            const int32_t m = bio_slot_agent(w, key, pc, BH_MMASK);
            if (m == i || (uint32_t)m >= w->n_slots || !(w->flags[m] & BA_ALIVE)) continue;
            const BioGenome* mg = &w->gen[w->gslot[m]];
            if (bio_popc16((uint32_t)(pg->mate_tag ^ mg->mate_tag)) > 3) continue;
            bio_cross(w, pg, mg, cg, cuid);
            bio_mutate_in_place(w, cg, cuid);
            sexual = 1;
        }
    }
    if (sexual) BIO_ADD64(&w->G->acc2[BACC2_SEXB], 1);
    else bio_mutate(w, pg, cg, cuid);
    w->x[j] = tx; w->z[j] = tz;
    w->cell[j] = (uint32_t)bio_cell_at(w, tx, tz);
    w->C[j] = cC; w->S[j] = 0; w->E[j] = cE - w->P.copy_cost; w->N[j] = cN; w->Pp[j] = cP;
    w->gut[j] = 0; w->W[j] = w->W[i];
    w->age[j] = 0u;
    w->flags[j] = BA_ALIVE | (fl & (BA_WET | BA_LAND));
    w->rbase[j] = 0; w->elast[j] = cE - w->P.copy_cost;
    w->gslot[j] = gs; w->uid[j] = cuid;
    w->tr[j] = bio_traits(cg);
    w->hid[j] = 0; w->out[j] = bio_out_bias(w, cg);
    for (int o = 0; o < BIO_OUT; ++o) w->dW[(int64_t)o * w->cap + j] = 0;
    w->lineage[j] = w->lineage[i]; w->parent[j] = w->uid[i]; w->bstep[j] = (uint32_t)w->step;
    w->tr2[j] = bio_traits2(cg);
    w->expl[j] = 0;
    uint32_t ce = 0u;
    if (w->P.feat & BIO_FEAT_PRED) {
        /* the diet record starts as the parent's; gate G4 counts generations in a row
         * born to a land animal (motile, on dry ground, living on land producers) */
        const uint32_t pe = w->eco[i];
        const int animal = (fl & BA_LAND) && pg->body[BG_MOTOR] > 128 && BE_GUTSHARE(pe) > 128u && BE_LANDPROD(pe) > 128u &&
                           bio_creature(pg->body, pe);
        const uint32_t gen = animal ? (BE_LANDGEN(pe) < 255u ? BE_LANDGEN(pe) + 1u : 255u) : 0u;
        ce = BE_MAKE(BE_TROPH(pe), BE_GUTSHARE(pe), BE_LANDPROD(pe), gen);
    }
    w->eco[j] = ce;
}

/* ==== k_free: dead agents push their genome slot (prefix order). Runs before
 * k_birth; the host then raises top by the number of deaths. ======================= */
BIO_FN void bio_agent_free(const BioWorld* w, int32_t i) {
    if (!(w->flags[i] & BA_DEAD)) return;
    const uint32_t d = (w->pre_bd[i] >> 16) + w->boff[(i / BIO_SCAN_BLOCK) * 3 + 1];
    w->fstack[w->top + d] = w->gslot[i];
    w->flags[i] = 0u;
}

/* scan inputs: births (low half) and deaths (high half) */
BIO_FN uint32_t bio_scan_bits(uint32_t fl) {
    return ((fl & BA_BIRTH) ? 1u : 0u) | ((fl & BA_DEAD) ? 0x10000u : 0u);
}

/* ==== statistics contributions (exact integer sums) =============================== */
BIO_FN void bio_agent_stats(const BioWorld* w, int32_t i, int64_t* st) {
    const uint32_t fl = w->flags[i];
    if (!(fl & BA_ALIVE)) return;
    const uint8_t* gb = w->gen[w->gslot[i]].body;
    const int32_t c = (int32_t)w->cell[i];
    const int64_t body = (int64_t)w->C[i] + w->S[i] + w->gut[i];
    BIO_ACC(&st[BS_LIVE], 1);
    if (gb[BG_MOTOR] > 0) BIO_ACC(&st[BS_MOTILE], 1);
    if (gb[BG_HOLDFAST] > 64 && (gb[BG_E_ANOX] >= 32 || gb[BG_E_OXY] >= 32) && gb[BG_PIGMENT] > 0) BIO_ACC(&st[BS_SPHOTO], 1);
    BIO_ACC(&st[BS_CMAR + bio_realm(w, c)], body);
    BIO_ACC(&st[BS_C], body);
    BIO_ACC(&st[BS_N], w->N[i]);
    BIO_ACC(&st[BS_P], w->Pp[i]);
    if (fl & BA_THOUGHT) BIO_ACC(&st[BS_THINK], 1);
    int64_t adw = 0;
    for (int o = 0; o < BIO_OUT; ++o) {
        const uint64_t row = w->dW[(int64_t)o * w->cap + i];
        if (!row) continue;
        for (int k = 0; k < 8; ++k) { const int32_t v = bio_lane(row, k); adw += v < 0 ? -v : v; }
    }
    if (adw) BIO_ACC(&st[BS_ABSDW], adw);
    if (w->C[i] < 0 || w->S[i] < 0 || w->E[i] < 0 || w->N[i] < 0 || w->Pp[i] < 0 || w->W[i] < 0 || w->gut[i] < 0) BIO_ACC(&st[BS_NEGA], 1);
    if (w->P.feat & BIO_FEAT_PRED) {
        const uint32_t e = w->eco[i];
        if (BE_GUTSHARE(e) > 128u) {            /* lives mostly on prey */
            BIO_ACC(&st[BS_CONSUMER], 1);
            BIO_ACC(&st[BS_TROPH], BE_TROPH(e));
            if (gb[BG_MOTOR] > 128 && bio_realm(w, c) == 2 && BE_LANDPROD(e) > 128u && BE_LANDGEN(e) >= 5u && bio_creature(gb, e))
                BIO_ACC(&st[BS_LANDANIM], 1);
        }
    }
    if ((w->P.feat & BIO_FEAT_SEX) && gb[BG_SEX] >= 128) BIO_ACC(&st[BS_SEXUAL], 1);
    if (body > 0 && bio_creature(gb, (w->P.feat & BIO_FEAT_PRED) ? w->eco[i] : 0u)) BIO_ACC(&st[BS_CREATURE], 1);
}
/* gene k of agent i (0 when not alive), for the mean gene values */
BIO_FN int32_t bio_agent_gene(const BioWorld* w, int32_t i, int32_t k) {
    return (w->flags[i] & BA_ALIVE) ? w->gen[w->gslot[i]].body[k] : 0;
}
/* per-cell statistics into st[0..BS_CELL_N): fields, vegetated dry cell, O2 cell */
#define BS_CELL_N (BF_N + 3)
BIO_FN void bio_cell_stats(const BioWorld* w, int32_t c, int64_t* st) {
    const int64_t n = w->n_cells;
    for (int f = 0; f < BF_N; ++f) BIO_ACC(&st[f], w->fld[f * n + c]);
    if (bio_realm(w, c) == 2) {
        int64_t a = 0;
        for (int b = 0; b < BIO_LAI_BANDS; ++b) a += w->lai[b * n + c];
        if (a >= 32768) BIO_ACC(&st[BF_N], 1);       /* cover >= 32 of 255 */
    }
    if (w->fld[BF_O2 * n + c] > bio_volume(w, c)) BIO_ACC(&st[BF_N + 1], 1);      /* > 1 uM */
    int neg = 0;
    for (int f = 0; f < BF_N; ++f) neg |= w->fld[f * n + c] < 0;
    if (neg) BIO_ACC(&st[BF_N + 2], 1);
}

#endif

# GME engine upgrade: smooth density surfaces and spectral light. Final implementation spec

**Base design:** integration-delivery.
**Taken from gpu-realtime:** geometry chosen per ray type, the frame pipeline (wavefront kernels, stream priorities, TAAU) and the measurement harness.
**Taken from physical-light:** the atmosphere model, optics, far field, pigment model and photon energetics.

**Phases:**
- **Phase A, engine only (M1–M6):** can start now. It never touches a file that `gme_genesis` or `gme_biotest` compiles.
- **Phases B and C, life side (M7–M8):** start after the integration builder's work has landed and the islands run stably.

**Names:** every file, function and struct named below was checked against the code on 2026-11-18. That includes `k_generate`, `k_commit`, `gpu_submit`, `build_batch`, `finish_batch`, `evict`, `scan_fn`, `cpu_gen_chunk`, `world_validate`, `ref_hash`, `world_get_scene`, `RenderScene`, `RenderCamera`, `KParams`, `trace`, `k_render`, `build_sky`, `bio_light`, `bio_globals_update`, `bio_sim_vegetation`, `BioParams.reserved[16]`, `BioTables.expn_q16`, `GMEBIO02`, `BIOMAP_VERSION` and `-DGME_BUILD_ENGINE=OFF` in `cluster/bio_launch.ps1` / `launch.ps1`.

**Where the numbers come from:**
- Values marked *measured* were produced for this spec on the current planet (system 59). The programs include the project's `worldgen.h` read-only. They are in `<scratch>\final\`: `surf_final.c`, `devstat.c`, `planet_consts.py`, `haze_est.py`, `pigment.py` and `water.py`.
- No project file was changed.
- No GPU timing was taken, because the 3060 is shared with the lab island. All ms values are estimates; M1 and M4 replace them with `--perf` measurements.

---

## 0. Decisions

| Topic | Decision |
|---|---|
| Truth | The uint8 material grid remains the gameplay and life truth. Phase A leaves `gme_height`, `gme_material`, `gme_voxel`, `gme_slot`, `gme_env_bin`, `biomap.*` and `planetfile.*` unedited. So `ref_hash`, every voxel check and the island env CRCs cannot change. |
| Smooth terrain | **T(x,z)** is the generator's own height without its intermediate floors, mean-shifted to h+1, plus ±0.25 m micro-relief. It is then **clamped into [h, h+2)** and **coast-locked**: a column renders wet iff its voxel column is wet. Storage is 2 B per column (8-bit h offset + int8 deviation in 1/128 m), in one 3,456 B tile per chunk column. |
| Smooth shapes | **Trees and shrubs:** analytic ellipsoid crown + cylinder trunk built from `GmeSlot`. They equal the voxel crown at every voxel centre (self-test D2) and are shifted onto T. **Kelp:** per-column capsules. **Caves:** voxel truth (blocky) until the optional M6. |
| Traversal | A 2.5D *column tracer*: 32 m tile → 8 m block (= one flora slot) → 1 m dual cell, with an exact ray/bilinear-patch quadratic. There is no per-voxel DDA above ground. Geometry is chosen per ray type: fine, 4 m LOD, or far field. |
| Light | 8 bands, 380–740 nm, 45 nm wide. Components: Planck star from `star_teff`; per-gas Rayleigh with King factors; CH4 absorption and a tholin haze calibrated to the bio's haze factor; Hillaire LUTs validated against Monte Carlo; spectral water; GGX + energy-compensated diffuse. The sun enters only through next-event estimation (NEE). Indirect light: 1 bounce at half resolution + a world-space radiance cache, SVGF denoising. Output: TAA/TAAU, histogram exposure, CAT16, AgX. |
| Buffers | Spectral data is kept where products matter: per-pixel shading, and half-resolution indirect irradiance in 8 bands. Every full-resolution buffer is XYZ/RGB or packed material ids. |
| Life, now | Organism colour = reflectance of its pigment (Soret + Q band, from the hue bin already exported). `gme --export-spectrum` writes an integer spectral light file for the bio. |
| Life, later | The bio core reads that file behind `BioParams.reserved[0]`, integer only; with the flag off, results are bit-identical. Morphology genes → biomap v2 → `flora.h` v2 shapes, which become voxel truth as well. |
| Fallback | `--quality classic` runs today's `k_render`, source untouched and pixel-identical. |

### 0.1 Flaws raised by the judges, and their fixes

| Raised against | Flaw | Fix in this spec |
|---|---|---|
| integration-delivery | T decoupled from the voxels (RMS 0.60 m, max 2.36 m), so the coast drifts | The [h, h+2) clamp *is* the storage format: ±1 m by construction. The coast lock makes wet/dry at column centres equal the voxel truth. *Measured:* 0 coast mismatches in 1,083,579 near-sea columns (unclamped: 56,855 = 5.2%). Smoothness stays at 0.077 m (unclamped 0.042, voxels 0.435). |
| integration-delivery | The bound \|mean dev\| ≤ 0.05 m is razor-thin | The bound is now structural. D1 gates the per-window mean at ≤ 0.10 m (*measured:* −0.034 … +0.004). |
| integration-delivery | No far field: the world edge is visible, since haze removes only ~32% at 3.4 km | M3 adds a far-field clipmap: 16 m spacing to ±8 km, 64 m to ±32 km (transmittance at 32 km ≈ 0.03), plus long-range shadows. M2 uses an edge fade until then. |
| integration-delivery | Weak pigment science and plain optics | Soret + Q pigments with leaf ρ/τ. GGX with height-correlated Smith, energy-compensated diffuse, exact Fresnel, Lekner–Dorf wet darkening, Snell window/TIR, caustics. |
| integration-delivery | 8-band fp16 buffers at full resolution | 8 bands only for half-res indirect irradiance (~15 MB, <0.1 ms of bandwidth). Full-res buffers are RGB/XYZ. |
| integration-delivery | `__reduce_or_sync` is 32-bit | The column tracer needs no brick masks, so they are off the critical path. If brick masks are built later for the classic renderer, use two 32-bit reductions. |
| integration-delivery | Fixed 512 MB reserve | Tiles are counted inside `world_create`'s budget. The renderer allocates its fixed buffers *before* `world_create`, so pool sizing sees the real free VRAM. No magic constant. |
| integration-delivery | M0 shows nothing on screen | M1 is the first milestone and puts smooth terrain on screen. |
| gpu-realtime | Fatal: the q8 sub-voxel height clamps on 88–90% of columns; any sign-locked surface terraces at 2 m steps (24.8% of non-zero steps, *measured*) | Not used for terrain. Sign-locked density is used only for caves (M6), where it is valid. |
| gpu-realtime | No smoothness gate | M1 gates on CPU smoothness statistics (D1) and on the on-screen share of axis-aligned normals. |
| gpu-realtime | `τ_h = −ln(haze)` treats a global transmission as direct-beam extinction | τ_h is solved so that the PAR-weighted *global* transmission equals `haze_q12/4096`. |
| gpu-realtime | Single-Gaussian pigments (no Soret band) | Soret + Q model. |
| physical-light | The B5 band-locked surface terraces gentle ground, and the corner spec is inconsistent | Not used. |
| physical-light | `GmeTreeRec` int16 coordinates overflow at the islands (14–111 km from origin) | Tiles store `GmeSlot` (int32) exactly as `flora.h` produces it. |
| physical-light | `τ_L = 0.5ω − 0.03` goes negative | ρ_L = 0.04 + 0.44(1−A), τ_L = 0.42(1−A). |
| physical-light | The bio's absorptance is linear in PIG, the engine's is exponential | The bio uses the same 10^(−OD) law through the existing `expn_q16` table. |
| physical-light | Mega-milestones, an optimistic ~11 ms, per-pixel free-flight noise in crowns | Six engine milestones, each with its own gates. Budgets are ranges and get measured. For primary rays crowns are opaque clumped surfaces; LAD media are used only for shadow and GI transmittance. |
| all three | Constants taken from the archived seed-2 planet | Every expected value is computed from the loaded planet at startup and printed. §A lists the current planet's values for reference only, not as gates. |
| gpu-realtime, physical-light | A "within 5% of bio i0" check collides with `par_q12 = 0.45` (the physical PAR share is 33.6%) | Gate on *relative* transmission. The absolute ratio (0.747) is printed as a known bio-side discrepancy, fixed in phase B. |

---

## 1. Overview and data flow

```
home.planet ─ planet_load ─► HomePlanet ─ spectral_build_atmo (host, double, 5 nm) ─► SpectralAtmo ─ renderer_set_atmo ─► LUTs
home.biomap ─ biomap_load ─► GmeBioView (unchanged upload in world_create)

world_update ─ build_batch: chunk jobs (unchanged) + tile jobs (NEW, ≤1 per chunk column, ≤1 in flight per tile cell)
           └─ gpu_submit on the generation stream (LOW priority):
                H2D jobs + tile jobs ─ k_col_tiles ─► d_tiles[page_n²] (GmeColTile, NEW)
                k_meta_init ─ k_generate ─ k_commit ─► d_pool + mapped staging         (unchanged)
                D2H meta ─ event
           └─ finish_batch: host mirror, page[] entries, tile_tag[] published (NEW), dirty lists (M4)
world_get_scene ─► RenderScene {page, pool, tiles, tile_tag, planet, seed, sea_level, bio (device cells), dirty info}
renderer_draw on the render stream (HIGH priority):
  upload/scatter dirty page + tile entries · k_far_update (M3) · k_sky_view / k_aerial (M2)
  k_primary   column tracer → G-buffer (t, normal, material pack, motion, water info)
  k_sun       disc-jittered shadow ray → visibility V (scalar)
  k_gi_trace → k_gi_nee → k_gi_shade (half res, 8 bands, M4) → E_ind; radiance cache update
  k_reflect   water / ice / wet (M4)
  k_temporal + k_atrous×3 + k_upsample (E_ind) · k_vis_filter (V)
  k_compose   spectral shading + water volume + aerial perspective + sky → XYZ → CAT16 → linear sRGB
  k_lum_hist → exposure · k_taa (TAAU) · k_tonemap (AgX, sRGB OETF, dither, crosshair) → BGRA
  async D2H (2 pinned frames, M4) ─► window_present (GDI, unchanged) / write_png
```

### 1.1 Files

| File | Status | Content | First used |
|---|---|---|---|
| `src/density.h` | **new** | GME_HD, integer code: `gme_surface_pair`, `gme_tile_dev_q7`, `gme_surface_q8`, `GmeColTile`, `gme_col_tile_build_column` / `_finish`, material-lookup helpers, flora inside-tests (integer + float twins), D1 statistics | M1 |
| `src/trace.cuh` | **new** | Device code: column tracer, patch solve, water plane, flora intersectors, `trace_voxels` (a *copy* of the classic `trace`, which itself stays untouched), material blend; LOD and far field from M3 | M1 |
| `src/spectral.h/.c` | **new** | Host code in double precision: Planck, CIE (1 nm table + Wyman fit check), gas data, atmosphere/LUT build, haze calibration, Monte Carlo sky reference, material and pigment spectra, water tables, self-tests, spectrum export | M2/M5 |
| `src/atmo.cu` | **new** | `k_sky_view`, `k_aerial`, LUT samplers | M2 |
| `src/gi.cu` | **new** | GI kernels, radiance hash cache, SVGF, TAA/TAAU | M4 |
| `src/spectrum_file.h` | **new** | `GMESPEC1` contract: integer-only, header-only static inline load/save, like `biomap.h` | M5 |
| `src/gpu.cu/.h` | changed | Tile pool, `k_col_tiles`, `gpu_read_tile`, `gpu_submit(..., ntiles, ...)`, low-priority generation streams | M1/M4 |
| `src/world.c/.h` | changed | Tile table, tile jobs, publication/eviction, `cpu_col_tile`, validation fields, `dens_hash`, `world_surface_q8`, dirty lists | M1/M4 |
| `src/render.cu/.h` | changed | Scene plumbing, `k_smooth` (M1), then the pass pipeline (M2+). `k_render`/`trace`/`sky`/`albedo` are kept byte-for-byte for classic mode | M1+ |
| `src/main.c` | changed | Options, keys, `--view`, `--perf`, `--metrics`, `--selftest`, `--export-spectrum`; the renderer is created before the world | M1+ |
| `CMakeLists.txt` | changed | Only the source list of `add_executable(gme …)`: `spectral.c`, `atmo.cu` (M2), `gi.cu` (M4). M1 needs no CMake change, because its new files are headers | M2 |

**Untouched in phase A:**
- `worldgen.h`, `flora.h`, `biomap.h/.c`, `planetfile.h/.c`
- `image.*`, `platform.*`, `jobs.*`, `rtcuda.*`, `window_win32.c`
- `src/bio/*`, `src/genesis/*`, `cluster/*`, `docs/biosphere-spec.md`

### 1.2 Structs (density.h / gpu.h / world.h / render.h)

```c
/* ---- src/density.h (GME_HD; includes worldgen.h + flora.h, edits neither; never included by gme_genesis) ---- */
#define GME_TILE_APRON 2
#define GME_TILE_N     36          /* columns x0-2 .. x0+33: bilinear corners + central-difference normals */
#define GME_DEV_UNIT   128         /* deviation T-(h+1) in 1/128 m, int8 => T in [h, h+2) */

typedef struct GmeColTile {        /* one per chunk column (all layers); 3,456 B */
    uint32_t tag;                  /* gme_page_tag(cx, cz) (informational; publication is host-side) */
    int32_t  cx, cz;
    int16_t  h_base;               /* min gme_height over the 36x36 columns */
    uint16_t flags;                /* 1 any wet column, 2 any tree/shrub, 4 any kelp, 0x8000 saturated (must stay 0) */
    int16_t  top, bot;             /* whole tile: max(T, flora top, water plane) ceil / min T floor, metres */
    int16_t  blk_top[16], blk_bot[16];   /* per 8 m block (= flora slot), row-major z*4+x, incl. 1-column overlap */
    GmeSlot  slot[16];             /* gme_slot() of the 4x4 slots, exactly as flora.h computes it (int32 fields) */
    uint16_t col[GME_TILE_N * GME_TILE_N]; /* bits 0-7: h - h_base; bits 8-15: int8 dev q7 */
    int16_t  lod4[9 * 9];          /* M3: T box-averaged over 4x4 columns at the 4 m lattice x0+4i, q4 m; 0 in M1 */
    uint8_t  pad[106];
} GmeColTile;                      /* GME_STATIC_ASSERT(sizeof(GmeColTile) == 3456) */

typedef struct GmeTileJob { int32_t cx, cz; uint32_t index, pad; } GmeTileJob;   /* index = gme_page_index(cx,0,cz,page_n,1) */

GME_HD void    gme_surface_pair(int32_t x, int32_t z, uint32_t seed, const GmePlanet* p, int32_t* h, int32_t* t_q8);
GME_HD int32_t gme_tile_dev_q7(int32_t t_q8, int32_t h, const GmePlanet* p);
GME_HD int32_t gme_surface_q8(int32_t x, int32_t z, uint32_t seed, const GmePlanet* p); /* what is drawn / walked */
GME_HD void    gme_col_tile_build_column(int32_t x, int32_t z, uint32_t seed, const GmePlanet* p, int32_t* h, int32_t* dev);
GME_HD void    gme_col_tile_finish(GmeColTile* t, const int32_t* h36, const int32_t* dev36,
                                   uint32_t seed, const GmePlanet* p, const GmeBioView* b); /* bounds, slots, lod4 */
GME_HD int     gme_crown_inside_f(float dx, float dy, float dz, int32_t trad, int32_t tvr);   /* D2 twins */
GME_HD int     gme_trunk_inside_f(float dx, float y, float dz, const GmeSlot* s);
GME_HD int32_t gme_kelp_len(int32_t x, int32_t z, int32_t h, uint32_t cell, uint32_t seed, const GmePlanet* p); /* 0 = none */

/* ---- world.h additions ---- */
typedef struct ValidateReport {            /* existing fields unchanged, appended: */
    /* ... */
    uint32_t tiles_checked;                /* published tiles compared GPU vs cpu_col_tile */
    uint64_t tile_mismatch;                /* differing bytes */
    uint32_t tile_h_bad, tile_coast_bad, tile_sat;
    double   dev_mean, dev_rms, dev_max, clamp_share;
    uint64_t dens_hash;                    /* FNV-1a 64 over cpu_col_tile of the WORLD_REF_COLS^2 reference columns */
} ValidateReport;
int world_surface_q8(const World* w, int32_t x, int32_t z, int32_t* t_q8);   /* analytic, = rendered surface */

/* ---- render.h additions ---- */
typedef struct RenderScene {               /* existing fields unchanged, appended: */
    /* ... */
    const GmeColTile* tiles;               /* device, page_n*page_n */
    const uint32_t*   tile_tag;            /* host copy, uploaded with the page table */
    int       tile_dirty;
    GmePlanet planet; uint32_t seed;       /* analytic fallback + far field */
    GmeBioView bio;                        /* cells = device pointer (gpu_world_biomap) */
    const uint32_t* page_dirty_idx; uint32_t page_dirty_n;   /* M4 scatter lists */
    const uint32_t* tile_dirty_idx; uint32_t tile_dirty_n;
} RenderScene;
typedef struct RenderCamera {              /* existing fields unchanged; main.c switches to designated initializers */
    int32_t cx, cy, cz; float lx, ly, lz; float yaw, pitch, fov_y; int shadows; const RenderSky* sky;
    int quality;        /* RQ_CLASSIC, RQ_LOW, RQ_MEDIUM, RQ_HIGH, RQ_REF */
    int debug;          /* RD_NONE, RD_NORMALS, RD_ALBEDO, RD_GI, RD_DEV, RD_BAND0 + b */
    uint32_t frame;     /* jitter / noise index */
    float sun_dir[3];   /* world space, from --time (M2); default = build_sky's direction */
    float ev;           /* fixed EV100, NAN = auto */
} RenderCamera;
typedef struct RendererConfig { int max_w, max_h, quality; } RendererConfig;
Renderer* renderer_create(const RendererConfig* cfg);       /* allocates every fixed buffer now */
int       renderer_set_atmo(Renderer* r, const SpectralAtmo* a);   /* M2 */
void      renderer_metrics(const Renderer* r, RenderMetrics* out); /* --metrics counters */
void      renderer_perf(const Renderer* r, RenderPerf* out);       /* --perf p50/p95 per kernel */
```

---

## 2. Density representation

### 2.1 What is computed

**1. Continuous height T.**
- The generator floors three times: `(cont*80)>>15`, `(detail*amp)>>16` with `amp` itself floored, then `*relief_q8>>8`. With relief 319 that makes 24.8% of the non-zero h steps 2 m tall (*measured*).
- So any surface locked to the top voxel must keep ledges of 45° or more. T is instead that same expression *without the floors*, in q8. `gme_surface_pair` evaluates the noise once and returns both values:

```c
GME_HD void gme_surface_pair(int32_t x, int32_t z, uint32_t seed, const GmePlanet* p, int32_t* h_out, int32_t* t_q8) {
    const int32_t cont = gme_noise2(x, z, 11, seed ^ 0xA511E9B3u), det = gme_fbm2(x, z, 8, 5, seed);
    const int32_t mount = cont > 0 ? cont : 0, amp = 20 + ((mount * 110) >> 15);
    int32_t h = ((((cont * 80) >> 15) + (int32_t)(((int64_t)det * amp) >> 16)) * p->relief_q8) >> 8;  /* == gme_height */
    h = h > GME_HEIGHT_MAX ? GME_HEIGHT_MAX : (h < -GME_HEIGHT_MAX ? -GME_HEIGHT_MAX : h);
    int64_t t = (((int64_t)cont * 80) >> 7) + (((int64_t)det * (20 * 32768 + (int64_t)mount * 110)) >> 23); /* a+b, q8 */
    t = ((t * p->relief_q8) >> 8) - (p->relief_q8 - 128);            /* mean(T) = h + 1 */
    t += gme_noise2(x, z, 2, seed ^ 0xD3A5u) >> 9;                   /* +-0.25 m micro relief */
    *h_out = h; *t_q8 = (int32_t)t;
}
GME_HD int32_t gme_tile_dev_q7(int32_t t_q8, int32_t h, const GmePlanet* p) {
    int32_t dev = (t_q8 - 256 * (h + 1) + 1) >> 1, lo = -128;        /* q7, rounded */
    if (h >= p->sea_level) { const int32_t c = 128 * (p->sea_level - h) + 1; if (c > lo) lo = c; }  /* coast lock */
    return dev < lo ? lo : (dev > 127 ? 127 : dev);                   /* T in [h, h+2) */
}
GME_HD int32_t gme_surface_q8(int32_t x, int32_t z, uint32_t seed, const GmePlanet* p) {
    int32_t h, t; gme_surface_pair(x, z, seed, p, &h, &t);
    return 256 * (h + 1) + 2 * gme_tile_dev_q7(t, h, p);
}
```

- `h` here is checked against `gme_height` on 25 M columns (*measured*, 0 differences), and again by `tile_h_bad` in every validation run.
- **Coast lock.**
  - A column is voxel-wet iff `h < sea_level`. The water plane is the top face of the sea-level voxel, `y_w = sea_level + 1`.
  - Dry columns with `h = sea` (T would otherwise go below y_w) get T ≥ y_w + 1/128.
  - Dry columns with `h = sea+1` get T ≥ y_w + 1/128.
  - Wet columns already satisfy T < h+2 ≤ y_w.
  - So wet/dry at every column centre equals the voxel truth, and the drawn shoreline (the bilinear contour) lies within one column of the voxel shoreline. Mats, beaches and kelp therefore appear exactly where the life sim put them.

**2. Caves.**
- Caves stay the voxel truth. `gme_material` makes caves only at depth > 6, so they are underground and reach the surface only through cliff mouths.
- Optional M6 adds a near-camera *sign-locked* cave density: sign from the voxel, magnitude from smoothed occupancy, so it is never more than 0.5 m from the truth.

**3. Flora shapes.**
- Analytic, from the tile's `GmeSlot`. The parameters are those of `gme_voxel`, so the analytic shapes match it exactly (§3.5, D2).

**4. Water surface.**
- Plane at `y_w`, present where the bilinear T < y_w (coast-locked as above). Gerstner wave *normals* only, no displacement.
- When `ocean_top_mat == MAT_ICE` the plane is an ice surface.

### 2.2 Encoding and precision

| Field | Encoding | Precision / range |
|---|---|---|
| deviation T−(h+1) | int8, 1/128 m | 7.8 mm; [−1, +0.992] m. Normal quantisation ≈ 0.22° |
| h | uint8 offset from `h_base` | exact. *Measured* max range inside one 36×36 tile: 30 m. Over 255 sets the saturation flag, which validation requires to be 0 |
| block/tile bounds | int16 metres (floor/ceil) | conservative |
| lod4 (M3) | int16, 1/16 m | box average of 16 T values (integer division) |

The CPU and GPU produce bit-identical tiles, because the code is integer only.

### 2.3 Memory cost and mitigation

Baseline from the real bench log:
- pool 6.47 GiB = 211,957 slots;
- `radius_max` 119, so `page_n = 2·119+8 = 246`;
- radius achieved 107 (106 with life);
- 5.57 non-uniform chunks per column.

| Option | Extra VRAM | View radius |
|---|---|---|
| 8-bit density per voxel in every slot | +100% | ×0.71 |
| 2 B/voxel SDF | +200% | ×0.58 |
| 4-bit density per voxel | +50% | ×0.82 |
| q7 density only in surface bricks (~20–28%) + index | +22–28% | ×0.88–0.90, plus a variable-size allocator |
| analytic at every ray step (6 noise evaluations × 4 columns per cell) | 0 | ×1.0, but +6–10 ms per frame |
| **chosen: column tiles (2 B/column + bounds + slots)** | 246² × 3,456 B = **209 MB = 3.0% of the pool** | **×0.985 (−1.5%)** |
| renderer fixed buffers, Medium at 1280×720 (§6.3) | ~110 MB | −0.8% (allocated first) |
| optional cave cache (M6, High) | 40 MB | −0.3% |

**Mitigation:**
- `world_create` computes the tile-table bytes from a first-pass `page_n` and subtracts them before dividing by `CHUNK_BYTES`. A smaller final `page_n` only over-reserves slightly.
- No host mirror of tiles is kept. Validation reads them back with `gpu_read_tile`, and the CPU computes T analytically.

### 2.4 Generation and streaming

**`k_col_tiles<<<ntiles, 256, 10.4 KB smem, stream>>>(jobs, tiles, seed, planet, bio)`**
- Runs on the batch stream before `k_meta_init`.
- Each thread builds ~5 of the 1,296 columns with `gme_col_tile_build_column`.
- A shared-memory reduction gives `h_base`, the block and tile bounds and `lod4`.
- 16 threads call `gme_slot`; the flora tops come from the slots:
  - tree/shrub: `tcy+tvr+2`;
  - kelp: `sea_level+1`;
  - plus the water plane where any column is wet.
- Cost: ~16 noise2 per column (with apron), against ~100 per column for the 16 `k_generate` passes. Target: generation ≥ 90% of baseline chunks/s.

**`gpu_submit`** becomes `gpu_submit(GpuWorld* g, int buf, uint32_t count, uint32_t ntiles, uint32_t seed, const GmePlanet*, const GmeBioView*)`. Tile jobs are staged in pinned `gpu_staging_tiles(g, buf)` (max_batch × 16 B).

**`gpu_world_create(slots, max_batch, tile_count)`** allocates `d_tiles`. `int gpu_read_tile(GpuWorld*, uint32_t index, GmeColTile* dst)` is synchronous, like `gpu_read_slot`.

**World residency** (new arrays `tile_tag[page_n²]` and `tile_job_tag[page_n²]`, both 0 = none):
1. `scan_fn` sets bit 31 of `missing[k]` when the column's published `tile_tag` ≠ its tag.
2. In `build_batch`, the first visit to such a column queues one tile job, if `tile_job_tag[ti] == 0`. It then sets `tile_job_tag[ti] = tag`. A tile-only column consumes no slot.
3. If the cell is busy with another column's in-flight job (toroidal collision), the job is deferred and `cursor` is reset on the next update.
4. `finish_batch`, for each tile job of the batch: if `tile_job_tag[ti] == tag`, then clear it; and if the column is within radius+1, set `tile_tag[ti] = tag` and `tile_dirty = 1`.
5. `evict`: when a freed chunk's column has a published tile, set `tile_tag[ti] = 0`.

The invariant: the renderer reads only published tiles, publication happens only after the batch event, and at most one job is in flight per cell. This is the same safety argument as today's page entries and slots.

**CPU mode (`--bench --cpu`):** no tile storage. `cpu_col_tile()` is evaluated on demand for `dens_hash` and D1.

### 2.5 Uniform chunks, unloaded chunks, determinism

- **Tiles are per column**, independent of how many layers are uniform or stored. Uniform chunks still take 0 B of voxel storage.
- **Material lookups at a hit:**
  - pool byte if the chunk is resident;
  - the uniform material from the page entry if it is uniform;
  - otherwise the analytic `gme_voxel(x, y, z, &col, seed, &planet)`, with `col` from `gme_column_in_slot` and the tile's slot, so the result is exact even before the chunk streams in.
- **Missing tile:** voxel mode in M1–M2; far field from M3.
- **Determinism:** tiles are integer-only and identical between `k_col_tiles` and `cpu_col_tile`. Rendering may use floats; validation never compares rendered output across devices.

### 2.6 Validation

**`world_validate` additions:**
- 256 published tiles, sampled half near and half far.
- Pass conditions:
  - byte-exact match between `gpu_read_tile` and `cpu_col_tile` (`tile_mismatch == 0`);
  - decoded h == `gme_height` for all 1,296 columns (`tile_h_bad == 0`);
  - `(T < y_w) == (h < sea)` at every column (`tile_coast_bad == 0`);
  - `tile_sat == 0`;
  - `tiles_checked > 0` when using the GPU.
- Statistics `dev_mean/rms/max/clamp_share` are printed.
- **`dens_hash`:** FNV-1a 64 over `cpu_col_tile` for the `WORLD_REF_COLS²` reference columns around `ref_cx/ref_cz`. It is a new fingerprint, printed next to an **unchanged `ref_hash`**.
- The function still returns 1 on any existing failure.

`--bench` prints:
```
Density: 256 tiles (331776 columns): tile mismatches 0, h 0, coast 0, saturated 0; dev T-(h+1) mean -0.015 m rms 0.56 m max 1.00 m, clamp 9.7%; dens_hash ...
```

**Measured on the current planet** (25,165,824 columns, 2048² windows at the centre of each of the 6 life islands; "gentle" = 8 m slope of h < 0.3, which is 84% of pairs):

| Surface | RMS 2nd difference, gentle (m) | Ledges ≥45° (\|ΔT\| ≥ 1 m, adjacent columns) | Exactly flat 4×4 (range < 1/64 m) | Mean normal error vs continuous field | Coast mismatches |
|---|---|---|---|---|---|
| voxel top faces (today) | 0.435 | 9.10% | 51.5% | 12.1° | 0 |
| T unclamped (integration-delivery as written) | 0.042 | 0.001% | 0.00% | 2.6° (the intended micro-relief) | 56,855 / 1,083,579 |
| **T clamped [h,h+2) + coast lock (this spec)** | **0.077** | **0.051%** | **0.95%** | **3.25°** | **0** |

Further measured values:
- Unclamped deviation: mean −0.019 m (window means −0.034 … +0.004), RMS 0.596 m, max 2.36 m, |dev| > 1 m on 9.46% of columns.
- Stored deviation: mean −0.015 m, RMS 0.556 m, clamp active on 9.7% of columns.

---

## 3. Smooth-surface rendering

### 3.1 Ray types and the geometry each one sees

| Ray | Fine 1 m patches | 4 m LOD (`lod4`, M3) | Far field (M3) | Flora | Cap |
|---|---|---|---|---|---|
| primary | to `d_fine` (Low 300 / Med 600 / High 900 m; dithered ±5%) | to `max_dist` | 16 m to 8 km, 64 m to 32 km | analytic to 1.5 km, canopy tint beyond | horizon |
| sun shadow (any-hit) | 32 m | to `max_dist` | 16 m level to 8 km | crowns to 256 m (porosity, M5) | above global top |
| GI | 8 m | to 256 m | none | crowns to 64 m | 256 m → sky |
| reflection | 64 m | to `max_dist` | yes | to 512 m | horizon |

Wavefront split: traversal kernels (`k_primary`, `k_sun`, `k_gi_trace`, `k_gi_nee`, `k_reflect`) carry no float[8] state and are built with `__launch_bounds__(128, 6)`. M1 verifies ≤ 64 registers with no spills via `-Xptxas -v`. All spectral math lives in `k_gi_shade` and `k_compose`.

### 3.2 Column tracer (`trace_column` in trace.cuh)

Coordinates are relative to the camera chunk, as today.

**Level 0, tiles (32 m):** 2D Amanatides–Woo over chunk columns.
- For tile cell [t0, t1]: compute y0/y1 and the tile record `ti = gme_page_index(ax, 0, az, n, 1)`.
- Skip the tile when `min(y0,y1) > tile.top` (and when `max(y0,y1) < tile.bot` in voxel-mode cases).
- Stop as a miss when `d.y ≥ 0` and y > the global top (`GME_HEIGHT_MAX + 2 + GME_FLORA_REACH`).
- A missing tile (`tile_tag ≠ tag`) is crossed in voxel mode (M1–M2) or the far field (M3+).

**Level 1, blocks (8 m = flora slot):** the same test with `blk_top`.
- On entry, if `slot.trad > 0`: closed-form ray–ellipsoid and ray–cylinder over the block's t-range (§3.5). Record `t_flora`.

**Level 2, dual cells (1 m, offset by 0.5):** cell (cx, cz) has its corners at the centres of columns cx..cx+1 and cz..cz+1. The apron covers every dual cell that overlaps the tile, so adjacent tiles evaluate identical patches (no seams).

**Patch solve:**
- Corners: `T = 256(h_base+h8+1) + 2·dev`, in metres relative to the camera.
- Surface: S(u,v) = a + b·u + c·v + k·u·v, with
  - a = T00,
  - b = T10−T00,
  - c = T01−T00,
  - k = T00−T10−T01+T11.
- On the clipped segment s∈[0, s1]: u = u0+du·s, v = v0+dv·s, and f(s) = y−S = A s² + B s + C, with
  - A = −k·du·dv,
  - B = dy − b·du − c·dv − k(u0·dv + v0·du),
  - C = y0 − a − b·u0 − c·v0 − k·u0·v0.
- The hit is the smallest root in range where f goes from ≥ 0 to < 0. If |A| < 1e−7, solve linearly. This is exact (about 30 FLOPs), with no iteration and no voxel fetches.

**Termination:**
- A terrain hit, flora at `t_flora` < the next cell's t, or a water event (§3.6) ends the walk at the smallest t.
- Kelp blocks also test the capsules of the dual cell's 4 corner columns (radius 0.2 + sway 0.3 < 0.5, so each capsule touches only those 4 cells).

**Voxel mode (`trace_voxels`, a copy of the classic DDA):**
- Entered when:
  - the camera is below its column's T, or
  - a hit's material lookup returns `MAT_AIR` (a cave mouth on a cliff), or
  - a tile is missing (M1–M2).
- LEAF, WOOD and KELP voxels are ignored there (they are drawn analytically). WATER is transparent; `water_t` is recorded.
- Exits back to column mode when the ray is above the column's T + 0.5 m.

### 3.3 Normals

- **Shading normal:**
  - corner gradients by central differences, gx = (T[x+1]−T[x−1])/2 and gz likewise (apron 2);
  - blended bilinearly at (u, v);
  - n = normalize(−gx, 1, −gz).
  - This is C0-continuous across cells and tiles.
- **Geometric normal**, used for ray offsets: from ∂S/∂u = b + k·v and ∂S/∂v = c + k·u.
- **Materials** may add a small micro-normal (M5): 0.5 m value noise for rock and soil, sand ripples.

### 3.4 Material at the hit (depth-anchored, with cliff layering)

For each of the 4 corner columns i of the hit's dual cell:
- weight w_i = bilinear weight;
- eligible if T_i ≥ y_hit − 0.25 m. This drops low columns for points high on a cliff face.

Voxel index and material:
- Δ_i = dev_i/128.
- iy_i = min(h_i, ⌊y_hit − Δ_i⌋).
- m_i = voxel(col_i, iy_i): pool, page-uniform, or analytic (§2.5).

On flat ground every eligible corner gives its ground voxel: grass, mat, sand or regolith. On a cliff the depth grows downward, so the grid's real layering shows: soil, then rock, then ore and sulfide veins.

- If any eligible m_i is `MAT_AIR`, the hit is a cave mouth and the ray continues in voxel mode.
- Material parameters (spectral albedo id, roughness, F0) are blended with weights w_i² over the eligible corners, then normalised. This is deterministic, with crisp but non-square borders and no noise for the denoiser.
- The G-buffer stores the 2 dominant materials + weight + flags, and the slot's palette index and cover. Spectra are rebuilt in `k_compose`.

### 3.5 Vegetation

**Trees and shrubs (kind 4/5, `slot.trad > 0`):**
- **Crown:** ellipsoid centred on (tx+.5, tcy+.5+Δ, tz+.5), with horizontal radius a = √(trad²+trad) and vertical radius b = √(tvr²+tvr). This is exactly `gme_voxel`'s test `(dx²+dz²)·vv + dy²·rr ≤ rr·vv` written as a float.
- **Trunk:** vertical cylinder, radius 0.18 + 0.06·trad (< 0.5), from y = tbase+1+Δ to tcy+1+Δ. It contains exactly the WOOD voxel centres.
- **Δ = dev(tx,tz)/128**, so |Δ| ≤ 1 m: the tree stands on the rendered ground.
- **M5 additions:**
  - clumps: inward 3D value-noise displacement ≤ 0.35·trad, sphere-traced in the shell, ≤ 12 steps;
  - leaf transmittance;
  - a porosity used for shadow rays only.

**Kelp (kind 2):**
- Per column: `colhash = gme_hash3(x,0,z,seed^0xF10Au)`; the strand is present iff `((colhash>>16)&255) < (cover>>1)`.
- `len` comes from `gme_kelp_len`, the same integer rules as `gme_voxel`.
- Capsule radius 0.2 m on the column axis, from the column's T to `h+len+1`, clipped 0.1 m below y_w.
- Sway ≤ 0.3 m in time (visual only).

**Ground cover:**
- Tufts and grass are shading (sheen + normal fuzz), not geometry.
- Mats are a ground material.

**Distance:** beyond 1.5 km, flora is a canopy tint and cover term on the LOD surface.

**D2 self-test:** for every ht in 2..32 (so every reachable (trad, tvr)) on flat ground, every voxel centre of the slot satisfies `gme_trunk_inside_f ? WOOD : gme_crown_inside_f ? LEAF : AIR` == `gme_voxel`. Likewise for the kelp capsule axis vs KELP voxels, for len 1..40. Required: 0 mismatches.

### 3.6 Water and ice surfaces

- **Water plane:** y_w = sea_level + 1. Per ray, t_w = (y_w − o.y)/d.y.
- **Water event:** when the dual cell containing t_w lies in a block flagged wet and the bilinear T(x_w, z_w) < y_w.
- **M1:** the classic look. The ray continues straight, with `water_t` and the classic absorption and Fresnel formula, so A/B comparisons are clean.
- **M2+:**
  - Gerstner normals: 8 waves, 0.3–20 m, with direction spread for wind 5 m/s.
  - Exact Fresnel, n = 1.333.
  - The refracted ray continues *in the same thread* from the interface.
  - From below: Snell window 48.6°, and TIR outside it.
- **Ice** (`ocean_top_mat == MAT_ICE`): an opaque dielectric plane, n = 1.31, α = 0.08.

### 3.7 Shadows

- **Sun:** one any-hit ray per pixel toward an IGN/R2-jittered point on the true stellar disc (θ = 5.88 mrad for this planet). Penumbra ≈ 1.2 cm per metre of occluder distance.
- **Points under water:** the ray uses the refracted sun direction up to y_w, then the true sun direction. Weighted by (1 − F) × band transmittance × caustic factor (High).
- **Long-range shadows (M3):** after the 4 m LOD region, march the 16 m far-field min/max mip up to 8 km, or until the ray is above the far-field maximum.

### 3.8 LOD and far field (M3)

- **4 m LOD:** `lod4` 9×9 per tile (corners at the tile edges, so neighbouring tiles share samples exactly). Bilinear 4 m cells, bounded by `blk_top/bot`. The fine/LOD switch is dithered per pixel and resolved by TAA.
- **Far-field clipmap** (renderer-owned, toroidal, needs no world residency):
  - L1: 1024² at 16 m (±8.2 km). L2: 1024² at 64 m (±32.8 km).
  - 4 B per sample: int16 height (q4), each sample the box average of 4×4 sub-samples of `gme_surface_q8`.
  - u8: kind/water bit from `gme_slot` at the sample centre plus h<sea. u8: palette/cover.
  - Min/max quadtree.
  - `k_far_update` rebuilds the rows and columns that enter as the camera moves; a full rebuild takes ~2–3 ms.
  - Curvature drop `(d² − d0²)/(2R_p)` (R_p = `radius_re`·6371 km), continuous at the loaded radius.
  - The far field also fills columns whose tiles have not streamed in yet, so there are no holes.
- **Edge:** at 32 km the haze transmittance at 550 nm is ≈ 0.03 (§4.3). Beyond 30 km the terrain blends into the sky-view LUT's below-horizon radiance.

### 3.9 What stays blocky, and why

- **Cave interiors** (voxel mode) until optional M6. Caves are 3D, not a heightfield; they lie ≥ 7 voxels deep and are seen only from inside or through rare cliff mouths. M6's sign-locked density (valid for caves) rounds them within 0.5 m of the truth.
- **The voxel grid itself** (gameplay and life) is unchanged.

---

## 4. Light transport

### 4.1 Spectral representation

8 render bands of 45 nm. The renderer carries band-mean spectral radiance L_b [W m⁻² sr⁻¹ nm⁻¹], converted as XYZ = 683·Σ_b L_b·(X̄_b, Ȳ_b, Z̄_b) in cd/m².
- The CMF band integrals come from the 1 nm CIE 1931 2° table (ship the 5 nm table). The values below are from the Wyman 2013 fit, within 1% of the table.
- XYZ→linear sRGB: [3.2406 −1.5372 −0.4986; −0.9689 1.8758 0.0415; 0.0557 −0.2040 1.0570].

| b | nm | centre | X̄_b | Ȳ_b | Z̄_b | photon eV | TOA E_b, current planet (W/m²) | τ_Rayleigh, current planet | a_w band mean (1/m) | T(10 m) incl. CDOM |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 380–425 | 402.5 | 1.99 | 0.10 | 9.25 | 3.08 | 60 | 2.03 | 0.0067 | 0.33 |
| 1 | 425–470 | 447.5 | 13.54 | 1.55 | 72.18 | 2.77 | 71 | 1.31 | 0.0078 | 0.48 |
| 2 | 470–515 | 492.5 | 2.49 | 12.10 | 23.67 | 2.52 | 77 | 0.88 | 0.0189 | 0.53 |
| 3 | 515–560 | 537.5 | 12.04 | 39.92 | 1.68 | 2.31 | 79 | 0.62 | 0.0472 | 0.45 |
| 4 | 560–605 | 582.5 | 40.62 | 37.29 | 0.04 | 2.13 | 79 | 0.44 | 0.1195 | 0.26 |
| 5 | 605–650 | 627.5 | 31.62 | 14.02 | 0 | 1.98 | 77 | 0.33 | 0.2892 | 0.045 |
| 6 | 650–695 | 672.5 | 4.37 | 1.85 | 0 | 1.84 | 73 | 0.25 | 0.4443 | 0.011 |
| 7 | 695–740 | 717.5 | 0.09 | 0.08 | 0 | 1.73 | 68 | 0.19 | 1.256 | 0.000 |
| 8 (bio only) | 740–900 | — | — | — | — | ~1.5 | 204 | — | 3.48 | 0 |

- An equal-energy spectrum gives X = Y = Z = 106.8.
- "Light is energy": each band carries its photon energy hc/λ. As the sun drops, Rayleigh scattering (λ⁻⁴) strips the 3 eV photons. CH4 removes the deep red. Water absorbs the low-energy red photons first. A pigment absorbs specific photon energies, and its colour is what it does not absorb.

### 4.2 Star

- E_λ = S_top·π·B_λ(T_eff)/(σT⁴), with S_top = 1361·L/a². This is the same formula as `bio_env_planet`.
- The disc's angular radius is θ = R★/a, with R★ = √L·(5772/T)² R☉.
- Linear limb darkening, I ∝ 1 − 0.6(1−μ), normalised.
- `--sun-disc k` scales the radius by k and divides the radiance by k², so energy is conserved.
- Current planet: S_top = 1484.7 W/m², θ = 5.88 mrad (0.337°). The planet file's `sun_size` (0.0488) is 8.3× the true radius.

### 4.3 Atmosphere (`spectral_build_atmo(const HomePlanet*, double p_bar, SpectralAtmo*)`)

**Column and geometry:**
- N = P/(m̄g), with P = 1 bar (the bio spec's tuned value; `--pressure` overrides).
- m̄ from `atm_mix`; g = `gravity_g`·9.80665.
- H = R·T_surf/(M·g).
- R_p = `radius_re`·6371 km; atmosphere top at R_p + 8H.

**Rayleigh:**
- σ_i(λ) = σ_air(λ)·((n_i−1)/2.92e−4)²·F_K,i/1.048, with σ_air from Bodhaine et al. (1999).
- (n−1)·10⁶ at STP: H2S 641, CH4 444, NH3 376, CO2 449, N2 298, O2 271, H2 132.
- King factors F_K: N2 1.034, O2 1.096, CO2 1.15, all others 1.0.
- σ = Σ x_i·σ_i. Phase function: (3/16π)(1+cos²θ).
- Earth preset (air, 1.01325 bar, g 9.80665): τ(550) must be 0.097 ± 0.005 (computed: 0.0969).

**Gas absorption:**
- CH4 uses Karkoschka (1998) room-temperature coefficients, shipped as `data/ch4_k98.txt` in km-amagat⁻¹ at 1 nm. If the file is missing, a built-in coarse band model is used (619, 667, 702, 727, 790, 840, 864–890 nm) with a warning.
- NH3 (552/645 nm) and H2S (UV) are below 1% in the bands and are listed but omitted.
- H2O is absent from `atm_mix`. `--h2o x` adds it (default 0).
- All gas data are table files, so they can be corrected later.

**Haze (CH4 photochemical, tholin-like):**
- τ_h(λ) = τ_h550·(λ/550)⁻².
- ω_h(λ) = 0.95 − 0.45·e^(−(λ−380)/90).
- Henyey–Greenstein g = 0.65; scale height H/2.
- **Calibration:** bisection on τ_h550 so that the *PAR-weighted global horizontal transmission* (direct + diffuse, sun at zenith, ground albedo 0.15, computed with the LUT model) equals `haze_q12/4096`, where `haze_q12 = 4096 − 2048·x/(x+0.2)` is the bio's formula. If Rayleigh + gas alone are already below the target, τ_h = 0 and a warning is printed.
- Current planet: target 0.637; Rayleigh-only global transmission ≈ 0.79; expected τ_h550 ≈ 0.7 (two-stream estimate). Ground extinction at 550 nm ≈ 0.11/km, giving transmittance 0.68 at 3.4 km, 0.16 at 16 km and 0.03 at 32 km. Hence the far field.

**LUTs (Hillaire 2020 layout, 8 channels):**

| LUT | Size | Built | Notes |
|---|---|---|---|
| transmittance | 256×64 | once per planet, host, double | Built at 5 nm over 380–900 and band-averaged *as transmittance*, weighted by the TOA spectrum, so saturated CH4 bands stay right |
| multi-scatter | 32×32 | once per planet | |
| sky-view | 192×108, fp16 | when the sun moves > 0.1° (~0.1 ms) | non-linear latitude mapping |
| aerial perspective | 32³ froxels × (8 in-scatter + 8 transmittance) | every frame | exponential slices to `max_dist` |

Beyond `max_dist` (far field) the aerial term is an 8-sample analytic march. Horizontal near-ground paths use a 1D table of band transmittance against CH4 column (64 log steps), which is exact for saturated bands.

**Validation:**
- Host Monte Carlo reference: spectral at 5 nm, multiple scattering, Rayleigh + HG haze, ground albedo 0.15; 3 sun elevations × 8 directions.
- Sky-view must be within 10% per band.
- Otherwise (τ is 2–3 in the blue) the multi-scatter LUT is computed by host iterated scattering orders (4 orders, ~1 s) instead of Hillaire's series approximation.
- Band error: the host compares 8-band vs 1 nm products for (leaf × noon sun), (seabed at 10 m) and (terrain at 8 km). If ΔE2000 > 3, move to 12 bands (only buffer sizes change).

### 4.4 Surfaces

**BRDF**, f = f_spec + f_diff:
- f_spec = D_GGX(α)·G2_heightcorrelated(α)·F/(4μ_iμ_o).
  - Dielectrics use exact unpolarised Fresnel: rock n = 1.55, leaf 1.45, ice 1.31, water 1.333.
  - Pyrite is a conductor with spectral F0 = 0.30, 0.32, 0.36, 0.42, 0.48, 0.52, 0.54, 0.56 and α = 0.35.
- f_diff = ρ_b/π·(1−E(μ_i))(1−E(μ_o))/(1−E_avg).
  - E is a 32×32 (μ, α) LUT of the specular lobe's directional albedo.
  - This is reciprocal and keeps albedo ≤ 1 (furnace-tested).

**Wet tidal band** (wetness w = smoothstep from y_w+1.5 down to y_w):
- Lekner–Dorf: ρ_wet = 0.493ρ/(1 − 0.472ρ), from F̄_e = 0.066 and F̄_i = 0.472.
- Plus a water film with F0 0.02 and α 0.05.

**Spectral albedo per band 0..7** (starting values, tuned):

| Material | Bands 0..7 | α |
|---|---|---|
| basalt | .050 .052 .055 .058 .060 .062 .065 .068 | 0.6 |
| granite | .20 .22 .24 .26 .28 .29 .30 .32 | 0.5 |
| dirt | .06 .07 .09 .12 .15 .18 .21 .23 | 0.85 |
| sand | .28 .32 .37 .42 .47 .50 .53 .55 | 0.7 |
| regolith | .12 .14 .16 .19 .22 .24 .26 .28 | 0.8 |
| hematite ore | .04 .05 .06 .08 .15 .28 .33 .35 | 0.6 |
| sulfide | diffuse .02 | conductor as above |
| ice | .80 .80 .79 .77 .74 .70 .65 .58 | 0.08 |
| bark | .05 .06 .07 .09 .12 .15 .18 .20 | 0.8 |

Pigmented tissue (leaf, mat, kelp, grass) is defined in §5.1.

### 4.5 Water optics

- **Pure water:** Pope & Fry (1997) from 380 to 727.5 nm, Kou (1993) above.
- **Added terms:**
  - CDOM: 0.05·e^(−0.014(λ−440)) per m (tuned);
  - particles: 0.02 per m;
  - scattering b = 0.0038·(400/λ)^4.32 + 0.02·(550/λ) per m;
  - phase function: molecular + Henyey–Greenstein 0.92.
- **Band transmittance** T_b(d) is a 64-entry log-d table per band, the band average of e^(−a(λ)d), *not* e^(−mean(a)·d). Its values at 10 m are in §4.1.
- **In-scatter along an underwater view segment** of length d, with κ = a + 0.2b:
  - L = β(θ)·E_w,b(z0)·(1 − e^(−(κ + K·m)d))/(κ + K·m);
  - E_w,b(z0) is the sun irradiance just below the interface;
  - K = κ_b, the sun path's attenuation; m = (rate of depth increase along the ray)/μ_s;
  - plus an isotropic sky term.
- **Caustics (High):** I = clamp(1/(1 + D(1 − 1/n)∇²η), 0.2, 5), using the analytic Gerstner Laplacian, with D = depth/μ_s.

### 4.6 Direct light, sky, GI, reflections

**Direct sun:**
- E_sun,b(x) = E_top,b·T_b(sun path from x)·V(x), times the limb-darkened disc.
- The sun enters *only* through this NEE term: escaped rays read the sky-view LUT with the disc excluded, so there are no fireflies.

**Sky light by preset:**
- Low: SH9 per band from the sky-view LUT × a heightfield AO (8 tile T samples at 2 m and 8 m, in 4 directions).
- Medium/High: via GI rays.

**GI (M4):**
- Half resolution, 640×360. 1 spp (Medium) or 2 spp (High). Cosine-weighted directions from interleaved gradient noise (IGN) + an R2 frame offset, so no data file is needed.
- At the secondary hit:
  - sun NEE with coarse shadow geometry;
  - plus ρ_b/π × the cache irradiance (multi-bounce).
- Escaped rays read the sky-view LUT without the sun disc.
- E_ind,b = π·L̄, stored as 8 fp16 bands.

**Radiance cache:**
- Hash grid with 2²⁰ cells × 24 B (8 fp16 E + checksum + count) = 24 MB.
- Cell size s = clamp(0.25·(1 + d/8 m), 0.25, 16) m; key = (⌊p/s⌋, normal octant).
- Written by `k_gi_shade` with E_ind as an EMA with weight 1/min(n, 32). Read at GI secondary hits.
- A miss falls back to sky SH × 0.5.

**Reflections:**
- Water, ice and wet ground with F above 0.02.
- Medium: one half-res ray per reflective pixel (mirror for water/ice, GGX-sampled for wet ground), shaded with NEE + cache.
- Low: sky-view only.
- High: full resolution + rough lobe sampling.

**Leaves:** back-lit transmission L_t = τ_L,b/π·E_sun,b·e^(−σ·chord), with σ = G·LAD, G = 0.5 and LAD set for LAI 4 per occupied slot. In High, crown shadow and GI transmittance use the same σ·chord (LAD medium), never free flight.

### 4.7 Temporal accumulation, reprojection, denoising

**Jitter and reprojection:**
- Halton(2,3) with 8 phases on primary rays.
- Motion: P_prev = P_rel + 32·(chunk_now − chunk_prev), projected with the previous camera.
- Disocclusion when |Δdepth| > 0.02t + 0.05 m, or n·n′ < 0.9, or the material class changed. History is also reset when the camera moves more than 2 m in a frame.

**GI temporal:** α = max(1/n, 0.05); moments α 0.2; variance ×4 while n < 4.

**À-trous:**
- 3 passes (Medium) or 4 (High); 5×5 B3 kernel (1/16, 1/4, 3/8, 1/4, 1/16) with steps 1, 2, 4 (, 8).
- Weights:
  - w_z = exp(−|Δz|/(|∇z|·step + 1e−3));
  - w_n = max(0, n·n′)^64;
  - w_l = exp(−|ΔY|/(4√var + 1e−4)).
- Then a joint-bilateral 2×2 upsample.

**Sun visibility V:** full resolution, temporal α 0.2, plus one 3×3 à-trous pass.

**TAA:**
- On exposed linear sRGB: YCoCg 3×3 variance clip with γ = 1.25, history weight 0.9 (0.8 under fast motion), luminance-weighted resolve.
- **TAAU** for `--scale` in [0.5, 1]: jittered samples are accumulated into output-resolution history with Catmull–Rom history fetch. This replaces StretchDIBits upscaling in the new path.

**Reference:** `--accum N` gives the reference image: N frames, 4 bounces, no denoising.

### 4.8 Exposure, white balance, tonemapping

- **White balance:** CAT16 with M16 = [0.401288 0.650173 −0.051461; −0.250268 1.204414 0.045854; −0.002079 0.048952 0.953127]. It adapts from the XYZ of the current global horizontal irradiance spectrum at the camera's ground to D65, with degree D = 0.8 (`--wb D`; 0 = raw alien light).
- XYZ → linear sRGB; negatives are desaturated toward Y.
- **Exposure:**
  - log2(Y) histogram, 64 bins over [−8, +16], from every 4th pixel in x and y (`k_lum_hist`).
  - Mean of the 5–95% range; k = 0.18/Y_avg.
  - Adapts in log space: τ = 0.5 s when the image darkens, 2 s when it brightens.
  - `--ev EV100` fixes k = 1/(1.2·2^EV100).
- **AgX:** Wrensch's "minimal AgX" for sRGB primaries.
  - Inset matrix (column-major as published): {0.842479062253094, 0.0423282422610123, 0.0423756549057051; 0.0784335999999992, 0.878468636469772, 0.0784336; 0.0792237451477643, 0.0791661274605434, 0.879142973793104}.
  - Encode log2, clamp to [−12.47393, 4.026069], normalise.
  - Sigmoid 15.5x⁶ − 40.14x⁵ + 31.96x⁴ − 6.868x³ + 0.4298x² + 0.1191x − 0.00232.
  - Outset {1.19687900512017, −0.0528968517574562, −0.0529716355144438; −0.0980208811401368, 1.15190312990417, −0.0980434501171241; −0.0990297440797205, −0.0989611768448433, 1.15107367264116}.
  - Then pow 2.2 → linear → piecewise sRGB OETF. This replaces today's `sqrt`.
- **Final output:** ±0.5 LSB triangular IGN dither. The crosshair is drawn after tonemapping, inverted as today.

---

## 5. Life coupling

### 5.1 Phase A, engine-only now (M5)

**Pigment from the exported hue bin** (the biomap palette index, `hue_index` in `bio_io.c`):
- λ_Q = 383 + 34.13·i nm for i = 1..15. The later gene form is 400 + 2·G_HUE.
- ε(λ) = 1.2·G(λ; 435, 30) + G(λ; λ_Q, 25) + 0.15·s(λ). G is a Gaussian with the given centre and σ in nm, and s(λ) = 1 below 690 nm, falling linearly to 0 at 740.
- OD_b = PIG/255 · 4.0 · ε. The factor 4.0 includes in-leaf path lengthening (tuned). PIG = 200 until biomap v2 carries the mean pigment.
- A_b = band mean of 1 − 10^(−OD) at 1 nm.

**Leaf optics:** ρ_L = 0.04 + 0.44(1−A), τ_L = 0.42(1−A). ρ + τ ≤ 0.9 and τ ≥ 0 always.

**Other tissues:**
- **Mat:** ρ = 0.03 + ρ_sand·(1−A)².
- **Grass ground voxel:** mix(ρ_soil, ρ_L, 0.8·cover/255).
- **Kelp:** leaf optics, under water.

The HSV palette survives only as a debug view.

**`gme --pigment-table` reference** (equal-energy light, von Kries to D65, sRGB):

| Bin | λ_Q | sRGB | Bin | λ_Q | sRGB |
|---|---|---|---|---|---|
| 1 | 417 | 127,117,46 | 9 | 690 | 114,118,47 |
| 2 | 451 | 129,114,43 | 10 | 724 | 125,117,47 |
| 3 | 485 | 136,97,45 | 11–15 | 758–895 | 127,117,46 (NIR absorbers: only Soret + background absorb visible light) |
| 4 | 520 | 133,70,52 | | | |
| 5 | 554 | 106,58,57 | | | |
| 6 | 588 | 55,82,58 | | | |
| 7 | 622 | 27,107,53 | | | |
| 8 | 656 (chlorophyll-like) | 81,118,49 | | | |

**Probe, key L:** spectral irradiance at the crosshair hit.
- Per band: W/m²/nm and µmol photons/m²/s.
- Sun/sky split, depth below water, total PAR photons.
- The ratio to the bio's scalar i0, printed as "render/bio".

**`gme --export-spectrum life/spectrum.bin`** (`GMESPEC1`, `src/spectrum_file.h`; integer-only, little-endian, header CRC32 + payload CRC32, written to a tmp file then renamed):

```c
#define SPEC_NB 9      /* 8 render bands + NIR 740-900 */
#define SPEC_NHAZE 8   /* haze_q12 bins 4096 - 256*k */
typedef struct SpectrumFile {
    uint64_t planet_hash; int32_t band_edge_nm[SPEC_NB + 1]; int32_t s_top_wm2, t_surf_ck;
    int32_t surf_q16[SPEC_NHAZE][24][SPEC_NB];  /* global horizontal irradiance / S_top at the bio's SIN24 elevation, q16 */
    int32_t dir_q16[SPEC_NHAZE][24][SPEC_NB];   /* its direct-beam part */
    int32_t water_T_q16[SPEC_NB][33];           /* band-averaged transmittance at 0..32 m */
    int32_t kw_q16[SPEC_NB];                    /* effective attenuation 0-20 m, per m */
    int32_t phot_q16[SPEC_NB], ev_q16[SPEC_NB]; /* umol photons per J; mean photon energy (eV) */
    int32_t uphill_q16[SPEC_NB][2];             /* min(1, exp(-(E_RC - E_ph)/kT)), RC 680 nm (OXY) / 870 nm (ANOX), kT from t_surf */
    int32_t eps_q12[16][SPEC_NB];               /* pigment shape per hue bin, for 10^-OD via expn_q16 */
} SpectrumFile;
```

- Current planet, kT = 25.4 meV: OXY gets 2.4% of band 7 (717 nm) and nothing from NIR; ANOX gets all of NIR.
- The doubles are computed once on the PC and shipped as integers, so the rigs never call libm.

### 5.2 Phase B, bio core later (M7; integer only, flag-gated)

> **Built.** The life side is described as built in [biosphere-spec.md](biosphere-spec.md) 10.5. What differs from the plan below:
> - **Switch and CRC:** the two reserved words are now `BioParams.light_mode` and `BioParams.spectrum_crc`. A new run takes `life/spectrum.bin` when it exists (`--spectrum FILE`, `--no-spectrum`); `life --resume --enable spectral` switches a running world over, logged as an intervention.
> - **Water:** each band uses its transmittance table at 0–32 m (interpolated), and the band's attenuation only below 32 m. A band average of exponentials is not an exponential, and the table is what the renderer computes.
> - **Canopy:** grey (phase 1). The per-band leaf area index (phase 2) is not built.
> - **Energy scale:** exact, not ±5%. The two efficiencies are computed when the run is created so that the reference leaf (pigment 200, hue bin 8, at the surface, over a day at the present haze) gains what the scalar model gave it. T13a (the test planet): 10176 against 10192 eu per day (−0.2%, integer rounding). Switching it on in the evolved island 1 moved plant production by +4% in a sim-year and left the vegetation where it was.
> - **Leaf area** no longer scales with the pigment gene when light comes in bands: the pigment acts through its absorptance.
> - **`BG_QWIDTH`** (gene 41) mixes in the neighbouring hue bins' pigment shapes (up to half of each) and up to doubles the pigment cost.
> - **T13** as built has four parts: calibration, the depth profile against the band sum (within 1.7%), selection (two clones on a shelf of 2 m and 12 m: the red one wins the shallows, the blue-green one the deep, and nothing separates them with scalar light), and CPU ≡ GPU with every feature plus the resume rules. The Spearman criterion of the plan was replaced: a saturated population of one pigment shows no shift within a test's time.
> - **T12:** see biosphere-spec 10.11.

**Switch:** `BioParams.reserved[0] = BIO_LIGHT_SPECTRAL`.
- `BioParams` is stored in `GMEBIO02` checkpoints. Every existing checkpoint therefore loads with 0 and keeps today's scalar `bio_light` path bit for bit.
- New runs opt in with `life --spectrum life/spectrum.bin`. Its CRC goes into `reserved[1]`, and a resume with a different file is refused.

**`bio_globals_update`:**
- k = clamp((4096 − haze_q12) >> 8, 0, 6) and f = (4096 − haze_q12) & 255.
- I0_b = s_top·lerp(surf[k][hr][b], surf[k+1][hr][b], f) >> 16.
- This replaces `par_q12`·haze for flagged runs. It fixes the 45% vs 33.6% PAR discrepancy, and the CH4 haze the island makes now changes the light spectrum.

**`bio_light_spectral`:**
- Water: I_b·EXPN(kw_b·zq8) with the same index scaling as `bio_light`.
- Canopy, phase 1: grey, using the existing `lai` index.
- Canopy, phase 2: `lai_abs[9][cell]` (+36 MB per island) for spectral understorey light.

**Absorptance:** A_b = 65536 − `expn_q16`[min(1023, PIG·eps_q12[hue][b]·K >> S)], with K and S chosen so the index is 147.4·OD.

**Harvest in photons:**
- Q = Σ_b ((I_b·A_b) >> 16)·phot_b >> 16·uphill_b[path] >> 16.
- Energy: eu = Q·E_RC[path]·eff′.
- eff′ is retuned so that the reference planet and pigment give today's mean yield within ±5%.
- Photon energy decides which metabolism a colour can drive: OXY gets nothing beyond ~720 nm, ANOX uses 740–900 nm.

**Files:**
- `bio_types.h`: flag, `BioSpectral` tables, a `const BioSpectral* SP` appended to `BioWorld`.
- `bio_core.h`: `bio_light_spectral`, `bio_absorb_q16`.
- `bio_kernels.cu`, `bio_sim.c`, `bio_cpu.c`: upload.
- `bio_api.h`: `BioConfig.spectrum` pointer.
- `bio_main.c`: CLI.
- `bio_test.c`: new tests.
- `docs/biosphere-spec.md`.

**Tests:**
- T1 lint (the tables are integers loaded from the file).
- T2/T2b CPU==GPU with the flag on.
- **T14:** flag off. A resumed checkpoint hashes identically to the pre-M7 binary, and T3–T9 and T12 are unchanged.
- **T13:** spectral selection. In a depth-gradient island, Spearman(depth, λ_Q of the dominant phototrophs) ≤ −0.5, or λ_Q shifts by at least 20 nm toward the 470–520 nm water window at 20 m, against a μ = 0 control.
- T12 must stay ≥ 150 steps/s.

### 5.3 Phase C, shapes to evolve into (M8)

> **Built.** Life side: [biosphere-spec.md](biosphere-spec.md) 10.6. Engine side: 5.5 below. What differs from the plan below:
> - **Gene value 0 is the plant as it was.** The shape genes sat unused through every evolved world's history and are still near 0, so `BG_ASPECT` and `BG_LEAFANG` wrap around their range: gene 0 is height : width 1 and the mid leaf angle (`BIOMORPH_ASPECT_OF`, `BIOMORPH_LEAFCLASS_OF` in `biomap.h`). Aspect has 8 steps: 0.5, 0.75, 1, 1.4, 1.8, 2.2, 2.6, 3.
> - **Selection** is a crown model, not only "light caught against a cost": the crown's top area (width² × class factor), interception `1 − e^(−κ·L)` with Goudriaan's projection per leaf-angle class and hour (never more than falls on the top), saturation of the lit leaves of every land plant, shading by what a plant intercepts, overtopping through height, and a structure cost per class. In weak light a thin all-zero shape gains what the plant gained before; an evolved vegetation is nevertheless re-sorted when the feature is switched on (biosphere-spec 10.6).
> - **Biomap v2** as planned; the shape word also carries a valid bit (`BIOMORPH_VALID`), and the header's atmosphere is `atm_ppm[7]` with a flag saying whether it is known.

**Genes in reserved body slots** (`BG_STORE` = 36, so 37–47 are free; `BG_NEUTRAL0` = 48):

| Gene | Meaning |
|---|---|
| 37 `BG_CROWN` | ellipsoid / cone / umbrella / column / dome; clumping Ω 0.8 / 0.6 / 0.9 / 0.7 / 1.0 |
| 38 `BG_ASPECT` | height:width 0.5–3; spreads leaf area over the height bands |
| 39 `BG_LEAFANG` | χ = −0.4 + g/255; Goudriaan G(θ) = φ1 + φ2·cosθ with φ1 = 0.5 − 0.633χ − 0.33χ² and φ2 = 0.877(1 − 2φ1), as an integer table G_Q16[16][24] |
| 40 `BG_POROSITY` | |
| 41 `BG_QWIDTH` | 15–60 nm; pigment cost ∝ PIG·width |

- **Selection:** light caught at the hour's sun elevation against a structure cost per shape.
- **Biomap v2 (`GMEBMAP2`, `BIOMAP_VERSION 2`):**
  - adds `uint32 morph[cell]`: crown 3 bits, aspect 3, leaf angle 2, porosity 2, mean PIG 8, Q width 4;
  - the header adds `atm_mix` in ppm, so the rendered sky follows the atmosphere life has made (the haze clears as CH4 falls);
  - `biomap_load` reads v1 and v2.
- **`flora.h` v2:**
  - `GmeBioView` gains `const uint32_t* morph`, NULL for v1, which leaves the code path unchanged;
  - `gme_slot`/`gme_voxel` apply integer inside-tests per crown class, all inscribed in the 8 m slot, so evolved shapes become voxel truth;
  - `density.h`/`trace.cuh` get the matching float intersectors, and D2 is extended to v2.

### 5.4 Which files change in which phase, and why the islands cannot break

| Phase | Files | Why the running islands are safe |
|---|---|---|
| **A** (M1–M6, now) | density.h, trace.cuh, spectral.*, atmo.cu, gi.cu, spectrum_file.h; gpu.*, world.*, render.*, main.c; the `gme` target in CMakeLists | None of these is a `gme_genesis` or `gme_biotest` source (those are genesis_main.c, cosmos.c, chem.c, planetfile.c, biomap.c, bio_env.c, bio_main.c, the bio core, rtcuda.c, platform.c, jobs.c and image.c). Every rig build uses `-DGME_BUILD_ENGINE=OFF`, so the `gme` block is skipped even when `src/` is redeployed. Gates: SHA-256 of all protected files is unchanged, and `gme_genesis env --island k --crc-only` equals `life/env_crc.txt` for k = 0..4 (fb0adde3, 92e3947f, 9b5d80f8, d6b757ad, 03277b3b). |
| **B** (M7, after the integration builder lands and the islands are stable) | bio_types.h, bio_core.h, bio_kernels.cu, bio_sim.c, bio_cpu.c, bio_api.h, bio_main.c, bio_test.c, the bio spec | Flag off is bit-identical (T14). Old checkpoints carry `reserved[0]` = 0. Env CRCs are untouched, since bio_env.c only uses gme_height, gme_material and gme_env_bin. Rigs change only through `bio_launch.ps1 -Deploy` after T1–T14 pass on the PC. Flagged runs are new runs only. |
| **C** (M8) | as B, plus bio_io.c (export v2), biomap.h/.c, flora.h, density.h, trace.cuh | v1 biomaps take the unchanged code path, so `ref_hash` is identical for `--no-life`, `--biomap-test` and the v1 home biomap. `gme_env_bin`, `gme_height` and `gme_material` are untouched, so env CRCs hold. |

### 5.5 Crown shapes and creatures in the engine (as built)

**Crown shapes (phase C, engine side).** A version 2 biomap gives every cell a shape word (`BIOMORPH`, `biomap.h`). A shrub or tree of a cell whose word is valid takes its crown class and aspect bin; a cell without a valid word, and every cell of a version 1 biomap, keeps the version 1 plant, voxel for voxel.

- **Voxel truth (`flora.h`).** Radius `r` = the version 1 radius (1..3; a column one less), so every crown stays inside its 8 m slot. Crown rows `ch` from the aspect: the ellipsoid is the version 1 crown stretched by the aspect (at 1 : 1 it is the version 1 plant), cone, column and dome have `ch = round(aspect × (2r + 1))`, the umbrella is a cap of half those rows on a bare trunk. With `X = dx² + dz²`, `a² = r² + r` and `u` the height of a voxel centre above the crown's bottom face, a voxel is leaf when

  | Class | Inside when |
  |---|---|
  | ellipsoid | `X/a² + (u − ch/2)² / ((ch² − 1)/4) ≤ 1` |
  | cone | `X/a² ≤ ((ch − u)/ch)²` |
  | umbrella | `X/a² ≤ u/ch` |
  | column | `X ≤ a²` |
  | dome | `X/a² + (u/ch)² ≤ 1` |

  all in integers (`gme_crown_v2_inside`). Porosity, leaf angle, pigment amount and Q band width never change a voxel; they are carried in the slot's shape word for the renderer.
- **Smooth surface (`density.h`, `trace.cuh`).** The same solids as quadrics for the column tracer. Self-test D2 covers version 2: 5 classes × 8 aspect bins × heights 2..32 m at every trunk position, 9.2 M voxel centres against the analytic solids with 0 mismatches; the ellipsoid at 1 : 1 against the version 1 plant, 249,600 voxels, 0 differ; ray chords within 2 cm of a 5 mm march.
- **Sky.** A version 2 biomap carries the atmosphere life has made; the sky uses it (`--sky-planet` keeps the planet file's).
- **Test pattern.** `--biomap-test2` is the version 1 pattern with every crown class × aspect bin; with `--sheet FILE` it renders one close-up per class and aspect.

**Creatures (`fauna_scene.*`, `render.cu`).** The creatures of a GMEFAUN1 file (`--fauna file`, default `life/home.fauna`; `--no-fauna`; `--fauna-test [N]` for a built-in herd) are drawn as spheres. They never touch the chunk pool or the tiles, so generation, validation and every hash are unaffected.

- **Placement** (the file carries no heights). A walker stands on the drawn ground: the bilinear patch of the smooth surface under it, its centre one radius along the slope's normal above it. A swimmer hangs at its depth below the water plane, kept between the seabed and the surface. A walker whose ground is under water and a swimmer over land are dropped. Radius = half the record's diameter, clamped to 0.2..6 m. A swarm is 6 smaller spheres on a jittered ring.
- **Motion.** An idle sway and hop (walkers) or drift (swimmers), amplitude and speed from the motor gene; `--still` holds it. The snapshot is frozen: creatures do not travel.
- **Tracing.** The spheres are binned into a hash grid of 32 m cells (one per chunk column; a sphere reaching into a neighbour is listed in both, with the cell's height range). The kernels walk the grid along each ray up to 1024 m, and along the sun ray up to 128 m for shadows. Colour: the palette index (pigment hue bin) with a per-creature brightness.
- **Classic view.** A walker stands on the top face of its column's ground voxel.

**Gates, as run on the PC before render tests moved to the cluster and before the merge with the reviewed renderer (commit ba450fc on the pre-review engine):** `ref_hash` and `dens_hash` of `--bench` unchanged against the first build for `--no-life` (GPU and `--cpu`), `--biomap-test` and the version 1 home biomap; `--biomap-test2` with the same hashes on GPU and CPU; D2, D5 and D6 passed. **Not yet done on the merged engine:** any run of it on a GPU. Its reference is the reviewed renderer (7a3a819): `ref_hash` and `dens_hash` of `--bench` must equal that commit's for `--no-life`, `--biomap-test` and a version 1 biomap (`dens_hash` no longer equals the first build's: the review's smoothed surface changed the tiles), frames without creatures and with a version 1 biomap must equal its frames, and D1 to D6 must pass. Also open: the headless Linux build of these changes, and a frame-time measurement on a quiet GPU (the launch bounds of `k_surface` and `k_sec_gen` were tuned before the review enlarged those kernels). Checked without a GPU: the merged engine compiles on Windows, `--export-spectrum` writes the committed `life/spectrum.bin` byte for byte, the light of a planet sky and of a biomap sky builds and is cached apart, and the creature placement code places the creatures of an exported file.

---

## 6. Performance, knobs, fallback

### 6.1 Budget per pass at 1280×720 on the RTX 3060 (GPU ms, estimates to be measured)

| Pass | Low (scale 0.67 + TAAU) | Medium (default) | High |
|---|---|---|---|
| uploads/scatter, LUT updates, far-field update | 0.1 | 0.15 | 0.2 |
| `k_primary` (column tracer, flora, water entry, G-buffer 24 B/px) | 1.0–1.8 | 2.2–3.8 | 2.4–4.2 |
| `k_sun` | 0.5–0.9 | 1.0–1.8 | 1.2–2.0 (long range) |
| GI (`trace`/`nee`/`shade` + cache) | SH×AO 0.2 | 1.3–2.3 | 2.5–4.0 (2 spp) |
| `k_reflect` | 0.05 (sky only) | 0.3–0.8 | 0.6–1.2 (+ caustics) |
| denoise (GI + V) | 0.15 | 0.6–0.8 | 0.8–1.0 |
| compose + aerial + histogram + TAA(U) + tonemap | 0.4–0.5 | 0.5–0.6 | 0.6 |
| async readback (overlapped) | 0.2 | 0.3 | 0.3 |
| **total** | **2.6–4.0** | **6.4–10.5** | **8.3–13.6** |

**Basis:**
- Today's `k_render` takes 5–10 ms for 1.84 M voxel-DDA rays (up to `MAX_STEPS` 1536).
- The column tracer replaces 3D per-voxel stepping with 2D tile and block skips plus ~2–20 exact patch solves per ray.
- Incoherent rays are priced at 2× coherent ones.

**CPU side:**
- `world_update` averages 0.6–1.4 ms per tick in the bench log.
- GDI present at 1280×720 is ~1–2 ms, overlapped by double-buffered readback (one frame of latency; screenshots synchronise).

**Contention:**
- The render stream is created with `cudaStreamCreateWithPriority` at the greatest priority; the generation streams get the least.
- Viewer batches are capped by `--batch` (512 recommended for the viewer, to be measured; the bench keeps 2048).
- The PC lab island is another process. Priorities do not apply across processes, so measure with it paused.

**Fixes to today's code:**
- **Page table:** the full ~7.5 MB upload on every dirty frame (pageable memory, 246²×16 entries) is replaced by dirty lists (`page_dirty_idx`, `tile_dirty_idx`) plus `k_page_scatter`. A full upload is used only when more than 1/8 of the table changed.
- **Readback:** the synchronous readback becomes double-buffered.

### 6.2 Quality knobs

- **Presets:** `--quality classic|low|medium|high|ref`; key K cycles them, key V toggles classic vs new for A/B.
- **Other options:** `--scale` (render scale, with TAAU in the new path), `--gi 0|1` (key G), `--lod-fine M`, `--far 0|1`, `--sun-disc k`, `--time H` (key T steps 1 h; the default is today's fixed `build_sky` direction), `--ev`, `--wb D`, `--pressure`, `--h2o`, `--accum N`, `--debug normals|albedo|gi|dev|band N`.
- **Existing** `F` (shadows), `P` (screenshot), `--fps`, `--frames`, `--screenshot` and `--spawn`/`--look` are kept.

### 6.3 VRAM of the new renderer (Medium, 1280×720)

| Item | Size |
|---|---|
| G-buffer | 22 MB |
| V + history | 4 MB |
| colour history (RGBA16F ×2) | 15 MB |
| output ×2 | 7 MB |
| half-res GI (8 fp16 bands, 5 buffers + moments + guides) | 22 MB |
| LUTs | 2 MB |
| far field | 11 MB |
| radiance cache | 24 MB |
| **total** | **≈110 MB** |
| High adds the cave cache (40 MB) | |

**Allocation order:**
1. `renderer_create(&cfg)` allocates all of this before `world_create`.
2. `main.c` re-runs `gpu_query`.
3. The pool is then sized from the true free memory.

### 6.4 Classic fallback

`--quality classic` calls the untouched `k_render`/`trace`/`sky`/`albedo` with `RenderSky`, as today. It must stay ≥ 99.99% pixel-identical to the pre-upgrade binary at the fixed views; target 100%.

---

## 7. Tests and done-checks

### 7.1 Harness (M1)

- **`--view coast|peak|forest|shelf|far`:** deterministic spawn and look computed from the generator.
  - `coast`: today's `spawn_coast`, or `spawn_default` without life.
  - `peak`: highest h on a 64 m lattice within 4 km, eye 60 m, looking at `coast`.
  - `forest`: the `--biomap-test` centre, eye 20 m.
  - `shelf`: nearest column with depth 8–15 m, eye 2 m under y_w via `--eye-y`.
  - `far`: `peak` with pitch 0, toward the longest open direction.
- **`--settle`:** wait for `fully_loaded` before counting `--frames`.
- **`--raw`:** also dump BGRA next to the PNG. `gme --compare A.raw B.raw` prints the identical-pixel %, PSNR and max difference.
- **`--metrics`** (GPU atomics; `k_crack` is a debug kernel running both tracers):
  - terrain pixels;
  - share of axis-aligned shading normals (max |n_i| > 0.9995);
  - crack pixels: classic voxel DDA hits terrain at t < 0.9·`max_dist` while the new tracer misses;
  - NaN count.
- **`--perf`:** per-kernel cudaEvent pairs over N frames on the bench flight path (250 m/s, heading 0.6·sin(0.05t)). Prints p50/p95/max per kernel, GPU total, `world_update` and present; writes `perf_<view>.csv`.
  - Measured with the PC lab island paused (Ctrl+C checkpoints it) and again with it running, for the record.
- **`--selftest`:** D1–D6 (below; a subset with `--selftest d2,d5`), exit code = number of failures.

### 7.2 Gates that apply to every milestone

- `gme --bench` gives `RESULT: OK`, and `ref_hash` equals the recorded pre-upgrade value for `--no-life`, `--biomap-test` and the home biomap.
- GPU pool, host and meta mismatches are 0; `flora_checked` and `flora_voxels` are unchanged.
- Density checks (§2.6) are 0, and `dens_hash` is identical across 2 runs. `--bench --cpu` gives OK with the same `ref_hash`/`dens_hash`.
- `gme_genesis env --island k --crc-only` equals `life/env_crc.txt` for k = 0..4, and the protected-file SHA-256 list is unchanged.
- Generation throughput: chunks/s ≥ 90% of baseline; slots ≥ 96% of baseline.

### 7.3 Self-tests

| Test | Checks | Pass |
|---|---|---|
| D1 density (CPU, the 6 island windows, 25 M columns) | h == `gme_height`; \|T−(h+1)\| ≤ 1; coast mismatches 0; gentle RMS 2nd diff ≤ 0.10 m (*meas.* 0.077); ledges ≥ 45° ≤ 0.2% (*meas.* 0.051%); flat 4×4 ≤ 2% (*meas.* 0.95%); \|window mean dev\| ≤ 0.10 m; tile saturation 0; 4,096 random GPU tiles == CPU | all |
| D2 flora | analytic trunk/crown/kelp vs `gme_voxel` at every voxel centre, all ht 2..32 and len 1..40 | 0 mismatches |
| D3 physics (M2) | Planck vs σT⁴ ±0.5%; Earth τ(550) 0.097 ± 0.005 with zenith bluer than horizon; equal-energy → x,y = 1/3 ± 0.002; BRDF directional albedo ≤ 1 (32 angles × materials); sky-view vs Monte Carlo ≤ 10%; haze residual ≤ 0.005; water T_b(10 m) vs 1 nm integration ±1%; Snell window 48.6 ± 0.5°; band-error ΔE2000 ≤ 3 | all |
| D4 render (M4) | furnace scene (flat, uniform sky L, ρ = 0.5, no sun) gives 0.5L ± 1% after 256 frames; 0 NaN in an 1800-frame flight | all |
| D5 export (M5) | spectrum.bin byte-identical ×2; Σ bands of `surf` at noon × S_top = LUT global irradiance ±1%; PAR transmission at the planet's haze bin = `haze_q12/4096` ± 0.005; pigment table = §5.1 ± 3 sRGB levels | all |
| D6 far field (M3) | GPU far-field clipmap samples (2 levels, 2 builds and camera moves) vs the CPU `gme_far_sample`; update time per 16 m camera step | 0 mismatches; ≤ 0.3 ms |

### 7.4 Screenshots

Captured with `gme --view V --settle --frames 120 --fps 0 --time 14 --ev <fixed from M2> --raw --screenshot shots/<M>/<V>_<quality>.png` for classic and new, at the same spawn and look, stored with `perf_<V>.csv` and the `--metrics` output.

### 7.5 fps gates

| Milestone | Gate |
|---|---|
| M1 | new-path p95 GPU ≤ classic p95 on all views and on the flight (hard fail above 1.3×) |
| M2 | Low ≤ 6 ms, Medium (direct only) ≤ 9 ms p95 |
| M3 | Medium ≤ 9.5 ms p95; far update ≤ 0.3 ms per move |
| M4 | 1800-frame flight with streaming: Low ≤ 8 ms, **Medium ≤ 16.7 ms (60 fps)**, High ≤ 25 ms (≥ 30 fps guaranteed); page/tile upload ≤ 0.1 ms p95; present-to-present p95 ≤ 16.7 ms at Medium with `--fps 60` |
| M5 | forest view ≤ +1.5 ms vs M4 |

---

## 8. Milestones (implementation order)

| M | Content | Files | Done-check | LOC |
|---|---|---|---|---|
| **M1 Smooth terrain on screen** | Baseline record (bench hashes, chunks/s, classic `--perf`, views). `density.h` (T, tiles, D1/D2). `k_col_tiles` + tile residency + validation + `dens_hash`. `trace.cuh` column tracer (patches, water plane with the classic look, depth-anchored materials, analytic trees/shrubs/kelp, voxel mode for caves and missing tiles). `k_smooth` with today's lighting model on smooth normals. Harness (`--view`, `--perf`, `--metrics`, `--selftest`, V toggle). | density.h, trace.cuh (new); gpu.cu/h, world.c/h, render.cu/h, main.c | §7.2 gates. D1 and D2 pass. Axis-aligned terrain normals < 2% (classic ≈ 100%). Cracks ≤ 1e−4 of pixels. p95 ≤ classic. Screenshot pairs at 5 views. Internal order: D1 on CPU → tiles + validation → tracer → flora + D2 → harness. | ~2,000 |
| **M2 Spectral sun and sky** | `spectral.c` (Planck, CIE, gases, LUT build, haze calibration, Monte Carlo reference, material spectra, water). `atmo.cu`. G-buffer pipeline (`k_primary`, `k_sun` with disc, `k_compose`). Refraction and spectral water. Sky-SH × heightfield AO ambient. Exposure, CAT16, AgX, sRGB. Time of day. Probe L. Edge fade at 0.85–1.0 of `max_dist`. Low/Medium (direct only). | spectral.h/.c, atmo.cu (new); render.cu/h, trace.cuh, main.c, CMakeLists (gme only) | D3 passes. Startup prints τ per band, H, N, haze τ, S_top and PAR share for the loaded planet. Screenshots at sun elevations 60/20/5/−3°. fps gate. | ~2,000 |
| **M3 Distance** | `lod4` in tiles. LOD traversal. Far-field clipmap (`k_far_update`, march, curvature). Long-range shadows. Edge fade removed. | density.h, gpu.cu, trace.cuh, render.cu, main.c | No sky pixels below the geometric horizon in `far`. Luminance step at the `max_dist` ring < 2%. Static 64-frame luminance σ < 1% for t > 1 km. Mountain shadows longer than 512 m visible at sun elevation 10°. fps gate. | ~800 |
| **M4 Indirect light and frame pipeline** | GI (trace/nee/shade), radiance cache, SVGF, TAA/TAAU, reflections, motion vectors, async readback, stream priorities, dirty-list scatter, High/Ref presets, `--accum`. | gi.cu (new); render.cu/h, trace.cuh, world.c/h, gpu.cu, main.c, CMakeLists | D4 passes. Static Medium after 64 frames vs `--accum 1024`: relative MSE < 2%, PSNR ≥ 35 dB; flicker < 1%. Disoccluded pixels carry no history older than 2 frames (debug view). fps gates (§7.5). | ~1,500 |
| **M5 Life colours and shapes (engine)** | Pigment and leaf optics, mats/grass/kelp spectra, clumped crowns, LAD shadow transmittance (High), kelp blades, `--pigment-table`, `--export-spectrum` (GMESPEC1), probe photon output. | spectral.c/h, spectrum_file.h (new), trace.cuh, render.cu, main.c | D2 = 0; D5 passes. `--biomap-test` sheet shows 5 kinds × 15 hues. `flora_checked` unchanged. fps gate. | ~900 |
| **M6 Smooth caves** (optional) | Near-camera sign-locked cave density: 34³ int8 per cave chunk, LRU 1,024 entries = 40 MB, built on the GPU from the pool, CPU twin. Trilinear crossing in voxel mode. | density.h, gpu.cu, trace.cuh, render.cu, world.c | Cave cache CPU == GPU byte-identical. Sign at every voxel centre equals the truth. Smoothed hits within 0.5 m (Chebyshev) of a solid voxel. High stays within budget. | ~600 |
| **M7 Bio spectral light** (phase B) | §5.2 | §5.4 row B | T1–T12 + T13 + T14; 50 sim-years on the lab island; env CRCs unchanged | ~700 |
| **M8 Morphology** (phase C) | §5.3 | §5.4 row C | v1 `ref_hash` unchanged; v2 validation 0 mismatches; D2 v2 = 0; ≥ 3 evolved crown classes on screen | ~1,100 |

M5 depends only on M2, so it can move ahead of M4 if life colours are wanted earlier.

---

## A. Current planet (system 59) reference values

These are printed by the engine at startup. They are reference values, not gates.

- **Star:** 5247.6 K, 0.4664 L☉, a = 0.6539 AU, S_top = 1484.7 W/m² (bio `s_top_wm2` 1485).
- **PAR:** 400–700 nm is 33.6% = 498 W/m². The bio assumes 45% = 668 W/m².
- **Band TOA irradiance:** 60, 71, 77, 79, 79, 77, 73, 68 W/m², NIR 204. Photon flux 203…410 µmol/m²/s per band.
- **Disc:** R★ = 0.826 R☉, θ = 5.88 mrad (0.337°).
- **Atmosphere:**
  - m̄ = 22.28 g/mol, g = 6.311 m/s², H = 17.45 km, N = 4.284e29 m⁻² (Earth 2.148e29), R_p = 7954 km.
  - Rayleigh cross-section = 2.909 × air; τ_R(550) = 0.562, per band 2.03 1.31 0.88 0.62 0.44 0.33 0.25 0.19.
  - CH4 vertical column 8.44 km-amagat.
- **Haze:** `haze_q12` = 2610 (0.637). Bio noon i0 = 426 W/m². Rendered ground PAR at the same transmission = 318 W/m², ratio 0.747.
- **Visibility:** estimated τ_h550 ≈ 0.7, giving transmittance 0.68 / 0.40 / 0.16 / 0.03 at 3.4 / 8 / 16 / 32 km.
- **Earth preset:** τ(550) = 0.0969.

**Cross-check with the archived seed-2 planet:** S_top 1702, PAR 38.3%, bands 103 108 107 103 97 89 81 73, H 16.85 km, τ_R(550) 0.605, θ 4.45 mrad. These reproduce physical-light's figures.

## B. Risks

1. **Column-tracer divergence or register pressure.** Mitigated by the split kernels, the M1 register check, and dropping to the LOD surface earlier.
2. **Clamp side effects.**
   - The [h, h+2) clamp leaves 0.95% of 4×4 neighbourhoods flat near 2 m steps; accepted.
   - The coast lock flattens `h = sea` beaches at the waterline. This is intended and reads as a tidal flat.
3. **Thick-sky multi-scatter error.** Covered by the Monte Carlo gate and the iterated-orders fallback.
4. **CH4 data quality.** Kept in a table file, with a fallback band model.
5. **Ghosting at 250 m/s and above.** Disocclusion rejection plus history reset on fast moves.
6. **Shared 3060 with the lab island.** Measure paused; stream priorities work only within one process.
7. **Spectral band error.** The ΔE check triggers the 12-band fallback.
8. **Phase B/C changes evolution.** New runs only, flag-gated, retuned efficiency, T13/T14.
9. **Future terrain edits.** An edit must rebuild the column's tile from the edited grid (T re-clamped to the new h).
10. **Collision.** Gameplay walking should use `world_surface_q8`, which is the drawn surface and stays within 1 m of the voxel truth.


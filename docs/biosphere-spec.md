# GME Biosphere: final implementation spec ("Seedbank on integer rails")

**Status:** sections 1–9 are the plan of the first release. Animals, sexual reproduction, learning, light in wavelength bands, plant shapes, the island hub, seed banks and the creature export were built after it and are described as built in **section 10**, which is the reference wherever it differs from the plan.

**Base design:** integration-delivery, the top-ranked design. It is all-integer, ships genesis stages, and does engine and validation work first.

**Grafted from ecology-emergence:**
- Pyrite sink and piston-velocity limit
- Tidal `wet_frac` bridge to land
- Kin-density pathogen and mate-tag speciation
- Ecological unit tests

**Grafted from gpu-throughput:**
- Evolvable think period
- Slot-pooled genomes with deduplicated checkpoints
- `bio-*` session names and `--duty`
- Distinct scouted terrain patches per GPU
- `rebase`
- Null controls and the brain-value metric
- An env-bin fallback table that vegetates the world outside the simulated regions

**Facts checked against the code in `<repo>`:**
- **Sea level is -35.** `genesis/home.planet` has `gen=-35 329 40 1284 1016 3 2 4 5 5 2600`, so sea_level = -35, relief_q8 = 329, soil_depth = 3, surface_mat = DIRT, beach_mat = SAND. **Nothing in this spec hard-codes sea level.** Every depth is `sea_level - h`, with sea_level read from the file.
- **Star and orbit.** The star is 6239 K at 1.84 Lsun, a = 1.212 AU, which gives a top-of-atmosphere flux S = 1361·L/a² ≈ 1701 W/m². Gravity is 0.603 g and t_surf is 304.0 K. The atmosphere exists only as text (`atmosphere=hydrogen sulfide 51%, methane 39%, ammonia 11%, oxygen O2 0%`). CO2 and H2 are absent.
- **Generation code.**
  - `gme_height`, `gme_material`, `gme_hash3` and `gme_noise2` are `GME_HD` and integer-only in `src/worldgen.h`.
  - `k_generate(jobs, out, meta, seed, GmePlanet planet)` calls `gme_height` once per column and then `gme_material` per voxel (`src/gpu.cu:59`).
  - `cpu_gen_chunk(cx,cy,cz,seed,pl,out,meta)` is in `src/world.c:193`.
  - `gpu_submit(g, buf, n, seed, planet)` is in `src/gpu.cu:190`.
  - `gpu_world_create(slots, max_batch)` is in `src/gpu.cu:145`.
- **Validation column check.** `world_validate` (`src/world.c:688`) takes the first voxel from the top that is not AIR, `ocean_mat` or `ocean_top_mat`, and requires it to equal `gme_height`.
- **Renderer.** `src/render.cu` has `albedo(int m)` at line 152 and a GRASS side-face blend at line 191. The hit voxel's absolute coordinate is `p.cx*32 + h.vx`.
- **NVRTC options.** `src/rtcuda.c:181` compiles with `-std=c++17 --use_fast_math -default-device` and no headers. The API is `rt_init, rt_compile, rt_func, rt_alloc, rt_upload, rt_download, rt_memset32, rt_launch(f, grid, block, shared, args), rt_sync`.
- **Kernel embedding.** CMake hex-embeds `cosmos_kernels.cu` to `COSMOS_KERNELS_SRC` inline, so there is no reusable embed function yet.
- **Genesis CLI.** `genesis_main.c`'s `main()` rejects unknown `--options`. Its `mats[MAT_COUNT]` name table is at line 323.
- **Search kill pattern.** `cluster/launch.ps1:38` kills every `^gme-[a-z0-9-]*` tmux session. Search sessions are named `gme-search<i>`.
- **status.ps1 deletes genesis/.** `cluster/status.ps1 -Fetch` **deletes `./genesis`** before copying a new system. Biosphere outputs therefore live in a new top-level `life/` folder, never in `genesis/`.
- **NVRTC on the rigs.** `genesis_seeds.sh` sets `LD_LIBRARY_PATH=$HOME/venvs/<venv>/.../nvidia/cu13/lib` for NVRTC.
- **Thread pool.** `jobs_parallel_for(count, grain, JobFn(ctx,begin,end,worker), ctx)`.

---

## 1. Architecture and data flow

```
genesis/home.planet ──planet_load──► HomePlanet (+ new atm_mix[], planet_hash)
        │
        ▼  gme_genesis env           (CPU, jobs_parallel_for, worldgen.h integer fns only)
life/regions.txt  (6 scouted 16.4 km patches)   runs/bio_<hash>/island_k/env.bin (+env.png, crc32)
        │
        ▼  gme_genesis life --island k --device d     (NVRTC kernels, integer-only sim)
runs/bio_<hash>/island_k/{ckpt_*.bin, metrics.csv, stages.log, maps/*.png, outbox/, inbox/}
        │            ▲   PC hub (cluster/bio_hub.ps1): migrants ring + global atmosphere
        ▼  gme_genesis bioexport (per island) → biopart_k.bin ; PC: bioexport --merge
life/home.biomap   (≤8 regions × 1024² cells u32 + 64-entry env-bin fallback + palette)
        │
        ▼  gme.exe (main.c loads it, hash-checked) → GmeBioView → k_generate / cpu_gen_chunk
voxels: MAT_GRASS / MAT_MAT / MAT_LEAF / MAT_WOOD / MAT_KELP   (bit-identical CPU≡GPU)
        │
        ▼  (M7) life/home.fauna → render overlay of motile organisms
```

**Process model.** Life is a set of new `gme_genesis` stages: `env`, `life`, `life-test`, `bioexport`, `biostat`, `biomap-test`, `life-rebase`, `life-seedbank`, `life-inoculate`. They are dispatched to `bio_main()` before the existing option parser runs, so they use their own options. `gme_genesis` builds with gcc only on the rigs and with MSVC on the PC. The engine never runs the simulation; it only reads the frozen `life/home.biomap`.

### New files

| File | Content | ~LOC |
|---|---|---|
| `src/flora.h` | `GME_HD` integer vegetation: `GmeBioRegion`, `GmeBioView`, `GmeColumn`, `gme_env_bin`, `gme_column`, `gme_voxel`, `gme_is_standing_flora` | 220 |
| `src/bio/bio_types.h` | Constants, gene enum, `BioGenome`, `BioParams`, `BioStats`, file headers; integer typedef shim under `__CUDACC_RTC__`; copy of `gme_hash3` as `bio_hash` | 300 |
| `src/bio/bio_core.h` | All per-agent and per-cell math as `BIO_FN` functions shared by gcc and NVRTC: metabolism, brain, mutation, LUT builders, fair-share, light | 900 |
| `src/bio/bio_kernels.cu` | Thin `extern "C" __global__` wrappers over `bio_core.h`, plus scan and sort kernels | 600 |
| `src/bio/bio_env.c/.h` | Region scout, static env, `env.bin` read/write, env PNG | 350 |
| `src/bio/bio_sim.c/.h` | Host driver: allocation (`rt_mem_info`), seeding, step loop, `--duty`, stats download, epochs | 600 |
| `src/bio/bio_cpu.c` | Serial CPU reference of the same step (calls the same `bio_core.h`) | 350 |
| `src/bio/bio_io.c/.h` | Checkpoint write/read (dedup, CRC32), migration files, biomap part/merge, maps | 500 |
| `src/bio/bio_stats.c` | Species clustering, Shannon/Simpson, ablation, `biostat` | 300 |
| `src/bio/bio_main.c` | CLI for all life stages | 250 |
| `src/bio/bio_test.c` | `life-test` suite (§8) | 400 |
| `cluster/bio_island.sh`, `bio_launch.ps1`, `bio_status.ps1`, `bio_hub.ps1` | Cluster | 300 |

### Changed files

- `src/worldgen.h`: append materials.
- `src/world.c/.h`: bio view, `cpu_gen_chunk`, `world_validate`.
- `src/gpu.cu/.h`: biomap upload; `gpu_submit` takes a view.
- `src/render.cu/.h`: albedo and palette.
- `src/main.c`: load the biomap, spawn point, `--biomap/--no-life`.
- `src/planetfile.c/.h`: `atm_mix`, `planet_hash`.
- `src/genesis/genesis_main.c`: stage dispatch, `mats[]` names, `finish` emits `atm=`.
- `src/rtcuda.c/.h`: add `rt_mem_info`.
- `CMakeLists.txt`: `gme_embed()` function, bio sources, ctest.
- `cluster/launch.ps1`: kill only `^gme-search`.

Total is about 5,300 lines.

---

## 2. Environment model

### 2.1 Planet inputs (`planetfile.h/.c`)

```c
enum { ATM_H2S, ATM_CH4, ATM_NH3, ATM_CO2, ATM_O2, ATM_N2, ATM_H2, ATM_N };
/* HomePlanet gains: */ double atm_mix[ATM_N]; int atm_known;
uint64_t planet_hash(const HomePlanet* p);
```

- **`atm=` key.** Parse `atm=f0 f1 ... f6` when present. Otherwise parse the `atmosphere=` text by gas name ("hydrogen sulfide", "methane", "ammonia", "carbon dioxide", "oxygen", "nitrogen", "hydrogen"; match the `CC_NAME` strings).
- **CO2 baseline.** If CO2 is missing, set it to 0.005 and renormalise. Mark it `/* tuned */`.
- **`finish`.** `run_finish` writes `atm=` from the `cm[]` fractions. Re-running `finish --dir genesis` updates the current file, because the chem shards are present.
- **`planet_hash`.** FNV-1a 64 over a canonical string built with `snprintf("%u %d %d..." )` of `system_seed`, `planet_index`, all 11 `gen` ints, `lround(t_surf*100)`, `lround(star_lum*1e4)`, `lround(a_au*1e4)` and `lround(atm_mix[i]*1e4)`. It ignores line endings, so the PC and the rigs agree.

### 2.2 Regions: islands as distinct real terrain patches

`gme_genesis env --scout` scans centres on a 2 km lattice within ±128 km of (0,0). For each window of 16,384 m it samples `gme_height(x,z,seed,&gen)` every 64 m, with `seed = system_seed` (= 2).

**Integer score:**
- `ocean_frac` must lie in [25%, 50%] (planet ocean cover is 11%, so this selects coasts).
- +2 per shelf sample (depth 1–60 m).
- +1 per land sample within 1 km of the coast.
- +8 per vent candidate (§2.4).

It picks the 6 best windows with centres at least 20 km apart. Islands 0–4 go to the rig GPUs and 5 is the PC lab. The result is written to `life/regions.txt`, one line per island: `k x0 z0`. The scout runs once on the PC and the file is deployed. Every derived number (depths, photic share) comes from sea_level as read from the planet file.

### 2.3 Grid

| Quantity | Value |
|---|---|
| Cell size | 16 m (`BIO_SHIFT=4`) |
| Grid | 1024×1024 cells (16.4 km) per island |
| Time step | dt = 1 h; 1 day = 24 steps; 1 sim year = 8760 steps |
| Island boundary | Reflecting edge |

**Static fields: `env.bin`.** These are computed on the CPU from 4×4 column samples per cell. They are integer-only and deterministic.

```c
typedef struct BioEnvCell {          /* 20 B, stored SoA */
    int16_t elev;        /* mean gme_height of 16 samples */
    int16_t depth;       /* sea_level - elev (<=0 on land) */
    uint8_t wet[3];      /* #samples(of 16)*16 with h <= sea_level + t, t = -1,0,+1 (tide) */
    uint8_t f_soil, f_ore, f_sulf, f_gran;  /* /255, gme_material at depth 1..8 below each sample */
    uint8_t vent;        /* 0..255 source strength */
    uint8_t flow_dir;    /* D8 downhill, 255 = sink/sea */
    uint8_t env_bin;     /* gme_env_bin() at cell centre (shared with engine fallback) */
    uint16_t coast_q;    /* distance to nearest ocean cell in cells (2-pass chamfer) */
    uint8_t pad[3];
} BioEnvCell;
/* env.bin header: "GMEENV01", ver, planet_hash u64, seed, x0, z0, shift, w, h, s_top_wm2 (int),
   t_surf_ck (int), crc32 of payload */
```

Each rig generates its own `env.bin`. `bio_launch.ps1` compares the CRC against the PC's and refuses to start on a mismatch.

### 2.4 Dynamic fields

Fields are int64 in mmol per cell, SoA, double-buffered for diffusion. They are int64 because an H2S-saturated mixed layer reaches about 10^11 mmol per cell.

`DIC, DOC, NH4, PO4, FE2, H2S, SO4 (S⁰+sulfate lumped), O2, CH4, H2, SOILW (L), DET` gives 12 fields. With the double buffer and 12 u64 demand sums per cell that is about 300 B/cell, roughly 310 MB per island.

The **active volume** of a cell is `256 m² × clamp(depth, 1, 10) m` for water, and `256 m² × 0.5 m` of soil for land.

**Derived per step:**
- Light: `PAR[cell][band 0..7]`, int32 W/m².
- `T_ck`, the temperature in centi-kelvin (int32).

**Global per island, in bulk:**
- `int64 atm[ATM_N]`, the island atmospheric box in mmol.
- Ledgers `SED_C, SED_S, SED_FE, IMPORT_C, VENT_C`.

**Vents.** A cell is a vent if `depth > 15 && f_sulf + f_ore > 96 && (bio_hash(i,j,0,seed) & 1023) < 64·density`. The target is about 1 vent per km²; the scout requires at least 20 vents.

### 2.5 Update equations

All arithmetic is integer: int64 intermediates, `>>` on q16 constants, and lookup tables built by integer routines (no libm, so the PC and the rigs build identical tables).

1. **Light** (`k_env_light`).
   - Hour of day: `hr = step % 24`.
   - `I0 = S_top · 0.45 · τ_haze · SIN24_Q12[hr] >> 12`. `SIN24_Q12` is a literal 24-entry table in `bio_core.h`.
   - `τ_haze_q12 = 4096 − 2048·x_CH4/(x_CH4 + 0.2)` (integer, in q12 mole fractions), which is about 0.55 now.
   - In water at agent depth z: `I = I0 · EXPN_Q16[(k_w·z) in 1/64 units] >> 16`, with `k_w = 0.06 /m + 0.02 /m per (mmol C/m³ phototroph)`. The self-shading term comes from the previous step's canopy sum.
   - On land, per canopy band b: `I_b = I0 · EXPN_Q16[0.5·LAI_above(b)]`.
   - Bands are height classes `{<0.5, 0.5–1, 1–2, 2–4, 4–8, 8–16, 16–32, ≥32 m}`.
2. **Temperature.**
   - `T_ck = t_surf_ck − 65·max(0, elev − sea_level)/10` (lapse 6.5 K/km), `+ diurnal`.
   - Diurnal swing is ±600 cK on land and ±50 cK at sea, using `SIN24`.
   - Below 20 m depth: `−5 cK/m`, down to −700.
   - Vent cell: `+1500`.
   - `fT = GAUSS_Q16[(T − T_opt)/T_tol]`. Maintenance uses `Q10_Q16[(T−30300)/100]` (Q10 = 2).
3. **Tide.** `t_idx = TIDE[(step mod 149)·...]`, which takes the values −1, 0, +1 over 12.4 h. A cell's wet fraction is `wet[t_idx+1]/256`. An agent on a coastal cell is **wet** iff `(bio_hash(uid,0,0,salt) & 255) < wet[t]`. Its hash is a fixed micro-elevation rank, so the tide sweeps the same agents every cycle. This is the daily selection gradient that bridges sea and land.
4. **Diffusion** (`k_env_diffuse`, every step).
   - `a'_i = a_i − Σ_{j∈4nbr} trunc((a_i − a_j)·D_ij >> 16)`. Truncation toward zero is antisymmetric, so mass is conserved to the bit.
   - Coefficients in q16: water–water 0.2 (13107), soil–soil 0.01, water–land 0.02. SOILW does not diffuse.
   - Non-motile planktonic agents get a hash random walk instead of advection.
5. **Sources and sinks** (`k_env_chem`). Rates are per step per cell and all are `/* tuned */` fields of `BioParams`.
   - **Vents:** `H2 += 40·v`, `H2S += 20·v`, `FE2 += 10·v`, `DIC += 20·v` (mmol, v = vent/255).
   - **Serpentinisation** on seafloor ore cells: `H2 += 2·f_ore/255`.
   - **Weathering** on land and seabed:
     - `FE2 += 0.5·f_ore`, `H2S += 0.3·f_sulf`, `PO4 += 0.05·(f_gran + f_soil)`.
     - `DOC += 0.2·f_soil·organic_q`, with organics 0.38% taken from the `crust=` text.
   - **Rain** (once per sim-day): `SOILW += R0·EXPN(coast_q·16/3000)`. Runoff moves along `flow_dir` to a sink or the sea.
   - **Air–sea and air–soil exchange (piston velocity).** For each gas g:
     - `flux = trunc((sat_g − aq_g)·kp_q16 >> 16)`.
     - `sat_g = H_g·p_g·V`, with H_g the Henry constants at 304 K as a `BioParams` table.
     - `kp`: 0.02 m/h over the mixed-layer depth for water (τ about 500 h for 10 m), and 0.5 /h for soil.
     - The low ocean `kp` is a deliberate knob: biota can draw dissolved H2S and O2 away from equilibrium locally.
     - Each block accumulates `−flux` into int64 `atm_partial[block][g]`.
   - **Pyrite sink:** `py = min(H2S/2, FE2, (H2S·FE2·k_py) >> 40)`, then `FE2 −= py`, `H2S −= 2py`, and `SED_S += 2py`, `SED_FE += py`. This is the Fe-limited sulfide sink that lets the ocean lose H2S far from vents.
   - **Abiotic oxidation:** `O2 + H2S → SO4`, `2O2 + CH4 → DIC`, `O2 + 4FE2 → SED_FE` (Fe³⁺ sediment). Each is a second-order integer rate.
   - **Burial** (daily, ocean cells with depth > 30): `SED_C += DET >> 9`, which is about 0.2%/day. Burial is the only lasting net oxidant source, as on Earth.
   - **Mineralisation:** `DET → DOC` at 1/64 per day; N goes straight back to NH4.
   - **Atmospheric photochemistry** in the island box: `O2 + CH4` and `O2 + H2S` at small integer rates. The ozone shield `uv_q8 = 255·0.001/(x_O2 + 0.001)` is a lookup on `x_O2` (ppm).
6. **Island atmosphere.** `k_atm_reduce` sums `atm_partial` in int64, which is exact and order-free. `atm[g] += Σ partials`.
   - **Box size.** `atm` initialises to `column_mol_per_m2 · area_region · atm_scale · mix[g]`, where `column = P/(g·M̄)` with `P = 1 bar /* tuned */`. That is about 6.6×10⁵ mol/m², so the island box is about 1.8×10¹⁴ mol × `atm_scale`.
   - **Default `atm_scale = 1e-4`.** It is logged in the checkpoint header and called out like the README's other compressions.
7. **Global atmosphere feedback through the PC hub.**
   - Every epoch, each island writes `atm_<epoch>.txt`: its per-m² net exchange, split into ocean cells and land cells.
   - The hub computes the planet delta as `ocean_cover·mean(ocean_deltas) + (1−ocean_cover)·mean(land_deltas)`. Planet ocean cover is 0.11, so the ocean-biased islands do not over-weight marine flux.
   - The hub updates `life/hub/atm_global.txt` as int64 mmol/m² × scale and pushes it back.
   - Islands relax at each epoch: `atm += (global·area − atm) >> 2`.
   - If the file is stale, an island keeps running on local values.
   - The hub prints **"years to O2 rise at current net reductant flux"**. Whether oxidation is reachable is therefore a measured number and a knob (`atm_scale`), not a hope.

---

## 3. Organism model

### 3.1 Units

| Quantity | Unit | Notes |
|---|---|---|
| Carbon C | mmol C, int32 | Biomass is carbon; cap 1e9 mmol ≈ 12 t C per agent |
| Energy E | eu = 1 mmol ATP-equivalent, int32 | |
| Water W | q8 of capacity | |
| N | mmol | C:N = 6 for building biomass, drawn from NH4 |
| P | mmol | C:P = 106 for building biomass, drawn from PO4 |

An agent is a **super-individual**: a clonal colony (mat, plankton, algae, plant ramet) or a single motile individual. There is no "type" field; genes decide everything.

### 3.2 Genome (`BioGenome`, 384 B, immutable for an agent's life, stored in a slot pool)

```c
#define BIO_IN 16
#define BIO_H 8
#define BIO_OUT 8
typedef struct BioGenome {
    uint8_t  body[64];                 /* gene table below */
    int8_t   w_ih[BIO_H][BIO_IN];      /* 128 */
    int8_t   w_hh[BIO_H][BIO_H];       /*  64  full recurrent */
    int8_t   b_h[BIO_H];               /*   8 */
    int8_t   w_ho[BIO_OUT][BIO_H];     /*  64 */
    int8_t   b_o[BIO_OUT];             /*   8 */
    uint16_t mate_tag;
    uint8_t  pad[46];                  /* = 384, 16-B aligned */
} BioGenome;
```

**Body genes** (uint8; the enum in `bio_types.h`):

| idx | gene | meaning |
|---|---|---|
| 0–7 | `E_METH, E_ANOX, E_OXY, E_FERM, E_SULFRED, E_AERO, E_METHOX, E_FEOX` | Enzyme expression, 0..255 |
| 8 | `G_SIZE` | Reproduction mass `m_r = 2^(g/10)` mmol C (1 … 3.4e7) |
| 9 | `G_STRUCT` | Share of growth into structure S (height or wood) |
| 10 | `G_HOLDFAST` | Anchored (no drift); benthic or rooted |
| 11 | `G_CUTICLE` | Desiccation resistance |
| 12 | `G_ROOT` | Soil water and nutrient uptake on land |
| 13 | `G_BUOY` | Water-column depth: `z = depth·(255−buoy)/255` |
| 14 | `G_MOTOR` | Speed and attack strength |
| 15 | `G_MOUTH` | Maximum prey/self mass ratio |
| 16 | `G_ARMOR` | Predation defence |
| 17 | `G_PIGMENT` | Light-harvesting amount |
| 18 | `G_HUE` | Pigment colour (exported palette; also sensed as a signal) |
| 19 | `G_UVSHIELD` | UV protection |
| 20 | `G_SENSE` | <128: own cell only; ≥128: 3×3 cells (cost) |
| 21 | `G_REPRO_E` | Energy needed to reproduce, `·4` eu per mmol of `m_r` |
| 22 | `G_OFFS` | Fraction of C and E given to offspring (16..240)/256 |
| 23 | `G_DISPERSE` | Offspring distance 0..`g/16` cells (spores, larvae) |
| 24 | `G_MUT` | Per-byte mutation probability `2^-(6 + g/32)` (1/64 … 1/16384) |
| 25 | `G_ETA` | Learning rate |
| 26 | `G_DECAY` | Decay of learned deltas toward the genome |
| 27 | `G_PLASTIC` | Bitmask: which output rows are plastic |
| 28 | `G_NEURONS` | Bitmask: which hidden neurons are on |
| 29 | `G_THINK` | Think period `1 + (g>>2)` steps (1..64) |
| 30 | `G_TOPT` | T_opt = 273 + g/4 K |
| 31 | `G_TTOL` | T_tol = 1 + g/16 K |
| 32 | `G_O2TOL` | O2 tolerance |
| 33 | `G_SEX` | ≥128: sexual (crossover with a compatible cellmate) |
| 34 | `G_LIFESPAN` | 2^(g/16) days |
| 35 | `G_KINSHARE` | Energy shared with kin cellmates (output 7 gate) |
| 36 | `G_STORE` | Reserve allocation |
| 37–47 | reserved | Neutral until used |
| 48–63 | neutral loci | Molecular clock and phylogeny |

Every non-neutral gene above 0 has a maintenance cost (§3.5), so versatility is expensive and guilds emerge.

### 3.3 Agent state (SoA in VRAM)

**Hot record, 72 B:**
```
int32 x_q8, z_q8;            // metres q8 inside region
uint32 cell;                 // sort key
int32 C, S, E, N, gut;       // carbon, structural C, energy, nitrogen, ingested C (internal DOC)
int16 W; uint16 age_h4, flags; int16 rbase;   // flags: alive|wet|land|due|dead_this_step
uint32 gslot, uid;
int8  hid[8], out[8];
uint64 claim;                // atomicMin predation claim
```

**Cold record, 76 B:** `int8 dW[8][8]` (learned deltas, output layer only), `uint32 lineage, parent, birth_step`.

**Totals:**
- Per agent: 72 + 76 + 384 (genome slot) ≈ **532 B**.
- Capacity from `rt_mem_info`:

| Card | Agent cap | Memory |
|---|---|---|
| 3060 | 2M | 1.1 GB |
| 4090 | 4M | 2.2 GB |
| 4080 | 3M | — |

  Fields add about 310 MB. Everything fits next to the search, whose cosmos buffers are under 200 MB.

### 3.4 Brain

Each step, `k_think` runs only for agents with `(step + uid) % period == 0`.

**Inputs (int8 q7), 16:**

| # | Input | # | Input |
|---|---|---|---|
| 0 | E/E_repro | 8 | gradient z |
| 1 | PAR at own band/depth | 9 | crowding (cell count / 32) |
| 2 | (T − T_opt)/T_tol | 10 | kin share (own-bucket count / cell count) |
| 3 | wet/W | 11 | threat dx (largest motile mouth-capable neighbour within 3×3, from summaries) |
| 4 | O2 | 12 | threat dz |
| 5 | H2S | 13 | prey dx (best prey biomass within 3×3) |
| 6 | DOC + DET | 14 | prey dz |
| 7 | gradient x of the agent's best substrate (from the 3×3 cell fields) | 15 | day phase (SIN24) |

All sensing reads the per-cell summary records (§5). It is order-free and costs about 300 B.

**Network:**
```
h_pre_j = 64·b_h[j] + Σ_i w_ih[j][i]·in[i] + Σ_k w_hh[j][k]·hid_prev[k]      (int32, __dp4a)
hid_j   = neuron_on(j) ? TANH_Q7[clamp(h_pre_j >> 7)] : 0
out_o   = TANH_Q7[clamp((64·b_o[o] + Σ_j (w_ho[o][j] + dW[o][j])·hid_j) >> 7)]
```

**Outputs, 8:**

| # | Output |
|---|---|
| 0 | move x |
| 1 | move z |
| 2 | attack |
| 3 | reproduce gate |
| 4 | growth allocation (structure vs reserve) |
| 5 | depth/buoyancy change |
| 6 | spore release timing / dispersal gate |
| 7 | share with kin |

Sessile autotrophs use outputs 3, 4 and 6, so plant behaviour (allocation, timing) is brain-driven from M2.

**Brain cost** per think: `0.05 eu × popcount(G_NEURONS) + 0.01 eu × G_SENSE/255`. Combined with `G_THINK`, cheap slow brains evolve for sessile life and fast ones for hunters. This is also the main throughput lever.

**Learning: neuromodulated Hebbian, Baldwinian.** (This rule failed test T9 and is kept only for worlds that have not switched the `learn` feature on; the rule that replaced it is in 10.4.)
- Reward: `m = clamp(((E_now − E_lastthink) − rbase)·128 / (1 + C/64), −127, 127)`, then `rbase += (r − rbase) >> 3`.
- For each plastic row o and active hidden j: `dW[o][j] += SR((ETA·m·out_o·hid_j) >> 21) − SR((dW·DECAY) >> 8)`, clamped to ±32.
- SR is stochastic rounding with `bio_hash(uid, step, o·8+j, salt)`.
- Learning costs `0.002 eu × popcount(G_PLASTIC)·8·ETA/255` per think.
- Offspring start with `dW = 0`. A `--lamarck` flag exists for experiments only.

### 3.5 Metabolism (`k_metab`, after fair-share grants)

**Rate law** for each enzymatic pathway:
`v_k = C · (e_k/255) · Vmax_k · Π_s MM(S_s, K_s) · fT`, with `MM(S,K) = S·65536/(S+K)` computed in int64. The **requested** v_k is posted as demand in `k_demand`, and the granted amount is `req·ratio >> 16` (§5).

| path | reaction (per mmol flux) | Vmax (/mmol C /h) | energy | C to body |
|---|---|---|---|---|
| METH | 4 H2 + DIC → CH4 | 0.5 | +1 eu per CH4 | autotroph: may fix DIC at 8 eu/mmol C |
| ANOX | light; fixes C using 2 H2S → 2 SO4 per C | light-limited | `eu = I·A_leaf·3600·0.03/50` | fix C at 8 eu/mmol |
| OXY | light; fixes C from H2O, releases 1 O2 per C | light-limited | `eu = I·A_leaf·3600·0.05·(psi)/50`, psi = `K_tox/(K_tox + H2S)` (sulfide inhibition) | fix C at 8 eu/mmol |
| FERM | DOC/gut → ½ DIC + ½ CH4 | 0.3 | +0.5 eu per mmol C | assimilate at 2 eu/mmol |
| SULFRED | DOC/gut + ½ SO4 → DIC + ½ H2S | 0.2 | +1 eu per mmol C | assimilate at 2 eu/mmol |
| AERO | DOC/gut + O2 → DIC | 0.3 | +5 eu per mmol C | assimilate at 2 eu/mmol |
| METHOX | CH4 + 2 O2 → DIC | 0.2 | +3 eu per mmol CH4 | assimilate CH4-C at 2 eu/mmol |
| FEOX | 4 FE2 + O2 → SED_FE | 1.0 | +0.1 eu per FE2 | autotroph fix at 8 eu/mmol |

**Notes on pathways:**
- **Leaf area.** `A_leaf = C·(PIGMENT/255)·SLA`, with SLA = 2.4e-4 m²/mmol C. It is entered into the canopy bins at height `H = icbrt(S/4000)` m (capped at 60 m). A mat or plankton has S = 0, so H = 0.
- **Sulfur cycle.** ANOX, SULFRED and pyrite together close it.
- **Ingested prey.** Prey carbon goes to `gut` and is processed by the predator's own FERM/SULFRED/AERO without competition. Predation therefore needs a respiration pathway, and aerobic predators get 10× the yield of fermenting ones.

**Growth.** Available energy is spent on C from fixation or assimilation, limited by N (C:N 6) and P (C:P 106) grants from NH4 and PO4. New C goes into S (structure) or C (reserve) according to `out[4]` and `G_STRUCT`.

**Costs per step** (eu, integer, with stochastic rounding):
- Maintenance: `0.002·C·Q10(T)·(1 + crowd/16)`. The crowding term is a soft, order-free density cap.
- Structure: `0.0002·S`.
- Enzymes: `0.001·C·Σ(e_k/255)²`. The quadratic penalty drives specialisation.
- Traits: `0.0005·C·(UVSHIELD + CUTICLE + ARMOR + ROOT)/255`.
- Motion: `C·v²·k_med`, with k_med = 1 in water and 3 on land, plus `C·0.6·0.02·v` for weight on land at 0.6 g.
- O2 toxicity for anaerobes: `0.01·C·O2/(O2 + K_O2)·(255 − O2TOL)/255`.
- UV on exposed agents (land, or water with z < 3 m): `0.01·C·uv_dose·(255 − UVSHIELD)/255`, with `uv_dose = uv_q8·EXPN(z/1.5 m)`.
- Desiccation on land, or tidal cells when not wet: `W −= dry·(255 − CUTICLE)·A/…`. W refills from SOILW through ROOT (fair-share) and fully when wet.

**Starvation.** If E < 0, the agent catabolises C at 4 eu/mmol (with O2 and AERO) or 0.5 eu/mmol otherwise.

### 3.6 Reproduction and mutation (`k_birth`)

**Condition.** All of the following must hold:
- `C ≥ m_r`
- `E ≥ REPRO_E·m_r/64`
- `out[3] > 0`
- the target cell's count from this step's sort is below `BIO_CELL_HARDCAP = 64`

This one rule covers both colony splitting and offspring.

**Offspring.**
- It gets `G_OFFS` of the parent's C, N and E, and **pays the copy cost of 0.02 eu per genome byte** (7.7 eu). That is the literal "self-copying algorithm with an energy budget".
- Position: the parent's position plus a hash-random offset of up to `G_DISPERSE/16` cells. On land, spores use a wind direction from `bio_hash(step/24)`.
- One offspring per parent per step.

**Mutation.**
- Each of the 336 non-pad bytes mutates with probability `2^-(6 + G_MUT/32)`: `±(1 + (h&7))` with probability 15/16, otherwise a uniform reset.
- `mate_tag` mutates one bit with probability 1/64.
- Macro mutation (p = 1/2048) is one of:
  - copy one hidden neuron's in/out rows over another (duplication);
  - toggle a bit in `G_NEURONS` or `G_PLASTIC`;
  - jump one enzyme gene by ±64. This lets OXY and new pathways cross fitness valleys.

**Sex.** If `G_SEX ≥ 128`, the partner is the first agent in the parent's ordered cell list (§5) that has `G_SEX ≥ 128`, `popcount(tag^tag') ≤ 3` and is not the parent. Crossover is uniform over 16-byte blocks, selected by hash bits. Speciation emerges from geography (islands, depth, coast) plus tag drift; no code assigns species.

### 3.7 Death

An agent dies when any of these holds:
- `C < m_r/64` after starvation
- `W < 0`
- `age > lifespan`
- it is eaten
- `|T − T_opt| > 3·T_tol`
- a background hazard of 1e-4/h
- the **kin-pathogen** hazard (Janzen–Connell) `0.002·k/(k + 20)` per day fires, where k is the number of cellmates in the agent's LSH bucket. The bucket is the top 4 bits of a hash of the quantised enzyme and morphology genes. This is the explicit anti-monoculture force.

On death, C goes to DET and N to NH4 by atomic add, and the genome slot is freed.

### 3.8 Predation (`k_claim` → `k_predate`)

*The plan. Predation was built differently (per-cell slots instead of ordered lists, edibility by individual size, a refuge, hunger): see 10.1.*

- An attacker needs `out[2] > 0`, `MOTOR > 0` and `MOUTH > 0`. It picks the best prey among the first 16 agents of its ordered cell list with `C_prey ≤ C_self·MOUTH/32`, then does `atomicMin(&prey.claim, (bio_hash(att,step)<<32) | att_uid)`. This is deterministic and gives one winner per prey. Each attacker claims at most one prey.
- The prey thread resolves the claim. Success probability is `p = SIGMOID_Q16[(MOTOR_att − MOTOR_prey)/8]·(300 − ARMOR)/300`, rolled with a hash.
- If the prey is a colony (`HOLDFAST > 128` or `C_prey > 4·bite`), `bite = min(C_prey, C_att/5)` moves to the attacker's `gut`.
- Otherwise the prey dies: 70% of its C goes to `gut`, 30% to DOC.
- Trophic level is a q4 exponential moving average: `troph = (3·troph + troph_prey + 16)/4`.

---

## 4. Staged emergence (marine → land plants → land animals)

**Seed (step 0).** 20,000 colonies of one **protocell genome**, placed only in habitable cells: vent cells, plus seafloor with depth 2–60 m (relative to file sea_level), 273–330 K, and H2 + H2S above the median.

The protocell genome:
- `E_METH = 160`, `E_FERM = 40`, all other enzymes 0
- `G_SIZE = 20` (4 mmol), no pigment, no motor, `G_THINK = 124` (period 32), `G_MUT = 64` (about 1/256 per byte)
- random brain within ±4, `G_NEURONS = 0x0F`

This is an **H2 + CO2 chemoautotroph**: it lives on vent and serpentinisation H2 and does not depend on the finite abiotic DOC. That fixes the fermenter-only seed flaw.

**Optional window.** `--abiogenesis-days N` reseeds 10 protocells per day at vents until the population reaches 1e5. It is logged as `INTERVENTION abiogenesis` in `stages.log`.

**Why the order emerges** (each step is driven by physics, not scripted):
1. **Marine chemotrophs → anoxygenic phototrophs.** Drift and dispersal carry lineages into the photic shelf. ANOX (continuous genes) pays wherever H2S and light coexist. Its SO4 output feeds SULFRED decomposers, and death feeds FERM: the first cross-feeding.
2. **Oxygenic photosynthesis.** OXY is costlier and sulfide-inhibited. It wins where the pyrite sink and ANOX have drawn H2S down: shallow water far from vents. The slow piston velocity allows local O2 pockets. O2 poisons anaerobes and opens AERO (10× yield), which makes motile phagotrophs (animals) affordable in water.
3. **Land plants (no O2 required).** Tidal cells select for CUTICLE and HOLDFAST in intertidal mats. Spores (DISPERSE) reach moist coastal soil (SOILW, DOC, PO4, full light). UVSHIELD lets them tolerate UV without ozone, at a cost. Canopy light competition then selects for G_STRUCT height: mats → ground cover → shrubs → trees. If O2 does rise, ozone lowers the UV tax and land expands further.
4. **Land animals.** Motile heterotrophs follow land producer biomass. They need CUTICLE, MOTOR and ARMOR, and enough yield to pay for 0.6 g locomotion. In practice that means AERO, so they track O2 (gate G4). If O2 stays near zero, SULFRED or FERM grazers on land are still possible but marginal. That outcome is reported as a result.

**Observed gates.** These are computed in `k_stats` and appended to `stages.log`. They never trigger behaviour.

| Gate | Condition |
|---|---|
| G0 established | live > 1e5 for one sim-year |
| G1 marine photosynthesis | ANOX + OXY ≥ 50% of C fixation for 5 sim-years |
| G2 oxygenic | OXY ≥ 10% of GPP in any 64×64 tile, and local O2 > 1 µM |
| G3 land plants | Sessile phototrophs cover > 1% of dry land cells (`wet[2] == 0`) at cover ≥ 32 |
| G4 land animals | ≥ 1000 motile (MOTOR > 128) heterotrophs on dry cells, > 50% of their energy from PHAGO of land producers, persisting > 5 generations |
| G5 oxidation (GOE) | Global atmospheric O2 > 1% (hub) |

**Anti-collapse safeguards** (all deterministic, all logged):
- **Order-free fair-share uptake.** Pools can never be overdrawn.
- **Crowding maintenance** plus the hard per-cell birth cap.
- **Kin-pathogen hazard.**
- **Capacity guard.** Births are refused above 0.95·cap, logged as `CAP`.
- **Monoculture alarm.** The top species cluster holding > 80% of biomass for 50 sim-years is logged as `ALARM`; nothing is done automatically.
- **Extinction handling.** When live = 0, `life` exits with code 3. `bio_island.sh` restarts from the previous checkpoint with a new `--salt` and logs `INTERVENTION rollback n`. The rollback count is a reported metric, so real extinctions are never hidden.
- **Per-island barrier sweeps.** Each island k gets multipliers on desiccation, UV and land-locomotion costs: islands 0..4 use ×0.7, 0.85, 1.0, 1.15, 1.3. They are recorded in the header, which turns "nothing reaches land" into a measured parameter study.
- **Operator "introduce" step.** For the user's "we introduce plants, then land life" step by step: `gme_genesis life-inoculate --from runs/.../island_2 --guild land_plant --to island_4 --n 500`. It copies **evolved genomes only** into matching-habitat cells and logs `INTERVENTION inoculate`. It never writes behaviour.

---

## 5. GPU execution

### 5.1 Packaging under NVRTC constraints

- **Embedding.** In CMake, `function(gme_embed OUT VAR)` takes a list of sources, concatenates them, hex-encodes them as cosmos does today, and writes `${CMAKE_BINARY_DIR}/gen/${OUT}`. `cosmos_kernels.inc` is regenerated through the same function, so behaviour is unchanged.
- **Bio kernel source.** `gme_embed(bio_kernels.inc BIO_KERNELS_SRC src/bio/bio_types.h src/bio/bio_core.h src/bio/bio_kernels.cu)`. `bio_sim.c` calls `rt_compile(BIO_KERNELS_SRC, "bio_kernels.cu")`.
- **`bio_types.h` shim:** `#ifdef __CUDACC_RTC__` typedefs `int8_t … uint64_t` and defines `BIO_FN` as `static __forceinline__`. Otherwise it `#include <stdint.h>` and defines `BIO_FN` as `static inline`. Atomics are macros: `BIO_ADD64(p, v)` is `atomicAdd((unsigned long long*)p, (unsigned long long)v)` on the GPU and `*p += v` on the CPU. `BIO_DP4A` is `__dp4a` on the GPU and a C loop on the CPU.
- **Kernels:** `extern "C" __global__`, no includes, no libm, **no float or double tokens**, which is enforced by test T1. `--use_fast_math` is therefore irrelevant, and the 4090, 4080, 3060 and CPU produce bit-identical results.
- **Lookup tables** (`EXPN_Q16[1024]`, `TANH_Q7[256]`, `GAUSS_Q16[256]`, `Q10_Q16[128]`, `SIGMOID_Q16[256]`) are built at init by integer routines in `bio_core.h` (fixed-point series) and uploaded. `SIN24_Q12` and `TIDE` are literal tables.
- **RNG:** stateless `bio_hash(uid, step, purpose, salt)`, a copy of `gme_hash3`. Resume is exact.
- **`rtcuda` addition:** `int rt_mem_info(size_t* free, size_t* total)` via `cuMemGetInfo_v2`, about 10 lines. Nothing else is needed: `rt_launch` 1-D grids and `rt_memset32` suffice.

### 5.2 Kernel order per step

Host-side state per step: `n_slots` (live plus holes) and `n_cells = 1M`.

| # | kernel | grid | work |
|---|---|---|---|
| 1 | `k_env_light` | cells | I0, tide index, T per cell |
| 2 | `k_env_diffuse` | cells | 12 fields, antisymmetric truncating flux, double-buffer swap |
| 3 | `k_env_chem` | cells | Vents, weathering, rain (daily), runoff (daily), exchange → `atm_partial`, pyrite, abiotic oxidation, burial (daily) |
| 4 | `k_atm_reduce` | 1 block | int64 sums → `atm`, ozone and haze factors |
| 5 | `k_bin_count` → `k_scan_a/b/c` → `k_bin_scatter` → `k_bin_order` | agents / cells | Counting sort by cell. `k_bin_order` does an insertion sort of each cell slice by `bio_hash(uid, step)`, so cell lists are deterministic despite atomic scatter order |
| 6 | `k_cell_clear` + `k_cell_summary` | cells / agents | Per-cell 32 B: count, biomass, LAI[8] (u16), kin buckets (16×u8 saturating), max predator `MOUTH·C` key, prey biomass. All are atomics of commutative or max type |
| 7 | `k_canopy_light` | cells | `PAR[cell][band]` from LAI above each band |
| 8 | `k_think` | agents | Early-out unless due; sense from summaries, MLP, Hebbian; writes `out`, `hid` |
| 9 | `k_move_claim` | agents | Motile movement and drift, land/water flags, wet test; predation claims (`atomicMin` u64) |
| 10 | `k_demand` | agents | Requests per pool → `BIO_ADD64(demand[cell][pool])` |
| 11 | `k_supply` | cells | `ratio_q16[cell][pool] = demand ≤ pool ? 65536 : (pool<<16)/demand` |
| 12 | `k_metab` | agents | Grants (floor, so Σ ≤ pool), pathways, growth, costs, damage; products and consumption via atomic adds to fields |
| 13 | `k_predate` | agents (prey) | Resolve claims, transfer to the attacker's `gut` |
| 14 | `k_fate` | agents | Death → DET, NH4, free flag; birth flag |
| 15 | `k_scan(birth)` → `k_birth` | agents | Offspring slot = `n_slots + prefix`; genome slot = `free_stack[top − 1 − prefix]`; copy, mutate, crossover; thread per birth (384 B) |
| 16 | `k_scan(dead)` → `k_free_push` | agents | Push dead genome slots onto the free stack in prefix order |
| 17 | `k_compact` | agents | Every 64 steps, or when holes exceed 10%: scan alive → gather hot and cold into the alternate buffers, swap |
| 18 | `k_stats` | blocks | Every 24 steps: int64 reductions into `BioStats` (about 4 KB downloaded) |

- **Host sync.** Each step downloads 8 bytes (`n_slots`, births); about 20 µs.
- **Epoch work** (every sim-year): sample 4096 genomes, write maps, run migration I/O and the atmosphere exchange. Checkpoints are time-based.
- **`--duty d`.** After each batch of 24 steps, the host sleeps `t_batch·(1−d)/d`. The default is 0.6 on the rigs and 0.5 on the PC.

### 5.3 Cost estimate: 1M agents on an RTC 3060, 1024² cells

| part | bytes/step | time at ~210 GB/s effective |
|---|---|---|
| env diffuse + chem + light (12 int64 fields r/w + neighbours) | ~220 MB | 1.05 ms |
| sort + order + summaries (1M × 24 B + cells × 48 B) | ~75 MB | 0.4 ms |
| think (≈1/8 due × (384 genome + 300 summary + 150 state)) | ~105 MB | 0.5 ms |
| demand / supply / metab (hot 72 B r/w, 12 grants, field atomics) | ~250 MB | 1.2 ms (+ atomic contention) |
| predate / fate / birth (1–2% births × 384 B) / free | ~40 MB | 0.2 ms |
| ~22 launches | | 0.15 ms |
| **total, card alone** | | **≈3.5–4.5 ms** |
| with 2 cosmos processes sharing the card, `--duty 0.6` | | **≈8–10 ms → 100–125 steps/s** |

**Throughput:**
- One sim-year (8760 steps) takes about 70–90 s, so a 3060 island makes about **1000–1200 sim-years per wall-day**.
- With colony generation times of 1–3 days, that is about 10⁵ microbial generations per day.
- The 4090 at 4M agents is about 1.5× the 3060 per step.

M2 must benchmark this. Exit check: on the PC 3060 at 1M agents with no search running, at least 150 steps/s; otherwise profile before booting.

---

## 6. Cluster

### 6.1 Island map

| island | host | device | cap | duty |
|---|---|---|---|---|
| 0 | rig A | 0 (4090) | 4M | 0.6 |
| 1 | rig A | 1 (4080) | 3M | 0.6 |
| 2, 3, 4 | rig B | 0, 1, 2 (3060) | 2M | 0.6 |
| 5 (lab) | PC | 0 (3060) | 1M | 0.5, optional; tests and sweeps |

The star search keeps all 11 `gme-search*` slots.

### 6.2 Run directory

`~/gme/runs/bio_<planet_hash>/island_<k>/` contains:
- `env.bin`
- `ckpt_<step>.bin`
- `metrics.csv`
- `stages.log`
- `maps/`
- `outbox/`, `inbox/`, `imported/`

### 6.3 Checkpoint `GMEBIO02`

- **Header (512 B):** magic, version, planet_hash u64, env_crc u32, island, salt, step u64, n_live, cap, `BioParams` (including atm_scale, sweep multipliers and seed), `atm[ATM_N]` i64, ledgers (`SED_*`, `IMPORT_C`, `VENT_C`, cumulative fluxes), last hub epoch applied.
- **Sections**, each `tag[4] u64 bytes crc32 payload`:
  - `FLDS`: 12 int64 fields
  - `HOT_`: compacted live agents
  - `COLD`: deltas, lineage
  - `GTAB`: unique genomes, deduplicated by FNV-64 plus memcmp; clonal populations collapse about 10×
  - `GIDX`: u32 index per agent
  - `INPT`: the migrant files and atmosphere values applied, by epoch, so a resumed island replays them identically
- **Writing:** download to pinned host memory, then a worker thread writes `.tmp`, fsyncs and renames. This happens every 30 wall-minutes and on SIGTERM/SIGINT. The last 3 are kept, plus one per 6 h. Expected size: about 0.4–1 GB on a 3060 and up to 2 GB on the 4090.
- **Resume.** `life --resume` loads the newest checkpoint whose CRCs are valid. It **refuses** a planet_hash or env_crc mismatch.

### 6.4 Migration and hub (PC-centred; no rig-to-rig SSH assumed)

*As built: 10.7.*

- **Island side.** Each epoch, an island writes `outbox/mig_<k>_<epoch>.bin`: 256 genomes per habitat class (marine, coastal, land), chosen as the top lineages by biomass (deterministic), with lineage IDs. At the next epoch it imports `inbox/*` sorted by name. Each migrant goes into a random matching-habitat cell with C = m_r and E = REPRO_E. Imported carbon is logged in `IMPORT_C`, and the files move to `imported/`.
- **`cluster/bio_hub.ps1`** (PC loop, every 10 min):
  - scp each island's outbox to `life/hub/in/`
  - push each file to ring neighbour k+1 and to one random island
  - merge `atm_<epoch>.txt` into `life/hub/atm_global.txt` and push it to all islands
  - print titration ETA, gates and species totals
- Islands never wait. If the PC sleeps, they run closed.

### 6.5 Scripts

- **`cluster/launch.ps1`:** change line 38 to `grep -o "^gme-search[0-9]*"`, so only search sessions are killed.
- **`cluster/bio_launch.ps1 [-Deploy] [-Duty 0.6] [-Islands …]`:**
  - scp `src`, `CMakeLists.txt`, `cluster`, `genesis/home.planet` and `life/regions.txt` to `~/gme`, then build with the same cmake line (`-DGME_BUILD_ENGINE=OFF`).
  - Per rig, run `gme_genesis env --island k` for its islands, and compare the CRC with `life/env_crc.txt` from the PC.
  - Start tmux `bio-isl<k>`.
  - Never touch `gme-*` sessions.
- **`cluster/bio_island.sh <dev> <island> <hash> <duty> <cap>`:**
  - Use the same `LD_LIBRARY_PATH` line as `genesis_seeds.sh`.
  - `while true; do ./build/gme_genesis life --device D --island K --run runs/bio_H/island_K --resume --duty X --cap C; rc=$?; ...`
  - On rc 3 (extinct): roll back with a new salt and log the intervention. On any other non-zero rc: sleep 30 and resume.
- **`cluster/bio_status.ps1 [-Fetch]`:**
  - Per island: `tmux ls | grep bio-`, `nvidia-smi`, the last metrics row, the `stages.log` tail, steps/s, and search seeds/hour from `runs/progress.log` as a coexistence check.
  - `-Fetch`: runs `gme_genesis bioexport --run …` on each rig, which writes `biopart_k.bin`. It scp's the parts, metrics and PNGs into `life/islands/k/`, then runs `gme_genesis bioexport --merge life/islands --out life/home.biomap` locally.

### 6.6 Better planet found ("let the cards keep running")

*As built: 10.8.*

1. `status.ps1 -Fetch`, then `chem` and `finish` produce a new `home.planet` with a new hash.
2. `gme_genesis life-seedbank --runs runs/bio_<old>` produces `seedbank.bin`: the top 2000 genomes per guild × habitat, deduplicated.
3. `env --scout` runs on the new planet.
4. `gme_genesis life-rebase --bank seedbank.bin --run runs/bio_<new>/island_k` seeds the genomes into matching habitats. Genomes carry no planet state.
5. Optionally the new run starts from protocells instead, for a "pure" run.

The old runs can keep going in parallel. Hash guards prevent wrong-world resumes.

---

## 7. Engine integration (bit-identical)

### 7.1 Materials (`worldgen.h`, appended; existing indices unchanged)

```c
MAT_REGOLITH, /* existing = 10 */
MAT_MAT,      /* 11 microbial / algal mat — replaces the ground voxel (depth 0) */
MAT_LEAF,     /* 12 standing: leaves, low cover tufts */
MAT_WOOD,     /* 13 standing: trunks */
MAT_KELP,     /* 14 standing: in water column, replaces water */
MAT_COUNT
```

- `MAT_GRASS` (3) finally gets used: it replaces the depth-0 surface voxel.
- `gme_is_standing_flora(m)` is true for `LEAF`, `WOOD` and `KELP`. `GRASS` and `MAT` are ground-replacing, so the surface height stays at `h`.
- Update `genesis_main.c`'s `mats[MAT_COUNT]` with "microbial mat", "leaf", "wood", "kelp".

### 7.2 `life/home.biomap`

**Header:**
```c
"GMEBMAP1", ver, planet_hash u64, seed u32, sim_year, gates u32,
nreg (≤8), shift=4, reg[8] {x0, z0, w, h, offset}, fallback[64] u32, palette[16][3] u8, crc32
```

**Payload:** `uint32 cell[]`, one per cell per region:

| bits | field |
|---|---|
| 0–3 | kind: 0 none, 1 mat, 2 kelp/algae, 3 ground cover, 4 shrub, 5 tree |
| 4–7 | palette index |
| 8–15 | cover 0–255 |
| 16–23 | height (m) |
| 24–31 | fauna density (later) |

Five regions take about 20 MB.

**`bioexport`** works per cell from live agents that are sessile phototrophs (`HOLDFAST > 64`, ANOX or OXY ≥ 32):
- Classify by realm and height: H < 0.5 m → mat (water or tidal) or ground cover (land). Water with H ≥ 1 → kelp. Land 0.5–4 m → shrub. ≥ 4 m → tree.
- The kind is the class with the most biomass.
- `cover = min(255, biomass_per_m²·255/ref)`, with ref = 50 g C/m².
- `height` is the biomass-weighted mean H.
- `palette` is the dominant hue bin (`G_HUE >> 4`). The RGB comes from an integer HSV table, so anoxygenic purple versus oxygenic green shows in the world.
- `fallback[env_bin]` is the dominant kind, cover, height and palette over all island cells in that env bin. It vegetates the world **outside** the simulated regions statistically. It can be disabled with `fallback_on = 0`.

### 7.3 `src/flora.h` (GME_HD, integer only, included by `worldgen.h` users)

```c
typedef struct GmeBioRegion { int32_t x0, z0, w, h; uint32_t offset; } GmeBioRegion;
typedef struct GmeBioView {             /* ~450 B, passed BY VALUE to k_generate */
    const uint32_t* cells;              /* host ptr on CPU, device ptr on GPU; NULL = no life */
    int32_t nreg, shift; uint32_t seed, fallback_on;
    GmeBioRegion reg[8]; uint32_t fallback[64];
} GmeBioView;
typedef struct GmeColumn { int32_t h; uint32_t cell; int32_t tx, tz, tbase, theight, trad; uint32_t colhash; } GmeColumn;
GME_HD uint32_t gme_env_bin(int32_t x, int32_t z, int32_t h, uint32_t seed, const GmePlanet* p);
GME_HD void     gme_column(int32_t x, int32_t z, int32_t h, uint32_t seed, const GmePlanet* p, const GmeBioView* b, GmeColumn* c);
GME_HD uint8_t  gme_voxel(int32_t x, int32_t y, int32_t z, const GmeColumn* c, uint32_t seed, const GmePlanet* p);
GME_HD int      gme_is_standing_flora(uint32_t m);
```

**`gme_env_bin`** is 2 bits × 3:
- moisture: `gme_noise2(x, z, 10, seed^0xB10C0001)` plus a bonus when `h − sea_level < 16`
- height band: `(h − sea_level)/40`, clamped 0..3
- substrate: rock province (the same `gme_noise2(...,9, seed^0x6A417E00)` granite test as `gme_material`) × soil

**`gme_column`** runs once per column, after `h`:
- Find the cell: region rectangle test (≤ 8 integer compares), otherwise the fallback, otherwise 0.
- `colhash = gme_hash3(x, 0, z, seed^0xF10A)`.
- For kinds 4 and 5: the tree slot is the 8 m grid square containing (x, z).
  - It is occupied if `gme_hash3(slot) & 255 < cover`.
  - `trad = clamp(height/4, 1, 3)`.
  - The trunk is jittered inside the slot with a margin of at least `trad`: `tx = sx·8 + trad + hash % (8 − 2·trad)`, likewise `tz`. Every crown voxel therefore lies in its own slot, and one slot suffices.
  - `tbase = gme_height(tx, tz)`: **one extra height call, only in vegetated slots**.

**`gme_voxel`** first computes `base = gme_material(x, y, z, c->h, seed, p)` unchanged, then:

| kind | rule |
|---|---|
| 1 mat | `y == h` and (seabed or beach) and `colhash & 255 < cover` → `MAT_MAT` |
| 3 cover | land, `y == h`, `base == surface_mat`, `colhash < cover` → `MAT_GRASS`; `y == h+1` and `(colhash >> 8 & 255) < cover/4` → `MAT_LEAF` |
| 2 kelp | `h < y < min(h + height, sea_level)` and `(colhash >> 16 & 255) < cover/2` → `MAT_KELP`; `y == h` → `MAT_MAT` as for kind 1 |
| 4/5 shrub/tree | land only (`tbase > sea_level`), only where `base == MAT_AIR`: `x == tx && z == tz && tbase < y ≤ tbase + theight − trad` → `MAT_WOOD`; crown around `(tx, tbase + theight − trad, tz)` with `dx² + dz² + (2dy)² ≤ trad²` → `MAT_LEAF` |

**Invariant:** with `cells == NULL && fallback_on == 0`, the output equals `gme_material` byte for byte.

### 7.4 Plumbing

- **`gpu.h/.cu`:**
  - Add `int gpu_world_set_biomap(GpuWorld*, const uint32_t* cells, size_t n)`, which does `cudaMalloc` plus a copy and keeps `d_bio`.
  - `gpu_submit(g, buf, n, seed, planet, const GmeBioView* view)` copies `*view`, sets `.cells = d_bio`, and passes it by value: `k_generate<<<…>>>(…, seed, *planet, bv)`.
  - In `k_generate`: `gme_column(wx, wz, h, seed, &planet, &bv, &col)` before the y loop, then `gme_voxel` in place of `gme_material`.
- **`world.h/.c`:**
  - `WorldConfig` gains `GmeBioView bio` (host view).
  - `world_create` uploads through `gpu_world_set_biomap`.
  - `cpu_gen_chunk(..., const GmeBioView* bio, out, meta)` uses the same `gme_column`/`gme_voxel`.
  - `val_fn` passes `&w->cfg.bio`, so the host, device and meta checks are unchanged.
- **Column check in `world_validate`:**
  - The top search becomes `m != AIR && m != ocean_mat && m != ocean_top_mat && !gme_is_standing_flora(m)`. GRASS and MAT sit at y = h, so the analytic `gme_height` still matches.
  - Add `ValidateReport.flora_checked` (columns where `gme_column` gave kind ≠ 0) and `flora_voxels` (flora-material voxels in the checked chunks).
  - Add `ref_hash`: FNV-64 over the reference chunks, printed by `--bench`.
  - Fail if `cfg.bio.cells` is set and `flora_checked == 0`, so the check cannot pass vacuously.
- **`main.c`:**
  - Options: `--biomap path` (default `life/home.biomap` if present) and `--no-life`.
  - Refuse a biomap whose planet_hash or seed does not match `hp`.
  - When a biomap is loaded, spawn at the centre of region 0, offset to the nearest coast land cell.
- **`render.h/.cu`:**
  - `RenderScene` gains `const uint32_t* bio_cells` (device pointer from `gpu.cu`), the regions and `float palette[16][3]`, carried into `KParams`.
  - `albedo()` adds `MAT_MAT (0.30, 0.22, 0.30)`, `LEAF (0.25, 0.45, 0.18)`, `WOOD (0.35, 0.25, 0.15)`, `KELP (0.20, 0.35, 0.18)`.
  - For GRASS, LEAF, MAT and KELP, look up the hit cell from `(p.cx·32 + h.vx, p.cz·32 + h.vz)`. If it is inside a region and its palette index is nonzero, tint with `mix3(base, palette, 0.7)`.
  - Extend the line-191 side-face dirt blend to `MAT_MAT`.
- **Cost.** One extra `gme_height` per vegetated slot column, plus about 10 integer compares per voxel. Measure with `gme --bench` (chunks/s with and without the biomap). The budget is at most 25% slower on the CPU path. More non-uniform chunks above ground are reported through the existing slots_used stat.

### 7.5 Creatures (M7)

*As built: the file is GMEFAUN1 (`src/fauna.h`, 10.9); the engine side is in render-upgrade-spec 5.5.*

- `bioexport --fauna` writes `life/home.fauna`: up to 64K motile agents `{x, y, z, size_m, palette}`. y is ground + 1 on land, or `sea_level − z` in water.
- `k_render` tests a coarse 32 m hash grid of spheres after the voxel hit. This does not touch the chunk pool, so validation is unaffected.
- Later: live biomap refresh via `world_set_biomap`, which flushes and regenerates resident chunks in the affected regions.

---

## 8. Metrics, CLI output and tests

### 8.1 Console output

Every 10 sim-days, `life` prints:
```
isl2 y=123.4 live=1.23M (motile 41k) C[mar 4.5e6 | land 1.2e3 mol] GPP 3.1e4 NPP 1.2e4 R 1.1e4 mol/d
  atm H2S 50.9% CH4 39.1% O2 0.0001%  O2aq max 0.8uM  sp 34 H 2.10 top 22%  troph max 3.2
  think 12% of agents  dW|mean| 1.7  rollbacks 0  128 st/s (duty .6)  gates G0 G1
```

### 8.2 `metrics.csv` (one row per sim-day)

- step, sim_day
- live, colonies, motile, births, deaths, migrants in
- C by guild × realm (marine/coastal/land × producer, consumer, decomposer, chemotroph)
- fluxes per pathway (8), GPP, NPP, respiration, burial, vent input
- `atm[7]`, mean aqueous O2/H2S/CH4
- C, N, S and Fe totals and their conservation residuals (must be 0)
- mean traits (pigment, cuticle, motor, think period, active neurons, plastic rows)
- mean |dW|
- max/mean trophic level, mean generation
- species richness, Shannon H, Simpson, top-cluster share, turnover (each epoch, from the sample)
- steps/s
- interventions

### 8.3 Species metric (`bio_stats.c`, each epoch on 4096 sampled genomes)

- Distance = `L1(enzyme + morphology genes)/255 + 0.25·L1(brain bytes)/127 + 0.5·[popcount(tag^tag') > 3]`.
- Single-linkage union-find at threshold 0.08 × gene count.
- Neutral loci 48–63 give the lineage age (molecular clock).

**Brain value** (`biostat --ablate`, offline on the PC from a checkpoint):
- Replay 2 sim-days of a 128² sub-window three ways: intact, brain-zeroed (outputs fixed at the bias), and η = 0.
- Report the energy-gain ratios. If brains do not matter, this catches it.

### 8.4 Images

Written to `maps/` each epoch through `write_png`:
- biomass per guild (64 m pixels)
- dominant kind
- pigment hue
- O2aq, H2S aq
- `env.png` once

### 8.5 Tests

Run with `gme_genesis life-test [--cpu|--device d]`; also wired into ctest `bio_tests` and `gme --bench`.

| # | test | pass criterion |
|---|---|---|
| T1 | lint | `bio_core.h` and `bio_kernels.cu` contain no `float` or `double` tokens |
| T2 | CPU≡GPU | 64² grid, 2k agents, 500 steps; FNV of fields, hot, cold, genomes and atm is equal on CPU and GPU. Rigs: `biostat --hash` gives equal results on 4090, 4080 and 3060 |
| T3 | conservation | Closed box (no vents, burial, weathering or atm exchange), 10k steps: total C, N, S and Fe are **exactly** constant |
| T4 | resume | 500 + checkpoint + 500 equals 1000 straight (hash) |
| T5 | chemostat | Methanogens at one vent with fixed H2 flux plateau within ±20% of the analytic K = flux·yield/maintenance |
| T6 | light profile | Phototroph biomass versus depth follows `EXPN` attenuation within ±25% |
| T7 | null controls | light = 0 → phototroph C = 0 by 2 sim-years. Vents off and no weathering → extinction. `G_MUT` forced to max (μ = 0 path) → species count does not rise. `atm_scale → 1` → no G5 in 100 years |
| T8 | selection | In a light-rich shelf, mean `G_PIGMENT` rises over 5k steps versus a μ = 0 control |
| T9 | learning | Fixed genome, moving food patch: intake per day with η > 0 beats η = 0 by > 10%; brain-zeroed is worse than both |
| T10 | predator–prey (M5) | Cross-correlation of prey and predator biomass peaks at a lag > 0 |
| T11 | engine | `gme_genesis biomap-test` writes a checkerboard of all kinds and palettes in a region around spawn. `gme --bench --biomap test.biomap` gives 0 host, device, meta and column mismatches and `flora_checked > 0`, on GPU and `--cpu`. `--no-life` gives a `ref_hash` equal to the pre-change baseline |
| T12 | benchmark | Steps/s at 1M, 2M and 4M per card, logged; the 3060 meets ≥ 150 steps/s alone at 1M |

---

## 9. Milestones (in implementation order)

| M | content | done-check | LOC | days |
|---|---|---|---|---|
| **M0** | `planetfile` `atm_mix` + `atm=` + `planet_hash`; `finish` emits `atm=` (re-run finish on the current planet); `rt_mem_info`; CMake `gme_embed()` (cosmos migrated to it); `src/bio/` skeleton and `bio_main` dispatch in `genesis_main.c`; `env --scout` and `env --island k` (`regions.txt`, `env.bin`, `env.png`); **`launch.ps1` kill filter changed to `^gme-search`**; `life/` folder convention | Scout prints sea_level −35 from the file and 6 regions with ≥ 20 vents each; the env CRC on rig B equals the PC's; cosmos still runs (one seed, unchanged output) | 550 | 2 |
| **M1** | Engine flora: materials, `flora.h`, `GmeBioView` plumbing (`gpu_world_set_biomap`, `gpu_submit`, `cpu_gen_chunk`, `world_validate` with `flora_checked`/`ref_hash`), `main.c` `--biomap`/spawn, render albedo and palette tint, `mats[]` names, `biomap-test` stage | **T11** green on GPU and `--cpu`; baseline `ref_hash` unchanged with `--no-life`; `--bench` slowdown ≤ 25% recorded; screenshot (P) shows the checkerboard | 650 | 3 |
| **M2** | Sim core with brains from day one: env kernels (diffusion, chem, pyrite, piston exchange, tide, light, T), atm box, sort/order/summaries, think (MLP + Hebbian + think period), motility and drift, fair-share demand/supply, all 8 pathways, growth, costs, kin-pathogen, death, scan births with genome pool and free stack, copy cost, mutation, compaction, stats, maps, checkpoint/resume with dedup, CPU reference `bio_cpu.c`, `life-test` T1–T9 and T12 | T1–T9 and T12 pass on the PC (GPU and CPU); 50 sim-years on the lab island reach G0 with zero ledger residual | 2300 | 6 |
| **M3** | `bio_island.sh`, `bio_launch.ps1`, `bio_status.ps1`, `bio_hub.ps1`, `--duty`, migration in/out, global atmosphere relax, `INPT` log, rollback on extinction; **boot islands 0–4 on all 5 rig GPUs** | All 5 `bio-isl*` sessions alive for 24 h with checkpoints rotating; hub moves migrants both ways; T2 cross-architecture hash equal; search seeds/hour drop < 25% versus the day before | 350 | 2 |
| **M4** | `bioexport` (parts → merge, fallback table, palette), `bio_status -Fetch` → `life/home.biomap` | `gme.exe` shows evolved mats and kelp on the home island's shelf; validation passes with the real biomap (`flora_checked > 0`); land cover appears once G3 is logged | 400 | 2 |
| **M5** | Predation (`k_claim`/`k_predate`, gut processing, trophic level), sexual crossover with mate tags, species clustering, `biostat --ablate`; resumed onto the running islands (checkpoint v2 is forward-compatible because all genes exist since M2) | T10 passes; species richness > 10 on at least 3 islands; brain-value ratio > 1.05 for motile guilds; G2/G4 tracked | 600 | 4 |
| **M6** | `life-seedbank`, `life-rebase`, `life-inoculate`; hub titration ETA | Rebase onto a test planet (a different seed) seeds all habitats; resume with a wrong planet hash is refused; inoculation is logged in `stages.log` | 250 | 2 |
| **M7** | `home.fauna` export and sphere overlay in `k_render`; optional live biomap refresh | Motile organisms visible near spawn; validation unaffected | 300 | 3 |

**Totals:** about 5,400 LOC and about 24 working days. Long boots start at M3 (about day 13) and keep accumulating. Every later milestone resumes from the live checkpoints.

### Risks and how this plan measures them

- **Emergent steps stall.**
  - O2 never rises: the hub prints a titration ETA, and `atm_scale` and `k_py` are logged knobs.
  - Land never colonised: per-island barrier sweeps, the tidal bridge, and logged inoculation.
- **Determinism slips.**
  - A float sneaks in: T1 lint.
  - Order-dependent code: T2 runs on every build; ordered cell lists and keyed `atomicMin`.
- **Contention with the search.** `--duty`, plus seeds/hour monitored in `bio_status`.
- **Integer overflow.** Fields are int64; agents are capped at 1e9 mmol C and E; demand is u64; reductions are int64.
- **Material enum growth.** Grep every `MAT_COUNT`-sized table in M1 (`genesis_main.c` `mats[]`, `render.cu` `albedo`).
- **Losing outputs.** `status.ps1 -Fetch` wipes `genesis/`, so every biosphere artefact lives in `life/` and `runs/`.

---

## 10. After the first release: animals, sex, learning, light in bands, shapes, the hub, seed banks

Sections 1–9 are the plan the first release was built from. This section describes what was built on top of it, as built. Where it differs from the plan (3.4 learning, 3.8 predation, 6.4 hub, 6.6 seed bank, 7.5 creatures), this section is the reference.

### 10.0 Features and the rule that protects running worlds

Each addition is a **feature**, a bit in `BioParams.feat` (or `BioParams.light_mode` for light in bands). `BioParams` is stored in every checkpoint.

| Feature (`life --features`, `--enable`) | Bit | What it switches on |
|---|---|---|
| `animals` | `BIO_FEAT_PRED` | predation and grazing, gut, trophic level, predator and prey senses, gate G4 (10.1) |
| `sex` | `BIO_FEAT_SEX` | crossover with a compatible cellmate (10.2) |
| `learn` | `BIO_FEAT_LEARN` | learning by exploration, replacing the plain Hebbian rule (10.4) |
| `morph` | `BIO_FEAT_MORPH` | crown shapes: genes 37–41 (10.6) |
| `hub` | `BIO_FEAT_HUB` | gas uptake ledgers by realm, the island's outbox and inbox (10.7) |
| `spectral` | `light_mode` | light in 9 wavelength bands from a GMESPEC1 file (10.5) |

- **A new run** gets all of them by default, `spectral` only when `life/spectrum.bin` exists. Named explicitly (`--features all` or a list with `spectral`), light in bands requires the file and the run refuses to start without it.
- **A resumed run keeps the features of its checkpoint.** A checkpoint of the first release carries none, so a new build alone changes nothing on a running island.
- **`life --resume --enable LIST`** switches features on in a living world. It is an intervention: it is logged in `stages.log` (`INTERVENTION enable ...`) once, and passing it again later is a no-op.
- **With every feature off the simulation is the first release, bit for bit.** Test T14 reruns the T2, T2b and T4 scenarios with `features = 0` and compares their state hashes with the ones the first release printed. On real data: the 2.85 M-agent checkpoint of island 1 (sim year 2751) resumed for two sim-days gives state hash `320b1cf0b31758e2` with the first release's binary and with the new one.
- State that only the features use (the ecology word and the exploration of each agent, the second accumulator block) is part of the state hash only while a feature is on.

**What switching a feature on does to an evolved world.** Island 1 (2.85 M agents, 89% O2), resumed from its year-2751 checkpoint for one sim-year with one feature switched on at a time, without a hub:

| Switched on | Plant production (mol C / day) | Land carbon (mol) | Cells of ground cover / shrub / tree | |
|---|---|---|---|---|
| nothing | 4.14e5 | 1.42e8 | 55.1% / 23.0% / 0.10% | |
| `animals` | 4.24e5 | 1.42e8 | 55.1% / 23.1% / 0.10% | 149 agents live mostly on prey; 20 kills and 1,170 bites a day move 4,500 mol C (10.1) |
| `sex,learn` | 4.17e5 | 1.42e8 | 55.1% / 23.1% / 0.10% | 143,500 sexual agents, 3.6 births a day with two parents |
| `spectral` | 4.32e5 | 1.42e8 | 55.1% / 23.0% / 0.10% | |
| `morph` | 5.36e5 | 1.38e8 | 66.8% / 11.2% / 0.15% | the vegetation re-sorts itself (10.6) |
| all | 5.48e5 | 1.38e8 | 67.2% / 10.9% / 0.14% | 119 agents live mostly on prey |

Animals, sex, learning and light in bands leave an evolved vegetation where it was. Crown shapes do not. In every run the population (2.85 M), the atmosphere and the vegetated area stayed put and the ledgers stayed exact.

Parameters of the features live in what used to be `BioParams.reserved[16]`. They are zero in an old checkpoint and are filled in with their defaults whenever a run is created or loaded (`bio_params_features`, `bio_features_init`).

**Checkpoint version 3** (magic still `GMEBIO02`): a new `AUX_` section (12 B per agent: ecology word, exploration) and a longer `GLOB` section. Version 2 files load: the missing parts are zero.

### 10.1 Animals (replaces 3.8)

The plan's per-cell ordered lists would have cost about 0.6 ms per step at 1 M agents (a tenth of the step). Predation is built on per-cell **slots** instead: no sort, every write is a commutative atomic or touches only the writer's own agent, and the CPU reference and every GPU agree to the bit.

**Per step** (`k_summary` → `k_pre` → `k_metab` → `k_feed`):

1. **Offer** (`k_summary`). An agent attacks when `MOUTH > 0`, `MOTOR > 0`, its attack output `out[2] > 0` and its gut is less than half full (`2·gut ≤ C`; a full animal does not hunt). It offers itself in one of the cell's 4 slots (chosen by hash). A slot keeps the highest key: a 7-bit hash priority, then the agent index scrambled per cell and step, so neither array order nor uid order is favoured.
2. **Claim** (`k_pre`, before anyone moves). Every agent looks at one slot of its cell (by hash). If an attacker stands there, the agent offers itself as prey when
   - it can be **found**: movers (a real motor, `MOTOR ≥ 16`, and no photosynthesis) always can; of the cell's `nn` others each is found with probability `1 − 1/nn`, and the last one never. This is the **refuge**: a cell hides one of its anchored or drifting inhabitants, so predation thins a cell but does not empty it (`pred_refuge1`, tuned);
   - and it is **edible**: an anchored colony (`HOLDFAST ≥ 128`) can always be grazed; anything else only if one of its individuals fits the attacker's gape: `m_r(prey) ≤ 2^(SIZE_att/10) · MOUTH_att / 32`. Edibility compares individual sizes (the reproduction mass, a gene), not colony biomass: a colony cannot escape by growing, only its lineage can by evolving larger individuals;
   - and the attacker can **reach** it: an attacker slower than a metre an hour (`MOTOR < 16`) cannot go to its food. It takes only what comes to it: prey that walks or swims at that speed, or that floats free in water. Grazing what stands still is for those that move. (Evolved forests carry mouths and motor genes of 1 or 2 that drifted in while the genes did nothing; without this rule the plants eat each other where they stand.)
   The attacker takes the claim with the lowest key (`atomicMin`): one prey per attacker and step.
3. **Outcome** (`k_metab`, the prey's thread). Success probability `SIGMOID[(MOTOR_att − MOTOR_prey)/4] · (300 − ARMOR)/300` (the plan's `/8` was too flat to select for speed; tuned). With `b0 = C_att · MOUTH/255 / 4`:
   - a free body of at most `4·b0` is **eaten whole**: 70% of `C + S + gut` goes to the attacker, 30% to the cell's DOC, its N and P to NH4 and PO4;
   - otherwise it loses a **bite**: `min(C/2, b0)` of its reserve carbon. A colony keeps at least half of its reserve per bite.
4. **Feed** (`k_feed`, the attacker's thread). The attacker adds what its prey gave up to its `gut`. An attacker that died in the same step passes the meal on to the detritus.

**Gut.** Gut carbon is the animal's private organic carbon. Its FERM, SULFRED and AERO pathways and its assimilation use it before the cell's DOC (the demand on the pool shrinks by what the gut covers), with half saturation at `gut = C/8`. An aerobic animal gets 5 eu per mmol, a fermenting one 0.5, as in 3.5.

**Space.** With animals in the world, the crowding maintenance and the 64-per-cell birth cap of a mover (`MOTOR ≥ 16` and no photosynthesis) count only the movers. A grazer in a meadow of 64 plants would otherwise pay five times its maintenance and never reproduce there. A phototroph needs its place in the light whatever its motor gene: an unused motor gene costs nothing and must not buy a plant room.

**Diet record** (one word per agent, inherited): the share of its organic carbon that came from its gut (moving average), its trophic level (moving average of prey level + 1; a meal of its own kin bucket does not count, or cannibals would climb without end), the share of its meals that were producers on dry land, and how many generations in a row were born to a land animal. A **consumer** is an agent with gut share above one half. A **creature** is what the engine is given to draw as an animal (10.9).

**Senses** (brain inputs 11–14, with `G_SENSE ≥ 128`): the gradient in x and z of the largest gape that fits an individual of the agent's size (danger), and of the number of agents (the crowd, i.e. food).

**Gate G4** (land animals): at least 1000 creatures (10.9) for one sim-year that are fast (`MOTOR > 128`), on dry land, consumers, with more than half of their meals land producers, and at least 5 generations in a row born to such a parent.

**What it does in an evolved world.** Island 1 at sim year 2751 (2.85 M agents, 89% O2), one sim-year after `animals` was switched on: 149 agents live mostly on prey at trophic level 2.6, 106 that go to it and 43 anchored ones that take what drifts past, with about 20 kills and 1,170 bites per day moving 4,500 mol C per day (1% of the plant production). The island has 102 creatures. Most of them still photosynthesise and carry a holdfast gene: plants by descent that have begun to walk and graze. No land animal in the sense of gate G4 yet. The population, the carbon by realm, the atmosphere and the vegetation stay where they were, and the ledgers stay exact. The raw material was there already, drifted in while the genes did nothing: about 160,000 anchored plants carry a motor gene of 1 or 2, and many a mouth. Without the reach rule they ate their anchored neighbours where they stood (10,400 consumers within a year, 10,363 of them anchored plants, 99 creatures); with it, grazing has to be earned by moving, and animals start small.

### 10.2 Sexual reproduction (as 3.6, with slots)

An agent with `G_SEX ≥ 128` offers itself as a mate in one of its cell's 2 mate slots (same key as above). A reproducing parent with `G_SEX ≥ 128` takes the first slot holder that is alive, is not itself and whose `mate_tag` differs from its own in at most 3 bits. The child gets each of the genome's 24 blocks of 16 bytes from one parent or the other by hash bits (a block is 16 body genes, one hidden neuron's input weights, two neurons' recurrent weights, ...), then mutates as usual. Without a compatible mate in the cell the parent reproduces alone: sex is facultative, so switching it on cannot strand a lineage. Species still come only from geography and tag drift.

### 10.3 Counting species (8.3, as built: `bio_stats.c`)

Each epoch, and on `biostat --species`: 4096 genomes sampled by a hash of their uid (the same agents on every device), distance = L1 of body genes 0–41 / 255 + 0.25 × L1 of the expressed brain bytes / 127 + 0.5 when the mate tags differ in more than 3 bits, single linkage at 0.08 × 42. Only the weights of hidden neurons that are switched on in both genomes count: the weights of a neuron that is off drift freely and say nothing about kinship. Output: clusters, clusters of two or more, Shannon H, Simpson, share of the largest (`census.csv`, console). Island 1 at year 2751: 169 clusters in the sample, 85 of them with two or more members, the largest 88%.

### 10.4 Learning by exploration (replaces the rule of 3.4)

**Why the plain rule failed T9.** `dW += η·m·out·hid` strengthens whatever the agent did whenever the reward beat its running average. It cannot tell a good action from a bad one, and over time its updates cancel. On the foraging test it was 4% worse than no learning.

**The rule** (node perturbation). At every think, each plastic output of a learner (`ETA > 0`) is nudged by a random amount of up to ±`explore_q7`/127 (default 40: ±0.31) and keeps that nudge until its next think. At the next think the weights of the row follow `ETA · m · nudge · hid >> 21`, minus the decay, clamped to ±32. A nudge that paid is made more likely in that situation, one that cost less likely. Offspring still start with `dW = 0` (Baldwinian).

**T9 as it now reads** (motile fermenters chasing a food patch that jumps every 2 days; intake per day over days 10–40; `ETA` 64):

| Run | Intake | |
|---|---|---|
| learning | 3470 | |
| no learning (`ETA` forced to 0) | 2757 | learning is +26% |
| exploring exactly as a learner does, weights frozen | 3184 | learning is +9% |
| brain zeroed | 1724 | worse than all |

The third row is a control the test did not have before. About 60% of the gain over "no learning" is the exploration itself (a wanderer finds food patches); the rest is learning proper. The test requires learning to beat no learning by more than 10% **and** the frozen-weights control by more than 5%. Over eight seeds learning beat no learning by 23–44% and the control by 7–15%. With random instead of pre-wired weights, learning more than doubles the intake of the control (1962–2125 against 930): chemotaxis learned from scratch within a lifetime. Too high a learning rate hurts (at eight times the best step learning falls behind the control), so `ETA` has an optimum for selection to find.

### 10.5 Light in wavelength bands (render-upgrade-spec 5.2, as built)

- **File.** `gme --export-spectrum life/spectrum.bin` writes GMESPEC1 from the renderer's atmosphere and water model: band irradiance at the surface for 8 haze bins × 24 hours × 9 bands (8 of 45 nm over 380–740 nm, plus 740–900 nm), water transmittance per band at 0–32 m, photons per joule, the share of a band's photons each reaction centre can use, and the pigment shape per hue bin. Integers only; the rigs never call libm.
- **Per step.** The surface light of each band is the table value at the hour, interpolated between the two haze bins around the island's present haze. The 45% PAR share of the scalar model is gone (the real share on the home planet is 33.6%), and the methane the island makes now colours its light.
- **Per agent.** Each band is attenuated by the water above the agent (the band's own transmittance table, the band's attenuation below 32 m) or by the grey canopy above it. The pigment absorbs `1 − 10^(−OD)` of a band, `OD = PIG/255 × 4 × ε(hue bin, band)`: a Soret band at 435 nm, a Q band at 383 + 34 × bin nm, and a little background. Absorbed light is counted in photons, and a photon counts only if the pathway's reaction centre can use it: oxygenic photosynthesis gets nothing beyond about 720 nm, anoxygenic uses the near infrared as well.
- **Energy scale.** The two efficiencies (eu per µmol of usable photons per second) are set when the run is created so that a reference leaf (pigment 200, Q band at 656 nm, at the surface, summed over a day at the present haze) yields exactly what the scalar model gave it. The world keeps its energy scale; what changes is which light a pigment can catch. The values belong to the run, because they are fitted at the island's haze of that moment: on the home planet a new run gets 1.030 (oxygenic) and 0.612 (anoxygenic), island 1 switched over under its clear air of year 2751 gets 1.227 and 0.653.
- **Leaf area.** With bands the leaf area no longer scales with the pigment gene; the pigment acts through its absorptance, which saturates. Canopy shading stays grey (the leaf area index of 2.5.1), and a plant shades its neighbours with the same whole leaf it harvests with.
- **Resume.** The CRC of the file's payload is stored in the checkpoint; a resume with another light file is refused. Settle the file (it comes from the engine's atmosphere model) before switching light in bands on in a running island: after that the island is tied to it.

**T13:** the reference leaf gains 10176 eu per day with bands and 10192 with the scalar model (−0.2%); the same leaf with its Q band in the near infrared gains 14% less; the gain of benthic phototrophs at 2–17 m follows the band sum within 1.7%; and of two clones that differ only in pigment colour (Q band at 656 nm and at 485 nm), started in equal numbers on a shelf that is 2 m deep on one side and 12 m on the other, the red one wins in the shallows and the blue-green one at depth (mean Q band 589 nm against 565 nm after 5000 steps, against 570 / 573 nm with scalar light, where the clones are the same organism).

### 10.6 Plant shapes (render-upgrade-spec 5.3, as built)

Five body genes that were neutral until now:

| Gene | Meaning | Gene value 0 |
|---|---|---|
| 37 `G_CROWN` | crown class: ellipsoid, cone, umbrella, column, dome (`g·5 >> 8`) | ellipsoid |
| 38 `G_ASPECT` | height : width, 8 steps: 0.5, 0.75, 1, 1.4, 1.8, 2.2, 2.6, 3 | 1 : 1 |
| 39 `G_LEAFANG` | leaf angle, 16 classes from erect to flat | the mid angle |
| 40 `G_POROSITY` | open crown: lets up to half the light through | closed |
| 41 `G_QWIDTH` | width of the pigment's Q band (with light in bands) | narrow |

**Gene value 0 is the shape the plant had.** These genes sat unused through every evolved world's history, and with the low mutation rates that evolved they are still near 0 (island 1, year 2751: means 11–24 of 255). The aspect and leaf angle genes therefore wrap around their range so that 0 is height : width 1 and the mid leaf angle, and the crowns of an evolved forest keep their form when shapes are switched on (how much light each plant then gets is another matter, see below), with a few percent of odd crowns as raw material (island 1: 91% ellipsoid, 1.3% cone, 2.5% umbrella, 3.8% column, 1.5% dome).

**What a shape does** (with shapes on; a land plant of 0.5 m and more has a crown):
- **Height.** The same structure builds a taller, narrower or a lower, wider plant: `H = cbrt(S/4000 × aspect²)`. Tall plants overtop their neighbours (the canopy bands of 2.5.1).
- **Crown.** The crown's top area `A` is a disc of the crown's width (`H/aspect`) times a class factor (ellipsoid 1, cone 0.6, umbrella 1.6, column 0.4, dome 1.2). The world's light is horizontal flux per unit of leaf area, as if every leaf lay flat; that stays the unit. With leaf projection `κ` and crown leaf area index `Lc` = leaf area / `A`, the crown intercepts the light falling on `A × (1 − e^(−κ·Lc))`: `κ ×` its leaf area while it is thin, never more than falls on its top however much leaf it holds. A tall narrow crown is high but small.
- **Leaf angle.** `κ = G(θ)/cos θ × clumping × gain`, Goudriaan's `G = φ1 + φ2·cos θ` at the hour's sun elevation as an integer table of 16 classes × 24 hours; an open crown (porosity) lowers it by up to half. The gain is a constant of the planet that gives the reference foliage (mid angle, ellipsoid) a light-weighted mean `κ` of 1 over the day. Flat leaves catch more per leaf when the sun is high; erect ones spread the light over more leaf.
- **Saturation.** With shapes on, the lit leaf of every land plant saturates: `harvest = cap × (1 − e^(−L/cap))`, `cap` = 720 eu per step per m² of lit leaf (200 W/m², about 25 µmol C/m²/s). The lit leaf is the whole leaf of ground cover and of a thin crown, `A/κ` of a full one. Alone in full sun a flat leaf keeps 45% of what the linear rule gave it over a day; where the light is shared or dimmed, little is lost. Erect leaves spread the noon light over more leaf and win in a full crown.
- **Shade.** A plant shades its neighbours with what it intercepts (a small full crown shades little ground, however much leaf it holds), and under a band of crowns exactly what they intercepted is missing. By night the bands hold the plain leaf area again, which is what the count of vegetated cells and gate G3 are taken from at the end of each day.
- **Cost.** Structure maintenance times a class factor (cone 0.85, umbrella 1.35, column 0.75, dome 1.1); a wide Q band up to doubles the pigment cost.
- **Calibration.** In light too weak to saturate a leaf, a thin crown with every shape gene at 0 gains over a day what its leaf area gained before the feature, and so does ground cover.

**What it does to an evolved world.** It is not neutral, and cannot be: until now a plant was a quantity of leaf sharing its cell's light, and plants evolved to carry far more leaf than a crown of their height can light. Island 1, one sim-year after `morph` was switched on: cells dominated by shrubs fall from 23.0% to 11.2%, ground cover rises from 55.1% to 66.8%, trees from 0.10% to 0.15%. Plant production rises by 29% (ground cover gets the light the over-leafed crowns no longer take), land carbon falls by 3%; the population, the atmosphere and the vegetated area do not move. With every feature and the hub the same island stands at 6.7% shrub cells and 0.07% trees after three sim-years: the over-leafed shrubs die back faster than new ones grow. From then on shapes are under selection (wider crowns, more structure per leaf), and whether a forest of the new kind grows back is for the islands to show. Switch `morph` on in a new or rebased world; in a running one expect this re-sorting, try it on one island first and keep a checkpoint.

**T16** checks it through the simulation, one plant and one day: in weak light a thin crown and ground cover gain what they gained without the feature (−5.8% and −2.8%); in full sun both keep about 45% (crown 46%, cover 44%: no step at the 0.5 m where a crown begins); a full crown gains under half as much per unit leaf carbon as a thin one, and no more than its top area would gain as bare leaf; flat leaves beat erect ones in a thin crown and lose in a full one; a squat crown beats a tall narrow one when nothing shades it; an open crown catches less than a closed one.

**Export.** A run with any feature writes **biomap version 2** (`GMEBMAP2`): per cell one more word with the dominant crown class, aspect, leaf angle and porosity bins, the mean pigment and the Q width of the cell's plants of the exported kind; the same per fallback class; and the island's atmosphere in ppm, so the rendered sky can follow the air life has made. A first-release run keeps writing version 1, byte for byte. `biomap_load` reads both.

### 10.7 The island hub (2.5.7 and 6.4, as built)

**Island side** (`bio_hub.c`, with the `hub` feature; once per sim-year, before anything of that step is reported or saved):
- writes `outbox/atm_<k>_<epoch>.txt` (its box, and its cumulative gas uptake from the atmosphere by all cells and by land cells) and `outbox/mig_<k>_<epoch>.bin`: its emigrants, up to 64 genomes of each guild in each habitat (marine, coastal, land). They are the distinct genomes with the most living biomass, one per species-like cluster first (no genome within the species distance of 10.3 of one already chosen), then the next heaviest. The plan chose by founder lineage; an old world descends from a handful of founders (island 1 at year 2751: three), so that sent nine genomes a year;
- imports what waits in `inbox/`: migrants into random cells of their habitat (the import ledger keeps the totals exact), and the planet's atmosphere: the box moves a quarter of the way to it (the hub-edit ledger keeps the totals exact);
- moves what it applied to `imported/e<epoch>_<name>` and notes the last epoch at which it read its inbox (`imported/last_live.txt`). **An island resumed from an older checkpoint applies the same files at the same epochs again**, and nothing at the epochs where nothing was applied, so its history repeats exactly (T17: identical state hash); what arrived in the meantime waits in `inbox/` until the island is past its old history. This folder is the plan's `INPT` log; it keeps the records of the last 1024 epochs (a checkpoint older than that replays without them).
- Of a backlog in `inbox/` (the island was away) only the newest planet atmosphere and each sender's two newest emigrant files are taken; the rest is dropped.
- An island never waits. Without a hub it runs closed.

**Hub side** (`gme_genesis life-hub --dir life/hub --islands 0,1,...`; integer arithmetic, plain files):
- The planet's atmosphere is one column per m², kept at the islands' box scale (`atm_scale`). It starts as the mean of the islands' columns.
- Each pass takes the newest figures of each island and turns the change since its last report into uptake per m² of ocean and per m² of land, weighted by the planet's ocean cover (10% on the home planet; the islands are coasts with far more sea). The planet is the mean of the islands taking part: every island-year counts for 1/n of a planet-year whenever it is reported, so the column and the clock do not depend on how the reports fall into passes. The column moves by that uptake, then by the same photochemistry as an island box.
- The islands run at different speeds. The hub's clock is the mean of the years they have reported.
- **One hub folder serves one planet.** The island files carry their planet; a file of another planet stops the pass, and an island refuses the air of another planet. An island whose epoch has gone backwards was founded anew (`life-rebase`): it is taken in again from where it now stands.
- Each island's emigrants go to the next island of the ring and to one more, chosen by hash.
- It prints the planet's air and the **titration**: sim-years until the air's reductants are used up at the present net oxidant supply, or that there is no O2 rise at the present fluxes.
- `--runs DIR,DIR,...` handles islands whose run folders are on the same machine (the hub carries the files itself). For rigs, `cluster/bio_hub.ps1` makes `life/hub/in` hold what each island's outbox holds before each pass, and copies `life/hub/out/<k>` into the inboxes after it, under another name first so an island never reads half a file.

**T17:** two islands, six epochs: migrants both ways, exact ledgers, the islands' O2 shares pulled together (gap 0.0000 against 0.0298 without the hub), and the replay from an older checkpoint identical. **T17b**, on hand-written island files: the same island-years reported in one pass or in two give the same column and the same clock; an island founded anew is taken in from its new epoch; a file of another planet is refused.

### 10.8 Seed bank, rebase, inoculation (6.6, as built: `bio_bank.c`)

**Guilds here are read from the genes, not from what an agent has eaten:** a `producer` can photosynthesise (pigment, and ANOX or OXY of 32 and more); a `chemotroph` leans to lithotrophy (METH, FEOX or METHOX at least as strong as FERM, SULFRED and AERO); of the other heterotrophs those with a motor are `consumer`, the rest `decomposer`. `land_animal` is `consumer@land`. These are not the consumer (gut share) and the land animal (gate G4) of 10.1: a motile heterotroph that never ate prey is a `consumer` here.

- **`life-seedbank --runs DIR [--out life/seedbank.bin] [--n 2000]`** reads the newest checkpoint of every `island_*` run under DIR (each on its own authority: `env.bin` says which planet it is, no planet file needed) and writes GMESEED1: one record per distinct genome and habitat with its guild, habitat (marine, coastal, land) and living biomass, the best `n` of each guild × habitat. Island 1 at year 2751: 790,392 distinct genome × habitat records, 6 s.
- **`life-rebase --bank FILE --island K [--n 20000]`** creates a run on the present planet founded from the bank instead of protocells: each founder goes into a cell of its genome's habitat with its reproduction mass, the energy to reproduce once and the nutrients for its body. With fewer founders than banked genomes the founders are spread evenly over the bank, so every guild and habitat gets its part. It writes the first checkpoint and stops; `life --island K --resume` takes it from there. `life --bank FILE` does the same and keeps running. Genomes carry no planet state. A resume with a wrong planet hash is refused as before (T4).
- **`life-inoculate --from RUN --to RUN --guild G [--n 500]`** copies evolved genomes of one guild (`producer`, `chemotroph`, `decomposer`, `consumer`, `any`, optionally `@marine`, `@coastal`, `@land`; `land_plant`, `land_animal`) from one run's newest checkpoint into the other's, into cells of their habitat. It keeps the replaced checkpoint in `before_inoculate/` and logs `INTERVENTION inoculate` in `stages.log`. It never writes behaviour. Stop the target island first.

**T18:** bank of an evolved test island, file round trip, rebase onto another planet (3000 founders, each in a cell of its habitat, exact ledger), inoculation of land producers only.

### 10.9 Creatures for the engine (7.5, as built)

`bioexport --run DIR` writes `fauna_<K>.bin` beside the biomap part, and `bioexport --merge` writes `home.fauna` beside the biomap (GMEFAUN1, `src/fauna.h`): the creatures, the largest bodies first, at most 65,536.

A **creature** is an agent with a motor that moves at 1 m per hour or more at full effort (`MOTOR ≥ 16`) or lives mostly on prey. A phototroph with a holdfast is a plant whatever its motor gene (evolved forests carry motor genes of 1 or 2 that drifted in) and is drawn as vegetation, unless it does both, walks and lives on prey: then it is a creature and is left out of the vegetation. The same rule counts `creatures` in the statistics and decides who can be a land animal (gate G4).

Per record: world position in 1/16 m, depth below the water plane for swimmers, body size (the diameter of a sphere of the body's volume at 0.12 cm³ per mmol C), palette index (pigment hue bin), trophic level, motor gene, and flags: swims, hunter (trophic level 2.5 and up), armoured, swarm (a body of 64 or more individuals). Positions carry no height: the engine puts a walker on its own ground. `bioexport` also prints who carries a motor gene. Island 1 one sim-year after `animals` was switched on: 160,301 agents carry one; 102 are creatures (the largest 0.94 m), 157,660 are plants with a holdfast, and 2,539 are free but slower than 1 m per hour and do not live on prey.

### 10.10 Tests added

| # | Test | Pass criterion |
|---|---|---|
| T2c | CPU ≡ GPU, every feature | plants with crowns, grazers, hunters, sex; 1500 steps; equal hashes; kills, bites and two-parent births all above 0; exact ledger |
| T3b | conservation with animals | closed box, 4000 steps, C N S Fe exactly constant with kills and bites |
| T4b | resume with animals and sex | grazers, hunters and sexual plants: 517 steps, a checkpoint, 150 more steps on the running island and on one loaded from the checkpoint give the same state hash (the save is off the 64-step compaction grid, so the running island has holes in its arrays that the loaded one has not) |
| T9 | learning | as 10.4 |
| T10 | predator and prey | plankton and hunters, fixed genomes, 200 days: both persist, the cross-correlation of the two biomass series peaks at a positive lag with r > 0.3 (measured: lag +6 days, r 0.99) |
| T13 | light in bands | as 10.5; CPU ≡ GPU with bands and every feature; resume rules |
| T14 | features off = first release | recorded hashes reproduced |
| T15 | sex | two clones with markers in two genome blocks, no mutation: recombinants with compatible tags (7107 of 18,189), none with tags 16 bits apart, none with the feature off |
| T16 | crown shapes | as 10.6 |
| T17, T17b | hub | as 10.7 |
| T18 | seed bank | as 10.8 |
| T19 | species, creatures, vegetation with shapes | cluster counts of known populations; creature file sorted, inside the island, round trip, no anchored plant in it; a shape word on every vegetated cell |

T6 of the first release (light profile under water) now takes its reference at 2 m: the 1 m cell falls dry at low tide, and a leaf in the air is not under water.

`GME_BIOFEAT=<bits>` replaces the default features of the tests that do not choose their own (for experiments; with 0 the tests of the features report failures by construction). That a build with every feature off is the first release is what T14 checks.

### 10.11 Cost

Per step on the RTX 3060 at 1 M agents (T12, protocells; three runs each; the gate is 150 steps/s): the first release 176 to 180 steps/s, this build 183 to 203 with the features off and 187 to 194 with all of them on. Protocells exercise the features little; the evolved island is the fairer measure.
On the evolved island 1 (2.85 M agents, 3060): 80 steps/s with the features off, 62 to 76 with one or all of them on (all: 62).

### 10.12 Known limits

- **Kin sharing, spore timing, wind-directed spores and `G_STORE`** are still genes without behaviour.
- **Canopy shading is grey** with light in bands (the plan's phase 2, a leaf area index per band, is not built).
- **Predation is local to a 16 m cell** and at most 4 attackers feed per cell and hour. Dense herds interfere with each other; that is the model's only predator self-limitation besides hunger.
- **The hub's global clock is a mean** of island epochs; islands far apart in sim time exchange fluxes as if they were contemporaries.
- **Switching features on changes a world's future** and cannot be undone except from an older checkpoint. Keep one (`milestones/`).
- **Animals start from very little.** In a world of the first release the only raw material is what drifted in while the genes were silent. One sim-year after switching on, island 1 has about 100 creatures and no land animal; whether and how fast grazers spread is for the islands to show (gate G4).
- **Switching crown shapes on re-sorts an evolved vegetation** within a sim-year (10.6).
- **Leaf saturation uses the mean light of the leaves that share it**, not the sunlit and shaded leaves separately: a closed stand loses little to saturation.

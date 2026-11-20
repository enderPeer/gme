# GME: a world booted from physics

GME is an experimental game engine in **C17 + CUDA**. It builds its world from first principles and then lets life evolve in it:

1. **A star system forms.** A star and a dust disk; planetesimals collide and merge into planets through GPU N-body gravity. Rock, ice and gas are sorted by the disk's temperature.
2. **Chemistry decides the materials.** The chosen planet's crust becomes atoms with real valences (H1, O2, N3, C4, Si4, …). They form, break and swap bonds by real bond energies while the planet cools. What survives becomes the building blocks: granite or basalt, iron ore, sulfides, water, the atmosphere's gases, and organics. Carbon is the atom with the most varied bonds.
3. **Life evolves.** Self-copying organisms carry an energy budget, a genome of enzymes and body genes, and a small neural network. They evolve on islands of the real terrain, starting at hydrothermal vents. Phototrophs, oxygenic photosynthesis and land plants emerge without scripting; grazers and hunters, sexual reproduction, learning within a lifetime, pigments tuned to the light they get and crown shapes follow from the same genes. The simulation is integer-only, so a run is bit-identical on every GPU and on the CPU.
4. **The world is streamed and rendered on the GPU.** A sparse voxel world is sized to the machine, and a CUDA ray tracer shows it. Evolved vegetation becomes part of the voxel world.

The heavy stages run headless on a small GPU cluster over SSH. The rigs need only the NVIDIA driver: GPU code is compiled at run time with NVRTC, so no CUDA toolkit is required there.

| Planet formation (12,288 planetesimals, 6,000 years) | Life on a test island (10 simulated years) |
|---|---|
| ![formation](docs/images/planet59_formation.png) | ![life](docs/images/alife_test_island_10y.png) |

| Smooth density terrain (render upgrade M1, before / after) | Vegetation kinds and pigment colours (test pattern) |
|---|---|
| ![render](docs/images/m1_before_after.png) | ![vegetation](docs/images/vegetation_test_pattern.png) |

| Evolved vegetation after ~3,000 simulated years (new renderer) | Sunset through the methane haze |
|---|---|
| ![evolved](docs/images/evolved_world.jpg) | ![sunset](docs/images/sunset.jpg) |

**Status:** research prototype, Windows-first (headless Linux build for render nodes). Star system formation, chemistry and the integer life simulation work end to end. On the current planet, life has run for 3,000+ simulated years on two cluster GPUs: oxygenic photosynthesis took over, an oxidation event followed, and 70–78% of the islands are vegetated. The renderer upgrade (smooth density surfaces, 8-band spectral light, far field to 32 km, bounced light, pigment colours; [docs/render-upgrade-spec.md](docs/render-upgrade-spec.md)) is built, reviewed and fixed; `--quality classic|low|medium|high|ref` picks the renderer (classic = the original voxel ray tracer). Animals (predation and grazing), sexual reproduction, learning, light in wavelength bands, crown shapes, the island hub and seed banks are built as features of the life simulation, and the engine draws evolved crown shapes and creatures. On the evolved islands animals start from very little (see [docs/biosphere-spec.md](docs/biosphere-spec.md) 10.1); a walkable player is not built yet.

## Requirements

- **Engine (`gme`):** Windows 10/11, an NVIDIA GPU (developed on an RTX 3060, sm_86), CUDA Toolkit 12+ (developed with 13.4), Visual Studio 2022 Build Tools (MSVC), CMake 3.24+ and Ninja.
- **Simulation tools (`gme_genesis`, `gme_biotest`):** plain C. They build with MSVC or gcc (Linux) and need only the NVIDIA driver plus an NVRTC library at run time.

## Build & run (Windows)

From PowerShell (the `.\` prefix is required there):

```
.\build.bat                             # all targets into .\build (or: .\build.bat <dir> <target>)
.\build\gme.exe                         # opens the 3D view (loads genesis/home.planet and life/home.biomap if present)
.\build\gme.exe --bench                 # headless: fill + 20 s flight + CPU/GPU validation
.\build\gme.exe --bench --cpu           # CPU-only generation path
```

3D view controls: hold the **right mouse button** to look around, **WASD** to move, **Space** / **C** to go up / down, **Shift** for 8× speed, **mouse wheel** to change base speed, **F** to toggle shadows, **V** to switch classic / new renderer, **K** to cycle the quality preset, **P** to save a screenshot, **Esc** to quit.

View options: `--width`, `--height`, `--scale 0.5` (render at a lower resolution), `--fps N` (0 means uncapped), `--spawn X Z [EYE]`, `--look YAW PITCH`, `--region K` (start at the coast of life island K), `--frames N --screenshot file.png` (render N frames, save, exit).

World options: `--vram 0..1` (share of free VRAM for chunks, default 0.6), `--ram 0..1` (share of free RAM for the host mirror, default 0.5), `--batch N`, `--radius N`, `--seed N`, `--planet file`, `--biomap file` / `--no-life` / `--biomap-test` / `--biomap-test2` (the test pattern with every crown class and aspect), `--sky-planet` (the planet file's atmosphere instead of the one life has made), `--fauna file` (creatures; default `life/home.fauna` if present) / `--no-fauna` / `--fauna-test [N]` (a built-in herd around the view point).

## Genesis: booting a world from physics

`gme_genesis` builds the world in stages, and each stage writes plain files into a run folder:

1. **cosmos**: a star of 0.8–1.2 solar masses forms with a dust disk. How much rock, ice and organics condense depends on the disk temperature (frost line). The planetesimals then merge under N-body gravity on the GPU, gas drag damps their orbits, and cores above about 8 Earth masses capture hydrogen and helium to become gas giants. Output: `system.txt`, `system.png`, disk frames.
2. **chem**: the chosen planet's crust is turned into atoms with valences. Monte Carlo moves form, break and swap bonds by real bond energies while the planet cools from 2500 K to its surface temperature. The molecules that survive are the materials. Reactors are independent, so `--shard i --shards n` splits them across machines.
3. **finish**: merges the chemistry shards into `home.planet`, which holds oceans, ice, soil, relief, atmosphere (with a volcanic CO2 floor), star colour and the other planets in the sky.

```
.\build\gme_genesis.exe all --dir genesis --seed 7          # everything locally (small)
.\build\gme_genesis.exe chem --dir genesis --reactors 256   # bigger chemistry for a chosen system
.\build\gme_genesis.exe finish --dir genesis
```

The included `genesis/home.planet` is system 59: a 1.00 Earth-mass planet at 0.65 AU, 22 °C, with oceans over 10% of the surface, soil on land and an early-Earth atmosphere of methane, hydrogen sulfide and ammonia. It was the best of 517 systems searched on the cluster (`search_archive/search_ranking.txt`).

Simplifications to know about: time is compressed (bodies merge inside half their Hill radius), crust volatile loss and the greenhouse effect are tuned estimates, and ocean coverage is a fitted curve. Each of these is marked in the code.

## Life: an evolving biosphere

Life is simulated offline by `gme_genesis` on square **islands** of real terrain (16.4 km, 1024 × 1024 cells of 16 m) and frozen into `life/home.biomap`, which `gme.exe` loads (hash-checked against the planet) to grow mats, kelp, ground cover, shrubs and trees. Organisms are colonies or individuals with a genome of enzymes, body genes and a small neural network; nothing is scripted: chemistry, light, tides and selection decide what evolves. Plan and formulas: [docs/biosphere-spec.md](docs/biosphere-spec.md).

```
.\build\gme_genesis.exe env --scout                      # pick 6 islands -> life/regions.txt (+ regions.png)
.\build\gme_genesis.exe env --island 5                   # island 5's environment -> runs/bio_<planet>/island_5/env.bin, env.png
.\build\gme_genesis.exe life --island 5 --cap 1000000    # run it on GPU 0 (Ctrl+C checkpoints and stops)
.\build\gme_genesis.exe life --island 5 --resume         # continue from the newest checkpoint
.\build\gme_genesis.exe bioexport --run runs\bio_<planet>\island_5 --out life\islands\5\biopart_5.bin
.\build\gme_genesis.exe bioexport --merge life\islands --out life\home.biomap
.\build\gme_genesis.exe life-test                        # test suite (--cpu for the CPU reference)
```

`life` options: `--device D` or `--cpu`, `--cap N` agents (0 = from free memory), `--duty 0..1` (share of GPU time), `--years Y` / `--days D` (0 = forever), `--minutes M` (wall-clock limit), `--checkpoint-min M` (default 30; the last 3 checkpoints are kept), `--salt S`, `--seed-protocells N` (default 20000), `--maps-every Y`, `--metrics-every D`, `--features LIST` / `--enable LIST`, `--spectrum FILE` / `--no-spectrum`, `--bank FILE` (found the run from a seed bank), `--no-hub`. It exits 3 when life goes extinct. One step is one hour, one sim-year is 8760 steps.

A run folder holds `env.bin`, `ckpt_*.bin`, `metrics.csv` (daily: population, carbon by realm, pathway fluxes, atmosphere, conservation residuals, mean genes, animals and sex), `census.csv` (yearly carbon by guild and realm, species count), `stages.log` (gates G0–G5, interventions), `maps/` (yearly PNGs) and, with the hub, `outbox/`, `inbox/` and `imported/`. `biostat --run DIR` summarises the newest checkpoint.

Compressions to know about: the island atmosphere is a box of 1e-4 of the real air column, tides are ±1 m, one agent is a clonal colony (super-individual) of up to 1e9 mmol C, and the population is held at 0.95 of `--cap` by a fair birth lottery. Each island has its own land-barrier multiplier (×0.7 … ×1.3) as a parameter study.

### Features: animals, sex, learning, light in bands, plant shapes, the hub

Six additions are **features** of a run ([docs/biosphere-spec.md](docs/biosphere-spec.md) section 10):

| Feature | What it adds |
|---|---|
| `animals` | Predation and grazing: a mouth, a motor and the will to attack make an animal. Prey is edible when one of its individuals fits the attacker's gape; anchored colonies are grazed in bites, but only by attackers that can move a metre an hour (the slower ones take what comes to them). A gut, trophic levels, senses for danger and food, and the land-animal gate G4. |
| `sex` | A parent with the sex gene crosses its genome with a compatible cellmate (mate tags within 3 bits), block by block. Alone it still reproduces alone. |
| `learn` | Learning by exploration: plastic outputs try small random nudges and keep what paid. On the foraging test it beats no learning by 26% and beats the same exploration with frozen weights by 9%. |
| `spectral` | Light in 9 wavelength bands from `life/spectrum.bin` (written by `gme --export-spectrum`): water takes the red first, methane haze colours the sky, and a pigment only harvests the photons its reaction centre can use. |
| `morph` | Crown shapes: class (ellipsoid, cone, umbrella, column, dome), height : width, leaf angle, porosity. A crown catches no more light than falls on it, leaves saturate in strong light, and plants shade each other with what they intercept; shapes are exported to the engine. Switching it on in an evolved world re-sorts its vegetation: on island 1 the shrub cells fell from 23% to 11% in one sim-year and to 7% in three (spec 10.6). Try it on one island first. |
| `hub` | The island keeps the books the planet-wide atmosphere needs and trades emigrants through its outbox and inbox. |

A **new** run has all of them by default, light in bands only when `life/spectrum.bin` is there (choose with `--features animals,sex,...` or `none`; an explicit `all` or `spectral` requires the file). A **resumed** run keeps the features of its checkpoint: a new build changes nothing on a running island. `life --resume --enable LIST` switches features on in a living world; that is logged in `stages.log` as an intervention. With every feature off the simulation is the first release, bit for bit, which the test suite checks against recorded state hashes.

```
.\build\gme.exe --export-spectrum life/spectrum.bin              # the planet's light in bands, once per planet
.\build\gme_genesis.exe life --island 5 --resume --enable animals,sex   # switch features on in a running world
.\build\gme_genesis.exe biostat --run DIR --species              # species = clusters of similar genomes
.\build\gme_genesis.exe life-hub --islands 0,1 --runs DIR0,DIR1  # one pass of the island hub on this machine
.\build\gme_genesis.exe life-seedbank --runs runs\bio_<planet>   # best genomes per guild x habitat -> life/seedbank.bin
.\build\gme_genesis.exe life-rebase --bank life\seedbank.bin --island 2   # a new run (also on another planet) founded from the bank
.\build\gme_genesis.exe life-inoculate --from DIR_A --to DIR_B --guild land_plant --n 500
```

- **Hub.** Each sim-year an island writes its gas uptake and its emigrants (its most successful genomes of each guild in each habitat) to `outbox/`. `life-hub` moves the planet's atmosphere by the islands' uptake per m² of ocean and of land, weighted by the planet's real ocean cover, and routes emigrants around the ring. One hub folder serves one planet. Islands move a quarter of the way to the planet's atmosphere and take in the migrants at their next sim-year. Without a hub an island runs closed.
- **Seed bank.** Genomes carry no planet state. A bank from one planet's islands founds runs on another (`life-rebase`), and `life-inoculate` moves one guild between islands. Both are logged.
- **Export.** `bioexport` writes the vegetation with its evolved shapes (biomap version 2) and the creatures (`fauna_<K>.bin`: what moves or lives on prey and is not vegetation); `bioexport --merge` also writes `life/home.fauna` for the engine.

## Running on a GPU cluster

The scripts in `cluster/` drive Linux rigs over SSH (key login) and tmux. Configure your machines first:

```
copy cluster\hosts.example.ps1 cluster\hosts.local.ps1   # then edit: SSH user, rig A, rig B (git-ignored)
```

On each rig: the NVIDIA driver, gcc, cmake, tmux, and an NVRTC library (for example `pip install nvidia-cuda-nvrtc` in a venv under `~/venvs`, or set `GME_NVRTC_DIR`).

```
pwsh cluster/launch.ps1 -Deploy         # star-system search on every rig GPU (endless; several processes per GPU)
pwsh cluster/status.ps1 [-Fetch]        # habitability ranking; -Fetch copies the best system to ./genesis
pwsh cluster/bio_launch.ps1 -Deploy     # life islands: copy + build in ~/gme/build-bio, check env CRCs, start tmux bio-isl<k>
pwsh cluster/bio_launch.ps1 -Deploy -Restart -Enable animals,sex   # new build, and switch features on in the running islands
pwsh cluster/bio_status.ps1 [-Fetch]    # per island progress; -Fetch exports and merges into life/home.biomap and life/home.fauna
pwsh cluster/bio_hub.ps1                # every 10 min: outboxes -> life/hub/<planet>, planet-wide atmosphere + migrants, -> inboxes
pwsh cluster/timeline.ps1               # every 30 min: fetch, screenshots of every island, charts; stops when "playable"
pwsh cluster/render_remote.ps1 -Run "--frames 600 --screenshot {out}/view.png --perf" -Out status/render/x
                                        # render tests on rig B: headless Linux build of gme, coolest free GPU
```

Render tests run headless on a Linux rig: `gme` builds there without a display (`src/window_headless.c`, frames go to PNG). The CUDA compiler comes from NVIDIA's pip wheels in user space (`pip install --target ~/gme/toolchain "nvidia-cuda-nvcc==13.2.*" "nvidia-nvvm==13.2.*" "nvidia-cuda-crt==13.2.*" "nvidia-cuda-runtime==13.2.*" "nvidia-cuda-cccl==13.2.*"`, plus a `libcudart.so` symlink in its `lib`). Pin every wheel to the same version, or `ptxas` rejects the front end's PTX. The wrapper allows at most two jobs at a time and waits while a GPU is above 78 °C, because that rig's three GPUs sit close together.

## Layout

| File | Role |
|---|---|
| `src/platform.*`, `src/jobs.*` | Hardware detection, timers, large allocations; a worker pool that uses every hardware thread |
| `src/worldgen.h` | Space layout (1 voxel = 1 m, 32³ chunks) and the fixed-point terrain generator shared by C and CUDA (bit-identical CPU/GPU) |
| `src/gpu.*`, `src/world.*` | CUDA chunk pool and batched generation; residency, streaming, eviction, adaptive view radius, validation |
| `src/render.*`, `src/window_win32.c` | CUDA ray tracer and the native window |
| `src/density.h`, `src/trace.cuh` | Render upgrade (in progress): smooth density surface and its column tracer |
| `src/flora.h`, `src/biomap.*` | Vegetation voxels from the frozen biomap: the `GMEBMAP1` file, and `GMEBMAP2` with evolved crown shapes |
| `src/fauna_scene.*` | Creatures in the 3D view: placement on the drawn ground or at depth, a hash grid of 32 m cells the renderer walks |
| `src/planetfile.*` | `home.planet` load/save and the planet hash |
| `src/rtcuda.*` | Run-time CUDA: driver API + NVRTC loaded dynamically (no toolkit needed) |
| `src/genesis/` | `gme_genesis`: cosmos (N-body), chem (bond Monte Carlo), planet derivation |
| `src/bio/` | Life simulation: `bio_api.h` (contract), `bio_core.h` + `bio_kernels.cu` (integer math, NVRTC kernels), `bio_sim.c` / `bio_cpu.c` (GPU driver / CPU reference), `bio_io.c` (checkpoints, vegetation), `bio_bank.c` (seed bank, seeding, creatures), `bio_hub.c` (island hub), `bio_stats.c` (species), `bio_env.c`, `bio_main.c`, `bio_test.c` |
| `src/fauna.h`, `src/spectrum_file.h` | File contracts between the life simulation and the engine: creatures (`GMEFAUN1`), light in bands (`GMESPEC1`) |
| `cluster/` | Cluster scripts (search, life islands, timeline) |
| `docs/` | Design specs: biosphere and render upgrade |

## Space model

- A position is a chunk coordinate (int32) plus a float offset inside the chunk, so precision is the same everywhere. The horizontal range is ±33,000 km. The vertical range is 16 layers (512 m) by default and can go up to 32.
- A chunk that is all one material (air, solid stone or open water) takes 24 bytes and no voxel storage.
- Each non-uniform chunk uses one 32 KB slot, with the same index in VRAM and in the host mirror (for CPU-side physics and gameplay queries).
- The view radius grows until the storage budget is full and shrinks if storage runs out.

## License

MIT, see [LICENSE](LICENSE).

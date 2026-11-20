# The life upgrades: start them and test them

This page is the short way in to what section 10 of [biosphere-spec.md](biosphere-spec.md) and section 5.5 of [render-upgrade-spec.md](render-upgrade-spec.md) describe: animals, sexual reproduction, learning, light in wavelength bands, crown shapes, the island hub, the seed bank, and creatures and crown shapes in the 3D view.

All commands run from the repository root in PowerShell.

## 1. Build

```
.\build.bat
```

That builds `gme.exe` (the engine), `gme_genesis.exe` (the simulation) and `gme_biotest.exe` (its tests) into `.\build`.

## 2. See it in the 3D view

No data is needed for the built-in patterns:

```
.\build\gme.exe --biomap-test2 --fauna-test
```

- `--biomap-test2` plants the test pattern with every crown class (ellipsoid, cone, umbrella, column, dome) in every height : width bin around the start point.
- `--fauna-test [N]` puts a herd of N test creatures (default 4096) on a 6 m grid around the view point: 0.25 to 4 m across, walkers on the ground and swimmers below the water plane, grazers and darker hunters, some armoured, some as swarms. A swimmer over land and a walker over water are left out, so over dry ground about half of the herd is drawn.
- Hold the right mouse button to look, WASD to move, Shift for speed. **L** prints what is under the crosshair (a creature reports as `creature`), **K** cycles the quality preset, **V** switches to the classic voxel renderer (the same crowns as voxels), **T** steps the time of day.

One image per crown class and aspect, without the window: `.\build\gme.exe --biomap-test2 --sheet sheet.png`.

A real world is opened the same way, with the files the tour below writes (or `life/home.biomap` and `life/home.fauna`, which load by default):

```
.\build\gme.exe --biomap runs\demo_upgrades\home.biomap --fauna runs\demo_upgrades\home.fauna
```

A version 2 biomap also carries the atmosphere life has made, and the sky is computed from it. `--sky-planet` shows the planet file's sky instead, `--no-fauna` leaves the creatures out.

## 3. The tour: every upgrade on a copy of an evolved island

```
.\demo\upgrades.ps1
```

It needs an evolved checkpoint. By default it takes `milestones\playable_v1\island_1` (sim year 2751, 2.85 M agents); another run is named with `-From <run folder> -Island <K>`. The source is only read: the tour works on a copy in `runs\demo_upgrades` (1.1 GB when it is done, 2 GB at its peak). On an RTX 3060 with nothing else on the GPU the whole tour takes about 9 minutes; `-Stages animals,export` is the short version (about 3 minutes).

| Stage | What it runs | What to look for |
|---|---|---|
| `animals` | `life --resume --enable all` on the copy, to the next whole sim-year, then `biostat --species` | the `animals:` line: creatures, agents that live mostly on prey, kills and bites per day, sexual agents and births with two parents; `species:` clusters of similar genomes |
| `bank` | `life-seedbank`, `life-rebase` onto island 5, one sim-year | the bank's table of genomes per guild and habitat; the new island passes its gates within a few sim-years instead of centuries |
| `hub` | `life-hub` over both islands, twice, with a sim-year after each | `hub:` the planet's atmosphere and the titration; `life: hub at year N: M migrants imported` on each island |
| `export` | `bioexport` of both islands and `bioexport --merge` | `shapes of N shrub and tree cells`, `version 2`, the atmosphere for the sky, the creature count; then the viewer commands, one per island placed where it has the most creatures |

Every command is printed before it runs, so the tour doubles as a list of the command lines. The complete tool output is in `runs\demo_upgrades\demo.log`. Run it again and the islands go one step further; `-Fresh` starts over.

The end of the reference run (RTX 3060, 2026-11-20, from `milestones\playable_v1\island_1`; the simulation is deterministic, so the same checkpoint gives the same numbers):

```
== done in 8.5 min; the complete output is in runs\demo_upgrades\demo.log
  island 1, the evolved one, at the end of the animals stage (features all): isl1 y=2752.00 live=2.85M (motile 161.8k, sessile photo 2.74M)
    animals: 74 creatures, 0 land animals; 132 agents live mostly on prey (trophic level 2.77); per day 20 kills, 832 bites, 2.01e+03 mol C eaten | sexual 144.6k, 6.5 births/d with two parents
    species: 154 clusters in a sample of 4096 genomes (88 with 2+ members), Shannon H 0.91, Simpson 0.228, largest 88%
  island 5, founded from the seed bank, at the end of the bank stage: isl5 y=1.00 live=379.9k (motile 15.8k, sessile photo 147.8k)
    species: 77 clusters in a sample of 4096 genomes (51 with 2+ members), Shannon H 1.82, Simpson 0.647, largest 58%
  hub, pass 2: planet atmosphere: H2S 4.4819% CH4 6.8601% NH3 3.9893% CO2 9.6731% O2 74.9954% N2 0.0001% H2 0.0000%
  hub, pass 2: titration: the planet's air is oxidised: O2 is in excess of the reductants
  island_1: step=24125040 year=2754.000 HUB 765 migrants imported, atmosphere moved a quarter toward the planet's
  island_5: step=26280 year=3.000 HUB 667 migrants imported, atmosphere moved a quarter toward the planet's
  export: shapes of 78850 shrub and tree cells (78850 with evolved shape): ellipsoid 90.8% cone 1.1% umbrella 2.7% column 3.9% dome 1.5%; mean height : width 1.04
  export: wrote runs\demo_upgrades\home.biomap (version 2): 2 region(s), sim year 2754, gates 2f, planet 319a5581e7884e3a seed 59
  export: atmosphere for the sky: H2S 1.85% CH4 11.14% NH3 6.47% CO2 13.668% O2 66.872% N2 0.00% H2 0.000%
  export: wrote runs\demo_upgrades\home.fauna: 101 of 101 creatures from 2 island file(s)
  creatures of island 1: 65 in the file; the 64 m square with the most holds 3 around (88415, -14336): largest 0.50 m, all on land or on the shore
  creatures of island 5: 36 in the file; the 64 m square with the most holds 14 around (-98302, -94213): largest 0.13 m, all on land or on the shore
  island 5's gates since it was founded:
    step=264 year=0.030 GATE G2 oxygenic reached (gates G2)
    step=8952 year=1.022 GATE G0 established reached (gates G0 G2)
    step=17544 year=2.003 GATE G5 oxidation reached (gates G0 G2 G5)
    step=19200 year=2.192 GATE G3 land plants reached (gates G0 G2 G3 G5)

Look at it (the 3D view needs the engine build and a CUDA GPU):
  .\build\gme.exe --biomap runs\demo_upgrades\home.biomap --fauna runs\demo_upgrades\home.fauna      # island 1 (region 0): evolved crown shapes, the sky of the air life has made
  .\build\gme.exe --biomap runs\demo_upgrades\home.biomap --fauna runs\demo_upgrades\home.fauna --region 1      # island 5, founded from the seed bank
  .\build\gme.exe --biomap runs\demo_upgrades\home.biomap --fauna runs\demo_upgrades\home.fauna --spawn 88415 -14336 6 --look 35 -30      # where island 1 has the most creatures
  .\build\gme.exe --biomap runs\demo_upgrades\home.biomap --fauna runs\demo_upgrades\home.fauna --spawn -98302 -94213 6 --look 35 -30      # where island 5 has the most creatures
  .\build\gme.exe --biomap-test2 --fauna-test      # built-in patterns: every crown shape, a test herd
```

Of the 101 creatures in that file the engine's placement keeps 100 (93 walkers on dry ground, 7 swimmers; 75 of them swarms, drawn as rings of six spheres). Fourteen stand within 32 m of the island 5 spot, three within 32 m of the island 1 spot and 22 within a kilometre of it.

Things to know before reading the numbers:

- **Animals start from very little.** A world of the first release never selected for a mouth or a motor, so the first sim-years show tens to a few hundred creatures among 2.85 M agents. Most are plants by descent that have begun to walk and graze, small (under a metre) and living in swarms. None is a land animal in the sense of gate G4, which asks for 1000 fast land consumers over a sim-year. Whether grazers spread is for long runs to show (spec 10.1 and 10.12).
- **Crown shapes re-sort an evolved vegetation.** With `morph` the share of shrub cells falls and ground cover rises within a sim-year (spec 10.6). The tour switches it on because it works on a copy. On the real islands, switch it on deliberately and keep a checkpoint.
- **The hub in the tour joins a 2750-year-old island and a newborn one.** The planet's air is the mean of the two, so the old island's air moves toward the young island's reducing mix. That is the hub doing its arithmetic on an odd pair, not a fault.

## 4. Test each upgrade

The tests are small deterministic worlds. Each prints the numbers it measured and ends in `pass` or `FAIL`; a selection takes seconds. Select them with `GME_BIOTEST`:

```
$env:GME_BIOTEST = 'T10'; .\build\gme_biotest.exe; Remove-Item Env:GME_BIOTEST
```

| Upgrade | Tests | What they check |
|---|---|---|
| Animals | `T10`, `T2`, `T3`, `T4` | T10: plankton and hunters cycle, the predator's biomass follows the prey's (cross-correlation peak at a positive lag). T2c: CPU and GPU give the same state hash with plants, grazers, hunters and sex. T3b: C, N, S and Fe exactly conserved with kills and bites. T4b: a run resumed from a checkpoint equals the uninterrupted one |
| Sexual reproduction | `T15` | recombinants appear between compatible mates, none between incompatible ones, none with the feature off |
| Learning | `T9` | on a moving food patch, learning foragers take in more than non-learners and more than exploration with frozen weights |
| Light in bands | `T13` | calibration against the scalar light, the water profile per band, pigments adapt to depth, CPU equals GPU, resume refuses another light file |
| Crown shapes | `T16`, `T19` | a crown never catches more light than falls on its top, weak-light parity with the old model; a shape word on every vegetated cell of the export |
| Island hub | `T17` | migrants both ways with exact ledgers, the islands' air pulled together, exact replay from an older checkpoint; T17b: the hub's arithmetic on hand-written files |
| Seed bank | `T18` | bank round trip, a new run founded on another planet, inoculation of one guild |
| Species and creatures | `T19` | cluster counts of known populations; the creature file is sorted, inside the island, and holds no anchored plant |
| Running worlds are safe | `T14` | with every feature off the simulation is the first release: the recorded state hashes are reproduced |

The whole suite (`.\build\gme_biotest.exe`, no selection) takes 2 minutes on an RTX 3060 and ends with `OK: 0 test(s) failed`. `--cpu` runs it on the CPU reference, which must print the same hashes. T12 is a speed gate (150 steps a second at 1 M agents on the 3060): run it with nothing else on the GPU.

The engine has its own self-tests (`gme --selftest`, or a subset such as `--selftest d2,d5`) and world hashes (`gme --bench`). On this project they run on the render rig, not on the workstation:

```
pwsh cluster/render_remote.ps1 -Run "--selftest d2,d5" -Out status/render/selftest
pwsh cluster/render_remote.ps1 -Run "--biomap-test2 --fauna-test --frames 300 --screenshot {out}/shapes.png" -Out status/render/shapes
```

## 5. On the real islands

```
pwsh cluster/bio_launch.ps1 -Deploy -Restart -Enable animals,sex,learn,spectral   # new build, features on in the running islands
pwsh cluster/bio_hub.ps1                                                          # the hub: every 10 min
pwsh cluster/bio_status.ps1 -Fetch                                                # export and merge into life/home.biomap and life/home.fauna
```

- A resumed island keeps the features of its checkpoint, so deploying the new build alone changes nothing. `-Enable` is the intervention, and it is logged in each island's `stages.log`.
- Switching a feature on cannot be undone except from an older checkpoint. `milestones\playable_v1` is that checkpoint for both islands.
- `spectral` needs `life/spectrum.bin` on the rigs; `-Deploy` copies it.
- Leave `morph` for one island first (see above).
- The new options of these three scripts (`-Enable`, the hub loop, the creature files in `-Fetch`) have been checked for syntax only. They have not been run against the rigs yet: watch the first run.

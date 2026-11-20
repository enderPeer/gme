# Boot the life simulation on the rigs: one tmux session bio-isl<k> per island, each
# running cluster/bio_island.sh (docs/biosphere-spec.md 6.1-6.5). The star search
# (gme-* sessions, ~/gme/build) is never touched.
#   pwsh cluster/bio_launch.ps1 [-Deploy] [-Duty 0.6] [-Islands 0,1,2,3,4] [-Restart] [-Enable LIST]
#     -Deploy   copy src, CMakeLists.txt, cluster/bio_island.sh, genesis/home.planet,
#               life/regions.txt and life/spectrum.bin (when present) to ~/gme on the rigs,
#               then build gme_genesis in ~/gme/build-bio
#     -Restart  stop running bio-isl<k> sessions of these islands (Ctrl+C: they checkpoint
#               first) and start them again, e.g. after -Deploy of a new build
#     -Enable   features to switch on in the islands' living worlds when they (re)start:
#               animals, sex, learn, morph, hub, spectral, or all (docs/biosphere-spec.md 10).
#               An island keeps the features of its checkpoint; a new build alone changes
#               nothing. Enabling is logged in each stages.log as an intervention and cannot
#               be undone except by going back to an older checkpoint. spectral needs
#               life/spectrum.bin (gme --export-spectrum life/spectrum.bin, then -Deploy).
# Before starting an island its env.bin is built on the rig and its CRC compared with
# this PC's (life/env_crc.txt); on a mismatch that island is not started.
param(
    [switch]$Deploy,
    [double]$Duty = 0.6,
    [object[]]$Islands = @(0, 1),   # focus: rig A only (two large GPUs); islands 2-4 (rig B) are off
    [switch]$Restart,
    [object[]]$Enable = @()
)
$ErrorActionPreference = 'Stop'
# "-Islands 0,1" arrives as the one string "0,1" when the script is started as pwsh <script> (file mode)
$Islands = @($Islands | ForEach-Object { "$_" -split '[,\s]+' } | Where-Object { $_ -ne '' } | ForEach-Object { [int]$_ })
. (Join-Path $PSScriptRoot 'hosts.ps1')   # $GmeUser, $GmeRigA, $GmeRigB
Set-Location (Split-Path $PSScriptRoot)

$user = $GmeUser
$map = @{   # island -> rig, CUDA device (driver order: rig A 0 = 24 GB card, 1 = 16 GB card), agent cap (spec 6.1)
    0 = @{ Host = $GmeRigA; Dev = 0; Cap = 4000000 }
    1 = @{ Host = $GmeRigA; Dev = 1; Cap = 3000000 }
    2 = @{ Host = $GmeRigB; Dev = 0; Cap = 2000000 }
    3 = @{ Host = $GmeRigB; Dev = 1; Cap = 2000000 }
    4 = @{ Host = $GmeRigB; Dev = 2; Cap = 2000000 }
}
foreach ($k in $Islands) { if (-not $map.ContainsKey($k)) { throw "island $k has no rig slot (0-4; 5 is the PC lab)" } }
$genesis = '.\build\gme_genesis.exe'
if (-not (Test-Path $genesis)) { throw "$genesis missing: run .\build.bat first" }
if (-not (Test-Path life/regions.txt)) { throw "life/regions.txt missing: run $genesis env --scout first" }

# the planet these islands belong to (runs/bio_<hash>/), and this PC's env CRC per island
$hash = (& $genesis env --info | Select-String '^planet_hash ([0-9a-f]{16})').Matches[0].Groups[1].Value
if (-not $hash) { throw "cannot read the planet hash (genesis/home.planet)" }
$crc = @{}
foreach ($k in $Islands) {
    $out = & $genesis env --island $k --crc-only 2>&1
    $m = $out | Select-String 'env crc ([0-9a-f]{8})'
    if (-not $m) { throw "env --island $k failed on this PC:`n$($out -join "`n")" }
    $crc[$k] = $m.Matches[0].Groups[1].Value
}
$known = @{}
if (Test-Path life/env_crc.txt) {
    foreach ($l in Get-Content life/env_crc.txt) { if ($l -match '^(\d+) ([0-9a-f]{8}) ([0-9a-f]{16})$' -and $Matches[3] -eq $hash) { $known[[int]$Matches[1]] = $Matches[2] } }
}
foreach ($k in $Islands) { $known[$k] = $crc[$k] }
($known.Keys | Sort-Object | ForEach-Object { "$_ $($known[$_]) $hash" }) | Set-Content life/env_crc.txt
Write-Host "planet $hash; PC env CRCs: $(($Islands | ForEach-Object { "$_=$($crc[$_])" }) -join ' ') (life/env_crc.txt)"

# -Enable: one lower-case, comma-separated string of known names (gme_genesis compares them exactly;
# "animals,sex" is an array when the script is called directly and one string through pwsh <script>)
$en = (@($Enable | ForEach-Object { "$_".ToLowerInvariant() -split '[,\s]+' } | Where-Object { $_ -ne '' }) -join ',')
foreach ($f in @($en -split ',' | Where-Object { $_ })) {
    if (@('animals', 'sex', 'learn', 'morph', 'hub', 'spectral', 'all') -notcontains $f) { throw "-Enable: unknown feature '$f' (animals, sex, learn, morph, hub, spectral, all)" }
}
if ($en -match 'spectral|all' -and -not (Test-Path life/spectrum.bin)) {
    throw "light in bands needs life/spectrum.bin: run .\build\gme.exe --export-spectrum life/spectrum.bin, then -Deploy"
}
$hosts = $Islands | ForEach-Object { $map[$_].Host } | Sort-Object -Unique
$buildOk = @{}
foreach ($h in $hosts) { $buildOk[$h] = $true }
if ($Deploy) {
    foreach ($h in $hosts) {
        ssh "$user@$h" 'mkdir -p ~/gme/cluster ~/gme/genesis ~/gme/life'
        scp -q -r src CMakeLists.txt "${user}@${h}:gme/"
        scp -q cluster/bio_island.sh "${user}@${h}:gme/cluster/"     # only this script: the search's scripts stay as they are
        scp -q genesis/home.planet "${user}@${h}:gme/genesis/home.planet"
        scp -q life/regions.txt "${user}@${h}:gme/life/regions.txt"
        if (Test-Path life/spectrum.bin) { scp -q life/spectrum.bin "${user}@${h}:gme/life/spectrum.bin" }   # light in bands
        ssh "$user@$h" 'cd ~/gme && chmod +x cluster/bio_island.sh && cmake -S . -B build-bio -DGME_BUILD_ENGINE=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null && nice -n 10 cmake --build build-bio -j 8 --target gme_genesis >/dev/null 2>&1 && echo "built ~/gme/build-bio/gme_genesis on $(hostname)" || echo "BUILD FAILED on $(hostname)"'
        if ($LASTEXITCODE -ne 0) { $buildOk[$h] = $false; Write-Warning "${h}: the build failed; its islands are left as they are" }
    }
}

foreach ($k in $Islands) {
    $p = $map[$k]; $h = $p.Host
    $run = "runs/bio_$hash/island_$k"
    $name = "bio-isl$k"
    if (-not $buildOk[$h]) { continue }       # a running island is never stopped for a build that failed
    if ($en -match 'spectral|all' -and (ssh "$user@$h" 'test -f ~/gme/life/spectrum.bin && echo yes') -ne 'yes') {
        Write-Warning "island ${k}: ~/gme/life/spectrum.bin is not on $h (run with -Deploy): left as it is"; continue
    }
    # environment on the rig (deterministic: must equal this PC's CRC)
    $out = ssh "$user@$h" "cd ~/gme && nice -n 10 ./build-bio/gme_genesis env --island $k --run $run 2>&1"
    $m = $out | Select-String 'env crc ([0-9a-f]{8})'
    if (-not $m) { Write-Warning "island ${k}: env failed on ${h}:`n$($out -join "`n")"; continue }
    $rcrc = $m.Matches[0].Groups[1].Value
    if ($rcrc -ne $crc[$k]) { Write-Warning "island ${k}: env crc $rcrc on $h differs from this PC's $($crc[$k]) - not started"; continue }
    $running = ssh "$user@$h" "tmux has-session -t $name 2>/dev/null && echo yes"
    if ($running -eq 'yes') {
        if (-not $Restart) { Write-Host "$h GPU $($p.Dev): island $k already running (tmux $name), env crc $rcrc ok"; continue }
        Write-Host "$h GPU $($p.Dev): stopping island $k (Ctrl+C, waiting for its checkpoint)..."
        ssh "$user@$h" "tmux send-keys -t $name C-c"
        for ($i = 0; $i -lt 180; $i++) {
            Start-Sleep -Seconds 5
            if ((ssh "$user@$h" "tmux has-session -t $name 2>/dev/null && echo yes") -ne 'yes') { break }
        }
        if ((ssh "$user@$h" "tmux has-session -t $name 2>/dev/null && echo yes") -eq 'yes') { Write-Warning "island ${k}: $name still running after 15 min, left alone"; continue }
    }
    ssh "$user@$h" "tmux new-session -d -s $name '~/gme/cluster/bio_island.sh $($p.Dev) $k $hash $Duty $($p.Cap) $en'"
    Write-Host "$h GPU $($p.Dev): island $k started (tmux $name, cap $($p.Cap), duty $Duty, env crc $rcrc, ~/gme/$run)$(if ($en) { ", switching on: $en" })"
}

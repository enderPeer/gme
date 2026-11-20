# Progress of the life-simulation islands on the rigs (docs/biosphere-spec.md 6.5),
# with the star search's seeds per hour as a coexistence check.
#   pwsh cluster/bio_status.ps1 [-Fetch] [-Islands 0,1,2,3,4] [-LabRun runs/bio_<hash>/island_5]
#     -Fetch  export each island's newest checkpoint on its rig (gme_genesis bioexport on the
#             CPU, so the running island keeps its GPU), copy the part, its creatures
#             (fauna_<k>.bin), metrics, census, stages.log and the newest maps into
#             life/islands/<k>/, export the PC lab island too when its run folder exists, and
#             merge all parts into life/home.biomap and life/home.fauna
param(
    [switch]$Fetch,
    [object[]]$Islands = @(0, 1),   # focus: rig A only (two large GPUs); islands 2-4 (rig B) are off
    [string]$LabRun = ''
)
$ErrorActionPreference = 'Stop'
# "-Islands 0,1" arrives as the one string "0,1" when the script is started as pwsh <script> (file mode)
$Islands = @($Islands | ForEach-Object { "$_" -split '[,\s]+' } | Where-Object { $_ -ne '' } | ForEach-Object { [int]$_ })
. (Join-Path $PSScriptRoot 'hosts.ps1')   # $GmeUser, $GmeRigA, $GmeRigB
Set-Location (Split-Path $PSScriptRoot)

$user = $GmeUser
$map = @{
    0 = @{ Host = $GmeRigA; Dev = 0 }
    1 = @{ Host = $GmeRigA; Dev = 1 }
    2 = @{ Host = $GmeRigB; Dev = 0 }
    3 = @{ Host = $GmeRigB; Dev = 1 }
    4 = @{ Host = $GmeRigB; Dev = 2 }
}
$genesis = '.\build\gme_genesis.exe'
$hash = (& $genesis env --info | Select-String '^planet_hash ([0-9a-f]{16})').Matches[0].Groups[1].Value
if (-not $hash) { throw "cannot read the planet hash (genesis/home.planet)" }
Write-Host "planet $hash"

# remote snippets: single-quoted (no PowerShell expansion), placeholders filled in below
$hostCmd = @'
cd ~/gme 2>/dev/null || exit 0
tmux ls 2>/dev/null | grep -E '^(bio-|gme-)' | cut -d: -f1 | tr '\n' ' '; echo
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
if [ -f runs/progress.log ]; then
  tail -n 3000 runs/progress.log | awk -v now="$(date +%H:%M:%S)" '
    function sec(t,  a) { split(t, a, ":"); return a[1] * 3600 + a[2] * 60 + a[3] }
    / exit / { d = sec(now) - sec($1); if (d < 0) d += 86400; if (d <= 3600) n++ }
    END { printf "search: %d seeds finished in the last hour\n", n }'
fi
'@
$islandCmd = @'
cd ~/gme/@RUN@ 2>/dev/null || { echo "  island @K@: no run folder ~/gme/@RUN@"; exit 0; }
ck=$(ls ckpt_*.bin 2>/dev/null | sort | tail -n 1)
rb=$(grep -c 'INTERVENTION rollback' stages.log 2>/dev/null)
echo "  island @K@: newest checkpoint ${ck:-none}, ${rb:-0} rollbacks"
[ -f metrics.csv ] && awk -F, 'END { if (NR > 1) printf "    year %.2f  live %s  sessile photo %s  C marine %.3g land %.3g mol  O2aq %.3g uM  %s steps/s  gate mask %s\n", $3, $6, $8, $11 / 1000, $13 / 1000, $37, $5, $NF }' metrics.csv
[ -f metrics.csv ] && awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i } END { if (NR > 1 && c["consumers"]) printf "    animals: %s living on prey (trophic level %s), %s land animals, %s kills and %s bites per day; %s sexual, %s births/day with two parents\n", $c["consumers"], $c["mean_troph"], $c["land_animals"], $c["kills"], $c["bites"], $c["sexual"], $c["sexual_births"] }' metrics.csv
[ -f stages.log ] && tail -n 3 stages.log | sed 's/^/    /'
'@
$exportCmd = @'
cd ~/gme && nice -n 10 ./build-bio/gme_genesis bioexport --run @RUN@ --out @RUN@/biopart_@K@.bin 2>&1 | tail -n 4
cd @RUN@/maps 2>/dev/null && ls kind_y*.png 2>/dev/null | sort | tail -n 1 | sed 's/^kind_\(y[0-9]*\)\.png$/\1/'
'@

$hostCmd = $hostCmd.Replace("`r", ''); $islandCmd = $islandCmd.Replace("`r", ''); $exportCmd = $exportCmd.Replace("`r", '')   # bash needs LF

$hosts = $Islands | ForEach-Object { $map[$_].Host } | Sort-Object -Unique
foreach ($h in $hosts) {
    Write-Host "=== $h ==="
    ssh "$user@$h" $hostCmd | ForEach-Object { Write-Host "  $_" }
    foreach ($k in ($Islands | Where-Object { $map[$_].Host -eq $h })) {
        $run = "runs/bio_$hash/island_$k"
        ssh "$user@$h" $islandCmd.Replace('@RUN@', $run).Replace('@K@', "$k") | ForEach-Object { Write-Host $_ }
    }
}

if ($Fetch) {
    foreach ($k in $Islands) {
        $h = $map[$k].Host
        $run = "runs/bio_$hash/island_$k"
        $dst = "life/islands/$k"
        New-Item -ItemType Directory -Force $dst | Out-Null
        $out = @(ssh "$user@$h" $exportCmd.Replace('@RUN@', $run).Replace('@K@', "$k"))
        $out | Where-Object { $_ -notmatch '^y\d+$' } | ForEach-Object { Write-Host "  island ${k}: $_" }
        if (-not ($out | Select-String 'bioexport: wrote')) { Write-Warning "island ${k}: export failed on $h"; continue }
        scp -q "${user}@${h}:gme/$run/biopart_$k.bin" "$dst/"
        scp -q "${user}@${h}:gme/$run/fauna_$k.bin" "$dst/" 2>$null      # its motile organisms (none before animals)
        foreach ($f in 'metrics.csv', 'census.csv', 'stages.log', 'env.png') { scp -q "${user}@${h}:gme/$run/$f" "$dst/" 2>$null }
        $year = $out | Where-Object { $_ -match '^y\d+$' } | Select-Object -Last 1
        if ($year) {
            New-Item -ItemType Directory -Force "$dst/maps" | Out-Null
            foreach ($m in 'biomass', 'kind', 'hue', 'o2', 'h2s') { scp -q "${user}@${h}:gme/$run/maps/${m}_$year.png" "$dst/maps/" 2>$null }
        }
        Write-Host "  island ${k}: copied into $dst"
    }
    # the PC lab island (5), if it has been run here
    $lab = if ($LabRun) { $LabRun } else { "runs/bio_$hash/island_5" }
    if (Test-Path "$lab/env.bin") {
        New-Item -ItemType Directory -Force life/islands/5 | Out-Null
        & $genesis bioexport --run $lab --out life/islands/5/biopart_5.bin | ForEach-Object { Write-Host "  lab: $_" }
    }
    & $genesis bioexport --merge life/islands --out life/home.biomap | ForEach-Object { Write-Host "  $_" }
    if ($LASTEXITCODE -eq 0) { Write-Host "life/home.biomap updated - gme.exe loads it automatically" }
}

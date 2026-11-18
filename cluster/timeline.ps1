# Timeline: follow the life islands until the world is playable.
# Every -IntervalMin minutes:
#   1. cluster/bio_status.ps1 -Fetch   (export each island's newest checkpoint, merge life/home.biomap)
#   2. screenshots of every island with the engine: a ground view at the coast and an overview
#   3. charts across all islands (cluster/timeline_charts.py)
#   4. one row per island in <Out>/timeline.csv
# It exits 0 when the playable criterion holds (PLAYABLE.txt written), 2 at -MaxHours.
#
# Playable (all tunable): on at least -IslandsNeeded islands, land vegetation
# (ground cover + shrubs + trees) covers >= -VegTarget % of the island's cells and
# woody plants (shrubs + trees) >= -WoodyTarget %, and no island is extinct.
# Focus: islands 0 and 1 on rig A (its two large GPUs); rig B's islands are off.
#   pwsh cluster/timeline.ps1 [-Islands 0,1] [-IntervalMin 30] [-MaxHours 36] [-VegTarget 10] [-WoodyTarget 1] [-IslandsNeeded 2]
param(
    [int[]]$Islands = @(0, 1),
    [int]$IntervalMin = 30,
    [double]$MaxHours = 36,
    [double]$VegTarget = 10.0,
    [double]$WoodyTarget = 1.0,
    [int]$IslandsNeeded = 2,
    [string]$Out = 'status/timeline'
)
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot)
$gme = '.\build\gme.exe'
if (-not (Test-Path $gme)) { throw "$gme missing: run .\build.bat first" }
New-Item -ItemType Directory -Force $Out | Out-Null
$csv = Join-Path $Out 'timeline.csv'
if (-not (Test-Path $csv)) { 'stamp,island,year,live,mat,kelp,cover,shrub,tree,mean_cover,tallest_m,land_veg_pct,woody_pct' | Set-Content $csv }
$start = Get-Date

function Shot([string[]]$GmeArgs, [string]$File) {
    $o = & $gme @GmeArgs --frames 600 --screenshot $File 2>&1 | ForEach-Object { "$_" }
    $m = $o | Select-String 'Spawn: \((-?\d+), (-?\d+), (-?\d+)\), heading (-?[\d.]+)' | Select-Object -First 1
    if ($m) { return @{ x = [int]$m.Matches[0].Groups[1].Value; y = [int]$m.Matches[0].Groups[2].Value; z = [int]$m.Matches[0].Groups[3].Value; yaw = [double]$m.Matches[0].Groups[4].Value } }
    return $null
}

while ($true) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmm'
    $dir = Join-Path $Out $stamp
    New-Item -ItemType Directory -Force $dir | Out-Null
    Write-Host "== ${stamp}: fetching islands"
    $log = pwsh -NoProfile -Command "& ./cluster/bio_status.ps1 -Fetch -Islands $($Islands -join ',')" *>&1 | ForEach-Object { "$_" }
    $log | Set-Content (Join-Path $dir 'status.txt')

    # per-island vegetation and year from the export lines
    $isl = @{}
    foreach ($l in $log) {
        if ($l -match 'island (\d): bioexport: island \d+, checkpoint step \d+ \(year ([\d.]+)\), (\d+) live') {
            $k = [int]$Matches[1]
            if (-not $isl.ContainsKey($k)) { $isl[$k] = @{} }
            $isl[$k].year = [double]$Matches[2]; $isl[$k].live = [long]$Matches[3]
        }
        if ($l -match 'island (\d): bioexport: .*cells: bare ([\d.]+)% mat ([\d.]+)% kelp ([\d.]+)% cover ([\d.]+)% shrub ([\d.]+)% tree ([\d.]+)%; mean cover (\d+)/255, tallest (\d+) m') {
            $k = [int]$Matches[1]
            if (-not $isl.ContainsKey($k)) { $isl[$k] = @{} }
            $isl[$k].mat = [double]$Matches[3]; $isl[$k].kelp = [double]$Matches[4]; $isl[$k].cover = [double]$Matches[5]
            $isl[$k].shrub = [double]$Matches[6]; $isl[$k].tree = [double]$Matches[7]
            $isl[$k].meancover = [int]$Matches[8]; $isl[$k].tallest = [int]$Matches[9]
        }
    }

    # screenshots: region r of the merged biomap = island r (parts merge in folder order)
    foreach ($k in ($isl.Keys | Sort-Object)) {
        $sp = Shot @('--region', "$k") (Join-Path $dir "island$k`_ground.png")
        if ($sp) { [void](Shot @('--spawn', "$($sp.x)", "$($sp.z)", '170', '--look', "$($sp.yaw)", '-34') (Join-Path $dir "island$k`_overview.png")) }
        $maps = "life/islands/$k/maps"
        if (Test-Path $maps) { Get-ChildItem $maps -Filter 'kind_*.png' | Sort-Object Name | Select-Object -Last 1 | Copy-Item -Destination (Join-Path $dir "island$k`_kindmap.png") }
    }
    python cluster/timeline_charts.py life/islands (Join-Path $dir 'charts.png') 2>&1 | Out-Null

    # rows + criterion
    $ok = 0; $extinct = $false
    foreach ($k in ($isl.Keys | Sort-Object)) {
        $v = $isl[$k]
        if (-not $v.ContainsKey('cover')) { continue }
        $land = $v.cover + $v.shrub + $v.tree
        $woody = $v.shrub + $v.tree
        if ($v.live -eq 0) { $extinct = $true }
        if ($land -ge $VegTarget -and $woody -ge $WoodyTarget) { $ok++ }
        "$stamp,$k,$($v.year),$($v.live),$($v.mat),$($v.kelp),$($v.cover),$($v.shrub),$($v.tree),$($v.meancover),$($v.tallest),$([math]::Round($land, 3)),$([math]::Round($woody, 3))" | Add-Content $csv
        Write-Host ("   island {0}: year {1,7:N1}  land veg {2,6:N2}%  woody {3,5:N2}%  tallest {4} m" -f $k, $v.year, $land, $woody, $v.tallest)
    }
    Write-Host "   playable islands: $ok of $($isl.Count) (need $IslandsNeeded)"
    if ($ok -ge $IslandsNeeded -and -not $extinct) {
        "playable at $stamp`: $ok islands meet land veg >= $VegTarget% and woody >= $WoodyTarget%" | Set-Content (Join-Path $Out 'PLAYABLE.txt')
        Copy-Item life/home.biomap (Join-Path $dir 'home.biomap')
        Write-Host "PLAYABLE reached at $stamp"
        exit 0
    }
    if (((Get-Date) - $start).TotalHours -ge $MaxHours) { Write-Host "time limit reached without the playable criterion"; exit 2 }
    Start-Sleep -Seconds ($IntervalMin * 60)
}

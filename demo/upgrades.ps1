# A tour of the life upgrades on a COPY of an evolved island (docs/upgrades-quickstart.md).
#
#   .\demo\upgrades.ps1                          the whole tour: about 12 min on an RTX 3060, 2.5 GB in runs\demo_upgrades
#   .\demo\upgrades.ps1 -Stages animals,export   the short one: about 4 min
#   .\demo\upgrades.ps1 -Fresh                   start again from the source checkpoint
#
# Stages
#   animals  copy the source island, switch every feature on (life --resume --enable all), run to the next
#            whole sim-year, then count creatures and species (biostat --species)
#   bank     seed bank of that island (life-seedbank), a new island founded from it (life-rebase), one sim-year
#   hub      two passes of the island hub over both islands (life-hub) with a sim-year after each:
#            the planet-wide air, emigrants both ways
#   export   biomap version 2 (crown shapes, the atmosphere for the sky) and the creature file (bioexport),
#            and the commands that open them in the 3D view
#
# The source run (-From) is only read. Every command is printed before it runs; the complete output of the
# tools is in <Out>\demo.log. Running the script again takes the islands one step further (-Fresh starts over).
param(
    [string]$From = 'milestones\playable_v1\island_1',   # a run folder with a checkpoint (ckpt_*.bin) and env.bin
    [int]$Island = 1,                                     # the island that run belongs to (life\regions.txt)
    [int]$NewIsland = 5,                                  # the island the seed bank founds anew
    [string]$Out = 'runs\demo_upgrades',
    [string]$Build = 'build',
    [string[]]$Stages = @('animals', 'bank', 'hub', 'export'),
    [string]$Enable = 'all',                              # features switched on in the copy (life --enable)
    [switch]$Fresh
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot)
$t0 = Get-Date

$known = 'animals', 'bank', 'hub', 'export'
$Stages = @($Stages | ForEach-Object { "$_" -split ',' } | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
foreach ($s in $Stages) { if ($known -notcontains $s) { throw "unknown stage '$s' (stages: $($known -join ', '))" } }

$G = Join-Path $Build 'gme_genesis.exe'
if (-not (Test-Path $G)) { throw "$G not found: build first (.\build.bat)" }
foreach ($f in 'genesis\home.planet', 'life\regions.txt') { if (-not (Test-Path $f)) { throw "$f is missing (run from a complete checkout)" } }
if ($Enable -match 'all|spectral' -and -not (Test-Path 'life\spectrum.bin')) {
    throw "light in bands needs life\spectrum.bin: .\$Build\gme.exe --export-spectrum life/spectrum.bin"
}
if ($Island -eq $NewIsland) { throw '-NewIsland must be another island than -Island' }

$A = Join-Path $Out "island_$Island"
$B = Join-Path $Out "island_$NewIsland"
$H = Join-Path $Out 'hub'
$bank = Join-Path $Out 'seedbank.bin'
$log = Join-Path $Out 'demo.log'

if ($Fresh -and (Test-Path $Out)) {
    if (-not (Test-Path $log)) { throw "$Out was not made by this script (no demo.log in it): not deleted" }
    Remove-Item -Recurse -Force $Out
}
New-Item -ItemType Directory -Force $Out | Out-Null
$logAbs = (Resolve-Path $Out).Path + '\demo.log'
[System.IO.File]::AppendAllText($logAbs, "`n==== demo/upgrades.ps1 $(Get-Date -Format s), stages $($Stages -join ',') ====`n")

function Say([string]$text) { Write-Host ''; Write-Host "== $text" -ForegroundColor Green }
function Fmt([double]$v, [string]$f = '0.0') { $v.ToString($f, [cultureinfo]::InvariantCulture) }   # a decimal point in every locale

# Runs gme_genesis, appends its complete output to demo.log, prints the lines matching -Show as they come
# (of the islands' periodic reports only every -Every-th), and returns all lines.
function Invoke-Genesis {
    param([string[]]$GArgs, [string]$Show = '.', [int]$Every = 1)
    Write-Host "> gme_genesis $($GArgs -join ' ')" -ForegroundColor Cyan
    [System.IO.File]::AppendAllText($logAbs, "`n> gme_genesis $($GArgs -join ' ')`n")
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'      # the tools write notes to stderr; only the exit code decides
    $reports = 0
    $shown = $false
    $global:LASTEXITCODE = -1                # stays -1 when the tool cannot be started at all
    $lines = & $G @GArgs 2>&1 | ForEach-Object {
        $line = "$_"
        [System.IO.File]::AppendAllText($logAbs, $line + "`n")
        if ($line -match '^isl\d+ y=') { $reports++; $shown = (($reports - 1) % $Every) -eq 0 }
        $periodic = $line -match '^isl\d+ y=|^\s+(mol/d|atm |genes |think |animals:)'
        if ($line -match $Show -and (-not $periodic -or $shown)) { Write-Host "  $line" }
        $line
    }
    $rc = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($rc -ne 0) {
        $lines | Select-Object -Last 12 | ForEach-Object { Write-Host "  $_" }
        throw "gme_genesis $($GArgs[0]) failed with exit code $rc (complete output: $log)"
    }
    return , @($lines)
}

function Get-NewestCkpt([string]$dir) {
    Get-ChildItem $dir -Filter 'ckpt_*.bin' -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
}
function Get-Day([string]$dir) {            # sim-day of the newest checkpoint (24 steps a day)
    $c = Get-NewestCkpt $dir
    if (-not $c) { return -1 }
    [int64][math]::Floor([int64]($c.BaseName -replace '^ckpt_', '') / 24)
}
function Next-Year([string]$dir) { ([math]::Floor((Get-Day $dir) / 365) + 1) * 365 }   # day of the next whole sim-year
function Remove-OlderCkpts([string]$dir) {  # the demo's own run folders keep only their newest checkpoint
    $all = @(Get-ChildItem $dir -Filter 'ckpt_*.bin' | Sort-Object Name)
    if ($all.Count -gt 1) { $all | Select-Object -First ($all.Count - 1) | Remove-Item -Force }
}
# One island to sim-day $day; prints every 8th report and returns the last "animals:" line
function Run-Island([int]$k, [string]$dir, [int64]$day, [string[]]$more = @()) {
    $cmd = @('life', '--island', "$k", '--run', $dir, '--resume', '--days', "$day", '--checkpoint-min', '0', '--maps-every', '0', '--duty', '1') + $more
    $res = Invoke-Genesis $cmd -Show '^life: (features|INTERVENTION|resumed|hub at|years reached|light in)|^isl\d+ y=|^\s+animals:' -Every 8
    Remove-OlderCkpts $dir
    ($res | Where-Object { $_ -match '^\s+animals:' } | Select-Object -Last 1)
}
$summary = [System.Collections.Generic.List[string]]::new()

# ---------------------------------------------------------------------------------------------------------
if ($Stages -contains 'animals') {
    Say "animals, sex, learning, light in bands, crown shapes: island $Island with the features switched on"
    if (-not (Get-NewestCkpt $A)) {
        $src = Get-NewestCkpt $From
        if (-not $src -or -not (Test-Path (Join-Path $From 'env.bin'))) {
            throw "$From has no checkpoint (ckpt_*.bin) and env.bin: pass -From <run folder of an evolved island> -Island <its number>"
        }
        New-Item -ItemType Directory -Force $A | Out-Null
        Write-Host "copying $($src.Name) ($([math]::Round($src.Length / 1MB)) MB) and env.bin from $From (the source is only read)"
        Copy-Item $src.FullName $A
        Copy-Item (Join-Path $From 'env.bin') $A
    } else {
        Write-Host "continuing $A (use -Fresh to start again from $From)"
    }
    $last = Run-Island $Island $A (Next-Year $A) @('--enable', $Enable)
    $st = Invoke-Genesis @('biostat', '--run', $A, '--species') -Show '^isl\d+ y=|^\s+(animals:|features|species:)'
    $summary.Add("island $Island, the evolved one, at the end of the animals stage (features $Enable): " + (($st | Where-Object { $_ -match '^isl\d+ y=' } | Select-Object -Last 1) -replace '\s+C\[.*$', ''))
    if ($last) { $summary.Add('  ' + $last.Trim()) }
    $sp = $st | Where-Object { $_ -match '^\s+species:' } | Select-Object -Last 1
    if ($sp) { $summary.Add('  ' + $sp.Trim()) }
}

# ---------------------------------------------------------------------------------------------------------
if ($Stages -contains 'bank') {
    Say "seed bank: the best genomes of island $Island, and island $NewIsland founded from them"
    if (-not (Get-NewestCkpt $A)) { throw "the bank stage needs the animals stage first ($A has no checkpoint)" }
    if (-not (Get-NewestCkpt $B)) {
        Invoke-Genesis @('life-seedbank', '--runs', $A, '--out', $bank) | Out-Null
        Invoke-Genesis @('life-rebase', '--bank', $bank, '--island', "$NewIsland", '--run', $B, '--cap', '400000', '--maps-every', '0') `
            -Show '^life: (island|founded|features|new run)|^isl\d+ y=' | Out-Null
    } else {
        Write-Host "continuing $B (use -Fresh to found it again)"
    }
    $last = Run-Island $NewIsland $B (Next-Year $B) @('--cap', '400000')
    $st = Invoke-Genesis @('biostat', '--run', $B, '--species') -Show '^isl\d+ y=|^\s+(atm |animals:|species:)'
    $summary.Add("island $NewIsland, founded from the seed bank, at the end of the bank stage: " + (($st | Where-Object { $_ -match '^isl\d+ y=' } | Select-Object -Last 1) -replace '\s+C\[.*$', ''))
    $sp = $st | Where-Object { $_ -match '^\s+species:' } | Select-Object -Last 1
    if ($sp) { $summary.Add('  ' + $sp.Trim()) }
}

# ---------------------------------------------------------------------------------------------------------
if ($Stages -contains 'hub') {
    Say "island hub: the planet-wide air and emigrants between island $Island and island $NewIsland"
    if (-not (Get-NewestCkpt $A) -or -not (Get-NewestCkpt $B)) { throw 'the hub stage needs the animals and bank stages first (two islands)' }
    foreach ($pass in 1, 2) {
        $hub = Invoke-Genesis @('life-hub', '--dir', $H, '--islands', "$Island,$NewIsland", '--runs', "$A,$B") -Show '^hub:'
        if ($pass -eq 2) { $hub | Where-Object { $_ -match 'planet atmosphere|titration' } | ForEach-Object { $summary.Add("hub, pass 2: " + ($_ -replace '^hub:\s*', '')) } }
        Run-Island $Island $A (Next-Year $A) | Out-Null
        Run-Island $NewIsland $B (Next-Year $B) @('--cap', '400000') | Out-Null
    }
    foreach ($d in $A, $B) {
        Get-Content (Join-Path $d 'stages.log') | Where-Object { $_ -match ' HUB ' } | Select-Object -Last 1 |
            ForEach-Object { $summary.Add("$(Split-Path $d -Leaf): " + $_) }
    }
}

# ---------------------------------------------------------------------------------------------------------
if ($Stages -contains 'export') {
    Say 'export for the 3D view: biomap version 2 (crown shapes, the sky from the air life has made) and the creature file'
    if (-not (Get-NewestCkpt $A)) { throw "the export stage needs the animals stage first ($A has no checkpoint)" }
    $parts = Join-Path $Out 'parts'
    if (Test-Path $parts) { Remove-Item -Recurse -Force $parts }
    New-Item -ItemType Directory -Force $parts | Out-Null
    $runsToExport = @(@{ k = $Island; dir = $A })
    if (Get-NewestCkpt $B) { $runsToExport += @{ k = $NewIsland; dir = $B } }
    $regions = @()                          # each island's square in world coordinates (metres)
    foreach ($r in $runsToExport) {
        $exp = Invoke-Genesis @('bioexport', '--run', $r.dir, '--out', (Join-Path $parts "biopart_$($r.k).bin")) -Show '^bioexport:'
        $hit = $exp | Select-String -Pattern 'region x (-?\d+) z (-?\d+), (\d+)x(\d+) cells of (\d+) m' | Select-Object -First 1
        if ($hit) {                         # (PowerShell names ignore case: no variable here may be called $g, $a, $b or $h)
            $v = @($hit.Matches[0].Groups | Select-Object -Skip 1 | ForEach-Object { [int]$_.Value })
            $regions += @{ k = $r.k; x0 = $v[0]; z0 = $v[1]; x1 = $v[0] + $v[2] * $v[4]; z1 = $v[1] + $v[3] * $v[4] }
        }
    }
    $bm = Join-Path $Out 'home.biomap'
    $fa = Join-Path $Out 'home.fauna'
    $mg = Invoke-Genesis @('bioexport', '--merge', $parts, '--out', $bm, '--fauna', $fa) -Show '^merge: (wrote|shapes|atmosphere)'
    $mg | Where-Object { $_ -match '^merge: (wrote|shapes|atmosphere)' } | ForEach-Object { $summary.Add(($_ -replace '^merge:\s*', 'export: ')) }

    # where the creatures are: on each island, the 64 m square of the creature file (GMEFAUN1, src/fauna.h) that
    # holds the most. A record: x, z in 1/16 m, depth and size in 1/16 m, flags (bit 0: swims)
    $spots = @()
    $by = [System.IO.File]::ReadAllBytes((Resolve-Path $fa).Path)
    $n = if ($by.Length -ge 128) { [int][BitConverter]::ToUInt32($by, 32) } else { 0 }
    if ($n -eq 0) { $summary.Add('creatures: none in the file') }
    foreach ($rg in $regions) {
        $boxes = @{}
        $inside = 0
        for ($i = 0; $i -lt $n; $i++) {
            $o = 128 + 16 * $i
            $x = [BitConverter]::ToInt32($by, $o) -shr 4
            $z = [BitConverter]::ToInt32($by, $o + 4) -shr 4
            if ($x -lt $rg.x0 -or $x -ge $rg.x1 -or $z -lt $rg.z0 -or $z -ge $rg.z1) { continue }
            $inside++
            $key = "$($x -shr 6),$($z -shr 6)"
            if (-not $boxes.ContainsKey($key)) { $boxes[$key] = [System.Collections.Generic.List[object]]::new() }
            $boxes[$key].Add(@($x, $z, [BitConverter]::ToUInt16($by, $o + 8), [BitConverter]::ToUInt16($by, $o + 10), $by[$o + 12]))
        }
        if (-not $inside) { $summary.Add("creatures of island $($rg.k): none"); continue }
        $best = $boxes.Values | Sort-Object Count -Descending | Select-Object -First 1
        $sx = [int](($best | ForEach-Object { $_[0] } | Measure-Object -Average).Average)
        $sz = [int](($best | ForEach-Object { $_[1] } | Measure-Object -Average).Average)
        $big = ($best | ForEach-Object { $_[3] } | Measure-Object -Maximum).Maximum / 16.0
        $swim = @($best | Where-Object { $_[4] -band 1 }).Count
        $deep = ($best | ForEach-Object { $_[2] } | Measure-Object -Maximum).Maximum / 16.0
        $spots += @{ k = $rg.k; at = "$sx $sz" }
        $where = if ($swim) { "$swim of them in water of 1 m and more (down to $(Fmt $deep) m below the water plane)" } else { 'all on land or on the shore' }
        $summary.Add("creatures of island $($rg.k): $inside in the file; the 64 m square with the most holds $($best.Count) around ($sx, $sz): largest $(Fmt $big '0.00') m, $where")
    }
}

# the gates the new island passed since it was founded (stages.log)
if ((Test-Path (Join-Path $B 'stages.log')) -and ($Stages -contains 'bank' -or $Stages -contains 'hub')) {
    $gates = @(Get-Content (Join-Path $B 'stages.log') | Where-Object { $_ -match ' GATE ' })
    if ($gates.Count) { $summary.Add("island ${NewIsland}'s gates since it was founded:") }
    $gates | ForEach-Object { $summary.Add('  ' + $_) }
}

# ---------------------------------------------------------------------------------------------------------
Say "done in $(Fmt ((Get-Date) - $t0).TotalMinutes) min; the complete output is in $log"
$summary | ForEach-Object { Write-Host "  $_" }
[System.IO.File]::AppendAllText($logAbs, "`n---- summary ----`n" + ($summary -join "`n") + "`n")
if ($Stages -contains 'export') {
    Write-Host ''
    Write-Host 'Look at it (the 3D view needs the engine build and a CUDA GPU):' -ForegroundColor Green
    $view = ".\$Build\gme.exe --biomap $Out\home.biomap --fauna $Out\home.fauna"
    Write-Host "  $view      # island $Island (region 0): evolved crown shapes, the sky of the air life has made"
    if ($regions.Count -gt 1) { Write-Host "  $view --region 1      # island $NewIsland, founded from the seed bank" }
    foreach ($sp in $spots) { Write-Host "  $view --spawn $($sp.at) 6 --look 35 -30      # where island $($sp.k) has the most creatures" }
    Write-Host "  .\$Build\gme.exe --biomap-test2 --fauna-test      # built-in patterns: every crown shape, a test herd"
}

# A tour of the life upgrades on a COPY of an evolved island (docs/upgrades-quickstart.md).
#
#   .\demo\upgrades.ps1                          the whole tour: about 9 min on an RTX 3060; runs\demo_upgrades
#                                                holds 1.1 GB when it is done, 2 GB at its peak
#   .\demo\upgrades.ps1 -Stages animals,export   the short one: about 3 min
#   .\demo\upgrades.ps1 -Fresh                   start the whole tour again from the source checkpoint
#   .\demo\upgrades.ps1 -From <run folder> -Island <K> [-NewIsland <another island>]   another evolved island
#
# Stages
#   animals  copy the source island, switch every feature on (life --resume --enable all), run to the next
#            whole sim-year, then count creatures and species (biostat --species)
#   bank     seed bank of that island (life-seedbank), a new island founded from it (life-rebase), one sim-year
#   hub      two passes of the island hub over both islands (life-hub) with a sim-year after each:
#            the planet-wide air, emigrants between the islands
#   export   biomap version 2 (crown shapes, the atmosphere for the sky) and the creature file (bioexport),
#            and the commands that open them in the 3D view
#
# The source run (-From) is only read. The tour works only in a folder it made itself (it leaves the file
# .upgrades-tour there) and refuses any other folder that is not empty. Every command is printed before it
# runs; the complete output of the tools is in <Out>\demo.log. Running the script again takes the islands one
# step further.
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
$root = Split-Path $PSScriptRoot
# The defaults are meant against the repository. A path the caller passed is meant against the caller's folder.
if ((Get-Location).ProviderPath -ne $root) {
    foreach ($name in 'From', 'Out', 'Build') {
        if ($PSBoundParameters.ContainsKey($name)) {
            Set-Variable $name ($ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PSBoundParameters[$name]))
        }
    }
}

function Say([string]$text) { Write-Host ''; Write-Host "== $text" -ForegroundColor Green }
function Fmt([double]$v, [string]$f = '0.0') { $v.ToString($f, [cultureinfo]::InvariantCulture) }   # a decimal point in every locale
function Full([string]$p) {                 # absolute path of a folder that need not exist (relative: against the repository)
    if ([System.IO.Path]::IsPathRooted($p)) { [System.IO.Path]::GetFullPath($p) } else { [System.IO.Path]::GetFullPath((Join-Path $root $p)) }
}
function Quote([string]$s) { if ($s -match '[\s"]') { '"' + $s + '"' } else { $s } }                 # an argument as one would type it
function Show-Exe([string]$exe) {           # an exe as one would start it from a PowerShell prompt
    $p = if ([System.IO.Path]::IsPathRooted($exe) -or $exe -match '^\.{1,2}[\\/]') { $exe } else { '.\' + $exe }
    if ($p -match '\s') { '& "' + $p + '"' } else { $p }
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
function Remove-OlderCkpts([string]$dir) {  # the tour's own run folders keep only their newest checkpoint
    $all = @(Get-ChildItem $dir -Filter 'ckpt_*.bin' | Sort-Object Name)
    if ($all.Count -gt 1) { $all | Select-Object -First ($all.Count - 1) | Remove-Item -Force }
}
# Runs gme_genesis, appends its complete output to demo.log, prints the lines matching -Show as they come
# (of the islands' periodic reports only every -Every-th), and returns all lines.
function Invoke-Genesis {
    param([string[]]$GArgs, [string]$Show = '.', [int]$Every = 1)
    $typed = 'gme_genesis ' + (($GArgs | ForEach-Object { Quote $_ }) -join ' ')
    Write-Host "> $typed" -ForegroundColor Cyan
    [System.IO.File]::AppendAllText($logAbs, "`n> $typed`n")
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'      # the tools write notes to stderr; only the exit code decides
    $reports = 0
    $shown = $false
    $global:LASTEXITCODE = -1                # stays -1 when the tool cannot be started at all
    $lines = & $Tool @GArgs 2>&1 | ForEach-Object {
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
# One island to sim-day $day; prints every 8th report and returns the last "animals:" line
function Run-Island([int]$k, [string]$dir, [int64]$day, [string[]]$more = @()) {
    $cmd = @('life', '--island', "$k", '--run', $dir, '--resume', '--days', "$day", '--checkpoint-min', '0', '--maps-every', '0', '--duty', '1') + $more
    $res = Invoke-Genesis $cmd -Show '^life: (features|INTERVENTION|resumed|hub at|years reached|light in)|^isl\d+ y=|^\s+animals:' -Every 8
    Remove-OlderCkpts $dir
    ($res | Where-Object { $_ -match '^\s+animals:' } | Select-Object -Last 1)
}

Push-Location $root
try {
    $t0 = Get-Date
    $known = 'animals', 'bank', 'hub', 'export'
    $Stages = @($Stages | ForEach-Object { "$_" -split ',' } | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
    foreach ($s in $Stages) { if ($known -notcontains $s) { throw "unknown stage '$s' (stages: $($known -join ', '))" } }

    # (PowerShell names ignore case: the script keeps clear of one-letter variables)
    $Tool = Join-Path $Build 'gme_genesis.exe'
    $Engine = Show-Exe (Join-Path $Build 'gme.exe')
    if (-not (Test-Path $Tool)) { throw "$Tool not found: build first (.\build.bat)" }
    foreach ($f in 'genesis\home.planet', 'life\regions.txt') { if (-not (Test-Path $f)) { throw "$f is missing (run from a complete checkout)" } }
    if ($Enable -match 'all|spectral' -and -not (Test-Path 'life\spectrum.bin')) {
        throw "light in bands needs life\spectrum.bin: $Engine --export-spectrum life/spectrum.bin"
    }
    if ($Island -eq $NewIsland) { throw '-NewIsland must be another island than -Island (for example -Island 5 -NewIsland 4)' }

    $RunA = Join-Path $Out "island_$Island"        # the copy of the evolved island
    $RunB = Join-Path $Out "island_$NewIsland"     # the island founded from the seed bank
    $HubDir = Join-Path $Out 'hub'
    $bank = Join-Path $Out 'seedbank.bin'
    $log = Join-Path $Out 'demo.log'
    $marker = Join-Path $Out '.upgrades-tour'

    # ---- nothing is created, continued or deleted before these hold ----
    $fromFull = (Full $From).TrimEnd('\')
    $outFull = (Full $Out).TrimEnd('\')
    if ($fromFull -eq $outFull -or $fromFull.StartsWith($outFull + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "-From ($From) lies inside -Out ($Out): the tour works on a copy, choose another -Out"
    }
    if ((Test-Path $Out) -and @(Get-ChildItem -Force $Out).Count -gt 0 -and -not (Test-Path $marker)) {
        throw "$Out exists, is not empty and was not made by this script (no .upgrades-tour in it): choose another -Out"
    }
    if ($Fresh) {
        if ($Stages -notcontains 'animals') { throw '-Fresh starts the whole tour again from the source checkpoint: it needs the animals stage (leave -Stages out, or include animals)' }
        if (-not (Get-NewestCkpt $From) -or -not (Test-Path (Join-Path $From 'env.bin'))) {
            throw "$From has no checkpoint (ckpt_*.bin) and env.bin: nothing deleted"
        }
        if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
    }
    New-Item -ItemType Directory -Force $Out | Out-Null
    if (-not (Test-Path $marker)) {
        Set-Content -Path $marker -Value "made by demo/upgrades.ps1 on $(Get-Date -Format s) from $From (island $Island)"
    }
    $logAbs = (Resolve-Path $Out).Path + '\demo.log'
    [System.IO.File]::AppendAllText($logAbs, "`n==== demo/upgrades.ps1 $(Get-Date -Format s), stages $($Stages -join ',') ====`n")
    $summary = [System.Collections.Generic.List[string]]::new()

    # -----------------------------------------------------------------------------------------------------
    if ($Stages -contains 'animals') {
        Say "animals, sex, learning, light in bands, crown shapes: island $Island with the features switched on"
        if (-not (Get-NewestCkpt $RunA)) {
            $src = Get-NewestCkpt $From
            if (-not $src -or -not (Test-Path (Join-Path $From 'env.bin'))) {
                throw "$From has no checkpoint (ckpt_*.bin) and env.bin: pass -From <run folder of an evolved island> -Island <its number>"
            }
            New-Item -ItemType Directory -Force $RunA | Out-Null
            Write-Host "copying $($src.Name) ($([math]::Round($src.Length / 1MB)) MB) and env.bin from $From (the source is only read)"
            Copy-Item $src.FullName $RunA
            Copy-Item (Join-Path $From 'env.bin') $RunA
        } else {
            Write-Host "continuing $RunA (-Fresh starts the whole tour again from $From)"
        }
        $last = Run-Island $Island $RunA (Next-Year $RunA) @('--enable', $Enable)
        $st = Invoke-Genesis @('biostat', '--run', $RunA, '--species') -Show '^isl\d+ y=|^\s+(animals:|features|species:)'
        $summary.Add("island $Island, the evolved one, at the end of the animals stage (features $Enable): " + (($st | Where-Object { $_ -match '^isl\d+ y=' } | Select-Object -Last 1) -replace '\s+C\[.*$', ''))
        if ($last) { $summary.Add('  ' + $last.Trim()) }
        $sp = $st | Where-Object { $_ -match '^\s+species:' } | Select-Object -Last 1
        if ($sp) { $summary.Add('  ' + $sp.Trim()) }
    }

    # -----------------------------------------------------------------------------------------------------
    if ($Stages -contains 'bank') {
        Say "seed bank: the best genomes of island $Island, and island $NewIsland founded from them"
        if (-not (Get-NewestCkpt $RunA)) { throw "the bank stage needs the animals stage first ($RunA has no checkpoint)" }
        if (-not (Get-NewestCkpt $RunB)) {
            Invoke-Genesis @('life-seedbank', '--runs', $RunA, '--out', $bank) | Out-Null
            Invoke-Genesis @('life-rebase', '--bank', $bank, '--island', "$NewIsland", '--run', $RunB, '--cap', '400000', '--maps-every', '0') `
                -Show '^life: (island|founded|features|new run)|^isl\d+ y=' | Out-Null
        } else {
            Write-Host "continuing $RunB (-Fresh starts the whole tour again; to found only this island again, delete $RunB)"
        }
        $last = Run-Island $NewIsland $RunB (Next-Year $RunB) @('--cap', '400000')
        $st = Invoke-Genesis @('biostat', '--run', $RunB, '--species') -Show '^isl\d+ y=|^\s+(atm |animals:|species:)'
        $summary.Add("island $NewIsland, founded from the seed bank, at the end of the bank stage: " + (($st | Where-Object { $_ -match '^isl\d+ y=' } | Select-Object -Last 1) -replace '\s+C\[.*$', ''))
        $sp = $st | Where-Object { $_ -match '^\s+species:' } | Select-Object -Last 1
        if ($sp) { $summary.Add('  ' + $sp.Trim()) }
    }

    # -----------------------------------------------------------------------------------------------------
    if ($Stages -contains 'hub') {
        Say "island hub: the planet-wide air and emigrants between island $Island and island $NewIsland"
        if (-not (Get-NewestCkpt $RunA) -or -not (Get-NewestCkpt $RunB)) { throw 'the hub stage needs the animals and bank stages first (two islands)' }
        foreach ($pass in 1, 2) {
            $hub = Invoke-Genesis @('life-hub', '--dir', $HubDir, '--islands', "$Island,$NewIsland", '--runs', "$RunA,$RunB") -Show '^hub:'
            if ($pass -eq 2) { $hub | Where-Object { $_ -match 'planet atmosphere|titration' } | ForEach-Object { $summary.Add("hub, pass 2: " + ($_ -replace '^hub:\s*', '')) } }
            Run-Island $Island $RunA (Next-Year $RunA) | Out-Null
            Run-Island $NewIsland $RunB (Next-Year $RunB) @('--cap', '400000') | Out-Null
        }
        foreach ($d in $RunA, $RunB) {
            Get-Content (Join-Path $d 'stages.log') | Where-Object { $_ -match ' HUB ' } | Select-Object -Last 1 |
                ForEach-Object { $summary.Add("$(Split-Path $d -Leaf): " + $_) }
        }
    }

    # -----------------------------------------------------------------------------------------------------
    $regions = @()                              # each exported island's square in world coordinates (metres)
    $spots = @()                                # where each island has the most creatures
    $order = @()                                # the islands in the order of the merged biomap's regions
    if ($Stages -contains 'export') {
        Say 'export for the 3D view: biomap version 2 (crown shapes, the sky from the air life has made) and the creature file'
        if (-not (Get-NewestCkpt $RunA)) { throw "the export stage needs the animals stage first ($RunA has no checkpoint)" }
        $parts = Join-Path $Out 'parts'
        if (Test-Path $parts) { Remove-Item -Recurse -Force $parts }
        New-Item -ItemType Directory -Force $parts | Out-Null
        $runsToExport = @(@{ k = $Island; dir = $RunA })
        if (Get-NewestCkpt $RunB) { $runsToExport += @{ k = $NewIsland; dir = $RunB } }
        foreach ($r in $runsToExport) {
            $exp = Invoke-Genesis @('bioexport', '--run', $r.dir, '--out', (Join-Path $parts "biopart_$($r.k).bin")) -Show '^bioexport:'
            $hit = $exp | Select-String -Pattern 'region x (-?\d+) z (-?\d+), (\d+)x(\d+) cells of (\d+) m' | Select-Object -First 1
            if ($hit) {
                $v = @($hit.Matches[0].Groups | Select-Object -Skip 1 | ForEach-Object { [int]$_.Value })
                $regions += @{ k = $r.k; x0 = $v[0]; z0 = $v[1]; x1 = $v[0] + $v[2] * $v[4]; z1 = $v[1] + $v[3] * $v[4] }
            }
        }
        $bm = Join-Path $Out 'home.biomap'
        $fa = Join-Path $Out 'home.fauna'
        $mg = Invoke-Genesis @('bioexport', '--merge', $parts, '--out', $bm, '--fauna', $fa) -Show '^merge: (wrote|shapes|atmosphere)'
        $mg | Where-Object { $_ -match '^merge: (wrote|shapes|atmosphere)' } | ForEach-Object { $summary.Add(($_ -replace '^merge:\s*', 'export: ')) }
        # the merge orders the regions by part file name, not by role: read the order from its own lines
        $order = @($mg | ForEach-Object { if ($_ -match 'biopart_(\d+)\.bin: \d+ region') { [int]$Matches[1] } })

        # where the creatures are: on each island, the 64 m square of the creature file (GMEFAUN1, src/fauna.h)
        # that holds the most. A record: x, z in 1/16 m, depth and size in 1/16 m, flags (bit 0: in water)
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
            # the fullest square; among equally full ones the one with the smallest x, then z (the same pick in every run)
            $best = ($boxes.GetEnumerator() | Sort-Object @{ Expression = { $_.Value.Count }; Descending = $true },
                                                          @{ Expression = { [int]($_.Key -split ',')[0] } },
                                                          @{ Expression = { [int]($_.Key -split ',')[1] } } | Select-Object -First 1).Value
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
    if ((Test-Path (Join-Path $RunB 'stages.log')) -and ($Stages -contains 'bank' -or $Stages -contains 'hub')) {
        $gates = @(Get-Content (Join-Path $RunB 'stages.log') | Where-Object { $_ -match ' GATE ' })
        if ($gates.Count) { $summary.Add("island ${NewIsland}'s gates since it was founded:") }
        $gates | ForEach-Object { $summary.Add('  ' + $_) }
    }

    # -----------------------------------------------------------------------------------------------------
    Say "done in $(Fmt ((Get-Date) - $t0).TotalMinutes) min; the complete output is in $log"
    $summary | ForEach-Object { Write-Host "  $_" }
    [System.IO.File]::AppendAllText($logAbs, "`n---- summary ----`n" + ($summary -join "`n") + "`n")
    if ($Stages -contains 'export') {
        Write-Host ''
        Write-Host 'Look at it (the 3D view needs the engine build and a CUDA GPU):' -ForegroundColor Green
        $view = "$Engine --biomap $(Quote (Join-Path $Out 'home.biomap')) --fauna $(Quote (Join-Path $Out 'home.fauna'))"
        $ri = [array]::IndexOf($order, $Island)
        $rn = [array]::IndexOf($order, $NewIsland)
        if ($ri -ge 0) { Write-Host "  $view$(if ($ri -gt 0) { " --region $ri" })      # island $Island (region $ri): evolved crown shapes, the sky of the air life has made" }
        if ($rn -ge 0) { Write-Host "  $view$(if ($rn -gt 0) { " --region $rn" })      # island $NewIsland (region $rn), founded from the seed bank" }
        foreach ($sp in $spots) { Write-Host "  $view --spawn $($sp.at) 6 --look 35 -30      # where island $($sp.k) has the most creatures" }
        Write-Host "  $Engine --biomap-test2 --fauna-test      # built-in patterns: every crown shape, a test herd"
    }
} finally {
    Pop-Location
}

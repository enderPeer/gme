# The island hub (docs/biosphere-spec.md 6.4): one planet-wide atmosphere and migrants
# between the islands. This script only carries files; the arithmetic is gme_genesis
# life-hub (integer, so every pass is reproducible from the files in life/hub).
#   pwsh cluster/bio_hub.ps1 [-Islands 0,1] [-IntervalMin 10] [-Once] [-Dir life/hub/<planet hash>] [-LabRun runs/bio_<hash>/island_5]
# Each pass:
#   1. make <Dir>/in hold what every island's outbox holds now (its gas uptake by realm, its
#      emigrants); an island founded anew is then taken in again from its new epoch
#   2. gme_genesis life-hub: move the planet's atmosphere by the islands' uptake per m^2 of
#      ocean and of land (weighted by the planet's ocean cover), route migrants to the next
#      island of the ring and to one more
#   3. copy <Dir>/out/<k> into island k's inbox (uploaded under another name, then renamed,
#      so an island never reads half a file)
# An island needs the hub feature (life --enable hub, or a run started with it) to write an
# outbox and read its inbox. Islands never wait: without the hub they simply run closed.
# -LabRun adds the PC's own island (island 5) from a local run folder.
# One hub folder serves one planet (the default is life/hub/<planet hash>): life-hub refuses
# files of another planet.
param(
    [object[]]$Islands = @(0, 1),
    [int]$IntervalMin = 10,
    [switch]$Once,
    [string]$Dir = '',
    [string]$LabRun = ''
)
$ErrorActionPreference = 'Stop'
# "-Islands 0,1" arrives as the one string "0,1" when the script is started as pwsh <script> (file mode)
$Islands = @($Islands | ForEach-Object { "$_" -split '[,\s]+' } | Where-Object { $_ -ne '' } | ForEach-Object { [int]$_ })
. (Join-Path $PSScriptRoot 'hosts.ps1')   # $GmeUser, $GmeRigA, $GmeRigB
Set-Location (Split-Path $PSScriptRoot)

$user = $GmeUser
$map = @{   # island -> rig (as cluster/bio_launch.ps1)
    0 = $GmeRigA; 1 = $GmeRigA
    2 = $GmeRigB; 3 = $GmeRigB; 4 = $GmeRigB
}
foreach ($k in $Islands) { if (-not $map.ContainsKey($k)) { throw "island $k has no rig slot (0-4; the PC lab island is -LabRun)" } }
$genesis = '.\build\gme_genesis.exe'
if (-not (Test-Path $genesis)) { throw "$genesis missing: run .\build.bat first" }
$hash = (& $genesis env --info | Select-String '^planet_hash ([0-9a-f]{16})').Matches[0].Groups[1].Value
if (-not $hash) { throw "cannot read the planet hash (genesis/home.planet)" }
if (-not $Dir) { $Dir = "life/hub/$hash" }        # one hub folder per planet
New-Item -ItemType Directory -Force (Join-Path $Dir 'in') | Out-Null
$all = @($Islands)
if ($LabRun) { $all += 5 }

while ($true) {
    $stamp = Get-Date -Format 'HH:mm:ss'
    # 1. outboxes -> in/ (finished files only: an island writes *.tmp and renames)
    foreach ($k in $Islands) {
        $h = $map[$k]
        $run = "gme/runs/bio_$hash/island_$k"
        $list = @(ssh -o BatchMode=yes "$user@$h" "ls $run/outbox 2>/dev/null | grep -E '\.(bin|txt)$'; echo __listed__")
        if ($list -notcontains '__listed__') { Write-Warning "island ${k}: $h not reachable; its files in $Dir/in stay as they are"; continue }
        $names = @($list | Where-Object { $_ -and $_ -ne '__listed__' })
        if (-not $names) { continue }      # nothing written yet: the island keeps its last files here
        foreach ($f in $names) {
            if (-not (Test-Path (Join-Path $Dir "in/$f"))) { scp -q "${user}@${h}:$run/outbox/$f" (Join-Path $Dir 'in/') 2>$null }
        }
        # in/ holds what the outbox holds now: an island founded anew must not hide behind its old files
        Get-ChildItem (Join-Path $Dir 'in') -File |
            Where-Object { ($_.Name -like "atm_${k}_*" -or $_.Name -like "mig_${k}_*") -and ($names -notcontains $_.Name) } | Remove-Item -Force
    }
    # 2. the hub's pass; the PC's own island is just another outbox and inbox, on this disk
    $hubArgs = @('life-hub', '--dir', $Dir, '--islands', ($all -join ','))
    if ($LabRun) {
        $ob = Join-Path $LabRun 'outbox'
        $lab = @(if (Test-Path $ob) { Get-ChildItem $ob -File | Where-Object { $_.Extension -in '.bin', '.txt' } })
        if ($lab) {
            $lab | Copy-Item -Destination (Join-Path $Dir 'in') -Force
            Get-ChildItem (Join-Path $Dir 'in') -File |
                Where-Object { ($_.Name -like 'atm_5_*' -or $_.Name -like 'mig_5_*') -and ($lab.Name -notcontains $_.Name) } | Remove-Item -Force
        }
    }
    & $genesis @hubArgs | ForEach-Object { Write-Host "$stamp $_" }
    if ($LASTEXITCODE -ne 0) { Write-Warning "life-hub failed (exit $LASTEXITCODE); trying again next pass" }
    # 3. out/<k> -> inboxes
    foreach ($k in $all) {
        $out = Join-Path $Dir "out/$k"
        if (-not (Test-Path $out)) { continue }
        $files = @(Get-ChildItem $out -File)
        if (-not $files) { continue }
        if ($k -eq 5) {
            New-Item -ItemType Directory -Force (Join-Path $LabRun 'inbox') | Out-Null
            $files | Move-Item -Destination (Join-Path $LabRun 'inbox') -Force
        } else {
            $h = $map[$k]
            $run = "gme/runs/bio_$hash/island_$k"
            ssh -o BatchMode=yes "$user@$h" "mkdir -p $run/inbox $run/inbox.part"
            scp -q ($files | ForEach-Object { $_.FullName }) "${user}@${h}:$run/inbox.part/"
            if ($LASTEXITCODE -eq 0) {
                ssh -o BatchMode=yes "$user@$h" "cd $run && for f in inbox.part/*; do [ -f `"`$f`" ] && mv -f `"`$f`" inbox/; done"
                $files | Remove-Item -Force
            } else { Write-Warning "island ${k}: upload to $h failed; the files stay in $out for the next pass" }
        }
        Write-Host "$stamp island ${k}: $($files.Count) file(s) delivered"
    }
    if ($Once) { break }
    Start-Sleep -Seconds ($IntervalMin * 60)
}

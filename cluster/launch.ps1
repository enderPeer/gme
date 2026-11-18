# Boot the universe on the rigs: tmux sessions running star-system seeds
# through the cosmos stage. By default the search never stops: every slot
# walks its own arithmetic sequence of seeds, so no two slots collide and
# status.ps1 can always report the best system found so far.
#
# The cosmos stage is latency-bound (it syncs with the GPU every step), so
# several processes share each card to keep it busy.
#   pwsh cluster/launch.ps1 [-Particles 12288] [-Years 6000] [-FirstSeed 25] [-Count 0] [-Deploy]
param(
    [int]$Particles = 12288,
    [double]$Years = 6000,
    [int]$FirstSeed = 25,
    [int]$Count = 0,          # seeds per slot; 0 = run forever
    [switch]$Deploy
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'hosts.ps1')   # $GmeUser, $GmeRigA, $GmeRigB
Set-Location (Split-Path $PSScriptRoot)

# one entry per process slot: host + CUDA device (driver order: rig A 0 = 24 GB card, 1 = 16 GB card)
$slots = @(
    @{ Host = $GmeRigA; Dev = 0 }, @{ Host = $GmeRigA; Dev = 0 }, @{ Host = $GmeRigA; Dev = 0 },
    @{ Host = $GmeRigA; Dev = 1 }, @{ Host = $GmeRigA; Dev = 1 },
    @{ Host = $GmeRigB; Dev = 0 }, @{ Host = $GmeRigB; Dev = 0 },
    @{ Host = $GmeRigB; Dev = 1 }, @{ Host = $GmeRigB; Dev = 1 },
    @{ Host = $GmeRigB; Dev = 2 }, @{ Host = $GmeRigB; Dev = 2 }
)
$stride = $slots.Count

if ($Deploy) {
    foreach ($h in ($slots.Host | Sort-Object -Unique)) {
        ssh "$GmeUser@$h" 'mkdir -p ~/gme'
        scp -q -r src CMakeLists.txt cluster "${GmeUser}@${h}:gme/"
        ssh "$GmeUser@$h" 'cd ~/gme && chmod +x cluster/*.sh && cmake -S . -B build -DGME_BUILD_ENGINE=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j >/dev/null 2>&1 && echo built on $(hostname)'
    }
}

foreach ($h in ($slots.Host | Sort-Object -Unique)) {   # stop earlier search sessions (only gme-search*: life islands run in bio-isl*)
    ssh "$GmeUser@$h" 'tmux ls 2>/dev/null | grep -o "^gme-search[0-9]*" | xargs -r -n1 tmux kill-session -t'
}
for ($i = 0; $i -lt $slots.Count; $i++) {
    $p = $slots[$i]
    $name = "gme-search$i"
    $run = "~/gme/cluster/genesis_seeds.sh $($p.Dev) $($FirstSeed + $i) $Count $Particles $Years $stride"
    ssh "$GmeUser@$($p.Host)" "tmux new-session -d -s $name '$run'"
    Write-Host "$($p.Host) GPU $($p.Dev): seeds $($FirstSeed + $i), +$stride ... (tmux $name)"
}

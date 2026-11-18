# Progress of the cosmos boot on the rigs, plus a habitability ranking of the
# finished systems. With -Fetch, copies the best run to ./genesis on this PC.
#   pwsh cluster/status.ps1 [-Fetch]
param([switch]$Fetch)
. (Join-Path $PSScriptRoot 'hosts.ps1')   # $GmeUser, $GmeRigA, $GmeRigB
Set-Location (Split-Path $PSScriptRoot)
$hosts = $GmeRigA, $GmeRigB
$best = $null
foreach ($h in $hosts) {
    Write-Host "=== $h ==="
    ssh "$GmeUser@$h" 'cd ~/gme 2>/dev/null || exit; tmux ls 2>/dev/null | grep gme- ; nvidia-smi --query-gpu=index,name,utilization.gpu --format=csv,noheader; for f in runs/seed_*.log; do [ -f "$f" ] && echo "$(basename $f .log): $(grep -E "cosmos: +[0-9]" $f | tail -1 | sed "s/cosmos: *//")"; done'
    $score = ssh "$GmeUser@$h" 'cd ~/gme 2>/dev/null && ls -d runs/seed_*/ 2>/dev/null | xargs -r ./build/gme_genesis score 2>/dev/null | tail -n +2'
    foreach ($line in $score) {
        Write-Host "  $line"
        if ($line -match '^(\S+)\s+([+-][\d.]+)') {
            $s = [double]$Matches[2]
            if (-not $best -or $s -gt $best.Score) { $best = @{ Host = $h; Dir = $Matches[1].TrimEnd('/'); Score = $s } }
        }
    }
}
if ($best) {
    Write-Host "`nbest so far: $($best.Dir) on $($best.Host) (score $($best.Score))"
    if ($Fetch) {
        if (Test-Path genesis) { Remove-Item -Recurse -Force genesis }
        scp -q -r "$GmeUser@$($best.Host):gme/$($best.Dir)" genesis
        Write-Host "copied to ./genesis - next: build\gme_genesis.exe chem --dir genesis --reactors 256, then finish"
    }
}

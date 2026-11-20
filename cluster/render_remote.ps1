# Render tests on the cluster's render node (rig B: headless Linux build of gme).
#
#   pwsh cluster/render_remote.ps1 -Sync
#       copy this tree (src, CMakeLists.txt, genesis/home.planet, life/home.biomap, life/home.fauna) to
#       ~/$Name on rig B and build gme there (skipped when nothing changed)
#   pwsh cluster/render_remote.ps1 -Run "<gme args>" -Out <local folder> [-Gpu K] [-Root <tree> -Name <workspace>]
#       sync if needed, run gme on a free GPU of rig B, copy the job's output folder back.
#       "{out}" in the args is that job folder on the rig, e.g.
#       -Run "--frames 600 --screenshot {out}/coast.png --perf" -Out status/render/final
#       gme's console output is saved as gme.log in the job folder.
#   -Root/-Name build another source tree in its own rig workspace, e.g. a reference build of
#   the last commit: git worktree add ..\gme-ref HEAD; then -Root ..\gme-ref -Name gme-render-ref
#
# Thermal guard (the rig's three GPUs are stacked): at most -MaxJobs jobs at once, each on the
# coolest free GPU of -Gpus (default 0,1: GPU 2 reached 87 C and throttled within 30 s under
# load, so it runs only when pinned with -Gpu 2) at or below -MaxTemp C; a job waits (up to
# 30 min) otherwise. While a job runs its GPU is sampled every 2 s: the job is stopped at
# -KillTemp C (exit 76), and thermal throttling is reported. -Exclusive holds every job slot
# for the whole job (no other job runs), for timing measurements. Needs the user-space CUDA toolchain in
# ~/gme/toolchain on the rig (pip wheels nvidia-cuda-nvcc/nvvm/crt/runtime/cccl 13.2).
param(
    [switch]$Sync,
    [string]$Run = '',
    [string]$Out = '',
    [int]$Gpu = -1,
    [int[]]$Gpus = @(0, 1),
    [int]$MaxTemp = 78,
    [int]$KillTemp = 83,
    [int]$MaxJobs = 2,
    [switch]$Exclusive,
    [string]$Root = '',
    [string]$Name = 'gme-render'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'hosts.ps1')   # $GmeUser, $GmeRigB
$OutAbs = if ($Out) { [System.IO.Path]::GetFullPath($Out) } else { '' }
Set-Location $(if ($Root) { $Root } else { Split-Path $PSScriptRoot })
$dst = "$GmeUser@$GmeRigB"

# PowerShell ends piped stdin lines with CRLF; strip the CRs on the rig before bash reads them
function Invoke-Rig([string]$script) { $script.Replace("`r", '') | ssh -o BatchMode=yes $dst "tr -d '\r' | bash -s" }

# ---- sync: stamp = hash of every input file; upload + build only when it changed ----
$inputs = @(Get-ChildItem src -Recurse -File | Sort-Object FullName) + @(Get-Item CMakeLists.txt)
foreach ($f in 'genesis/home.planet', 'life/home.biomap', 'life/home.fauna') { if (Test-Path $f) { $inputs += Get-Item $f } }
$sha = [System.Security.Cryptography.SHA256]::Create()
$all = ($inputs | ForEach-Object { "$($_.FullName.Substring((Get-Location).Path.Length)) $((Get-FileHash $_.FullName -Algorithm SHA256).Hash)" }) -join "`n"
$stamp = -join ($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($all)) | ForEach-Object { $_.ToString('x2') })
$remoteStamp = (Invoke-Rig "cat ~/$Name/.stamp 2>/dev/null || true") -join ''
if ($remoteStamp.Trim() -ne $stamp) {
    $stage = "/tmp/gme-render-stage-$([guid]::NewGuid().ToString('N').Substring(0, 12))"
    Invoke-Rig "mkdir -p $stage/genesis $stage/life" | Out-Null
    scp -q -r src CMakeLists.txt "${dst}:$stage/"
    if (Test-Path genesis/home.planet) { scp -q genesis/home.planet "${dst}:$stage/genesis/" }
    if (Test-Path life/home.biomap) { scp -q life/home.biomap "${dst}:$stage/life/" }
    if (Test-Path life/home.fauna) { scp -q life/home.fauna "${dst}:$stage/life/" }
    $build = @"
set -e
exec 7>/tmp/$Name.build.lock; flock 7
mkdir -p ~/$Name
if [ "`$(cat ~/$Name/.stamp 2>/dev/null)" != "$stamp" ]; then
  rm -rf ~/$Name/src && mv $stage/src ~/$Name/src
  mv -f $stage/CMakeLists.txt ~/$Name/
  mkdir -p ~/$Name/genesis ~/$Name/life
  [ -f $stage/genesis/home.planet ] && mv -f $stage/genesis/home.planet ~/$Name/genesis/
  [ -f $stage/life/home.biomap ] && mv -f $stage/life/home.biomap ~/$Name/life/
  if [ -f $stage/life/home.fauna ]; then mv -f $stage/life/home.fauna ~/$Name/life/; else rm -f ~/$Name/life/home.fauna; fi
  T=~/gme/toolchain/nvidia/cu13
  cmake -S ~/$Name -B ~/$Name/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=`$T/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 > ~/$Name/build.log 2>&1
  nice -n 10 cmake --build ~/$Name/build -j 12 --target gme >> ~/$Name/build.log 2>&1 || { echo "render: BUILD FAILED on rig B (tail of ~/$Name/build.log):"; grep -iE "error" ~/$Name/build.log | head -n 20; exit 2; }
  echo "$stamp" > ~/$Name/.stamp
  echo "render: built gme on rig B"
else
  echo "render: rig B already up to date"
fi
rm -rf $stage
"@
    Invoke-Rig $build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} elseif ($Sync) {
    Write-Host 'render: rig B already up to date'
}
if (-not $Run) { exit 0 }
if (-not $Out) { throw '-Out <local folder> is required with -Run' }

# ---- run on the coolest allowed, free GPU under the job-slot and per-GPU locks ----
$job = "$(Get-Date -Format 'yyyyMMdd-HHmmss')-$([guid]::NewGuid().ToString('N').Substring(0, 6))"
$args2 = $Run.Replace('{out}', "out/$job")
if ($args2.Contains("'")) { throw 'gme arguments must not contain single quotes' }
$allowed = if ($Gpu -ge 0) { @($Gpu) } else { $Gpus }
$runScript = @'
set -u
cd ~/__NAME__
mkdir -p out/__JOB__
export LD_LIBRARY_PATH=~/gme/toolchain/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
allowed=" __ALLOWED__ "
start=$(date +%s)
if [ __EXCL__ -eq 1 ]; then
  for s in $(seq 0 $((__MAXJOBS__ - 1))); do
    eval "exec $((20 + s))>/tmp/gme-render.slot$s.lock"
    flock -w 1800 $((20 + s)) || { echo "render: could not get exclusive use of the rig in 30 min"; exit 75; }
  done
  echo "render: exclusive (all __MAXJOBS__ job slots held)"
fi
while true; do
  got=0
  if [ __EXCL__ -eq 1 ]; then got=1; else
    for s in $(seq 0 $((__MAXJOBS__ - 1))); do
      exec 8>/tmp/gme-render.slot$s.lock
      if flock -n 8; then got=1; break; fi
      exec 8>&-
    done
  fi
  if [ $got -eq 1 ]; then
    for g in $(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits | sort -t, -k2 -n | awk -F', *' -v m=__MAXTEMP__ '$2+0 <= m {print $1}'); do
      case "$allowed" in *" $g "*) ;; *) continue ;; esac
      exec 9>/tmp/gme-render.gpu$g.lock
      if flock -n 9; then
        echo "render: job __JOB__ on rig B GPU $g ($(nvidia-smi -i $g --query-gpu=temperature.gpu --format=csv,noheader) C at start)"
        CUDA_VISIBLE_DEVICES=$g ./build/gme __ARGS__ > out/__JOB__/gme.log 2>&1 &
        gp=$!
        peak=0; throttled=0; stopped=0
        while kill -0 $gp 2>/dev/null; do
          line=$(nvidia-smi -i $g --query-gpu=temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null)
          t=$(echo "$line" | cut -d, -f1 | tr -d ' '); r=$(echo "$line" | cut -d, -f2 | tr -d ' ')
          case "$t" in ''|*[!0-9]*) t=0 ;; esac
          [ "$t" -gt "$peak" ] && peak=$t
          if [ -n "$r" ] && [ $(( r & 0x68 )) -ne 0 ]; then throttled=1; fi   # HW slowdown, SW or HW thermal
          if [ "$t" -ge __KILLTEMP__ ]; then kill $gp 2>/dev/null; stopped=1; echo "render: STOPPED at $t C (limit __KILLTEMP__ C)"; break; fi
          sleep 2
        done
        wait $gp; rc=$?
        [ $stopped -eq 1 ] && rc=76
        note=""; [ $throttled -eq 1 ] && note=", thermal throttling seen"
        echo "render: exit $rc, peak GPU temperature $peak C$note"
        exit $rc
      fi
      exec 9>&-
    done
    [ __EXCL__ -eq 1 ] || exec 8>&-
  fi
  if [ $(( $(date +%s) - start )) -gt 1800 ]; then echo "render: no allowed GPU (__ALLOWED__) free at or below __MAXTEMP__ C for 30 min"; exit 75; fi
  sleep 5
done
'@
$runScript = $runScript.Replace('__NAME__', $Name).Replace('__JOB__', $job).Replace('__ARGS__', $args2).Replace('__ALLOWED__', ($allowed -join ' ')).Replace('__EXCL__', $(if ($Exclusive) { '1' } else { '0' })).Replace('__MAXJOBS__', "$MaxJobs").Replace('__MAXTEMP__', "$MaxTemp").Replace('__KILLTEMP__', "$KillTemp")
Invoke-Rig $runScript
$rc = $LASTEXITCODE
New-Item -ItemType Directory -Force $OutAbs | Out-Null
scp -q -r "${dst}:$Name/out/$job/*" "$OutAbs/"
Write-Host "render: outputs of job $job copied to $Out (gme.log has the gme console output)"
exit $rc

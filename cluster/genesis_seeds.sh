#!/usr/bin/env bash
# Runs the cosmos stage for star-system seeds on one GPU of this rig.
# usage: genesis_seeds.sh <cuda_device> <first_seed> <count|0=forever> <particles> <years> [stride]
#   seeds visited: first, first+stride, first+2*stride, ...  (stride default 1)
# Results: ~/gme/runs/seed_<N>/ (system.txt, system.png, frames), logs next to them.
set -u
DEV=$1; FIRST=$2; COUNT=$3; PARTICLES=$4; YEARS=$5; STRIDE=${6:-1}
cd "$HOME/gme"
# NVRTC (the rigs need no CUDA toolkit): set GME_NVRTC_DIR to the folder holding
# libnvrtc.so, e.g. from "pip install nvidia-cuda-nvrtc" in a venv; otherwise the
# first such folder under ~/venvs is used.
if [ -z "${GME_NVRTC_DIR:-}" ]; then
  GME_NVRTC_DIR=$(ls -d "$HOME"/venvs/*/lib/python3*/site-packages/nvidia/cu1*/lib 2>/dev/null | head -n 1)
fi
[ -n "${GME_NVRTC_DIR:-}" ] && export LD_LIBRARY_PATH="$GME_NVRTC_DIR:${LD_LIBRARY_PATH:-}"
mkdir -p runs
s=$FIRST; n=0
while [ "$COUNT" -eq 0 ] || [ "$n" -lt "$COUNT" ]; do
  out="runs/seed_$s"
  if [ ! -f "$out/system.txt" ]; then             # resumable: skip finished seeds
    echo "$(date +%T) gpu $DEV seed $s start" >> runs/progress.log
    ./build/gme_genesis cosmos --device "$DEV" --seed "$s" --particles "$PARTICLES" \
        --years "$YEARS" --snapshots 24 --dir "$out" > "runs/seed_$s.log" 2>&1
    echo "$(date +%T) gpu $DEV seed $s exit $?" >> runs/progress.log
  fi
  s=$((s + STRIDE)); n=$((n + 1))
done

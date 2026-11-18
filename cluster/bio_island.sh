#!/usr/bin/env bash
# One life-simulation island on one GPU of this rig, resumed forever
# (docs/biosphere-spec.md 6.5). Started by cluster/bio_launch.ps1 in tmux bio-isl<island>.
# usage: bio_island.sh <cuda_device> <island> <planet_hash> <duty> <cap>
#   run folder ~/gme/runs/bio_<planet_hash>/island_<island>/ (env.bin is made by bio_launch.ps1)
#   life exit 0 (Ctrl+C, SIGTERM, tmux kill: it checkpoints first) ends the loop;
#   exit 3 (extinct): roll back to the newest checkpoint with a new salt, logged as
#     "INTERVENTION rollback <n>" in stages.log (after 3 extinctions in a row from the
#     same checkpoint, that checkpoint moves to rolled_back/ and the one before is used);
#   any other exit: wait 30 s and resume.
# The binary lives in ~/gme/build-bio, so rebuilding it never touches the star search.
set -u
DEV=$1; ISL=$2; HASH=$3; DUTY=$4; CAP=$5
cd "$HOME/gme"
# NVRTC (the rigs need no CUDA toolkit): set GME_NVRTC_DIR to the folder holding
# libnvrtc.so, e.g. from "pip install nvidia-cuda-nvrtc" in a venv; otherwise the
# first such folder under ~/venvs is used.
if [ -z "${GME_NVRTC_DIR:-}" ]; then
  GME_NVRTC_DIR=$(ls -d "$HOME"/venvs/*/lib/python3*/site-packages/nvidia/cu1*/lib 2>/dev/null | head -n 1)
fi
[ -n "${GME_NVRTC_DIR:-}" ] && export LD_LIBRARY_PATH="$GME_NVRTC_DIR:${LD_LIBRARY_PATH:-}"
RUN="runs/bio_$HASH/island_$ISL"
BIN=./build-bio/gme_genesis
mkdir -p "$RUN"
LOG="$RUN/life.log"
rb=$(grep -c "INTERVENTION rollback" "$RUN/stages.log" 2>/dev/null)
rb=${rb:-0}
streak=0        # extinctions in a row from the same newest checkpoint
streak_ck=""
extra=""        # --salt S after an extinction, kept until a newer checkpoint exists
salt_ck=""
newest_ck() { ls "$RUN"/ckpt_*.bin 2>/dev/null | sort | tail -n 1; }
while true; do
  echo "$(date '+%F %T') start: life --device $DEV --island $ISL --duty $DUTY --cap $CAP $extra" >> "$LOG"
  # one metrics row per 10 sim-days and maps every 20 sim-years keep the files at tens of MB per wall-day
  nice -n 5 $BIN life --device "$DEV" --island "$ISL" --run "$RUN" --resume --duty "$DUTY" --cap "$CAP" \
      --checkpoint-min 30 --metrics-every 10 --maps-every 20 $extra >> "$LOG" 2>&1
  rc=$?
  if [ -n "$extra" ] && [ "$(newest_ck)" != "$salt_ck" ]; then extra=""; fi   # the new salt is in a checkpoint now
  case $rc in
    0) echo "$(date '+%F %T') life stopped (exit 0)" >> "$LOG"
       break ;;
    3) rb=$((rb + 1))
       salt=$(( (ISL + 1) * 100003 + rb * 7919 ))
       newest=$(newest_ck)
       if [ "$newest" != "$streak_ck" ]; then streak=0; streak_ck="$newest"; fi
       streak=$((streak + 1))
       if [ "$streak" -ge 3 ] && [ -n "$newest" ]; then
         mkdir -p "$RUN/rolled_back" && mv "$newest" "$RUN/rolled_back/"
         echo "$(date '+%F %T') INTERVENTION skip $(basename "$newest"): extinct 3 times from it, moved to rolled_back/" >> "$RUN/stages.log"
         streak=0; streak_ck=""
       fi
       echo "$(date '+%F %T') INTERVENTION rollback $rb salt $salt (extinct)" >> "$RUN/stages.log"
       extra="--salt $salt"; salt_ck=$(newest_ck) ;;
    *) echo "$(date '+%F %T') life exited $rc, resuming in 30 s" >> "$LOG"
       sleep 30 ;;
  esac
done

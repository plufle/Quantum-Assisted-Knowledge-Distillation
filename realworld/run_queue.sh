#!/bin/bash
# Real-world RPS queue: teacher -> KD student -> PQK student, strictly sequential, halt on failure.
cd /e/projects/Project1 || exit 1
LOG=realworld/runs/queue.log
mkdir -p realworld/runs
run() {
  echo "[start $(date +%H:%M)] $*" >> "$LOG"
  if python realworld/train.py "$@" >> realworld/runs/train_stdout.log 2>&1; then
    echo "[done  $(date +%H:%M)] $*" >> "$LOG"
  else
    echo "[FAIL  $(date +%H:%M)] $* — queue halted" >> "$LOG"; exit 1
  fi
}
run --role teacher
run --role student --method kd
run --role student --method pqk_adaptive
echo "=== QUEUE COMPLETE $(date +%H:%M) ===" >> "$LOG"

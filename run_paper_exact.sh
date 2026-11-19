#!/usr/bin/env bash
# FPTNN with the paper's stated settings only (Qian et al., TGRS 2022):
#   DnCNN 16 layers x 64 filters (Fig. 12 optimum), Adam, constant lr 1e-3, batch 32,
#   SURE with a 1-probe forward-difference Monte-Carlo divergence (Soltanayev & Chun),
#   sigma per band from Eq. (24), linear B-spline framelet (1 level), (x, y) frontal slices.
# The paper does not state the training length; we use STEPS (default 100k) and log every 10k.
# One process per noise level, all on one GPU, each with its own results file.
# Usage:  CUDA_VISIBLE_DEVICES=1 bash run_paper_exact.sh          (5-D: Table III)
#         WITH4D=1 CUDA_VISIBLE_DEVICES=1 bash run_paper_exact.sh  (+ 4-D: Fig. 5 analogue)
# Progress: tail -n 2 results/logs/paper_*.log
set -uo pipefail
cd "$(dirname "$0")/experiments"
mkdir -p ../results/logs
L=../results/logs
STEPS=${STEPS:-100000}
if [[ "${QUICK:-0}" == "1" ]]; then STEPS=30; fi
PAPER="--depth 16 --width 64 --batch 32 --lr 1e-3 --lr_final 1e-3 --div_probes 1 --div_forward --sigma_mode mad_band --steps $STEPS"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
log() { echo "[$(date +%H:%M:%S)] $*"; }

pids=()
for s in -5 -2.5 0 2.5 5; do          # Table III input SNRs
  python run_benchmark.py --data synth5d --snrs $s --methods fptnn fptnn_mse --tag paper5d_s$s --save_snr 99 $PAPER \
    > $L/paper_5d_s$s.log 2>&1 & pids+=($!)
done
if [[ "${WITH4D:-0}" == "1" ]]; then
  for s in -5 -2.5 0 2.5 5 7.5 10; do  # Fig. 5 input SNRs (VSP -> open 4-D substitute)
    python run_benchmark.py --data hyper4d --snrs $s --methods fptnn --tag paper4d_s$s --save_snr 99 $PAPER \
      > $L/paper_4d_s$s.log 2>&1 & pids+=($!)
  done
fi
log "started ${#pids[@]} runs (logs: results/logs/paper_*.log)"
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
grep -l "Traceback" $L/paper_*.log && fail=1
log "finished (fail=$fail)"
grep -h -E "===|-> " $L/paper_*.log
exit $fail

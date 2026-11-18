#!/usr/bin/env bash
# Same experiments as run_all.sh, split into 5 lanes that run side by side on ONE GPU.
# Each lane writes its own results files, so lanes never clash; everything is resumable.
# Usage:  CUDA_VISIBLE_DEVICES=1 bash run_parallel.sh
#         QUICK=1 CUDA_VISIBLE_DEVICES=1 bash run_parallel.sh   (smoke test)
# Progress: tail -f results/logs/lane_*.log
set -uo pipefail
cd "$(dirname "$0")/experiments"
mkdir -p ../results/logs
L=../results/logs

DEPTH=${DEPTH:-17}; WIDTH=${WIDTH:-64}; STEPS=${STEPS:-10000}; BATCH=${BATCH:-128}; PROBES=${PROBES:-4}
GSTEPS=${GSTEPS:-5000}
SNRS="-5 -2.5 0 2.5 5 7.5 10"; TAG=main; ATAG=e4; PU=50
if [[ "${QUICK:-0}" == "1" ]]; then DEPTH=5; WIDTH=8; STEPS=30; GSTEPS=30; BATCH=16; SNRS="0"; TAG=quick; ATAG=quick; PU=2; fi
NET="--depth $DEPTH --width $WIDTH --steps $STEPS --batch $BATCH --div_probes $PROBES"
# keep CPU baselines from oversubscribing cores when 5 lanes run at once
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
log() { echo "[$(date +%H:%M:%S)] $*"; }

lane_a() {  # E1: 5-D sweep + F-tSVT
  log "A: E1 5-D sweep"
  python run_benchmark.py --data synth5d --snrs $SNRS --methods tnn sgk patchunet fptnn fptnn_mse --tag $TAG --save_snr 5 --pu_epochs $PU $NET
  log "A: E1 F-tSVT"
  python run_benchmark.py --data synth5d --snrs 0 5 --methods ftsvt --tag $TAG --save_snr 5
  log "A: done"
}
lane_b() {  # E2: 4-D sweep, then E3 field data
  log "B: E2 4-D sweep"
  python run_benchmark.py --data hyper4d --snrs $SNRS --methods tnn sgk patchunet fptnn fptnn_mse --tag $TAG --save_snr 0 --pu_epochs $PU $NET
  log "B: E3 field"
  python run_field.py $NET
  log "B: done"
}
lane_c() {  # E4 ablations 4-D
  log "C: E4 ablations 4-D"
  python run_ablation.py --data hyper4d --snr 0 --tag $ATAG $NET --variants base haar cubic linear_L2 dct identity sig_global sig_oracle loss_mse slices_tx div_fwd1 holdout
  log "C: done"
}
lane_e() {  # E4 ablations 5-D
  log "E: E4 ablations 5-D"
  python run_ablation.py --data synth5d --snr 0 --tag $ATAG $NET --variants base haar dct identity sig_global sig_oracle loss_mse slices_tx div_fwd1 holdout
  log "E: done"
}
lane_d() {  # E5 theory + depth x width grid
  log "D: E5 theory"
  python e5_theory.py
  log "D: E4 grid"
  python run_ablation.py --data synth5d --snr 0 --grid --depths 8 12 17 20 --widths 16 32 64 --tag $ATAG --steps $GSTEPS --batch $BATCH --div_probes $PROBES
  log "D: done"
}

log "starting 5 lanes (logs: results/logs/lane_{a,b,c,d,e}.log)"
lane_a > $L/lane_a.log 2>&1 & pa=$!
lane_b > $L/lane_b.log 2>&1 & pb=$!
lane_c > $L/lane_c.log 2>&1 & pc=$!
lane_d > $L/lane_d.log 2>&1 & pd=$!
lane_e > $L/lane_e.log 2>&1 & pe=$!
fail=0
for p in $pa $pb $pc $pd $pe; do wait $p || fail=1; done
grep -l "Traceback" $L/lane_*.log && fail=1
log "figures + tables"
python make_figures.py --tag $TAG --abl_tag $ATAG > $L/figures.log 2>&1 || fail=1
log "finished (fail=$fail) -> results/"
exit $fail

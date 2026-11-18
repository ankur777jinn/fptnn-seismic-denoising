#!/usr/bin/env bash
# Full reproduction of FPTNN (Qian et al., IEEE TGRS 2022) on the open datasets.
# Uses CUDA automatically if available (FPTNN, PATCHUNET); TNN/SGK run on CPU.
# Defaults: SURE with a 4-probe central-difference divergence, 10k steps x batch 128.
# Estimated time: ~8-12 h on one modern GPU (>=12 GB); ~50 h on an RTX 3050 laptop.
# Resumable: finished runs are stored in results/*.json and skipped on restart.
# Usage:  bash run_all.sh            (full)
#         QUICK=1 bash run_all.sh    (smoke test, a few minutes)
set -euo pipefail
cd "$(dirname "$0")/experiments"
mkdir -p ../results/logs

DEPTH=${DEPTH:-17}; WIDTH=${WIDTH:-64}; STEPS=${STEPS:-10000}; BATCH=${BATCH:-128}; PROBES=${PROBES:-4}
GSTEPS=${GSTEPS:-5000}   # depth x width grid uses fewer steps
SNRS="-5 -2.5 0 2.5 5 7.5 10"; TAG=main; ATAG=e4; PU=50
if [[ "${QUICK:-0}" == "1" ]]; then DEPTH=5; WIDTH=8; STEPS=30; GSTEPS=30; BATCH=16; SNRS="0"; TAG=quick; ATAG=quick; PU=2; fi
NET="--depth $DEPTH --width $WIDTH --steps $STEPS --batch $BATCH --div_probes $PROBES"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "E5 theory checks";                       python e5_theory.py                     2>&1 | tee ../results/logs/e5.log
log "E1 5-D synthetic sweep";                 python run_benchmark.py --data synth5d --snrs $SNRS --methods tnn sgk patchunet fptnn fptnn_mse --tag $TAG --save_snr 5 --pu_epochs $PU $NET 2>&1 | tee ../results/logs/e1.log
log "E1 F-tSVT (0, 5 dB)";                    python run_benchmark.py --data synth5d --snrs 0 5 --methods ftsvt --tag $TAG --save_snr 5 2>&1 | tee -a ../results/logs/e1.log
log "E2 4-D curved-event sweep";              python run_benchmark.py --data hyper4d --snrs $SNRS --methods tnn sgk patchunet fptnn fptnn_mse --tag $TAG --save_snr 0 --pu_epochs $PU $NET 2>&1 | tee ../results/logs/e2.log
log "E3 field data";                          python run_field.py $NET                 2>&1 | tee ../results/logs/e3.log
log "E4 ablations (4-D, 0 dB)";               python run_ablation.py --data hyper4d --snr 0 --tag $ATAG $NET --variants base haar cubic linear_L2 dct identity sig_global sig_oracle loss_mse slices_tx div_fwd1 holdout 2>&1 | tee ../results/logs/e4a.log
# cubic / 2-level framelets are skipped on 5-D: 125x redundancy needs >8 GB RAM
log "E4 ablations (5-D, 0 dB)";               python run_ablation.py --data synth5d --snr 0 --tag $ATAG $NET --variants base haar dct identity sig_global sig_oracle loss_mse slices_tx div_fwd1 holdout 2>&1 | tee ../results/logs/e4b.log
log "E4 depth x width grid (5-D, 0 dB)";      python run_ablation.py --data synth5d --snr 0 --grid --depths 8 12 17 20 --widths 16 32 64 --tag $ATAG --steps $GSTEPS --batch $BATCH --div_probes $PROBES 2>&1 | tee ../results/logs/e4g.log
log "figures + tables";                       python make_figures.py --tag $TAG --abl_tag $ATAG
log "done -> results/"

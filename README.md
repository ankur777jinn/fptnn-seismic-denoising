# FPTNN seismic denoising: a reproduction

Course project by Ankur Dahiya and Saketh. We reproduce

> Qian et al., "Multidimensional Seismic Data Denoising Using Framelet-Based
> Order-p Tensor Deep Learning," *IEEE TGRS*, vol. 60, 2022,
> doi:10.1109/TGRS.2022.3197287.

There is no official code; everything here is written from the paper.

## Method (Algorithm 1)
1. Undecimated tight framelet transform (WᵀW = I) along modes 3..p.
2. Denoise every frontal slice with one shared DnCNN trained **without clean
   data** using Monte-Carlo SURE.
3. Inverse transform Wᵀ.

## Layout
- `fptnn/`: framelets, t-product, DnCNN + SURE, noise estimation, metrics,
  baselines (TNN, SGK, PATCHUNET port).
- `experiments/`: E1/E2 noise sweeps, E3 field data, E4 ablations, E5 theory
  checks, figures and tables.
- `data/`: open data from chenyk1990/reproducible_research (see `data/SOURCES.md`).
- `results/`: JSON results, figures, LaTeX tables.

## Run
```bash
pip install -r requirements.txt
QUICK=1 bash run_all.sh   # smoke test, ~10 min
bash run_all.sh           # full pipeline, sequential
CUDA_VISIBLE_DEVICES=0 bash run_parallel.sh   # same, 5 lanes on one GPU
```
Defaults: SURE with a 4-probe central-difference divergence, 10k steps x batch 128.

## Status (work in progress)
5-D synthetic data, 5 dB input, 17×64 DnCNN (paper Table III: SURE 20.41 dB, MSE 21.50 dB):

| Training | Output SNR |
|---|---|
| SURE, 5k steps × batch 32 | 8.13 dB |
| SURE, 20k steps × batch 32 | 12.46 dB |
| SURE, 20k steps × batch 128 | 13.97 dB |
| SURE, 20k steps × batch 128, true σ | 15.51 dB |
| Supervised MSE, 5k steps × batch 32 | 19.79 dB |
| TNN / SGK baselines | 14.07 / 15.47 dB |

The supervised network nearly matches the paper, but SURE training converges
slowly and has not reached the paper's SURE numbers yet. See `results/diag5_*.json`.

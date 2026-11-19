"""E1/E2: synthetic benchmarks -- FPTNN vs TNN, SGK, PATCHUNET over input SNRs.

usage: python run_benchmark.py --data synth5d --snrs -5 -2.5 0 2.5 5 \
          --methods tnn sgk patchunet fptnn fptnn_mse --tag e1
Results are appended to results/<tag>_<data>.json (resumable); denoised volumes at
--save_snr are stored in results/vol_<tag>_<data>_<method>.npy for the figures.
"""
import argparse
import os

import numpy as np

from common import RES, Timer, env_info, load_json, save_json
from fptnn.data import IMAGE_MODES, load
from fptnn.metrics import add_noise_snr, snr_db

CFG = {
    "synth5d": {"sgk": dict(psize=(6, 4, 4, 4, 4), step=(2, 2, 2, 2, 2), K=512),
                "pu": dict(psize=(8, 4, 4, 4, 4), step=(4, 2, 2, 2, 2))},
    "hyper4d": {"sgk": dict(psize=(6, 4, 4, 4), step=(2, 2, 2, 2), K=256),
                "pu": dict(psize=(8, 8, 8, 4), step=(2, 2, 2, 1))},
    "field3d": {"sgk": dict(psize=(6, 4, 4), step=(2, 2, 2), K=64),
                "pu": dict(psize=(16, 16, 8), step=(2, 2, 2))},
}
TNN_GRID = [0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
SGK_PERC = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]


def run_method(name, X, Y, sigma, data, a):
    im = IMAGE_MODES[data]
    info = {}
    if name == "tnn":
        from fptnn.baselines.tnn import oracle_grid, tnn_denoise
        with Timer() as t:
            out, c, curve = oracle_grid(tnn_denoise, Y, X, TNN_GRID, sigma=sigma, image_modes=im)
        info = {"c": c, "curve": curve, "time_s": t.s / len(TNN_GRID)}
    elif name == "ftsvt":
        from fptnn.baselines.tnn import ftsvt_denoise, oracle_grid
        with Timer() as t:
            out, c, curve = oracle_grid(ftsvt_denoise, Y, X, TNN_GRID, sigma=sigma, image_modes=im)
        info = {"c": c, "curve": curve, "time_s": t.s / len(TNN_GRID)}
    elif name == "sgk":
        from fptnn.baselines.sgk import sgk_denoise
        cfg = CFG[data]["sgk"]
        with Timer() as t:
            outs = sgk_denoise(Y, cfg["psize"], cfg["step"], K=cfg["K"], T=3, niter=10, perc=SGK_PERC)
        snrs = {k: snr_db(X, v) for k, v in outs.items()}
        best = max(snrs, key=snrs.get)
        out = outs[best]
        info = {"perc": best, "curve": snrs, "time_s": t.s}
    elif name == "patchunet":
        from fptnn.baselines.patchunet import patchunet_denoise
        cfg = CFG[data]["pu"]
        out, info = patchunet_denoise(Y, cfg["psize"], cfg["step"], epochs=a.pu_epochs, seed=a.seed)
    elif name.startswith("fptnn"):
        from fptnn.model import FPTNN
        loss = "mse" if name.endswith("_mse") else "sure"
        sig_mode = a.sigma_mode
        if name.endswith("_oracle"):
            sig_mode = "oracle"
        m = FPTNN(image_modes=im, kind=a.kind, ftype=a.ftype, levels=a.levels, bc="symmetric",
                  depth=a.depth, width=a.width, steps=a.steps, batch=a.batch, lr=a.lr, lr_final=a.lr_final,
                  loss=loss, sigma_mode=sig_mode, seed=a.seed, threads=a.threads,
                  log_every=max(a.steps // 10, 1),
                  div_probes=a.div_probes, div_central=not a.div_forward)
        out, info = m.fit_denoise(Y, X=X, sigma_true=sigma)
        info["time_s"] = info["total_time_s"]
    else:
        raise ValueError(name)
    return out, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="synth5d")
    p.add_argument("--snrs", type=float, nargs="+", default=[-5, -2.5, 0, 2.5, 5])
    p.add_argument("--methods", nargs="+", default=["tnn", "sgk", "patchunet", "fptnn", "fptnn_mse"])
    p.add_argument("--tag", default="bench")
    p.add_argument("--save_snr", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise_seed", type=int, default=2026)
    # FPTNN
    p.add_argument("--kind", default="framelet")
    p.add_argument("--ftype", default="linear")
    p.add_argument("--levels", type=int, default=1)
    p.add_argument("--depth", type=int, default=17)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_final", type=float, default=1e-4, help="cosine end LR; set = --lr for a constant LR (paper)")
    p.add_argument("--sigma_mode", default="mad_band")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--pu_epochs", type=int, default=50)
    p.add_argument("--div_probes", type=int, default=4)
    p.add_argument("--div_forward", action="store_true", help="1-sided divergence (paper/Soltanayev); default central")
    a = p.parse_args()

    X = load(a.data)
    fname = f"{a.tag}_{a.data}.json"
    res = load_json(fname) if os.path.exists(os.path.join(RES, fname)) else {"env": env_info(), "runs": []}
    res["args"] = vars(a)
    done = {(r["snr_in"], r["method"]) for r in res["runs"]}
    for s in a.snrs:
        Y, N, sigma = add_noise_snr(X, s, seed=a.noise_seed + int(10 * s))
        for mth in a.methods:
            if (s, mth) in done:
                continue
            print(f"=== {a.data}  input {s:+.1f} dB  {mth}", flush=True)
            out, info = run_method(mth, X, Y, sigma, a.data, a)
            r = {"snr_in": s, "method": mth, "snr_out": snr_db(X, out), "sigma": sigma,
                 "info": {k: v for k, v in info.items()}}
            print(f"    -> {r['snr_out']:.2f} dB  ({info.get('time_s', 0):.1f}s)", flush=True)
            res["runs"].append(r)
            save_json(res, fname)
            if abs(s - a.save_snr) < 1e-6:
                np.save(os.path.join(RES, f"vol_{a.tag}_{a.data}_{mth}.npy"), out.astype(np.float32))
                np.save(os.path.join(RES, f"vol_{a.tag}_{a.data}_noisy.npy"), Y.astype(np.float32))


if __name__ == "__main__":
    main()

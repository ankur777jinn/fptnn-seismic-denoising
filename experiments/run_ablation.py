"""E4: ablations of FPTNN on one dataset / noise level.

  transform : framelet (linear B-spline, 1 level) | Haar framelet | cubic framelet |
              2-level linear | DCT (orthonormal, Fourier-type) | identity (no transform)
  sigma     : Eq.(24) per band (paper) | global MAD propagated through W | oracle
  loss      : SURE (ground-truth free) | MSE with clean data (supervised bound)
  depth x width grid (Fig. 12 of the paper)
"""
import argparse
import os

from common import RES, env_info, load_json, save_json
from fptnn.data import IMAGE_MODES, load
from fptnn.metrics import add_noise_snr, snr_db

VARIANTS = {
    "base":         {},
    "haar":         {"ftype": "haar", "bc": "periodic"},
    "cubic":        {"ftype": "cubic"},
    "linear_L2":    {"levels": 2},
    "dct":          {"kind": "dct"},
    "identity":     {"kind": "identity"},
    "sig_global":   {"sigma_mode": "mad_global"},
    "sig_oracle":   {"sigma_mode": "oracle"},
    "loss_mse":     {"loss": "mse"},
    "slices_tx":    {"image_modes": "tx"},   # (time, x) frontal slices instead of (x, y)
    # SURE-training choices (default is now a 4-probe central-difference divergence)
    "div_fwd1":     {"div_probes": 1, "div_central": False},   # paper-style 1-probe forward diff
    "div_c4":       {"div_probes": 4, "div_central": True},    # explicit 4-probe central diff
    "holdout":      {"val_frac": 0.1},                         # hold-out SURE (gap diagnostic)
    "no_bn":        {"bn": False},                             # no BatchNorm
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="hyper4d")
    p.add_argument("--snr", type=float, default=0.0)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--grid", action="store_true", help="depth x width grid instead")
    p.add_argument("--depths", type=int, nargs="+", default=[8, 12, 17])
    p.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--depth", type=int, default=17)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--sigma_mode", default="mad_band")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--div_probes", type=int, default=4)
    p.add_argument("--div_forward", action="store_true", help="1-sided divergence (paper/Soltanayev); default central")
    p.add_argument("--tag", default="e4")
    a = p.parse_args()

    from fptnn.model import FPTNN
    X = load(a.data)
    Y, _, sigma = add_noise_snr(X, a.snr, seed=2026 + int(10 * a.snr))
    fname = f"{a.tag}_{'grid' if a.grid else 'ablation'}_{a.data}.json"
    res = load_json(fname) if os.path.exists(os.path.join(RES, fname)) else {"env": env_info(), "runs": {}}
    res["args"] = vars(a)
    base = dict(image_modes=IMAGE_MODES[a.data], kind="framelet", ftype="linear", levels=1,
                bc="symmetric", depth=a.depth, width=a.width, steps=a.steps, batch=a.batch,
                sigma_mode=a.sigma_mode, threads=a.threads, log_every=max(a.steps // 5, 1),
                div_probes=a.div_probes, div_central=not a.div_forward)
    jobs = []
    if a.grid:
        for d in a.depths:
            for w in a.widths:
                jobs.append((f"d{d}_w{w}", {"depth": d, "width": w}))
    else:
        jobs = [(v, VARIANTS[v]) for v in a.variants]
    for name, over in jobs:
        if name in res["runs"]:
            continue
        cfg = dict(base, **over)
        if cfg["image_modes"] == "tx":
            cfg["image_modes"] = (0, IMAGE_MODES[a.data][0])
        print(f"=== {a.data} {a.snr:+.1f} dB  {name}: {over}", flush=True)
        m = FPTNN(**cfg)
        out, info = m.fit_denoise(Y, X=X, sigma_true=sigma)
        s = snr_db(X, out)
        print(f"    -> {s:.2f} dB  ({info['total_time_s']:.0f}s)", flush=True)
        res["runs"][name] = {"snr_out": s, "cfg": {k: v for k, v in cfg.items() if k != 'image_modes'},
                             "info": info}
        save_json(res, fname)


if __name__ == "__main__":
    main()

"""E3: open 3-D post-stack field data (300 x 100 x 10, dt = 4 ms).

(i)  semi-synthetic test: add white noise (sigma = 0.1 of max amplitude, the setting
     of the source repository) and measure SNR w.r.t. the original field record;
     baseline parameters are tuned here (oracle) and re-used in (ii).
(ii) raw field data (no ground truth): denoise, keep the removed noise and compute
     local similarity between denoised data and removed noise (Chen & Fomel, 2015).
"""
import argparse
import os

import numpy as np

from common import RES, Timer, env_info, save_json
from fptnn.data import IMAGE_MODES, load
from fptnn.metrics import local_similarity, mad_sigma, haar_hh, snr_db

from run_benchmark import CFG, SGK_PERC, TNN_GRID


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=17)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--sigma_mode", default="mad_band")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--rect", type=int, nargs=3, default=[20, 10, 3])
    a = p.parse_args()

    from fptnn.baselines.patchunet import patchunet_denoise
    from fptnn.baselines.sgk import sgk_denoise
    from fptnn.baselines.tnn import tnn_denoise
    from fptnn.model import FPTNN

    X0 = load("field3d")
    im = IMAGE_MODES["field3d"]
    res = {"env": env_info(), "args": vars(a), "semi": {}, "raw": {}}

    def fptnn_run(Y, X=None, sig=None):
        m = FPTNN(image_modes=im, kind="framelet", ftype="linear", levels=1, bc="symmetric",
                  depth=a.depth, width=a.width, steps=a.steps, batch=a.batch,
                  sigma_mode=a.sigma_mode, threads=a.threads, log_every=max(a.steps // 10, 1))
        return m.fit_denoise(Y, X=X, sigma_true=sig)

    # ---------------- (i) semi-synthetic
    rng = np.random.default_rng(201314)
    sigma = 0.1
    Y = X0 + sigma * rng.standard_normal(X0.shape)
    res["semi"]["snr_in"] = snr_db(X0, Y)
    print(f"semi-synthetic input SNR {res['semi']['snr_in']:.2f} dB", flush=True)
    best_c, best_s = None, -1e9
    for c in TNN_GRID:
        s = snr_db(X0, tnn_denoise(Y, sigma, im, c))
        if s > best_s:
            best_c, best_s = c, s
    res["semi"]["tnn"] = {"snr": best_s, "c": best_c}
    cfg = CFG["field3d"]
    outs = sgk_denoise(Y, cfg["sgk"]["psize"], cfg["sgk"]["step"], K=cfg["sgk"]["K"], perc=SGK_PERC)
    snrs = {k: snr_db(X0, v) for k, v in outs.items()}
    best_p = max(snrs, key=snrs.get)
    res["semi"]["sgk"] = {"snr": snrs[best_p], "perc": best_p}
    out_pu, _ = patchunet_denoise(Y, cfg["pu"]["psize"], cfg["pu"]["step"])
    res["semi"]["patchunet"] = {"snr": snr_db(X0, out_pu)}
    out_f, info = fptnn_run(Y, X=X0, sig=sigma)
    res["semi"]["fptnn"] = {"snr": snr_db(X0, out_f), "info": info}
    print({k: v["snr"] for k, v in res["semi"].items() if isinstance(v, dict)}, flush=True)
    save_json(res, "e3_field3d.json")

    # ---------------- (ii) raw field data
    Yr = X0
    # robust global noise estimate (Eq. (24)) on the (t, x) sections
    secs = np.moveaxis(Yr, 2, 0)  # (y, t, x)
    s_hat = mad_sigma(haar_hh(secs))
    res["raw"]["sigma_hat"] = s_hat
    outs = {}
    with Timer() as t:
        outs["tnn"] = tnn_denoise(Yr, s_hat, im, best_c)
    res["raw"]["tnn_time"] = t.s
    with Timer() as t:
        outs["sgk"] = sgk_denoise(Yr, cfg["sgk"]["psize"], cfg["sgk"]["step"], K=cfg["sgk"]["K"], perc=best_p)
    res["raw"]["sgk_time"] = t.s
    outs["patchunet"], i_pu = patchunet_denoise(Yr, cfg["pu"]["psize"], cfg["pu"]["step"])
    res["raw"]["patchunet_time"] = i_pu["time_s"]
    outs["fptnn"], info = fptnn_run(Yr)
    res["raw"]["fptnn_info"] = info
    for k, v in outs.items():
        noise = Yr - v
        ls = local_similarity(v, noise, a.rect)
        res["raw"][k] = {"removed_rms": float(np.sqrt(np.mean(noise ** 2))),
                         "ls_mean": float(ls.mean()), "ls_p95": float(np.percentile(ls, 95))}
        np.save(os.path.join(RES, f"vol_e3_field3d_{k}.npy"), v.astype(np.float32))
        np.save(os.path.join(RES, f"ls_e3_field3d_{k}.npy"), ls.astype(np.float32))
        print(k, res["raw"][k], flush=True)
    save_json(res, "e3_field3d.json")


if __name__ == "__main__":
    main()

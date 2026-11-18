"""Figures and LaTeX tables from the JSON/NPY results.

usage: python make_figures.py [--tag main]
"""
import argparse
import json
import os

import numpy as np

from common import FIG, RES
from fptnn.data import load

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "font.family": "serif"})
NAMES = {"tnn": "TNN", "sgk": "SGK", "patchunet": "PATCHUNET", "ftsvt": "F-tSVT",
         "fptnn": "FPTNN (SURE)", "fptnn_mse": "FPTNN (MSE, supervised)",
         "fptnn_oracle": "FPTNN (SURE, true $\\sigma$)"}
COLORS = {"tnn": "#3a923a", "sgk": "#1f5fa8", "patchunet": "#e08b00", "ftsvt": "#7a5195",
          "fptnn": "#d1495b", "fptnn_mse": "#8c1c2e", "fptnn_oracle": "#ef8a9a"}
MARK = {"tnn": "v", "sgk": "o", "patchunet": "x", "ftsvt": "s", "fptnn": "D", "fptnn_mse": "^",
        "fptnn_oracle": "d"}
TAB = os.path.join(RES, "tables")
os.makedirs(TAB, exist_ok=True)


def jload(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ SNR curves + tables
def snr_curves(tag, datasets):
    fig, axs = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 2.7))
    axs = np.atleast_1d(axs)
    for ax, data in zip(axs, datasets):
        r = jload(f"{tag}_{data}.json")
        if r is None:
            continue
        runs = r["runs"]
        meths = [m for m in NAMES if any(x["method"] == m for x in runs)]
        snrs = sorted({x["snr_in"] for x in runs})
        rows = []
        for m in meths:
            pts = sorted((x["snr_in"], x["snr_out"]) for x in runs if x["method"] == m)
            ax.plot(*zip(*pts), marker=MARK[m], color=COLORS[m], label=NAMES[m], lw=1.2, ms=4,
                    mfc="none" if m != "fptnn" else COLORS[m])
            d = dict(pts)
            rows.append((m, [d.get(s) for s in snrs]))
        ax.plot(snrs, snrs, "k:", lw=0.7, label="no denoising")
        ax.set_xlabel("input SNR (dB)")
        ax.set_ylabel("output SNR (dB)")
        ax.set_title({"synth5d": "5-D synthetic", "hyper4d": "4-D curved-event synthetic"}.get(data, data))
        ax.grid(alpha=0.3)
        # LaTeX table
        with open(os.path.join(TAB, f"{tag}_{data}_snr.tex"), "w") as f:
            f.write("\\begin{tabular}{l" + "c" * len(snrs) + "}\n\\hline\n")
            f.write("Method & " + " & ".join(f"{s:g} dB" for s in snrs) + " \\\\\n\\hline\n")
            best = [max((v[i] for _, v in rows if v[i] is not None), default=None) for i in range(len(snrs))]
            for m, v in rows:
                cells = []
                for i, x in enumerate(v):
                    if x is None:
                        cells.append("--")
                    elif m != "fptnn_mse" and best[i] is not None and abs(x - max(
                            y[i] for mm, y in rows if mm != "fptnn_mse" and y[i] is not None)) < 1e-9:
                        cells.append(f"\\textbf{{{x:.2f}}}")
                    else:
                        cells.append(f"{x:.2f}")
                f.write(NAMES[m].replace("$\\sigma$", "$\\sigma$") + " & " + " & ".join(cells) + " \\\\\n")
            f.write("\\hline\n\\end{tabular}\n")
    axs[-1].legend(fontsize=6.5, frameon=False, loc="upper left")
    plt.tight_layout()
    save(fig, f"{tag}_snr_curves")


def runtime_table(tag, datasets):
    lines = []
    for data in datasets:
        r = jload(f"{tag}_{data}.json")
        if r is None:
            continue
        for m in NAMES:
            ts = [x["info"].get("time_s") for x in r["runs"] if x["method"] == m and x["info"].get("time_s")]
            if ts:
                lines.append((data, m, float(np.median(ts))))
    with open(os.path.join(TAB, f"{tag}_runtime.tex"), "w") as f:
        f.write("\\begin{tabular}{llr}\n\\hline\nData & Method & Time (s) \\\\\n\\hline\n")
        for d, m, t in lines:
            f.write(f"{d} & {NAMES[m]} & {t:.1f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    return lines


# ------------------------------------------------------------------ sections
def _clip(a, q=99):
    return np.percentile(np.abs(a), q)


def sections(tag, data, meths=("sgk", "tnn", "patchunet", "fptnn")):
    p = lambda m: os.path.join(RES, f"vol_{tag}_{data}_{m}.npy")
    if not os.path.exists(p("noisy")):
        return
    X = load(data)
    Y = np.load(p("noisy")).astype(float)
    outs = {m: np.load(p(m)).astype(float) for m in meths if os.path.exists(p(m))}
    if data == "synth5d":
        sec = lambda v: v[:, :, :, 0, 0].reshape(100, 100, order="F")
        xl = "trace ($h_x$, $h_y$)"
    else:
        sec = lambda v: v[:, :, 16, 2]
        xl = "x"
    c = _clip(sec(X))
    cols = ["clean", "noisy"] + list(outs)
    fig, axs = plt.subplots(2, len(cols), figsize=(1.35 * len(cols) + 0.3, 3.8), sharey=True)
    for j, name in enumerate(cols):
        v = X if name == "clean" else Y if name == "noisy" else outs[name]
        axs[0, j].imshow(sec(v), aspect="auto", cmap="gray", vmin=-c, vmax=c)
        axs[0, j].set_title({"clean": "Clean", "noisy": "Noisy"}.get(name, NAMES.get(name, name)))
        if name == "clean":
            rem, t = np.zeros_like(sec(X)), ""
        elif name == "noisy":
            rem, t = sec(Y - X), "Added noise"
        else:
            rem, t = sec(Y - v), "Removed by " + NAMES[name].split(" ")[0]
        axs[1, j].imshow(rem, aspect="auto", cmap="gray", vmin=-c, vmax=c)
        axs[1, j].set_title(t)
        for a in axs[:, j]:
            a.set_xticks([])
        axs[1, j].set_xlabel(xl)
    axs[0, 0].set_ylabel("time sample")
    axs[1, 0].set_ylabel("time sample")
    axs[1, 0].axis("off")
    plt.tight_layout()
    save(fig, f"{tag}_{data}_sections")


def field_figure():
    r = jload("e3_field3d.json")
    if r is None:
        return
    X0 = load("field3d")
    meths = [m for m in ("tnn", "sgk", "patchunet", "fptnn") if os.path.exists(os.path.join(RES, f"vol_e3_field3d_{m}.npy"))]
    y = 5
    c = _clip(X0[:, :, y])
    fig, axs = plt.subplots(3, len(meths) + 1, figsize=(1.45 * (len(meths) + 1) + 0.4, 5.6), sharey=True)
    axs[0, 0].imshow(X0[:, :, y], aspect="auto", cmap="gray", vmin=-c, vmax=c)
    axs[0, 0].set_title("Raw field data")
    axs[1, 0].axis("off")
    axs[2, 0].axis("off")
    im = None
    for j, m in enumerate(meths, 1):
        v = np.load(os.path.join(RES, f"vol_e3_field3d_{m}.npy")).astype(float)
        ls = np.load(os.path.join(RES, f"ls_e3_field3d_{m}.npy")).astype(float)
        axs[0, j].imshow(v[:, :, y], aspect="auto", cmap="gray", vmin=-c, vmax=c)
        axs[0, j].set_title(NAMES[m].split(" ")[0])
        axs[1, j].imshow((X0 - v)[:, :, y], aspect="auto", cmap="gray", vmin=-c, vmax=c)
        axs[1, j].set_title("removed noise")
        im = axs[2, j].imshow(ls[:, :, y], aspect="auto", cmap="jet", vmin=0, vmax=0.5)
        axs[2, j].set_title(f"LS, mean {ls.mean():.3f}")
    for a in axs.ravel():
        a.set_xticks([])
    axs[0, 0].set_ylabel("time sample")
    if im is not None:
        fig.colorbar(im, ax=axs[2, 1:].tolist(), shrink=0.8, pad=0.01)
    save(fig, "e3_field3d_sections")
    with open(os.path.join(TAB, "e3_field3d.tex"), "w") as f:
        f.write("\\begin{tabular}{lccc}\n\\hline\nMethod & semi-synth. SNR (dB) & LS mean & LS 95th pct. \\\\\n\\hline\n")
        f.write(f"Input & {r['semi']['snr_in']:.2f} & -- & -- \\\\\n")
        for m in meths:
            s = r["semi"].get(m, {}).get("snr")
            raw = r["raw"].get(m, {})
            f.write(f"{NAMES[m].split(' ')[0]} & {s:.2f} & {raw.get('ls_mean', float('nan')):.3f} & "
                    f"{raw.get('ls_p95', float('nan')):.3f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")


# ------------------------------------------------------------------ SURE vs true MSE
def sure_curve(tag, data, snr):
    r = jload(f"{tag}_{data}.json")
    if r is None:
        return
    runs = [x for x in r["runs"] if x["method"] in ("fptnn", "fptnn_oracle") and x["snr_in"] == snr]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for x in runs:
        h = x["info"]["hist"]
        ax.plot(h["step"], h["loss"], color=COLORS[x["method"]], label="SURE estimate (" + NAMES[x["method"]] + ")")
        if h["true_mse"][0] is not None:
            ax.plot(h["step"], h["true_mse"], "--", color=COLORS[x["method"]], label="true MSE (not used)")
    ax.set_xlabel("training step")
    ax.set_ylabel("per-pixel error ($\\sigma$-normalised)")
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, f"{tag}_{data}_sure_vs_mse")


# ------------------------------------------------------------------ ablations
ABL_LABEL = {"base": "Base: linear framelet, Eq.(24) $\\sigma$, SURE",
             "haar": "Haar framelet", "cubic": "Cubic B-spline framelet", "linear_L2": "Linear framelet, 2 levels",
             "dct": "DCT instead of framelet", "identity": "No transform (2-D DnCNN on slices)",
             "sig_global": "$\\sigma$: global MAD propagated through $W$", "sig_oracle": "$\\sigma$: true value",
             "loss_mse": "MSE loss with clean data", "slices_tx": "(t, x) frontal slices"}


def ablation(tag, datasets):
    for data in datasets:
        r = jload(f"{tag}_ablation_{data}.json")
        if r is None:
            continue
        names = [k for k in ABL_LABEL if k in r["runs"]]
        vals = [r["runs"][k]["snr_out"] for k in names]
        fig, ax = plt.subplots(figsize=(3.5, 0.28 * len(names) + 0.6))
        base = r["runs"].get("base", {}).get("snr_out")
        ax.barh(range(len(names)), vals, color=["#d1495b" if k == "base" else "#777777" for k in names])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([ABL_LABEL[k] for k in names], fontsize=6.5)
        ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=6.5)
        ax.set_xlabel("output SNR (dB)")
        ax.set_xlim(min(vals) - 2, max(vals) + 2)
        if base is not None:
            ax.axvline(base, color="#d1495b", lw=0.7, ls=":")
        plt.tight_layout()
        save(fig, f"{tag}_ablation_{data}")
        with open(os.path.join(TAB, f"{tag}_ablation_{data}.tex"), "w") as f:
            f.write("\\begin{tabular}{lc}\n\\hline\nVariant & SNR (dB) \\\\\n\\hline\n")
            for k, v in zip(names, vals):
                f.write(f"{ABL_LABEL[k]} & {v:.2f} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n")


def grid(tag, data):
    r = jload(f"{tag}_grid_{data}.json")
    if r is None:
        return
    pts = {}
    for k, v in r["runs"].items():
        d, w = k.split("_")
        pts.setdefault(int(w[1:]), []).append((int(d[1:]), v["snr_out"]))
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    for w in sorted(pts):
        ax.plot(*zip(*sorted(pts[w])), marker="o", label=f"{w} filters", mfc="none")
    ax.set_xlabel("network depth")
    ax.set_ylabel("output SNR (dB)")
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, f"{tag}_grid_{data}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="main")
    p.add_argument("--abl_tag", default="e4")
    a = p.parse_args()
    ds = ["synth5d", "hyper4d"]
    snr_curves(a.tag, ds)
    runtime_table(a.tag, ds)
    for d in ds:
        sections(a.tag, d)
        sure_curve(a.tag, d, 0.0)
    field_figure()
    ablation(a.abl_tag, ["hyper4d", "synth5d"])
    grid(a.abl_tag, "synth5d")
    print("figures ->", FIG)

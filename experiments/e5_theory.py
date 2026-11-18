"""E5 -- numerical checks of the theory behind FPTNN (no training).

 (a) tight-frame property of the framelet matrices: W^T W = I, W W^T != I
 (b) order-p t-product: block-circulant definition == FFT slice-wise products;
     framelet-domain product (Def. 2) vs block-circulant formula (Def. 1)
 (c) noise in the framelet domain: per-band std = sigma * ||row of W||, white inside
     every frontal slice, correlated across bands
 (d) Eq. (24) MAD estimate per band vs the true per-band sigma
 (e) energy compaction: per-band SNR of the transformed 5-D data
"""
import numpy as np

from common import FIG, save_json
from fptnn.data import IMAGE_MODES, load
from fptnn.framelet import FILTERS, ModeTransform, framelet_matrix, uep_residual
from fptnn.metrics import add_noise_snr, haar_hh, mad_sigma
from fptnn.noise_est import slice_sigmas
from fptnn.tensor_ops import tprod_definition, tprod_fft, tprod_transform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = {}

# (a) ---------------------------------------------------------------
rows = []
for f in FILTERS:
    for L in (1, 2):
        for bc in ("periodic", "symmetric"):
            W, _ = framelet_matrix(100, f, L, bc)
            rows.append({"filter": f, "levels": L, "bc": bc, "shape": list(W.shape),
                         "uep": uep_residual(f),
                         "WtW_minus_I": float(np.abs(W.T @ W - np.eye(100)).max()),
                         "WWt_minus_I": float(np.abs(W @ W.T - np.eye(W.shape[0])).max())})
out["tight_frame"] = rows
print("(a) tight frame:")
for r in rows:
    print(f"   {r['filter']:6s} L{r['levels']} {r['bc']:9s} W{tuple(r['shape'])}  "
          f"|WtW-I|={r['WtW_minus_I']:.1e}  |WWt-I|={r['WWt_minus_I']:.2f}")

# (b) ---------------------------------------------------------------
rng = np.random.default_rng(0)
tp = []
for shp in [(4, 3, 6), (4, 3, 6, 5), (4, 3, 6, 5, 4)]:
    A = rng.standard_normal(shp)
    B = rng.standard_normal((shp[1], 2) + shp[2:])
    Cd = tprod_definition(A, B)
    Cf = tprod_fft(A, B)
    mats = {m: framelet_matrix(shp[m], "linear", 1, "symmetric")[0] for m in range(2, len(shp))}
    Cw = tprod_transform(A, B, mats)
    tp.append({"order": len(shp), "def_vs_fft": float(np.abs(Cd - Cf).max()),
               "framelet_vs_circulant_rel": float(np.linalg.norm(Cw - Cd) / np.linalg.norm(Cd))})
out["tproduct"] = tp
print("(b) t-product:", tp)

# (c) ---------------------------------------------------------------
X = load("synth5d")
im = IMAGE_MODES["synth5d"]
perm = list(im) + [m for m in range(5) if m not in im]
Xp = np.transpose(X, perm)
T = ModeTransform(Xp.shape, range(2, 5), "framelet", "linear", 1, "symmetric")
noise = np.random.default_rng(1).standard_normal(Xp.shape)
Nt = T.forward(noise)
lab, nb = T.band_index()
lab_full = np.broadcast_to(lab, Nt.shape)
std_an = np.broadcast_to(T.coef_noise_std(1.0), Nt.shape)
band_std_emp = np.array([Nt[lab_full == b].std() for b in range(nb)])
band_std_an = np.array([std_an[lab_full == b].mean() for b in range(nb)])
# whiteness inside a frontal slice: correlation between neighbouring pixels
S = np.moveaxis(Nt.reshape(10, 10, -1), 2, 0)
c_in = np.mean([np.corrcoef(S[:, :-1, :].ravel(), S[:, 1:, :].ravel())[0, 1],
                np.corrcoef(S[:, :, :-1].ravel(), S[:, :, 1:].ravel())[0, 1]])
# correlation across neighbouring time-framelet coefficients (same band)
c_t = np.corrcoef(Nt[:, :, 1:99].ravel(), Nt[:, :, 2:100].ravel())[0, 1]
out["noise_in_framelet_domain"] = {
    "band_std_empirical": band_std_emp, "band_std_analytic": band_std_an,
    "max_rel_err": float(np.max(np.abs(band_std_emp / band_std_an - 1))),
    "neighbour_corr_within_slice": float(c_in), "neighbour_corr_along_time_band": float(c_t)}
print(f"(c) band std: max rel err {out['noise_in_framelet_domain']['max_rel_err']:.3f}; "
      f"corr within slice {c_in:.3f}; corr along transformed time axis {c_t:.3f}")

# (d) + (e) ----------------------------------------------------------
est = []
for snr_in in (-5, 0, 5):
    Y, Nn, sig = add_noise_snr(X, snr_in, seed=10)
    Yp = np.transpose(Y, perm)
    Yt = T.forward(Yp)
    Sy = np.moveaxis(Yt.reshape(10, 10, -1), 2, 0)
    lab_s = np.broadcast_to(lab, (1, 1) + Yt.shape[2:]).reshape(-1)
    s_mad = slice_sigmas(Sy, lab_s, "mad_band", T)
    s_glob = slice_sigmas(Sy, lab_s, "mad_global", T, Y_orig_slices=np.moveaxis(Yp.reshape(10, 10, -1), 2, 0))
    s_or = slice_sigmas(Sy, lab_s, "oracle", T, sigma_true=sig)
    est.append({"snr_in": snr_in,
                "mad_band_rel_err_median": float(np.median(np.abs(s_mad / s_or - 1))),
                "mad_band_rel_err_max": float(np.max(np.abs(s_mad / s_or - 1))),
                "mad_global_rel_err": float(np.median(np.abs(s_glob / s_or - 1)))})
out["sigma_estimation"] = est
print("(d) sigma estimation:", est)

Xt = T.forward(Xp)
Ntt = T.forward(np.transpose(add_noise_snr(X, 0, seed=10)[1], perm))
lab_full = np.broadcast_to(lab, Xt.shape)
eS = np.array([np.sum(Xt[lab_full == b] ** 2) for b in range(nb)])
eN = np.array([np.sum(Ntt[lab_full == b] ** 2) for b in range(nb)])
band_snr = 10 * np.log10(eS / eN)
out["energy_compaction_0dB"] = {"signal_energy_frac": eS / eS.sum(), "noise_energy_frac": eN / eN.sum(),
                                "band_snr_db": band_snr}
print("(e) per-band SNR at 0 dB input: min %.1f max %.1f dB; low-pass band holds %.1f%% of signal energy"
      % (band_snr.min(), band_snr.max(), 100 * eS[0] / eS.sum()))

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
idx = np.arange(nb)
ax[0].bar(idx - 0.2, 100 * eS / eS.sum(), 0.4, label="clean signal", color="#1f5fa8")
ax[0].bar(idx + 0.2, 100 * eN / eN.sum(), 0.4, label="noise", color="#d1495b")
ax[0].set_xlabel("framelet band (t, h_x, h_y)")
ax[0].set_ylabel("energy share (%)")
ax[0].legend(frameon=False, fontsize=7)
ax[1].bar(idx, band_snr, color="#444444")
ax[1].axhline(0, color="k", lw=0.6)
ax[1].set_xlabel("framelet band")
ax[1].set_ylabel("band SNR (dB)")
for a in ax:
    a.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig(f"{FIG}/e5_energy_compaction.pdf")
plt.savefig(f"{FIG}/e5_energy_compaction.png", dpi=150)
save_json(out, "e5_theory.json")
print("saved")

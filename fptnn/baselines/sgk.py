"""SGK dictionary-learning denoiser for M-D seismic data (Chen, GJI 2020), Python port
of yc_sgk.m / yc_patch5d.m / yc_pthresh.m from github.com/chenyk1990/reproducible_research
(sgk5d), generalised to any order.

Pipeline: overlapping M-D patches -> SGK dictionary (DCT-Kronecker initialisation,
`niter` rounds of 1-sparse coding + closed-form atom update) -> T-sparse OMP coding ->
percentile hard thresholding of the coefficients -> patch averaging.
Atom update uses the coefficient-weighted sum d_k ∝ sum_i g_ik x_i (least-squares
solution for 1-sparse codes, Sahoo & Makur 2013); Chen's code uses the plain sum,
which coincides when all g_ik > 0.
"""
import math

import numpy as np
import scipy.sparse as sp
from numpy.lib.stride_tricks import sliding_window_view


def patch_extract(D, psize, step):
    """Zero-pad each dim so that (n - l) % s == 0 (as yc_patch5d), return
    (patches N x P, meta)."""
    pads = []
    for n, l, s in zip(D.shape, psize, step):
        r = (n - l) % s
        pads.append((0, (s - r) % s))
    Dp = np.pad(D, pads)
    V = sliding_window_view(Dp, psize)
    V = V[tuple(slice(None, None, s) for s in step)]
    grid = V.shape[: D.ndim]
    X = V.reshape(-1, int(np.prod(psize))).copy()
    return X, {"shape": D.shape, "padded": Dp.shape, "grid": grid, "psize": psize, "step": step}


def patch_inverse(X, meta):
    psize, step, grid = meta["psize"], meta["step"], meta["grid"]
    acc = np.zeros(meta["padded"])
    cnt = np.zeros(meta["padded"])
    Xr = X.reshape(grid + tuple(psize))
    for idx in np.ndindex(*grid):
        sl = tuple(slice(i * s, i * s + l) for i, s, l in zip(idx, step, psize))
        acc[sl] += Xr[idx]
        cnt[sl] += 1
    out = acc / np.maximum(cnt, 1)
    return out[tuple(slice(0, n) for n in meta["shape"])]


def dct_kron_dict(psize, K):
    """DCT-Kronecker initial dictionary of yc_sgk_recon5d.m (first K columns)."""
    c = math.ceil(K ** (1.0 / len(psize)))
    D = np.ones((1, 1))
    for l in psize:
        B = np.zeros((l, c))
        for k in range(c):
            v = np.cos(np.arange(l) * k * np.pi / c)
            if k > 0:
                v = v - v.mean()
            B[:, k] = v / (np.linalg.norm(v) + 1e-12)
        D = np.kron(D, B)
    D = D[:, :K]
    return D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)


def omp_batch(D, X, T, DtX=None, G=None):
    """Vectorised orthogonal matching pursuit: X (P x N) ~ D (P x K) C, ||c_n||_0 <= T."""
    K, N = D.shape[1], X.shape[1]
    if DtX is None:
        DtX = D.T @ X
    if G is None:
        G = D.T @ D
    R = X.copy()
    sel = np.zeros((T, N), dtype=np.int64)
    ar = np.arange(N)
    coef = None
    for t in range(T):
        corr = np.abs(D.T @ R)
        for j in range(t):
            corr[sel[j], ar] = -1.0
        sel[t] = np.argmax(corr, axis=0)
        I = sel[: t + 1]                          # (t+1, N)
        Gs = G[I.T[:, :, None], I.T[:, None, :]]  # (N, t+1, t+1)
        rhs = DtX[I, ar]                          # (t+1, N)
        coef = np.linalg.solve(Gs + 1e-10 * np.eye(t + 1), rhs.T[:, :, None])[:, :, 0]  # (N, t+1)
        R = X - np.einsum("pnj,nj->pn", D[:, I.T], coef)
    rows = sel.T.reshape(-1)
    cols = np.repeat(ar, T)
    return sp.csr_matrix((coef.reshape(-1), (rows, cols)), shape=(K, N))


def sgk_train(X, K=512, T=3, niter=10, D0=None, psize=None):
    """SGK dictionary learning (yc_sgk.m). X: P x N patch matrix."""
    D = D0 if D0 is not None else dct_kron_dict(psize, K)
    N = X.shape[1]
    ar = np.arange(N)
    for _ in range(niter):
        corr = D.T @ X
        k = np.argmax(np.abs(corr), axis=0)
        g = corr[k, ar]
        Gm = sp.csr_matrix((g, (k, ar)), shape=(D.shape[1], N))
        Dn = np.asarray((Gm @ X.T).T)              # sum_i g_ik x_i
        nrm = np.linalg.norm(Dn, axis=0)
        used = nrm > 1e-12
        D[:, used] = Dn[:, used] / nrm[used]
    C = omp_batch(D, X, T)
    return D, C


def pthresh_hard(C, perc):
    """yc_pthresh(x,'ph',perc): keep the `perc` percent largest |coefficients|."""
    vals = np.abs(C.data)
    n_total = C.shape[0] * C.shape[1]
    keep = int(round(perc / 100.0 * n_total))
    if keep >= vals.size:
        return C
    thr = np.partition(vals, vals.size - keep)[vals.size - keep] if keep > 0 else np.inf
    C2 = C.copy()
    C2.data = np.where(np.abs(C2.data) >= thr, C2.data, 0.0)
    C2.eliminate_zeros()
    return C2


def sgk_denoise(Y, psize, step, K=512, T=3, niter=10, perc=0.5, return_parts=False):
    Xp, meta = patch_extract(Y, psize, step)
    X = Xp.T  # P x N
    D, C = sgk_train(X, K, T, niter, psize=psize)
    out = {}
    percs = perc if isinstance(perc, (list, tuple, np.ndarray)) else [perc]
    for pc in percs:
        Ct = pthresh_hard(C, pc)
        Xh = np.asarray(D @ Ct.toarray()) if Ct.nnz else np.zeros_like(X)
        out[float(pc)] = patch_inverse(Xh.T, meta)
    if return_parts:
        return out, D, C
    return out if len(percs) > 1 else out[float(percs[0])]

"""FPTNN: framelet-based order-p tensor neural network (Qian et al., IEEE TGRS 2022).

Algorithm 1 of the paper, implemented as
  1) X~ = Y x_3 W_3 x_4 W_4 ... x_p W_p          (framelet transform of modes 3..p)
  2) every frontal slice X~(:,:,i3,..,ip) is denoised by one shared matrix-DL model
     MDL = DnCNN trained *without clean data* by minimising Stein's unbiased risk
     estimate (SURE, Eq. (6)) with a Monte-Carlo divergence
  3) X_hat = X~ x_p W_p^T ... x_3 W_3^T             (inverse transform, W^T W = I)

By Definition 2 the order-p framelet t-product is a slice-wise matrix product in
the framelet domain, so a network acting on the frontal slices of X~ is the
practical realisation of the order-p tNN of Definition 4 / Eq. (21).
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn

from .framelet import ModeTransform
from .noise_est import slice_sigmas


# ---------------------------------------------------------------- network
class DnCNN(nn.Module):
    """DnCNN (Zhang et al., 2017): Conv+ReLU, (D-2) x [Conv+BN+ReLU], Conv;
    residual learning, h(y) = y - R(y)."""

    def __init__(self, depth=17, width=64, bn=True):
        super().__init__()
        layers = [nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(width, width, 3, padding=1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(width, momentum=0.05))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(width, 1, 3, padding=1))
        self.body = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.body[-1].weight)  # start as identity denoiser h(y)=y
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, y):
        return y - self.body(y)


def sure_loss(model, y, sigma, eps):
    """Monte-Carlo SURE (Ramani et al., 2008; Soltanayev & Chun, 2018), Eq. (6):
       (1/K)||y - h(y)||^2 - sigma^2 + (2 sigma^2 / K) div_y h(y),
       div_y h(y) ~= b^T (h(y + eps b) - h(y)) / eps,  b ~ N(0, I).
    Returns (mean loss, h(y)).  y: (B,1,H,W); sigma, eps: (B,)."""
    out = model(y)
    b = torch.randn_like(y)
    out_p = model(y + eps.view(-1, 1, 1, 1) * b)
    K = y[0].numel()
    div = (b * (out_p - out)).flatten(1).sum(1) / eps
    data = ((y - out) ** 2).flatten(1).sum(1) / K
    loss = data - sigma ** 2 + 2.0 * sigma ** 2 * div / K
    return loss.mean(), out


# ---------------------------------------------------------------- the method
class FPTNN:
    def __init__(self, image_modes=(0, 1), kind="framelet", ftype="linear", levels=1,
                 bc="symmetric", depth=17, width=64, bn=True, steps=3000, batch=32,
                 lr=1e-3, lr_final=1e-4, loss="sure", sigma_mode="mad_band",
                 eps_rel=1e-3, crop=None, seed=0, threads=None, device=None, log_every=100,
                 infer_batch=4096, verbose=True):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

    # --------------------------------------------------------------
    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def fit_denoise(self, Y, X=None, sigma_true=None):
        """Denoise the order-p tensor Y (ground-truth free if loss='sure').
        X (clean) is only used for loss='mse' (supervised upper bound) and for
        logging the true error next to the SURE estimate."""
        t0 = time.time()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        if self.threads:
            torch.set_num_threads(self.threads)
        p = Y.ndim
        perm = list(self.image_modes) + [m for m in range(p) if m not in self.image_modes]
        inv_perm = np.argsort(perm)
        Yp = np.transpose(Y, perm).astype(np.float64)
        T = ModeTransform(Yp.shape, modes=range(2, p), kind=self.kind, ftype=self.ftype,
                          levels=self.levels, bc=self.bc)
        Yt = T.forward(Yp)
        n1, n2 = Yt.shape[:2]
        S = np.ascontiguousarray(np.moveaxis(Yt.reshape(n1, n2, -1), 2, 0))  # (N, n1, n2)
        N = S.shape[0]
        lab, _ = T.band_index()
        lab = np.broadcast_to(lab, (1, 1) + Yt.shape[2:]).reshape(-1)
        Yorig_slices = np.moveaxis(Yp.reshape(n1, n2, -1), 2, 0)
        sig = slice_sigmas(S, lab, self.sigma_mode, T, sigma_true, Yorig_slices)
        sig = np.maximum(sig, 1e-8 * (np.abs(S).max() + 1e-12))
        info = {"n_slices": int(N), "slice_shape": [int(n1), int(n2)],
                "redundancy": float(np.prod(Yt.shape) / np.prod(Yp.shape)),
                "sigma_slices_mean": float(sig.mean())}
        if sigma_true is not None:
            sig_or = slice_sigmas(S, lab, "oracle", T, sigma_true)
            info["sigma_rel_err"] = float(np.median(np.abs(sig / sig_or - 1)))
        Sn = torch.from_numpy((S / sig[:, None, None]).astype(np.float32)).unsqueeze(1)
        Xn = None
        if X is not None:
            Xt = T.forward(np.transpose(X, perm).astype(np.float64))
            Xs = np.moveaxis(Xt.reshape(n1, n2, -1), 2, 0)
            Xn = torch.from_numpy((Xs / sig[:, None, None]).astype(np.float32)).unsqueeze(1)
        if self.loss == "mse" and Xn is None:
            raise ValueError("loss='mse' needs the clean tensor X")
        self._log(f"[FPTNN] {self.kind}/{self.ftype}/L{self.levels}: {N} slices of "
                  f"{n1}x{n2}, redundancy {info['redundancy']:.1f}, sigma={self.sigma_mode}")

        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        info["device"] = str(dev)
        model = DnCNN(self.depth, self.width, self.bn).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps, eta_min=self.lr_final)
        hist = {"step": [], "loss": [], "true_mse": []}
        model.train()
        run_l, run_t, cnt = 0.0, 0.0, 0
        t_train = time.time()
        for it in range(1, self.steps + 1):
            idx = torch.from_numpy(rng.integers(0, N, self.batch))
            y = Sn[idx].to(dev, non_blocking=True)
            if self.crop is not None:  # random spatial crop (same window for the batch)
                ch, cw = min(self.crop[0], n1), min(self.crop[1], n2)
                r0 = int(rng.integers(0, n1 - ch + 1))
                c0 = int(rng.integers(0, n2 - cw + 1))
                y = y[:, :, r0:r0 + ch, c0:c0 + cw]
            ones = torch.ones(len(idx), device=dev)
            if self.loss == "sure":
                loss, out = sure_loss(model, y, ones, self.eps_rel * ones)
            else:
                out = model(y)
                xt = (Xn[idx] if self.crop is None else Xn[idx][:, :, r0:r0 + ch, c0:c0 + cw]).to(dev)
                loss = ((out - xt) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_l += float(loss.detach())
            if Xn is not None:
                with torch.no_grad():
                    xt = (Xn[idx] if self.crop is None else Xn[idx][:, :, r0:r0 + ch, c0:c0 + cw]).to(dev)
                    run_t += float(((out - xt) ** 2).mean())
            cnt += 1
            if it % self.log_every == 0 or it == self.steps:
                hist["step"].append(it)
                hist["loss"].append(run_l / cnt)
                hist["true_mse"].append(run_t / cnt if Xn is not None else None)
                self._log(f"  step {it:5d}  loss {run_l / cnt:.4f}" +
                          (f"  true-mse {run_t / cnt:.4f}" if Xn is not None else "") +
                          f"  ({time.time() - t_train:.0f}s)")
                run_l, run_t, cnt = 0.0, 0.0, 0
        info["train_time_s"] = time.time() - t_train

        model.eval()
        t_inf = time.time()
        out = torch.empty_like(Sn)
        with torch.no_grad():
            for s in range(0, N, self.infer_batch):
                out[s:s + self.infer_batch] = model(Sn[s:s + self.infer_batch].to(dev)).cpu()
        Xs_hat = out.squeeze(1).numpy().astype(np.float64) * sig[:, None, None]
        Xt_hat = np.moveaxis(Xs_hat, 0, 2).reshape(Yt.shape)
        Xp_hat = T.inverse(Xt_hat)
        info["infer_time_s"] = time.time() - t_inf
        info["total_time_s"] = time.time() - t0
        info["hist"] = hist
        self.model_ = model
        return np.transpose(Xp_hat, inv_perm), info

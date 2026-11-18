"""PATCHUNET (Saad et al., 2021): fully unsupervised patch-based U-Net autoencoder.
PyTorch port of github.com/omarmohamed15/PATCHUNET (Pytorch/PATCHUNET.py, Utils.py),
generalised from 2-D (48x48 patches) to M-D patches that are flattened to vectors.
Training: noisy patch -> same noisy patch, MSE, Adam(1e-3, betas=(0.5, 0.9)),
batch 128, 50 epochs, weights of the lowest-loss epoch kept; patches averaged back.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from .sgk import patch_extract, patch_inverse


class PatchUNET(nn.Module):
    def __init__(self, P):
        super().__init__()
        enc = [P, 128, 64, 32, 16, 8, 4]
        self.enc = nn.ModuleList([nn.Linear(a, b) for a, b in zip(enc[:-1], enc[1:])])
        dec_in = [4, 8, 16, 32, 64, 128]
        self.dec = nn.ModuleList([nn.Linear(d, d) for d in dec_in])
        self.out = nn.Linear(256, P)
        self.act = nn.ELU(1.0)

    def forward(self, x):
        es = []
        h = x
        for l in self.enc:
            h = self.act(l(h))
            es.append(h)
        d = es[-1]
        skips = es[::-1]  # e6, e5, ..., e1
        for i, l in enumerate(self.dec):
            d = self.act(l(d))
            d = torch.cat([d, skips[i]], dim=-1)
        return self.out(d)


def patchunet_denoise(Y, psize, step, epochs=50, batch=128, seed=0, verbose=False, device=None):
    t0 = time.time()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    Xp, meta = patch_extract(Y, psize, step)
    scale = np.abs(Xp).max() + 1e-12
    data = torch.from_numpy((Xp / scale).astype(np.float32)).to(dev)
    net = PatchUNET(data.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.5, 0.9), eps=1e-8)
    best, best_state = np.inf, None
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(data), generator=g).to(dev)
        tot = 0.0
        for s in range(0, len(data), batch):
            dd = data[perm[s:s + batch]]
            loss = ((net(dd) - dd) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(dd)
        if tot < best:
            best = tot
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  [PATCHUNET] epoch {ep} loss {tot / len(data):.5f}", flush=True)
    net.load_state_dict(best_state)
    with torch.no_grad():
        out = net(data).cpu().numpy().astype(np.float64) * scale
    return patch_inverse(out, meta), {"time_s": time.time() - t0, "n_patches": len(data),
                                       "patch_dim": int(data.shape[1])}

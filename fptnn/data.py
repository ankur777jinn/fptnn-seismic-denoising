"""Loaders for the open datasets used in this study (see data/SOURCES.md).

All tensors are returned with time as axis 0 and normalised to max |x| = 1.
  synth5d : (t, hx, hy, x, y) = 100 x 10 x 10 x 10 x 10, dt = 4 ms  [Qian et al. Sec. IV-D]
  hyper4d : (t, x, y, hx)     = 101 x 32 x 32 x 5, curved events  [substitute for the VSP set]
  field3d : (t, x, y)         = 300 x 100 x 10 post-stack field data, dt = 4 ms
Each entry also gives the two 'image' (frontal-slice) modes used by FPTNN/TNN:
the spatial midpoint coordinates, as in Fig. 1 of the paper (inline x xline).
"""
import os

import numpy as np
import scipy.io as sio

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

IMAGE_MODES = {"synth5d": (3, 4), "hyper4d": (1, 2), "field3d": (1, 2)}


def load(name, root=ROOT):
    if name == "synth5d":
        d = sio.loadmat(os.path.join(root, "yc_synth5d.mat"))["data5d"]
    elif name == "hyper4d":
        d = sio.loadmat(os.path.join(root, "yc_hyper5d.mat"))["hyper5d"][:, :, :, :, 2]
    elif name == "field3d":
        a = np.fromfile(os.path.join(root, "real3d_field.bin"), dtype=np.float32)
        # MATLAB: fread(fid,[300,1000]); reshape(d,300,100,10)  (column-major)
        d = a.reshape(10, 100, 300).transpose(2, 1, 0).astype(np.float64)
    else:
        raise ValueError(name)
    d = np.asarray(d, dtype=np.float64)
    return d / np.abs(d).max()

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)


def save_json(obj, name):
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return path


def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.s = time.time() - self.t


def env_info():
    import platform
    info = {"python": platform.python_version(), "machine": platform.machine(),
            "cpus": os.cpu_count(), "numpy": np.__version__}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    return info

"""Pool-based datasets from Liang et al. (2021), turned into ground-truth pools.

Each dataset becomes a finite pool of K unique candidates:
  * X  : (K, d) features min-max scaled to [0, 1]
  * f  : (K,)  hidden "true" objective, oriented so HIGHER IS BETTER and
               standardized to zero mean / unit variance
  * replicate_sd : pooled std of replicate measurements, in the same
               standardized units. This calibrates the measurement-noise
               sigma in the simulator with real lab noise.

Replicated rows (same composition / geometry) are averaged to get f_k.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DATASETS = {
    # P3HT/CNT films: maximize electrical conductivity (heavy tailed -> log)
    "p3ht": dict(file="P3HT_dataset.csv", target="Conductivity (measured) (S/cm)",
                 goal="max", transform="log"),
    # Perovskite films: minimize the instability index (heavy tailed -> log)
    "perovskite": dict(file="Perovskite_dataset.csv", target="Instability index",
                       goal="min", transform="log"),
    # 3D-printed crossed-barrel structures: maximize toughness
    "crossed_barrel": dict(file="Crossed_barrel_dataset.csv", target="toughness",
                           goal="max", transform=None),
}


@dataclass
class Pool:
    name: str
    X: np.ndarray            # (K, d) in [0, 1]
    f: np.ndarray            # (K,) standardized, higher is better
    replicate_sd: float      # pooled replicate noise, standardized units
    n_replicates: np.ndarray  # (K,) how many raw rows went into each f_k
    feature_names: list
    f_mean: float            # to undo standardization (transformed units)
    f_std: float

    @property
    def K(self) -> int:
        return len(self.f)

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.f, q))


def load_pool(name: str, data_dir: str = DATA_DIR) -> Pool:
    spec = DATASETS[name]
    df = pd.read_csv(os.path.join(data_dir, spec["file"]))
    feats = [c for c in df.columns if c != spec["target"]]
    y = df[spec["target"]].to_numpy(float)
    if spec["transform"] == "log":
        y = np.log(y)
    if spec["goal"] == "min":
        y = -y
    df = df[feats].round(6).copy()
    df["_y"] = y
    g = df.groupby(feats, sort=False)["_y"]
    agg = g.agg(["mean", "count", "var"]).reset_index()

    # pooled replicate variance: sum (n_i - 1) s_i^2 / sum (n_i - 1)
    rep = agg[agg["count"] > 1]
    dof = (rep["count"] - 1).sum()
    pooled_var = float(((rep["count"] - 1) * rep["var"]).sum() / dof) if dof > 0 else 0.0

    f_raw = agg["mean"].to_numpy(float)
    mu, sd = f_raw.mean(), f_raw.std()
    f = (f_raw - mu) / sd

    X = agg[feats].to_numpy(float)
    lo, hi = X.min(0), X.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    X = (X - lo) / span

    return Pool(name=name, X=X, f=f, replicate_sd=float(np.sqrt(pooled_var) / sd),
                n_replicates=agg["count"].to_numpy(int), feature_names=feats,
                f_mean=float(mu), f_std=float(sd))


if __name__ == "__main__":
    for n in DATASETS:
        p = load_pool(n)
        print(f"{n:15s} K={p.K:4d} d={p.X.shape[1]} replicate_sd={p.replicate_sd:.3f} "
              f"top5%-threshold={p.quantile(0.95):.3f} max={p.f.max():.3f}")

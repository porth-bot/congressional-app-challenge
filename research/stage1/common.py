"""Shared helpers: loading features, metrics, bootstrap intervals."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONDITIONS = ["clean", "g711", "opus", "g723"]
COND_NAMES = {
    "clean": "Studio (16 kHz)",
    "g711": "Landline (G.711)",
    "opus": "VoIP (Opus 8 kbps)",
    "g723": "Cellular-style (G.723.1)",
}
TRAIN_CODECS = ["g711", "opus"]  # g723 is never seen in training
SETS = {"asv_train": "ASVspoof 2019 train", "asv_test": "ASVspoof 2019 eval (unseen attacks)", "itw": "In-the-Wild"}


def manifest():
    m = pd.read_csv(ROOT / "data" / "manifest.csv")
    m["y"] = (m["label"] == "spoof").astype(int)
    return m


def load(cond, cols=None):
    df = pd.read_parquet(ROOT / "data" / "features" / f"{cond}.parquet")
    df = manifest().merge(df, on="clip_id")
    df["cond"] = cond
    return df if cols is None else df[["clip_id", "corpus", "speaker", "system", "y", "cond", *cols]]


def eer(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def auc(y, s):
    return float(roc_auc_score(y, s))


def bootstrap(y, s, fn, n=1000, seed=0):
    """Point estimate and 95% percentile interval, resampling clips with replacement
    separately within each class so every resample keeps both classes."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    stats = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats.append(fn(y[idx], s[idx]))
    return float(fn(y, s)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

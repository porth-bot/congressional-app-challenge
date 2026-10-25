"""Pick which WavLM layer to use, by cross-validation on the training split only.

For each of the 13 layers, fit a logistic regression on ASVspoof train
(studio + G.711 + Opus copies) with 5 speaker-grouped folds and keep the layer
with the lowest out-of-fold EER. Then write that layer's embeddings as
data/features/ssl_<cond>.parquet so benchmark.py can use them like any other
feature set.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CONDITIONS, RESULTS, ROOT, TRAIN_CODECS, eer, manifest

FEAT = ROOT / "data" / "features"


def load(cond):
    z = np.load(FEAT / f"ssl_layers_{cond}.npz")
    return z["ids"], z["emb"].astype(np.float32)


def main():
    man = manifest().set_index("clip_id")
    parts = []
    for c in ["clean", *TRAIN_CODECS]:
        ids, emb = load(c)
        keep = np.array([man.at[i, "corpus"] == "asv_train" for i in ids])
        parts.append((ids[keep], emb[keep]))
    ids = np.concatenate([p[0] for p in parts])
    emb = np.concatenate([p[1] for p in parts])
    y = man.loc[ids, "y"].to_numpy()
    groups = man.loc[ids, "speaker"].to_numpy()
    scores = {}
    for layer in range(emb.shape[1]):
        X = emb[:, layer]
        oof = np.zeros(len(y))
        for a, b in GroupKFold(5).split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[a], y[a])
            oof[b] = m.predict_proba(X[b])[:, 1]
        scores[layer] = eer(y, oof)[0]
        print(f"layer {layer:2d}  CV EER {scores[layer]:.4f}", flush=True)
    best = min(scores, key=scores.get)
    print("best layer", best)
    json.dump({"model": "microsoft/wavlm-base-plus", "cv_eer_by_layer": scores, "best_layer": best},
              open(RESULTS / "ssl_layer_selection.json", "w"), indent=1)
    for c in CONDITIONS:
        ids, emb = load(c)
        df = pd.DataFrame(emb[:, best], columns=[f"ssl_{k}" for k in range(emb.shape[2])])
        df.insert(0, "clip_id", ids)
        df.to_parquet(FEAT / f"ssl_{c}.parquet", index=False)


if __name__ == "__main__":
    sys.exit(main())

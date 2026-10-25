"""Two checks a reviewer asked for on the Stage 1 headline numbers.

1. Recalibration control. Is the drop in false alarms from phone training just a
   better cut-off? Take the studio-trained WavLM model and set its cut-off on
   out-of-fold scores of the PHONE copies (landline + VoIP) of the training clips:
   for each speaker-grouped fold, train on the studio copies of the other folds and
   score this fold's phone copies. No test data is used.

2. Speaker-clustered intervals. Test clips share speakers (and, across channels,
   the same recordings), so resampling clips makes intervals too narrow. Here we
   resample speakers, keeping all of a speaker's clips together.

Writes results/cluster_ci.json.
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
N_BOOT = 1000


def ssl(cond):
    df = pd.read_parquet(FEAT / f"ssl_{cond}.parquet")
    return manifest().merge(df, on="clip_id")


def recal_threshold():
    cols = None
    clean = ssl("clean")
    clean = clean[clean.corpus == "asv_train"].reset_index(drop=True)
    cols = [c for c in clean.columns if c.startswith("ssl_")]
    phone = pd.concat([ssl(c) for c in TRAIN_CODECS])
    phone = phone[phone.corpus == "asv_train"].reset_index(drop=True)
    oof_s, oof_y = [], []
    for a, b in GroupKFold(5).split(clean, clean.y, clean.speaker):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        m.fit(clean.loc[a, cols].to_numpy(float), clean.y[a])
        held = set(clean.speaker[b])
        ph = phone[phone.speaker.isin(held)]
        oof_s.append(m.predict_proba(ph[cols].to_numpy(float))[:, 1])
        oof_y.append(ph.y.to_numpy())
    return eer(np.concatenate(oof_y), np.concatenate(oof_s))[1]


def rates(y, s, thr):
    pred = s >= thr
    fa = pred[y == 0].mean()
    miss = 1 - pred[y == 1].mean()
    return fa, miss, 1 - (fa + miss) / 2  # balanced accuracy


def cluster_boot(df, fn, rng):
    spk = df.speaker.unique()
    groups = {s: g.index.to_numpy() for s, g in df.groupby("speaker")}
    stats = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[s] for s in rng.choice(spk, len(spk))])
        d = df.loc[idx]
        if d.y.nunique() < 2:
            continue
        stats.append(fn(d))
    stats = np.array(stats)
    return [float(x) for x in np.percentile(stats, [2.5, 97.5], axis=0).T.ravel()] if stats.ndim > 1 else \
        [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def main():
    rng = np.random.default_rng(0)
    bench = json.load(open(RESULTS / "benchmark_ssl.json"))["runs"]
    scores = pd.read_parquet(ROOT / "data" / "benchmark_ssl_scores.parquet")
    man = manifest().set_index("clip_id")
    thr_recal = recal_threshold()
    models = {"studio_trained": ("ssl_clean", bench["ssl_clean"]["threshold"]),
              "studio_trained_recalibrated": ("ssl_clean", thr_recal),
              "phone_trained": ("ssl_aug", bench["ssl_aug"]["threshold"])}
    out = {"recalibrated_threshold": thr_recal, "n_boot": N_BOOT, "unit": "speaker", "results": {}}
    for corpus in ["asv_test", "itw"]:
        for cond in CONDITIONS:
            for name, (model, thr) in models.items():
                d = scores[(scores.model == model) & (scores.corpus == corpus) & (scores.cond == cond)].copy()
                d["speaker"] = man.loc[d.clip_id, "speaker"].to_numpy()
                d = d.reset_index(drop=True)
                y, s = d.y.to_numpy(), d.score.to_numpy()
                fa, miss, bacc = rates(y, s, thr)
                ci = cluster_boot(d, lambda x: rates(x.y.to_numpy(), x.score.to_numpy(), thr) + (eer(x.y, x.score)[0],), rng)
                out["results"][f"{corpus}/{cond}/{name}"] = {
                    "false_alarm": float(fa), "false_alarm_ci": ci[0:2], "miss": float(miss), "miss_ci": ci[2:4],
                    "balanced_accuracy": float(bacc), "balanced_accuracy_ci": ci[4:6],
                    "eer": eer(y, s)[0], "eer_ci": ci[6:8]}
            # paired difference in balanced accuracy, phone-trained minus studio-trained (same clips)
            a = scores[(scores.model == "ssl_aug") & (scores.corpus == corpus) & (scores.cond == cond)].set_index("clip_id")
            b = scores[(scores.model == "ssl_clean") & (scores.corpus == corpus) & (scores.cond == cond)].set_index("clip_id")
            j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner").reset_index()
            j["speaker"] = man.loc[j.clip_id, "speaker"].to_numpy()
            j["y"] = j.y_a

            def diff(x, ta=models["phone_trained"][1], tb=models["studio_trained"][1]):
                return (rates(x.y.to_numpy(), x.score_a.to_numpy(), ta)[2] - rates(x.y.to_numpy(), x.score_b.to_numpy(), tb)[2],
                        eer(x.y, x.score_a)[0] - eer(x.y, x.score_b)[0])
            point = diff(j)
            ci = cluster_boot(j, diff, rng)
            out["results"][f"{corpus}/{cond}/phone_minus_studio"] = {
                "balanced_accuracy_diff": float(point[0]), "balanced_accuracy_diff_ci": ci[0:2],
                "eer_diff": float(point[1]), "eer_diff_ci": ci[2:4]}
            print(corpus, cond, {k.split("/")[-1]: {kk: round(vv, 3) if isinstance(vv, float) else [round(z, 3) for z in vv]
                                                      for kk, vv in v.items()}
                                 for k, v in out["results"].items() if k.startswith(f"{corpus}/{cond}/")}, flush=True)
    json.dump(out, open(RESULTS / "cluster_ci.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())

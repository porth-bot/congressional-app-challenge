"""Phone-channel benchmark for Stage 1 (the voice model).

Every model is trained on the ASVspoof 2019 LA train subset only, then tested on
  - ASVspoof 2019 eval: new speakers and 13 attack systems never seen in training
  - In-the-Wild: real and fake celebrity / politician audio from the internet
each in four channels: studio, landline G.711, VoIP Opus, and G.723.1. G.723.1 is
never used in training, so it doubles as an unseen-codec test.

Two kinds of numbers:
  - EER and AUC: threshold-free, how well the scores rank real vs fake
  - false-alarm and miss rates at a FIXED threshold chosen on training data. This
    is what a deployed app would do, and it shows calibration drift: a threshold
    that worked on studio audio can flag real family voices once the codec hits.

Writes results/benchmark_<featureset>.json and prints a summary table.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar

from common import CONDITIONS, RESULTS, TRAIN_CODECS, auc, bootstrap, eer, load
from features import BIO, SPECTRAL


def feature_sets(cols):
    ceps = [c for c in cols if c.startswith(("mfcc", "dmfcc", "lfcc"))]
    surv = json.load(open(RESULTS / "survival.json"))
    return {
        "bio": BIO,
        "bio_spectral": BIO + SPECTRAL,
        "cepstral": ceps,
        "all": BIO + SPECTRAL + ceps,
        "survivors": surv["survivors"],
        "survivors_bio_spectral": surv["survivors_bio_spectral"],
    }


def make_model(kind):
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              l2_regularization=1.0, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def fit_eval(name, X_cols, kind, train_conds, data, verbose=True):
    """Train on ASVspoof train in train_conds, test everywhere."""
    tr = pd.concat([data[c][data[c].corpus == "asv_train"] for c in train_conds])
    X, y = tr[X_cols].to_numpy(float), tr.y.to_numpy()
    if kind == "lr":
        med = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), med, X)
    # out-of-fold scores, grouped by speaker so a voice never sits on both sides
    oof = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, tr.speaker):
        m = make_model(kind).fit(X[a], y[a])
        oof[b] = m.predict_proba(X[b])[:, 1]
    cv_eer, thr = eer(y, oof)
    model = make_model(kind).fit(X, y)
    decisions = {("asv_train_oof", "+".join(train_conds)): (tr.clip_id.to_numpy() + "|" + tr.cond.to_numpy(), y, (oof >= thr).astype(int), oof)}
    res = {"name": name, "features": len(X_cols), "model": kind, "train_conditions": train_conds,
           "cv_eer": cv_eer, "threshold": thr, "test": {}}
    for corpus in ["asv_test", "itw"]:
        res["test"][corpus] = {}
        for c in CONDITIONS:
            d = data[c][data[c].corpus == corpus]
            Xt = d[X_cols].to_numpy(float)
            if kind == "lr":
                Xt = np.where(np.isnan(Xt), med, Xt)
            s = model.predict_proba(Xt)[:, 1]
            yt = d.y.to_numpy()
            e = bootstrap(yt, s, lambda a, b: eer(a, b)[0])
            u = bootstrap(yt, s, auc)
            pred = (s >= thr).astype(int)
            res["test"][corpus][c] = {
                "n": int(len(yt)),
                "eer": e, "auc": u,
                "false_alarm": float(pred[yt == 0].mean()),  # real voice flagged as fake
                "miss": float(1 - pred[yt == 1].mean()),     # fake voice let through
            }
            decisions[(corpus, c)] = (d.clip_id.to_numpy(), yt, pred, s)
    if verbose:
        print(f"\n{name}  [{kind}, {len(X_cols)} feats, train={'+'.join(train_conds)}]  CV EER {cv_eer:.3f}")
        for corpus in ["asv_test", "itw"]:
            row = "  ".join(f"{c}:{res['test'][corpus][c]['eer'][0]:.3f}/FA{res['test'][corpus][c]['false_alarm']:.2f}"
                            for c in CONDITIONS)
            print(f"  {corpus:9s} EER/false-alarm  {row}")
    return res, decisions


def compare(da, db):
    """Exact McNemar on correct/incorrect at each model's own fixed threshold."""
    out = {}
    for key in da:
        if key[0] == "asv_train_oof":
            continue
        ida, y, pa, _ = da[key]
        idb, _, pb, _ = db[key]
        assert (ida == idb).all()
        ca, cb = pa == y, pb == y
        tab = [[int((ca & cb).sum()), int((ca & ~cb).sum())], [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
        p = mcnemar(tab, exact=True).pvalue
        out[f"{key[0]}/{key[1]}"] = {"acc_a": float(ca.mean()), "acc_b": float(cb.mean()), "table": tab, "p": float(p)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="extra feature parquet prefixes, e.g. ssl")
    args = ap.parse_args()
    data = {c: load(c) for c in CONDITIONS}
    for pre in args.extra:
        for c in CONDITIONS:
            e = pd.read_parquet(RESULTS.parent / "data" / "features" / f"{pre}_{c}.parquet")
            data[c] = data[c].merge(e, on="clip_id")
    fs = feature_sets(data["clean"].columns)
    for pre in args.extra:
        fs[pre] = [k for k in data["clean"].columns if k.startswith(pre + "_")]

    aug = ["clean", *TRAIN_CODECS]
    runs = [
        ("cepstral_clean", fs["cepstral"], "gbm", ["clean"]),
        ("all_clean", fs["all"], "gbm", ["clean"]),
        ("bio_spectral_clean", fs["bio_spectral"], "gbm", ["clean"]),
        ("cepstral_aug", fs["cepstral"], "gbm", aug),
        ("all_aug", fs["all"], "gbm", aug),
        ("bio_spectral_aug", fs["bio_spectral"], "gbm", aug),
        ("survivors_aug", fs["survivors"], "gbm", aug),
        ("survivors_bio_spectral_aug", fs["survivors_bio_spectral"], "gbm", aug),
    ]
    for pre in args.extra:
        runs += [(f"{pre}_clean", fs[pre], "lr", ["clean"]), (f"{pre}_aug", fs[pre], "lr", aug)]
    results, decs = {}, {}
    for name, cols, kind, conds in runs:
        if not cols:
            print("skip", name, "(no features)")
            continue
        results[name], decs[name] = fit_eval(name, cols, kind, conds, data)
    pairs = [("survivors_aug", "cepstral_clean"), ("survivors_aug", "all_aug"), ("all_aug", "all_clean"),
             ("cepstral_aug", "cepstral_clean")]
    pairs += [(f"{p}_aug", f"{p}_clean") for p in args.extra]
    pairs += [(f"{p}_aug", "survivors_aug") for p in args.extra]
    mc = {f"{a} vs {b}": compare(decs[a], decs[b]) for a, b in pairs if a in decs and b in decs}
    tag = "_".join(["benchmark", *args.extra])
    json.dump({"runs": results, "mcnemar": mc}, open(RESULTS / f"{tag}.json", "w"), indent=1)
    # per-clip scores, for fusion and the app's examples
    rows = []
    for name, d in decs.items():
        for (corpus, c), (ids, y, pred, s) in d.items():
            rows += [{"model": name, "corpus": corpus, "cond": c, "clip_id": i, "y": int(t), "score": float(v)}
                     for i, t, v in zip(ids, y, s)]
    pd.DataFrame(rows).to_parquet(RESULTS.parent / "data" / f"{tag}_scores.parquet", index=False)
    print("\nwrote", RESULTS / f"{tag}.json")


if __name__ == "__main__":
    sys.exit(main())

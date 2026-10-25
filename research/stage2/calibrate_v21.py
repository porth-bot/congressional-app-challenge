"""Stage 2 version 2.1: the same model, calibrated with how much has been heard.

Version 2's calibration layer was one Platt fit over 10%..100% prefixes pooled.
TF-IDF vectors are length-normalised, so three words like "bail money tonight"
can produce as large a raw score z as a whole call, and the single Platt slope
had to stay low (a = 0.59) to cover both. A full, blatant call then landed near
p = 0.6. Version 2.1 lets confidence grow with the number of caller words heard:

    u = z - z0                                          z0 = the score of an empty transcript
    p = 1 / (1 + exp(-(a*u + d*u*ln(1 + n) + b)))      n = words in the normalised caller text

Measuring z from the empty-transcript score z0 means that before the caller has
said anything, p is the validation prior: no words, no evidence. There is no
term in n alone, so length by itself (which leaks the label a little in the
family calls) cannot move the score. The model weights are untouched.

Three calibration recipes are compared on every test set:
  v2          version 2's recipe refit here (z only, shakeleoatmeal/validation +
              held-out family scenarios, equal weight per source)
  v21_pooled  the length-aware formula on the same pooled validation data
  v21_family  the length-aware formula on the held-out family scenarios only
The app ships v21_family. Liveline screens calls that claim to be family, so it
is calibrated on validation calls of that kind; the pooled recipes are pulled
toward caution by shakeleoatmeal, whose real refund and Social Security calls
sound like scams. The switch was prompted by version 2 raising no alarms on the
family test set, so that test set is no longer a clean holdout for this version;
its family numbers are development results. The calibration itself is fit only
on validation data. Thresholds follow version 2's rule on each recipe's own validation
sources (version 2 also checked the training pool's out-of-fold scores, which
never set a threshold; they are not recomputed here).

Writes model.json (calibration + thresholds), test_scores.csv, val_scores.csv
(cross-fitted, for tuning the app's fused cut-offs), results_v21.json and
research/data/stage2/parity_cases_v21.json.

    cd research/stage2 && ../.venv/bin/python calibrate_v21.py && node parity_v21.js
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from common import DATA, HERE, analyzer, normalize, prefix_caller, tag_tactics
from train import FAR_RED, FAR_YELLOW, FRACS10, GRID, SEED, auc_fast, brier, build_splits, ece, eer, with_ci

VALS = ("val", "val_family")
RECIPES = {"v2": (("val", "val_family"), "z"), "v21_pooled": (("val", "val_family"), "len"),
           "v21_family": (("val_family",), "len")}
SHIP = "v21_family"
TESTS = ("test_in", "shake_test", "shake_trainplustest", "menaattia_test", "menaattia_all", "youtube", "family_test")
HARD_NEG = {"XN03", "XN04", "XN05", "XN06", "XN07"}  # family test hard negatives (see RESULTS.md table)


def raw_scores(m, df, fracs):
    """Raw model score z and caller word count n at each prefix, with the same
    math as stage2_model.js."""
    names, greets = set(m["tokenizer"]["name_tokens"]), set(m["tokenizer"]["greeting_tokens"])
    idx = {t: i for i, t in enumerate(m["vocab"])}
    idf, coef = np.array(m["idf"]), np.array(m["coef"])
    Z = np.zeros((len(df), len(fracs)))
    N = np.zeros_like(Z)
    for c, f in enumerate(fracs):
        for r, t in enumerate(df.turns):
            clean = normalize(prefix_caller(t, f), True, names, greets)
            cnt = {}
            for g in analyzer(clean):
                j = idx.get(g)
                if j is not None:
                    cnt[j] = cnt.get(j, 0) + 1
            z = m["intercept"]
            if cnt:
                cols = np.array(list(cnt))
                v = (1 + np.log(np.array(list(cnt.values()), float))) * idf[cols]
                z += float((v / np.sqrt((v * v).sum())) @ coef[cols])
            Z[r, c] = z
            N[r, c] = len(clean.split())
    return Z, N


Z0 = None  # raw score of an empty transcript (the model's intercept), set in main()


def feats(Z, N, kind):
    z = np.asarray(Z, float).ravel()
    if kind == "z":
        return z[:, None]
    u = z - Z0
    return np.column_stack([u, u * np.log1p(np.asarray(N, float).ravel())])


def fit(parts, kind):
    X, Y, W = [], [], []
    for Z, N, y in parts:
        yy = np.repeat(np.asarray(y)[:, None], Z.shape[1], axis=1).ravel()
        X.append(feats(Z, N, kind))
        Y.append(yy)
        W.append(np.full(len(yy), 1.0 / len(yy)))
    return LogisticRegression(C=1e6, max_iter=10000).fit(np.vstack(X), np.concatenate(Y), sample_weight=np.concatenate(W))


def prob(lr, Z, N, kind):
    return lr.predict_proba(feats(Z, N, kind))[:, 1].reshape(np.shape(Z))


def halves(S):
    out = {}
    for n in VALS:
        df = S[n]
        if n == "val_family":
            k = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED)
            out[n] = list(k.split(np.zeros(len(df)), df.label.values, df.type.values))
        else:
            k = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
            out[n] = list(k.split(np.zeros(len(df)), df.label.values))
    return out


def crossfit(S, ZN, H, recipe):
    """Each validation half scored by a calibrator fit on the other halves."""
    srcs, kind = RECIPES[recipe]
    out = {n: np.zeros_like(ZN[n][0]) for n in VALS}
    for h in range(2):
        lr = fit([(ZN[n][0][H[n][h][0]], ZN[n][1][H[n][h][0]], S[n].label.values[H[n][h][0]]) for n in srcs], kind)
        for n in VALS:
            b = H[n][h][1]
            out[n][b] = prob(lr, ZN[n][0][b], ZN[n][1][b], kind)
    return out


def thr_for(max_p, alpha):
    m_ = np.sort(np.asarray(max_p))[::-1]
    k = int(math.floor(alpha * len(m_)))
    return 0.0 if k >= len(m_) else (math.floor(m_[k] * 1000) + 1) / 1000.0


def main():
    rng = np.random.default_rng(SEED)
    global Z0
    m = json.load(open(os.path.join(HERE, "model.json")))
    Z0 = float(m["intercept"])
    S, _ = build_splits()
    g10 = [GRID.index(f) for f in FRACS10]
    ZN = {n: raw_scores(m, S[n], GRID) for n in set(VALS) | set(TESTS)}
    H = halves(S)
    res = {"version": "2.1", "formula": "p = 1/(1+exp(-(a*u + d*u*ln(1+n) + b))), u = z - z0", "z0": Z0,
           "calibration_quality": {}}

    # calibration quality, cross-fitted, old vs new, by prefix bucket
    buckets = {"10-30%": [0, 1, 2], "40-70%": [3, 4, 5, 6], "80-100%": [7, 8, 9]}
    for kind in RECIPES:
        P = crossfit(S, {n: (ZN[n][0][:, g10], ZN[n][1][:, g10]) for n in VALS}, H, kind)
        q = {}
        for n in VALS:
            y = S[n].label.values
            for bn, cols in buckets.items():
                yy = np.repeat(y[:, None], len(cols), axis=1).ravel()
                pp = P[n][:, cols].ravel()
                q[f"{n} {bn}"] = {"brier": brier(yy, pp), "ece": ece(yy, pp), "mean_p_scam": float(pp[yy == 1].mean()),
                                  "mean_p_normal": float(pp[yy == 0].mean())}
        res["calibration_quality"][kind] = q

    # final calibrators, fit on all validation rows
    cal, KIND = {}, {r: RECIPES[r][1] for r in RECIPES}
    for r, (srcs, kind) in RECIPES.items():
        cal[r] = fit([(ZN[n][0][:, g10], ZN[n][1][:, g10], S[n].label.values) for n in srcs], kind)
    srcs = RECIPES[SHIP][0]
    prior = float(np.mean([S[n].label.mean() for n in srcs]))
    lp = math.log(prior / (1 - prior))
    a, d = (float(x) for x in cal[SHIP].coef_[0])
    b = float(cal[SHIP].intercept_[0])
    res["calibration"] = {r: {"coef": [float(x) for x in cal[r].coef_[0]], "intercept": float(cal[r].intercept_[0]),
                              "sources": list(RECIPES[r][0]), "kind": RECIPES[r][1]} for r in RECIPES}
    res["calibration"]["shipped"] = {"recipe": SHIP, "a": a, "d": d, "b": b, "prior": prior}
    res["calibration"]["v2_refit_check"] = {"refit": res["calibration"]["v2"],
                                            "model_json_v2": {"a": m["calibration"]["a"], "b": m["calibration"]["b"]}}

    # thresholds: same rule as version 2, on the two validation sources
    th = {}
    for kind in RECIPES:
        ever = {n: prob(cal[kind], *[x[S[n].label.values == 0] for x in ZN[n]], KIND[kind]).max(axis=1)
                for n in RECIPES[kind][0]}
        t = {}
        for nm, alpha in (("yellow", FAR_YELLOW), ("red", FAR_RED)):
            per = {src: thr_for(v, alpha) for src, v in ever.items()}
            t[nm] = max(per.values())
            t.update({f"{nm}_from_{src}": v for src, v in per.items()})
        th[kind] = t
    res["thresholds"] = th
    print("thresholds", json.dumps(th))

    # test evaluation, both calibrations
    res["evaluation"] = {}
    for key in TESTS:
        df, y = S[key], S[key].label.values
        r = {"n": int(len(y)), "n_scam": int(y.sum())}
        for kind in RECIPES:
            P = prob(cal[kind], ZN[key][0], ZN[key][1], KIND[kind])
            pf, ev = P[:, -1], P.max(axis=1)
            e = {}
            if 0 < y.sum() < len(y):
                e["auc_full"] = with_ci(lambda ix: auc_fast(y[ix], pf[ix]), y, rng)
                e["eer_full"] = with_ci(lambda ix: eer(y[ix], pf[ix]), y, rng)
            for tn in ("yellow", "red"):
                t = th[kind][tn]
                if y.sum():
                    e[f"scams_ever_{tn}"] = float(np.mean(ev[y == 1] >= t))
                if (y == 0).any():
                    e[f"normal_ever_{tn}"] = float(np.mean(ev[y == 0] >= t))
            if key == "family_test":
                hard = (y == 0) & df.type.isin(HARD_NEG).values
                e["hard_negatives"] = int(hard.sum())
                for tn in ("yellow", "red"):
                    e[f"hard_negatives_ever_{tn}"] = float(np.mean(ev[hard] >= th[kind][tn]))
            if y.sum():
                e["median_p_full_scam"] = float(np.median(pf[y == 1]))
            if (y == 0).any():
                e["median_p_full_normal"] = float(np.median(pf[y == 0]))
            r[kind] = e
        res["evaluation"][key] = r
        print(key, {k: {kk: (round(vv["value"], 3) if isinstance(vv, dict) else round(vv, 3)) for kk, vv in v.items()}
                    for k, v in r.items() if k in RECIPES})

    # exports: per-dialogue streaming scores (test and cross-fitted validation)
    def table(key, P):
        df = S[key]
        src = ["family_test"] * len(df) if key == "family_test" else (
            ["family_val"] * len(df) if key == "val_family" else list(df.source.values))
        t = pd.DataFrame({"source": src, "dialogue_id": [f"{s_}/{sp_}/{r}" for s_, sp_, r in zip(df.source, df.split, df.row)],
                          "label": df.label.values, "n_caller_words": ZN[key][1][:, -1].astype(int)})
        for c, f in enumerate(FRACS10):
            t[f"p_{int(round(f * 100))}"] = P[:, c]
        for c, f in enumerate(FRACS10):
            t[f"llr_{int(round(f * 100))}"] = np.log(P[:, c] / (1 - P[:, c])) - lp
        return t

    pd.concat([table(k, prob(cal[SHIP], ZN[k][0][:, g10], ZN[k][1][:, g10], KIND[SHIP]))
               for k in ("test_in", "shake_test", "menaattia_test", "family_test")]).to_csv(
        os.path.join(HERE, "test_scores.csv"), index=False, float_format="%.10g")
    PX = crossfit(S, {n: (ZN[n][0][:, g10], ZN[n][1][:, g10]) for n in VALS}, H, SHIP)
    pd.concat([table(n, PX[n]) for n in VALS]).to_csv(os.path.join(HERE, "val_scores.csv"), index=False, float_format="%.10g")

    # model.json: new calibration and thresholds, same weights
    m["format"] = "liveline-stage2-text-v2.1"
    m["calibration"] = {"method": "length-aware platt", "a": a, "d": d, "c": 0.0, "b": b, "z0": Z0,
                        "fit_on": "held-out family scenarios (validation), 10%..100% prefixes",
                        "formula": "p = 1 / (1 + exp(-(a * u + d * u * ln(1 + n) + b))), u = z - z0, "
                                   "z = coef . x + intercept, z0 = intercept (score of an empty transcript), "
                                   "n = number of words in the normalised caller text"}
    m["prior"] = prior
    m["thresholds"] = {"yellow": th[SHIP]["yellow"], "red": th[SHIP]["red"]}
    json.dump(m, open(os.path.join(HERE, "model.json"), "w"), separators=(",", ":"))

    # parity cases for the JavaScript port
    cases = []
    for key in ("family_test", "menaattia_test", "shake_test", "test_in"):
        df = S[key]
        for r in rng.choice(len(df), size=min(150, len(df)), replace=False):
            for f in (0.3, 0.6, 1.0):
                text = prefix_caller(df.turns.iloc[r], f)
                z, n = raw_scores(m, df.iloc[[r]], [f])
                u = z[0, 0] - Z0
                s_ = a * u + d * u * math.log1p(n[0, 0]) + b
                cases.append({"text": text, "p": 1 / (1 + math.exp(-s_))})
    json.dump(cases, open(os.path.join(DATA, "parity_cases_v21.json"), "w"))
    res["parity_cases"] = len(cases)
    json.dump(res, open(os.path.join(HERE, "results_v21.json"), "w"), indent=1)
    print("calibration", res["calibration"])


if __name__ == "__main__":
    sys.exit(main())

"""Stage 2 of Liveline (version 2): a scam-tactic text model that scores a call transcript.

One command reruns everything (fixed seeds):

    cd research/stage2 && ../.venv/bin/python train.py

Version 2 adds family-emergency ("grandparent") calls written by AI
(research/data/stage2/family/, see RESULTS.md) and an optional
"tactic count" input. Version 1 lives in v1/.

Steps
  1. Load the public sources plus the family calls, find near-duplicates,
     build the splits.
  2. Choose (variant, C, family weight) on the two validation sources only,
     by the rule stored in results.json["selection_rule"].
  3. Calibrate on the pooled validation sources, set yellow/red thresholds.
  4. Evaluate once on the test sets, with bootstrap 95% CIs and streaming.
  5. Run the version 1 model on the family sets for a before / after.
  6. Export model.json, results.json, test_scores.csv, streaming_curve.png
     and the parity cases.

Splits
  train pool      BothBosu train splits (one per near-duplicate cluster) +
                  family TRAIN batch minus the held-out scenarios
  validation      shakeleoatmeal/validation + held-out family scenarios
                  (TS03, TS06, TS09, TS11, TN03, TN06, TN09, TN11)
  tests           BothBosu test (de-duplicated), shakeleoatmeal train+test,
                  menaattia, youtube, family TEST batch (separate writers,
                  scenarios and names)
"""

import json
import math
import os
import platform
import re
import time

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from common import (GREETING_TOKENS, HERE, NAME_TOKENS, TACTICS, analyzer, caller_text,
                    full_text, load_all, normalize, parse_turns, prefix_caller, tag_tactics,
                    DATA)
from neardup import components, near_pairs, shingle_matrix

SEED = 0
NEAR = 0.5                    # Jaccard (word 5-gram shingles) for near-duplicates
LOOSE = 0.3                   # stricter de-dup level, used for a sensitivity check
FRACS10 = [round(0.1 * k, 1) for k in range(1, 11)]        # streaming curve points
GRID = [round(0.02 * k, 2) for k in range(1, 51)]          # 2% steps for "ever red"
AUG = [0.25, 0.5, 0.75, 1.0]  # prefix augmentation fractions (variant C)
N_BOOT = 2000
FAR_YELLOW, FAR_RED = 0.10, 0.02
BB = ["scam_dialogue", "single_agent", "multi_agent"]

if os.environ.get("STAGE2_SMOKE"):   # quick end-to-end check of the code, not for results
    N_BOOT = 20
    GRID = list(FRACS10)
    C_GRID2 = [1.0]
    FAM_W_GRID = [1]

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def char_analyzer(text):
    """Character 3-5 grams inside each word padded with spaces (" gift ")."""
    out = []
    for w in text.split():
        w = " " + w + " "
        for n in (3, 4, 5):
            for i in range(len(w) - n + 1):
                out.append(w[i:i + n])
    return out


class TextModel:
    """TF-IDF (word 1-2 grams, optional char 3-5 grams) + logistic regression."""

    def __init__(self, word_max, char_max=0, C=1.0):
        self.word_max, self.char_max, self.C = word_max, char_max, C

    def fit(self, vocab_texts, train_texts, y, sample_weight=None):
        # Vocabulary and idf come from full-length training dialogues only,
        # so prefix copies (variant C) do not inflate document frequencies.
        self.wv = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, min_df=2,
                                  max_features=self.word_max, dtype=np.float64)
        self.wv.fit(vocab_texts)
        if self.char_max:
            self.cv = TfidfVectorizer(analyzer=char_analyzer, sublinear_tf=True, min_df=2,
                                      max_features=self.char_max, dtype=np.float64)
            self.cv.fit(vocab_texts)
        self.clf = LogisticRegression(C=self.C, max_iter=5000)
        self.clf.fit(self.transform(train_texts), y, sample_weight=sample_weight)
        return self

    def transform(self, texts):
        X = self.wv.transform(texts)
        if self.char_max:
            X = sp.hstack([X, self.cv.transform(texts)]).tocsr()
        return X

    def decision(self, texts):
        return self.clf.decision_function(self.transform(texts))

    def n_features(self):
        return len(self.wv.vocabulary_) + (len(self.cv.vocabulary_) if self.char_max else 0)


# Version 2 model choice. Version 1 already showed prefix augmentation and
# character n-grams were worse on validation, so the grid is word n-grams only.
VARIANTS2 = ["A_word12", "D_word12_plus_tactic_count"]
C_GRID2 = [0.1, 0.3, 1.0, 3.0]
FAM_W_GRID = [1, 3]      # weight of each family-call training row vs a BothBosu row


# ---------------------------------------------------------------------------
# Data and splits
# ---------------------------------------------------------------------------

def build_splits():
    df = load_all()
    df["turns"] = [parse_turns(d) for d in df.dialogue]
    X = shingle_matrix(df.dialogue.tolist())
    i, j, s = near_pairs(X, threshold=LOOSE)
    k = s >= NEAR
    df["cluster"] = components(len(df), i[k], j[k])
    df["cluster_loose"] = components(len(df), i, j)

    is_bb = df.source.isin(BB)
    tr_all = df[is_bb & (df.split == "train")]
    train = tr_all.drop_duplicates("cluster").reset_index(drop=True)
    te_all = df[is_bb & (df.split == "test")]
    te = te_all[~te_all.cluster.isin(set(tr_all.cluster))]
    test_in = te.drop_duplicates("cluster").reset_index(drop=True)
    # Stricter version: also drop test dialogues sharing a looser (>= 0.3)
    # template cluster with any training dialogue.
    test_in_strict = test_in[~test_in.cluster_loose.isin(set(tr_all.cluster_loose))].reset_index(drop=True)

    # Overlap of every evaluation source with the training pool at >= 0.3
    tr_idx = set(np.where(is_bb & (df.split == "train"))[0])
    ext_overlap = {}
    for name in ("shakeleoatmeal", "menaattia", "youtube"):
        idx = set(np.where(df.source == name)[0])
        hits = {a for a, b in zip(i, j) if a in idx and b in tr_idx} | \
               {b for a, b in zip(i, j) if b in idx and a in tr_idx}
        ext_overlap[name] = len(hits)

    fam = df[df.source == "family"]
    fam_tr = fam[fam.split == "train"]
    train = pd.concat([train, fam_tr], ignore_index=True)
    tr_fam_idx = set(fam_tr.index)
    idx = set(fam[fam.split == "test"].index)
    ext_overlap["family_test_vs_family_train"] = len(
        {a for a, b in zip(i, j) if a in idx and b in tr_fam_idx} | {b for a, b in zip(i, j) if b in idx and a in tr_fam_idx})

    sh = df[df.source == "shakeleoatmeal"]
    me = df[df.source == "menaattia"]
    splits = {
        "train": train,
        "val_family": fam[fam.split == "validation"].reset_index(drop=True),
        "family_test": fam[fam.split == "test"].reset_index(drop=True),
        "test_in": test_in,
        "test_in_strict": test_in_strict,
        "val": sh[sh.split == "validation"].reset_index(drop=True),
        "shake_test": sh[sh.split == "test"].reset_index(drop=True),
        "shake_trainplustest": sh[sh.split.isin(["train", "test"])].reset_index(drop=True),
        "menaattia_all": me.reset_index(drop=True),
        "menaattia_test": me[me.split == "test"].reset_index(drop=True),
        "youtube": df[df.source == "youtube"].reset_index(drop=True),
    }
    info = {
        "near_duplicate_threshold_jaccard_5gram": NEAR,
        "bothbosu_train_dialogues_before_dedup": int(len(tr_all)),
        "train_pool_after_dedup": int(len(train)),
        "of_which_family_train": int(len(fam_tr)),
        "bothbosu_test_dialogues_before_dedup": int(len(te_all)),
        "test_removed_as_near_copy_of_train": int(len(te_all) - len(te)),
        "test_removed_as_within_test_duplicate": int(len(te) - len(test_in)),
        "in_domain_test_after_dedup": int(len(test_in)),
        "in_domain_test_strict_jaccard_0.3": int(len(test_in_strict)),
        "external_dialogues_with_any_train_neighbor_jaccard_0.3": ext_overlap,
    }
    return splits, info


def texts_at(turns_list, frac, mode="caller", neutral=True):
    if mode == "caller":
        return [normalize(prefix_caller(t, frac), neutral) for t in turns_list]
    # both speakers (leakage ablation only)
    out = []
    for t in turns_list:
        words = [w for _, x in t for w in x.split()]
        keep = int(math.ceil(frac * len(words) - 1e-9))
        out.append(normalize(" ".join(words[:keep]), neutral))
    return out


def aug_rows(df, fracs, mode="caller", neutral=True):
    texts, ys, groups = [], [], []
    for f in fracs:
        texts += texts_at(df.turns, f, mode, neutral)
        ys += list(df.label)
        groups += list(df.cluster)
    return texts, np.array(ys), np.array(groups)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auc_fast(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def ece(y, p, bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def boot_idx(y, rng, n=N_BOOT):
    """Stratified bootstrap: resample scams and non-scams separately."""
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n):
        parts = []
        if len(pos):
            parts.append(rng.choice(pos, len(pos), replace=True))
        if len(neg):
            parts.append(rng.choice(neg, len(neg), replace=True))
        yield np.concatenate(parts)


def ci(vals):
    v = np.asarray([x for x in vals if not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return [None, None]
    # Linear interpolation is undefined between inf values ("never reached
    # red"), so samples containing inf use the plain empirical quantile.
    method = "linear" if np.all(np.isfinite(v)) else "inverted_cdf"
    lo, hi = np.percentile(v, [2.5, 97.5], method=method)
    return [float(lo), float(hi)]


def with_ci(fn, y, rng, n=N_BOOT):
    """Point estimate plus percentile bootstrap CI of fn(indices)."""
    point = fn(np.arange(len(y)))
    boots = [fn(ix) for ix in boot_idx(y, rng, n)]
    return {"value": point, "ci95": ci(boots)}


def first_cross(P, t):
    """P: docs x GRID calibrated scores. Fraction heard when p first >= t (inf if never)."""
    hit = P >= t
    any_hit = hit.any(axis=1)
    first = np.argmax(hit, axis=1)
    out = np.array(GRID)[first].astype(float)
    out[~any_hit] = np.inf
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def fit_platt(z, y):
    lr = LogisticRegression(C=1e6, max_iter=10000)
    lr.fit(np.asarray(z).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt(z, a, b):
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(z) + b)))


def sig6(x):
    return float(f"{x:.6g}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED, "version": 2,
               "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                            "numpy": np.__version__, "scipy": scipy.__version__}}
    S, info = build_splits()
    results["splits"] = info
    results["split_sizes"] = {k: {"n": int(len(v)), "scam": int(v.label.sum()),
                                  "nonscam": int((v.label == 0).sum())} for k, v in S.items()}
    log("splits", json.dumps(results["split_sizes"]))
    log("dedup", json.dumps(info))

    train = S["train"]
    y_tr = train.label.values
    is_fam = (train.source == "family").values
    full_tr = texts_at(train.turns, 1.0)
    VALS = {"val": S["val"], "val_family": S["val_family"]}   # the two validation sources

    def weights(fam_w):
        return np.where(is_fam, float(fam_w), 1.0)

    def tactic_k(texts):
        return np.array([len(tag_tactics(t)) for t in texts], dtype=float)

    def scores_at(df, m, fracs):
        """docs x fracs raw model scores Z and tactic counts K."""
        Z = np.zeros((len(df), len(fracs))); K = np.zeros((len(df), len(fracs)))
        for c, f in enumerate(fracs):
            tx = texts_at(df.turns, f)
            Z[:, c] = m.decision(tx)
            K[:, c] = tactic_k(tx)
        return Z, K

    def feats(Z, K, variant):
        z = np.asarray(Z, float).ravel()
        if variant == "D_word12_plus_tactic_count":
            return np.column_stack([z, np.asarray(K, float).ravel()])
        return z.reshape(-1, 1)

    def fit_cal(parts, variant):
        """Logistic calibrator on pooled validation rows; each source gets equal total weight.
        parts: list of (Z, K, y) with Z, K docs x fracs."""
        X, Y, W = [], [], []
        for Z, K, y in parts:
            yy = np.repeat(np.asarray(y)[:, None], Z.shape[1], axis=1).ravel()
            X.append(feats(Z, K, variant)); Y.append(yy); W.append(np.full(len(yy), 1.0 / len(yy)))
        lr = LogisticRegression(C=1e6, max_iter=10000)
        lr.fit(np.vstack(X), np.concatenate(Y), sample_weight=np.concatenate(W))
        return lr

    def cal_halves(name, df):
        if name == "val_family":   # whole scenarios stay together
            k = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED)
            return list(k.split(np.zeros(len(df)), df.label.values, df.type.values))
        k = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
        return list(k.split(np.zeros(len(df)), df.label.values))

    halves = {n: cal_halves(n, d) for n, d in VALS.items()}

    def crossfit_probs(ZK, variant):
        """Calibrated validation probabilities where each half is scored by a
        calibrator fit on the other halves of both sources."""
        out = {n: np.zeros_like(ZK[n][0]) for n in VALS}
        for h in range(2):
            parts = [(ZK[n][0][halves[n][h][0]], ZK[n][1][halves[n][h][0]], VALS[n].label.values[halves[n][h][0]])
                     for n in VALS]
            lr = fit_cal(parts, variant)
            for n in VALS:
                b = halves[n][h][1]
                out[n][b] = lr.predict_proba(feats(ZK[n][0][b], ZK[n][1][b], variant))[:, 1].reshape(len(b), -1)
        return out

    # ---- 2. model choice, on validation only ---------------------------------
    comp = []
    for fam_w in FAM_W_GRID:
        for C in C_GRID2:
            m = TextModel(5000, 0, C).fit(full_tr, full_tr, y_tr, sample_weight=weights(fam_w))
            ZK = {n: scores_at(d, m, FRACS10) for n, d in VALS.items()}
            for variant in VARIANTS2:
                P = crossfit_probs(ZK, variant)
                per = {n: [auc_fast(VALS[n].label.values, P[n][:, c]) for c in range(len(FRACS10))] for n in VALS}
                r = {"variant": variant, "C": C, "family_weight": fam_w,
                     "val_auc_by_frac": per,
                     "val_auc_mean_over_fracs": {n: float(np.mean(v)) for n, v in per.items()},
                     "val_auc_full": {n: v[-1] for n, v in per.items()}}
                r["score"] = float(np.mean(list(r["val_auc_mean_over_fracs"].values())))
                comp.append(r)
                log(variant, "C", C, "fam_w", fam_w, "score %.4f" % r["score"],
                    {n: round(v, 4) for n, v in r["val_auc_mean_over_fracs"].items()},
                    "full", {n: round(v, 4) for n, v in r["val_auc_full"].items()})
    results["variant_comparison"] = comp
    top = max(r["score"] for r in comp)
    near_top = [r for r in comp if r["score"] >= top - 0.005]
    chosen = min(near_top, key=lambda r: (VARIANTS2.index(r["variant"]), r["family_weight"], r["C"]))
    results["selection_rule"] = (
        "over every (variant, C, family weight): mean over the two validation sources (shakeleoatmeal/validation "
        "and the held-out family scenarios) of the AUC averaged over 10%..100% prefixes, using cross-fitted "
        "calibrated probabilities; settings within 0.005 of the best are tied and go to the simplest "
        "(no tactic count, then lower family weight, then smallest C)")
    results["chosen_variant"] = {"variant": chosen["variant"], "C": chosen["C"], "family_weight": chosen["family_weight"]}
    VAR, C, FW = chosen["variant"], chosen["C"], chosen["family_weight"]
    log("CHOSEN", VAR, "C", C, "fam_w", FW)

    # ---- leakage ablations, validation only (raw model score AUC) ------------
    abl = {}
    for label, mode, neutral in (("chosen (caller words, names/greetings neutralized)", "caller", True),
                                 ("caller words, names/greetings NOT neutralized", "caller", False),
                                 ("both speakers (receiver lines included)", "full", True)):
        ft = texts_at(train.turns, 1.0, mode, neutral)
        m = TextModel(5000, 0, C).fit(ft, ft, y_tr, sample_weight=weights(FW))
        abl[label] = {}
        for n, d in VALS.items():
            v = [auc_fast(d.label, m.decision(texts_at(d.turns, f, mode, neutral))) for f in FRACS10]
            abl[label][n] = {"val_auc_full": v[-1], "val_auc_mean_over_fracs": float(np.mean(v))}
        log("ablation", label, abl[label])
    results["ablations_on_validation"] = abl

    # ---- 3. final model, OOF, calibration -------------------------------------
    model = TextModel(5000, 0, C).fit(full_tr, full_tr, y_tr, sample_weight=weights(FW))
    model.wv.idf_ = np.array([sig6(x) for x in model.wv.idf_])
    model.clf.coef_ = np.array([[sig6(x) for x in model.clf.coef_[0]]])
    model.clf.intercept_ = np.array([sig6(model.clf.intercept_[0])])

    def grid_scores(df, m=model):
        return scores_at(df, m, GRID)

    log("OOF grid scores (training pool) for thresholds")
    cvk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cvk.split(np.zeros(len(train)), y_tr, groups=train.cluster.values))
    oofZ = np.zeros((len(train), len(GRID))); oofK = np.zeros_like(oofZ)
    w_all = weights(FW)
    for tr_i, te_i in folds:
        m = TextModel(5000, 0, C).fit([full_tr[q] for q in tr_i], [full_tr[q] for q in tr_i], y_tr[tr_i],
                                      sample_weight=w_all[tr_i])
        oofZ[te_i], oofK[te_i] = grid_scores(train.iloc[te_i], m)
    VG = {n: grid_scores(d) for n, d in VALS.items()}
    g10 = [GRID.index(f) for f in FRACS10]

    lr_cal = fit_cal([(VG[n][0][:, g10], VG[n][1][:, g10], VALS[n].label.values) for n in VALS], VAR)
    A = sig6(float(lr_cal.coef_[0, 0]))
    Ck = sig6(float(lr_cal.coef_[0, 1])) if VAR == "D_word12_plus_tactic_count" else 0.0
    B = sig6(float(lr_cal.intercept_[0]))
    prior = float(np.mean([VALS[n].label.mean() for n in VALS]))   # equal source weights

    def to_p(Z, K):
        return 1.0 / (1.0 + np.exp(-(A * np.asarray(Z) + Ck * np.asarray(K) + B)))

    cal = {"variant": VAR, "a": A, "c_tactic_count": Ck, "b": B, "prior": prior,
           "fit_on": "shakeleoatmeal/validation + held-out family scenarios, 10%..100% prefixes, equal weight per source"}
    ZK10 = {n: (VG[n][0][:, g10], VG[n][1][:, g10]) for n in VALS}
    PX = crossfit_probs(ZK10, VAR)
    for n in VALS:
        y = VALS[n].label.values
        yy = np.repeat(y[:, None], len(g10), axis=1).ravel()
        cal[f"crossfit_brier_{n}"] = brier(yy, PX[n].ravel())
        cal[f"crossfit_ece_{n}"] = ece(yy, PX[n].ravel())
    yy = np.repeat(y_tr[:, None], len(g10), axis=1).ravel()
    cal["brier_on_training_pool_oof"] = brier(yy, to_p(oofZ[:, g10], oofK[:, g10]).ravel())
    results["calibration"] = cal
    log("calibration", json.dumps(cal))

    # ---- thresholds -------------------------------------------------------------
    def thr_for(max_p, alpha):
        m_ = np.sort(np.asarray(max_p))[::-1]
        k = int(math.floor(alpha * len(m_)))
        if k >= len(m_):
            return 0.0
        return (math.floor(m_[k] * 1000) + 1) / 1000.0

    ever = {"training_pool_oof": to_p(oofZ[y_tr == 0], oofK[y_tr == 0]).max(axis=1)}
    for n in VALS:
        ns = VALS[n].label.values == 0
        ever[n] = to_p(VG[n][0][ns], VG[n][1][ns]).max(axis=1)
    th = {}
    for nm, alpha in (("yellow", FAR_YELLOW), ("red", FAR_RED)):
        per = {src: thr_for(v, alpha) for src, v in ever.items()}
        th[nm] = max(per.values())
        for src, t in per.items():
            th[f"{nm}_from_{src}"] = t
    th["rule"] = (f"smallest p (3 decimals) such that at most {int(FAR_YELLOW*100)}% (yellow) / "
                  f"{int(FAR_RED*100)}% (red) of non-scam calls EVER reach it at any 2% checkpoint; the strictest of "
                  "training-pool out-of-fold, shakeleoatmeal/validation and the held-out family scenarios")
    YEL, RED = th["yellow"], th["red"]
    results["thresholds"] = th
    log("thresholds", th)

    # ---- 4. test evaluation ------------------------------------------------
    def evaluate(df, name, n_boot=N_BOOT):
        y = df.label.values
        Z, K = grid_scores(df)
        P = to_p(Z, K)
        pf = P[:, -1]
        has_both = 0 < y.sum() < len(y)
        r = {"n": int(len(y)), "n_scam": int(y.sum()), "n_nonscam": int((y == 0).sum())}
        sc, ns = y == 1, y == 0

        def rate(mask_fn, cls):
            return lambda ix: float(np.mean(mask_fn(ix)[y[ix] == cls])) if (y[ix] == cls).any() else float("nan")

        if has_both:
            r["auc"] = with_ci(lambda ix: auc_fast(y[ix], pf[ix]), y, rng, n_boot)
            r["eer"] = with_ci(lambda ix: eer(y[ix], pf[ix]), y, rng, n_boot)
            r["brier_full"] = brier(y, pf)
            r["ece_full"] = ece(y, pf)
            pp = P[:, g10].ravel(); yy = np.repeat(y[:, None], len(g10), axis=1).ravel()
            r["brier_prefixes"] = brier(yy, pp)
            r["ece_prefixes"] = ece(yy, pp)
            r["accuracy_at_0.5"] = with_ci(lambda ix: float(np.mean((pf[ix] >= .5) == y[ix])), y, rng, n_boot)
        r["median_p_full_scam"] = float(np.median(pf[sc])) if sc.any() else None
        r["median_p_full_nonscam"] = float(np.median(pf[ns])) if ns.any() else None
        ever = P.max(axis=1)
        for tn, t in (("yellow", YEL), ("red", RED)):
            d = {}
            if sc.any():
                d["detect_full"] = with_ci(rate(lambda ix: pf[ix] >= t, 1), y, rng, n_boot)
                d["detect_ever"] = with_ci(rate(lambda ix: ever[ix] >= t, 1), y, rng, n_boot)
            if ns.any():
                d["false_alarm_full"] = with_ci(rate(lambda ix: pf[ix] >= t, 0), y, rng, n_boot)
                d["false_alarm_ever"] = with_ci(rate(lambda ix: ever[ix] >= t, 0), y, rng, n_boot)
            if has_both:
                d["accuracy_full"] = with_ci(lambda ix: float(np.mean((pf[ix] >= t) == y[ix])), y, rng, n_boot)
            r["at_" + tn] = d
        # streaming curve
        st = []
        for c, f in zip(g10, FRACS10):
            row = {"frac": f}
            if has_both:
                row["auc"] = with_ci(lambda ix: auc_fast(y[ix], P[ix, c]), y, rng, 1000)
            for tn, t in (("yellow", YEL), ("red", RED)):
                if ns.any():
                    row[f"false_alarm_now_{tn}"] = float(np.mean(P[ns, c] >= t))
                    row[f"false_alarm_by_now_{tn}"] = float(np.mean(P[ns, :c + 1].max(axis=1) >= t))
                if sc.any():
                    row[f"detect_now_{tn}"] = float(np.mean(P[sc, c] >= t))
                    row[f"detect_by_now_{tn}"] = float(np.mean(P[sc, :c + 1].max(axis=1) >= t))
            st.append(row)
        r["streaming"] = st
        if sc.any():
            for tn, t in (("yellow", YEL), ("red", RED)):
                fc = first_cross(P[sc], t)
                ysc = np.ones(sc.sum(), int)
                r[f"time_to_{tn}"] = {
                    "share_of_scams_ever_reaching": float(np.isfinite(fc).mean()),
                    "median_fraction_heard": with_ci(lambda ix: float(np.median(fc[ix])), ysc, rng, n_boot),
                    "median_fraction_heard_among_detected": float(np.median(fc[np.isfinite(fc)])) if np.isfinite(fc).any() else None}
        # per call type
        if df.type.nunique() > 1:
            pt = {}
            for (tp, lab), g in df.groupby(["type", "label"]):
                ix = g.index.values
                t = {"label": int(lab), "n": int(len(ix)),
                     "flag_red_full": float(np.mean(pf[ix] >= RED)), "flag_red_ever": float(np.mean(ever[ix] >= RED)),
                     "flag_yellow_full": float(np.mean(pf[ix] >= YEL)), "flag_yellow_ever": float(np.mean(ever[ix] >= YEL))}
                pt[f"{tp} ({'scam' if lab else 'nonscam'})"] = t
            r["by_type"] = pt
        # rules-only baseline: number of tactic rules that fire on the caller's words
        nt = np.array([len(tag_tactics(normalize(caller_text(d)))) for d in df.dialogue])
        if has_both:
            r["rules_only_baseline_auc"] = with_ci(lambda ix: auc_fast(y[ix], nt[ix]), y, rng, n_boot)
        # can the app give a plain-language reason when the meter is not green?
        flagged = pf >= YEL
        r["why_available"] = {
            "n_flagged_yellow_or_red_full": int(flagged.sum()),
            "share_flagged_with_a_tactic_tag": float(np.mean(nt[flagged] > 0)) if flagged.any() else None,
            "share_green_with_a_tactic_tag": float(np.mean(nt[~flagged] > 0)) if (~flagged).any() else None}
        log("eval", name, "AUC", r.get("auc", {}).get("value"), "EER", r.get("eer", {}).get("value"))
        return r, (Z, K), P

    evals, full_scores = {}, {}
    for key, title in (("test_in", "in-domain test (BothBosu, de-duplicated)"),
                       ("test_in_strict", "in-domain test, strict de-dup (no shared template >= 0.3)"),
                       ("shake_test", "shakeleoatmeal test split"),
                       ("shake_trainplustest", "shakeleoatmeal train+test (not used for training)"),
                       ("menaattia_all", "menaattia train+test (external, never used)"),
                       ("menaattia_test", "menaattia test split"),
                       ("youtube", "youtube scambait transcripts (scams only)"),
                       ("family_test", "family-emergency TEST batch (AI-written, separate writers and scenarios)")):
        r, Z, P = evaluate(S[key], key)
        r["title"] = title
        evals[key] = r
        full_scores[key] = Z
    # validation sources, for reference only (selection, calibration and thresholds were set here)
    rv, _, _ = evaluate(S["val"], "val", n_boot=500)
    rv["title"] = "shakeleoatmeal validation (used for selection, calibration, thresholds)"
    evals["validation_reference"] = rv
    rv, _, _ = evaluate(S["val_family"], "val_family", n_boot=500)
    rv["title"] = "held-out family scenarios (used for selection, calibration, thresholds)"
    evals["validation_family_reference"] = rv
    results["evaluation"] = evals
    write_test_scores(S, full_scores, g10, (A, Ck, B), prior)

    # ---- 5. explanations --------------------------------------------------
    vocab = model.wv.get_feature_names_out()
    coef_w = model.clf.coef_[0][:len(vocab)]
    order = np.argsort(coef_w)
    def tactic_of(term):
        return [t for t in tag_tactics(term)]
    top_pos = [{"ngram": vocab[i], "weight": float(coef_w[i]), "tactic_rule_match": tactic_of(vocab[i])}
               for i in order[::-1][:40]]
    top_neg = [{"ngram": vocab[i], "weight": float(coef_w[i])} for i in order[:40]]
    results["top_ngrams"] = {
        "scam": top_pos, "safe": top_neg,
        "share_of_top40_scam_ngrams_matching_a_tactic_rule": float(np.mean([bool(t["tactic_rule_match"]) for t in top_pos]))}

    tag_stats = {}
    for key in ("train", "test_in", "shake_trainplustest", "menaattia_all", "youtube", "family_test"):
        df = S[key]
        tags = [set(tag_tactics(normalize(caller_text(d)))) for d in df.dialogue]
        y = df.label.values
        row = {}
        for t in TACTICS + [{"id": "any"}]:
            tid = t["id"]
            hit = np.array([(len(s) > 0) if tid == "any" else (tid in s) for s in tags])
            row[tid] = {"scam": float(hit[y == 1].mean()) if (y == 1).any() else None,
                        "nonscam": float(hit[y == 0].mean()) if (y == 0).any() else None}
        tag_stats[key] = row
    results["tactic_rule_fire_rates"] = tag_stats

    # ---- version 1 model on the family calls (before / after) ----------------
    results["v1_on_family"] = v1_on_family(S, rng)

    # ---- demo calls from the app (sanity only; nothing is tuned on these) --
    results["demo_calls"], results["demo_calls_meta"] = demo_calls(model, (A, Ck, B), prior, YEL, RED, S)

    # ---- export -------------------------------------------------------------
    export(model, (A, Ck, B), prior, YEL, RED, cal["fit_on"], chosen, results)
    write_parity_cases(S, model, (A, Ck, B), prior)
    plot(evals)
    results["runtime_seconds"] = round(time.time() - T0, 1)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(clean_json(results), f, indent=1, allow_nan=False)
    log("wrote results.json")


# ---------------------------------------------------------------------------
# Demo calls, export, parity cases, figure
# ---------------------------------------------------------------------------

def clean_json(x):
    """Strict JSON: inf ("never reached the threshold") -> "never", NaN -> null."""
    if isinstance(x, dict):
        return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "never"
        return x
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def extract_demo_calls():
    html = open(os.path.join(HERE, "..", "..", "atelier-securities.html"), encoding="utf-8").read()
    block = html[html.index("var SC={"):html.index("var call=null")]
    calls = {}
    for m in re.finditer(r"\n    (\w+):\{number:", block):
        calls[m.group(1)] = m.start()
    names = list(calls)
    out = {}
    for k, nm in enumerate(names):
        seg = block[calls[nm]: calls[names[k + 1]] if k + 1 < len(names) else len(block)]
        def lines(s):
            return [(w, t) for w, t in re.findall(r'who:"(caller|rose)",text:"((?:[^"\\]|\\.)*)"', s)]
        ev = seg[seg.index("events:["):]
        end = min([x for x in (ev.find("],reply"), ev.find("],\n      pass")) if x > 0])
        out[nm] = {"events": lines(ev[:end])}
        if "pass:[" in seg:
            out[nm]["pass"] = lines(seg[seg.index("pass:["):seg.index("fail:[")])
            out[nm]["fail"] = lines(seg[seg.index("fail:["):seg.index("reply:", seg.index("fail:["))])
    return out


def score_text(model, cal, prior, caller_words):
    A, Ck, B = cal
    clean = normalize(caller_words)
    z = float(model.decision([clean])[0])
    s_ = A * z + Ck * len(tag_tactics(clean)) + B
    p = 1.0 / (1.0 + math.exp(-s_))
    return p, s_ - math.log(prior / (1 - prior))


def demo_calls(model, cal, prior, YEL, RED, S):
    import hashlib
    from neardup import shingle_matrix as _sm
    calls = extract_demo_calls()
    html = open(os.path.join(HERE, "..", "..", "atelier-securities.html"), encoding="utf-8").read()
    sc_block = html[html.index("var SC={"):html.index("var call=null")]
    meta = {"sc_block_sha256": hashlib.sha256(sc_block.encode()).hexdigest(),
            "skipped_no_scripted_caller_lines": []}
    # How much of each demo call's caller wording also appears in the family
    # TRAIN batch: share of the call's word 4-grams found in any training call.
    fam_tr = S["train"][S["train"].source == "family"]
    grams = set()
    for d in fam_tr.dialogue:
        t = normalize(caller_text(d)).split()
        grams |= {" ".join(t[i:i + 4]) for i in range(len(t) - 3)}
    out = {}

    def band(p):
        return "red" if p >= RED else ("yellow" if p >= YEL else "green")
    for nm, c in calls.items():
        if not any(w == "caller" for w, _ in c["events"] + c.get("pass", []) + c.get("fail", [])):
            meta["skipped_no_scripted_caller_lines"].append(nm)
            continue
        cl = normalize(" ".join(t for w, t in c["events"] + c.get("fail", []) if w == "caller")).split()
        g4 = [" ".join(cl[i:i + 4]) for i in range(len(cl) - 3)]
        meta.setdefault("share_of_caller_4grams_seen_in_family_train", {})[nm] = (
            float(np.mean([g in grams for g in g4])) if g4 else None)
        variants = {"main": c["events"]}
        if "pass" in c:
            variants = {"with_pass_branch": c["events"] + c["pass"],
                        "with_fail_branch": c["events"] + c["fail"]}
        out[nm] = {}
        for vn, lines in variants.items():
            said, steps = [], []
            for who, text in lines:
                if who == "caller":
                    said.append(text)
                p, llr = score_text(model, cal, prior, " ".join(said))
                steps.append({"who": who, "text": text, "p": round(p, 4), "llr": round(llr, 3),
                              "band": band(p), "tactics": tag_tactics(normalize(" ".join(said)))})
            out[nm][vn] = steps
            log("demo", nm, vn, [(s_["band"], s_["p"]) for s_ in steps])
    return out, meta


def json_model_scores(m, df, fracs):
    """Score calls with a saved model.json (any version) using the same math as
    stage2_model.js. Used to run the version 1 model on the version 2 family sets."""
    names, greets = set(m["tokenizer"]["name_tokens"]), set(m["tokenizer"]["greeting_tokens"])
    idx = {t: i for i, t in enumerate(m["vocab"])}
    idf, coef = np.array(m["idf"]), np.array(m["coef"])
    cal = m["calibration"]
    a, b, ck = cal["a"], cal["b"], cal.get("c", 0.0)
    P = np.zeros((len(df), len(fracs)))
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
                cols = np.array(list(cnt)); v = (1 + np.log(np.array(list(cnt.values()), float))) * idf[cols]
                z += float((v / np.sqrt((v * v).sum())) @ coef[cols])
            k = len(tag_tactics(clean))
            P[r, c] = 1.0 / (1.0 + math.exp(-(a * z + ck * k + b)))
    return P


def v1_on_family(S, rng):
    m1 = json.load(open(os.path.join(HERE, "v1", "model.json")))
    r1 = json.load(open(os.path.join(HERE, "v1", "results.json")))
    out = {"thresholds_v1": m1["thresholds"]}
    # self-check: this scorer must reproduce version 1's own menaattia test AUC
    Pm = json_model_scores(m1, S["menaattia_test"], [1.0])
    out["self_check_menaattia_test_auc"] = {"recomputed": auc_fast(S["menaattia_test"].label.values, Pm[:, 0]),
                                            "v1_results_json": r1["evaluation"]["menaattia_test"]["auc"]["value"]}
    for key in ("family_test", "val_family"):
        df = S[key]; y = df.label.values
        P = json_model_scores(m1, df, GRID)
        pf, ev_ = P[:, -1], P.max(axis=1)
        d = {"auc": with_ci(lambda ix: auc_fast(y[ix], pf[ix]), y, rng),
             "eer": with_ci(lambda ix: eer(y[ix], pf[ix]), y, rng),
             "auc_by_frac": [auc_fast(y, P[:, GRID.index(f)]) for f in FRACS10]}
        for tn in ("yellow", "red"):
            t = m1["thresholds"][tn]
            d[f"detect_ever_{tn}"] = float(np.mean(ev_[y == 1] >= t))
            d[f"false_alarm_ever_{tn}"] = float(np.mean(ev_[y == 0] >= t))
            d[f"detect_full_{tn}"] = float(np.mean(pf[y == 1] >= t))
            d[f"false_alarm_full_{tn}"] = float(np.mean(pf[y == 0] >= t))
        d["median_p_full_scam"] = float(np.median(pf[y == 1]))
        d["median_p_full_nonscam"] = float(np.median(pf[y == 0]))
        out[key] = d
        log("v1 on", key, "AUC", d["auc"]["value"])
    return out


def export(model, cal, prior, YEL, RED, cal_name, chosen, results):
    A, Ck, B = cal
    vocab = model.wv.get_feature_names_out().tolist()
    coef = model.clf.coef_[0]
    m = {
        "format": "liveline-stage2-text-v2",
        "about": ("Scam-tactic text model for Liveline Stage 2. TF-IDF over the caller's words + logistic "
                  "regression, then a calibrated logistic layer that may also use the number of rule-tagged "
                  "tactics. Trained on LLM-generated public dialogues (BothBosu, apache-2.0) plus family-emergency "
                  "calls written by AI for this project; see RESULTS.md for what the numbers do and do not show."),
        "variant": chosen["variant"], "C": chosen["C"], "family_weight": chosen["family_weight"],
        "input": "the CALLER's words only (receiver lines are ignored)",
        "tokenizer": {
            "steps": ["lowercase", "delete apostrophes ' \u2018 \u2019 \u02bc `",
                      "replace each run of digits 0-9 with ' num '",
                      "replace each run of characters outside a-z with one space",
                      "split on spaces", "words in name_tokens -> 'name'",
                      "words in greeting_tokens -> 'hello'"],
            "name_tokens": NAME_TOKENS, "greeting_tokens": GREETING_TOKENS,
            "word_ngrams": [1, 2], "char_ngrams": None,
            "speaker_tags": {"caller": ["caller", "suspect"], "receiver": ["receiver", "innocent"]},
        },
        "tfidf": {"tf": "1 + ln(count)", "idf": "smooth: ln((1+n)/(1+df)) + 1", "norm": "l2"},
        "vocab": vocab,
        "idf": [float(x) for x in model.wv.idf_],
        "coef": [float(x) for x in coef],
        "intercept": float(model.clf.intercept_[0]),
        "calibration": {"method": "logistic on z and k" if Ck else "platt", "a": A, "c": Ck, "b": B,
                        "fit_on": cal_name,
                        "formula": "p = 1 / (1 + exp(-(a * z + c * k + b))), z = coef . x + intercept, "
                                   "k = number of distinct tactic rules that fire"},
        "prior": prior,
        "llr_formula": "llr = logit(p) - logit(prior); prior is the (source-weighted) scam share of the calibration set",
        "thresholds": {"yellow": YEL, "red": RED},
        "tactics": TACTICS,
    }
    path = os.path.join(HERE, "model.json")
    with open(path, "w") as f:
        json.dump(m, f, separators=(",", ":"))
    results["model_json_bytes"] = os.path.getsize(path)
    log("model.json", os.path.getsize(path), "bytes")


def write_test_scores(S, full_scores, g10, cal, prior):
    """Per-dialogue streaming scores on the held-out test sets, for the fusion
    study. Only dialogues never used for training, CV, calibration or
    thresholds: de-duplicated in-domain test, shakeleoatmeal test, menaattia
    test, and the family-emergency test batch. p_XX / llr_XX are computed on
    the caller's words inside the first XX% of the call's words (both
    speakers), the same prefixes as the streaming curve. No dialogue text."""
    A, Ck, B = cal
    lp = math.log(prior / (1 - prior))
    out = []
    for key in ("test_in", "shake_test", "menaattia_test", "family_test"):
        df, (Z, K) = S[key], full_scores[key]
        src = ["family_test"] * len(df) if key == "family_test" else list(df.source.values)
        rows = pd.DataFrame({
            "source": src,
            "dialogue_id": [f"{s_}/{sp_}/{r}" for s_, sp_, r in zip(df.source, df.split, df.row)],
            "label": df.label.values,
            "n_caller_words": [len(caller_text(d).split()) for d in df.dialogue]})
        S_ = A * Z[:, g10] + Ck * K[:, g10] + B
        for c, f in enumerate(FRACS10):
            rows[f"p_{int(round(f * 100))}"] = 1.0 / (1.0 + np.exp(-S_[:, c]))
        for c, f in enumerate(FRACS10):
            rows[f"llr_{int(round(f * 100))}"] = S_[:, c] - lp
        out.append(rows)
    pd.concat(out).to_csv(os.path.join(HERE, "test_scores.csv"), index=False, float_format="%.10g")
    log("wrote test_scores.csv")


def write_parity_cases(S, model, cal, prior):
    """Cases for parity_test.js. Written under research/data/ (not research/stage2/)
    because they contain dataset text, and menaattia has no license to redistribute."""
    A, Ck, B = cal
    cases = []
    for key in ("test_in", "shake_test", "menaattia_test", "youtube", "family_test"):
        df = S[key]
        for f in (None, 0.1, 0.5):
            texts = [caller_text(d) for d in df.dialogue] if f is None else [prefix_caller(t, f) for t in df.turns]
            clean = [normalize(t) for t in texts]
            z = model.decision(clean)
            k = np.array([len(tag_tactics(c)) for c in clean])
            p = 1.0 / (1.0 + np.exp(-(A * z + Ck * k + B)))
            for i in range(len(df)):
                c = {"set": key if f is None else f"{key}@{f}", "p": float(p[i]), "z": float(z[i]), "k": int(k[i]),
                     "tactics": tag_tactics(clean[i])}
                if f is None:
                    c["dialogue"] = df.dialogue.iloc[i]
                else:
                    c["caller"] = texts[i]
                cases.append(c)
    path = os.path.join(DATA, "parity_cases.json")
    with open(path, "w") as fh:
        json.dump({"prior": prior, "a": A, "c": Ck, "b": B, "cases": cases}, fh)
    log("parity cases", len(cases), "->", path)


def plot(evals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.4))
    sets = [("test_in", "In-domain test (BothBosu)", "#1f4e9c", "-"),
            ("shake_trainplustest", "Cross-dataset: shakeleoatmeal", "#d1495b", "-"),
            ("menaattia_all", "Cross-dataset: menaattia", "#2a9d8f", "-"),
            ("family_test", "Family-emergency test (AI-written)", "#e9a03b", "-")]
    x = [100 * f for f in FRACS10]
    for key, lab, col, ls in sets:
        st = evals[key]["streaming"]
        v = [r["auc"]["value"] for r in st]
        lo = [r["auc"]["ci95"][0] for r in st]; hi = [r["auc"]["ci95"][1] for r in st]
        ax[0].plot(x, v, ls, color=col, lw=2.5, marker="o", label=lab)
        ax[0].fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
    ax[0].axhline(0.5, color="grey", lw=1, ls=":")
    ax[0].set_xlabel("% of the call heard")
    ax[0].set_ylabel("AUC")
    ax[0].set_ylim(0.4, 1.01)
    ax[0].set_xticks(range(10, 101, 10))
    for key, lab, col, ls in sets:
        st = evals[key]["streaming"]
        ax[1].plot(x, [100 * r["detect_by_now_red"] for r in st], "-", color=col, lw=2.5, marker="o")
        ax[1].plot(x, [100 * r["false_alarm_by_now_red"] for r in st], "--", color=col, lw=2, marker="s", ms=5)
    ax[1].plot([], [], "-", color="k", lw=2.5, label="scam calls flagged red")
    ax[1].plot([], [], "--", color="k", lw=2, label="normal calls flagged red (false alarm)")
    ax[1].set_xlabel("% of the call heard")
    ax[1].set_ylabel("% of calls that have turned red")
    ax[1].set_ylim(-2, 102)
    ax[1].set_xticks(range(10, 101, 10))
    ax[1].legend(frameon=False, fontsize=12, loc="upper left")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=13, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(HERE, "streaming_curve.png"), dpi=150)
    log("wrote streaming_curve.png")


if __name__ == "__main__":
    main()

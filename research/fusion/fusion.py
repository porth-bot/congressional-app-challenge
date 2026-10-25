"""Does combining the voice and language stages beat either one alone?

There is no public dataset of phone calls with both audio and transcripts of
grandparent scams, so we build test calls by pairing held-out data from the
two stages:

  real family call   = real voice clip      + non-scam transcript   (label 0)
  voice-clone scam   = computer-made clip   + scam transcript       (label 1)
  human impostor     = real voice clip      + scam transcript       (label 1)

Voice clips come from In-the-Wild (real internet audio, never trained on),
each passed through one of the three phone codecs at random. Transcripts come
from Stage 2's held-out test dialogues. Pairing is random, which makes the two
stages independent given the label by construction: this tests whether the
fusion math does its job, not whether real scams satisfy the independence
assumption.

Each stage gives a log-likelihood ratio (LLR). The voice model's LLR is for
"computer-made vs real voice", calibrated on ASVspoof TRAIN out-of-fold scores
only, so In-the-Wild never informs it. That is not yet evidence about "scam vs
not scam": human scammers have real voices. If a share PI_CLONE of scam calls
use a cloned voice and real family calls never do, then

    P(voice | scam) / P(voice | family) = PI_CLONE * LR_fake + (1 - PI_CLONE)

so a real-sounding voice can lower the odds of a scam by at most a factor of
(1 - PI_CLONE), while a computer-made voice can raise them a lot.
Fused log-odds = log prior odds + log(that ratio) + LLR_words.

Outputs results/fusion.json and results/fig_fusion.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage1"))
from common import RESULTS, ROOT, auc, bootstrap, eer  # noqa: E402

PRIOR = 0.10  # starting scam probability for an unknown number claiming to be family (an assumption, shown in the app)
PI_CLONE = 0.5  # share of family-impersonation scams that use a cloned voice (an assumption; the test mix matches it)
CODECS = ["g711", "opus", "g723"]
PREFIXES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# The app's yellow and red cut-offs on the combined probability are tuned on
# VALIDATION calls (In-the-Wild calibration-half voices + Stage 2 validation
# transcripts): the lowest cut-offs that keep real family calls that ever turn
# yellow at or under 10%, and ever red at or under 2%.
MAX_YELLOW_FA, MAX_RED_FA = 0.10, 0.02

# Stage 3, the family safe word. It is only asked when the combined score lands
# in the unsure (yellow) zone, because it costs the grandparent effort. Its
# likelihood ratio comes from two assumptions we have not measured: how often a
# real family member remembers the word, and how often a scammer knows it.
# P(gives word | real) = KNOW_FAMILY, P(gives word | scam) = KNOW_SCAM, so
#   said it:   LR = KNOW_SCAM / KNOW_FAMILY
#   couldn't:  LR = (1 - KNOW_SCAM) / (1 - KNOW_FAMILY)
KNOW_FAMILY, KNOW_SCAM = 0.90, 0.01
SENSITIVITY = [(0.90, 0.01), (0.70, 0.01), (0.90, 0.05), (0.70, 0.05), (0.90, 0.20), (0.90, 0.50)]
N_CLUSTER_BOOT = 300


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def voice_llrs(scores_file, model, calibration):
    """Voice LLRs (computer-made vs real) for In-the-Wild phone clips.

    calibration="asv_train": calibrate on ASVspoof train out-of-fold scores, and
      test on every In-the-Wild clip. Nothing about In-the-Wild is used.
    calibration="itw_half": split In-the-Wild by speaker into two halves, calibrate
      on one half and test only on the other, the way a deployed system would be
      calibrated on real phone audio before use. No speaker is in both halves.
    """
    s = pd.read_parquet(scores_file)
    s = s[s.model == model]
    itw = s[(s.corpus == "itw") & s.cond.isin(CODECS)].copy()
    man = pd.read_csv(ROOT / "data" / "manifest.csv").set_index("clip_id")
    itw["speaker"] = man.loc[itw.clip_id, "speaker"].to_numpy()
    if calibration == "asv_train":
        cal_rows, test = s[s.corpus == "asv_train_oof"], itw
    else:
        speakers = sorted(itw.speaker.unique())
        cal_speakers = set(speakers[0::2])
        cal_rows, test = itw[itw.speaker.isin(cal_speakers)], itw[~itw.speaker.isin(cal_speakers)].copy()
    cal = LogisticRegression(C=1e6).fit(logit(cal_rows.score.to_numpy())[:, None], cal_rows.y.to_numpy())
    prior = cal_rows.y.mean()
    test["llr"] = cal.decision_function(logit(test.score.to_numpy())[:, None]) - logit(prior)
    cal_rows = cal_rows.copy()
    cal_rows["llr"] = cal.decision_function(logit(cal_rows.score.to_numpy())[:, None]) - logit(prior)
    return test, cal_rows, {"slope": float(cal.coef_[0][0]), "intercept": float(cal.intercept_[0]), "prior": float(prior),
                  "n_calibration_clips": int(len(cal_rows)), "n_test_clips": int(len(test))}


def main():
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    model = cfg.get("voice_model", "ssl_aug")
    scores_file = ROOT / "data" / cfg.get("scores", "benchmark_ssl_scores.parquet")
    n_calls = cfg.get("n_calls", 6000)
    words_test = pd.read_csv(ROOT / "stage2" / "test_scores.csv")
    val_file = ROOT / "stage2" / "val_scores.csv"
    words_val = pd.read_csv(val_file) if val_file.exists() else None
    text_sets = {"external": ["menaattia", "shakeleoatmeal"], "family": ["family_test"]}

    out = {"prior": PRIOR, "pi_clone": PI_CLONE, "voice_model": model, "n_calls": n_calls, "runs": {}}
    cut = weights = None
    for mode in ["asv_train", "itw_half"]:
        voice, voice_cal, calinfo = voice_llrs(scores_file, model, mode)
        print("voice calibration:", mode, calinfo)
        if mode == "itw_half" and words_val is not None:
            fam_val = words_val[words_val.source.str.contains("family")]
            vcalls = make_calls(voice_cal, fam_val, n_calls, np.random.default_rng(11))
            weights = learn_weights(vcalls)
            out["learned_weights"] = weights
            print("learned fusion weights on validation:", weights)
            cut = {name: tune_cutoffs(pk, vcalls) for name, pk in peaks(vcalls, weights).items()}
            out["cutoffs"] = cut
            print("cut-offs tuned on validation:", {k: (round(v["warn"], 3), round(v["stop"], 3)) for k, v in cut.items()})
        for tname, sources in text_sets.items():
            w = words_test[words_test.source.isin(sources)]
            if not len(w):
                continue
            calls = make_calls(voice, w, n_calls, np.random.default_rng(7))
            key = f"{mode}/{tname}"
            print("==", key)
            out["runs"][key] = {"calibration": calinfo, "text_sources": sources,
                                **evaluate(calls, cut, weights if mode == "itw_half" else None)}
            if cut:
                out["runs"][key]["cluster_auc"] = cluster_auc(calls, np.random.default_rng(5))
                out["runs"][key]["matched_red"] = matched_red(calls, cut["fused"])
                print("cluster AUC gain fused-words:", {q: [round(x, 3) for x in v["fused_minus_words"]] for q, v in out["runs"][key]["cluster_auc"]["by_prefix"].items() if q in (10, 30, 50, 100)})
                print("matched red:", out["runs"][key]["matched_red"])
                out["runs"][key]["cascade"] = {f"family_knows_{kf}_scammer_knows_{ks}": cascade(calls, cut["fused"], kf, ks)
                                               for kf, ks in SENSITIVITY}
                print("cascade:", json.dumps(out["runs"][key]["cascade"][f"family_knows_{KNOW_FAMILY}_scammer_knows_{KNOW_SCAM}"]))
    json.dump(out, open(RESULTS / cfg.get("out", "fusion.json"), "w"), indent=1)


def make_calls(voice, words, n_calls, rng):
    """Random test calls: half real family, a quarter voice-clone scams, a quarter human impostors."""
    vb, vs = voice[voice.y == 0], voice[voice.y == 1]
    wn, ws = words[words.label == 0], words[words.label == 1]
    kinds = rng.choice(["family", "clone", "impostor"], size=n_calls, p=[0.5, 0.25, 0.25])
    rows = []
    for k in kinds:
        codec = rng.choice(CODECS)
        pool = vs if k == "clone" else vb
        v = pool[pool.cond == codec].iloc[rng.integers(len(pool[pool.cond == codec]))]
        wpool = wn if k == "family" else ws
        w = wpool.iloc[rng.integers(len(wpool))]
        r = {"kind": k, "y": int(k != "family"), "codec": codec, "speaker": v.speaker, "transcript": w.dialogue_id,
             "llr_fake": float(v.llr),
             "llr_voice": float(np.log(PI_CLONE * np.exp(min(v.llr, 50)) + (1 - PI_CLONE)))}
        for q in PREFIXES:
            r[f"llr_words_{q}"] = float(w[f"llr_{q}"])
        rows.append(r)
    return pd.DataFrame(rows)


def post_naive(df, q):
    """Ablation: treat the voice model's "real vs fake" ratio as if it were the
    "scam vs not" ratio (no mixture). This is what our first version did."""
    z = logit(PRIOR) + df.llr_fake + df[f"llr_words_{q}"]
    return 1 / (1 + np.exp(-z))


def post(df, use_voice, q, weights=None):
    """Combined scam probability. Without weights this is the plain Bayesian
    update (each stage's LLR counts once). With weights, each LLR is scaled by
    how much validation data says to trust it (linear logistic fusion)."""
    wv, ww, b = (weights["voice"], weights["words"], weights["bias"]) if weights else (1.0, 1.0, 0.0)
    z = logit(PRIOR) + b + (wv * df.llr_voice if use_voice else 0) + (ww * df[f"llr_words_{q}"] if q else 0)
    return 1 / (1 + np.exp(-z))


def peak(df, use_voice=True, weights=None):
    """Highest combined probability at any update during the call (the meter shows every update)."""
    return np.max(np.stack([post(df, use_voice, q, weights).to_numpy() for q in PREFIXES]), axis=0)


def learn_weights(calls):
    """Fit one weight per stage plus a bias on validation calls, every prefix stacked."""
    X = np.concatenate([np.c_[calls.llr_voice, calls[f"llr_words_{q}"]] for q in PREFIXES])
    y = np.tile(calls.y.to_numpy(), len(PREFIXES))
    lr = LogisticRegression(C=1e6).fit(X, y)
    # the validation mix is half scams; remove that prior so the bias is a pure correction
    return {"voice": float(lr.coef_[0][0]), "words": float(lr.coef_[0][1]),
            "bias": float(lr.intercept_[0] - logit(y.mean())), "fit_on": "validation calls, all prefixes"}


def peaks(calls, weights):
    out = {"fused": peak(calls), "words_only": peak(calls, use_voice=False)}
    if weights:
        out["fused_learned"] = peak(calls, True, weights)
    return out


def tune_cutoffs(pk, calls):
    pk = pk[calls.kind.to_numpy() == "family"]
    warn = float(np.quantile(pk, 1 - MAX_YELLOW_FA))
    stop = float(np.quantile(pk, 1 - MAX_RED_FA))
    return {"warn": warn, "stop": stop, "rule": f"family calls ever yellow <= {MAX_YELLOW_FA}, ever red <= {MAX_RED_FA}",
            "tuned_on": "In-the-Wild calibration-half voices + Stage 2 family validation transcripts"}


def cascade(calls, cut, know_family, know_scam, seed=3):
    """Run the three-stage cascade on each test call, update by update.
    Stages 1 and 2 (voice and words) run on every update. The first time the
    combined probability reaches the yellow cut-off, either it is already red
    (warn right away) or Stage 3 asks for the safe word once; the answer
    multiplies the odds for the rest of the call. A call 'warns' if it is ever
    red. The word answer is random with the assumed probabilities."""
    rng = np.random.default_rng(seed)
    lr_pass, lr_fail = know_scam / know_family, (1 - know_scam) / (1 - know_family)
    traj = np.stack([post(calls, True, q).to_numpy() for q in PREFIXES], axis=1)  # calls x updates
    kinds = calls.kind.to_numpy()
    asked = np.zeros(len(calls), bool)
    warned = np.zeros(len(calls), bool)
    for i in range(len(calls)):
        hit = np.where(traj[i] >= cut["warn"])[0]
        if not len(hit):
            continue
        j = hit[0]
        if traj[i, j] >= cut["stop"]:
            warned[i] = True
            continue
        asked[i] = True
        knows = rng.random() < (know_family if kinds[i] == "family" else know_scam)
        odds = traj[i, j:] / (1 - traj[i, j:]) * (lr_pass if knows else lr_fail)
        warned[i] = bool((odds / (1 + odds) >= cut["stop"]).any())
    return {"lr_said_it": lr_pass, "lr_couldnt": lr_fail,
            **{kind: {"asked_for_word": float(asked[kinds == kind].mean()), "warned_red": float(warned[kinds == kind].mean())}
               for kind in ["family", "clone", "impostor"]}}


def cluster_auc(calls, rng):
    """AUC of voice, words and combined, and the combined-minus-single gains, with
    intervals that resample speakers and transcripts (each call is weighted by how
    many times its speaker and its transcript were drawn). The test calls reuse a
    limited set of speakers and transcripts, so resampling calls would be too narrow."""
    from sklearn.metrics import roc_auc_score
    spk, trn = calls.speaker.to_numpy(), calls.transcript.to_numpy()
    us, ut = np.unique(spk), np.unique(trn)
    y = calls.y.to_numpy()
    scores = {q: {"voice": post(calls, True, 0).to_numpy(), "words": post(calls, False, q).to_numpy(),
                  "fused": post(calls, True, q).to_numpy()} for q in PREFIXES}
    draws = []
    for _ in range(N_CLUSTER_BOOT):
        cs = pd.Series(rng.choice(us, len(us))).value_counts()
        ct = pd.Series(rng.choice(ut, len(ut))).value_counts()
        w = cs.reindex(spk).fillna(0).to_numpy() * ct.reindex(trn).fillna(0).to_numpy()
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        draws.append({q: {k: roc_auc_score(y, v, sample_weight=w) for k, v in d.items()} for q, d in scores.items()})
    out = {}
    for q in PREFIXES:
        a = {k: np.array([d[q][k] for d in draws]) for k in ["voice", "words", "fused"]}
        pct = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        out[q] = {**{k: pct(v) for k, v in a.items()},
                  "fused_minus_words": pct(a["fused"] - a["words"]), "fused_minus_voice": pct(a["fused"] - a["voice"])}
    return {"n_boot": len(draws), "n_speakers": int(len(us)), "n_transcripts": int(len(ut)), "by_prefix": out}


def matched_red(calls, cut):
    """Compare 'ever red' catch rates at the SAME rate of real family calls ending
    red: move the words-only red cut-off until its family red rate equals the
    combined system's, then compare scam catch rates."""
    fam = calls.kind.to_numpy() == "family"
    pf, pw = peak(calls), peak(calls, use_voice=False)
    fa = float((pf[fam] >= cut["stop"]).mean())
    tw = float(np.quantile(pw[fam], 1 - fa))
    return {"family_red_rate": fa, "words_only_matched_cutoff": tw,
            **{kind: {"combined": float((pf[calls.kind == kind] >= cut["stop"]).mean()),
                      "words_only": float((pw[calls.kind == kind] >= tw).mean())} for kind in ["clone", "impostor"]}}


def evaluate(calls, cut, weights=None):
    out = {"mix": calls.kind.value_counts().to_dict(), "by_prefix": {}}
    y = calls.y.to_numpy()
    for q in PREFIXES:
        rec = {}
        variants = [("voice", True, 0, None), ("words", False, q, None), ("fused", True, q, None)]
        if weights:
            variants.append(("fused_learned", True, q, weights))
        variants.append(("fused_naive", None, q, None))
        for name, uv, qq, wts in variants:
            p = (post_naive(calls, qq) if uv is None else post(calls, uv, qq, wts)).to_numpy()
            rec[name] = {"auc": bootstrap(y, p, auc, n=500), "eer": bootstrap(y, p, lambda a, b: eer(a, b)[0], n=500)}
            # detection at a fixed 5% false-alarm rate on real family calls
            thr = np.quantile(p[y == 0], 0.95)
            rec[name]["catch_at_5pct_false_alarm"] = {
                kind: float((p[calls.kind == kind] > thr).mean()) for kind in ["clone", "impostor"]}
        out["by_prefix"][q] = rec
        print(q, {k: round(v["auc"][0], 3) for k, v in rec.items()},
              {k: v["catch_at_5pct_false_alarm"] for k, v in rec.items()})
    if cut:
        for name, pk in peaks(calls, weights).items():
            c = cut[name]
            out[f"ever_reached_{name}"] = {
                kind: {"yellow_or_red": float((pk[calls.kind == kind] >= c["warn"]).mean()),
                       "red": float((pk[calls.kind == kind] >= c["stop"]).mean())}
                for kind in ["family", "clone", "impostor"]}
            print(f"ever reached ({name}):", {k: {a: round(b, 3) for a, b in v.items()} for k, v in out[f"ever_reached_{name}"].items()})
    return out


if __name__ == "__main__":
    sys.exit(main())

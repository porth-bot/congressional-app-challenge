"""Feature survival study: which voice cues still separate real from fake
after a phone codec?

For each feature and channel we compute the AUC of the raw feature value
(no model), oriented so that 1.0 means perfect separation in the direction the
feature points on studio audio. Separation = 2 * (AUC - 0.5), so 0 is chance
and 1 is perfect; a negative value means the codec flipped the direction.

Survivors are chosen on the ASVspoof TRAIN split only, using only the two
training codecs (G.711, Opus). G.723.1 and both test sets never influence the
choice, so they stay honest held-out checks.

Rule, fixed before looking at any test number: a feature survives if its
separation is at least 0.2 (oriented AUC >= 0.60) under both training codecs.
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

from common import CONDITIONS, RESULTS, TRAIN_CODECS, auc, load
from features import BIO, SPECTRAL

MIN_SEP = 0.2


def main():
    data = {c: load(c) for c in CONDITIONS}
    ceps = [k for k in data["clean"].columns if k.startswith(("mfcc", "dmfcc", "lfcc"))]
    feats = BIO + SPECTRAL + ceps
    out = {"rule": f"separation >= {MIN_SEP} on ASVspoof train under {TRAIN_CODECS}", "features": {}}
    for f in feats:
        rec = {"family": "bio" if f in BIO else "spectral" if f in SPECTRAL else "cepstral"}
        for corpus in ["asv_train", "asv_test", "itw"]:
            base = data["clean"][data["clean"].corpus == corpus]
            ok = base[f].notna()
            a0 = auc(base.y[ok], base[f][ok])
            sign = 1 if a0 >= 0.5 else -1
            seps = {}
            for c in CONDITIONS:
                d = data[c][data[c].corpus == corpus]
                ok = d[f].notna()
                if ok.sum() < 50 or d.y[ok].nunique() < 2:
                    seps[c] = None
                    continue
                a = auc(d.y[ok], d[f][ok])
                seps[c] = round(2 * ((a if sign > 0 else 1 - a) - 0.5), 4)
            rec[corpus] = seps
            if corpus == "asv_train":
                rec["direction"] = "higher in fakes" if sign > 0 else "lower in fakes"
                # does the codec keep each clip's value in the same order? (same clips, both channels)
                m = data["clean"][data["clean"].corpus == corpus][["clip_id", f]]
                rho = {}
                for c in CONDITIONS[1:]:
                    j = m.merge(data[c][["clip_id", f]], on="clip_id", suffixes=("_a", "_b")).dropna()
                    rho[c] = round(float(spearmanr(j[f + "_a"], j[f + "_b"]).correlation), 4)
                rec["stability_rho"] = rho
        tr = rec["asv_train"]
        rec["survives"] = bool(all(tr[c] is not None and tr[c] >= MIN_SEP for c in TRAIN_CODECS))
        out["features"][f] = rec
    surv = [f for f, r in out["features"].items() if r["survives"]]
    out["survivors"] = surv
    out["survivors_bio_spectral"] = [f for f in surv if f in BIO + SPECTRAL]
    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / "survival.json", "w"), indent=1)

    print(f"{'feature':16s} {'clean':>6s} {'g711':>6s} {'opus':>6s} {'g723':>6s}  rho_g711  survives")
    for f in BIO + SPECTRAL:
        r = out["features"][f]
        t = r["asv_train"]
        fmt = lambda v: f"{v:6.2f}" if v is not None else "   n/a"
        print(f"{f:16s} {fmt(t['clean'])} {fmt(t['g711'])} {fmt(t['opus'])} {fmt(t['g723'])}  {r['stability_rho']['g711']:7.2f}   {r['survives']}")
    print("survivors:", len(surv), "of", len(feats), "| bio/spectral:", out["survivors_bio_spectral"])


if __name__ == "__main__":
    sys.exit(main())

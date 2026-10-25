"""Leakage audit for the Stage 2 text data. Run before any modeling.

    ../.venv/bin/python audit.py

Writes audit.json and prints a readable report. Checks:
  1. what each source is (size, labels, types, how speakers are marked)
  2. exact and near-duplicate dialogues within and across sources/splits
  3. speaker tags and other formatting by label
  4. length differences by label (and how well length alone separates them)
  5. whether the receiver's lines give the label away
  6. caller words that are strongly tied to one label (names, templates)
"""

import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from common import HERE, TAG_RE, analyzer, caller_text, load_all, normalize, parse_turns
from neardup import components, near_pairs, shingle_matrix

NEAR = 0.5      # Jaccard on word 5-gram shingles: at or above this = near-duplicate
LOOSE = 0.3     # looser "same template" level, reported for sensitivity
BB = ["scam_dialogue", "single_agent", "multi_agent"]


def receiver_text(d):
    return " ".join(t for r, t in parse_turns(d) if r == "receiver")


def main():
    df = load_all()
    rep = {}
    print("=" * 70, "\n1. SOURCES\n")
    summ = []
    for (src, split), g in df.groupby(["source", "split"], sort=False):
        tags = Counter(m.lower() for d in g.dialogue for m in TAG_RE.findall(d))
        row = {"source": src, "split": split, "n": len(g),
               "n_scam": int(g.label.sum()), "n_nonscam": int((g.label == 0).sum()),
               "types_scam": sorted(g[g.label == 1].type.unique().tolist()),
               "types_nonscam": sorted(g[g.label == 0].type.unique().tolist()),
               "speaker_tags": dict(tags.most_common()),
               "mean_chars": round(float(g.dialogue.str.len().mean()), 1)}
        summ.append(row)
        print(f"{src:15s} {split:10s} n={row['n']:5d} scam={row['n_scam']:5d} "
              f"nonscam={row['n_nonscam']:5d} chars={row['mean_chars']:7.1f} tags={row['speaker_tags']}")
        print(f"{'':27s}scam types={row['types_scam']} nonscam types={row['types_nonscam']}")
    rep["sources"] = summ

    # ------------------------------------------------------------------
    print("\n" + "=" * 70, "\n2. DUPLICATES\n")
    norm_full = [" ".join(normalize(t, False) for r, t in parse_turns(d)) for d in df.dialogue]
    df["_norm"] = norm_full
    ex = df.groupby("_norm").agg(n=("source", "size"),
                                 where=("source", lambda s: sorted(set(s))))
    ex = ex[ex.n > 1]
    print(f"exact duplicate groups (after removing speaker tags and case): {len(ex)}, "
          f"covering {int(ex.n.sum())} dialogues")
    exact_pairs = Counter(" + ".join(w) for w in ex["where"])
    print("  by source combination:", dict(exact_pairs))
    rep["exact_duplicate_groups"] = {"groups": int(len(ex)), "dialogues": int(ex.n.sum()),
                                     "by_sources": dict(exact_pairs)}

    X = shingle_matrix(df.dialogue.tolist())
    i, j, s = near_pairs(X, threshold=LOOSE)
    grp = (df.source + "/" + df.split).values
    lab = df.label.values
    pairs = pd.DataFrame({"a": grp[i], "b": grp[j], "s": s, "same_label": lab[i] == lab[j],
                          "ta": df.type.values[i], "tb": df.type.values[j]})
    pairs["combo"] = [" | ".join(sorted(x)) for x in zip(pairs.a, pairs.b)]
    near = {}
    for th in (LOOSE, NEAR, 0.95):
        c = pairs[pairs.s >= th].groupby("combo").size().sort_values(ascending=False)
        near[str(th)] = {k: int(v) for k, v in c.items()}
        print(f"\npairs with Jaccard >= {th}:")
        print(c.to_string())
    rep["near_duplicate_pairs"] = near
    disc = int(((pairs.s >= NEAR) & ~pairs.same_label).sum())
    print(f"\nnear-duplicate pairs (>= {NEAR}) with DIFFERENT labels: {disc}")
    rep["near_duplicate_label_conflicts"] = disc
    same = pairs[(pairs.s >= NEAR) & (pairs.a.str.split("/").str[0] == pairs.b.str.split("/").str[0])]
    same_src = same.a.str.split("/").str[0]
    bt = same.groupby([same_src, same.ta]).size()
    print("\nsame-source near-duplicates by source and call type:\n" + bt.to_string())
    rep["same_source_near_dups_by_type"] = {f"{a}|{b}": int(v) for (a, b), v in bt.items()}
    x = pairs[(pairs.s >= NEAR) & (pairs.a.str.split("/").str[0] != pairs.b.str.split("/").str[0])]
    bytype = x.groupby(["ta", "tb"]).size()
    print("\ncross-source near-duplicates by call type:\n" + bytype.to_string())
    rep["cross_source_near_dups_by_type"] = {f"{a}|{b}": int(v) for (a, b), v in bytype.items()}

    # Version 2: hand-written family-emergency calls. How close is each TEST
    # call to its nearest TRAIN-batch call (train + validation clusters)?
    fam = df[df.source == "family"].reset_index(drop=True)
    Xf = shingle_matrix(fam.dialogue.tolist()).astype(np.float32)
    inter = (Xf @ Xf.T).toarray()
    sz = np.asarray(Xf.sum(axis=1)).ravel()
    jac = inter / np.maximum(sz[:, None] + sz[None, :] - inter, 1)
    te_i = np.where(fam.split == "test")[0]
    tr_i = np.where(fam.split != "test")[0]
    nn_te = jac[np.ix_(te_i, tr_i)].max(axis=1)
    jw = jac[np.ix_(tr_i, tr_i)].copy()
    np.fill_diagonal(jw, 0)
    nn_tr = jw.max(axis=1)
    fam_ov = {
        "test_to_nearest_train": {"max": float(nn_te.max()), "p99": float(np.percentile(nn_te, 99)),
                                  "median": float(np.median(nn_te)), "n_ge_0.3": int((nn_te >= LOOSE).sum()),
                                  "n_ge_0.5": int((nn_te >= NEAR).sum()), "n_test": int(len(te_i))},
        "train_to_nearest_other_train": {"max": float(nn_tr.max()), "median": float(np.median(nn_tr)),
                                         "n_ge_0.3": int((nn_tr >= LOOSE).sum()), "n_ge_0.5": int((nn_tr >= NEAR).sum()),
                                         "n_train": int(len(tr_i))}}
    print("\nfamily calls, nearest-neighbor Jaccard (word 5-gram shingles):", json.dumps(fam_ov))
    rep["family_batch_overlap"] = fam_ov

    # Effect on a pooled BothBosu split
    k = s >= NEAR
    df["cluster"] = components(len(df), i[k], j[k])
    bb = df[df.source.isin(BB)]
    tr, te = bb[bb.split == "train"], bb[bb.split == "test"]
    leaked = te[te.cluster.isin(set(tr.cluster))]
    print(f"\nBothBosu pooled: {len(tr)} train dialogues form {tr.cluster.nunique()} near-dup clusters "
          f"({len(tr) - tr.cluster.nunique()} redundant copies)")
    print(f"BothBosu pooled: {len(leaked)} of {len(te)} test dialogues are near-copies of a train dialogue")
    print("  leaked test dialogues by source:", leaked.source.value_counts().to_dict())
    rep["pooled_bothbosu"] = {
        "train_dialogues": int(len(tr)), "train_clusters": int(tr.cluster.nunique()),
        "test_dialogues": int(len(te)), "test_near_copies_of_train": int(len(leaked)),
        "leaked_by_source": {k2: int(v) for k2, v in leaked.source.value_counts().items()}}

    # ------------------------------------------------------------------
    print("\n" + "=" * 70, "\n3/4. FORMATTING AND LENGTH BY LABEL\n")
    fmt = []
    for src, g in df.groupby("source", sort=False):
        cw = np.array([len(caller_text(d).split()) for d in g.dialogue])
        nt = np.array([len(parse_turns(d)) for d in g.dialogue])
        nl = np.array(["\n" in d for d in g.dialogue])
        y = g.label.values
        row = {"source": src}
        for v, name in ((0, "nonscam"), (1, "scam")):
            if (y == v).any():
                row[f"caller_words_median_{name}"] = float(np.median(cw[y == v]))
                row[f"turns_median_{name}"] = float(np.median(nt[y == v]))
                row[f"has_newlines_{name}"] = round(float(nl[y == v].mean()), 3)
        if len(set(y)) == 2:
            row["auc_caller_length_only"] = round(float(roc_auc_score(y, cw)), 3)
            row["auc_turn_count_only"] = round(float(roc_auc_score(y, nt)), 3)
        fmt.append(row)
        print(row)
    rep["format_and_length"] = fmt

    # ------------------------------------------------------------------
    print("\n" + "=" * 70, "\n5. DOES THE RECEIVER GIVE THE LABEL AWAY?\n")
    rx = re.compile(r"\b(scam|scams|scammer|scammers|fraud|fraudster)\b", re.I)
    rr = []
    for src, g in df.groupby("source", sort=False):
        hit = np.array([bool(rx.search(receiver_text(d))) for d in g.dialogue])
        y = g.label.values
        row = {"source": src}
        for v, name in ((0, "nonscam"), (1, "scam")):
            if (y == v).any():
                row[f"receiver_says_scam_{name}"] = round(float(hit[y == v].mean()), 3)
        rr.append(row)
        print(row)
    rep["receiver_says_scam_or_fraud"] = rr

    # Caller-only vs receiver-only models, grouped 5-fold CV on pooled BothBosu train
    tr = tr.reset_index(drop=True)
    y = tr.label.values
    gkf = GroupKFold(n_splits=5)
    cvres = {}
    for name, fn in (("caller_only", caller_text), ("receiver_only", receiver_text)):
        texts = [normalize(fn(d)) for d in tr.dialogue]
        vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, min_df=2)
        Xt = vec.fit_transform(texts)
        p = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), Xt, y,
                              groups=tr.cluster.values, cv=gkf, method="decision_function")
        cvres[name] = round(float(roc_auc_score(y, p)), 4)
    print("grouped 5-fold CV AUC on pooled BothBosu train:", cvres)
    rep["cv_auc_by_speaker"] = cvres

    # ------------------------------------------------------------------
    print("\n" + "=" * 70, "\n6. CALLER WORDS TIED TO ONE LABEL (per BothBosu source)\n")
    skew = {}
    for src in BB:
        g = df[(df.source == src) & (df.split == "train")]
        docs = [set(normalize(caller_text(d), False).split()) for d in g.dialogue]
        y = g.label.values
        c1, c0 = Counter(), Counter()
        for d, v in zip(docs, y):
            (c1 if v else c0).update(d)
        n1, n0 = int(y.sum()), int((y == 0).sum())
        rows = []
        for w in set(c1) | set(c0):
            a, b = c1[w], c0[w]
            if a + b < 25:
                continue
            lo = np.log((a + .5) / (n1 - a + .5)) - np.log((b + .5) / (n0 - b + .5))
            rows.append((w, a, b, lo))
        rows.sort(key=lambda r: r[3])
        top_scam = [(w, a, b) for w, a, b, _ in rows[::-1][:30]]
        top_non = [(w, a, b) for w, a, b, _ in rows[:30]]
        skew[src] = {"scam": top_scam, "nonscam": top_non}
        print(f"-- {src}: (word, #scam docs, #nonscam docs)")
        print("   scam-leaning:   ", top_scam)
        print("   nonscam-leaning:", top_non)
    rep["label_skewed_caller_words"] = skew

    names = ["john", "johnson", "karen", "david", "rachel", "emily", "smith", "officer",
             "thompson", "lee", "mike", "sarah", "xyz", "abc", "jake", "ethan"]
    nm = {}
    for src in BB + ["shakeleoatmeal", "menaattia"]:
        g = df[df.source == src]
        docs = [set(normalize(caller_text(d), False).split()) for d in g.dialogue]
        y = g.label.values
        nm[src] = {w: {"scam": round(float(np.mean([w in d for d, v in zip(docs, y) if v == 1])), 3),
                       "nonscam": round(float(np.mean([w in d for d, v in zip(docs, y) if v == 0])), 3)}
                   for w in names}
    print("\nshare of caller texts containing common generated names, by label:")
    for src, d in nm.items():
        print(" ", src, {w: (v["scam"], v["nonscam"]) for w, v in d.items() if v["scam"] + v["nonscam"] > 0.02})
    rep["name_rates"] = nm

    # Which scam stories exist at all in the BothBosu training/test data?
    print("\n" + "=" * 70, "\n7. SCAM STORIES PRESENT IN BOTHBOSU (caller words, any split)\n")
    stories = {
        "family_words (grandma, grandpa, grandson, granddaughter, grandmother, grandfather, grandchild)":
            r"\b(grandma|grandpa|grandson|granddaughter|grandmother|grandfather|grandchild|grandkid)",
        "bail": r"\bbail\b",
        "arrest": r"\barrest",
        "crypto (bitcoin, crypto, cryptocurrency)": r"\b(bitcoin|crypto)",
        "gift card": r"\bgift ?cards?\b",
    }
    bbd = df[df.source.isin(BB)]
    st = {}
    for name, pat in stories.items():
        rx2 = re.compile(pat, re.I)
        hit = np.array([bool(rx2.search(caller_text(d))) for d in bbd.dialogue])
        st[name] = {"scam_dialogues": int(hit[bbd.label.values == 1].sum()),
                    "normal_dialogues": int(hit[bbd.label.values == 0].sum()),
                    "of_total": int(len(bbd))}
        print(f"  {name}: {st[name]}")
    rep["scam_story_mentions_bothbosu"] = st

    with open(os.path.join(HERE, "audit.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("\nwrote audit.json")


if __name__ == "__main__":
    main()

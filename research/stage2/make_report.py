"""Builds RESULTS.md from results.json, audit.json and parity_result.json.

    ../.venv/bin/python make_report.py

Every number in RESULTS.md is filled in here from those files, so nothing is
typed by hand. The prose around the numbers was written after reading them.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
A = json.load(open(os.path.join(HERE, "audit.json")))
P = json.load(open(os.path.join(HERE, "parity_result.json")))
E = R["evaluation"]
V1 = json.load(open(os.path.join(HERE, "v1", "results.json")))
E1 = V1["evaluation"]


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def vci(m, fmt=f3):
    """value [lo, hi]"""
    if m is None:
        return "n/a"
    lo, hi = m["ci95"]
    return f"{fmt(m['value'])} [{fmt(lo)}, {fmt(hi)}]"


def pci(m):
    return vci(m, lambda x: pct(x))


def frac_heard(x):
    if x is None:
        return "n/a"
    if isinstance(x, str) or x == float("inf"):
        return "never"
    return f"{100 * x:.0f}%"


def median_heard(m):
    v = m["value"]; lo, hi = m["ci95"]
    return f"{frac_heard(v)} [{frac_heard(lo)}, {frac_heard(hi)}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


import re

SETS = [("test_in", "In-domain test (BothBosu, de-duplicated)"),
        ("test_in_strict", "In-domain test, strict de-dup"),
        ("shake_test", "shakeleoatmeal test split"),
        ("shake_trainplustest", "shakeleoatmeal train+test"),
        ("menaattia_test", "menaattia test split"),
        ("menaattia_all", "menaattia train+test"),
        ("youtube", "YouTube scambait transcripts (scams only)"),
        ("family_test", "Family-emergency TEST batch (AI-written)"),
        ("validation_reference", "shakeleoatmeal validation (tuning, reference)"),
        ("validation_family_reference", "Held-out family scenarios (tuning, reference)")]

SCEN = {
    "TS01": "arrested after a car accident, bail, 'lawyer' takes over", "TS02": "hospital after a crash, pay before treatment",
    "TS03": "stuck abroad, passport and wallet stolen", "TS04": "drugs in a friend's car, bail, don't tell parents",
    "TS05": "'officer' says DUI custody, cash courier", "TS06": "caused an accident, settle quietly to avoid lawsuit",
    "TS07": "adult child on work trip, lost wallet, voice sounds different, payment app",
    "TS08": "bar fight, lawyer retainer in gift cards", "TS09": "owes dangerous people, crypto ATM",
    "TS10": "'ER doctor' wants surgery deposit by wire", "TS11": "SUBTLE: chit-chat then small ask to a friend's account",
    "TS12": "'bail bondsman' follow-up, cash in a box, gag order",
    "TN01": "plan a visit or airport ride", "TN02": "check-in after a doctor visit", "TN03": "new phone number, nothing asked",
    "TN04": "birthday call or party planning", "TN05": "HARD: paying back a small loan by Venmo/check",
    "TN06": "HARD: 'did you get the check Mom mailed?'", "TN07": "HARD: car broke down, parent already coming",
    "TN08": "HARD: chip in on a group gift, no pressure", "TN09": "HARD: real doctor's office reminder",
    "TN10": "HARD: real bank fraud alert, 'call the number on your card'", "TN11": "good news, past trouble resolved",
    "TN12": "HARD: ride to the store, paying back for groceries",
    "XS01": "hurt on a trip, clinic won't discharge until paid, Cash App/wire",
    "XS02": "relative 'held' over a debt, stay on the line and pay", "XS03": "arrested at protest/concert, bail in prepaid cards",
    "XS04": "abroad, customs found something, fine by wire/crypto, 'been crying'",
    "XS05": "in-law says your child crashed, tow + co-pay, child's phone broken",
    "XS06": "'insurance adjuster' private settlement, cash courier",
    "XS07": "SUBTLE: catch-up then rent 'loan', keep from parents, gift cards",
    "XS08": "new number, jail on old traffic warrant, gift cards",
    "XN01": "asks for a family recipe", "XN02": "pharmacy pickup reminder, offers a ride",
    "XN03": "HARD: speeding ticket already paid, no ask", "XN04": "HARD: real pharmacy, prescription ready, copay at pickup",
    "XN05": "HARD: open ask to chip in to a family trip fund", "XN06": "HARD: credit union, new card, call number on back",
    "XN07": "HARD: lost wallet last week, already handled", "XN08": "HARD: calling from the hospital with happy news",
}


def n_str(e):
    return f"{e['n']} ({e['n_scam']} / {e['n_nonscam']})"


def g(e, *path):
    for k in path:
        if e is None or k not in e:
            return None
        e = e[k]
    return e


def demote(md):
    """Version 1 report as an appendix: push every heading down one level."""
    md = md.replace("](streaming_curve.png)", "](v1/streaming_curve.png)")
    return re.sub(r"^(#+) ", lambda m: "#" + m.group(1) + " ", md, flags=re.M)


def build():
    L = []
    add = L.append
    th, cal, ch, sz, sp = R["thresholds"], R["calibration"], R["chosen_variant"], R["split_sizes"], R["splits"]
    m = json.load(open(os.path.join(HERE, "model.json")))
    fam, fam1 = E["family_test"], R["v1_on_family"]["family_test"]
    sh, me, yt, ti = E["shake_trainplustest"], E["menaattia_all"], E["youtube"], E["test_in"]
    th1 = V1["thresholds"]
    demo_max = {k: max(st['p'] for v in R["demo_calls"][k].values() for st in v) for k in R["demo_calls"]}

    add("# Liveline Stage 2: scam-tactic text model (version 2)")
    add("")
    add("Stage 2 reads what the caller says and scores how much it sounds like a scam. It is a small TF-IDF plus "
        "logistic regression model that runs in plain browser JavaScript (`stage2_model.js`, no libraries) and returns a "
        "calibrated probability, a log-likelihood ratio for the Bayesian fusion step, a green / yellow / red band, and "
        "rule-based tactic tags for the plain-language \"why\".")
    add("")
    add("Version 2 adds family-emergency (\"grandparent\") calls, which no public dataset has, and tries a tactic-count "
        "input. Version 1 (public data only) is kept at the end of this file and in `v1/`. Every number below is filled "
        "in by `make_report.py` from files written by `audit.py`, `train.py` and `parity_test.js`. Nothing was tuned on "
        "a test set.")
    add("")
    add("> **Read this before quoting any family-call number.** All family-emergency calls (training, validation and "
        "test) were written by AI for this project. The app's demo calls were also written "
        "with AI help. So the family test result says the model learned what an AI model thinks these calls sound like, not "
        "that it catches real grandparent scams, and the demo-call scores below are a sanity check, not evidence.")
    add("")
    add("## The short version")
    add("")
    add(f"- **Family-emergency calls:** on the held-out family test batch (different writers, scenarios and names), "
        f"AUC went from {vci(fam1['auc'])} with version 1 to **{vci(fam['auc'])}** with version 2; EER "
        f"{vci(fam1['eer'])} to {vci(fam['eer'])}. Scams that turned red at any point: {pct(fam1['detect_ever_red'])} to "
        f"{pct(fam['at_red']['detect_ever']['value'])}; yellow: {pct(fam1['detect_ever_yellow'])} to "
        f"{pct(fam['at_yellow']['detect_ever']['value'])}. Normal family calls that ever turned red: "
        f"{pct(fam1['false_alarm_ever_red'])} to {pct(fam['at_red']['false_alarm_ever']['value'])}; yellow: "
        f"{pct(fam1['false_alarm_ever_yellow'])} to {pct(fam['at_yellow']['false_alarm_ever']['value'])}.")
    add(f"- **But the alarm still never fires on family calls.** Family scams get a median p of "
        f"{f3(fam['median_p_full_scam'])} against {f3(fam['median_p_full_nonscam'])} for normal family calls, so they rank "
        f"well, but the shipped thresholds (yellow {th['yellow']}, red {th['red']}) come from the strictest validation "
        f"source, shakeleoatmeal, and sit above every family test call. The held-out family scenarios alone would have put "
        f"yellow at {th['yellow_from_val_family']} and red at {th['red_from_val_family']}. Changing the threshold rule now, "
        "after seeing test results, would be tuning on test, so it was not done; it is the first thing to decide for version 3.")
    vc = R["variant_comparison"]
    worse = all(d["score"] <= a["score"] for a in vc for d in vc
                if a["variant"] == "A_word12" and d["variant"] != "A_word12" and a["C"] == d["C"] and a["family_weight"] == d["family_weight"])
    add(f"- **The tactic-count input did not help:** on validation, variant D scored "
        f"{'at or below' if worse else 'not consistently above'} variant A at every C and family weight.")
    add(f"- **Public test sets, before and after:** shakeleoatmeal AUC {f3(E1['shake_trainplustest']['auc']['value'])} to "
        f"{f3(sh['auc']['value'])}, menaattia {f3(E1['menaattia_all']['auc']['value'])} to {f3(me['auc']['value'])}, "
        f"in-domain {f3(E1['test_in']['auc']['value'])} to {f3(ti['auc']['value'])}. menaattia got slightly worse "
        f"({vci(E1['menaattia_all']['auc'])} to {vci(me['auc'])}); shakeleoatmeal is unchanged within its intervals. "
        "Full table below.")
    add(f"- **Chosen model:** {ch['variant']}, C = {ch['C']}, family rows weighted {ch['family_weight']}x. Thresholds: "
        f"yellow {th['yellow']}, red {th['red']} (version 1: {th1['yellow']}, {th1['red']}).")
    add("- **Demo calls:** " + ", ".join(f"{k} {demo_max[k]:.2f}" for k in demo_max)
        + f" at most (yellow starts at {th['yellow']}). Not evidence either way (see the box above).")
    add(f"- **JavaScript parity:** max |p_js - p_sklearn| = {P['max_abs_diff_p']:.1e} on {P['cases']} cases, "
        f"{'PASS' if P['pass'] else 'FAIL'}. `model.json` is {R['model_json_bytes']:,} bytes.")
    add("")
    add("## Rerun everything")
    add("")
    add("```bash")
    add("cd research/stage2")
    add("../.venv/bin/python fetch_data.py    # public datasets at pinned commits -> research/data/stage2/")
    add("# family calls: research/data/stage2/family/{train,test}/batch_*.txt (written once by AI; checked with family_check.py)")
    add("../.venv/bin/python audit.py         # leakage audit -> audit.json, audit_output.txt")
    add(f"../.venv/bin/python train.py         # about {R['runtime_seconds'] / 60:.0f} min -> model.json, results.json, test_scores.csv, streaming_curve.png")
    add("node parity_test.js                  # JavaScript vs scikit-learn -> parity_result.json")
    add("../.venv/bin/python make_report.py   # this file")
    add("```")
    add("")
    add(f"Seed {R['seed']}. Python {R['versions']['python']}, scikit-learn {R['versions']['sklearn']}, numpy "
        f"{R['versions']['numpy']}, scipy {R['versions']['scipy']}, node {P['node']}.")
    add("")

    # ---------------- before / after
    add("## Before and after (version 1 vs version 2)")
    add("")
    add("Version 1 numbers come from `v1/results.json`; version 1 on the family sets was scored by running `v1/model.json` "
        f"through the same math as the JavaScript (self-check: it reproduces version 1's menaattia test AUC, "
        f"{R['v1_on_family']['self_check_menaattia_test_auc']['recomputed']:.4f} vs "
        f"{R['v1_on_family']['self_check_menaattia_test_auc']['v1_results_json']:.4f}). Each version uses its own thresholds.")
    add("")
    rows = []
    for k, t in SETS[:7]:
        a, b = E1[k], E[k]
        def cell(e, *path, f=pct):
            v = g(e, *path)
            return f(v) if v is not None else "n/a"
        rows.append([t,
                     f"{cell(a, 'auc', 'value', f=f3)} / {cell(b, 'auc', 'value', f=f3)}",
                     f"{cell(a, 'eer', 'value', f=f3)} / {cell(b, 'eer', 'value', f=f3)}",
                     f"{cell(a, 'at_yellow', 'detect_ever', 'value')} / {cell(b, 'at_yellow', 'detect_ever', 'value')}",
                     f"{cell(a, 'at_red', 'detect_ever', 'value')} / {cell(b, 'at_red', 'detect_ever', 'value')}",
                     f"{cell(a, 'at_yellow', 'false_alarm_ever', 'value')} / {cell(b, 'at_yellow', 'false_alarm_ever', 'value')}",
                     f"{cell(a, 'at_red', 'false_alarm_ever', 'value')} / {cell(b, 'at_red', 'false_alarm_ever', 'value')}"])
    rows.append(["Family-emergency TEST batch", f"{f3(fam1['auc']['value'])} / {f3(fam['auc']['value'])}",
                 f"{f3(fam1['eer']['value'])} / {f3(fam['eer']['value'])}",
                 f"{pct(fam1['detect_ever_yellow'])} / {pct(fam['at_yellow']['detect_ever']['value'])}",
                 f"{pct(fam1['detect_ever_red'])} / {pct(fam['at_red']['detect_ever']['value'])}",
                 f"{pct(fam1['false_alarm_ever_yellow'])} / {pct(fam['at_yellow']['false_alarm_ever']['value'])}",
                 f"{pct(fam1['false_alarm_ever_red'])} / {pct(fam['at_red']['false_alarm_ever']['value'])}"])
    add(table(["Set (v1 / v2)", "AUC", "EER", "Scams ever yellow", "Scams ever red", "Normal ever yellow (false alarm)",
               "Normal ever red (false alarm)"], rows))
    add("")
    add("Confidence intervals for every version 2 number are in the results tables below; version 1's are in the appendix.")
    add("")

    # ---------------- family data
    fo = A["family_batch_overlap"]
    fl = {r["source"]: r for r in A["format_and_length"]}
    rr = {r["source"]: r for r in A["receiver_says_scam_or_fraud"]}
    add("## The family-emergency calls (new in version 2)")
    add("")
    add("**Who wrote them.** Six separate AI writing sessions, each writing the calls directly (no generator scripts) in the same "
        "`caller:` / `receiver:` format. Four wrote the TRAIN batch (each 30 scam + 30 normal from its own 3 + 3 "
        "scenarios); two wrote the TEST batch (each 20 + 20) from a different scenario list and a different set of names. "
        "Every writer wrote both classes, so writing style cannot stand in for the label. Writers were told to read no "
        "other file, so the test writers never saw the training calls and no writer saw the app or its demo calls. The "
        "scenario lists and writing rules were written by the agent running this study, who *had* read the demo calls "
        "earlier for the version 1 sanity run, so the themes overlap with the demo by design (the demo's lead scenario is "
        "a grandparent bail scam). The coordinator asked for both batches written in order; they were written in "
        "parallel instead, with the test writers isolated from the training files, which serves the same purpose.")
    add("")
    add("**Rules given to writers (both classes).** Same relationship words and openings, same mix of tones, 3 to 14 "
        "caller turns split evenly short / medium / long. Scams: varied payment (gift cards, wire, cash courier or "
        "overnight mail, crypto ATM, Zelle / Venmo / Cash App), secrecy, urgency, excuses for sounding different, a "
        "second person taking over, about a quarter with a subtle ask. Normal calls: about 1 in 10 innocently mention "
        "sounding different, and the hard negatives mention money, trouble, a hospital, a new number or a bank innocently.")
    add("")
    add(f"**Sizes.** Training {sp['of_which_family_train']} (from 16 scenarios), validation {sz['val_family']['n']} "
        f"(8 held-out scenarios, chosen before any model saw the data), test {sz['family_test']['n']} "
        f"({sz['family_test']['scam']} scam / {sz['family_test']['nonscam']} normal, 16 scenarios).")
    add("")
    add(table(["Scenario", "Class", "Batch", "What it is"],
              [[k, "scam" if k[1] == "S" else "normal",
                "test" if k.startswith("X") else ("validation" if k in ("TS03", "TS06", "TS09", "TS11", "TN03", "TN06", "TN09", "TN11") else "train"),
                v] for k, v in SCEN.items()]))
    add("")
    add(f"**Overlap check.** Word 5-gram Jaccard from each test call to its nearest train-batch call: max "
        f"{fo['test_to_nearest_train']['max']:.3f}, median {fo['test_to_nearest_train']['median']:.3f}; "
        f"{fo['test_to_nearest_train']['n_ge_0.3']} of {fo['test_to_nearest_train']['n_test']} at 0.3 or more "
        f"(near-copy level is 0.5). Inside the train batch the nearest-neighbor max is "
        f"{fo['train_to_nearest_other_train']['max']:.3f}. No copying, by this measure.")
    add("")
    add(f"**Leaks in the family data.** Family scam callers talk longer: median caller words "
        f"{fl['family']['caller_words_median_scam']:.0f} vs {fl['family']['caller_words_median_nonscam']:.0f}, and caller "
        f"word count alone gives AUC {fl['family']['auc_caller_length_only']} (all family calls). The receiver says "
        f"\"scam\" or \"fraud\" in {pct(rr['family']['receiver_says_scam_scam'])} of scams and "
        f"{pct(rr['family']['receiver_says_scam_nonscam'])} of normal calls, so that leak is absent (and receiver lines "
        "are ignored anyway). Writer names: the TRAIN-batch names are in the name list that becomes `name`; the TEST "
        "names are deliberately not, so to the model they are unknown words, as a real caller's name would be.")
    add("")

    # ---------------- model choice
    add("## Model choice (validation only)")
    add("")
    add("Training pool: BothBosu (de-duplicated, as in version 1) plus the family training scenarios. Two variants: "
        "**A**, TF-IDF + logistic regression with Platt calibration (version 1's model), and **D**, the same plus the "
        "number of distinct rule-tagged tactics as a second input to the calibration layer. D's tactic count enters only "
        "the small calibration model fit on validation data, because on BothBosu the rules alone already reach AUC 0.98, "
        "and putting the count into the big model would let it lean on that.")
    add("")
    add(f"Selection rule, fixed before the run: {R['selection_rule']}.")
    add("")
    rows = [[r["variant"], r["C"], r["family_weight"], f3(r["val_auc_full"]["val"]), f3(r["val_auc_mean_over_fracs"]["val"]),
             f3(r["val_auc_full"]["val_family"]), f3(r["val_auc_mean_over_fracs"]["val_family"]), f"{r['score']:.4f}"]
            for r in R["variant_comparison"]]
    add(table(["Variant", "C", "Family weight", "shakeleoatmeal val AUC (full)", "shakeleoatmeal val AUC (mean over prefixes)",
               "Family val AUC (full)", "Family val AUC (mean over prefixes)", "Score"], rows))
    add("")
    add(f"**Chosen: {ch['variant']}, C = {ch['C']}, family weight {ch['family_weight']}.**")
    add("")
    abl = R["ablations_on_validation"]
    add("Text-choice ablations on validation (raw model score AUC, full call / mean over prefixes):")
    add("")
    add(table(["Input", "shakeleoatmeal val", "Family val"],
              [[k, f"{f3(v['val']['val_auc_full'])} / {f3(v['val']['val_auc_mean_over_fracs'])}",
                f"{f3(v['val_family']['val_auc_full'])} / {f3(v['val_family']['val_auc_mean_over_fracs'])}"]
               for k, v in abl.items()]))
    add("")

    # ---------------- calibration and thresholds
    add("## Calibration, likelihood ratio, thresholds")
    add("")
    add(f"Calibration layer fit on {cal['fit_on']}: `p = 1 / (1 + exp(-(a z + c k + b)))` with a = {m['calibration']['a']}, "
        f"c = {m['calibration']['c']} (tactic count k), b = {m['calibration']['b']}; `llr = logit(p) - logit(prior)`, prior = "
        f"{m['prior']:.4f}. Cross-fitted Brier / ECE: shakeleoatmeal val {f3(cal['crossfit_brier_val'])} / "
        f"{f3(cal['crossfit_ece_val'])}, family val {f3(cal['crossfit_brier_val_family'])} / {f3(cal['crossfit_ece_val_family'])}; "
        f"Brier on training-pool out-of-fold scores {f3(cal['brier_on_training_pool_oof'])}.")
    add("")
    add(f"Threshold rule: {th['rule']}.")
    add("")
    add(table(["Band", "Training pool OOF", "shakeleoatmeal val", "Family val", "Shipped (strictest)", "Version 1"],
              [[b, th[f"{b}_from_training_pool_oof"], th[f"{b}_from_val"], th[f"{b}_from_val_family"], th[b], th1[b]]
               for b in ("yellow", "red")]))
    add("")
    add("Calibration on the test sets (full call / pooled over prefixes):")
    add("")
    rows = []
    for k, t in SETS:
        e = E[k]
        if "brier_full" in e:
            rows.append([t, f"{f3(e['brier_full'])} / {f3(e['brier_prefixes'])}", f"{f3(e['ece_full'])} / {f3(e['ece_prefixes'])}",
                         f3(e["median_p_full_scam"]), f3(e["median_p_full_nonscam"])])
    add(table(["Set", "Brier", "ECE", "Median p, scams", "Median p, normal"], rows))
    add("")

    # ---------------- results
    add("## Results (version 2)")
    add("")
    add("95% stratified bootstrap intervals (2,000 resamples; 1,000 for per-fraction AUCs; 500 for the two reference "
        "validation rows).")
    add("")
    add(table(["Set", "Calls (scam / normal)", "AUC", "EER", "Accuracy at p = 0.5", "Rules-only baseline AUC"],
              [[t, n_str(E[k]), vci(E[k].get("auc")), vci(E[k].get("eer")), vci(E[k].get("accuracy_at_0.5")),
                vci(E[k].get("rules_only_baseline_auc"))] for k, t in SETS]))
    add("")
    for tn in ("yellow", "red"):
        add(f"**{tn.capitalize()} (p >= {th[tn]})**")
        add("")
        add(table(["Set", "Scams caught (end)", "Scams caught (any time)", "False alarms (end)", "False alarms (any time)",
                   "Accuracy (end)"],
                  [[t, pci(E[k]["at_" + tn].get("detect_full")), pci(E[k]["at_" + tn].get("detect_ever")),
                    pci(E[k]["at_" + tn].get("false_alarm_full")), pci(E[k]["at_" + tn].get("false_alarm_ever")),
                    pci(E[k]["at_" + tn].get("accuracy_full"))] for k, t in SETS]))
        add("")
    add("### Family test batch by scenario (hard negatives marked HARD)")
    add("")
    add(table(["Scenario", "What it is", "Calls", "Yellow or red (end)", "Yellow or red (any time)", "Red (end)", "Red (any time)"],
              [[t.split(" ")[0], SCEN.get(t.split(" ")[0], ""), v["n"], pct(v["flag_yellow_full"]), pct(v["flag_yellow_ever"]),
                pct(v["flag_red_full"]), pct(v["flag_red_ever"])] for t, v in sorted(E["family_test"]["by_type"].items())]))
    add("")
    hn = [v for t, v in E["family_test"]["by_type"].items() if "HARD" in SCEN.get(t.split(" ")[0], "")]
    n_hn = sum(v["n"] for v in hn)
    add(f"Across the {n_hn} hard-negative test calls: ever yellow or red "
        f"{pct(sum(v['flag_yellow_ever'] * v['n'] for v in hn) / n_hn)}, ever red {pct(sum(v['flag_red_ever'] * v['n'] for v in hn) / n_hn)}.")
    add("")
    add("### Streaming")
    add("")
    add("Each call is cut at 10%, 20%, ... 100% of its words (both speakers count toward the length); the model scores the "
        "caller's words in that prefix.")
    add("")
    add("![AUC and red-flag rate vs share of the call heard](streaming_curve.png)")
    add("")
    keys = [("test_in", "in-domain"), ("shake_trainplustest", "shakeleoatmeal"), ("menaattia_all", "menaattia"),
            ("family_test", "family test")]
    hdr = ["% heard"] + [f"AUC {nm}" for _, nm in keys] + [f"scams yellow / red by now ({nm})" for _, nm in keys] + \
          [f"normal yellow / red by now ({nm})" for _, nm in keys]
    rows = []
    for i in range(10):
        row = [f"{int(round(E['test_in']['streaming'][i]['frac'] * 100))}%"]
        row += [vci(E[k]["streaming"][i]["auc"]) for k, _ in keys]
        row += [f"{pct(E[k]['streaming'][i]['detect_by_now_yellow'])} / {pct(E[k]['streaming'][i]['detect_by_now_red'])}" for k, _ in keys]
        row += [f"{pct(E[k]['streaming'][i]['false_alarm_by_now_yellow'])} / {pct(E[k]['streaming'][i]['false_alarm_by_now_red'])}" for k, _ in keys]
        rows.append(row)
    add(table(hdr, rows))
    add("")
    add("Time to first yellow / red for scam calls (\"never\" when fewer than half ever got there):")
    add("")
    rows = []
    for k, t in SETS:
        e = E[k]
        if "time_to_red" not in e:
            continue
        r_ = [t]
        for tn in ("yellow", "red"):
            tt = e[f"time_to_{tn}"]
            r_ += [pct(tt["share_of_scams_ever_reaching"]), median_heard(tt["median_fraction_heard"]),
                   frac_heard(tt["median_fraction_heard_among_detected"])]
        rows.append(r_)
    add(table(["Set", "Ever yellow", "Median % heard to yellow [CI]", "Median, among those that got there",
               "Ever red", "Median % heard to red [CI]", "Median, among those that got there"], rows))
    add("")

    # ---------------- explanations
    TR = R["tactic_rule_fire_rates"]
    labels = {t["id"]: t["label"] for t in m["tactics"]}
    add("## Tactic tags (rule-based, not machine learning)")
    add("")
    add(f"Same hand-written rules as version 1. In version 2 the *count* of distinct tags is an input to the calibration "
        f"layer only if variant D was chosen (c = {m['calibration']['c']}). Fire rates on the caller's words (scam / normal):")
    add("")
    rows = []
    for tid in list(labels) + ["any"]:
        rr_ = [labels.get(tid, "any rule")]
        for k in ("train", "shake_trainplustest", "menaattia_all", "family_test"):
            v = TR[k][tid]
            rr_.append(f"{pct(v['scam'])} / {pct(v['nonscam'])}")
        rows.append(rr_)
    add(table(["Rule", "Training pool", "shakeleoatmeal", "menaattia", "Family test"], rows))
    add("")
    tn_ = R["top_ngrams"]
    add(table(["Most scam-weighted n-grams", "weight"], [[f"`{t['ngram']}`", f"{t['weight']:+.3f}"] for t in tn_["scam"][:30]]))
    add("")
    add(table(["Most safe-weighted n-grams", "weight"], [[f"`{t['ngram']}`", f"{t['weight']:+.3f}"] for t in tn_["safe"][:30]]))
    add("")

    # ---------------- demo
    meta = R["demo_calls_meta"]
    add("## Demo-call sanity run (not evidence)")
    add("")
    add(f"Scored after each line of the `SC` scripts in `atelier-securities.html` (SC block sha256 "
        f"`{meta['sc_block_sha256'][:16]}...`; skipped for having no scripted caller lines: "
        f"{', '.join(meta['skipped_no_scripted_caller_lines']) or 'none'}). These scripts were written with AI help, and "
        "the family training data was written by AI to a scenario list drafted by an agent that had read them, so a good "
        "score here would prove little and a bad one is still worth knowing. Share of each demo call's caller word "
        "4-grams that also appear somewhere in the family training batch: "
        + ", ".join(f"{k} {pct(v)}" for k, v in meta["share_of_caller_4grams_seen_in_family_train"].items())
        + ". Nothing was changed because of these results.")
    add("")
    for call, variants in R["demo_calls"].items():
        for vn, steps in variants.items():
            add(f"**{call}** ({vn.replace('_', ' ')})")
            add("")
            add(table(["Who", "Line", "p", "LLR", "Band", "Tactic tags so far"],
                      [[s_["who"], s_["text"].replace("|", "/"), f"{s_['p']:.3f}", f"{s_['llr']:+.2f}", s_["band"],
                        ", ".join(s_["tactics"])] for s_ in steps]))
            add("")

    # ---------------- export
    add("## Export and parity")
    add("")
    add(f"- `model.json`: {R['model_json_bytes']:,} bytes, {len(m['vocab'])} n-grams, calibration a / c / b, prior, "
        "thresholds, tokenizer spec and tactic rules. Weights rounded to 6 significant digits and put back into the "
        "scikit-learn model before evaluation.")
    add("- `stage2_model.js`: unchanged API (`scoreTranscript`, `scoreLines`, `scoreDialogue` -> `{p, llr, band, tactics}`); "
        "it now adds `c * (number of tactic tags)` when model.json has a `c`.")
    add(f"- `parity_test.js`: {P['cases']} cases (all test sets incl. the family test batch, full calls through the JS "
        f"parser plus 10% and 50% caller prefixes). Max |p| difference {P['max_abs_diff_p']:.2e}, max |llr| difference "
        f"{P['max_abs_diff_llr']:.2e}, tactic mismatches {P['tactic_mismatches']}, band mismatches {P['band_mismatches']}, "
        f"demo steps {P['demo_steps']}. **{'PASS' if P['pass'] else 'FAIL'}**.")
    add("- `test_scores.csv`: per-call p / llr at 10%..100% (same prefixes as the streaming curve) for the in-domain test, "
        "shakeleoatmeal test split, menaattia test split, and the family test batch (`source = family_test`). Nothing in "
        "it was used for training, model choice, calibration or thresholds. No dialogue text.")
    add("")

    # ---------------- limitations
    add("## Limitations")
    add("")
    add("- **The family data is AI-written, by the same model family for training and test.** Separate writers, scenarios "
        "and names keep the test batch from being a copy, but not from sharing one model's idea of how these calls go. "
        "The family test number is an upper bound on what to expect from real calls.")
    add("- **The rest of the training data is LLM-written too** (BothBosu), with the topic confound described in the "
        "version 1 audit: its scams and normal calls are about different subjects.")
    add("- **Tuning used shakeleoatmeal validation and held-out family scenarios.** menaattia and YouTube were used only "
        "for the final test.")
    add(f"- **Only {yt['n']} real-call transcripts**, all scams, and no real normal calls, so no real-world false-alarm "
        "rate can be claimed.")
    add("- **Clean transcripts, not live speech-to-text.** Recognition errors are not modeled.")
    add("")
    add("## Files")
    add("")
    add("| File | What it is |")
    add("|---|---|")
    for f_, d_ in [("fetch_data.py", "downloads the public datasets at pinned commits"),
                   ("family_check.py", "format checker the family-call writers ran"),
                   ("common.py", "parsing, normalization, n-grams, tactic rules, data loading (incl. family calls)"),
                   ("neardup.py", "word 5-gram Jaccard near-duplicate detection"),
                   ("audit.py / audit.json / audit_output.txt", "leakage audit (incl. family batch overlap)"),
                   ("train.py / results.json / train_log.txt", "model choice, calibration, thresholds, evaluation, export"),
                   ("model.json / stage2_model.js", "the shipped model and the browser / node scorer"),
                   ("parity_test.js / parity_result.json", "JS vs scikit-learn check"),
                   ("test_scores.csv", "per-call streaming scores on held-out test sets, for the fusion study"),
                   ("streaming_curve.png", "AUC and red-flag rates vs share of the call heard"),
                   ("v1/", "version 1 outputs (public data only)"),
                   ("make_report.py", "writes this file")]:
        add(f"| `{f_}` | {d_} |")
    add("")
    add("Raw data, the family calls and the parity fixture (which contain dataset text) live in `research/data/stage2/`.")
    add("")
    add("---")
    add("")
    add("# Appendix: version 1 (public data only), unchanged")
    add("")
    add(demote(open(os.path.join(HERE, "v1", "RESULTS.md")).read()))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    out = build()
    with open(os.path.join(HERE, "RESULTS.md"), "w") as f:
        f.write(out)
    print(f"wrote RESULTS.md ({len(out)} chars)")

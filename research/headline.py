"""Collect every number the app, the video and the written answers quote into
results/headline.json, rounded the way they are shown. Quote from this file,
never from memory."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"


def load(name):
    p = R / name
    return json.load(open(p)) if p.exists() else None


def r0(x):
    return round(100 * x)


def r1(x):
    return round(100 * x, 1)


def main():
    out = {}
    h = load("human_baseline.json")
    out["human"] = {"judgments": h["judgments"], "listeners": h["listeners"],
                    "fakes_missed_pct": r0(h["fakes_missed"]), "real_called_fake_pct": r0(h["real_called_fake"])}

    s = load("survival.json")["features"]
    keep = lambda f, c: s[f]["asv_train"][c]
    out["survival"] = {f: {"studio": round(keep(f, "clean"), 2),
                           "phone_worst_train_codec": round(min(keep(f, "g711"), keep(f, "opus")), 2),
                           "survives": s[f]["survives"]}
                       for f in ["f0_std", "f0_range", "shimmer", "flatness_mean", "pause_floor", "pause_flatness",
                                 "pause_rate", "pause_frac", "hf_ratio", "syll_rate", "jitter"]}
    out["survival"]["n_survivors_bio_spectral"] = len(load("survival.json")["survivors_bio_spectral"])

    b = load("benchmark_ssl.json")
    runs = b["runs"]

    def m(name, corpus, cond, key):
        v = runs[name]["test"][corpus][cond][key]
        return r1(v[0]) if isinstance(v, list) else r0(v)

    out["voice"] = {}
    for name in ["cepstral_clean", "ssl_clean", "ssl_aug", "all_aug", "bio_spectral_aug"]:
        out["voice"][name] = {corpus: {c: {"eer": m(name, corpus, c, "eer"), "false_alarm": m(name, corpus, c, "false_alarm"),
                                           "miss": m(name, corpus, c, "miss")}
                                       for c in ["clean", "g711", "opus", "g723"]}
                              for corpus in ["asv_test", "itw"]}
    mc = b["mcnemar"]["ssl_aug vs ssl_clean"]
    out["voice"]["mcnemar_aug_vs_clean"] = {k: {"acc_aug": r1(v["acc_a"]), "acc_clean": r1(v["acc_b"]), "p": v["p"]}
                                            for k, v in mc.items()}
    sel = load("ssl_layer_selection.json")
    out["voice"]["wavlm_layer"] = sel["best_layer"]

    f = load("fusion.json")
    fam = f["runs"]["itw_half/family"]
    out["fusion"] = {
        "cutoffs": {k: round(v, 2) for k, v in f["cutoffs"]["fused"].items() if k in ("warn", "stop")},
        "learned_weights": {k: round(v, 2) for k, v in f["learned_weights"].items() if k in ("voice", "words", "bias")},
        "family_auc": {q: {k: round(v["auc"][0], 2) for k, v in fam["by_prefix"][q].items()} for q in ("10", "30", "50", "100")},
        "family_catch_at_5pct_fa": {q: {k: {kk: r0(vv) for kk, vv in v["catch_at_5pct_false_alarm"].items()}
                                        for k, v in fam["by_prefix"][q].items()} for q in ("30", "100")},
        "family_ever": {name: {kind: {k: r0(v) for k, v in d.items()} for kind, d in fam[f"ever_reached_{name}"].items()}
                        for name in ("fused", "words_only")},
        "cascade_family": {name: {kind: {k: r0(v) for k, v in d.items()} for kind, d in c.items() if isinstance(d, dict)}
                           for name, c in fam["cascade"].items()},
        "cascade_lr": {k: round(v, 3) for k, v in fam["cascade"]["family_knows_0.9_scammer_knows_0.01"].items() if k.startswith("lr_")},
        "external_ever_fused": {kind: {k: r0(v) for k, v in d.items()}
                                for kind, d in f["runs"]["itw_half/external"]["ever_reached_fused"].items()},
    }
    out["fusion"]["family_cluster_auc_gain_fused_minus_words"] = {
        q: [round(x, 3) for x in fam["cluster_auc"]["by_prefix"][q]["fused_minus_words"]] for q in ("10", "30", "50", "100")}
    ext = f["runs"]["itw_half/external"]
    out["fusion"]["external_cluster_auc_gain_fused_minus_words"] = {
        q: [round(x, 3) for x in ext["cluster_auc"]["by_prefix"][q]["fused_minus_words"]] for q in ("10", "30", "50", "100")}
    out["fusion"]["family_matched_red"] = {k: ({kk: r0(vv) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 3))
                                           for k, v in fam["matched_red"].items()}
    cc = load("cluster_ci.json")["results"]
    out["voice_g723_asv"] = {name: {k: (r0(v) if not isinstance(v, list) else [r0(x) for x in v])
                                    for k, v in cc[f"asv_test/g723/{name}"].items()}
                             for name in ("studio_trained", "studio_trained_recalibrated", "phone_trained")}
    out["voice_g723_asv"]["phone_minus_studio"] = {k: (round(100 * v, 1) if not isinstance(v, list) else [round(100 * x, 1) for x in v])
                                                   for k, v in cc["asv_test/g723/phone_minus_studio"].items()}
    out["voice_itw_eer_phone_minus_studio"] = {c: [round(100 * x, 1) for x in cc[f"itw/{c}/phone_minus_studio"]["eer_diff_ci"]]
                                               for c in ("clean", "g711", "opus", "g723")}
    av = load("app_voice_values.json")
    out["app_voice_llr"] = {k: round(av[k], 3) for k in ("clone", "human", "poorline")}
    s2 = json.load(open(ROOT / "stage2" / "results_v21.json"))
    ev = s2["evaluation"]
    out["stage2"] = {
        "auc_full": {k: round(ev[k]["v21_family"]["auc_full"]["value"], 3) for k in ("family_test", "shake_trainplustest", "menaattia_all")},
        "family_test_shipped": {k: (r0(v) if isinstance(v, float) and v <= 1 else v)
                                for k, v in ev["family_test"]["v21_family"].items() if not isinstance(v, dict)},
        "v1_family_auc": round(json.load(open(ROOT / "stage2" / "v2" / "results.json"))["v1_on_family"]["family_test"]["auc"]["value"], 3),
    }
    json.dump(out, open(R / "headline.json", "w"), indent=1)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    sys.exit(main())

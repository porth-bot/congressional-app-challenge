"""Print the Stage 1 tables for RESULTS.md straight from the result JSON files,
so no number is typed by hand."""
import json
import sys

from common import CONDITIONS, RESULTS

NAMES = {
    "cepstral_clean": "Standard cepstral features (MFCC/LFCC), studio-trained",
    "all_clean": "All 125 handcrafted features, studio-trained",
    "bio_spectral_clean": "Voice-biology + spectral features, studio-trained",
    "cepstral_aug": "Standard cepstral features, phone-trained",
    "all_aug": "All 125 handcrafted features, phone-trained",
    "bio_spectral_aug": "Voice-biology + spectral features, phone-trained",
    "survivors_aug": "Codec survivors only (20 features), phone-trained",
    "survivors_bio_spectral_aug": "Surviving biology features only (2), phone-trained",
    "ssl_clean": "WavLM embedding, studio-trained",
    "ssl_aug": "WavLM embedding, phone-trained",
}
SHORT = {"clean": "Studio", "g711": "Landline", "opus": "VoIP", "g723": "Cellular*"}


def pct(x):
    return f"{100 * x:.1f}"


def table(runs, corpus, metric):
    head = "| Model | " + " | ".join(SHORT[c] for c in CONDITIONS) + " |"
    lines = [head, "|---|" + "---|" * len(CONDITIONS)]
    for name, r in runs.items():
        cells = []
        for c in CONDITIONS:
            v = r["test"][corpus][c][metric]
            cells.append(f"{pct(v[0])} ({pct(v[1])}-{pct(v[2])})" if isinstance(v, list) else pct(v))
        lines.append(f"| {NAMES.get(name, name)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "benchmark"
    b = json.load(open(RESULTS / f"{tag}.json"))
    runs = b["runs"]
    for corpus, title in [("asv_test", "ASVspoof 2019 eval"), ("itw", "In-the-Wild")]:
        print(f"\n#### {title}: equal error rate, % (95% CI)\n")
        print(table(runs, corpus, "eer"))
        print(f"\n#### {title}: real voices flagged as fake at the training threshold, %\n")
        print(table(runs, corpus, "false_alarm"))
        print(f"\n#### {title}: fake voices missed at the training threshold, %\n")
        print(table(runs, corpus, "miss"))
    print("\n#### Cross-validated EER on ASVspoof train (speaker-grouped 5-fold), %\n")
    for name, r in runs.items():
        print(f"- {NAMES.get(name, name)}: {pct(r['cv_eer'])}")
    print("\n#### McNemar tests (exact), accuracy at each model's training threshold\n")
    for pair, res in b["mcnemar"].items():
        for k, v in res.items():
            if k.endswith(("g723", "g711")):
                print(f"- {pair}, {k}: {pct(v['acc_a'])}% vs {pct(v['acc_b'])}%, p = {v['p']:.2g}")


if __name__ == "__main__":
    sys.exit(main())

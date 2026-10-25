"""How well do people hear deepfakes? Recomputed from the public data of
Mueller & Choong (2026), "Human Audio Deepfake Perception 2026"
(huggingface.co/datasets/mueller91/human-perception-audio-deepfake-2026,
CC BY-NC 4.0): 35,532 real-or-fake judgments by 1,768 listeners.

Writes results/human_baseline.json.
"""
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/mueller91/human-perception-audio-deepfake-2026/resolve/main/data.csv"


def main():
    path = ROOT / "data" / "human" / "data.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    df = pd.read_csv(path)
    fake, real = df[df.true_label == "fake"], df[df.true_label == "real"]
    out = {
        "judgments": int(len(df)),
        "listeners": int(df.uid.nunique()),
        "fake_clips_judged": int(len(fake)),
        "real_clips_judged": int(len(real)),
        "fakes_caught": float((fake.user_decision == "fake").mean()),
        "fakes_missed": float((fake.user_decision == "real").mean()),
        "real_called_fake": float((real.user_decision == "fake").mean()),
        "overall_accuracy": float((df.user_decision == df.true_label).mean()),
    }
    json.dump(out, open(ROOT / "results" / "human_baseline.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    sys.exit(main())

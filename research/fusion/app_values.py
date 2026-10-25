"""The voice evidence used by the app's scripted test calls.

The test calls have no audio, so each one uses the voice model's typical output
for that kind of caller: the median calibrated log-likelihood ratio (computer-made
vs real) on In-the-Wild phone-codec clips. We take the medians from the
calibration half of the speakers, so no test clip is involved.

Writes results/app_voice_values.json; the app's VOICE_LLR copies these values.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion import ROOT, RESULTS, voice_llrs  # noqa: E402


def main():
    _, cal, info = voice_llrs(ROOT / "data" / "benchmark_ssl_scores.parquet", "ssl_aug", "itw_half")
    out = {
        "clone": float(np.median(cal[cal.y == 1].llr)),
        "human": float(np.median(cal[cal.y == 0].llr)),
        "poorline": float(np.median(cal[cal.cond == "g723"].llr)),
        "source": "median calibrated LLR on the In-the-Wild calibration-half speakers (no test clips)",
        "n": {"computer_made": int((cal.y == 1).sum()), "real": int((cal.y == 0).sum()), "g723": int((cal.cond == "g723").sum())},
        "calibration": info,
    }
    json.dump(out, open(RESULTS / "app_voice_values.json", "w"), indent=1)
    print(json.dumps({k: round(v, 3) for k, v in out.items() if isinstance(v, float)}))


if __name__ == "__main__":
    sys.exit(main())

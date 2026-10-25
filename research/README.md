# Liveline research

This folder holds everything behind Liveline's numbers: three studies, the scripts that run them, and the results. The findings are in [RESULTS.md](RESULTS.md). This file explains what each study asks and how to rerun it.

## The three studies

| Study | Question | Code | Main output |
|---|---|---|---|
| 1. Voice on phone audio | Which voice cues survive a phone codec, and does training on phone-quality audio make a deepfake detector hold up? | `stage1/` | `results/benchmark_ssl.json`, `results/cluster_ci.json`, `results/fig_false_alarms_asv_test.png` |
| 2. Scam words | Can a small model that runs in the browser spot scam tactics in a transcript, including on datasets it never saw? | `stage2/` | `stage2/results_v21.json`, `stage2/RESULTS.md` |
| 3. Putting them together | Does combining voice and words beat either one, and what does the safe-word step add? | `fusion/` | `results/fusion.json`, `results/fig_fusion.png` |

`headline.py` collects every number the app, the video, and the written answers quote into `results/headline.json`. Quote from there, never from memory.

## Setup

```bash
cd research
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

You also need ffmpeg (`brew install ffmpeg`) and Node (for the JavaScript check). Downloaded audio, transcripts, and features go in `research/data/`, which is not committed; the scripts fetch everything from public sources.

## Rerun everything

Run from `research/`. The slow step is the WavLM embedding, about an hour on an M-series Mac.

```bash
# Study 1: voice
.venv/bin/python stage1/fetch_data.py      # 6,000 clips: ASVspoof 2019 train + eval, In-the-Wild
.venv/bin/python stage1/codecs.py          # landline, VoIP, and cellular-style copies of every clip
.venv/bin/python stage1/features.py        # 125 hand-measured voice features per clip
.venv/bin/python stage1/survival.py        # which features survive the codecs
.venv/bin/python stage1/ssl_embed.py       # WavLM speech features
.venv/bin/python stage1/select_layer.py    # pick the WavLM layer on training data only
.venv/bin/python stage1/benchmark.py --extra ssl
.venv/bin/python stage1/cluster_ci.py      # cut-off-only control and speaker-level intervals
.venv/bin/python stage1/figures.py '{"tag":"benchmark_ssl","models":{"cepstral_clean":["Standard features, studio-trained","#afb8c1"],"ssl_clean":["WavLM, studio-trained","#6e7781"],"ssl_aug":["Liveline: WavLM, phone-trained","#2a78d6"]}}'

# Study 2: words
cd stage2
../.venv/bin/python fetch_data.py
../.venv/bin/python audit.py               # checks the data for copies and label leaks
../.venv/bin/python train.py               # version 2 model
../.venv/bin/python calibrate_v21.py       # version 2.1 calibration (shipped)
node parity_v21.js                         # browser JavaScript matches Python
cd ..

# Study 3: together
.venv/bin/python fusion/fusion.py '{"n_calls":4000}'
.venv/bin/python fusion/fusion_fig.py
.venv/bin/python fusion/app_values.py      # the app's voice constants

# numbers and app
.venv/bin/python human/human_baseline.py
.venv/bin/python headline.py
.venv/bin/python embed_model.py            # copy the words model into the app
```

The family-emergency calls for Study 2 were written by AI and live in `data/stage2/family/`. They are not downloaded by `fetch_data.py`.

## Rules we kept

- **No test data touches a decision, with one exception.** Feature selection, the WavLM layer, model settings, calibration, and every cut-off come from training or validation data. The exception: after the version 2 words model raised no alarms on the family test calls, we switched to calibrating it on family validation calls (version 2.1). That fit used validation data only, but the decision came from a test result, so the family numbers are somewhat optimistic.
- **The cellular-style codec (G.723.1) is never used for training,** so it doubles as an unseen-codec test.
- **Silence is trimmed before measuring anything,** because ASVspoof 2019 leaks the label through the length of the silence around each clip.
- **The words model only sees what the caller says,** because in the public datasets the other speaker's lines give the label away.
- **Intervals resample speakers (and transcripts), not clips,** for the headline comparisons, because clips share speakers and recordings.
- **Copies are removed.** Two of the public transcript sets share 1,127 identical calls; 177 test calls were near-copies of training calls and were dropped.
- **Results that went against the plan are reported.** The breath and pause features failed, and the first fusion version made things worse. Both are in RESULTS.md.

## Folder map

| Path | Contents |
|---|---|
| `stage1/fetch_data.py` | Reads only the needed parts of the remote dataset files (HTTP range requests) |
| `stage1/codecs.py` | ffmpeg encode/decode round trips: G.711, Opus 8 kbps, G.723.1 |
| `stage1/features.py` | Pitch, jitter, shimmer, pauses, breath, rhythm, spectral shape, MFCC, LFCC |
| `stage1/benchmark.py` | Trains each model, reports EER, AUC, 95% bootstrap intervals, false alarms, McNemar tests |
| `stage2/train.py`, `calibrate_v21.py` | TF-IDF and logistic regression, then calibration that grows with how much of the call has been heard |
| `stage2/stage2_model.js` | The same model in plain browser JavaScript |
| `fusion/fusion.py` | Builds paired test calls, combines the stages, tunes the app's cut-offs on validation calls, simulates the safe-word step |
| `results/` | Figures and JSON results |

# Liveline Stage 2: scam-tactic text model

Stage 2 reads what the caller says and scores how much it sounds like a scam. It is a small TF-IDF plus logistic regression model that runs in plain browser JavaScript (`stage2_model.js`, no libraries) and returns a calibrated probability, a log-likelihood ratio for the Bayesian fusion step, a green / yellow / red band, and rule-based tactic tags for the plain-language "why".

Every number below is filled in by `make_report.py` from `results.json`, `audit.json` and `parity_result.json`, which `audit.py`, `train.py` and `parity_test.js` wrote. Nothing was tuned on a test set.

## The short version

- **In-domain it is perfect, and that means little.** On held-out BothBosu calls the AUC is 1.000 [1.000, 1.000] and the EER is 0.000 [0.000, 0.000]. The audit shows why: in BothBosu, scam calls and normal calls are about different topics (Social Security, refunds, tech support, prizes versus deliveries, insurance, appointments, wrong numbers), so topic alone separates them.
- **Across datasets it still ranks scams above normal calls most of the time.** Trained only on BothBosu, it scores AUC 0.927 [0.913, 0.939] on shakeleoatmeal (scam and normal versions of the *same* call types) and 0.917 [0.909, 0.925] on menaattia (short calls with many other scam types). The EERs are 0.140 [0.123, 0.161] and 0.168 [0.154, 0.178]. This is the honest generalization number, and it is a ranking number, not an alarm number.
- **Its probabilities do not transfer, so it rarely turns red outside its training data.** With the red threshold set for at most 2% false alarms on validation calls, it turned red on 11.4% of shakeleoatmeal scams at any point in the call, 0.0% of menaattia scams, and 0.0% of the 20 YouTube scam transcripts. It turned red on 1.5% and 0.0% of normal calls in those two sets.
- **Early detection, in-domain:** half the scams had turned yellow by 32% of the call (92.5% ever did), while 0.0% of normal calls ever turned yellow. Red is slower: 56.9% of scams ever reached it.
- **The app's demo calls all stay green.** Highest score reached: clone 0.23, real 0.14, unsure 0.13, medicare 0.47 (yellow starts at 0.76). The training data has no grandparent or family-emergency scams at all, which is the app's lead scenario.
- **Bottom line for the team:** use the LLR as one input to fusion, where a weak but real signal is still worth something, and do not show the text score alone as a verdict. The JavaScript port matches scikit-learn to 1.3e-15 in probability on 4908 cases.

## Rerun everything

```bash
cd research/stage2
../.venv/bin/python fetch_data.py    # download the six datasets (pinned commits) to research/data/stage2/
../.venv/bin/python audit.py         # leakage audit -> audit.json, audit_output.txt
../.venv/bin/python train.py         # about 16 min -> model.json, results.json, test_scores.csv, streaming_curve.png
node parity_test.js                  # JavaScript vs scikit-learn -> parity_result.json
../.venv/bin/python make_report.py   # this file
```

Seed 0 everywhere. Python 3.12.13, scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1, node v25.9.0.

## Data

| Dataset | License | What it is | Dialogues | Scam / normal | Call types | Speaker tags | How we used it |
|---|---|---|---|---|---|---|---|
| `BothBosu/scam-dialogue` | apache-2.0 | LLM-written phone dialogues (Llama-3-70B-instruct per its card) | 1600 (train 1280, test 320) | 800 / 800 | scam: refund, reward, ssn, support; normal: delivery, insurance, telemarketing, wrong | caller, receiver | training pool + in-domain test |
| `BothBosu/single-agent-scam-conversations` | apache-2.0 | LLM-written dialogues; mostly a copy of scam-dialogue with renamed speaker tags (see audit) | 1600 (train 1280, test 320) | 800 / 800 | scam: refund, reward, ssn, support; normal: appointment, delivery, insurance, wrong | innocent, suspect | training pool + in-domain test |
| `BothBosu/multi-agent-scam-conversation` | apache-2.0 | two LLM agents talking, receiver given one of 8 personalities; about twice as long | 1600 (train 1280, test 320) | 800 / 800 | scam: refund, reward, ssn, support; normal: appointment, delivery, insurance, wrong | innocent, suspect | training pool + in-domain test |
| `BothBosu/youtube-scam-conversations` | apache-2.0 | transcripts of 20 scambaiting YouTube videos (real scammers; the 'victim' is a baiter); scams only | 20 (all 20) | 20 / 0 | scam: refund, reward, ssn, support; normal: none | innocent, suspect | external test (detection rate only) |
| `menaattia/phone-scam-dataset` | none listed | very short dialogues (2 to 3 turns) covering many scam themes | 4000 (train 3200, test 800) | 2000 / 2000 | not given | caller, receiver | external test only, text never copied |
| `shakeleoatmeal/phone-scam-detection-synthetic` | mit | Llama-8B-written legit and fraud versions of the same three call types | 1800 (train 1259, validation 361, test 180) | 900 / 900 | scam: refund, ssn, support; normal: refund, ssn, support | caller, receiver | validation split = model choice, calibration, thresholds; train+test = external test |

Splits after de-duplication:

| Split | Dialogues | Scam | Normal |
|---|---|---|---|
| train | 2956 | 1408 | 1548 |
| test_in | 636 | 253 | 383 |
| test_in_strict | 598 | 253 | 345 |
| val | 361 | 181 | 180 |
| shake_test | 180 | 90 | 90 |
| shake_trainplustest | 1439 | 719 | 720 |
| menaattia_all | 4000 | 2000 | 2000 |
| menaattia_test | 800 | 400 | 400 |
| youtube | 20 | 20 | 0 |

## Leakage audit (done before any modeling)

`audit.py` checks duplicates, formatting, length, the receiver's lines and label-skewed words. Near-duplicates are measured with Jaccard similarity of word 5-gram shingles (speaker tags removed); 0.5 or more counts as a near-copy.

1. **single-agent is mostly a copy of scam-dialogue.** 1127 dialogues are word-for-word identical once `caller:`/`receiver:` is renamed to `Suspect:`/`Innocent:`. Cross-source near-copies by call type: delivery 200, insurance 200, refund 182, reward 190, ssn 177, support 189 (each set has 200 calls per type; the wrong-number, telemarketing and appointment calls are not shared). The copies cross the official splits: 81 scam-dialogue test calls are copies of single-agent training calls and 78 the other way round. scam-dialogue also repeats itself (115 near-copy pairs inside its training split, 39 between its test and training splits; 157 of its 158 internal near-copy pairs are templated wrong-number calls). Pooled naively, **177 of 960 test calls would be near-copies of a training call.** Fix: training keeps one call per near-duplicate cluster (3840 to 2956); the test set drops the 177 near-copies of training calls and 147 duplicates inside the test split, leaving 636. A stricter version that also drops test calls sharing a looser template (Jaccard 0.3 or more) with any training call keeps 598. No near-duplicate pair had conflicting labels (0).
2. **Topic gives the label away in BothBosu.** In all three sets the scam call types (ssn, refund, support, reward) never appear as normal calls, and the normal types (delivery, insurance, appointment, telemarketing, wrong number) never appear as scams. A model can learn "insurance means safe" instead of learning tactics. Cleaning cannot fix this, which is why the in-domain result is saturated and why the cross-dataset sets matter: in shakeleoatmeal the normal and scam calls share the same three types.
3. **Speaker tags.** `Suspect:` and `Innocent:` mark the caller and the person answering in *both* scam and normal calls (so they are not a direct label), but they are dataset-specific formatting. We parse the turns, drop the tags, and use only the caller's words.
4. **The receiver's lines leak.** In multi-agent, the person answering says "scam" or "fraud" in 47.6% of scam calls and 4.9% of normal calls; in scam-dialogue it is reversed (19.4% vs 30.9%). A model given only the receiver's lines gets grouped-CV AUC 0.9998. The generator knew the label when it wrote the reactions. So the model only sees what the caller says, which is also all the app should rely on.
5. **Made-up names and greetings.** The generator reused the same fake people unevenly: in scam-dialogue "Johnson" is in 36.9% of scam callers vs 3.1% of normal callers, "XYZ" in 2.9% vs 53.5% and "hello" appears in the caller's lines of 639 of 640 scam calls vs 256 of 640 normal calls. Fix: a fixed list of generated person names and placeholder companies becomes the word `name`, and hi / hey / hello become `hello` (the list is in `common.py` and `model.json`). Real organizations and titles (Microsoft, Walmart, Social Security, "officer") are kept, because impersonating them is a real tactic.
6. **Scam callers talk longer in BothBosu.** Median caller words, scam vs normal: 169 vs 112 (scam-dialogue), 175 vs 95 (single-agent), 368 vs 192 (multi-agent). Caller word count alone gives AUC 0.839, 0.909 and 0.794, but 0.458 on shakeleoatmeal and 0.493 on menaattia. TF-IDF vectors are length-normalized, and the streaming test scores partial calls, so length is not what the model is graded on.
7. **External sets barely overlap the training data:** dialogues with any training neighbor at Jaccard 0.3 or more: shakeleoatmeal 4, menaattia 0, youtube 0.

Effect of the two text choices on the validation split (not on any test set):

| Input | Validation AUC, full call | Validation AUC, mean over 10%..100% |
|---|---|---|
| chosen (caller words, names/greetings neutralized) | 0.915 | 0.782 |
| caller words, names/greetings NOT neutralized | 0.913 | 0.786 |
| both speakers (receiver lines included) | 0.901 | 0.774 |

Dropping the receiver's lines raises validation AUC by +0.014 (full call). Neutralizing names and greetings changes it by +0.003 on full calls and -0.004 averaged over prefixes, so it is not why the model works; we kept it because it removes a known artifact.

## Model and how it was chosen

Caller's words, then `normalize()` (lowercase, drop apostrophes, digits become `num`, everything else outside a-z becomes a space, names and greetings neutralized), then word unigrams and bigrams, TF-IDF (tf = 1 + ln count, smooth idf, L2 norm, min_df 2, the 5,000 most frequent n-grams), logistic regression (L2), then Platt scaling.

Three variants times six regularization strengths C were compared with grouped 5-fold cross-validation on the training pool (near-duplicates kept in the same fold) and by refitting on the whole pool and scoring the shakeleoatmeal validation split at 10%, 20%, ... 100% of each call.

Selection rule, fixed in `train.py` before any test set was scored: over every (variant, C): max mean shakeleoatmeal/validation AUC over 10%..100% prefixes, among pairs with grouped-CV AUC >= 0.99; pairs within 0.005 of the best count as tied and go to the simplest (no char n-grams, then no prefix augmentation, then smallest C).

| Variant | C | Features | CV AUC (full call) | CV log-loss (all prefixes) | Val AUC (full call) | Val AUC (mean over prefixes) |
|---|---|---|---|---|---|---|
| A_word12 | 0.1 | 5000 | 1.0000 | 0.2856 | 0.915 | 0.782 |
| A_word12 | 0.3 | 5000 | 1.0000 | 0.1718 | 0.915 | 0.784 |
| A_word12 | 1.0 | 5000 | 1.0000 | 0.0958 | 0.913 | 0.784 |
| A_word12 | 3.0 | 5000 | 1.0000 | 0.0583 | 0.913 | 0.785 |
| A_word12 | 10.0 | 5000 | 1.0000 | 0.0361 | 0.907 | 0.782 |
| A_word12 | 30.0 | 5000 | 1.0000 | 0.0277 | 0.904 | 0.780 |
| B_word12_prefixaug | 0.1 | 5000 | 1.0000 | 0.1227 | 0.876 | 0.757 |
| B_word12_prefixaug | 0.3 | 5000 | 1.0000 | 0.0619 | 0.867 | 0.752 |
| B_word12_prefixaug | 1.0 | 5000 | 1.0000 | 0.0292 | 0.857 | 0.745 |
| B_word12_prefixaug | 3.0 | 5000 | 1.0000 | 0.0156 | 0.851 | 0.741 |
| B_word12_prefixaug | 10.0 | 5000 | 1.0000 | 0.0108 | 0.866 | 0.746 |
| B_word12_prefixaug | 30.0 | 5000 | 1.0000 | 0.0091 | 0.863 | 0.744 |
| C_word12_char35 | 0.1 | 6000 | 1.0000 | 0.1787 | 0.897 | 0.767 |
| C_word12_char35 | 0.3 | 6000 | 1.0000 | 0.1090 | 0.897 | 0.768 |
| C_word12_char35 | 1.0 | 6000 | 1.0000 | 0.0671 | 0.893 | 0.764 |
| C_word12_char35 | 3.0 | 6000 | 1.0000 | 0.0517 | 0.887 | 0.760 |
| C_word12_char35 | 10.0 | 6000 | 1.0000 | 0.0358 | 0.890 | 0.762 |
| C_word12_char35 | 30.0 | 6000 | 1.0000 | 0.0363 | 0.887 | 0.760 |

**Chosen: A_word12 with C = 0.1.** Every setting reaches CV AUC 1.0000 because the BothBosu classes are separable, and CV log-loss always prefers the weakest regularization, so in-domain CV cannot pick a model. Prefix augmentation (B) and character n-grams (C) were both worse on the validation set than plain word n-grams (A), and the A settings were within the 0.005 tie band, so the rule took the most regularized one.

## Calibration and the likelihood ratio

Two Platt calibrators were compared, each fit on scores at 10%, 20%, ... 100% of every call:

| Calibrator | a | b | Brier on its own source (cross-fit halves) | Brier on the other source | ECE own / other |
|---|---|---|---|---|---|
| fit on in-domain out-of-fold scores | 9.746 | 3.206 | 0.005 | 0.408 | 0.002 / 0.413 |
| fit on shakeleoatmeal validation | 1.920 | -0.826 | 0.201 | 0.050 | 0.039 / 0.142 |

Rule: lower mean of the two out-of-sample Brier scores (own source cross-fitted in halves, other source scored directly). Means: in-domain fit 0.207, validation fit 0.125. **Chosen: validation (shakeleoatmeal/validation, 10%..100% prefixes).** The in-domain calibrator learns that BothBosu is easy and becomes wildly overconfident on anything else. Isotonic regression, for comparison, got cross-fit Brier 0.202 on validation; Platt was kept because it is smooth and gives a finite LLR.

Shipped formula: `z = coef . x + intercept`, `p = 1 / (1 + exp(-(a z + b)))` with a = 1.92032, b = -0.826323, and `llr = logit(p) - logit(prior)` with prior = 0.5014 (the scam share of the calibration set). For fusion: the LLR is the evidence from the words alone; multiply exp(llr) into the app's own prior odds.

Calibration on the test sets (full call / pooled over 10%..100% prefixes):

| Set | Brier | ECE (10 bins) | Median p, scams | Median p, normal calls |
|---|---|---|---|---|
| In-domain test (BothBosu, de-duplicated) | 0.013 / 0.035 | 0.076 / 0.108 | 0.872 | 0.022 |
| In-domain test, strict de-dup | 0.013 / 0.037 | 0.079 / 0.114 | 0.872 | 0.024 |
| shakeleoatmeal test split | 0.152 / 0.201 | 0.166 / 0.069 | 0.774 | 0.446 |
| shakeleoatmeal train+test | 0.151 / 0.197 | 0.179 / 0.054 | 0.764 | 0.435 |
| menaattia test split | 0.304 / 0.328 | 0.333 / 0.335 | 0.216 | 0.090 |
| menaattia train+test | 0.303 / 0.329 | 0.336 / 0.337 | 0.214 | 0.091 |
| shakeleoatmeal validation (tuning set, for reference) | 0.156 / 0.201 | 0.163 / 0.034 | 0.762 | 0.452 |

The probabilities are roughly right on shakeleoatmeal (same generator as the calibration set) and badly off on menaattia: its scams get a median p of 0.214 against 0.091 for its normal calls, so they still rank higher but almost never reach a threshold. We have not isolated why; the likely reasons are that menaattia's calls are only a couple of sentences and cover scam types BothBosu does not have (investment tips, customs fees, tax threats, account verification).

## Thresholds

Rule: smallest p (3 decimals) such that at most 10% (yellow) / 2% (red) of validation non-scam calls EVER reach it at any 2% checkpoint of the call; stricter of in-domain OOF and shakeleoatmeal/validation.

| Band starts at | From in-domain OOF | From shakeleoatmeal validation | Shipped (stricter) |
|---|---|---|---|
| yellow | 0.189 | 0.76 | 0.76 |
| red | 0.229 | 0.873 | 0.873 |

The validation source sets both thresholds: its normal calls are legitimate refund, SSN and support calls that sound a lot like scams, so keeping false alarms that low forces high thresholds.

## Results

All intervals are 95% stratified bootstrap intervals (2,000 resamples; 1,000 for the per-fraction AUCs).

### How well it separates scams from normal calls (full call)

| Set | Calls (scam / normal) | AUC | EER | Accuracy at p = 0.5 | Rules-only baseline AUC |
|---|---|---|---|---|---|
| In-domain test (BothBosu, de-duplicated) | 636 (253 / 383) | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.997 [0.992, 1.000] | 0.979 [0.969, 0.988] |
| In-domain test, strict de-dup | 598 (253 / 345) | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.997 [0.992, 1.000] | 0.977 [0.966, 0.987] |
| shakeleoatmeal test split | 180 (90 / 90) | 0.924 [0.886, 0.958] | 0.139 [0.094, 0.206] | 0.789 [0.733, 0.844] | 0.798 [0.738, 0.852] |
| shakeleoatmeal train+test | 1439 (719 / 720) | 0.927 [0.913, 0.939] | 0.140 [0.123, 0.161] | 0.798 [0.780, 0.817] | 0.751 [0.729, 0.774] |
| menaattia test split | 800 (400 / 400) | 0.910 [0.890, 0.930] | 0.180 [0.152, 0.205] | 0.516 [0.509, 0.526] | 0.783 [0.757, 0.807] |
| menaattia train+test | 4000 (2000 / 2000) | 0.917 [0.909, 0.925] | 0.168 [0.154, 0.178] | 0.515 [0.511, 0.519] | 0.800 [0.789, 0.810] |
| YouTube scambait transcripts (scams only) | 20 (20 / 0) | n/a | n/a | n/a | n/a |
| shakeleoatmeal validation (tuning set, for reference) | 361 (181 / 180) | 0.915 [0.886, 0.942] | 0.147 [0.105, 0.180] | 0.767 [0.730, 0.806] | 0.771 [0.731, 0.813] |

The rules-only baseline scores a call by how many tactic rules fire. The model beats it on every cross-dataset set.

### At the yellow and red thresholds

"End" is the score after the whole call; "any time" means the band was reached at some 2% checkpoint during the call, which is what a person watching the meter would see.

**Yellow (p >= 0.76)**

| Set | Scams caught (end) | Scams caught (any time) | False alarms (end) | False alarms (any time) | Accuracy (end) |
|---|---|---|---|---|---|
| In-domain test (BothBosu, de-duplicated) | 89.7% [85.8%, 93.3%] | 92.5% [89.3%, 95.7%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 95.9% [94.5%, 97.3%] |
| In-domain test, strict de-dup | 89.7% [85.8%, 93.3%] | 92.5% [89.3%, 95.7%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 95.7% [94.1%, 97.2%] |
| shakeleoatmeal test split | 53.3% [43.3%, 63.3%] | 61.1% [51.1%, 71.1%] | 1.1% [0.0%, 3.3%] | 8.9% [3.3%, 14.4%] | 76.1% [70.6%, 81.1%] |
| shakeleoatmeal train+test | 51.2% [47.3%, 54.8%] | 59.5% [55.9%, 62.9%] | 2.9% [1.8%, 4.2%] | 7.8% [5.8%, 9.7%] | 74.1% [72.2%, 76.1%] |
| menaattia test split | 0.0% [0.0%, 0.0%] | 0.2% [0.0%, 0.8%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 50.0% [50.0%, 50.0%] |
| menaattia train+test | 0.0% [0.0%, 0.0%] | 0.4% [0.1%, 0.6%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 50.0% [50.0%, 50.0%] |
| YouTube scambait transcripts (scams only) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | n/a | n/a | n/a |
| shakeleoatmeal validation (tuning set, for reference) | 50.8% [43.4%, 57.5%] | 56.4% [49.7%, 63.5%] | 4.4% [1.7%, 7.8%] | 10.0% [6.1%, 14.4%] | 73.1% [69.5%, 77.0%] |

**Red (p >= 0.873)**

| Set | Scams caught (end) | Scams caught (any time) | False alarms (end) | False alarms (any time) | Accuracy (end) |
|---|---|---|---|---|---|
| In-domain test (BothBosu, de-duplicated) | 49.8% [43.9%, 56.1%] | 56.9% [50.6%, 63.2%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 80.0% [77.5%, 82.4%] |
| In-domain test, strict de-dup | 49.8% [43.9%, 56.1%] | 56.9% [51.0%, 63.2%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 78.8% [76.1%, 81.4%] |
| shakeleoatmeal test split | 6.7% [2.2%, 12.2%] | 12.2% [5.6%, 18.9%] | 0.0% [0.0%, 0.0%] | 1.1% [0.0%, 3.3%] | 53.3% [51.1%, 56.1%] |
| shakeleoatmeal train+test | 7.9% [6.1%, 10.0%] | 11.4% [9.2%, 13.8%] | 0.0% [0.0%, 0.0%] | 1.5% [0.7%, 2.5%] | 54.0% [53.0%, 55.0%] |
| menaattia test split | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 50.0% [50.0%, 50.0%] |
| menaattia train+test | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | 50.0% [50.0%, 50.0%] |
| YouTube scambait transcripts (scams only) | 0.0% [0.0%, 0.0%] | 0.0% [0.0%, 0.0%] | n/a | n/a | n/a |
| shakeleoatmeal validation (tuning set, for reference) | 9.4% [5.0%, 13.8%] | 16.6% [11.3%, 22.1%] | 0.0% [0.0%, 0.0%] | 1.7% [0.0%, 3.3%] | 54.6% [52.6%, 56.7%] |

### Streaming: how fast it catches a scam

Each call is cut at 10%, 20%, ... 100% of its words (both speakers count toward the length), and the model scores the caller's words inside that prefix.

![AUC and red-flag rate vs share of the call heard](streaming_curve.png)

Left: AUC with 95% bootstrap bands; the dotted line is chance. Right: share of calls that have turned red at any point so far. The dashed false-alarm lines for in-domain (0.0%) and menaattia (0.0%) lie under the solid menaattia line at zero.

| % heard | AUC in-domain | yellow / red by now, scams (in-domain) | yellow / red by now, normal (in-domain) | AUC shakeleoatmeal | yellow / red by now, scams (shakeleoatmeal) | yellow / red by now, normal (shakeleoatmeal) | AUC menaattia | yellow / red by now, scams (menaattia) | yellow / red by now, normal (menaattia) |
|---|---|---|---|---|---|---|---|---|---|
| 10% | 0.998 [0.995, 0.999] | 14.2% / 5.9% | 0.0% / 0.0% | 0.582 [0.553, 0.611] | 0.6% / 0.0% | 1.5% / 0.0% | 0.549 [0.532, 0.567] | 0.1% / 0.0% | 0.0% / 0.0% |
| 20% | 1.000 [1.000, 1.000] | 30.4% / 13.8% | 0.0% / 0.0% | 0.663 [0.635, 0.690] | 4.5% / 0.3% | 2.9% / 0.7% | 0.549 [0.531, 0.567] | 0.2% / 0.0% | 0.0% / 0.0% |
| 30% | 1.000 [1.000, 1.000] | 49.0% / 21.7% | 0.0% / 0.0% | 0.726 [0.699, 0.754] | 13.4% / 1.1% | 4.6% / 1.1% | 0.640 [0.625, 0.657] | 0.2% / 0.0% | 0.0% / 0.0% |
| 40% | 1.000 [1.000, 1.000] | 60.5% / 29.6% | 0.0% / 0.0% | 0.766 [0.740, 0.789] | 22.9% / 2.8% | 5.7% / 1.2% | 0.737 [0.722, 0.752] | 0.2% / 0.0% | 0.0% / 0.0% |
| 50% | 1.000 [1.000, 1.000] | 71.9% / 34.8% | 0.0% / 0.0% | 0.795 [0.773, 0.817] | 28.7% / 4.0% | 6.4% / 1.2% | 0.808 [0.794, 0.822] | 0.4% / 0.0% | 0.0% / 0.0% |
| 60% | 1.000 [1.000, 1.000] | 80.6% / 37.9% | 0.0% / 0.0% | 0.830 [0.809, 0.852] | 35.7% / 5.3% | 6.9% / 1.4% | 0.851 [0.840, 0.863] | 0.4% / 0.0% | 0.0% / 0.0% |
| 70% | 1.000 [1.000, 1.000] | 85.4% / 46.6% | 0.0% / 0.0% | 0.859 [0.840, 0.878] | 43.4% / 6.7% | 7.4% / 1.4% | 0.882 [0.871, 0.892] | 0.4% / 0.0% | 0.0% / 0.0% |
| 80% | 1.000 [1.000, 1.000] | 88.5% / 50.6% | 0.0% / 0.0% | 0.884 [0.867, 0.902] | 50.5% / 8.5% | 7.4% / 1.5% | 0.901 [0.892, 0.911] | 0.4% / 0.0% | 0.0% / 0.0% |
| 90% | 1.000 [1.000, 1.000] | 90.9% / 54.2% | 0.0% / 0.0% | 0.909 [0.892, 0.923] | 55.8% / 9.7% | 7.6% / 1.5% | 0.912 [0.903, 0.920] | 0.4% / 0.0% | 0.0% / 0.0% |
| 100% | 1.000 [1.000, 1.000] | 92.5% / 56.9% | 0.0% / 0.0% | 0.927 [0.914, 0.940] | 59.5% / 11.4% | 7.8% / 1.5% | 0.917 [0.908, 0.926] | 0.4% / 0.0% | 0.0% / 0.0% |

Time to the first yellow / red, for scam calls ("never" when fewer than half the scams ever got there):

| Set | Ever yellow | Median % heard to yellow [CI] | Median, among those that got there | Ever red | Median % heard to red [CI] | Median, among those that got there |
|---|---|---|---|---|---|---|
| In-domain test (BothBosu, de-duplicated) | 92.5% | 32% [28%, 36%] | 30% | 56.9% | 78% [68%, 100%] | 38% |
| In-domain test, strict de-dup | 92.5% | 32% [28%, 36%] | 30% | 56.9% | 78% [68%, 100%] | 38% |
| shakeleoatmeal test split | 61.1% | 73% [56%, 97%] | 42% | 12.2% | never [never, never] | 66% |
| shakeleoatmeal train+test | 59.5% | 80% [74%, 86%] | 52% | 11.4% | never [never, never] | 65% |
| menaattia test split | 0.2% | never [never, never] | 42% | 0.0% | never [never, never] | n/a |
| menaattia train+test | 0.4% | never [never, never] | 12% | 0.0% | never [never, never] | n/a |
| YouTube scambait transcripts (scams only) | 0.0% | never [never, never] | n/a | 0.0% | never [never, never] | n/a |
| shakeleoatmeal validation (tuning set, for reference) | 56.4% | 92% [78%, never] | 53% | 16.6% | never [never, never] | 65% |

### By call type

**In-domain test (BothBosu, de-duplicated)**

| Call type | Calls | Yellow or red (end) | Yellow or red (any time) | Red (end) | Red (any time) |
|---|---|---|---|---|---|
| appointment (nonscam) | 77 | 0.0% | 0.0% | 0.0% | 0.0% |
| delivery (nonscam) | 80 | 0.0% | 0.0% | 0.0% | 0.0% |
| insurance (nonscam) | 80 | 0.0% | 0.0% | 0.0% | 0.0% |
| refund (scam) | 83 | 85.5% | 89.2% | 37.3% | 48.2% |
| reward (scam) | 54 | 85.2% | 90.7% | 31.5% | 44.4% |
| ssn (scam) | 61 | 100.0% | 100.0% | 98.4% | 98.4% |
| support (scam) | 55 | 89.1% | 90.9% | 32.7% | 36.4% |
| telemarketing (nonscam) | 40 | 0.0% | 0.0% | 0.0% | 0.0% |
| wrong (nonscam) | 106 | 0.0% | 0.0% | 0.0% | 0.0% |

**shakeleoatmeal train+test**

| Call type | Calls | Yellow or red (end) | Yellow or red (any time) | Red (end) | Red (any time) |
|---|---|---|---|---|---|
| refund (nonscam) | 240 | 0.0% | 1.2% | 0.0% | 0.0% |
| refund (scam) | 240 | 56.2% | 63.3% | 9.2% | 11.7% |
| ssn (nonscam) | 240 | 8.8% | 22.1% | 0.0% | 4.6% |
| ssn (scam) | 239 | 55.6% | 66.9% | 9.6% | 15.5% |
| support (nonscam) | 240 | 0.0% | 0.0% | 0.0% | 0.0% |
| support (scam) | 240 | 41.7% | 48.3% | 5.0% | 7.1% |

In-domain, 100.0% of SSN scams reach yellow and no normal call of any type is flagged. On shakeleoatmeal, the hardest normal calls are the legitimate SSN calls (22.1% turn yellow at some point), and the hardest scams are the tech-support ones (48.3% ever turn yellow).

## Explanations: rule-based tactic tags (not machine learning)

The "why" comes from hand-written keyword rules in `common.py` (copied into `model.json` and run by `stage2_model.js`). They never change the score. How often each rule fires on the caller's words:

| Rule | Training pool (scam / normal) | shakeleoatmeal (scam / normal) | menaattia (scam / normal) | YouTube scams |
|---|---|---|---|---|
| Rushing you or saying it is urgent | 57.5% / 11.6% | 53.4% / 15.3% | 10.5% / 0.3% | 25.0% |
| Telling you to keep it secret | 0.9% / 0.0% | 0.1% / 0.0% | 0.1% / 0.0% | 0.0% |
| Asking for gift cards | 26.7% / 1.2% | 0.6% / 0.3% | 2.2% / 0.0% | 5.0% |
| Asking for a wire transfer or money app | 4.1% / 0.0% | 1.1% / 0.1% | 1.8% / 0.0% | 0.0% |
| Asking for Bitcoin or crypto | 0.0% / 0.0% | 0.0% / 0.0% | 1.0% / 0.0% | 5.0% |
| Asking you to send money or pay a fee | 25.8% / 0.1% | 1.4% / 0.1% | 6.9% / 0.0% | 0.0% |
| Talking about arrest, bail, or legal trouble | 10.4% / 0.4% | 2.6% / 0.0% | 5.5% / 0.1% | 5.0% |
| Claiming to be the government, your bank, or a big company | 72.4% / 7.0% | 67.3% / 68.3% | 16.5% / 0.9% | 50.0% |
| Asking for personal numbers or passwords | 74.9% / 8.7% | 93.2% / 55.3% | 36.4% / 0.7% | 20.0% |
| Telling you not to hang up or call back | 19.5% / 0.3% | 0.8% / 0.0% | 0.1% / 0.0% | 0.0% |
| Wanting remote access to your computer | 22.4% / 0.1% | 6.0% / 2.8% | 2.1% / 0.0% | 15.0% |
| any rule | 99.4% / 24.7% | 99.2% / 86.8% | 61.5% / 1.8% | 75.0% |

The impersonation and personal-info rules fire on most legitimate shakeleoatmeal calls too (a real bank or support line also says who it is and asks to verify you), so those two tags explain rather than accuse. The crypto rule never fires in the training pool: BothBosu has no crypto scams.

Does a flagged call come with a reason? (end-of-call score)

| Set | Calls flagged yellow or red | Flagged calls with at least one tag | Green calls with at least one tag |
|---|---|---|---|
| In-domain test (BothBosu, de-duplicated) | 227 | 99.6% | 28.6% |
| In-domain test, strict de-dup | 227 | 99.6% | 31.3% |
| shakeleoatmeal test split | 49 | 100.0% | 96.2% |
| shakeleoatmeal train+test | 389 | 100.0% | 90.4% |
| menaattia test split | 0 | n/a | 29.2% |
| menaattia train+test | 0 | n/a | 31.6% |
| YouTube scambait transcripts (scams only) | 0 | n/a | 75.0% |
| shakeleoatmeal validation (tuning set, for reference) | 100 | 100.0% | 91.2% |

Model weights versus the rules: only 2.5% of the 40 most scam-weighted n-grams match a rule on their own, partly because the rules need longer phrases ("social security" alone does not match the "social security number" rule). Many of the top n-grams are about the same things as the rules (Social Security numbers, computers, verifying accounts, gift cards), and many are style words the generator gave its scammers ("but", "assure you", "ma'am", "uh"). The safe side is dominated by the BothBosu normal-call topics (insurance, delivery, appointments, wrong numbers), which is the topic confound from the audit showing up in the weights.

| Most scam-weighted n-grams | weight | rule match |
|---|---|---|
| `refund` | +0.892 |  |
| `process` | +0.806 |  |
| `social security` | +0.790 |  |
| `security` | +0.788 |  |
| `social` | +0.776 |  |
| `security number` | +0.719 |  |
| `your computer` | +0.715 |  |
| `your social` | +0.707 |  |
| `need` | +0.701 |  |
| `the refund` | +0.688 |  |
| `but` | +0.682 |  |
| `card` | +0.640 |  |
| `need to` | +0.637 |  |
| `computer` | +0.633 |  |
| `we need` | +0.605 |  |
| `gift card` | +0.579 | gift_cards |
| `verify` | +0.577 |  |
| `been` | +0.573 |  |
| `to verify` | +0.569 |  |
| `is a` | +0.553 |  |
| `information` | +0.553 |  |
| `account` | +0.546 |  |
| `issue` | +0.541 |  |
| `your account` | +0.536 |  |
| `this is` | +0.534 |  |
| `gift` | +0.527 |  |
| `now` | +0.527 |  |
| `weve` | +0.525 |  |
| `maam` | +0.520 |  |
| `information to` | +0.519 |  |
| `system` | +0.512 |  |
| `process the` | +0.511 |  |
| `assure` | +0.510 |  |
| `assure you` | +0.510 |  |
| `please` | +0.504 |  |
| `dont` | +0.498 |  |
| `prize` | +0.497 |  |
| `uh` | +0.491 |  |
| `identity` | +0.477 |  |
| `you please` | +0.476 |  |

| Most safe-weighted n-grams | weight |
|---|---|
| `insurance` | -0.829 |
| `num pm` | -0.726 |
| `pm` | -0.726 |
| `from name` | -0.663 |
| `sorry` | -0.653 |
| `delivery` | -0.631 |
| `appointment` | -0.617 |
| `at num` | -0.498 |
| `i was` | -0.491 |
| `package` | -0.486 |
| `scheduled` | -0.479 |
| `wrong` | -0.460 |
| `dialed` | -0.453 |
| `offering a` | -0.452 |
| `so sorry` | -0.442 |
| `offering` | -0.440 |
| `hello im` | -0.438 |
| `im so` | -0.433 |
| `were offering` | -0.428 |
| `your appointment` | -0.424 |
| `the wrong` | -0.424 |
| `wrong number` | -0.418 |
| `dr` | -0.416 |
| `have dialed` | -0.412 |
| `tomorrow` | -0.402 |
| `must have` | -0.401 |
| `oh` | -0.400 |
| `dr name` | -0.393 |
| `i must` | -0.382 |
| `dialed the` | -0.377 |
| `today` | -0.375 |
| `your package` | -0.372 |
| `great` | -0.370 |
| `sorry i` | -0.369 |
| `im` | -0.369 |
| `thanks` | -0.351 |
| `ha` | -0.346 |
| `coverage` | -0.342 |
| `i think` | -0.340 |
| `for` | -0.340 |

## Sanity run on the app's scripted demo calls

Scored after each line of the `SC` scripts in `atelier-securities.html` (SC block sha256 `11b35918c8691405...`; skipped because they have no scripted caller lines: live). The model only reads caller lines, so the score does not move on Rose's lines. Nothing was changed because of these results.

**clone** (main)

| Who | Line | p | LLR | Band | Tactic tags so far |
|---|---|---|---|---|---|
| caller | Grandma? It’s me, Ethan. | 0.230 | -1.21 | green |  |
| caller | I’m in real trouble. I got arrested and I need bail money tonight. | 0.103 | -2.16 | green | urgency, legal_trouble |
| rose | Oh no. Are you hurt? Let me call your mother. | 0.103 | -2.16 | green | urgency, legal_trouble |
| caller | Please don’t tell Mom. She can’t find out. | 0.127 | -1.93 | green | urgency, secrecy, legal_trouble |
| caller | The lawyer says you can pay with gift cards from the pharmacy. | 0.182 | -1.51 | green | urgency, secrecy, gift_cards, legal_trouble |
| caller | Can you go now? I only have a few minutes. | 0.207 | -1.35 | green | urgency, secrecy, gift_cards, legal_trouble |

**real** (main)

| Who | Line | p | LLR | Band | Tactic tags so far |
|---|---|---|---|---|---|
| caller | Hi Grandma, it’s Ethan. My phone died, so I’m using Jake’s. | 0.137 | -1.85 | green |  |
| caller | Are we still on for pancakes Saturday? Mom said she’ll drive me over. | 0.104 | -2.16 | green |  |
| rose | Of course, sweetheart. Nine o’clock? | 0.104 | -2.16 | green |  |
| caller | Nine works. Love you, see you then. | 0.088 | -2.34 | green |  |

**unsure** (with pass branch)

| Who | Line | p | LLR | Band | Tactic tags so far |
|---|---|---|---|---|---|
| caller | Grandma? It’s Ethan. Sorry, the signal here is really bad. | 0.076 | -2.51 | green |  |
| caller | I missed the last bus. Could you send me money for a ride home? | 0.086 | -2.36 | green | payment |
| rose | Where are you, honey? | 0.086 | -2.36 | green | payment |
| caller | Near the mall. I can explain later. | 0.093 | -2.28 | green | payment |
| caller | (whispering) {word}. | 0.093 | -2.28 | green | payment |
| rose | Okay. Let me call your dad and we’ll come get you. | 0.093 | -2.28 | green | payment |

**unsure** (with fail branch)

| Who | Line | p | LLR | Band | Tactic tags so far |
|---|---|---|---|---|---|
| caller | Grandma? It’s Ethan. Sorry, the signal here is really bad. | 0.076 | -2.51 | green |  |
| caller | I missed the last bus. Could you send me money for a ride home? | 0.086 | -2.36 | green | payment |
| rose | Where are you, honey? | 0.086 | -2.36 | green | payment |
| caller | Near the mall. I can explain later. | 0.093 | -2.28 | green | payment |
| caller | What? I can’t hear you. Please just send the money. | 0.131 | -1.90 | green | payment |

**medicare** (main)

| Who | Line | p | LLR | Band | Tactic tags so far |
|---|---|---|---|---|---|
| caller | Hello, this is the Medicare benefits office. Your benefits will be suspended today. | 0.212 | -1.32 | green | urgency, impersonation |
| caller | To stop that, I need to verify your Medicare number right now. | 0.378 | -0.51 | green | urgency, impersonation, personal_info |
| rose | Suspended? Why would it be suspended? | 0.378 | -0.51 | green | urgency, impersonation, personal_info |
| caller | There is a reactivation fee. You can pay it with gift cards. | 0.454 | -0.19 | green | urgency, gift_cards, payment, impersonation, personal_info |
| caller | Please stay on the line. Do not hang up. | 0.474 | -0.11 | green | urgency, gift_cards, payment, impersonation, personal_info, stay_on_line |

All demo calls stay green. The tactic rules do tag the scam lines (see the last column), but the model's score barely moves. The scripts are short (3 to 5 caller lines), and most of the words they lean on are not even in the model's vocabulary, because BothBosu never uses them. Weights of some of those n-grams (for scale, the largest scam weight is +0.892): `bail` not in vocabulary, `arrested` not in vocabulary, `lawyer` not in vocabulary, `grandma` not in vocabulary, `dont tell` not in vocabulary, `gift cards` +0.107, `gift card` +0.579, `medicare` not in vocabulary, `suspended` +0.202, `hang up` +0.126, `right now` +0.122, `fee` +0.440.

## Export and parity

- `model.json`: 154,671 bytes. 5000 n-grams with idf and weight, intercept, Platt a and b, prior, thresholds, tokenizer spec (steps, name and greeting lists, speaker tags) and the tactic rules. Weights and idf are rounded to 6 significant digits, and the rounded values were put back into the scikit-learn model before evaluation, so the evaluated model is the shipped model.
- `stage2_model.js`: `Stage2.load(model)`, then `scoreTranscript(callerWords)`, `scoreLines([{who, text}])` or `scoreDialogue(taggedTranscript)`, each returning `{p, llr, band, tactics}`. In the browser it loads itself from `window.LIVELINE_STAGE2_MODEL`.
- `parity_test.js`: 4908 cases (every in-domain test call, the shakeleoatmeal and menaattia test splits and the YouTube transcripts, each in full through the JS transcript parser, plus 10% and 50% caller prefixes). Max |p_js - p_sklearn| = 1.33e-15, max |llr difference| = 1.11e-14, tactic mismatches 0, band mismatches 0, demo steps checked 26. **PASS** (target < 1e-6).
- `test_scores.csv`: one row per held-out test call (in-domain test, shakeleoatmeal test split, menaattia test split; nothing used for training, CV, calibration or thresholds), with label, caller word count, and p / llr at 10%..100% using the same prefixes as the streaming curve. No dialogue text.

## Limitations

- **Almost all the training data is written by language models**, from a handful of prompt templates. The in-domain AUC of 1.0 says the model learned those templates, not that it can catch real scams.
- **The topic confound.** BothBosu scams and normal calls are about different things, so part of what the model learned is "refund or tech support sounds risky, deliveries and insurance sound safe". A real delivery scam or a real refund call would confuse it.
- **Missing scam types.** BothBosu's scams are SSN, refund, tech-support and prize calls. Across all 4800 BothBosu calls, "bail" appears in 0, crypto words in 0, and family words (grandma, grandson ...) in 18; we read all of those and in every one the relative is someone the caller mentions, never someone the caller pretends to be. The app's lead demo scenario is exactly the scam the model never saw.
- **Ranking transfers, probabilities do not.** Cross-dataset AUC of 0.917 to 0.927 means the score is informative, but the calibrated probability depends on the source and on how much of the call has been heard. Outside BothBosu it rarely leaves green. The LLR fed to fusion will therefore usually be small, which errs on the side of fewer false alarms and more missed scams.
- **Tuning used shakeleoatmeal validation**, so the shakeleoatmeal test numbers come from the same generator as the tuning data. menaattia and YouTube were never used for anything except the final test.
- **Only 20 real-call transcripts**, all scams, from scambaiting videos where the person answering is playing along. There is no real normal-call data at all, so no real-world false-alarm rate can be claimed.
- **Words, not audio.** These are clean transcripts. In the app, Stage 2 would read live speech-to-text output, with recognition errors this evaluation does not include.

## Files

| File | What it is |
|---|---|
| `fetch_data.py` | downloads the datasets at pinned commits |
| `common.py` | transcript parsing, text normalization, n-grams, tactic rules, data loading |
| `neardup.py` | word 5-gram Jaccard near-duplicate detection |
| `audit.py / audit.json / audit_output.txt` | leakage audit |
| `train.py / results.json / train_log.txt` | model selection, calibration, thresholds, evaluation, export |
| `model.json` | the shipped model |
| `stage2_model.js` | browser / node scorer |
| `parity_test.js / parity_result.json` | JS vs scikit-learn check |
| `test_scores.csv` | per-call streaming scores on held-out test sets, for the fusion study |
| `streaming_curve.png` | AUC and red-flag rates vs share of the call heard |
| `make_report.py` | writes this file |

Raw data and the parity fixture (which contains dataset text) live in `research/data/stage2/`, not here.

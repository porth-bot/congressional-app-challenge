"""Voice features for every clip in every channel condition.

Three families:
  bio      - cues from the physiology literature: pitch and its micro-variation,
             jitter, shimmer, harmonics-to-noise, voicing, pause and breath
             behaviour, syllable rate and rhythm
  spectral - frame-level spectral shape (centroid, flatness, tilt, high band)
  cepstral - MFCC and LFCC statistics, the standard anti-spoofing front ends

Every clip is trimmed of leading and trailing silence and RMS-normalised first.
ASVspoof 2019 LA is known to leak the label through the length of the silence
around each utterance (Mueller et al., Interspeech 2021), and we don't want the
pause features to learn that shortcut.

Writes data/features/<condition>.parquet.
"""
import csv
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
from parselmouth.praat import call
from scipy.fft import dct
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SR = 16000
HOP = 160  # 10 ms
WIN = 400  # 25 ms
NFFT = 512
CONDITIONS = ["clean", "g711", "opus", "g723"]

BIO = ["f0_mean", "f0_std", "f0_range", "f0_microvar", "jitter", "shimmer", "hnr", "voiced_frac",
       "pause_frac", "pause_rate", "pause_floor", "pause_flatness", "syll_rate", "rhythm_npvi", "env_mod"]
SPECTRAL = ["centroid_mean", "centroid_std", "rolloff_mean", "flatness_mean", "flatness_std",
            "flux_mean", "flux_std", "tilt_phoneband", "hf_ratio", "lf_ratio"]


def clip_path(rel, cond):
    p = ROOT / rel
    return p if cond == "clean" else ROOT / "data" / cond / p.parent.name / p.name


def lfcc_bank(n_filters=40):
    freqs = np.linspace(0, SR / 2, NFFT // 2 + 1)
    edges = np.linspace(0, SR / 2, n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return fb


LFCC_FB = lfcc_bank()
FREQS = np.linspace(0, SR / 2, NFFT // 2 + 1)


def safe(x):
    x = float(x)
    return x if np.isfinite(x) else np.nan


def prosody(y):
    snd = parselmouth.Sound(y.astype(np.float64), SR)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    voiced = f0 > 0
    out = {"voiced_frac": voiced.mean() if len(f0) else np.nan}
    if voiced.sum() >= 5:
        st = 12 * np.log2(f0[voiced] / 100.0)
        out["f0_mean"] = st.mean()
        out["f0_std"] = st.std()
        out["f0_range"] = np.percentile(st, 95) - np.percentile(st, 5)
        # frame-to-frame change inside voiced runs only
        st_all = np.where(voiced, 12 * np.log2(np.maximum(f0, 1) / 100.0), np.nan)
        d = np.abs(np.diff(st_all))
        out["f0_microvar"] = np.nanmean(d) if np.isfinite(d).any() else np.nan
    else:
        out.update(f0_mean=np.nan, f0_std=np.nan, f0_range=np.nan, f0_microvar=np.nan)
    try:
        pp = call([snd, pitch], "To PointProcess (cc)")
        out["jitter"] = safe(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        out["shimmer"] = safe(call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception:
        out["jitter"] = out["shimmer"] = np.nan
    try:
        harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        out["hnr"] = safe(call(harm, "Get mean", 0, 0))
    except Exception:
        out["hnr"] = np.nan
    return out, voiced


def runs(mask):
    """(start, length) of True runs."""
    m = np.concatenate([[0], mask.astype(int), [0]])
    d = np.diff(m)
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return list(zip(s, e - s))


def extract(y):
    y, _ = librosa.effects.trim(y, top_db=35, frame_length=WIN, hop_length=HOP)
    if len(y) < SR * 0.5:
        return None
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * 0.05
    f = {}
    pro, voiced = prosody(y)
    f.update(pro)

    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP, win_length=WIN)) ** 2  # (257, T)
    T = S.shape[1]
    rms = np.sqrt(S.sum(0) / NFFT + 1e-12)
    edb = 20 * np.log10(rms + 1e-9)
    edb -= np.percentile(edb, 95)
    speech = edb > -30

    # pauses: internal non-speech stretches of at least 150 ms
    pauses = [(s, l) for s, l in runs(~speech) if l >= 15 and s > 0 and s + l < T]
    dur = T / 100.0
    f["pause_frac"] = sum(l for _, l in pauses) / T
    f["pause_rate"] = len(pauses) / dur
    pmask = np.zeros(T, bool)
    for s, l in pauses:
        pmask[s:s + l] = True
    flat = librosa.feature.spectral_flatness(S=S, power=2.0)[0]
    if pmask.any():
        f["pause_floor"] = np.median(edb[pmask])  # breath and room noise between phrases
        f["pause_flatness"] = np.median(np.log10(flat[pmask]))
    else:
        f["pause_floor"] = f["pause_flatness"] = np.nan

    # syllable nuclei: intensity peaks while voiced (after de Jong & Wempe 2009)
    sm = np.convolve(edb, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(sm, prominence=2.0, distance=10)
    v = np.zeros(T, bool)
    v[: min(T, len(voiced))] = voiced[: min(T, len(voiced))]
    peaks = peaks[v[peaks] & speech[peaks]] if len(peaks) else peaks
    sp_dur = speech.sum() / 100.0
    f["syll_rate"] = len(peaks) / sp_dur if sp_dur > 0 else np.nan
    if len(peaks) >= 4:
        iv = np.diff(peaks).astype(float)
        f["rhythm_npvi"] = 100 * np.mean(np.abs(np.diff(iv)) / ((iv[1:] + iv[:-1]) / 2))
    else:
        f["rhythm_npvi"] = np.nan
    env = rms - rms.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    mf = np.fft.rfftfreq(len(env), d=0.01)
    band = (mf >= 2) & (mf <= 8)
    tot = (mf >= 0.5) & (mf <= 20)
    f["env_mod"] = spec[band].sum() / (spec[tot].sum() + 1e-12)

    Ss = S[:, speech] if speech.sum() >= 5 else S
    cen = librosa.feature.spectral_centroid(S=np.sqrt(Ss), sr=SR)[0]
    f["centroid_mean"], f["centroid_std"] = cen.mean(), cen.std()
    f["rolloff_mean"] = librosa.feature.spectral_rolloff(S=np.sqrt(Ss), sr=SR, roll_percent=0.85)[0].mean()
    fl = np.log10(flat[speech] if speech.sum() >= 5 else flat) + 0.0
    f["flatness_mean"], f["flatness_std"] = fl.mean(), fl.std()
    Sn = Ss / (Ss.sum(0, keepdims=True) + 1e-12)
    flux = np.sqrt((np.diff(Sn, axis=1) ** 2).sum(0)) if Sn.shape[1] > 1 else np.array([np.nan])
    f["flux_mean"], f["flux_std"] = flux.mean(), flux.std()
    pb = (FREQS >= 300) & (FREQS <= 3400)
    lp = 10 * np.log10(Ss[pb].mean(1) + 1e-12)
    f["tilt_phoneband"] = np.polyfit(np.log2(FREQS[pb]), lp, 1)[0]
    tot_e = Ss.sum() + 1e-12
    f["hf_ratio"] = 10 * np.log10(Ss[FREQS >= 4000].sum() / tot_e + 1e-12)
    f["lf_ratio"] = 10 * np.log10(Ss[FREQS < 300].sum() / tot_e + 1e-12)

    mel = librosa.feature.melspectrogram(S=Ss, sr=SR, n_mels=40, fmax=SR / 2)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    dm = librosa.feature.delta(mfcc, width=min(9, mfcc.shape[1] - (1 - mfcc.shape[1] % 2))) if mfcc.shape[1] >= 3 else np.zeros_like(mfcc)
    lf = dct(np.log(LFCC_FB @ Ss + 1e-10), type=2, axis=0, norm="ortho")[:20]
    for i in range(20):
        f[f"mfcc{i}_mean"], f[f"mfcc{i}_std"] = mfcc[i].mean(), mfcc[i].std()
        f[f"dmfcc{i}_std"] = dm[i].std()
        f[f"lfcc{i}_mean"], f[f"lfcc{i}_std"] = lf[i].mean(), lf[i].std()
    f["duration"] = len(y) / SR
    return f


def job(args):
    clip_id, rel, cond = args
    try:
        y, _ = sf.read(clip_path(rel, cond), dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        f = extract(y)
    except Exception as e:
        print("fail", clip_id, cond, e, flush=True)
        f = None
    if f is None:
        return None
    f["clip_id"] = clip_id
    return f


def main():
    conds = sys.argv[1:] or CONDITIONS
    rows = list(csv.DictReader(open(ROOT / "data" / "manifest.csv")))
    out = ROOT / "data" / "features"
    out.mkdir(parents=True, exist_ok=True)
    for cond in conds:
        jobs = [(r["clip_id"], r["path"], cond) for r in rows]
        feats = []
        with ProcessPoolExecutor(10) as ex:
            for k, f in enumerate(ex.map(job, jobs, chunksize=8)):
                if f is not None:
                    feats.append(f)
                if (k + 1) % 1000 == 0:
                    print(cond, k + 1, "/", len(jobs), flush=True)
        df = pd.DataFrame(feats)
        df.to_parquet(out / f"{cond}.parquet", index=False)
        print(cond, "rows", len(df), "features", df.shape[1] - 1, flush=True)


if __name__ == "__main__":
    sys.exit(main())

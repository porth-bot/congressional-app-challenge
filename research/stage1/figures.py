"""Figures for the write-up and the demo video. Everything is drawn from
results/*.json and the real audio; nothing is typed in by hand.

  fig_spectrogram.png  one real ASVspoof eval clip, studio vs cellular codec
  fig_survival.png     per-feature separation, studio vs phone
  fig_benchmark.png    EER by channel for the main models
  fig_false_alarms.png share of REAL voices flagged as fake at a fixed threshold
"""
import json
import sys

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from common import COND_NAMES, CONDITIONS, RESULTS, TRAIN_CODECS
from features import BIO, SPECTRAL, clip_path

INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
DARKGRAY = "#6e7781"
BLUE, GRAY, RED, GREEN = "#2a78d6", "#9aa4ae", "#e34948", "#1a7f37"
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

NICE = {
    "f0_mean": "Average pitch", "f0_std": "Pitch variability", "f0_range": "Pitch range",
    "f0_microvar": "Pitch micro-movement", "jitter": "Jitter (cycle-to-cycle pitch)",
    "shimmer": "Shimmer (cycle-to-cycle loudness)", "hnr": "Harmonics-to-noise", "voiced_frac": "Share of time voiced",
    "pause_frac": "Share of time paused", "pause_rate": "Pauses per second", "pause_floor": "Breath/noise level in pauses",
    "pause_flatness": "Breath noisiness in pauses", "syll_rate": "Syllable rate", "rhythm_npvi": "Rhythm regularity",
    "env_mod": "Syllable-rate loudness rhythm", "centroid_mean": "Spectral brightness", "centroid_std": "Brightness variation",
    "rolloff_mean": "Spectral roll-off", "flatness_mean": "Spectral flatness", "flatness_std": "Flatness variation",
    "flux_mean": "Spectral change", "flux_std": "Spectral change variation", "tilt_phoneband": "Spectral tilt (phone band)",
    "hf_ratio": "Energy above 4 kHz", "lf_ratio": "Energy below 300 Hz",
}


def spectrogram(clip="data/raw/asv_test/LA_E_8189776.flac"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, cond in zip(axes, ["clean", "g723"]):
        y, sr = sf.read(clip_path(clip, cond), dtype="float32")
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=128)), ref=np.max)
        t = np.arange(S.shape[1]) * 128 / sr
        f = np.linspace(0, sr / 2, S.shape[0]) / 1000
        ax.pcolormesh(t, f, S, vmin=-80, vmax=0, cmap="magma", shading="auto", rasterized=True)
        ax.set_title(COND_NAMES[cond], color=INK, loc="left")
        ax.set_xlabel("Seconds")
        ax.axhline(4, color="white", lw=1, ls=(0, (4, 3)), alpha=0.7)
    axes[0].set_ylabel("Frequency (kHz)")
    fig.savefig(RESULTS / "fig_spectrogram.png")
    plt.close(fig)


def app_spectrogram(clip="data/raw/asv_test/LA_E_8189776.flac"):
    """Small JPEG of the same real clip for the app's Signal page."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.8), sharey=True, dpi=100)
    for ax, cond, title in zip(axes, ["clean", "g723"], ["Studio", "Cellular-style phone codec"]):
        y, sr = sf.read(clip_path(clip, cond), dtype="float32")
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=128)), ref=np.max)
        t = np.arange(S.shape[1]) * 128 / sr
        f = np.linspace(0, sr / 2, S.shape[0]) / 1000
        ax.pcolormesh(t, f, S, vmin=-80, vmax=0, cmap="magma", shading="auto", rasterized=True)
        ax.set_title(title, loc="left", color=INK, fontsize=13)
        ax.set_xlabel("Seconds", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.axhline(4, color="white", lw=1, ls=(0, (4, 3)), alpha=0.7)
    axes[0].set_ylabel("kHz", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "app_spectrogram.jpg", dpi=100, pil_kwargs={"quality": 72, "optimize": True})
    plt.close(fig)


def survival():
    s = json.load(open(RESULTS / "survival.json"))["features"]
    rows = []
    for f in BIO + SPECTRAL:
        t = s[f]["asv_train"]
        phone = min(t[c] for c in TRAIN_CODECS)  # the same number the survival rule uses
        rows.append((f, t["clean"], phone, s[f]["survives"]))
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, (f, a, b, ok) in enumerate(rows):
        col = BLUE if ok else GRAY
        ax.plot([a, b], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter([a], [i], s=60, color="white", edgecolor=MUTED, lw=1.5, zorder=2)
        ax.scatter([b], [i], s=70, color=col, edgecolor="white", lw=2, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([NICE[r[0]] for r in rows])
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(0.2, color=BLUE, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("How well the feature alone separates real from fake\n(0 = coin flip, 1 = perfect; phone = worse of landline and VoIP)")
    ax.scatter([], [], s=60, color="white", edgecolor=MUTED, lw=1.5, label="Studio audio")
    ax.scatter([], [], s=70, color=BLUE, label="Phone audio, kept")
    ax.scatter([], [], s=70, color=GRAY, label="Phone audio, dropped")
    ax.text(0.2, len(rows) - 0.3, " keep line", color=BLUE, fontsize=12, va="bottom")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(RESULTS / "fig_survival.png")
    plt.close(fig)


def benchmark(tag, models):
    b = json.load(open(RESULTS / f"{tag}.json"))["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, corpus, title in zip(axes, ["asv_test", "itw"], ["ASVspoof 2019 eval (unseen attacks)", "In-the-Wild (real internet audio)"]):
        for name, (label, col) in models.items():
            e = [b[name]["test"][corpus][c]["eer"] for c in CONDITIONS]
            x = np.arange(len(CONDITIONS))
            ax.plot(x, [v[0] * 100 for v in e], color=col, lw=2, marker="o", ms=8, mec="white", mew=2, label=label)
            ax.fill_between(x, [v[1] * 100 for v in e], [v[2] * 100 for v in e], color=col, alpha=0.1, lw=0)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(["Studio", "Landline", "VoIP", "Cellular*"])
        ax.set_title(title, loc="left", color=INK)
        ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Equal error rate (%)\nlower is better, 50 = coin flip")
    axes[0].set_ylim(0, 65)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=13)
    fig.text(0.01, -0.16, "*G.723.1 was never used in training. Shaded bands are 95% bootstrap intervals.", color=MUTED, fontsize=12)
    fig.savefig(RESULTS / "fig_benchmark.png")
    plt.close(fig)


def false_alarms(tag, models, corpus="asv_test"):
    b = json.load(open(RESULTS / f"{tag}.json"))["runs"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.8 / len(models)
    x = np.arange(len(CONDITIONS))
    for k, (name, (label, col)) in enumerate(models.items()):
        v = [b[name]["test"][corpus][c]["false_alarm"] * 100 for c in CONDITIONS]
        bars = ax.bar(x + (k - (len(models) - 1) / 2) * w, v, width=w * 0.9, color=col, label=label)
        for r, val in zip(bars, v):
            ax.text(r.get_x() + r.get_width() / 2, val + 1.5, f"{val:.0f}%", ha="center", color=INK, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(["Studio", "Landline", "VoIP", "Cellular*"])
    ax.set_ylabel("Real voices flagged as fake (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper left")
    src = {"asv_test": "ASVspoof 2019 eval", "itw": "In-the-Wild"}[corpus]
    fig.text(0.01, -0.04, f"{src}, about 1,000 real voices per channel. Cut-off chosen on training data only. "
             "*G.723.1 was never used in training.", color=MUTED, fontsize=12)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.savefig(RESULTS / f"fig_false_alarms_{corpus}.png")
    plt.close(fig)


def main():
    spectrogram()
    app_spectrogram()
    survival()
    if len(sys.argv) > 1:
        cfg = json.loads(sys.argv[1])
        benchmark(cfg["tag"], cfg["models"])
        for corpus in ["asv_test", "itw"]:
            false_alarms(cfg["tag"], cfg["models"], corpus)


if __name__ == "__main__":
    sys.exit(main())

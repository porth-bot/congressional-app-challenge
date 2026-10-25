"""Figure: how well each signal separates scam calls from real family calls as
the call goes on. Reads results/fusion.json; nothing typed by hand."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1] / "results"
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
})
SERIES = [("voice", "Voice only", "#afb8c1"), ("words", "Words only", "#6e7781"),
          ("fused", "Combined (Liveline)", "#2a78d6")]


def main():
    f = json.load(open(R / "fusion.json"))
    panels = [(k, t) for k, t in [("itw_half/family", "Family-emergency calls*"), ("itw_half/external", "Public scam-call sets")]
              if k in f["runs"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.8), sharey=True, squeeze=False)
    for ax, (key, title) in zip(axes[0], panels):
        bp = f["runs"][key]["by_prefix"]
        cl = f["runs"][key]["cluster_auc"]["by_prefix"]
        qs = sorted(int(q) for q in bp)
        for name, label, col in SERIES:
            v = np.array([bp[str(q)][name]["auc"][0] for q in qs])
            lo = np.array([cl[str(q)][name][0] for q in qs])
            hi = np.array([cl[str(q)][name][1] for q in qs])
            ls = (0, (4, 3)) if name == "voice" else "-"
            ax.plot(qs, v, color=col, lw=2, ls=ls, label=label)
            ax.fill_between(qs, lo, hi, color=col, alpha=0.12, lw=0)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Share of the call heard (%)")
        ax.set_xticks([10, 30, 50, 70, 90])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("AUC, scam vs real family call\n(0.5 = coin flip, 1 = perfect)")
    axes[0][0].set_ylim(0.5, 1.0)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=13)
    fig.text(0.01, -0.2, "Test calls pair held-out In-the-Wild voices (phone codecs) with held-out transcripts. "
             "*Family-emergency transcripts were written by AI.\nBands are 95% intervals that resample speakers and "
             "transcripts (80 family transcripts, about 26 speakers), not individual calls.", color=MUTED, fontsize=11)
    fig.savefig(R / "fig_fusion.png")
    print("wrote", R / "fig_fusion.png")


if __name__ == "__main__":
    sys.exit(main())

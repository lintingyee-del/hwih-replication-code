# -*- coding: utf-8 -*-
"""hz_figstyle — house style for empirical figures (hzhang, settled 2026-07-11).

Journal-grade vector PDF, grayscale-safe. Origin: 6B paper step 20/76.
Conventions:
  - serif type matched to article body (Times/STIX), base size 9pt
  - monochrome: INK 0.10 points, WHISK 0.50 CIs, SHADE 0.955 post band,
    GUIDE 0.55 zero/reference rules
  - two-tier CIs: 90% thick (lw 2.0) inside 95% thin (lw 0.8)
  - event studies: light shading over the post period, dashed rule at
    treatment, hollow square at the reference-bin midpoint
  - panel tags "(a) ..." inside the axes at (0.03, 1.01); titles and notes
    live in the LaTeX caption, never inside the figure
  - sizes: 5.8x3.1 single panel, 6.9x3.0 for 1x2; savefig bbox_inches="tight",
    pad_inches=0.03; pdf.fonttype 42

Usage:
    import hz_figstyle as hz
    hz.apply()
    fig, ax = plt.subplots(figsize=hz.SINGLE)
    hz.espanel(ax, lo, hi, est, se, ylabel="asinh(cases) per SD of exposure")
    hz.save(fig, "fig_x.pdf")
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, WHISK, SHADE, GUIDE = "0.10", "0.50", "0.955", "0.55"
SINGLE, DOUBLE = (5.8, 3.1), (6.9, 3.0)

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.edgecolor": "0.35",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.color": "0.35", "ytick.color": "0.35",
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "0.90", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 200, "savefig.dpi": 200,
    "pdf.fonttype": 42,
}


def apply():
    plt.rcParams.update(RC)


def style_ticklabels(ax):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color("0.15")


def espanel(ax, bins_lo, bins_hi, est, se, ref=(-6, -1), ylabel="",
            post_from=-0.5):
    """Event-study panel: two-tier CIs, post shading, hollow reference marker."""
    est = np.asarray(est, float); se = np.asarray(se, float)
    mid = np.array([(a + b) / 2 for a, b in zip(bins_lo, bins_hi)])
    xmin, xmax = mid.min() - 3, mid.max() + 3
    ax.axvspan(post_from, xmax, color=SHADE, lw=0, zorder=0)
    ax.axhline(0, color=GUIDE, lw=0.7, zorder=1)
    ax.axvline(post_from, color=GUIDE, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.vlines(mid, est - 1.96 * se, est + 1.96 * se, color=WHISK, lw=0.8, zorder=3)
    ax.vlines(mid, est - 1.645 * se, est + 1.645 * se, color=WHISK, lw=2.0, zorder=3)
    ax.plot(mid, est, "o", color=INK, ms=4.6, mec="white", mew=0.6, zorder=4)
    if ref is not None:
        ax.plot([sum(ref) / 2], [0], marker="s", ms=5.0, mfc="white", mec=INK,
                mew=1.0, zorder=5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.12)
    style_ticklabels(ax)


def coefpanel(ax, labels, est, se, ylabel="", connect=True):
    """Categorical coefficient panel (e.g. by-bin gradients): connected dots."""
    est = np.asarray(est, float); se = np.asarray(se, float)
    x = np.arange(len(est))
    ax.axhline(0, color=GUIDE, lw=0.7)
    if connect:
        ax.plot(x, est, "-", color="0.62", lw=1.0, zorder=2)
    ax.vlines(x, est - 1.96 * se, est + 1.96 * se, color=WHISK, lw=0.8, zorder=3)
    ax.vlines(x, est - 1.645 * se, est + 1.645 * se, color=WHISK, lw=2.0, zorder=3)
    ax.plot(x, est, "o", color=INK, ms=4.8, mec="white", mew=0.6, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(est) - 0.5)
    ax.margins(y=0.12)
    ax.set_ylabel(ylabel)
    style_ticklabels(ax)


def tag(ax, text):
    """Panel tag '(a) ...' inside the axes, house position."""
    ax.annotate(text, xy=(0.03, 1.01), xycoords="axes fraction", fontsize=9,
                va="bottom", color=INK)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

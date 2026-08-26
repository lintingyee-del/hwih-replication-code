# -*- coding: utf-8 -*-
"""6B step 76 — redraw fig_twoclock in the house style of step 20.

Same data (output/twoclock_dynamics.csv), same two panels; style matched to
20_figures.py: serif/monochrome, two-tier CIs (90% thick / 95% thin), light
post-period shading, hollow reference marker, panel tags inside, notes in the
LaTeX caption.
"""

# Replication-package paths
from pathlib import Path as _ReplicationPath
import os as _ReplicationOS
_REP_PROJECT = _ReplicationPath(__file__).resolve().parents[1]
_REP_PACKAGE = _REP_PROJECT.parent
_REP_RESTRICTED = _REP_PACKAGE / 'restricted_data'
_REP_JUDGMENTS = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_JUDGMENT_ARCHIVE', _REP_RESTRICTED / 'judgment_archive'))
_REP_CASE_ARCHIVE = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_CASE_LEVEL_ARCHIVE', _REP_RESTRICTED / 'case_level_archive.parquet'))
_REP_MORTALITY = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_CDC_SOURCE_ROOT', _REP_RESTRICTED / 'mortality_volumes'))
_REP_REGISTRY = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_REGISTRY_ROOT', _REP_RESTRICTED / 'firm_registry'))
_REP_BAIDU = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_BAIDU_ROOT', _REP_RESTRICTED / 'baidu'))
_REP_INTERVIEWS = _ReplicationPath(_ReplicationOS.environ.get(
    'HWIH_INTERVIEWS_ROOT', _REP_RESTRICTED / 'interviews'))
import pandas as pd, numpy as np, os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTD = str(_REP_PROJECT / "output")
FIGD = f"{OUTD}/figures"
MANUSCRIPT = str(_REP_PACKAGE / "manuscript" / "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.edgecolor": "0.35",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.color": "0.35", "ytick.color": "0.35",
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "0.90", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 200, "savefig.dpi": 200,
    "pdf.fonttype": 42,
})
INK, WHISK, SHADE, GUIDE = "0.10", "0.50", "0.955", "0.55"


def _style_ticklabels(ax):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color("0.15")


def espanel(ax, bins_lo, bins_hi, est, se, ref=(-6, -1), ylabel=""):
    est = np.asarray(est, float); se = np.asarray(se, float)
    mid = np.array([(a + b) / 2 for a, b in zip(bins_lo, bins_hi)])
    xmin, xmax = mid.min() - 3, mid.max() + 3
    ax.axvspan(-0.5, xmax, color=SHADE, lw=0, zorder=0)
    ax.axhline(0, color=GUIDE, lw=0.7, zorder=1)
    ax.axvline(-0.5, color=GUIDE, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.vlines(mid, est - 1.96 * se, est + 1.96 * se, color=WHISK, lw=0.8, zorder=3)
    ax.vlines(mid, est - 1.645 * se, est + 1.645 * se, color=WHISK, lw=2.0, zorder=3)
    ax.plot(mid, est, "o", color=INK, ms=4.6, mec="white", mew=0.6, zorder=4)
    rm = sum(ref) / 2
    ax.plot([rm], [0], marker="s", ms=5.0, mfc="white", mec=INK, mew=1.0, zorder=5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.12)
    _style_ticklabels(ax)


dd = pd.read_csv(f"{OUTD}/twoclock_dynamics.csv")
enf = dd[dd["series"] == "enforceN_pros"].sort_values("lo")
civ = dd[dd["series"] == "civilFlow_filing"].sort_values("lo")

fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))
espanel(axes[0], enf["lo"].values, enf["hi"].values, enf["est"].values,
        enf["se"].values, ylabel="asinh(cases) per SD of exposure")
espanel(axes[1], civ["lo"].values, civ["hi"].values, civ["est"].values,
        civ["se"].values, ylabel="asinh(cases) per SD of exposure")
axes[0].annotate("(a) Enforcement caseload, prosecution clock", xy=(0.03, 1.01),
                 xycoords="axes fraction", fontsize=9, va="bottom", color=INK)
axes[1].annotate("(b) Relational filings, filing clock", xy=(0.03, 1.01),
                 xycoords="axes fraction", fontsize=9, va="bottom", color=INK)
fig.supxlabel("Months since inspection arrival (behavior clocks)", fontsize=9,
              y=-0.02, color="0.15")
fig.subplots_adjust(wspace=0.28)
fig.savefig(f"{FIGD}/fig_twoclock.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
shutil.copy(f"{FIGD}/fig_twoclock.pdf", f"{MANUSCRIPT}/fig_twoclock.pdf")
print("fig_twoclock.pdf redrawn in house style and copied to paper figures")

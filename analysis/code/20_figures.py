# -*- coding: utf-8 -*-
"""6B step 20 — paper figures (journal-grade vector PDF, grayscale-safe) +
P2P-battery and RR tables.

Style: serif type matched to the article body; two-tier CIs (thick 90% / thin
95%); light post-period shading; hollow reference marker; titles and notes live
in the LaTeX captions, never inside the figure. Monochrome throughout.
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
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUTD = str(_REP_PROJECT / "output")
FIGD = f"{OUTD}/figures"
TABD = f"{OUTD}/tables"
os.makedirs(FIGD, exist_ok=True)

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
    # light post-period band
    ax.axvspan(-0.5, xmax, color=SHADE, lw=0, zorder=0)
    ax.axhline(0, color=GUIDE, lw=0.7, zorder=1)
    ax.axvline(-0.5, color=GUIDE, lw=0.7, ls=(0, (4, 3)), zorder=1)
    # two-tier CI: 95% thin, 90% thick
    ax.vlines(mid, est - 1.96 * se, est + 1.96 * se, color=WHISK, lw=0.8, zorder=3)
    ax.vlines(mid, est - 1.645 * se, est + 1.645 * se, color=WHISK, lw=2.0, zorder=3)
    ax.plot(mid, est, "o", color=INK, ms=4.6, mec="white", mew=0.6, zorder=4)
    # hollow reference marker
    rm = sum(ref) / 2
    ax.plot([rm], [0], marker="s", ms=5.0, mfc="white", mec=INK, mew=1.0, zorder=5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.12)
    _style_ticklabels(ax)

# ---- Fig 1: civil flow event study ------------------------------------------
es = pd.read_csv(f"{OUTD}/eventstudy_v2.csv")
fig, ax = plt.subplots(figsize=(5.8, 3.1))
espanel(ax, es["bin_lo"], es["bin_hi"], es["est"].values, es["se"].values,
        ylabel="asinh(cases) per SD of exposure")
ax.set_xlabel("Months since inspection arrival")
fig.savefig(f"{FIGD}/fig_es_civil.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# ---- Fig 2: criminal backstop event study -----------------------------------
es1 = pd.read_csv(f"{OUTD}/eventstudy.csv")
esb = es1[es1["outcome"] == "y_backstop"]
fig, ax = plt.subplots(figsize=(5.8, 3.1))
espanel(ax, esb["bin_lo"], esb["bin_hi"], esb["est"].values, esb["se"].values,
        ylabel="Backstop share per SD of exposure")
ax.set_xlabel("Months since inspection arrival")
fig.savefig(f"{FIGD}/fig_es_backstop.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# ---- Fig 3: enforcement-crime caseload dynamics (two panels) ----------------
dyn = pd.read_csv(f"{OUTD}/dynamics_enforce.csv")
blo = dyn["bin"].str.extract(r"b_(m?\d+)_")[0].str.replace("m", "-").astype(int)
bhi = dyn["bin"].str.extract(r"_(m?\d+)$")[0].str.replace("m", "-").astype(int)
o = np.argsort(blo.values)
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))
espanel(axes[0], blo.values[o], bhi.values[o], dyn["est_raw"].values[o],
        dyn["se_raw"].values[o], ylabel="asinh(cases)")
espanel(axes[1], blo.values[o], bhi.values[o], dyn["est_dose"].values[o],
        dyn["se_dose"].values[o], ylabel="asinh(cases) per SD of exposure")
axes[0].annotate("(a) Raw profile", xy=(0.03, 1.01), xycoords="axes fraction",
                 fontsize=9, va="bottom", color=INK)
axes[1].annotate("(b) Exposure-interacted", xy=(0.03, 1.01), xycoords="axes fraction",
                 fontsize=9, va="bottom", color=INK)
fig.supxlabel("Months since inspection arrival", fontsize=9, y=-0.02, color="0.15")
fig.subplots_adjust(wspace=0.28)
fig.savefig(f"{FIGD}/fig_dynamics_enforce.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# The current stake-gradient figure is generated by step 100 from the declared
# five-band analysis.  The former C4 block here used superseded result tags and
# is intentionally not retained as a second implementation.
r = pd.read_csv(f"{OUTD}/results_v2.csv")

# ---- Fig 4: campaign timeline (clean Gantt) ---------------------------------
fig, ax = plt.subplots(figsize=(6.4, 2.1))
t0 = pd.Timestamp("2018-01-01")
mo = lambda d: (pd.Timestamp(d) - t0).days / 30.44
waves = [("Wave 1", "10 provinces", "2018-07", "2018-09", 3),
         ("Wave 2", "11 provinces", "2019-04", "2019-04", 2),
         ("Wave 3", "10 provinces", "2019-05", "2019-06", 1)]
for name, cnt, a, b, y in waves:
    xa, xb = mo(a), mo(b) + 1
    ax.add_patch(plt.Rectangle((xa, y - 0.26), xb - xa, 0.52, facecolor="0.30",
                               edgecolor="none", zorder=3))
    ax.text(xa - 0.4, y, name, ha="right", va="center", fontsize=8.5, color=INK)
    ax.text(xb + 0.4, y, cnt, ha="left", va="center", fontsize=8, color="0.4",
            style="italic")
ax.axvline(0, color=GUIDE, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.text(0.3, 3.7, "Campaign launch", fontsize=8, color="0.4", style="italic")
ticks = pd.date_range("2018-01-01", "2019-12-01", freq="3MS")
ax.set_xticks([mo(t) for t in ticks])
ax.set_xticklabels([t.strftime("%Y-%m") for t in ticks])
ax.set_yticks([]); ax.set_ylim(0.3, 4.1); ax.set_xlim(-3.5, 27)
ax.grid(False)
for s_ in ["left", "right", "top"]:
    ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color("0.35")
_style_ticklabels(ax)
fig.savefig(f"{FIGD}/fig_timeline.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# ---- tables: P2P battery + RR bounds (unchanged) ----------------------------
def cell(row):
    st = "***" if row["p"] < .01 else "**" if row["p"] < .05 else "*" if row["p"] < .1 else ""
    wp = f"{row['wild_p']:.3f}" if not np.isnan(row["wild_p"]) else "--"
    return f"{row['est']:.4f}{st} & ({row['se']:.4f}) & {wp} & {int(row['n']):,}"
r = r.set_index("tag")
L = [r"\begin{tabular}{lcccc}", r"\toprule",
     r"Specification & Coefficient & (SE) & Wild $p$ & $N$ \\ \midrule",
     r"\multicolumn{5}{l}{\emph{Panel A. Clean-window stacked flow (headline)}} \\[2pt]"]
for lab, tag in [("Baseline", "P1_stacked_baseline_reprint"),
                 ("+ Post$\\times$DFI-credit control", "P1_stacked_dficontrol"),
                 ("Drop top-quartile DFI prefectures", "P1_stacked_dropTopDFI")]:
    L.append(f"{lab} & {cell(r.loc[tag])} \\\\")
L += [r"\midrule", r"\multicolumn{5}{l}{\emph{Panel B. Full-sample triple difference}} \\[2pt]"]
for lab, tag in [("Baseline", "P1_C1_baseline_reprint"),
                 ("+ Post$\\times$DFI-credit control", "P1_C1_dficontrol"),
                 ("Drop top-quartile DFI prefectures", "P1_C1_dropTopDFI")]:
    L.append(f"{lab} & {cell(r.loc[tag])} \\\\")
L += [r"\midrule", r"\multicolumn{5}{l}{\emph{Panel C. Criminal enforcement caseload}} \\[2pt]",
      f"+ Post$\\times$DFI-credit control & {cell(r.loc['P1_enforceN_dficontrol'])} \\\\",
      r"\midrule",
      f"corr(exposure, DFI credit 2017) & \\multicolumn{{4}}{{c}}{{{r.loc['P1_corr_H_DFI']['est']:.3f}}} \\\\",
      r"\bottomrule", r"\end{tabular}"]
open(f"{TABD}/tab_p2p.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

rr = pd.read_csv(f"{OUTD}/rr_bounds.csv")
L = [r"\begin{tabular}{llcccc}", r"\toprule",
     r"Design & Target & $\hat\theta$ & (SE) & $B$ & Breakdown $\bar M$ \\ \midrule"]
for _, row in rr.iterrows():
    L.append(f"{row['design']} & {row['target']} & {row['theta']:.4f} & "
             f"({row['se']:.4f}) & {row['B']:.4f} & {row['breakdown_Mbar']:.2f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{TABD}/tab_rr.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("figures:", sorted(os.listdir(FIGD)))

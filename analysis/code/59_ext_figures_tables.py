# -*- coding: utf-8 -*-
"""6B step 59 — extension exhibits for the Version A integration.
Figures (house style of 20_figures.py):
  fig_es_backstop.pdf   original in-window series (filled) + post-2020
                        extension bins (open markers), rule at +28
  fig_release_series.pdf monthly releases by docket, source deaths marked
Tables (printed to ext2124/appendix_tables.tex): D1 release series, D2 gates,
E1 extension event study, E2 diagnostics.
Figures written to analysis/output/figures/ and copied to the submission dir.
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
import pandas as pd, numpy as np, os, shutil, sys, io, duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
OUTD = str(_REP_PROJECT / "output")
EXT = f"{OUTD}/ext2124"
FIGD = f"{OUTD}/figures"
SUB = str(_REP_PACKAGE / "manuscript" / "figures")
SRC6A = str(_REP_CASE_ARCHIVE)
CRIM_MONTHLY = f"{EXT}/criminal_release_monthly.csv"

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

def _style(ax):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color("0.15")

# ---------------- Fig: extended backstop event study ----------------
es1 = pd.read_csv(f"{OUTD}/eventstudy.csv")
esb = es1[es1["outcome"] == "y_backstop"]
ext = pd.read_csv(f"{EXT}/persist_es.csv")
ext = ext[(ext.spec == "coercive_share") & (ext.bin_lo >= 24)]

fig, ax = plt.subplots(figsize=(6.4, 3.1))
mid_o = (esb["bin_lo"] + esb["bin_hi"]) / 2
mid_e = (ext["bin_lo"] + ext["bin_hi"]) / 2
ax.axvspan(-0.5, 28, color=SHADE, lw=0, zorder=0)
ax.axvspan(28, 70, color="0.975", lw=0, zorder=0)
ax.axhline(0, color=GUIDE, lw=0.7, zorder=1)
ax.axvline(-0.5, color=GUIDE, lw=0.7, ls=(0, (4, 3)), zorder=1)
ax.axvline(28, color="0.30", lw=0.9, ls=(0, (2, 2)), zorder=2)
for m, e, s in [(mid_o, esb["est"].values, esb["se"].values)]:
    ax.vlines(m, e - 1.96 * s, e + 1.96 * s, color=WHISK, lw=0.8, zorder=3)
    ax.vlines(m, e - 1.645 * s, e + 1.645 * s, color=WHISK, lw=2.0, zorder=3)
    ax.plot(m, e, "o", color=INK, ms=4.6, mec="white", mew=0.6, zorder=4)
e, s = ext["est"].values, ext["se"].values
ax.vlines(mid_e, e - 1.96 * s, e + 1.96 * s, color=WHISK, lw=0.8, zorder=3)
ax.vlines(mid_e, e - 1.645 * s, e + 1.645 * s, color=WHISK, lw=2.0, zorder=3)
ax.plot(mid_e, e, "o", mfc="white", mec=INK, mew=1.0, ms=4.6, zorder=4)
ax.plot([-3.5], [0], marker="s", ms=5.0, mfc="white", mec=INK, mew=1.0, zorder=5)
ax.set_xlim(-28, 70)
ax.set_xlabel("Months since inspection arrival")
ax.set_ylabel("Backstop share per SD of exposure")
_style(ax)
fig.savefig(f"{FIGD}/fig_es_backstop.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# ---------------- Fig: release series ----------------
if os.path.exists(SRC6A):
    con = duckdb.connect()
    crim = con.sql(f"""SELECT strftime(judgment_date, '%Y-%m') ym, COUNT(*) n
      FROM '{SRC6A}' WHERE judgment_date BETWEEN DATE '2014-01-01' AND DATE '2024-10-31'
      GROUP BY 1 ORDER BY 1""").df()
    crim.to_csv(CRIM_MONTHLY, index=False)
    print(f"refreshed public criminal release aggregate -> {CRIM_MONTHLY}")
elif os.path.exists(CRIM_MONTHLY):
    crim = pd.read_csv(CRIM_MONTHLY, dtype={"ym": str})
    if list(crim.columns) != ["ym", "n"] or crim["ym"].duplicated().any():
        raise ValueError(f"invalid released criminal monthly aggregate: {CRIM_MONTHLY}")
    print(f"using released criminal release aggregate <- {CRIM_MONTHLY}")
else:
    raise FileNotFoundError(
        "Release-series figure needs either the restricted case archive or "
        f"the public monthly aggregate {CRIM_MONTHLY}"
    )
mb = pd.read_csv(f"{EXT}/mjd_baseline_2014_2020.csv")[["ym", "n"]]
mn = pd.read_csv(f"{EXT}/mjd.csv", header=None).iloc[:, :2]
mn.columns = ["ym", "n"]
civ = pd.concat([mb, mn]).sort_values("ym")
civ = civ[civ.ym != "2024-10"]  # backfill batch, not a judgment-month observation

def tonum(df):
    x = pd.PeriodIndex(df["ym"], freq="M")
    return x.year + (x.month - 1) / 12

fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.plot(tonum(crim), crim["n"], color=INK, lw=1.0, label="Criminal (17 offenses)")
ax.plot(tonum(civ), civ["n"], color="0.55", lw=1.0, label="Civil lending judgments")
ax.set_yscale("log")
for x, lab in [(2021 + 9.5 / 12, "source 1 ends"), (2023 + 11.5 / 12, "source 2 ends")]:
    ax.axvline(x, color="0.30", lw=0.8, ls=(0, (2, 2)))
ax.axvline(2021, color=GUIDE, lw=0.7, ls=(0, (4, 3)))
ax.set_xlim(2014, 2025)
ax.set_ylabel("Documents per month (log scale)")
ax.legend(frameon=False, fontsize=8, loc="lower left")
_style(ax)
fig.savefig(f"{FIGD}/fig_release_series.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

for f in ("fig_es_backstop.pdf", "fig_release_series.pdf"):
    shutil.copy(f"{FIGD}/{f}", f"{SUB}/{f}")
print("figures written and copied")

# ---------------- Tables ----------------
L = []
def w(s):
    L.append(s)

# D1
cy = crim.assign(y=crim["ym"].str[:4].astype(int)).groupby("y")["n"].sum()
civ["yr"] = civ["ym"].str[:4].astype(int)
cb = pd.read_csv(f"{EXT}/causes_baseline_2014_2020.csv")
cb = cb[cb.cause == "机动车交通事故责任纠纷"]
cn = pd.read_csv(f"{EXT}/causes.csv", header=None,
                 names=["ym", "cause", "n_all", "n_judg", "n_med"])
cn = cn[(cn.cause == "机动车交通事故责任纠纷") & (cn.ym != "2024-10")]
tr = pd.concat([cb[["ym", "n_judg"]], cn[["ym", "n_judg"]]])
tr["yr"] = tr["ym"].str[:4].astype(int)
civ_y = civ.groupby("yr")["n"].sum()
tr_y = tr.groupby("yr")["n_judg"].sum()
srcs = {y: "macro" for y in range(2014, 2021)}
srcs[2021] = "macro / ws"; srcs[2022] = "ws"; srcs[2023] = "ws"; srcs[2024] = "s41"
w("% ---- Table D1 ----")
w("\\begin{table}[htbp]\\centering")
w("\\caption{Judgment release by year, docket, and source.}\\label{tab:relseries}")
w("\\small \\tabin{\\begin{tabular}{lrrrl}")
w("\\toprule")
w("Year & Criminal (17 offenses) & Lending judgments & Traffic judgments & Source \\\\ \\midrule")
for y in range(2014, 2025):
    note = " (Jan--Oct)" if y == 2024 else ""
    w(f"{y}{note} & {cy.get(y, 0):,} & {int(civ_y.get(y, 0)):,} & {int(tr_y.get(y, 0)):,} & {srcs[y]} \\\\")
w("\\bottomrule")
w("\\end{tabular}}")
w("\\par\\smallskip\\centerline{\\parbox{\\tabinwd}{\\footnotesize\\emph{Notes:} Documents in the analysis dockets"
  " by judgment year. Criminal: archive judgments in the seventeen offense categories."
  " Civil columns: first-instance judgments in the private-lending and traffic-tort causes."
  " The primary source ceases operation in October 2021; a second source covers November 2021"
  " through December 2023; a third covers January through October 2024. The October 2024 file"
  " is a late-publication batch of 2022--2024 judgments and is excluded from monthly series."
  " Figure~\\ref{fig:relseries} plots the monthly series with the transitions marked.}}")
w("\\end{table}")
w("")

# D2
g = pd.read_csv(f"{EXT}/gates_composition.csv")
t = pd.read_csv(f"{EXT}/gates_tost.csv")
g = g[g.horizon == "2021-24"].set_index("attribute")
t = t[t.horizon == "2021-24"].set_index("attribute")
NAMES = [("log_release", "Log release volume"),
         ("sh_fact", "Fact-section availability"),
         ("violent_offense_mix", "Violent-offense mix"),
         ("placebo_mix", "Placebo-docket mix"),
         ("med_factlen", "Median fact length (chars)")]
w("% ---- Table D2 ----")
w("\\begin{table}[htbp]\\centering")
w("\\caption{Exposure-neutrality gates: post-2020 changes in released-docket attributes.}\\label{tab:gates}")
w("\\small \\tabin{\\begin{tabular}{lccc}")
w("\\toprule")
w("Attribute (2019 $\\rightarrow$ 2021--24 change) & $\\beta_{H}$ & s.e. & TOST margin \\\\ \\midrule")
for k, nm in NAMES:
    w(f"{nm} & {g.loc[k, 'beta_H']:.4f} & ({g.loc[k, 'se']:.4f}) & {t.loc[k, 'tost_margin_5pct']:.3f} \\\\")
w("\\bottomrule")
w("\\end{tabular}}")
w("\\par\\smallskip\\centerline{\\parbox{\\tabinwd}{\\footnotesize\\emph{Notes:} Each row regresses the prefecture-level"
  " change in one predetermined attribute of the released docket (2019 baseline to the 2021--2024"
  " average) on exposure $H_c$; CRV1 standard errors by province (31 clusters). The TOST margin is"
  " the smallest symmetric equivalence bound certified at the 5 percent level. Court-level release"
  " survival (courts with at least ten lending judgments in 2019, $n=2{,}937$) is 91 percent, flat"
  " in $H_c$ ($p=0.39$) and in the court's own 2019 relational and documentation shares ($p=0.46$,"
  " $0.50$). Judgments dated January 2022 through June 2023 published only in later batches"
  " (12 percent) match contemporaneously published ones on relational, coercive-collection, and"
  " documentation flags interacted with $H_c$ ($p=0.68$, $0.72$, $0.57$).}}")
w("\\end{table}")
w("")

# E1
e = pd.read_csv(f"{EXT}/persist_es.csv")
specs = [("coercive_share", "Main"), ("coercive_share_nodonut", "No donut"),
         ("coercive_share_min10", "Min cell 10")]
bins = e[e.spec == "coercive_share"][["bin_lo", "bin_hi"]].values
w("% ---- Table E1 ----")
w("\\begin{table}[htbp]\\centering")
w("\\caption{Hard-backstop share through event time $+66$: extension event study.}\\label{tab:persistes}")
w("\\small \\tabin{\\begin{tabular}{lcccccc}")
w("\\toprule")
w(" & \\multicolumn{2}{c}{Main} & \\multicolumn{2}{c}{No donut} & \\multicolumn{2}{c}{Min cell 10} \\\\")
w("Event-time bin & est. & s.e. & est. & s.e. & est. & s.e. \\\\ \\midrule")
for lo, hi in bins:
    row = [f"$[{lo:+d},{hi:+d}]$"]
    for sp, _ in specs:
        r = e[(e.spec == sp) & (e.bin_lo == lo)]
        if len(r):
            row += [f"{r.est.iloc[0]:.4f}", f"({r.se.iloc[0]:.4f})"]
        else:
            row += ["--", "--"]
    w(" & ".join(row) + " \\\\")
w("\\midrule")
w("Pooled post-2020 $\\times H$ & \\multicolumn{6}{l}{$\\PersistPooled$ (s.e.\\ $\\PersistPooledSE$);"
  " CRV1 $p=\\PersistPooledP$, score-flip bootstrap $p=\\PersistPooledWild$} \\\\")
w("\\bottomrule")
w("\\end{tabular}}")
w("\\par\\smallskip\\centerline{\\parbox{\\tabinwd}{\\footnotesize\\emph{Notes:} Event-time bins interacted with"
  " exposure $H_c$; reference window $[-6,-1]$. Outcome: hard-backstop share among target-docket"
  " judgments with fact sections, 2014--2024 coding held fixed across the 2020 boundary. Prefecture"
  " and province$\\times$month fixed effects, fact-length control in the pooled specification, cells"
  " with at least twenty (main) fact-section judgments, weights equal to cell denominators, CRV1 by"
  " province. The donut drops September--December 2021 (source transition). The pooled row reports"
  " the single post-2020 coefficient of Section~\\ref{sec:persist}.}}")
w("\\end{table}")
w("")

# E2
pt = pd.read_csv(f"{EXT}/persist_tests.csv").set_index("spec")
w("% ---- Table E2 ----")
w("\\begin{table}[htbp]\\centering")
w("\\caption{Extension diagnostics: drift, relative-magnitude bounds, worst-case suppression.}\\label{tab:persistdiag}")
w("\\small \\tabin{\\begin{tabular}{lcc}")
w("\\toprule")
w("Statistic & Value & s.e. \\\\ \\midrule")
for sp, nm in specs:
    w(f"Drift per month across post-2020 bins ({nm.lower()}) & "
      f"{pt.loc[sp, 'drift_per_month']:.5f} & ({pt.loc[sp, 'drift_se']:.5f}) \\\\")
w("Largest pre-period violation $B$ & 0.0249 & \\\\")
w("Breakdown $\\bar M$ (pooled coefficient) & 0.18 & \\\\")
w("Worst-case violent-content non-release $\\delta^{*}$ needed to")
w("\\quad generate the pooled coefficient & 0.13 & \\\\")
w("Observable margin: violent-offense mix, 95\\% bound & 0.011 & \\\\")
w("Observable margin: release volume, 95\\% bound (log points) & 0.17 & \\\\")
w("\\bottomrule")
w("\\end{tabular}}")
w("\\par\\smallskip\\centerline{\\parbox{\\tabinwd}{\\footnotesize\\emph{Notes:} Drift is the linear trend across the"
  " six post-2020 bins of Table~\\ref{tab:persistes}. $B$ and the breakdown $\\bar M$ follow"
  " Rambachan and Roth (2023) relative magnitudes on the pooled post-2020 coefficient."
  " $\\delta^{*}=|\\theta|/s(1-s)$ with post-2020 coercive share $s=0.16$: the per-SD rate of"
  " violent-content-only non-release that would fully generate the pooled coefficient; it exceeds"
  " every observable selection margin by an order of magnitude, but selection within offense"
  " categories cannot be excluded by observables, which is why the text reads the extension as"
  " no reversion in the released record rather than as a causal magnitude.}}")
w("\\end{table}")

open(f"{EXT}/appendix_tables.tex", "w", encoding="utf-8").write("\n".join(L))
print(f"tables -> {EXT}/appendix_tables.tex ({len(L)} lines)")

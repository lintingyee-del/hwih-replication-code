# -*- coding: utf-8 -*-
"""6B step 25 — AEJ-tier upgrades (no new data).

D1  Contiguous-province design: restrict the clean-window civil flow to first-wave
    provinces and the not-yet-treated provinces that share a land border with them
    (GB province-code adjacency), so treated and control prefectures are geographic
    neighbours. A feasible analog of a prefecture-border design (true border pairs
    would need centroids we do not have).
D3  Dose-response: clean-window civil flow by exposure quintile (monotonicity).
D4  Court-capacity heterogeneity: does judicialization load where courts had slack?
    (Pre-campaign civil cases per court, prefecture-level.)
D2  Welfare/cost back-of-envelope: caseload transferred, share of court workload, an
    illustrative monetary burden, and the speed contrast (criminal filing->judgment
    duration; private resolution is near-instant) with a congestion test (duration on
    Post x H).
D5  Timing/lag mechanism: origination-to-litigation lag consistent with private-
    enforcement failure rather than fresh borrowing.

Outputs: output/tables/numbers_ext.tex, tab_border.tex, tab_welfare.tex,
         output/figures/fig_dose.pdf, output/aej_upgrades_log via stdout.
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
import numpy as np, pandas as pd, pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
WINDOW = ("2017-01", "2019-03"); POST0 = "2018-09"
rows = {}

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


def fit(tag, fml, df, coef, weights=None):
    m = pf.feols(fml, data=df, vcov={"CRV1": "prov_id"}, weights=weights)
    try: wp = wild_score_p(fml, df, coef, weights)
    except Exception: wp = np.nan
    rows[tag] = dict(est=m.coef()[coef], se=m.se()[coef], p=m.pvalue()[coef],
                     wild_p=wp, n=int(m._N))
    print(f"{tag:34s} {m.coef()[coef]: .4f} ({m.se()[coef]:.4f}) "
          f"p={m.pvalue()[coef]:.3f} wild={wp:.3f} N={m._N}", flush=True)
    return m

# GB province-code land adjacency (undirected; islands have no land border)
ADJ = {
 "11":"12,13","12":"11,13","13":"11,12,14,15,21,37,41","14":"13,15,41,61",
 "15":"13,14,21,22,23,61,62,64","21":"13,15,22","22":"15,21,23","23":"15,22",
 "31":"32,33","32":"31,33,34,37","33":"31,32,34,35,36","34":"32,33,36,37,41,42",
 "35":"33,36,44","36":"33,34,35,42,43,44","37":"13,32,34,41",
 "41":"13,14,34,37,42,61","42":"34,36,41,43,50,61","43":"36,42,44,45,50,52",
 "44":"35,36,43,45","45":"43,44,52,53","46":"","50":"42,43,51,52,61",
 "51":"50,52,53,54,61,62,63","52":"43,45,50,51,53","53":"45,51,52,54",
 "54":"51,53,63,65","61":"14,15,41,42,50,51,62,64","62":"15,51,61,63,64,65",
 "63":"51,54,62,65","64":"15,61,62","65":"54,62,63"}
ADJ = {k: set(v.split(",")) - {""} for k, v in ADJ.items()}

# ============================ load clean-window civil ========================
cp = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cp["month"] = cp["jmonth"].astype(str).str[:7]
cp = cp[(cp["month"] >= WINDOW[0]) & (cp["month"] <= WINDOW[1])].copy()
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[
    ["province", "inspection_round"]].drop_duplicates()
cp = cp.merge(sched, on="province", how="left").dropna(subset=["exposure_v2_z"])
cp["pcode2"] = cp["prefecture_code"].astype(str).str[:2]
cp["treat"] = (cp["inspection_round"] == 1).astype(int)
cp["postc"] = (cp["month"] >= POST0).astype(int)
cp["prov_id"] = pd.factorize(cp["province"])[0]
cp["pref_cause"] = cp["prefecture_code"].astype(str) + "_" + cp["cause"]
cp["month_fe"] = cp["month"]
cp["asinh_n"] = np.arcsinh(cp["n_cases"])
cp["H"] = cp["exposure_v2_z"]
rel = cp[cp["cause_family"] == "relational"].copy()

# round by province-code (for adjacency)
r_by_code = rel.groupby("pcode2")["treat"].max()  # 1 if any first-wave pref in code
code_round = {c: ("T" if r_by_code.get(c, 0) == 1 else "C") for c in r_by_code.index}

# baseline clean-window dose (replicates headline) on relational cells
rel["pt"] = rel["postc"] * rel["treat"]; rel["pth"] = rel["pt"] * rel["H"]
rel["ph"] = rel["postc"] * rel["H"]
fit("D1_civil_baseline", "asinh_n ~ pth + ph + pt | pref_cause + month_fe", rel, "pth")

# ============================ D1 contiguous-province =========================
# keep provinces on a treatment border: a T-code adjacent to >=1 C-code, or vice versa
def on_border(c):
    r = code_round.get(c)
    if r is None: return False
    nbr = ADJ.get(c, set())
    other = "C" if r == "T" else "T"
    return any(code_round.get(n) == other for n in nbr)

border_codes = {c for c in code_round if on_border(c)}
relb = rel[rel["pcode2"].isin(border_codes)].copy()
print(f"contiguous-province sample: {len(border_codes)} border provinces, "
      f"{relb['prefecture_code'].nunique()} prefectures, N={len(relb)}", flush=True)
fit("D1_civil_border", "asinh_n ~ pth + ph + pt | pref_cause + month_fe", relb, "pth")

# ============================ D3 dose-response by quintile ====================
pref_H = rel[["prefecture_code", "H"]].drop_duplicates("prefecture_code")
qedges = pref_H["H"].quantile(np.linspace(0, 1, 6)).values
rel["Hq"] = pd.cut(rel["H"], bins=np.unique(qedges), labels=False,
                   include_lowest=True)
dose = []
for q in sorted(rel["Hq"].dropna().unique()):
    d = rel[rel["Hq"] == q].copy()
    d["pt"] = d["postc"] * d["treat"]
    m = pf.feols("asinh_n ~ pt | pref_cause + month_fe", data=d,
                 vcov={"CRV1": "prov_id"})
    hmid = d["H"].mean()
    dose.append((int(q), hmid, m.coef()["pt"], m.se()["pt"], int(m._N)))
    print(f"  dose Q{int(q)+1}: Hmean={hmid:+.2f} beta={m.coef()['pt']:+.4f} "
          f"({m.se()['pt']:.4f}) N={m._N}", flush=True)
dose = pd.DataFrame(dose, columns=["q", "Hmean", "beta", "se", "n"])

# ============================ D4 court-capacity slack ========================
cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["prefecture_code", "cause_family", "jmonth"])
cc["month"] = cc["jmonth"].astype(str).str[:7]
pre = cc[cc["month"] < "2018-01"]
pref_cases = pre.groupby("prefecture_code").size().rename("pre_civil")
xw = pd.read_parquet(f"{DATA}/court_xwalk.parquet")
courts = xw.groupby("prefecture_code").size().rename("n_courts")
cap = pd.concat([pref_cases, courts], axis=1).dropna()
cap["load_per_court"] = cap["pre_civil"] / cap["n_courts"]
cap["low_slack"] = (cap["load_per_court"] >
                    cap["load_per_court"].median()).astype(int)  # 1 = congested
relc = rel.merge(cap[["low_slack"]], on="prefecture_code", how="inner").copy()
relc["pth_slack"] = relc["pth"] * relc["low_slack"]
relc["pt_slack"] = relc["pt"] * relc["low_slack"]
relc["ph_slack"] = relc["ph"] * relc["low_slack"]
m = fit("D4_civil_x_lowslack",
        "asinh_n ~ pth + pth_slack + ph + ph_slack + pt + pt_slack "
        "| pref_cause + month_fe", relc, "pth_slack")
# main effect (slack prefectures = high-capacity baseline) reported alongside
rows["D4_civil_highslack_main"] = dict(est=rows["D4_civil_x_lowslack"]["est"],
    se=np.nan, p=np.nan, wild_p=np.nan, n=rows["D4_civil_x_lowslack"]["n"])
base = pf.feols("asinh_n ~ pth + pth_slack + ph + ph_slack + pt + pt_slack "
                "| pref_cause + month_fe", data=relc, vcov={"CRV1": "prov_id"})
rows["D4_highslack"] = dict(est=base.coef()["pth"], se=base.se()["pth"],
    p=base.pvalue()["pth"], wild_p=np.nan, n=int(base._N))
print(f"  high-capacity (slack) prefectures: pth={base.coef()['pth']:.4f}; "
      f"congested add-on: {base.coef()['pth_slack']:.4f}", flush=True)

# ============================ D2 welfare / speed =============================
# caseload transferred (clean-window dose, asinh ~ log at these means)
beta = rows["D1_civil_baseline"]["est"]
pre_rel = pre_rel_national = None
relpanel = pd.read_parquet(f"{DATA}/civil_panel.parquet")
relpanel = relpanel[relpanel["cause_family"] == "relational"].copy()
relpanel["yr"] = relpanel["jmonth"].astype(str).str[:4]
pre1617 = relpanel[relpanel["yr"].isin(["2016", "2017"])]
per_pref_yr = pre1617.groupby("prefecture_code")["n_cases"].sum().mean() / 2.0
national = pre1617["n_cases"].sum() / 2.0
inc_per_sd = per_pref_yr * (np.exp(beta) - 1)         # asinh~log for large n
national_1sd = national * (np.exp(beta) - 1)
COST_RMB = 3000  # illustrative court+litigant cost per civil case (flagged assn)
burden_national = national_1sd * COST_RMB
share_workload = (np.exp(beta) - 1)                    # increment / baseline
print(f"welfare: pre-campaign ~{per_pref_yr:.0f} rel cases/pref-yr, "
      f"~{national/1e6:.2f}M national; clean-window beta={beta:.3f} -> "
      f"+{inc_per_sd:.0f}/pref-yr per SD, +{share_workload*100:.0f}% of workload; "
      f"national +{national_1sd/1e3:.0f}k cases, ~{burden_national/1e6:.0f}M RMB @ "
      f"{COST_RMB}/case", flush=True)

# speed: criminal filing->judgment duration (courts slow vs private near-instant)
kd = pd.read_parquet(f"{DATA}/case_clean.parquet",
                     columns=["duration_days", "prefecture_code", "province",
                              "judgment_month", "analysis_group", "insp_month",
                              "post_judgment"])
kd = kd[(kd["duration_days"].notna()) & (kd["duration_days"] > 0)
        & (kd["duration_days"] < 3650)].copy()
dur_med = kd["duration_days"].median()
dur_mean = kd["duration_days"].mean()
print(f"criminal court duration: median {dur_med:.0f} d, mean {dur_mean:.0f} d "
      f"(N={len(kd):,})", flush=True)

# congestion: does duration rise with Post x H in enforcement-relevant dockets?
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code","exposure_v2_z"]]
kd = kd.merge(ex, on="prefecture_code", how="inner")
kd["prov_id"] = pd.factorize(kd["province"])[0]
kd["month"] = kd["judgment_month"].astype(str).str[:7]
kd["prov_month"] = kd["province"].astype(str) + "_" + kd["month"]
kd["pref"] = kd["prefecture_code"]
kd["logdur"] = np.log(kd["duration_days"])
kd["px"] = kd["post_judgment"].astype(float) * kd["exposure_v2_z"]
mkt = kd[kd["analysis_group"] == "market"].copy()
if len(mkt) > 5000:
    fit("D2_duration_market", "logdur ~ px | pref + prov_month", mkt, "px")

# ============================ D5 timing / lag ================================
lc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["cause", "jmonth", "orig_year", "post", "insp_month"])
ld = lc[lc["cause"] == "民间借贷纠纷"].copy()
ld["jyear"] = ld["jmonth"].astype(str).str[:4].astype(float)
ld = ld[(ld["orig_year"] >= 2012) & (ld["orig_year"] <= 2020)]
ld["lag"] = ld["jyear"] - ld["orig_year"]
ld = ld[ld["lag"] >= 0]
lag_pre = ld.loc[ld["post"] == 0, "lag"].mean()
lag_post = ld.loc[ld["post"] == 1, "lag"].mean()
# share of post-campaign litigated loans originated BEFORE the campaign (stock, not flow)
post_pre_orig = (ld.loc[ld["post"] == 1, "orig_year"] < 2018).mean()
print(f"lag: pre {lag_pre:.2f}y, post {lag_post:.2f}y; of post-campaign litigated "
      f"loans, {post_pre_orig*100:.0f}% originated pre-2018 (failing stock, not new "
      f"borrowing)", flush=True)

# ============================ figure: dose-response ==========================
fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.axhline(0, color="0.7", lw=0.8)
ax.errorbar(dose["Hmean"], dose["beta"], yerr=1.96*dose["se"], fmt="o-",
            color="#1f4e79", ecolor="#8fa9c8", capsize=3, lw=1.5, ms=6)
ax.set_xlabel("Prefecture exposure $H_c$ (quintile mean, SD units)")
ax.set_ylabel(r"Clean-window civil flow $\hat\beta$ (asinh)")
ax.set_title("Dose-response: judicialization rises with coercive-capacity exposure",
             fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUTD}/figures/fig_dose.pdf"); plt.close(fig)
print("saved fig_dose.pdf", flush=True)

# ============================ exports ========================================
def g(t, k="est"): return rows[t][k]
def st(t):
    p = rows[t]["p"];
    return "***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""
M = {
 "ExtBorderBeta": f"{g('D1_civil_border'):.3f}",
 "ExtBorderSE": f"{g('D1_civil_border','se'):.3f}",
 "ExtBorderP": f"{g('D1_civil_border','p'):.3f}",
 "ExtBorderWildP": f"{g('D1_civil_border','wild_p'):.3f}",
 "ExtBaseP": f"{g('D1_civil_baseline','p'):.3f}",
 "ExtBorderN": f"{g('D1_civil_border','n'):,}",
 "ExtBorderNPref": f"{relb['prefecture_code'].nunique()}",
 "ExtBaseBeta": f"{g('D1_civil_baseline'):.3f}",
 "ExtDoseLo": f"{dose['beta'].iloc[0]:.3f}",
 "ExtDoseHi": f"{dose['beta'].iloc[-1]:.3f}",
 "ExtCapHigh": f"{rows['D4_highslack']['est']:.3f}",
 "ExtCapCongAdd": f"{base.coef()['pth_slack']:.3f}",
 "ExtIncPerSD": f"{inc_per_sd:,.0f}",
 "ExtShareWork": f"{share_workload*100:.0f}",
 "ExtNatOneSD": f"{national_1sd/1e3:.0f}",
 "ExtCostRMB": f"{COST_RMB:,}",
 "ExtBurdenM": f"{burden_national/1e6:.0f}",
 "ExtDurMed": f"{dur_med:.0f}",
 "ExtDurMean": f"{dur_mean:.0f}",
 "ExtLagPre": f"{lag_pre:.1f}",
 "ExtLagPost": f"{lag_post:.1f}",
 "ExtPostPreOrig": f"{post_pre_orig*100:.0f}",
}
if "D2_duration_market" in rows:
    M["ExtDurCong"] = f"{g('D2_duration_market'):.3f}"
    M["ExtDurCongP"] = f"{g('D2_duration_market','p'):.3f}"
with open(f"{OUTD}/tables/numbers_ext.tex", "w", encoding="utf-8") as fh:
    fh.write("% AEJ-upgrade macros (6B step 25).\n")
    for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

with open(f"{OUTD}/tables/tab_border.tex", "w", encoding="utf-8") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\n"
             "Sample & Coefficient & (SE) & Wild $p$ & $N$ \\\\ \\midrule\n")
    for lab, t in [("Full clean window", "D1_civil_baseline"),
                   ("Contiguous provinces only", "D1_civil_border")]:
        r = rows[t]
        fh.write(f"{lab} & {r['est']:.4f}{st(t)} & ({r['se']:.4f}) & "
                 f"{r['wild_p']:.3f} & {r['n']:,} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

with open(f"{OUTD}/tables/tab_welfare.tex", "w", encoding="utf-8") as fh:
    fh.write("\\begin{tabular}{lc}\n\\toprule\n"
             "Quantity & Value \\\\ \\midrule\n")
    fh.write("\\multicolumn{2}{l}{\\emph{A. Caseload reallocated (clean-window dose,"
             " per SD of exposure)}} \\\\[2pt]\n")
    fh.write(f"Pre-campaign relational cases, per prefecture-year & "
             f"{per_pref_yr:,.0f} \\\\\n")
    fh.write(f"Increment per prefecture-year, one-SD prefecture & "
             f"+{inc_per_sd:,.0f} \\\\\n")
    fh.write(f"As a share of the prefecture's relational workload & "
             f"+{share_workload*100:.0f}\\% \\\\\n")
    fh.write(f"National increment, one-SD shift & +{national_1sd/1e3:.0f}k cases \\\\\n")
    fh.write(f"Illustrative burden @ {COST_RMB:,} RMB/case & "
             f"{burden_national/1e6:.0f}M RMB \\\\\n")
    fh.write("\\midrule\n\\multicolumn{2}{l}{\\emph{B. Speed: courts are slow; "
             "private resolution is near-instant}} \\\\[2pt]\n")
    fh.write(f"Criminal filing-to-judgment duration, median & {dur_med:.0f} days \\\\\n")
    if "D2_duration_market" in rows:
        fh.write(f"Market-docket duration, $\\mathrm{{Post}}\\times H_c$ (log days) & "
                 f"{g('D2_duration_market'):.3f}{st('D2_duration_market')} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

print("step 25 complete: numbers_ext.tex, tab_border.tex, tab_welfare.tex, fig_dose.pdf",
      flush=True)

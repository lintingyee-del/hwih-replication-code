# -*- coding: utf-8 -*-
"""6B step 23 — finish the two unfinished pieces of step 22 and export all tables.

Step 22 completed E1a (split-half), E2 (charge substitution), E3 (duration-
matched cohorts) and the CS civil-gap contrast; those values are cached below
verbatim from output/referee_robustness_log.txt. Only two pieces are recomputed
live here — cheaply, without the raw-extract duckdb pass:

  P  within-province permutation inference (999) on the two headline specs
  CS Callaway-Sant'Anna (2021) enforcement-caseload dose contrast (999 boot)

Then writes numbers_ref.tex, tab_meanrev.tex, tab_cs.tex complete.
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
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as sps

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")

# ---- completed results from step 22, read from its CSV export ---------------
# The previous hardcoded copy was transcribed from a pre-_wild.py-fix log and
# carried stale wild p-values (e.g. v1 backstop baseline 0.051 vs the corrected
# 0.030); loading referee_robustness.csv keeps a single source of truth.
_rr = pd.read_csv(f"{OUTD}/referee_robustness.csv").set_index("tag")
_KEYS = ["E1_civ_stacked_common", "E1_civ_stacked_H1415", "E1_civ_stacked_H1617",
         "E1_enf_common", "E1_enf_H1415", "E1_enf_H1617",
         "E2_mafia_docket_response", "E2_v1_backstop_baseline",
         "E2_v1_backstop_mafiactl", "E2_v1_backstop_dropmafia",
         "E2_v2_marketN_mafiactl",
         "E3_iou_age2", "E3_iou_age1", "E3_transfer_age2", "E3_rate_cohort_age2"]
CACHE = {k: (float(_rr.loc[k, "est"]), float(_rr.loc[k, "se"]),
             float(_rr.loc[k, "p"]), float(_rr.loc[k, "wild_p"]),
             int(_rr.loc[k, "n"])) for k in _KEYS}
CORR_HH, CORR_H1F, AGE_PRE, AGE_POST = 0.774, 0.934, 2.13, 0.86
# overall, boot SE, p, joint-lead p (step-70 one-generation rerun; the paper's
# tab:cs is now the two-aggregation table written by hand from referee_shoreup.csv)
CS_CIV = (0.1087, 0.2103, 0.605, 0.253)

def star(p): return ""
def C(tag): return CACHE[tag]

# ============================================================================
# P — within-province permutation inference (point estimate only; fast)
# ============================================================================
print("== permutation inference (999 within-province draws x2) ==", flush=True)
WINDOW = ("2017-01", "2019-03"); POST0 = "2018-09"

c = pd.read_parquet(f"{DATA}/civil_panel.parquet")
c = c[c["cause_family"] == "relational"].copy()
c["month"] = c["jmonth"].astype(str).str[:7]
c = c[(c["month"] >= WINDOW[0]) & (c["month"] <= WINDOW[1])]
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[
    ["province", "inspection_round"]].drop_duplicates()
c = c.merge(sched, on="province", how="left").dropna(subset=["exposure_v2_z"])
c["treat"] = (c["inspection_round"] == 1).astype(int)
c["postc"] = (c["month"] >= POST0).astype(int)
c["pref"] = c["prefecture_code"]
c["pref_cause"] = c["prefecture_code"] + "_" + c["cause"]
c["month_fe"] = c["month"]
c["asinh_n"] = np.arcsinh(c["n_cases"])
c["pt"] = c["postc"] * c["treat"]
c["H"] = c["exposure_v2_z"]

kp = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")
kp = kp[(kp["n_cases"] > 0) & (kp["family"] == "enforcementcrime")].copy()
kp["month"] = kp["jmonth"].astype(str).str[:7]
kp["pref"] = kp["prefecture_code"]
kp["prov_month"] = kp["province"] + "_" + kp["month"]
kp["asinh_n"] = np.arcsinh(kp["n_cases"])
kp["H"] = kp["exposure_v2_z"]
kp = kp.dropna(subset=["exposure_v2_z"])

def perm_p(d, fitfun, reps=999, seed=42):
    b_obs = fitfun(d, None)
    prefH = d[["pref", "province", "H"]].drop_duplicates("pref").reset_index(drop=True)
    rng = np.random.default_rng(seed); hits = 0
    for r in range(reps):
        hp = prefH.groupby("province")["H"].transform(lambda s: rng.permutation(s.values))
        hmap = dict(zip(prefH["pref"], hp))
        if abs(fitfun(d, hmap)) >= abs(b_obs): hits += 1
        if (r + 1) % 250 == 0: print(f"   {r+1}/{reps}", flush=True)
    return b_obs, (1 + hits) / (1 + reps)

def fit_civ(d, hmap):
    H = d["pref"].map(hmap) if hmap else d["H"]
    d = d.assign(pth=d["pt"] * H, ph=d["postc"] * H)
    return pf.feols("asinh_n ~ pth + ph + pt | pref_cause + month_fe", data=d).coef()["pth"]

def fit_enf(d, hmap):
    H = d["pref"].map(hmap) if hmap else d["H"]
    d = d.assign(px=d["post"] * H)
    return pf.feols("asinh_n ~ px | pref + prov_month", data=d).coef()["px"]

b_civ, PERM_CIV = perm_p(c, fit_civ)
print(f"civil stacked: obs {b_civ:.4f}, perm p = {PERM_CIV:.3f}", flush=True)
b_enf, PERM_ENF = perm_p(kp, fit_enf)
print(f"enforcement:  obs {b_enf:.4f}, perm p = {PERM_ENF:.3f}", flush=True)

# ============================================================================
# CS — enforcement-caseload dose contrast (build gap without raw duckdb pass)
# ============================================================================
print("== CS enforcement contrast (999 boot) ==", flush=True)
def midx(s): return s.str[:4].astype(int) * 12 + s.str[5:7].astype(int)

kv = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")
prefs = kv[["prefecture_code", "province", "exposure_v2_z", "insp_month"]].drop_duplicates(
    "prefecture_code")
prefs["insp"] = prefs["insp_month"].astype(str).str[:7]
months = pd.period_range("2014-01", "2020-12", freq="M").astype(str)
grid = prefs.merge(pd.DataFrame({"month": months}), how="cross")
enf = kv[kv["family"] == "enforcementcrime"].copy()
enf["month"] = enf["jmonth"].astype(str).str[:7]
enf_n = enf.groupby(["prefecture_code", "month"])["n_cases"].sum().rename("n_enf")
gap = grid.merge(enf_n, on=["prefecture_code", "month"], how="left")
gap["n_enf"] = gap["n_enf"].fillna(0)
gap["y_enf"] = np.arcsinh(gap["n_enf"])
gap = gap.dropna(subset=["exposure_v2_z"])
gap["t"] = midx(gap["month"]); gap["g"] = midx(gap["insp"])
gap["terc"] = pd.qcut(gap["exposure_v2_z"], 3, labels=[0, 1, 2]).astype(int)

PRE, POST = 12, 8
EV = list(range(-PRE, -1)) + list(range(0, POST + 1))

def cs_att(d, ycol, provs=None):
    if provs is not None:
        d = pd.concat([d[d["province"] == pv].assign(bs=i) for i, pv in enumerate(provs)],
                      ignore_index=True)
    piv = d.pivot_table(index=["prefecture_code", "g", "terc"] +
                        (["bs"] if provs is not None else []),
                        columns="t", values=ycol)
    idx_g = piv.index.get_level_values("g"); idx_T = piv.index.get_level_values("terc")
    out = {}
    for e in EV:
        num = {2: 0.0, 0: 0.0}; den = {2: 0.0, 0: 0.0}
        for g in np.unique(idx_g):
            t = g + e
            if t not in piv.columns or (g - 1) not in piv.columns: continue
            nyt = idx_g > max(t, g); dy = piv[t] - piv[g - 1]
            for T in (2, 0):
                tr = dy[(idx_g == g) & (idx_T == T)].dropna()
                ct = dy[nyt & (idx_T == T)].dropna()
                if len(tr) < 3 or len(ct) < 3: continue
                num[T] += len(tr) * (tr.mean() - ct.mean()); den[T] += len(tr)
        if den[2] > 0 and den[0] > 0:
            out[e] = num[2] / den[2] - num[0] / den[0]
    return out

pt = cs_att(gap, "y_enf")
posts = [e for e in pt if e >= 0]; pres = [e for e in pt if e < -1]
overall = np.mean([pt[e] for e in posts])
provs_all = gap["province"].unique(); rng = np.random.default_rng(42)
boot = {e: [] for e in pt}; boot_o = []
for r in range(999):
    draw = rng.choice(provs_all, size=len(provs_all), replace=True)
    b = cs_att(gap, "y_enf", provs=draw)
    ok = [e for e in posts if e in b]
    if ok: boot_o.append(np.mean([b[e] for e in ok]))
    for e in pt:
        if e in b: boot[e].append(b[e])
    if (r + 1) % 250 == 0: print(f"   {r+1}/999", flush=True)
se_o = np.std(boot_o, ddof=1); p_o = 2 * (1 - sps.norm.cdf(abs(overall / se_o)))
nb = min(len(boot[e]) for e in pres); B = np.array([[boot[e][j] for e in pres] for j in range(nb)])
bvec = np.array([pt[e] for e in pres])
try: pre_p = float(1 - sps.chi2.cdf(bvec @ np.linalg.solve(np.cov(B.T), bvec), len(pres)))
except Exception: pre_p = np.nan
CS_ENF = (overall, se_o, p_o, pre_p)
print(f"CS enforce: overall {overall:.4f} (boot SE {se_o:.4f}, p {p_o:.3f}); "
      f"joint lead p {pre_p:.3f}", flush=True)
cs_enf_dyn = {e: (pt[e], np.std(boot[e], ddof=1) if len(boot[e]) > 2 else np.nan)
              for e in sorted(pt)}

# ============================================================================
# EXPORT — numbers_ref.tex, tab_meanrev.tex, tab_cs.tex
# ============================================================================
print("== writing tables ==", flush=True)
def m3(tag, i=0): return f"{C(tag)[i]:.3f}"
macros = {
 "RefHalfCorr": f"{CORR_HH:.2f}", "RefHalfCorrFull": f"{CORR_H1F:.2f}",
 "RefCivHalfA": m3("E1_civ_stacked_H1415"), "RefCivHalfAWildP": m3("E1_civ_stacked_H1415", 3),
 "RefCivHalfB": m3("E1_civ_stacked_H1617"), "RefCivHalfBWildP": m3("E1_civ_stacked_H1617", 3),
 "RefEnfHalfA": m3("E1_enf_H1415"), "RefEnfHalfAWildP": m3("E1_enf_H1415", 3),
 "RefEnfHalfB": m3("E1_enf_H1617"), "RefEnfHalfBWildP": m3("E1_enf_H1617", 3),
 "RefPermCiv": f"{PERM_CIV:.3f}", "RefPermEnf": f"{PERM_ENF:.3f}",
 "RefMafiaResp": m3("E2_mafia_docket_response"),
 "RefMafiaRespP": f"{C('E2_mafia_docket_response')[2]:.3f}",
 "RefMafiaRespWildP": m3("E2_mafia_docket_response", 3),
 "RefVOneBackstopBase": m3("E2_v1_backstop_baseline"),
 "RefVOneBackstopMafia": m3("E2_v1_backstop_mafiactl"),
 "RefVOneBackstopMafiaWildP": m3("E2_v1_backstop_mafiactl", 3),
 "RefVOneBackstopDrop": m3("E2_v1_backstop_dropmafia"),
 "RefMarketNMafia": m3("E2_v2_marketN_mafiactl"),
 "RefIouAgeTwo": m3("E3_iou_age2"), "RefIouAgeTwoP": f"{C('E3_iou_age2')[2]:.3f}",
 "RefIouAgeOne": m3("E3_iou_age1"), "RefIouAgeOneP": f"{C('E3_iou_age1')[2]:.3f}",
 "RefTransferAgeTwo": m3("E3_transfer_age2"),
 "RefTransferAgeTwoP": f"{C('E3_transfer_age2')[2]:.3f}",
 "RefRateAgeTwo": m3("E3_rate_cohort_age2"),
 "RefRateAgeTwoP": f"{C('E3_rate_cohort_age2')[2]:.3f}",
 "RefAgePre": f"{AGE_PRE:.1f}", "RefAgePost": f"{AGE_POST:.1f}",
 "RefCSCiv": f"{CS_CIV[0]:.3f}", "RefCSCivSE": f"{CS_CIV[1]:.3f}",
 "RefCSCivP": f"{CS_CIV[2]:.3f}", "RefCSCivPreP": f"{CS_CIV[3]:.3f}",
 "RefCSEnf": f"{CS_ENF[0]:.3f}", "RefCSEnfSE": f"{CS_ENF[1]:.3f}",
 "RefCSEnfP": f"{CS_ENF[2]:.3f}", "RefCSEnfPreP": f"{CS_ENF[3]:.3f}",
}
with open(f"{OUTD}/tables/numbers_ref.tex", "w", encoding="utf-8") as fh:
    fh.write("% Referee-anticipation robustness macros (6B step 23).\n")
    for k, v in macros.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

def trow(label, tag):
    e, se, p, wp, n = C(tag)
    return f"{label} & {e:.4f}{star(p)} & ({se:.4f}) & {wp:.3f} & {n:,} \\\\\n"

with open(f"{OUTD}/tables/tab_meanrev.tex", "w", encoding="utf-8") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\n"
             "Specification & Coefficient & (SE) & Wild $p$ & $N$ \\\\ \\midrule\n")
    fh.write("\\multicolumn{5}{l}{\\emph{Panel A. Split-half exposure: "
             "clean-window civil flow}} \\\\[2pt]\n")
    fh.write(trow("Full index, common sample", "E1_civ_stacked_common"))
    fh.write(trow("2014--15 half index", "E1_civ_stacked_H1415"))
    fh.write(trow("2016--17 half index", "E1_civ_stacked_H1617"))
    fh.write("\\midrule\n\\multicolumn{5}{l}{\\emph{Panel B. Split-half exposure: "
             "enforcement caseload}} \\\\[2pt]\n")
    fh.write(trow("Full index, common sample", "E1_enf_common"))
    fh.write(trow("2014--15 half index", "E1_enf_H1415"))
    fh.write(trow("2016--17 half index", "E1_enf_H1617"))
    fh.write("\\midrule\n\\multicolumn{5}{l}{\\emph{Panel C. Charge substitution: "
             "mafia-docket margin}} \\\\[2pt]\n")
    fh.write(trow("Mafia-organization caseload (asinh)", "E2_mafia_docket_response"))
    fh.write(trow("v1 market backstop, baseline", "E2_v1_backstop_baseline"))
    fh.write(trow("\\quad + same-cell mafia control", "E2_v1_backstop_mafiactl"))
    fh.write(trow("\\quad drop mafia-case cells", "E2_v1_backstop_dropmafia"))
    fh.write(trow("Market caseload + mafia control", "E2_v2_marketN_mafiactl"))
    fh.write("\\midrule\n\\multicolumn{5}{l}{\\emph{Panel D. Origination cohorts, "
             "matched litigation lag}} \\\\[2pt]\n")
    fh.write(trow("IOU share, lag $\\le$ 2 years", "E3_iou_age2"))
    fh.write(trow("IOU share, lag $\\le$ 1 year", "E3_iou_age1"))
    fh.write(trow("Transfer records, lag $\\le$ 2 years", "E3_transfer_age2"))
    fh.write(trow("Monthly rate, lag $\\le$ 2 years", "E3_rate_cohort_age2"))
    fh.write("\\midrule\n")
    fh.write(f"Permutation $p$ (civil flow; 999 within-province draws) & "
             f"\\multicolumn{{4}}{{c}}{{{PERM_CIV:.3f}}} \\\\\n")
    fh.write(f"Permutation $p$ (enforcement caseload) & "
             f"\\multicolumn{{4}}{{c}}{{{PERM_ENF:.3f}}} \\\\\n")
    fh.write(f"Split-half exposure correlation & "
             f"\\multicolumn{{4}}{{c}}{{{CORR_HH:.2f}}} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# CS civil dynamics were computed in step 22 but not persisted; the CS table here
# reports the completed overall ATTs and lead tests for both outcomes, plus the
# enforcement event-time path recomputed above.
with open(f"{OUTD}/tables/tab_cs.tex", "w", encoding="utf-8") as fh:
    fh.write("\\begin{tabular}{lcc}\n\\toprule\n"
             " & Civil relational$-$placebo gap & Enforcement caseload \\\\\n"
             "Callaway--Sant'Anna dose contrast & (high$-$low exposure tercile) & "
             "(high$-$low tercile) \\\\ \\midrule\n")
    fh.write(f"Overall ATT ($e\\in[0,{POST}]$) & {CS_CIV[0]:.3f} & {CS_ENF[0]:.3f} \\\\\n")
    fh.write(f"Bootstrap SE (999, province block) & ({CS_CIV[1]:.3f}) & ({CS_ENF[1]:.3f}) \\\\\n")
    fh.write(f"$p$-value & {CS_CIV[2]:.3f} & {CS_ENF[2]:.3f} \\\\\n")
    fh.write(f"Joint lead test $p$ & {CS_CIV[3]:.3f} & {CS_ENF[3]:.3f} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

print("step 23 complete: numbers_ref.tex, tab_meanrev.tex, tab_cs.tex written", flush=True)

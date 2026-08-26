# -*- coding: utf-8 -*-
"""6B step 27 — prefecture-level staggered design from mobilization dates.

User-supplied prefecture mobilization-meeting months (docs/prefecture_campaign_filled.csv;
the local saohei chu'e dong-yuan bu-shu hui, the most exogenous local launch timing,
web-sourced with per-row citations in prefecture_campaign_evidence.csv). 151 prefectures
across 21 provinces, with within-province variation in 16 of them.

We use the mobilization month as a PREFECTURE-level treatment onset and identify off
within-province cross-prefecture timing (province x month FE absorb the province-level
central-inspection clock and macro shocks). This lifts the design from 31 province
clusters to prefecture granularity.

Outcomes: civil relational flow, criminal market backstop (v2), enforcement caseload.
Reports binary post-mobilization and dose (x H), clustered by prefecture and by province,
plus an event study. Compares to the province-level baseline.
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
from scipy import stats as sps

DATA = str(_REP_PROJECT / "data")
DOCS = str(_REP_PROJECT / 'docs').replace('\\', '/')
OUTD = str(_REP_PROJECT / "output")
rows = {}

def mi(s):  # 'YYYY-MM' -> month index
    s = s.astype(str)
    return s.str[:4].astype(int) * 12 + s.str[5:7].astype(int)

def run(tag, fml, df, coef, cl, weights=None):
    m = pf.feols(fml, data=df, vcov={"CRV1": cl}, weights=weights)
    rows[tag] = dict(est=m.coef()[coef], se=m.se()[coef], p=m.pvalue()[coef],
                     n=int(m._N), ncl=df[cl].nunique())
    print(f"{tag:40s} {m.coef()[coef]: .4f} ({m.se()[coef]:.4f}) "
          f"p={m.pvalue()[coef]:.3f} N={m._N} clusters={df[cl].nunique()}({cl})",
          flush=True)
    return m

# ---------------- mobilization months ----------------------------------------
mob = pd.read_csv(f"{DOCS}/prefecture_campaign_filled.csv",
                  dtype={"prefecture_code": str})
mob = mob[mob["campaign_start_month"].notna()][
    ["prefecture_code", "campaign_start_month"]].copy()
mob["mob_mi"] = mi(mob["campaign_start_month"])
print(f"mobilization coverage: {len(mob)} prefectures, "
      f"{mob['campaign_start_month'].nunique()} distinct months "
      f"({mob['campaign_start_month'].min()}..{mob['campaign_start_month'].max()})",
      flush=True)

def prep(panel_path, keep):
    p = pd.read_parquet(panel_path)
    p = p[keep(p)].copy()
    p["month"] = p["jmonth"].astype(str).str[:7]
    p = p.merge(mob, on="prefecture_code", how="inner")   # 151-pref subsample
    p["t_mi"] = mi(p["month"])
    p["post_mob"] = (p["t_mi"] >= p["mob_mi"]).astype(int)
    p["etime"] = p["t_mi"] - p["mob_mi"]
    p["prov"] = p["prefecture_code"].str[:2]
    p["prov_month"] = p["prov"] + "_" + p["month"]
    p["pref"] = p["prefecture_code"]
    p["H"] = p["exposure_v2_z"]
    p["pmH"] = p["post_mob"] * p["H"]
    return p.dropna(subset=["exposure_v2_z"])

# ================= CIVIL relational flow =====================================
cv = prep(f"{DATA}/civil_panel.parquet", lambda p: p["cause_family"] == "relational")
cv["asinh_n"] = np.arcsinh(cv["n_cases"])
cv["pref_cause"] = cv["prefecture_code"] + "_" + cv["cause"]
print(f"\n== CIVIL relational flow ({cv['pref'].nunique()} prefectures, "
      f"{cv['prov'].nunique()} provinces) ==", flush=True)
# binary post-mobilization, within-province identification (prov x month FE)
run("civ_postmob_provmonth_clpref",
    "asinh_n ~ post_mob | pref_cause + prov_month", cv, "post_mob", "pref")
run("civ_postmob_provmonth_clprov",
    "asinh_n ~ post_mob | pref_cause + prov_month", cv, "post_mob", "prov")
# plain month FE (uses both within- and cross-province timing)
run("civ_postmob_month_clpref",
    "asinh_n ~ post_mob | pref_cause + month", cv, "post_mob", "pref")
# dose interaction
run("civ_dose_provmonth_clprov",
    "asinh_n ~ pmH + post_mob | pref_cause + prov_month", cv, "pmH", "prov")

# ================= CRIMINAL market backstop (v2) =============================
kp = prep(f"{DATA}/crim_panel_v2.parquet",
          lambda p: (p["n_cases"] > 0) & (p["family"] == "market"))
print(f"\n== CRIMINAL market backstop ({kp['pref'].nunique()} prefectures) ==",
      flush=True)
run("crim_backstop_postmob_clpref",
    "y_backstop ~ post_mob + x_doclen | pref + prov_month", kp, "post_mob", "pref",
    weights="n_cases")
run("crim_backstop_postmob_clprov",
    "y_backstop ~ post_mob + x_doclen | pref + prov_month", kp, "post_mob", "prov",
    weights="n_cases")

# ================= CRIMINAL enforcement caseload =============================
en = prep(f"{DATA}/crim_panel_v2.parquet",
          lambda p: (p["n_cases"] > 0) & (p["family"] == "enforcementcrime"))
en["asinh_n"] = np.arcsinh(en["n_cases"])
print(f"\n== CRIMINAL enforcement caseload ({en['pref'].nunique()} prefectures) ==",
      flush=True)
run("enf_postmob_clpref",
    "asinh_n ~ post_mob | pref + prov_month", en, "post_mob", "pref")
run("enf_postmob_clprov",
    "asinh_n ~ post_mob | pref + prov_month", en, "post_mob", "prov")

# ================= event study (civil, within-province) =====================
print("\n== civil event study around mobilization (prov x month FE) ==", flush=True)
BINS = [(-24, -13), (-12, -7), (-6, -1), (0, 5), (6, 11), (12, 23)]
terms = []
for lo, hi in BINS:
    if (lo, hi) == (-6, -1):
        continue
    nm = f"e_{lo}_{hi}".replace("-", "m")
    cv[nm] = ((cv["etime"] >= lo) & (cv["etime"] <= hi)).astype(int)
    terms.append((nm, lo, hi))
es = pf.feols("asinh_n ~ " + " + ".join(t[0] for t in terms) +
              " | pref_cause + prov_month", data=cv, vcov={"CRV1": "prov"})
leads = [t[0] for t in terms if t[2] < 0]
names = list(es.coef().index)
b = es.coef()[leads].values
V = es._vcov[np.ix_([names.index(x) for x in leads], [names.index(x) for x in leads])]
pre_p = float(1 - sps.chi2.cdf(float(b @ np.linalg.solve(V, b)), len(leads)))
print(f"joint lead test p = {pre_p:.3f}", flush=True)
for nm, lo, hi in terms:
    print(f"  e[{lo:>3},{hi:>3}] {es.coef()[nm]:+.4f} ({es.se()[nm]:.4f})", flush=True)

# ---------------- exports ----------------------------------------------------
def g(t, k="est"): return rows[t][k]
def stp(t):
    p = rows[t]["p"]; return "***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""
M = {
 "PrefNnpref": f"{cv['pref'].nunique()}", "PrefNprov": f"{cv['prov'].nunique()}",
 "PrefCivBeta": f"{g('civ_postmob_provmonth_clprov'):.3f}",
 "PrefCivSE": f"{g('civ_postmob_provmonth_clprov','se'):.3f}",
 "PrefCivPprov": f"{g('civ_postmob_provmonth_clprov','p'):.3f}",
 "PrefCivPpref": f"{g('civ_postmob_provmonth_clpref','p'):.3f}",
 "PrefCivNcl": f"{g('civ_postmob_provmonth_clpref','ncl')}",
 "PrefCrimBeta": f"{g('crim_backstop_postmob_clprov'):.3f}",
 "PrefCrimPprov": f"{g('crim_backstop_postmob_clprov','p'):.3f}",
 "PrefCrimPpref": f"{g('crim_backstop_postmob_clpref','p'):.3f}",
 "PrefEnfBeta": f"{g('enf_postmob_clprov'):.3f}",
 "PrefEnfPprov": f"{g('enf_postmob_clprov','p'):.3f}",
 "PrefEnfPpref": f"{g('enf_postmob_clpref','p'):.3f}",
 "PrefEsPreP": f"{pre_p:.3f}",
}
with open(f"{OUTD}/tables/numbers_pref.tex", "w", encoding="utf-8") as fh:
    fh.write("% prefecture-staggered macros (6B step 27).\n")
    for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

with open(f"{OUTD}/tables/tab_pref.tex", "w", encoding="utf-8") as fh:
    fh.write("\\begin{tabular}{lccc}\n\\toprule\n"
             " & Coefficient & (SE) & $N$ \\\\\n"
             "Outcome & \\multicolumn{3}{c}{$\\mathrm{Post\\text{-}mobilization}$, "
             "prov.$\\times$month FE} \\\\ \\midrule\n")
    for lab, t in [("Relational civil flow (asinh)", "civ_postmob_provmonth_clprov"),
                   ("\\quad clustered by prefecture", "civ_postmob_provmonth_clpref"),
                   ("Market hard-backstop share", "crim_backstop_postmob_clprov"),
                   ("Enforcement caseload (asinh)", "enf_postmob_clprov")]:
        r = rows[t]
        fh.write(f"{lab} & {r['est']:.4f}{stp(t)} & ({r['se']:.4f}) & {r['n']:,} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print("\nstep 27 complete: numbers_pref.tex, tab_pref.tex", flush=True)

# -*- coding: utf-8 -*-
"""6B step 18 — P2P / concurrent-policy confound battery.

Concern: the 2018-2020 P2P collapse correlates geographically with violent-collection
exposure and mechanically raises lending litigation. Battery:
  (a) control Post x DigitalFinance(2017) [PKU DFI credit + total sub-indices]
  (b) drop top-quartile DFI-credit prefectures
  (c) rerun BOTH the full staggered C1 and the clean-window stacked flow spec
  (d) criminal enforcement-caseload spec with the same control (placebo for the
      confound: P2P collapse has no reason to reduce violent-enforcement caseloads)
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
import duckdb, pandas as pd, numpy as np, pyfixest as pf

SIX_A = str(_REP_CASE_ARCHIVE)
DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
con = duckdb.connect()
rows = []

dfi = con.sql(f"""
SELECT prefecture_code,
  AVG(pku_dfi_credit) FILTER (WHERE year(judgment_date)=2017) AS dfi_credit_2017,
  ANY_VALUE(pku_dfi_total_2017)  AS dfi_total_2017
FROM '{SIX_A}' WHERE prefecture_code<>'' GROUP BY 1
""").df()
for c in ["dfi_credit_2017", "dfi_total_2017"]:
    dfi[c + "_z"] = (dfi[c] - dfi[c].mean()) / dfi[c].std()
print("DFI coverage:", dfi["dfi_credit_2017"].notna().mean().round(3), "of",
      len(dfi), "prefectures")

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


def run(tag, fml, df, coef, weights=None):
    m = pf.feols(fml, data=df, vcov={"CRV1": "prov_id"}, weights=weights)
    try: wp = wild_score_p(fml, df, coef, weights)
    except Exception: wp = np.nan
    rows.append(dict(tag=tag, coef=coef, est=m.coef()[coef], se=m.se()[coef],
                     p=m.pvalue()[coef], wild_p=wp, n=int(m._N)))
    print(f"{tag:44s} {m.coef()[coef]: .5f} ({m.se()[coef]:.5f}) "
          f"p={m.pvalue()[coef]:.4f} wild={wp:.3f} N={m._N}")

# ---------- civil triple-diff (full staggered) --------------------------------
cp = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cp = cp[cp["cause_family"].isin(["relational", "placebo"])].copy()
cp = cp.merge(dfi, on="prefecture_code", how="left")
cp["month"] = cp["jmonth"].astype(str)
cp["prov_id"] = pd.factorize(cp["province"])[0]
cp["rel"] = (cp["cause_family"] == "relational").astype(int)
cp["px"] = cp["post"] * cp["exposure_v2_z"]
cp["pxr"] = cp["px"] * cp["rel"]
cp["pr"] = cp["post"] * cp["rel"]
cp["pdfi"] = cp["post"] * cp["dfi_credit_2017_z"]
cp["pdfir"] = cp["pdfi"] * cp["rel"]
cp["asinh_n"] = np.arcsinh(cp["n_cases"])
cp["pref_cause"] = cp["prefecture_code"] + "_" + cp["cause"]
cp["prov_month"] = cp["province"] + "_" + cp["month"]
cp["cause_month"] = cp["cause"] + "_" + cp["month"]

base = "asinh_n ~ pxr + px + pr | pref_cause + prov_month + cause_month"
run("P1_C1_baseline_reprint", base, cp, "pxr")
run("P1_C1_dficontrol",
    "asinh_n ~ pxr + px + pr + pdfir + pdfi | pref_cause + prov_month + cause_month",
    cp.dropna(subset=["dfi_credit_2017_z"]), "pxr")
q75 = cp["dfi_credit_2017_z"].quantile(0.75)
run("P1_C1_dropTopDFI", base, cp[cp["dfi_credit_2017_z"] < q75], "pxr")

# corr between exposures
cc = cp.drop_duplicates("prefecture_code")[["exposure_v2_z", "dfi_credit_2017_z"]].dropna()
rho = cc.corr().iloc[0, 1]
print(f"corr(H, DFI credit 2017) = {rho:.3f}")
rows.append(dict(tag="P1_corr_H_DFI", coef="rho", est=rho, se=np.nan, p=np.nan,
                 wild_p=np.nan, n=len(cc)))

# ---------- clean-window stacked flow with DFI control ------------------------
CLEAN_START = pd.Timestamp("2017-01-01")
CLEAN_END = pd.Timestamp("2019-04-01")
c = pd.read_parquet(f"{DATA}/civil_panel.parquet")
c = c[c["cause_family"] == "relational"].copy()
c = c.merge(dfi, on="prefecture_code", how="left")
c["judgment_date"] = pd.to_datetime(c["jmonth"], errors="coerce")
c = c[(c["judgment_date"] >= CLEAN_START) &
      (c["judgment_date"] < CLEAN_END)].copy()
c["month"] = c["judgment_date"].dt.strftime("%Y-%m")
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]] \
    .drop_duplicates()
c = c.merge(sched, on="province", how="left")
c["treat"] = (c["inspection_round"] == 1).astype(int)
c["postc"] = (c["month"] >= "2018-09").astype(int)
c["prov_id"] = pd.factorize(c["province"])[0]
c["pref_cause"] = c["prefecture_code"] + "_" + c["cause"]
c["month_fe"] = c["month"]
c["asinh_n"] = np.arcsinh(c["n_cases"])
c["pt"] = c["postc"] * c["treat"]
c["pth"] = c["pt"] * c["exposure_v2_z"]
c["ph"] = c["postc"] * c["exposure_v2_z"]
c["pdfi"] = c["postc"] * c["dfi_credit_2017_z"]
c["ptdfi"] = c["pt"] * c["dfi_credit_2017_z"]
run("P1_stacked_baseline_reprint",
    "asinh_n ~ pth + ph + pt | pref_cause + month_fe", c, "pth")
run("P1_stacked_dficontrol",
    "asinh_n ~ pth + ph + pt + ptdfi + pdfi | pref_cause + month_fe",
    c.dropna(subset=["dfi_credit_2017_z"]), "pth")
run("P1_stacked_dropTopDFI",
    "asinh_n ~ pth + ph + pt | pref_cause + month_fe",
    c[c["dfi_credit_2017_z"] < q75], "pth")

# ---------- criminal enforcement caseload with DFI control --------------------
kp = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")
kp = kp[(kp["family"] == "enforcementcrime") & (kp["n_cases"] > 0)].copy()
kp = kp.merge(dfi, on="prefecture_code", how="left")
kp["month"] = kp["jmonth"].astype(str)
kp["prov_id"] = pd.factorize(kp["province"])[0]
kp["px"] = kp["post"] * kp["exposure_v2_z"]
kp["pdfi"] = kp["post"] * kp["dfi_credit_2017_z"]
kp["pref"] = kp["prefecture_code"]
kp["prov_month"] = kp["province"] + "_" + kp["month"]
kp["asinh_n"] = np.arcsinh(kp["n_cases"])
run("P1_enforceN_dficontrol",
    "asinh_n ~ px + pdfi | pref + prov_month",
    kp.dropna(subset=["dfi_credit_2017_z"]), "px")

old = pd.read_csv(f"{OUTD}/results_v2.csv")
new = pd.concat([old[~old["tag"].str.startswith("P1_")], pd.DataFrame(rows)])
new.to_csv(f"{OUTD}/results_v2.csv", index=False)
print("P2P confound battery saved")

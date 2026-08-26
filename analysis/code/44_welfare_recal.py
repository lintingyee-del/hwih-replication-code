# -*- coding: utf-8 -*-
"""6B step 44 (b + a-salvage) — welfare recalibration macros. (b): replace the illustrative
3,000 RMB/case with the STATUTORY court fee applied to the observed relational-claim
distribution (external + data-driven), plus the 2021 all-in litigant benchmark. (a-salvage):
emit the civil filing-to-judgment duration response (congestion test: it did NOT rise), and
the filing-extractor audit precision. Writes manuscript.../tables/numbers_welfare2.tex."""

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
import pandas as pd, numpy as np
DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PACKAGE / "manuscript" / "tables" / "numbers_welfare2.tex")
NAT_ONE_SD_K = 117  # national one-SD caseload increment, thousand cases (Panel A, existing)

def fee(a):  # 财产案件受理费, 诉讼费用交纳办法 2007 (cumulative marginal brackets)
    if a is None or a != a or a <= 0: return np.nan
    if a <= 10000: return 50.0
    br = [(100000,.025),(200000,.02),(500000,.015),(1000000,.01),(2000000,.009),(5000000,.008),(10000000,.007),(20000000,.006)]
    f = 50.0; lo = 10000
    for hi, r in br:
        if a > lo: f += (min(a, hi) - lo) * r; lo = hi
        else: break
    if a > 20000000: f += (a - 20000000) * .005
    return f

d = pd.read_parquet(f"{DATA}/civil_case.parquet", columns=["cause","amount_yuan"])
lend = d[d["cause"] == "民间借贷纠纷"]["amount_yuan"].dropna()
lend = lend[(lend > 0) & (lend < 5e7)]
fees = lend.map(fee)
fee_med, fee_mean = fees.median(), fees.mean()
burden_fee_M = NAT_ONE_SD_K * 1000 * fee_mean / 1e6   # aggregate = count x mean fee
claim_med = lend.median()

sp = pd.read_csv(f"{DATA.replace('data','output')}/e42_congestion_spillover.csv")
civ = sp[sp["spec"].str.startswith("traffic-placebo: log-dur capped")].iloc[0]
civ_b, civ_wild = civ["b"], civ["wild_p"]

macros = {
    "ExtFeeMed": f"{fee_med:,.0f}", "ExtFeeMean": f"{fee_mean:,.0f}",
    "ExtClaimMed": f"{claim_med:,.0f}", "ExtBurdenFee": f"{burden_fee_M:,.0f}",
    "ExtAllIn": "30,578",  # 2021 national avg civil-litigation cost (court big-data, 南方都市报)
    "ExtCourtOp": "1,000",  # order-of-mag court-operating cost/case: 2007 基层法院人均经费 83k RMB/staff (PKU CCJ 司法经费与司法公正; 中国基层法院财政制度实证研究) at a few thousand cases/court; a cross-check on the fee, NOT an additive component (the fee funds the court -> same cost block)
    "ExtDurCivil": f"{civ_b:.2f}", "ExtDurCivilWildP": f"{civ_wild:.2f}", "ExtDurCivilMed": "64",
    "ExtFilingPrec": "97.5", "ExtFilingFN": "1.7",
}
with open(OUT, "w", encoding="utf-8") as f:
    f.write("% welfare recalibration (step 44): statutory-fee anchor + civil-duration null + audit\n")
    for k, v in macros.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
for k, v in macros.items(): print(f"  {k} = {v}")
print(f"[done] -> {OUT}")

# -*- coding: utf-8 -*-
"""6B step 62 — referee battery around the clean-window headline.
A. FILING-MONTH rerun of S1_civil_flow_dose: relational-cause cells dated by
   extracted filing month (civil_filing.parquet), symmetric 9-month judgment
   observation horizon (duration <= 270 days; judgment frame ends 2019-12 so
   every filing month in the window has a full horizon). Coverage-neutrality
   check: extraction share regressed on the treatment interaction.
B. Leave-one-province-out on the judgment-month headline (31 re-runs).
C. Wave-selection balance: province means of pre-period characteristics,
   wave 1 vs waves 2-3.
D. Clean-window event study: bins x Treat x H on relational cells.
Output: output/ext2124/filing_lopo_balance.csv (+ printed table)
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
import pandas as pd, numpy as np, pyfixest as pf, sys, io
from _wild import wild_p

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
WINDOW = ("2017-01", "2019-03")
CLEAN_START = pd.Timestamp("2017-01-01")
CLEAN_END = pd.Timestamp("2019-04-01")
POST0 = "2018-09"
rows = []

cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["case_no", "cause", "cause_family", "prefecture_code",
                              "province", "jmonth"])
rel = cc[cc["cause_family"] == "relational"].copy()
rel["jm"] = rel["jmonth"].astype(str).str[:7]
fil = pd.read_parquet(f"{DATA}/civil_filing.parquet")
fil = fil.rename(columns={"案号": "case_no", "案由": "cause_f"})
rel = rel.merge(fil[["case_no", "filing_ymd"]], on="case_no", how="left")
rel["fdate"] = pd.to_datetime(rel["filing_ymd"], errors="coerce")
rel["jdate"] = pd.to_datetime(rel["jmonth"], errors="coerce")
rel["dur"] = (rel["jdate"] - rel["fdate"]).dt.days
ok = rel["fdate"].notna() & rel["dur"].between(0, 270)
rel["fm"] = rel["fdate"].dt.strftime("%Y-%m")
print(f"relational cases {len(rel):,}; filing extracted {rel['fdate'].notna().mean():.3f}; "
      f"usable (0<=dur<=270) {ok.mean():.3f}")

sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code", "exposure_v2_z"]]

def cells(df, mcol):
    g = (df.groupby(["prefecture_code", "province", "cause", mcol]).size()
         .rename("n").reset_index().rename(columns={mcol: "month"}))
    g = g[(g["month"] >= WINDOW[0]) & (g["month"] <= WINDOW[1])]
    g = g.merge(sched, on="province").merge(ex, on="prefecture_code")
    g = g.dropna(subset=["exposure_v2_z", "inspection_round"])
    g["treat"] = (g["inspection_round"] == 1).astype(int)
    g["postc"] = (g["month"] >= POST0).astype(int)
    g["prov_id"] = pd.factorize(g["province"])[0]
    g["pref_cause"] = g["prefecture_code"] + "_" + g["cause"]
    g["asinh_n"] = np.arcsinh(g["n"])
    g["pt"] = g["postc"] * g["treat"]
    g["pth"] = g["pt"] * g["exposure_v2_z"]
    g["ph"] = g["postc"] * g["exposure_v2_z"]
    return g

FML = "asinh_n ~ pth + ph + pt | pref_cause + month"

def run(tag, g):
    m = pf.feols(FML, data=g, vcov={"CRV1": "prov_id"})
    wp = wild_p(FML, g, "pth")
    print(f"{tag:34s} pth={m.coef()['pth']: .4f} (se {m.se()['pth']:.4f}) "
          f"p={m.pvalue()['pth']:.4f} wild={wp:.3f} N={int(m._N):,}")
    rows.append(dict(part=tag, est=m.coef()["pth"], se=m.se()["pth"],
                     p_crv1=m.pvalue()["pth"], p_wild=wp, n=int(m._N)))
    return m

# A. filing-month vs judgment-month, same sample discipline
g_j = cells(rel[ok], "jm")
g_f = cells(rel[ok], "fm")
run("A_judgment_month(dur<=270)", g_j)
run("A_filing_month(dur<=270)", g_f)
# coverage neutrality: extraction share cell-level on pth
cov = (rel.assign(okx=ok.astype(int), month=rel["jm"])
       .groupby(["prefecture_code", "province", "month"])["okx"].mean()
       .rename("share").reset_index())
cov = cov[(cov["month"] >= WINDOW[0]) & (cov["month"] <= WINDOW[1])]
cov = cov.merge(sched, on="province").merge(ex, on="prefecture_code").dropna()
cov["treat"] = (cov["inspection_round"] == 1).astype(int)
cov["postc"] = (cov["month"] >= POST0).astype(int)
cov["prov_id"] = pd.factorize(cov["province"])[0]
cov["pt"] = cov["postc"] * cov["treat"]
cov["pth"] = cov["pt"] * cov["exposure_v2_z"]
cov["ph"] = cov["postc"] * cov["exposure_v2_z"]
mc = pf.feols("share ~ pth + ph + pt | prefecture_code + month", data=cov,
              vcov={"CRV1": "prov_id"})
print(f"A_coverage_neutrality              pth={mc.coef()['pth']: .4f} "
      f"(se {mc.se()['pth']:.4f}) p={mc.pvalue()['pth']:.4f}")
rows.append(dict(part="A_coverage_neutrality", est=mc.coef()["pth"],
                 se=mc.se()["pth"], p_crv1=mc.pvalue()["pth"], p_wild=np.nan,
                 n=int(mc._N)))

# B. leave-one-province-out on the standard judgment-month headline sample
cp = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cp = cp[cp["cause_family"] == "relational"].copy()
cp["judgment_date"] = pd.to_datetime(cp["jmonth"], errors="coerce")
cp = cp[(cp["judgment_date"] >= CLEAN_START) &
        (cp["judgment_date"] < CLEAN_END)].copy()
cp["month"] = cp["judgment_date"].dt.strftime("%Y-%m")
cp = cp.merge(sched, on="province", how="left")
cp["treat"] = (cp["inspection_round"] == 1).astype(int)
cp["postc"] = (cp["month"] >= POST0).astype(int)
cp["prov_id"] = pd.factorize(cp["province"])[0]
cp["pref_cause"] = cp["prefecture_code"] + "_" + cp["cause"]
cp["asinh_n"] = np.arcsinh(cp["n_cases"])
cp["pt"] = cp["postc"] * cp["treat"]
cp["pth"] = cp["pt"] * cp["exposure_v2_z"]
cp["ph"] = cp["postc"] * cp["exposure_v2_z"]
lopo = []
for pr in sorted(cp["province"].unique()):
    sub = cp[cp["province"] != pr]
    m = pf.feols(FML, data=sub, vcov={"CRV1": "prov_id"})
    lopo.append(dict(dropped=pr, est=m.coef()["pth"], p=m.pvalue()["pth"]))
ld_ = pd.DataFrame(lopo)
print(f"B_LOPO: est range [{ld_.est.min():.4f}, {ld_.est.max():.4f}], "
      f"all-positive={int((ld_.est > 0).all())}, max p={ld_.p.max():.3f}, "
      f"n p<0.10: {(ld_.p < 0.10).sum()}/31")
ld_.to_csv(f"{OUT}/lopo_detail.csv", index=False)
rows.append(dict(part="B_LOPO_min", est=ld_.est.min(), se=np.nan,
                 p_crv1=ld_.p.max(), p_wild=np.nan, n=31))

# C. wave balance (province level, 2017 pre-period)
pre = cp[cp["month"] <= "2017-12"].groupby("province").agg(
    rel_cases=("n_cases", "sum")).reset_index()
hbar = (cp.groupby("province")["exposure_v2_z"].mean().rename("H_mean").reset_index())
bal = pre.merge(hbar, on="province").merge(sched, on="province")
bal["wave1"] = (bal["inspection_round"] == 1).astype(int)
print("C_balance (wave1 vs waves2-3):")
for v in ("rel_cases", "H_mean"):
    a = bal.loc[bal.wave1 == 1, v]; b = bal.loc[bal.wave1 == 0, v]
    nd = (a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()))
    from scipy import stats as sps
    t, p = sps.ttest_ind(a, b, equal_var=False)
    print(f"   {v:12s} w1={a.mean():.2f} w23={b.mean():.2f} normdiff={nd:.3f} p={p:.3f}")
    rows.append(dict(part=f"C_balance_{v}", est=nd, se=np.nan, p_crv1=p,
                     p_wild=np.nan, n=len(bal)))

# D. clean-window event study (calendar bins around 2018-09, Treat x H)
esd = cp.copy()
et = (pd.PeriodIndex(esd["month"], freq="M")
      - pd.Period(POST0, freq="M")).map(lambda x: x.n)
esd["et"] = et
BINS = [(-20, -13), (-12, -7), (0, 6)]
terms = []
for lo, hi in BINS:
    nmH = f"b{lo}_{hi}H".replace("-", "m")
    nmT = f"b{lo}_{hi}T".replace("-", "m")
    inb = ((esd["et"] >= lo) & (esd["et"] <= hi)).astype(float)
    esd[nmH] = inb * esd["treat"] * esd["exposure_v2_z"]
    esd[nmT] = inb * esd["treat"]
    esd[f"b{lo}_{hi}X".replace('-', 'm')] = inb * esd["exposure_v2_z"]
    terms += [nmH, nmT, f"b{lo}_{hi}X".replace('-', 'm')]
mes = pf.feols(f"asinh_n ~ {' + '.join(terms)} | pref_cause + month",
               data=esd, vcov={"CRV1": "prov_id"})
print("D_clean_window_ES (Treat x H bins, ref [-6,-1]):")
for lo, hi in BINS:
    nm = f"b{lo}_{hi}H".replace("-", "m")
    print(f"   [{lo:+d},{hi:+d}] {mes.coef()[nm]: .4f} ({mes.se()[nm]:.4f})")
    rows.append(dict(part=f"D_ES_{lo}_{hi}", est=mes.coef()[nm], se=mes.se()[nm],
                     p_crv1=mes.pvalue()[nm], p_wild=np.nan, n=int(mes._N)))

pd.DataFrame(rows).to_csv(f"{OUT}/filing_lopo_balance.csv", index=False)
print("written:", f"{OUT}/filing_lopo_balance.csv")

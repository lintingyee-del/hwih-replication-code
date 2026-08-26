# -*- coding: utf-8 -*-
"""6B step 65 — three referee additions.
A. PPML (Poisson FE) recheck of the clean-window flow: judgment-month cells,
   filing-month cells, and the 250 km DLR border-pair stack.
B. Border-sample diagnostics: calendar lead bins and covariate balance
   (2017 digital-finance index, bank branches) across the treatment line.
C. Wave-assignment analysis: wave-1 on province observables; fitted propensity
   vs pre-period civil growth.
Output: output/ext2124/ppml_border_assign.csv
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
import duckdb, sys, io
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as sps

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
SRC = str(_REP_CASE_ARCHIVE)
WINDOW = ("2017-01", "2019-03"); POST0 = "2018-09"
rows = []

sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code", "exposure_v2_z"]]

def prep(cells):
    g = cells.merge(sched, on="province").merge(ex, on="prefecture_code")
    g = g.dropna(subset=["exposure_v2_z", "inspection_round"])
    g = g[(g["month"] >= WINDOW[0]) & (g["month"] <= WINDOW[1])]
    g["treat"] = (g["inspection_round"] == 1).astype(int)
    g["postc"] = (g["month"] >= POST0).astype(int)
    g["prov_id"] = pd.factorize(g["province"])[0]
    g["pref_cause"] = g["prefecture_code"].astype(str) + "_" + g["cause"]
    g["pt"] = g["postc"] * g["treat"]
    g["pth"] = g["pt"] * g["exposure_v2_z"]
    g["ph"] = g["postc"] * g["exposure_v2_z"]
    return g

def pois(tag, g, fml="n ~ pth + ph + pt | pref_cause + month"):
    m = pf.fepois(fml, data=g, vcov={"CRV1": "prov_id"})
    print(f"{tag:32s} pth={m.coef()['pth']: .4f} (se {m.se()['pth']:.4f}) "
          f"p={m.pvalue()['pth']:.4f} N={int(m._N):,}")
    rows.append(dict(part=tag, est=m.coef()["pth"], se=m.se()["pth"],
                     p=m.pvalue()["pth"], n=int(m._N)))

# A1: judgment-month cells (paper's panel)
cp = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cp = cp[cp["cause_family"] == "relational"].copy()
cp["month"] = cp["jmonth"].astype(str)
cp = cp.rename(columns={"n_cases": "n"})
gj = prep(cp[["prefecture_code", "province", "cause", "month", "n"]])
pois("A_PPML_judgment_month", gj)

# A2: filing-month cells
cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["case_no", "cause", "cause_family", "prefecture_code",
                              "province", "jmonth"])
rel = cc[cc["cause_family"] == "relational"].copy()
fil = pd.read_parquet(f"{DATA}/civil_filing.parquet").rename(columns={"案号": "case_no"})
rel = rel.merge(fil[["case_no", "filing_ymd"]], on="case_no", how="left")
rel["fdate"] = pd.to_datetime(rel["filing_ymd"], errors="coerce")
rel["dur"] = (pd.to_datetime(rel["jmonth"]) - rel["fdate"]).dt.days
ok = rel["fdate"].notna() & rel["dur"].between(0, 270)
rel = rel[ok]
rel["month"] = rel["fdate"].dt.strftime("%Y-%m")
gf = (rel.groupby(["prefecture_code", "province", "cause", "month"]).size()
      .rename("n").reset_index())
gf = prep(gf)
pois("A_PPML_filing_month", gf)

# A3: PPML on the 250 km DLR pair stack
cen = pd.read_csv(f"{DATA}/pref_centroids.csv", dtype={"prefecture_code": str})
pref = gj[["prefecture_code", "treat"]].drop_duplicates()
pref["pcode2"] = pref["prefecture_code"].astype(str).str[:2]
pref = pref.merge(cen, on="prefecture_code").dropna(subset=["lat"])
def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return 2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2
           + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
T = pref[pref.treat == 1].reset_index(drop=True)
C = pref[pref.treat == 0].reset_index(drop=True)
DM = hav(np.array(T.lat)[:, None], np.array(T.lon)[:, None],
         np.array(C.lat)[None, :], np.array(C.lon)[None, :])
DM = np.where(np.array(T.pcode2)[:, None] != np.array(C.pcode2)[None, :], DM, np.inf)
pairs = set()
tn, cn = DM.min(1), DM.min(0)
for i in np.where(tn <= 250)[0]:
    pairs.add((T.loc[i, "prefecture_code"], C.loc[np.argmin(DM[i]), "prefecture_code"]))
for j in np.where(cn <= 250)[0]:
    pairs.add((T.loc[np.argmin(DM[:, j]), "prefecture_code"], C.loc[j, "prefecture_code"]))
stack = []
for k, (a, b) in enumerate(sorted(pairs)):
    seg = gj[gj["prefecture_code"].isin([a, b])].copy()
    seg["pair_month"] = f"{k}_" + seg["month"]
    stack.append(seg)
st = pd.concat(stack, ignore_index=True)
pois("A_PPML_border_pairs_250km", st, "n ~ pth + ph + pt | pref_cause + pair_month")

# B1: border-sample calendar leads (asinh spec, 250 km restricted dose sample)
bset = set(T.loc[tn <= 250, "prefecture_code"]) | set(C.loc[cn <= 250, "prefecture_code"])
bs = gj[gj["prefecture_code"].isin(bset)].copy()
bs["asinh_n"] = np.arcsinh(bs["n"])
et = (pd.PeriodIndex(bs["month"], freq="M") - pd.Period(POST0, freq="M")).map(lambda x: x.n)
terms = []
for lo, hi in [(-20, -13), (-12, -7), (0, 6)]:
    nm = f"b{lo}_{hi}H".replace("-", "m")
    inb = ((et >= lo) & (et <= hi)).astype(float)
    bs[nm] = inb * bs["treat"] * bs["exposure_v2_z"]
    bs[nm + "T"] = inb * bs["treat"]
    bs[nm + "X"] = inb * bs["exposure_v2_z"]
    terms += [nm, nm + "T", nm + "X"]
mb = pf.feols(f"asinh_n ~ {' + '.join(terms)} | pref_cause + month", data=bs,
              vcov={"CRV1": "prov_id"})
for lo, hi in [(-20, -13), (-12, -7), (0, 6)]:
    nm = f"b{lo}_{hi}H".replace("-", "m")
    print(f"B_border_ES [{lo:+d},{hi:+d}]: {mb.coef()[nm]: .4f} ({mb.se()[nm]:.4f})")
    rows.append(dict(part=f"B_borderES_{lo}_{hi}", est=mb.coef()[nm],
                     se=mb.se()[nm], p=mb.pvalue()[nm], n=int(mb._N)))

# B2: covariate balance across the line (border sample)
con = duckdb.connect()
cov = con.sql(f"""
  SELECT prefecture_code, AVG(pku_dfi_total_2017) dfi2017,
         AVG(bank_branch_total_count) bank
  FROM '{SRC}' WHERE prefecture_code IS NOT NULL GROUP BY 1""").df()
bb = pref[pref["prefecture_code"].isin(bset)].merge(cov, on="prefecture_code", how="left")
for v in ("dfi2017", "bank"):
    a = bb.loc[bb.treat == 1, v].dropna(); b = bb.loc[bb.treat == 0, v].dropna()
    nd = (a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()))
    t, p = sps.ttest_ind(a, b, equal_var=False)
    print(f"B_balance {v}: T={a.mean():.1f} C={b.mean():.1f} normdiff={nd:.3f} p={p:.3f}")
    rows.append(dict(part=f"B_balance_{v}", est=nd, se=np.nan, p=p, n=len(bb)))

# C: wave assignment
prov = (gj.groupby("province").agg(H_mean=("exposure_v2_z", "mean"),
                                   treat=("treat", "max")).reset_index())
pre17 = (gj[gj["month"] <= "2017-12"].groupby("province")["n"].sum()
         .rename("pre_cases").reset_index())
cov_p = (gj[["prefecture_code", "province"]].drop_duplicates()
         .merge(cov, on="prefecture_code", how="left")
         .groupby("province")[["dfi2017", "bank"]].mean().reset_index())
prov = prov.merge(pre17, on="province").merge(cov_p, on="province")
prov["ln_pre"] = np.log(prov["pre_cases"])
X = prov[["H_mean", "ln_pre", "dfi2017", "bank"]].apply(lambda s: (s - s.mean()) / s.std())
X.insert(0, "const", 1.0)
y = prov["treat"].to_numpy(float)
b, res, *_ = np.linalg.lstsq(X.to_numpy(float), y, rcond=None)
fit = X.to_numpy(float) @ b
r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"C_assignment: R2={r2:.3f}; coefs " +
      ", ".join(f"{c}={v:.3f}" for c, v in zip(X.columns, b)))
rows.append(dict(part="C_assignment_R2", est=r2, se=np.nan, p=np.nan, n=len(prov)))
# does the fitted propensity predict PRE-period civil growth (2017H2 vs 2017H1)?
g17 = gj[gj["month"] <= "2017-12"].copy()
g17["h2"] = (g17["month"] >= "2017-07").astype(int)
gg = (g17.groupby(["province", "h2"])["n"].sum().unstack()
      .assign(growth=lambda d: np.log(d[1]) - np.log(d[0])).reset_index())
gg = gg.merge(prov[["province"]].assign(prop=fit), on="province")
sl, icpt, r, p, se = sps.linregress(gg["prop"], gg["growth"])
print(f"C_propensity_vs_pregrowth: slope={sl:.3f} (p={p:.3f})")
rows.append(dict(part="C_prop_vs_pregrowth", est=sl, se=se, p=p, n=len(gg)))

pd.DataFrame(rows).to_csv(f"{OUT}/ppml_border_assign.csv", index=False)
print("written:", f"{OUT}/ppml_border_assign.csv")

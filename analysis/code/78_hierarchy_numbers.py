# -*- coding: utf-8 -*-
"""6B step 78 — two pre-specified summary statistics for the evidence hierarchy.

P1  Composite de-militarization index (Kling-Liebman-Katz): prefecture x month,
    equal-weight mean of three pre-period-standardized components, all signed so
    that de-militarization is NEGATIVE: asinh enforcement caseload,
    detention-for-debt share (enforcement docket), audited hard-backstop share
    (market docket). Dose spec px | pref + prov_month; CRV1 + wild + wave-timing
    randomization inference (999 draws, reassign province inspection months).

P2  Stake-gradient shape contrast (pre-specified by Prop. 3: the response
    concentrates in the marginal band): mean(20-50k, 50-200k) minus
    mean(<20k, 200k-1m, >1m) from ONE stacked five-slope regression
    (pref x band and prov-month x band FE), delta-method CRV1 t plus the same
    wave-timing RI on the contrast.

Output: output/hierarchy_numbers.csv + log prints. NO tex writes.
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
import os, sys, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as sps
from _wild import wild_p

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
rows = []

# ============================================================================
# P1 — composite de-militarization index
# ============================================================================
print("== P1: composite de-militarization index ==", flush=True)
kp = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")
kp = kp[kp["n_cases"] > 0].copy()
kp["month"] = kp["jmonth"].astype(str).str[:7]
kp["insp"] = kp["insp_month"].astype(str).str[:7]

en = kp[kp["family"] == "enforcementcrime"]
mk = kp[kp["family"] == "market"]
base = (en.groupby(["prefecture_code", "province", "month", "insp"])
        .agg(n_enf=("n_cases", "sum"), det=("y_detention_debt", "mean"))
        .reset_index())
base["asinh_enf"] = np.arcsinh(base["n_enf"])
mkc = (mk.groupby(["prefecture_code", "month"])
       .agg(back=("y_backstop", "mean")).reset_index())
d = base.merge(mkc, on=["prefecture_code", "month"], how="left")
d = d.merge(kp[["prefecture_code", "exposure_v2_z"]].drop_duplicates(
    "prefecture_code"), on="prefecture_code").dropna(subset=["exposure_v2_z"])
d["post"] = (d["month"] >= d["insp"]).astype(int)

comps = ["asinh_enf", "det", "back"]
pre = d[d["post"] == 0]
for c in comps:  # pre-period standardization, all signed down-is-demilitarized
    mu, sd = pre[c].mean(), pre[c].std()
    d[f"z_{c}"] = (d[c] - mu) / sd
d["index"] = d[[f"z_{c}" for c in comps]].mean(axis=1, skipna=True)
d["ncomp"] = d[[f"z_{c}" for c in comps]].notna().sum(axis=1)
print(f"cells {len(d):,}; components per cell: "
      f"{d['ncomp'].value_counts().to_dict()}", flush=True)

d["px"] = d["post"] * d["exposure_v2_z"]
d["pref"] = d["prefecture_code"]
d["prov_month"] = d["province"] + "_" + d["month"]
d["prov_id"] = pd.factorize(d["province"])[0]
FML = "index ~ px | pref + prov_month"
m = pf.feols(FML, data=d, vcov={"CRV1": "prov_id"})
wp = wild_p(FML, d, "px")
print(f"P1 index: {m.coef()['px']:+.4f} ({m.se()['px']:.4f}) "
      f"CRV1 p={m.pvalue()['px']:.4f} wild={wp:.3f} N={int(m._N):,}", flush=True)
for c in comps:
    mc = pf.feols(f"z_{c} ~ px | pref + prov_month", data=d.dropna(subset=[f"z_{c}"]),
                  vcov={"CRV1": "prov_id"})
    print(f"   component {c:10s}: {mc.coef()['px']:+.4f} ({mc.se()['px']:.4f})",
          flush=True)

# wave-timing RI: reassign the 31 provinces' inspection months, refit
prov_insp = d[["province", "insp"]].drop_duplicates().reset_index(drop=True)
rng = np.random.default_rng(42)
b_obs = m.coef()["px"]; hits = 0; R = 999
provs = prov_insp["province"].values; insps = prov_insp["insp"].values
for r in range(R):
    perm = rng.permutation(insps)
    imap = dict(zip(provs, perm))
    dp = d.assign(insp2=d["province"].map(imap))
    dp["px"] = (dp["month"] >= dp["insp2"]).astype(int) * dp["exposure_v2_z"]
    b = pf.feols(FML, data=dp, vcov="iid").coef()["px"]
    if abs(b) >= abs(b_obs): hits += 1
    if (r + 1) % 100 == 0: print(f"   RI {r+1}/{R}", flush=True)
ri_p = (1 + hits) / (1 + R)
print(f"P1 index RI p = {ri_p:.3f}", flush=True)
rows.append(dict(part="P1", tag="demil_index", est=b_obs, se=m.se()["px"],
                 p=m.pvalue()["px"], wild_p=wp, ri_p=ri_p, n=int(m._N)))

# ============================================================================
# P2 — stake-gradient shape contrast (mid vs ends)
# ============================================================================
print("== P2: stake shape contrast ==", flush=True)
cc = pd.read_parquet(f"{DATA}/civil_case.parquet",
                     columns=["cause", "prefecture_code", "province", "jmonth",
                              "amount_yuan", "post", "insp_month"])
ld = cc[(cc["cause"] == "民间借贷纠纷") & cc["amount_yuan"].notna()
        & (cc["amount_yuan"] > 0)].copy()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[
    ["prefecture_code", "exposure_v2_z"]]
BANDS = [(0, 2e4, "q1"), (2e4, 5e4, "q2"), (5e4, 2e5, "q3"),
         (2e5, 1e6, "q4"), (1e6, np.inf, "q5")]
ld["band"] = None
for lo, hi, nm in BANDS:
    ld.loc[(ld["amount_yuan"] > lo) & (ld["amount_yuan"] <= hi), "band"] = nm
ld["month"] = ld["jmonth"].astype(str).str[:7]
g = (ld.groupby(["prefecture_code", "province", "month", "band"]).size()
     .rename("n").reset_index()
     .merge(ex, on="prefecture_code").dropna(subset=["exposure_v2_z"]))
insp = cc[["province", "insp_month"]].drop_duplicates()
insp["insp"] = insp["insp_month"].astype(str).str[:7]
g = g.merge(insp[["province", "insp"]].drop_duplicates(), on="province")
g["post"] = (g["month"] >= g["insp"]).astype(int)
g["y"] = np.arcsinh(g["n"])
g["px"] = g["post"] * g["exposure_v2_z"]
for _, _, nm in BANDS:
    g[f"px_{nm}"] = g["px"] * (g["band"] == nm)
g["pref_band"] = g["prefecture_code"] + "_" + g["band"]
g["pm_band"] = g["province"] + "_" + g["month"] + "_" + g["band"]
g["prov_id"] = pd.factorize(g["province"])[0]
FML2 = ("y ~ px_q1 + px_q2 + px_q3 + px_q4 + px_q5 | pref_band + pm_band")
m2 = pf.feols(FML2, data=g, vcov={"CRV1": "prov_id"})
names = list(m2.coef().index)
W = np.zeros(len(names))
for nm, w in (("px_q2", .5), ("px_q3", .5), ("px_q1", -1/3),
              ("px_q4", -1/3), ("px_q5", -1/3)):
    W[names.index(nm)] = w
L = float(W @ m2.coef().values)
seL = float(np.sqrt(W @ m2._vcov @ W))
tL = L / seL
pL = float(2 * (1 - sps.t.cdf(abs(tL), 30)))
print(f"P2 per-band: " + "  ".join(
    f"{nm}={m2.coef()[f'px_{nm}']:+.4f}" for _, _, nm in BANDS), flush=True)
print(f"P2 mid-minus-ends contrast: {L:+.4f} (se {seL:.4f}) "
      f"t={tL:.2f} p(t30)={pL:.3f} N={int(m2._N):,}", flush=True)

hits2 = 0
for r in range(R):
    perm = rng.permutation(insps)
    imap = dict(zip(provs, perm))
    gp = g.assign(insp2=g["province"].map(imap))
    gp["post2"] = (gp["month"] >= gp["insp2"]).astype(int)
    for _, _, nm in BANDS:
        gp[f"px_{nm}"] = gp["post2"] * gp["exposure_v2_z"] * (gp["band"] == nm)
    mb = pf.feols(FML2, data=gp, vcov="iid")
    Lb = float(W @ mb.coef().values)
    if abs(Lb) >= abs(L): hits2 += 1
    if (r + 1) % 100 == 0: print(f"   RI {r+1}/{R}", flush=True)
ri2 = (1 + hits2) / (1 + R)
print(f"P2 contrast RI p = {ri2:.3f}", flush=True)
rows.append(dict(part="P2", tag="stake_mid_vs_ends", est=L, se=seL, p=pL,
                 wild_p=np.nan, ri_p=ri2, n=int(m2._N)))

pd.DataFrame(rows).to_csv(f"{OUTD}/hierarchy_numbers.csv", index=False)
print("step 78 complete", flush=True)

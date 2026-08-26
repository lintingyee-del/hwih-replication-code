# -*- coding: utf-8 -*-
"""6B step 26 — true prefecture-border design (web-sourced centroids).

Upgrades step 25's province-contiguity check to a prefecture-level border
discontinuity, using prefecture centroids (data/pref_centroids.csv, built from a
public China administrative-division coordinate list). Two specs:

  A  Border-restricted dose: clean-window civil flow on prefectures whose nearest
     differently-timed prefecture ACROSS a province line is within D km.
  B  Border pairs (Dube-Lester-Reich): match each first-wave border prefecture to
     its nearest not-yet-treated prefecture across the line and vice versa, stack
     the pairs, and absorb pair x month fixed effects, so each treated prefecture
     is compared to its neighbour on the other side of the same boundary.

Supersedes step 25's 2-row tab_border.tex with a comprehensive identification
table (full / contiguous provinces / prefecture-border / border pairs).
Outputs: output/tables/tab_border.tex, numbers_pborder.tex.
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

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
WINDOW = ("2017-01", "2019-03"); POST0 = "2018-09"
D_KM = 200.0
rows = {}

from _wild import wild_score_p, wild_p  # corrected shared WCR bootstrap


def fit(tag, fml, df, coef="pth", cluster="prov_id"):
    m = pf.feols(fml, data=df, vcov={"CRV1": cluster})
    try: wp = wild_score_p(fml, df, coef, cluster=cluster)
    except Exception: wp = np.nan
    rows[tag] = dict(est=m.coef()[coef], se=m.se()[coef], p=m.pvalue()[coef],
                     wild_p=wp, n=int(m._N))
    print(f"{tag:30s} {m.coef()[coef]: .4f} ({m.se()[coef]:.4f}) "
          f"p={m.pvalue()[coef]:.3f} wild={wp:.3f} N={m._N}", flush=True)

def haversine(la1, lo1, la2, lo2):
    r = np.pi/180; R = 6371.0
    dla = (la2-la1)*r; dlo = (lo2-lo1)*r
    a = np.sin(dla/2)**2 + np.cos(la1*r)*np.cos(la2*r)*np.sin(dlo/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# GB province-code land adjacency (for the contiguous row)
ADJ = {"11":"12,13","12":"11,13","13":"11,12,14,15,21,37,41","14":"13,15,41,61",
 "15":"13,14,21,22,23,61,62,64","21":"13,15,22","22":"15,21,23","23":"15,22",
 "31":"32,33","32":"31,33,34,37","33":"31,32,34,35,36","34":"32,33,36,37,41,42",
 "35":"33,36,44","36":"33,34,35,42,43,44","37":"13,32,34,41",
 "41":"13,14,34,37,42,61","42":"34,36,41,43,50,61","43":"36,42,44,45,50,52",
 "44":"35,36,43,45","45":"43,44,52,53","46":"","50":"42,43,51,52,61",
 "51":"50,52,53,54,61,62,63","52":"43,45,50,51,53","53":"45,51,52,54",
 "54":"51,53,63,65","61":"14,15,41,42,50,51,62,64","62":"15,51,61,63,64,65",
 "63":"51,54,62,65","64":"15,61,62","65":"54,62,63"}
ADJ = {k: set(v.split(",")) - {""} for k, v in ADJ.items()}

# ---------------- clean-window relational panel ------------------------------
cp = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cp["month"] = cp["jmonth"].astype(str).str[:7]
cp = cp[(cp["month"] >= WINDOW[0]) & (cp["month"] <= WINDOW[1])].copy()
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[
    ["province", "inspection_round"]].drop_duplicates()
cp = cp.merge(sched, on="province", how="left").dropna(subset=["exposure_v2_z"])
cp = cp[cp["cause_family"] == "relational"].copy()
cp["pcode2"] = cp["prefecture_code"].astype(str).str[:2]
cp["treat"] = (cp["inspection_round"] == 1).astype(int)
cp["postc"] = (cp["month"] >= POST0).astype(int)
cp["prov_id"] = pd.factorize(cp["province"])[0]
cp["pref_cause"] = cp["prefecture_code"].astype(str) + "_" + cp["cause"]
cp["month_fe"] = cp["month"]
cp["asinh_n"] = np.arcsinh(cp["n_cases"])
cp["H"] = cp["exposure_v2_z"]
cp["pt"] = cp["postc"] * cp["treat"]; cp["pth"] = cp["pt"] * cp["H"]
cp["ph"] = cp["postc"] * cp["H"]

fit("full", "asinh_n ~ pth + ph + pt | pref_cause + month_fe", cp, )

# contiguous provinces (matches step 25)
r_by_code = cp.groupby("pcode2")["treat"].max()
cr = {c: ("T" if r_by_code.get(c, 0) == 1 else "C") for c in r_by_code.index}
def on_border_prov(c):
    o = "C" if cr.get(c) == "T" else "T"
    return any(cr.get(n) == o for n in ADJ.get(c, set()))
bc = {c for c in cr if on_border_prov(c)}
fit("contig", "asinh_n ~ pth + ph + pt | pref_cause + month_fe",
    cp[cp["pcode2"].isin(bc)].copy())

# ---------------- prefecture centroids + cross-province distances -------------
cen = pd.read_csv(f"{DATA}/pref_centroids.csv",
                  dtype={"prefecture_code": str})
pref = cp[["prefecture_code", "pcode2", "treat"]].drop_duplicates("prefecture_code")
pref = pref.merge(cen, on="prefecture_code", how="left").dropna(subset=["lat"])
T = pref[pref["treat"] == 1].reset_index(drop=True)
C = pref[pref["treat"] == 0].reset_index(drop=True)
Tla = np.asarray(T["lat"], float); Tlo = np.asarray(T["lon"], float)
Cla = np.asarray(C["lat"], float); Clo = np.asarray(C["lon"], float)
Tp = np.asarray(T["pcode2"], dtype=object); Cp = np.asarray(C["pcode2"], dtype=object)
# pairwise treated x control distances, different province only
DM = haversine(Tla[:, None], Tlo[:, None], Cla[None, :], Clo[None, :])
diffprov = (Tp[:, None] != Cp[None, :])
DM = np.where(diffprov, DM, np.inf)
t_near = DM.min(axis=1); c_near = DM.min(axis=0)
print(f"nearest cross-province differently-treated distance (km): "
      f"treated p50={np.median(t_near):.0f} p25={np.percentile(t_near,25):.0f}; "
      f"control p50={np.median(c_near):.0f}", flush=True)

# A: border-restricted dose (prefectures within D km of the treatment line)
Tb = set(T.loc[t_near <= D_KM, "prefecture_code"])
Cb = set(C.loc[c_near <= D_KM, "prefecture_code"])
border_pref = Tb | Cb
cpb = cp[cp["prefecture_code"].isin(border_pref)].copy()
print(f"prefecture-border sample (<= {D_KM:.0f} km): {len(border_pref)} prefectures "
      f"({len(Tb)} treated, {len(Cb)} control), N={len(cpb)}", flush=True)
fit("pborder", "asinh_n ~ pth + ph + pt | pref_cause + month_fe", cpb)

# B: Dube-Lester-Reich pairs — each treated border pref -> nearest control across
# the line, and each control -> nearest treated; stack with pair x month FE
pairs = []
ti = np.argmin(DM, axis=1)
for i in range(len(T)):
    if t_near[i] <= D_KM:
        pairs.append((T.loc[i, "prefecture_code"], C.loc[ti[i], "prefecture_code"]))
ci = np.argmin(DM, axis=0)
for j in range(len(C)):
    if c_near[j] <= D_KM:
        pairs.append((T.loc[ci[j], "prefecture_code"], C.loc[j, "prefecture_code"]))
pairs = sorted(set(pairs))
print(f"border pairs (DLR): {len(pairs)}", flush=True)
stack = []
for k, (a, b) in enumerate(pairs):
    seg = cp[cp["prefecture_code"].isin([a, b])].copy()
    seg["pair"] = k
    stack.append(seg)
st = pd.concat(stack, ignore_index=True)
st["pair_month"] = st["pair"].astype(str) + "_" + st["month"]
st["pair_pref_cause"] = st["pair"].astype(str) + "_" + st["pref_cause"]
fit("dlr", "asinh_n ~ pth + ph + pt | pair_pref_cause + pair_month", st)

# ---------------- exports ----------------------------------------------------
def st_(t):
    p = rows[t]["p"]; return "***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""
def R(t, k="est"): return rows[t][k]

with open(f"{OUTD}/tables/tab_border.tex", "w", encoding="utf-8") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\n"
             "Sample & Coefficient & (SE) & Wild $p$ & $N$ \\\\ \\midrule\n")
    for lab, t in [("Full clean window", "full"),
                   ("Contiguous provinces", "contig"),
                   (f"Prefecture-border ($\\le${D_KM:.0f} km)", "pborder"),
                   ("Border pairs (pair FE)", "dlr")]:
        r = rows[t]
        fh.write(f"{lab} & {r['est']:.4f}{st_(t)} & ({r['se']:.4f}) & "
                 f"{r['wild_p']:.3f} & {r['n']:,} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

M = {
 "ExtPBorderBeta": f"{R('pborder'):.3f}", "ExtPBorderP": f"{R('pborder','p'):.3f}",
 "ExtPBorderWildP": f"{R('pborder','wild_p'):.3f}",
 "ExtPBorderNPref": f"{len(border_pref)}", "ExtPBorderKm": f"{D_KM:.0f}",
 "ExtDLRBeta": f"{R('dlr'):.3f}", "ExtDLRP": f"{R('dlr','p'):.3f}",
 "ExtDLRWildP": f"{R('dlr','wild_p'):.3f}", "ExtDLRNPairs": f"{len(pairs)}",
}
with open(f"{OUTD}/tables/numbers_pborder.tex", "w", encoding="utf-8") as fh:
    fh.write("% prefecture-border macros (6B step 26).\n")
    for k, v in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("step 26 complete: tab_border.tex (4 rows), numbers_pborder.tex", flush=True)

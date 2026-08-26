# -*- coding: utf-8 -*-
"""6B step 63 — border bandwidth sensitivity + exposure map.
A. Re-estimate the prefecture-border dose and the DLR pair specs at cross-line
   distance bandwidths 100/150/200/250 km.
B. Dot map of prefecture exposure H (grayscale, wave-1 vs later markers).
Outputs: output/ext2124/border_bandwidth.csv, figures/fig_map_exposure.pdf
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
import pandas as pd, numpy as np, pyfixest as pf, sys, io, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _wild import wild_p

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DATA = str(_REP_PROJECT / "data")
OUT = str(_REP_PROJECT / "output" / "ext2124")
FIGD = str(_REP_PROJECT / 'output' / 'figures').replace('\\', '/')
SUB = str(_REP_PACKAGE / "manuscript" / "figures")
WINDOW = ("2017-01", "2019-03"); POST0 = "2018-09"

cp = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cp["month"] = cp["jmonth"].astype(str)
cp = cp[(cp["month"] >= WINDOW[0]) & (cp["month"] <= WINDOW[1])]
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]].drop_duplicates()
cp = cp.merge(sched, on="province", how="left").dropna(subset=["exposure_v2_z"])
cp = cp[cp["cause_family"] == "relational"].copy()
cp["pcode2"] = cp["prefecture_code"].astype(str).str[:2]
cp["treat"] = (cp["inspection_round"] == 1).astype(int)
cp["postc"] = (cp["month"] >= POST0).astype(int)
cp["prov_id"] = pd.factorize(cp["province"])[0]
cp["pref_cause"] = cp["prefecture_code"].astype(str) + "_" + cp["cause"]
cp["asinh_n"] = np.arcsinh(cp["n_cases"])
cp["pt"] = cp["postc"] * cp["treat"]
cp["pth"] = cp["pt"] * cp["exposure_v2_z"]
cp["ph"] = cp["postc"] * cp["exposure_v2_z"]

cen = pd.read_csv(f"{DATA}/pref_centroids.csv", dtype={"prefecture_code": str})
pref = cp[["prefecture_code", "pcode2", "treat"]].drop_duplicates("prefecture_code")
pref = pref.merge(cen, on="prefecture_code", how="left").dropna(subset=["lat"])

def haversine(la1, lo1, la2, lo2):
    r = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

T = pref[pref.treat == 1].reset_index(drop=True)
C = pref[pref.treat == 0].reset_index(drop=True)
DM = haversine(np.asarray(T.lat, float)[:, None], np.asarray(T.lon, float)[:, None],
               np.asarray(C.lat, float)[None, :], np.asarray(C.lon, float)[None, :])
DM = np.where(np.asarray(T.pcode2, object)[:, None] != np.asarray(C.pcode2, object)[None, :],
              DM, np.inf)
t_near, c_near = DM.min(axis=1), DM.min(axis=0)

rows = []
for D_KM in (100, 150, 200, 250):
    Tb = set(T.loc[t_near <= D_KM, "prefecture_code"])
    Cb = set(C.loc[c_near <= D_KM, "prefecture_code"])
    sub = cp[cp["prefecture_code"].isin(Tb | Cb)]
    fml = "asinh_n ~ pth + ph + pt | pref_cause + month"
    sub = sub.rename(columns={"month": "month"})
    m = pf.feols(fml.replace("month", "month"), data=sub.assign(month=sub["month"]),
                 vcov={"CRV1": "prov_id"})
    wp = wild_p(fml, sub, "pth")
    print(f"pborder <= {D_KM}km: {len(Tb)}T/{len(Cb)}C  pth={m.coef()['pth']:.4f} "
          f"(se {m.se()['pth']:.4f}) p={m.pvalue()['pth']:.4f} wild={wp:.3f} N={int(m._N):,}")
    rows.append(dict(band_km=D_KM, spec="pborder", est=m.coef()["pth"],
                     se=m.se()["pth"], p=m.pvalue()["pth"], p_wild=wp,
                     nT=len(Tb), nC=len(Cb), n=int(m._N)))
    # DLR nearest pairs at this bandwidth
    pairs = set()
    ti = np.argmin(DM, axis=1)
    for i in range(len(T)):
        if t_near[i] <= D_KM:
            pairs.add((T.loc[i, "prefecture_code"], C.loc[ti[i], "prefecture_code"]))
    ci = np.argmin(DM, axis=0)
    for j in range(len(C)):
        if c_near[j] <= D_KM:
            pairs.add((T.loc[ci[j], "prefecture_code"], C.loc[j, "prefecture_code"]))
    stack = []
    for k, (a, b) in enumerate(sorted(pairs)):
        seg = cp[cp["prefecture_code"].isin([a, b])].copy()
        seg["pair_month"] = f"{k}_" + seg["month"]
        stack.append(seg)
    st = pd.concat(stack, ignore_index=True)
    fml2 = "asinh_n ~ pth + ph + pt | pref_cause + pair_month"
    m2 = pf.feols(fml2, data=st, vcov={"CRV1": "prov_id"})
    wp2 = wild_p(fml2, st, "pth")
    print(f"  DLR pairs ({len(pairs)}): pth={m2.coef()['pth']:.4f} "
          f"(se {m2.se()['pth']:.4f}) p={m2.pvalue()['pth']:.4f} wild={wp2:.3f}")
    rows.append(dict(band_km=D_KM, spec="dlr_pairs", est=m2.coef()["pth"],
                     se=m2.se()["pth"], p=m2.pvalue()["pth"], p_wild=wp2,
                     nT=len(pairs), nC=np.nan, n=int(m2._N)))
pd.DataFrame(rows).to_csv(f"{OUT}/border_bandwidth.csv", index=False)

# ---------------- map ----------------
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code", "exposure_v2_z"]]
mp = pref.merge(ex, on="prefecture_code", how="left").dropna(subset=["exposure_v2_z"])
plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.2, 4.6))
for tr, mk, lab in [(1, "o", "Wave-1 province"), (0, "^", "Waves 2--3")]:
    d = mp[mp.treat == tr]
    sc = ax.scatter(d.lon, d.lat, c=d.exposure_v2_z, cmap="Greys", vmin=-1.5, vmax=2.5,
                    s=22, marker=mk, edgecolors="0.25", linewidths=0.4, label=lab)
cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("Exposure $H_c$ (SD)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.legend(frameon=False, fontsize=8, loc="lower left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(f"{FIGD}/fig_map_exposure.pdf", bbox_inches="tight", pad_inches=0.03)
shutil.copy(f"{FIGD}/fig_map_exposure.pdf", f"{SUB}/fig_map_exposure.pdf")
print("map written")

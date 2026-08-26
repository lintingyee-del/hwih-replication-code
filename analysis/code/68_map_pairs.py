# -*- coding: utf-8 -*-
"""6B step 68 — regenerate the exposure map with 250 km DLR pair segments."""

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
import pandas as pd, numpy as np, sys, io, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DATA = str(_REP_PROJECT / "data")
FIGD = str(_REP_PROJECT / 'output' / 'figures').replace('\\', '/')
SUB = str(_REP_PACKAGE / "manuscript" / "figures")

cp = pd.read_parquet(f"{DATA}/civil_panel.parquet")[["prefecture_code", "province"]].drop_duplicates()
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[["province", "inspection_round"]].drop_duplicates()
ex = pd.read_parquet(f"{DATA}/exposure_v2.parquet")[["prefecture_code", "exposure_v2_z"]]
cen = pd.read_csv(f"{DATA}/pref_centroids.csv", dtype={"prefecture_code": str})
p = (cp.merge(sched, on="province").merge(ex, on="prefecture_code")
     .merge(cen, on="prefecture_code").dropna(subset=["lat", "exposure_v2_z"]))
p["treat"] = (p["inspection_round"] == 1).astype(int)
p["pcode2"] = p["prefecture_code"].astype(str).str[:2]

def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return 2 * 6371 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2
           + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))

T = p[p.treat == 1].reset_index(drop=True)
C = p[p.treat == 0].reset_index(drop=True)
DM = hav(np.array(T.lat)[:, None], np.array(T.lon)[:, None],
         np.array(C.lat)[None, :], np.array(C.lon)[None, :])
DM = np.where(np.array(T.pcode2)[:, None] != np.array(C.pcode2)[None, :], DM, np.inf)
pairs = set()
tn, cn = DM.min(1), DM.min(0)
for i in np.where(tn <= 250)[0]:
    pairs.add((i, int(np.argmin(DM[i]))))
for j in np.where(cn <= 250)[0]:
    pairs.add((int(np.argmin(DM[:, j])), j))

plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.2, 4.6))
for i, j in sorted(pairs):
    ax.plot([T.loc[i, "lon"], C.loc[j, "lon"]], [T.loc[i, "lat"], C.loc[j, "lat"]],
            color="0.75", lw=0.5, zorder=1)
for tr, mk, lab in [(1, "o", "Wave-1 province"), (0, "^", "Waves 2--3")]:
    d = p[p.treat == tr]
    sc = ax.scatter(d.lon, d.lat, c=d.exposure_v2_z, cmap="Greys", vmin=-1.5,
                    vmax=2.5, s=22, marker=mk, edgecolors="0.25", linewidths=0.4,
                    label=lab, zorder=2)
cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("Exposure $H_c$ (SD)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.legend(frameon=False, fontsize=8, loc="lower left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(f"{FIGD}/fig_map_exposure.pdf", bbox_inches="tight", pad_inches=0.03)
shutil.copy(f"{FIGD}/fig_map_exposure.pdf", f"{SUB}/fig_map_exposure.pdf")
print(f"map with {len(pairs)} pair segments written")

# -*- coding: utf-8 -*-
"""6B step 24 — replace within-province exposure permutation with wave-timing
(Fisherian) randomization inference, aligned with the design's identifying
variation (treatment timing), not the exposure characteristic.

  Civil clean-window: permute first-wave TREATMENT status across the 31
    provinces (hold the treated count fixed at its actual value), recompute
    Post x Treat and Post x Treat x H, refit; 999 draws.

Only the civil clean-window headline gets a wave-timing permutation. The
enforcement caseload count is a supporting margin (its identification runs
through the backstop content decline and charge-substitution robustness, not raw
counts), and mean reversion in it is already ruled out by the split-half check,
so no permutation is reported for it. Patches only RefPermCiv in numbers_ref.tex
and the civil permutation footer row in tab_meanrev.tex; the enforcement footer
row written by step 23 is removed. All other cached results are left untouched.
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
import re
import numpy as np
import pandas as pd
import pyfixest as pf

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")
WINDOW = ("2017-01", "2019-03"); POST0 = "2018-09"

# ---------------- civil clean-window: permute first-wave treatment -----------
c = pd.read_parquet(f"{DATA}/civil_panel.parquet")
c = c[c["cause_family"] == "relational"].copy()
c["month"] = c["jmonth"].astype(str).str[:7]
c = c[(c["month"] >= WINDOW[0]) & (c["month"] <= WINDOW[1])]
sched = pd.read_parquet(f"{DATA}/panel_month.parquet")[
    ["province", "inspection_round"]].drop_duplicates()
c = c.merge(sched, on="province", how="left").dropna(subset=["exposure_v2_z"])
c["postc"] = (c["month"] >= POST0).astype(int)
c["pref_cause"] = c["prefecture_code"] + "_" + c["cause"]
c["month_fe"] = c["month"]
c["asinh_n"] = np.arcsinh(c["n_cases"])
c["H"] = c["exposure_v2_z"]
c["treat"] = (c["inspection_round"] == 1).astype(int)

provs_c = c[["province", "treat"]].drop_duplicates().reset_index(drop=True)
n_treat = int(provs_c["treat"].sum())
print(f"civil: {len(provs_c)} provinces, {n_treat} first-wave (treated)", flush=True)

def fit_civ(treat_map):
    d = c.copy()
    tr = d["province"].map(treat_map)
    d["pt"] = d["postc"] * tr
    d["pth"] = d["pt"] * d["H"]
    d["ph"] = d["postc"] * d["H"]
    return pf.feols("asinh_n ~ pth + ph + pt | pref_cause + month_fe",
                    data=d).coef()["pth"]

obs_treat = dict(zip(provs_c["province"], provs_c["treat"]))
b_civ = fit_civ(obs_treat)
rng = np.random.default_rng(42); provlist = provs_c["province"].values
hits = 0
for r in range(999):
    lab = np.zeros(len(provlist), int)
    lab[rng.choice(len(provlist), n_treat, replace=False)] = 1
    if abs(fit_civ(dict(zip(provlist, lab)))) >= abs(b_civ): hits += 1
    if (r + 1) % 250 == 0: print(f"   civil {r+1}/999", flush=True)
PERM_CIV = (1 + hits) / (1 + 999)
print(f"civil wave-timing perm: obs {b_civ:.4f}, p = {PERM_CIV:.3f}", flush=True)

# ---------------- patch numbers_ref.tex and tab_meanrev.tex ------------------
# Update the civil permutation to wave-timing; drop RefPermEnf and the
# enforcement permutation footer row that step 23 emitted.
nref = f"{OUTD}/tables/numbers_ref.tex"
with open(nref, encoding="utf-8") as fh: txt = fh.read()
txt = re.sub(r"\\newcommand\{\\RefPermCiv\}\{[^}]*\}",
             f"\\\\newcommand{{\\\\RefPermCiv}}{{{PERM_CIV:.3f}}}", txt)
txt = re.sub(r"\n\\newcommand\{\\RefPermEnf\}\{[^}]*\}", "", txt)
with open(nref, "w", encoding="utf-8") as fh: fh.write(txt)

tmr = f"{OUTD}/tables/tab_meanrev.tex"
with open(tmr, encoding="utf-8") as fh: txt = fh.read()
txt = re.sub(r"Permutation \$p\$ \(civil flow;[^&]*& \\multicolumn\{4\}\{c\}\{[^}]*\}",
             f"Permutation $p$ (civil flow; 999 wave-timing draws) & "
             f"\\\\multicolumn{{4}}{{c}}{{{PERM_CIV:.3f}}}", txt)
txt = re.sub(r"Permutation \$p\$ \(enforcement caseload[^\\]*\\\\multicolumn\{4\}\{c\}\{[^}]*\} \\\\\n",
             "", txt)
with open(tmr, "w", encoding="utf-8") as fh: fh.write(txt)
print(f"patched RefPermCiv={PERM_CIV:.3f}; dropped enforcement permutation", flush=True)
print("step 24 complete", flush=True)

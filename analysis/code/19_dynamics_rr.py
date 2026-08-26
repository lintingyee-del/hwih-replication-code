# -*- coding: utf-8 -*-
"""6B step 19 — (A) spike-then-decline dynamics for enforcement-crime caseloads;
(B) Rambachan-Roth (2023) relative-magnitudes sensitivity for the three key event
studies, using the transparent conservative implementation of e14 (bias bound =
multiplier * Mbar * max|pre step|; breakdown Mbar* where robust CI first covers 0).

Outputs: output/dynamics_enforce.csv, output/rr_bounds.csv
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
import pandas as pd, numpy as np, pyfixest as pf

DATA = str(_REP_PROJECT / "data")
OUTD = str(_REP_PROJECT / "output")

BINS = [(-24,-19),(-18,-13),(-12,-7),(-6,-1),(0,5),(6,11),(12,17),(18,28)]
REF = (-6,-1)

def es_fit(df, y, dose_col, fes, weights=None):
    """Event-study with exposure-interacted bins; returns coef vec, vcov, names."""
    d = df[(df["event_time"] >= -24) & (df["event_time"] <= 28)].copy()
    terms = []
    for lo, hi in BINS:
        if (lo, hi) == REF: continue
        nm = f"b_{lo}_{hi}".replace("-", "m")
        base = ((d["event_time"] >= lo) & (d["event_time"] <= hi)).astype(float)
        d[nm] = base * (d[dose_col] if dose_col else 1.0)
        terms.append(nm)
    m = pf.feols(f"{y} ~ {' + '.join(terms)} | {fes}", data=d,
                 vcov={"CRV1": "prov_id"}, weights=weights)
    names = list(m.coef().index)
    idx = [names.index(t) for t in terms]
    return m.coef().values[idx], m._vcov[np.ix_(idx, idx)], terms, m

def rm_bounds(beta, V, label, out):
    """Pre bins: (-24,-19),(-18,-13),(-12,-7) -> steps between consecutive pre bins
    and the step into the (omitted) reference. Post: bin0=[0,5], bins01 avg."""
    k = list(range(len(beta)))  # 0,1,2 pre; 3,4,5,6 post (ref omitted)
    steps = []
    for a, b in [(0, 1), (1, 2)]:
        c = np.zeros(len(beta)); c[b] = 1; c[a] = -1; steps.append(c)
    c = np.zeros(len(beta)); c[2] = -1; steps.append(c)  # last pre bin -> ref(0)
    svals = [c @ beta for c in steps]
    B = float(np.max(np.abs(svals)))
    E = np.eye(len(beta))
    for name, cvec, mult in [
        ("post bin [0,5]", E[3], 1.0),
        ("post avg [0,28]", (E[3]+E[4]+E[5]+E[6])/4, 2.5),
        ("post bin [12,17]", E[5], 3.0),
    ]:
        th = float(cvec @ beta); se = float(np.sqrt(cvec @ V @ cvec))
        # breakdown Mbar*: |th| = mult*Mbar*B + 1.96*se
        mstar = (abs(th) - 1.96 * se) / (mult * B) if B > 0 else np.inf
        out.append(dict(design=label, target=name, theta=th, se=se, B=B,
                        breakdown_Mbar=max(mstar, 0.0)))
        print(f"  {label:28s} {name:16s} th={th: .4f} se={se:.4f} "
              f"B={B:.4f} Mbar*={max(mstar,0):.2f}")

rr = []

# ---------- (A) enforcement-crime caseload dynamics ---------------------------
kp = pd.read_parquet(f"{DATA}/crim_panel_v2.parquet")
kp = kp[(kp["family"] == "enforcementcrime") & (kp["n_cases"] > 0)].copy()
kp["month"] = kp["jmonth"].astype(str)
kp["prov_id"] = pd.factorize(kp["province"])[0]
kp["pref"] = kp["prefecture_code"]
kp["prov_month"] = kp["province"] + "_" + kp["month"]
kp["asinh_n"] = np.arcsinh(kp["n_cases"])
kp["one"] = 1.0

# raw profile: calendar-month FE (identified off the three-wave stagger)
b_raw, V_raw, terms, m_raw = es_fit(kp, "asinh_n", None, "pref + month")
dyn = pd.DataFrame({"bin": [t for t in terms], "est_raw": b_raw,
                    "se_raw": np.sqrt(np.diag(V_raw))})
# exposure-interacted profile
kp2 = kp.rename(columns={"exposure_v2_z": "H"})
b_h, V_h, terms_h, m_h = es_fit(kp2, "asinh_n", "H", "pref + prov_month")
dyn["est_dose"] = b_h; dyn["se_dose"] = np.sqrt(np.diag(V_h))
dyn.to_csv(f"{OUTD}/dynamics_enforce.csv", index=False)
print("enforcement dynamics (raw / dose):")
print(dyn.round(4).to_string(index=False))
print("\nRR relative-magnitudes bounds:")
rm_bounds(b_h, V_h, "enforceN x H", rr)

# ---------- (B) RR for civil flow and criminal backstop -----------------------
cp = pd.read_parquet(f"{DATA}/civil_panel.parquet")
cp = cp[cp["cause_family"] == "relational"].copy()
cp["month"] = cp["jmonth"].astype(str)
cp["prov_id"] = pd.factorize(cp["province"])[0]
cp["pref_cause"] = cp["prefecture_code"] + "_" + cp["cause"]
cp["prov_month"] = cp["province"] + "_" + cp["month"]
cp["cause_month"] = cp["cause"] + "_" + cp["month"]
cp["asinh_n"] = np.arcsinh(cp["n_cases"])
cp = cp.rename(columns={"exposure_v2_z": "H"})
b_c, V_c, _, _ = es_fit(cp, "asinh_n", "H", "pref_cause + prov_month + cause_month")
rm_bounds(b_c, V_c, "civil flow x H", rr)

pm = pd.read_parquet(f"{DATA}/panel_month.parquet")
pm = pm[(pm["analysis_group"] == "market") & (pm["n_fact"] > 0)].copy()
pm["month"] = pm["judgment_month"].astype(str)
pm["prov_id"] = pd.factorize(pm["province"])[0]
pm["pref"] = pm["prefecture_code"]
pm["prov_month"] = pm["province"] + "_" + pm["month"]
pm = pm.rename(columns={"exposure_z": "H"})
b_k, V_k, _, _ = es_fit(pm, "y_backstop", "H", "pref + prov_month", weights="n_fact")
rm_bounds(b_k, V_k, "crim backstop x H (v1)", rr)

pd.DataFrame(rr).to_csv(f"{OUTD}/rr_bounds.csv", index=False)
print("saved dynamics_enforce.csv / rr_bounds.csv")

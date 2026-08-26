# -*- coding: utf-8 -*-
"""Frozen specification universe for the audited criminal core.

The admissible specifications are declared in
97_specification_universe_manifest.json before this script is run.  The script
writes every fit, whether or not it rejects, applies BH correction, and keeps
PPML results separate from the full-cell linear specifications.

Outputs (analysis/output/spec_universe/):
  criminal_spec_universe.csv
  criminal_pretrend_tests.csv
  criminal_spec_universe_summary.csv
  criminal_spec_universe_log.txt
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
import os
import sys

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from _wild import wild_score_p

sys.stdout.reconfigure(encoding="utf-8")
BASE = str(_REP_PROJECT)
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output", "spec_universe")
os.makedirs(OUT, exist_ok=True)
ROWS = []
LOG = []


def say(msg):
    print(msg, flush=True)
    LOG.append(str(msg))


def record_common(name, family, outcome, method, timing, start, end,
                  weighted, doc_control, fe, htrend, expected_sign, d,
                  m, coef, wild=np.nan, note=""):
    b = float(m.coef()[coef])
    se = float(m.se()[coef])
    p = float(m.pvalue()[coef])
    n_input = len(d)
    n_fit = int(m._N)
    ROWS.append({
        "spec": name,
        "family": family,
        "outcome": outcome,
        "method": method,
        "timing": timing,
        "start": start,
        "end": end,
        "weighted": weighted,
        "doc_control": doc_control,
        "fixed_effects": fe,
        "htrend": htrend,
        "expected_sign": expected_sign,
        "beta": b,
        "se": se,
        "p_crv1": p,
        "p_wild": wild,
        "n_input": n_input,
        "n_fit": n_fit,
        "fit_share": n_fit / n_input if n_input else np.nan,
        "note": note,
    })
    wp = f"{wild:.4f}" if np.isfinite(wild) else "--"
    say(f"{name:48s} b={b:+.5f} se={se:.5f} p={p:.4f} "
        f"wild={wp} N={n_fit}/{n_input}")


kp = pd.read_parquet(os.path.join(DATA, "crim_panel_v2.parquet"))
kp = kp[kp["n_cases"] > 0].copy()
kp["jmonth"] = pd.to_datetime(kp["jmonth"])
kp["insp_month"] = pd.to_datetime(kp["insp_month"])
kp["month"] = kp["jmonth"].dt.strftime("%Y-%m")
kp["year"] = kp["jmonth"].dt.year
kp["H"] = kp["exposure_v2_z"]
kp["t"] = ((kp["jmonth"].dt.year - 2014) * 12
           + kp["jmonth"].dt.month - 1).astype(float)
kp["Ht"] = kp["H"] * (kp["t"] - kp["t"].mean())
kp["post_staggered"] = (kp["jmonth"] >= kp["insp_month"]).astype(int)
kp["post_national"] = (kp["jmonth"] >= pd.Timestamp("2018-01-01")).astype(int)
kp["pref"] = kp["prefecture_code"].astype(str)
kp["prov_month"] = kp["province"] + "_" + kp["month"]
kp["prov_id"] = pd.factorize(kp["province"])[0]

SHARE_VARIANTS = [
    dict(tag="baseline", start=2014, end=2020, timing="staggered",
         weighted=True, doc=True, fe="strict", trend=False),
    dict(tag="end2019", start=2014, end=2019, timing="staggered",
         weighted=True, doc=True, fe="strict", trend=False),
    dict(tag="start2015", start=2015, end=2020, timing="staggered",
         weighted=True, doc=True, fe="strict", trend=False),
    dict(tag="unweighted", start=2014, end=2020, timing="staggered",
         weighted=False, doc=True, fe="strict", trend=False),
    dict(tag="no_doc", start=2014, end=2020, timing="staggered",
         weighted=True, doc=False, fe="strict", trend=False),
    dict(tag="month_fe", start=2014, end=2020, timing="staggered",
         weighted=True, doc=True, fe="month", trend=False),
    dict(tag="htrend", start=2014, end=2020, timing="staggered",
         weighted=True, doc=True, fe="strict", trend=True),
    dict(tag="national", start=2014, end=2020, timing="national",
         weighted=True, doc=True, fe="strict", trend=False),
    dict(tag="national_htrend", start=2014, end=2020, timing="national",
         weighted=True, doc=True, fe="strict", trend=True),
    dict(tag="end2019_htrend", start=2014, end=2019, timing="staggered",
         weighted=True, doc=True, fe="strict", trend=True),
]


def fit_share(family, outcome, label, expected_sign):
    base = kp[kp["family"] == family].copy()
    for v in SHARE_VARIANTS:
        d = base[(base["year"] >= v["start"]) & (base["year"] <= v["end"])].copy()
        post = "post_staggered" if v["timing"] == "staggered" else "post_national"
        d["X"] = d[post] * d["H"]
        controls = []
        if v["doc"]:
            controls.append("x_doclen")
        if v["trend"]:
            controls.append("Ht")
        rhs = "X" + (" + " + " + ".join(controls) if controls else "")
        fe = "pref + prov_month" if v["fe"] == "strict" else "pref + month"
        fml = f"{outcome} ~ {rhs} | {fe}"
        weights = "n_cases" if v["weighted"] else None
        name = f"{label}__{v['tag']}"
        try:
            m = pf.feols(fml, data=d, weights=weights,
                         vcov={"CRV1": "prov_id"})
            wp = wild_score_p(fml, d, "X", weights=weights,
                              cluster="prov_id", reps=9_999, seed=42)
            record_common(name, label, outcome, "share_ols", v["timing"],
                          v["start"], v["end"], v["weighted"], v["doc"],
                          fe, v["trend"], expected_sign, d, m, "X", wp)
        except Exception as exc:
            say(f"{name}: FAILED {type(exc).__name__}: {exc}")
            ROWS.append({"spec": name, "family": label, "outcome": outcome,
                         "method": "share_ols", "timing": v["timing"],
                         "start": v["start"], "end": v["end"],
                         "weighted": v["weighted"], "doc_control": v["doc"],
                         "fixed_effects": fe, "htrend": v["trend"],
                         "expected_sign": expected_sign,
                         "note": f"FAILED {type(exc).__name__}: {exc}"})


fit_share("market", "y_backstop", "market_backstop", -1)
fit_share("enforcementcrime", "y_detention_debt", "detention_debt", -1)


COUNT_VARIANTS = [
    dict(tag="baseline", start=2014, end=2020, timing="staggered", fe="strict", trend=False, method="asinh"),
    dict(tag="end2019", start=2014, end=2019, timing="staggered", fe="strict", trend=False, method="asinh"),
    dict(tag="start2015", start=2015, end=2020, timing="staggered", fe="strict", trend=False, method="asinh"),
    dict(tag="month_fe", start=2014, end=2020, timing="staggered", fe="month", trend=False, method="asinh"),
    dict(tag="htrend", start=2014, end=2020, timing="staggered", fe="strict", trend=True, method="asinh"),
    dict(tag="national", start=2014, end=2020, timing="national", fe="strict", trend=False, method="asinh"),
    dict(tag="national_htrend", start=2014, end=2020, timing="national", fe="strict", trend=True, method="asinh"),
    dict(tag="end2019_htrend", start=2014, end=2019, timing="staggered", fe="strict", trend=True, method="asinh"),
    dict(tag="ppml", start=2014, end=2020, timing="staggered", fe="strict", trend=False, method="ppml"),
    dict(tag="ppml_end2019", start=2014, end=2019, timing="staggered", fe="strict", trend=False, method="ppml"),
    dict(tag="ppml_htrend", start=2014, end=2020, timing="staggered", fe="strict", trend=True, method="ppml"),
    dict(tag="ppml_national", start=2014, end=2020, timing="national", fe="strict", trend=False, method="ppml"),
]


def fit_count():
    base = kp[kp["family"] == "enforcementcrime"].copy()
    for v in COUNT_VARIANTS:
        d = base[(base["year"] >= v["start"]) & (base["year"] <= v["end"])].copy()
        post = "post_staggered" if v["timing"] == "staggered" else "post_national"
        d["X"] = d[post] * d["H"]
        rhs = "X" + (" + Ht" if v["trend"] else "")
        fe = "pref + prov_month" if v["fe"] == "strict" else "pref + month"
        name = f"enforcement_count__{v['tag']}"
        try:
            if v["method"] == "asinh":
                d["y_count"] = np.arcsinh(d["n_cases"])
                fml = f"y_count ~ {rhs} | {fe}"
                m = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"})
                wp = wild_score_p(fml, d, "X", cluster="prov_id",
                                  reps=9_999, seed=42)
                record_common(name, "enforcement_count", "n_cases", "asinh_ols",
                              v["timing"], v["start"], v["end"], False, False,
                              fe, v["trend"], -1, d, m, "X", wp)
            else:
                fml = f"n_cases ~ {rhs} | {fe}"
                m = pf.fepois(fml, data=d, vcov={"CRV1": "prov_id"})
                record_common(name, "enforcement_count", "n_cases", "ppml",
                              v["timing"], v["start"], v["end"], False, False,
                              fe, v["trend"], -1, d, m, "X", np.nan,
                              note="PPML has no wild-score p in this universe")
        except Exception as exc:
            say(f"{name}: FAILED {type(exc).__name__}: {exc}")
            ROWS.append({"spec": name, "family": "enforcement_count",
                         "outcome": "n_cases", "method": v["method"],
                         "timing": v["timing"], "start": v["start"],
                         "end": v["end"], "weighted": False,
                         "doc_control": False, "fixed_effects": fe,
                         "htrend": v["trend"], "expected_sign": -1,
                         "note": f"FAILED {type(exc).__name__}: {exc}"})


fit_count()


DDD_VARIANTS = [
    dict(tag="baseline", start=2014, end=2020, weighted=True, doc=True, trend=False),
    dict(tag="end2019", start=2014, end=2019, weighted=True, doc=True, trend=False),
    dict(tag="start2015", start=2015, end=2020, weighted=True, doc=True, trend=False),
    dict(tag="unweighted", start=2014, end=2020, weighted=False, doc=True, trend=False),
    dict(tag="no_doc", start=2014, end=2020, weighted=True, doc=False, trend=False),
    dict(tag="htrend", start=2014, end=2020, weighted=True, doc=True, trend=True),
    dict(tag="end2019_htrend", start=2014, end=2019, weighted=True, doc=True, trend=True),
]


def fit_market_theft():
    base = kp[kp["family"].isin(["market", "theft"])].copy()
    base["target"] = (base["family"] == "market").astype(int)
    base["pref_group"] = base["pref"] + "_" + base["family"]
    base["group_month"] = base["family"] + "_" + base["month"]
    for v in DDD_VARIANTS:
        d = base[(base["year"] >= v["start"]) & (base["year"] <= v["end"])].copy()
        d["px"] = d["post_staggered"] * d["H"]
        d["X"] = d["px"] * d["target"]
        d["pt"] = d["post_staggered"] * d["target"]
        controls = ["px", "pt"]
        if v["doc"]:
            controls.append("x_doclen")
        if v["trend"]:
            d["Ht_target"] = d["Ht"] * d["target"]
            controls.extend(["Ht", "Ht_target"])
        rhs = "X + " + " + ".join(controls)
        fe = "pref_group + prov_month + group_month"
        fml = f"y_backstop ~ {rhs} | {fe}"
        weights = "n_cases" if v["weighted"] else None
        name = f"market_minus_theft__{v['tag']}"
        try:
            m = pf.feols(fml, data=d, weights=weights,
                         vcov={"CRV1": "prov_id"})
            wp = wild_score_p(fml, d, "X", weights=weights,
                              cluster="prov_id", reps=9_999, seed=42)
            record_common(name, "market_minus_theft", "y_backstop", "ddd_ols",
                          "staggered", v["start"], v["end"], v["weighted"],
                          v["doc"], fe, v["trend"], -1, d, m, "X", wp)
        except Exception as exc:
            say(f"{name}: FAILED {type(exc).__name__}: {exc}")
            ROWS.append({"spec": name, "family": "market_minus_theft",
                         "outcome": "y_backstop", "method": "ddd_ols",
                         "timing": "staggered", "start": v["start"],
                         "end": v["end"], "weighted": v["weighted"],
                         "doc_control": v["doc"], "fixed_effects": fe,
                         "htrend": v["trend"], "expected_sign": -1,
                         "note": f"FAILED {type(exc).__name__}: {exc}"})


fit_market_theft()


# Placebo and neighboring-family diagnostics use the same three declared rows.
DIAG_VARIANTS = [v for v in SHARE_VARIANTS if v["tag"] in
                 ("baseline", "end2019", "htrend")]
saved_variants = SHARE_VARIANTS
SHARE_VARIANTS = DIAG_VARIANTS
fit_share("violence", "y_backstop", "diag_violence_backstop", -1)
fit_share("theft", "y_backstop", "diag_theft_backstop", 0)
SHARE_VARIANTS = saved_variants


def pretrend_test(family, outcome, label, weighted, doc_control):
    d = kp[(kp["family"] == family) & (kp["event_time"] >= -24)
           & (kp["event_time"] <= 28)].copy()
    bins = [(-24, -19), (-18, -13), (-12, -7), (0, 5),
            (6, 11), (12, 17), (18, 28)]
    terms = []
    leads = []
    for lo, hi in bins:
        nm = f"e_{str(lo).replace('-', 'm')}_{str(hi).replace('-', 'm')}"
        d[nm] = ((d["event_time"] >= lo) & (d["event_time"] <= hi)).astype(int) * d["H"]
        terms.append(nm)
        if hi < 0:
            leads.append(nm)
    if outcome == "asinh_n":
        d["pre_y"] = np.arcsinh(d["n_cases"])
    else:
        d["pre_y"] = d[outcome]
    rhs = " + ".join(terms)
    if doc_control:
        rhs += " + x_doclen"
    fml = f"pre_y ~ {rhs} | pref + prov_month"
    weights = "n_cases" if weighted else None
    m = pf.feols(fml, data=d, weights=weights,
                 vcov={"CRV1": "prov_id"})
    names = list(m.coef().index)
    idx = [names.index(x) for x in leads]
    b = m.coef()[leads].values
    V = m._vcov[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.solve(V, b))
    p = float(stats.chi2.sf(stat, len(leads)))
    return {"family": label, "lead_bins": len(leads), "wald": stat,
            "p_crv1": p, "n": int(m._N)}


PRE = []
for args in [
    ("market", "y_backstop", "market_backstop", True, True),
    ("enforcementcrime", "y_detention_debt", "detention_debt", True, True),
    ("enforcementcrime", "asinh_n", "enforcement_count", False, False),
    ("violence", "y_backstop", "diag_violence_backstop", True, True),
    ("theft", "y_backstop", "diag_theft_backstop", True, True),
]:
    try:
        rec = pretrend_test(*args)
        PRE.append(rec)
        say(f"pretrend {rec['family']:30s} p={rec['p_crv1']:.4f} N={rec['n']}")
    except Exception as exc:
        say(f"pretrend {args[2]} FAILED {type(exc).__name__}: {exc}")


res = pd.DataFrame(ROWS)
ok = res["beta"].notna() & res["p_crv1"].notna()
res["p_for_bh"] = np.where(res["p_wild"].notna(), res["p_wild"], res["p_crv1"])
res["q_family"] = np.nan
for fam, idx in res[ok].groupby("family").groups.items():
    res.loc[idx, "q_family"] = multipletests(
        res.loc[idx, "p_for_bh"].astype(float), method="fdr_bh")[1]
res["q_global"] = np.nan
idx = res.index[ok]
res.loc[idx, "q_global"] = multipletests(
    res.loc[idx, "p_for_bh"].astype(float), method="fdr_bh")[1]
res["direction_ok"] = np.where(
    res["expected_sign"] == 0, np.nan,
    np.sign(res["beta"]) == np.sign(res["expected_sign"]))
res["linear_claim_gate"] = (
    res["method"].isin(["share_ols", "asinh_ols", "ddd_ols"])
    & res["direction_ok"].fillna(False)
    & (res["p_wild"] < 0.05)
    & (res["q_family"] < 0.10)
)
res.to_csv(os.path.join(OUT, "criminal_spec_universe.csv"), index=False,
           encoding="utf-8-sig")
pre = pd.DataFrame(PRE)
pre.to_csv(os.path.join(OUT, "criminal_pretrend_tests.csv"), index=False,
           encoding="utf-8-sig")

summ = (res[ok].groupby("family")
        .agg(specs=("spec", "size"), median_beta=("beta", "median"),
             min_beta=("beta", "min"), max_beta=("beta", "max"),
             share_expected_sign=("direction_ok", "mean"),
             p05_rows=("p_for_bh", lambda x: int((x < 0.05).sum())),
             q10_rows=("q_family", lambda x: int((x < 0.10).sum())),
             claim_gate_rows=("linear_claim_gate", "sum"))
        .reset_index())
summ.to_csv(os.path.join(OUT, "criminal_spec_universe_summary.csv"),
            index=False, encoding="utf-8-sig")
say("\n=== family summary ===")
say(summ.to_string(index=False))
say("\n=== rows passing the mechanical claim gate (still require stability review) ===")
gate = res[res["linear_claim_gate"]]
say(gate[["spec", "beta", "se", "p_wild", "q_family"]].to_string(index=False)
    if len(gate) else "none")

with open(os.path.join(OUT, "criminal_spec_universe_log.txt"), "w",
          encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
say(f"outputs -> {OUT}")

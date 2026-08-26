# -*- coding: utf-8 -*-
"""Formal triple difference on the death registry, replacing narrative use.

The paper currently uses the death registry only through Table tab:cdcexact, a
2^6 exact sign test on stratum-level deviations from a projected pre-trend. That
test is inferentially correct here (there are only six region-by-residence
strata, so clustered standard errors are not usable), but it throws away three
things the registry can support:

  1. a single pooled triple-difference estimate,
     post x male x homicide, with the comparator causes used as controls rather
     than reported as separate rows;
  2. a year-by-year profile, so the reader can see when the break occurs;
  3. a magnitude in lives rather than in percent.

Inference is by placebo-in-time randomization: the estimator is refit with every
possible contiguous three-year "post" window inside 2014-2021, and the realised
2018-2020 statistic is ranked against that distribution. This keeps the six
strata intact and makes no asymptotic claim. The 2^6 stratum sign test is
reported alongside for continuity with the existing table.

Data: output/cdc_homicide/cdc_homicide_panel.csv
  8 years x 3 regions x 2 urban/rural x 2 sexes = 96 analysis cells, each with
  homicide, homicide 15-59, suicide and road-traffic counts and implied
  population. National and "both sexes" aggregate rows are dropped.

Outputs (analysis/output/cdc_homicide/)
  cdc_ddd_estimates.csv, cdc_ddd_eventstudy.csv, cdc_ddd_lives.csv,
  cdc_ddd_log.txt
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
import io
import itertools
import os
import sys

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = str(_REP_PROJECT)
OUT = os.path.join(BASE, "output", "cdc_homicide")
PANEL = os.path.join(OUT, "cdc_homicide_panel.csv")
POST_YEARS = (2018, 2019, 2020)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


d = pd.read_csv(PANEL, encoding="utf-8-sig")
d = d[(d["region"] != "全国") & (d["urbrur"] != "城乡合计")
      & (d["sex"] != "合计")].copy()
d["stratum"] = d["region"] + "_" + d["urbrur"]
d["male"] = (d["sex"] == "男性").astype(int)
say(f"analysis cells: {len(d)} = {d['year'].nunique()} years x "
    f"{d['stratum'].nunique()} strata x 2 sexes")

# ---- stack causes ----------------------------------------------------------
CAUSES = [("homicide_15_59_n", "homicide", 1),
          ("suicide_n", "suicide", 0),
          ("traffic_acc_n", "traffic", 0)]
rows = []
for col, name, is_focal in CAUSES:
    t = d[["year", "stratum", "male", "sex", "pop_implied", col]].copy()
    t = t.rename(columns={col: "deaths"})
    t["cause"] = name
    t["focal"] = is_focal
    rows.append(t)
s = pd.concat(rows, ignore_index=True)
s["rate"] = s["deaths"] / s["pop_implied"] * 1e5
s = s[s["deaths"] > 0].copy()
s["y"] = np.log(s["rate"])
s["cell"] = s["stratum"] + "_" + s["sex"] + "_" + s["cause"]
s["stratum_year"] = s["stratum"] + "_" + s["year"].astype(str)
s["sex_year"] = s["sex"] + "_" + s["year"].astype(str)
s["cause_year"] = s["cause"] + "_" + s["year"].astype(str)
s["sex_cause"] = s["sex"] + "_" + s["cause"]
say(f"stacked: {len(s)} cell-cause-year observations, "
    f"{s['cause'].nunique()} causes")


def ddd(frame, post_years, label):
    """Saturated triple difference: post x male x homicide, with all two-way
    interactions absorbed. Homicide uses the 15-59 male-relevant series."""
    f = frame.copy()
    f["post"] = f["year"].isin(post_years).astype(int)
    f["T"] = f["post"] * f["male"] * f["focal"]
    formula = ("y ~ T | cell + stratum_year + sex_year + cause_year "
               "+ sex_cause")
    m = pf.feols(formula, data=f, vcov="hetero")
    return float(m.coef()["T"]), formula, int(m._N)


beta, formula, nobs = ddd(s, POST_YEARS, "realised")
say(f"\n=== pooled triple difference ===")
say(f"  post({POST_YEARS[0]}-{POST_YEARS[-1]}) x male x homicide: "
    f"beta = {beta:+.4f} log points = {100 * (np.exp(beta) - 1):+.1f} percent")
say(f"  N = {nobs}; formula: {formula}")

# ---- placebo-in-time randomization inference -------------------------------
years = sorted(s["year"].unique())
windows = [tuple(years[i:i + 3]) for i in range(len(years) - 2)]
placebo = []
for w in windows:
    b, _, _ = ddd(s, w, str(w))
    placebo.append({"window": f"{w[0]}-{w[-1]}", "beta": b,
                    "realised": w == POST_YEARS})
pl = pd.DataFrame(placebo)
say(f"\n=== placebo-in-time randomization ({len(windows)} contiguous "
    f"three-year windows) ===")
for _, r in pl.iterrows():
    mark = "  <-- realised" if r["realised"] else ""
    say(f"  {r['window']}: {r['beta']:+.4f}{mark}")
rank = int((pl["beta"] <= beta).sum())
p_ri = rank / len(pl)
say(f"  the realised window is rank {rank} of {len(pl)} from the bottom; "
    f"one-sided RI p = {p_ri:.3f}")

# ---- year-by-year profile --------------------------------------------------
f = s.copy()
ref = 2017
terms = []
for yr in years:
    if yr == ref:
        continue
    col = f"T_{yr}"
    f[col] = ((f["year"] == yr) * f["male"] * f["focal"]).astype(float)
    terms.append(col)
formula_es = ("y ~ " + " + ".join(terms)
              + " | cell + stratum_year + sex_year + cause_year + sex_cause")
mes = pf.feols(formula_es, data=f, vcov="hetero")
es = []
for yr in years:
    if yr == ref:
        es.append({"year": yr, "coefficient": 0.0, "pct": 0.0,
                   "reference": True})
        continue
    b_ = float(mes.coef()[f"T_{yr}"])
    es.append({"year": yr, "coefficient": b_,
               "pct": 100 * (np.exp(b_) - 1), "reference": False})
esd = pd.DataFrame(es)
say(f"\n=== year-by-year profile (reference {ref}) ===")
for _, r in esd.iterrows():
    say(f"  {int(r['year'])}: {r['coefficient']:+.4f} "
        f"({r['pct']:+.1f}%){'  [ref]' if r['reference'] else ''}")

# ---- magnitude in lives ----------------------------------------------------
male_hom = d[d["male"].eq(1)][["year", "stratum", "pop_implied",
                               "homicide_15_59_n"]].copy()
post = male_hom[male_hom["year"].isin(POST_YEARS)]
observed = post["homicide_15_59_n"].sum() / len(POST_YEARS)
counterfactual = observed / np.exp(beta)
averted_dsp = counterfactual - observed
dsp_pop = male_hom[male_hom["year"].eq(2019)]["pop_implied"].sum()
say(f"\n=== magnitude in lives ===")
say(f"  DSP male 15-59 population (2019): {dsp_pop:,.0f}")
say(f"  observed male 15-59 homicide deaths, {POST_YEARS[0]}-{POST_YEARS[-1]} "
    f"average: {observed:,.0f} per year")
say(f"  implied counterfactual without the differential: "
    f"{counterfactual:,.0f} per year")
say(f"  averted within the surveillance population: {averted_dsp:,.0f} per year")
for cov, lab in [(0.25, "DSP covers about a quarter of the population")]:
    say(f"  scaled nationally ({lab}): {averted_dsp / cov:,.0f} per year")

# ---- 2^6 stratum sign test, for continuity with tab:cdcexact ---------------
sign_rows = []
for st, g in d[d["male"].eq(1)].groupby("stratum"):
    pre = g[g["year"] < POST_YEARS[0]]
    fem = d[(d["male"].eq(0)) & (d["stratum"].eq(st))]
    gap_pre = (np.log(pre["homicide_15_59_n"] / pre["pop_implied"]).values
               - np.log(fem[fem["year"] < POST_YEARS[0]]["homicide_15_59_n"]
                        / fem[fem["year"] < POST_YEARS[0]]["pop_implied"]).values)
    yr_pre = pre["year"].values
    slope, intercept = np.polyfit(yr_pre, gap_pre, 1)
    po = g[g["year"].isin(POST_YEARS)]
    fpo = fem[fem["year"].isin(POST_YEARS)]
    gap_post = (np.log(po["homicide_15_59_n"] / po["pop_implied"]).values
                - np.log(fpo["homicide_15_59_n"] / fpo["pop_implied"]).values)
    pred = intercept + slope * po["year"].values
    sign_rows.append({"stratum": st,
                      "deviation_pct": 100 * np.mean(gap_post - pred)})
sg = pd.DataFrame(sign_rows)
devs = sg["deviation_pct"].values
stat = devs.mean()
draws = [np.mean(devs * np.array(sgn))
         for sgn in itertools.product([-1, 1], repeat=len(devs))]
p_exact = float(np.mean(np.array(draws) <= stat))
say(f"\n=== 2^6 stratum sign test (continuity check with tab:cdcexact) ===")
say(f"  mean stratum deviation {stat:+.1f}%, "
    f"{int((devs < 0).sum())} of {len(devs)} negative, one-sided p = {p_exact:.3f}")

pd.DataFrame([{"estimate": "post x male x homicide (log points)", "value": beta,
               "pct": 100 * (np.exp(beta) - 1), "n_obs": nobs,
               "ri_p_onesided": p_ri, "ri_windows": len(windows),
               "sign_test_p": p_exact, "sign_test_stat_pct": stat}]
             ).to_csv(os.path.join(OUT, "cdc_ddd_estimates.csv"), index=False,
                      encoding="utf-8-sig")
esd.to_csv(os.path.join(OUT, "cdc_ddd_eventstudy.csv"), index=False,
           encoding="utf-8-sig")
pl.to_csv(os.path.join(OUT, "cdc_ddd_placebo_windows.csv"), index=False,
          encoding="utf-8-sig")
pd.DataFrame([{"dsp_male_pop_2019": dsp_pop, "observed_per_year": observed,
               "counterfactual_per_year": counterfactual,
               "averted_dsp_per_year": averted_dsp,
               "averted_national_per_year_at_quarter_coverage":
                   averted_dsp / 0.25}]
             ).to_csv(os.path.join(OUT, "cdc_ddd_lives.csv"), index=False,
                      encoding="utf-8-sig")
with open(os.path.join(OUT, "cdc_ddd_log.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
say(f"\nDONE -> {OUT}")

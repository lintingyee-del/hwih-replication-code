# -*- coding: utf-8 -*-
"""Validation battery for the pre-COVID Baidu candidate from step 98.

The preferred design is fixed before this script runs: 2014--2019, staggered
inspection clock, prefecture and province-by-month fixed effects, headline H.
Validation consists of keyword decomposition, event-time leads, leave-one-
province-out stability, and 999 permutations of inspection months across the 31
provinces. No result is selected or dropped.
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
import re
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

xw = pd.read_parquet(os.path.join(DATA, "court_xwalk.parquet"))
pat = re.compile(r"(?:.*?省|.*?自治区)?(.+?(?:市|州|盟|地区))中级人民法院$")
names = {}
for _, row in xw.iterrows():
    match = pat.match(str(row["court_name"]))
    if match:
        names.setdefault(match.group(1), []).append(str(row["prefecture_code"]))
name2code = {nm: max(set(v), key=v.count) for nm, v in names.items()}
name2code.update({"北京市": "110100", "天津市": "120100",
                  "上海市": "310100", "重庆市": "500100"})


def city2code(city):
    city = str(city)
    cand = city if city.endswith(("市", "州", "盟", "地区")) else city + "市"
    if cand in name2code:
        return name2code[cand]
    for nm, code in name2code.items():
        if nm[:-1] and nm[:-1] in city:
            return code
    return None


exp = pd.read_parquet(os.path.join(DATA, "exposure_v2.parquet"))
exp["prefecture_code"] = exp["prefecture_code"].astype(str)
cp = pd.read_parquet(os.path.join(DATA, "civil_panel.parquet"),
                     columns=["province", "insp_month"]).drop_duplicates()
insp = cp.groupby("province")["insp_month"].first()
raw = pd.read_csv(str(_REP_PROJECT / "data" / "derived" / "baidu_index_city_month.csv"),
                  dtype={"ym": str})
raw["prefecture_code"] = raw["city"].map(city2code)
raw = raw[raw["prefecture_code"].isin(set(exp["prefecture_code"]))].copy()


def panel(keywords):
    d = raw[raw["keyword"].isin(keywords)].copy()
    d = (d.groupby(["prefecture_code", "ym"], as_index=False)["mean"].sum()
         .merge(exp[["prefecture_code", "province", "exposure_v2_z"]],
                on="prefecture_code", how="inner"))
    d["insp_month"] = pd.to_datetime(d["province"].map(insp))
    d["m"] = pd.to_datetime(d["ym"] + "-01")
    d = d[(d["m"] >= "2014-01-01") & (d["m"] <= "2019-12-01")].copy()
    d["H"] = d["exposure_v2_z"]
    d["post"] = (d["m"] >= d["insp_month"]).astype(int)
    d["X"] = d["post"] * d["H"]
    d["pref"] = d["prefecture_code"]
    d["provm"] = d["province"] + "_" + d["ym"]
    d["prov_id"] = pd.factorize(d["province"])[0]
    d["y"] = np.arcsinh(d["mean"])
    return d


rows = []
keyword_sets = [
    ("two_terms", ["讨债公司", "讨债"]),
    ("collection_company", ["讨债公司"]),
    ("collect_debt", ["讨债"]),
    ("shoushu_company", ["收数公司"]),
    ("all_three", ["讨债公司", "讨债", "收数公司"]),
]
for label, kws in keyword_sets:
    base = panel(kws)
    for transform in ("asinh", "log1p"):
        d = base.copy()
        d["y"] = (np.arcsinh(d["mean"]) if transform == "asinh"
                  else np.log1p(d["mean"]))
        fml = "y ~ X | pref + provm"
        model = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"})
        wild = wild_score_p(fml, d, "X", cluster="prov_id", reps=9_999, seed=42)
        rows.append(dict(keyword_set=label, transform=transform,
                         beta=float(model.coef()["X"]), se=float(model.se()["X"]),
                         p_crv1=float(model.pvalue()["X"]), p_wild=wild,
                         n=int(model._N)))
        print(f"{label:20s} {transform:6s} b={float(model.coef()['X']):+.5f} "
              f"se={float(model.se()['X']):.5f} "
              f"p={float(model.pvalue()['X']):.4f} wild={wild:.4f}", flush=True)

keyword_results = pd.DataFrame(rows)
keyword_results["q_bh"] = multipletests(
    keyword_results["p_wild"], method="fdr_bh")[1]
keyword_results.to_csv(os.path.join(OUT, "baidu_candidate_keywords.csv"),
                       index=False, encoding="utf-8-sig")

# Event study, with [-6,-1] as the omitted bin.
d = panel(["讨债公司", "讨债"])
d["event_time"] = ((d["m"].dt.year - d["insp_month"].dt.year) * 12
                   + d["m"].dt.month - d["insp_month"].dt.month)
d = d[(d["event_time"] >= -24) & (d["event_time"] <= 17)].copy()
bins = [(-24, -19), (-18, -13), (-12, -7), (0, 5), (6, 11), (12, 17)]
terms = []
leads = []
for lo, hi in bins:
    name = f"e_{str(lo).replace('-', 'm')}_{str(hi).replace('-', 'm')}"
    d[name] = ((d["event_time"] >= lo) & (d["event_time"] <= hi)).astype(int) * d["H"]
    terms.append((name, lo, hi))
    if hi < 0:
        leads.append(name)
fml = "y ~ " + " + ".join(item[0] for item in terms) + " | pref + provm"
event_model = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"})
coef_names = list(event_model.coef().index)
lead_idx = [coef_names.index(name) for name in leads]
lead_beta = event_model.coef()[leads].values
lead_vcov = event_model._vcov[np.ix_(lead_idx, lead_idx)]
lead_stat = float(lead_beta @ np.linalg.solve(lead_vcov, lead_beta))
lead_p = float(stats.chi2.sf(lead_stat, len(leads)))
event = []
for name, lo, hi in terms:
    event.append(dict(bin_lo=lo, bin_hi=hi,
                      beta=float(event_model.coef()[name]),
                      se=float(event_model.se()[name]),
                      pretrend_wald=lead_stat, pretrend_p=lead_p,
                      n=int(event_model._N)))
event_results = pd.DataFrame(event)
event_results.to_csv(os.path.join(OUT, "baidu_candidate_eventstudy.csv"),
                     index=False, encoding="utf-8-sig")
print(f"event-study joint leads: chi2={lead_stat:.3f}, p={lead_p:.4f}", flush=True)
print(event_results.to_string(index=False), flush=True)

# Leave one province out.
d = panel(["讨债公司", "讨债"])
loo = []
for province in sorted(d["province"].unique()):
    z = d[d["province"] != province].copy()
    model = pf.feols("y ~ X | pref + provm", data=z)
    loo.append(dict(omitted_province=province,
                    beta=float(model.coef()["X"]), n=int(model._N)))
loo = pd.DataFrame(loo)
loo.to_csv(os.path.join(OUT, "baidu_candidate_loo.csv"), index=False,
           encoding="utf-8-sig")
print(f"LOO: {int((loo['beta'] < 0).sum())}/{len(loo)} negative, "
      f"range=[{loo['beta'].min():+.5f}, {loo['beta'].max():+.5f}]", flush=True)

# Permute the 31 province inspection clocks, preserving the schedule multiset.
d = panel(["讨债公司", "讨债"])
observed = float(pf.feols("y ~ X | pref + provm", data=d).coef()["X"])
sched = (d[["province", "insp_month"]].drop_duplicates()
         .sort_values("province").reset_index(drop=True))
provinces = sched["province"].to_numpy()
clocks = sched["insp_month"].to_numpy()
rng = np.random.default_rng(20260717)
draws = []
for rep in range(999):
    clock_map = dict(zip(provinces, rng.permutation(clocks)))
    z = d.copy()
    perm_clock = pd.to_datetime(z["province"].map(clock_map))
    z["X_perm"] = (z["m"] >= perm_clock).astype(int) * z["H"]
    beta = float(pf.feols("y ~ X_perm | pref + provm", data=z).coef()["X_perm"])
    draws.append(beta)
    if (rep + 1) % 100 == 0:
        print(f"RI {rep + 1}/999", flush=True)
draws = np.asarray(draws)
null_mean = float(draws.mean())
p_two = float((1 + np.sum(np.abs(draws - null_mean)
                          >= abs(observed - null_mean))) / 1000)
p_one = float((1 + np.sum(draws <= observed)) / 1000)
pd.DataFrame({"draw": np.arange(1, 1000), "beta": draws}).to_csv(
    os.path.join(OUT, "baidu_candidate_timing_ri_draws.csv"), index=False,
    encoding="utf-8-sig")
pd.DataFrame([dict(observed=observed, permutation_mean=null_mean,
                   permutation_sd=float(draws.std(ddof=1)),
                   p_two_sided=p_two, p_one_sided=p_one, reps=999)]).to_csv(
    os.path.join(OUT, "baidu_candidate_timing_ri.csv"), index=False,
    encoding="utf-8-sig")
print(f"timing RI: observed={observed:+.5f}, null mean={null_mean:+.5f}, "
      f"two-sided p={p_two:.4f}, one-sided p={p_one:.4f}", flush=True)

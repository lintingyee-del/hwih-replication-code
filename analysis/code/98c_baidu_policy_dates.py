# -*- coding: utf-8 -*-
"""Common policy-date diagnostic for the pre-COVID Baidu dose.

Dates are institutional milestones frozen in the manifest.  These estimates
diagnose a nationwide shock interacted with pre-campaign H; they do not use or
claim random province inspection assignment.
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

xw = pd.read_parquet(os.path.join(DATA, "court_xwalk.parquet"))
pat = re.compile(r"(?:.*?省|.*?自治区)?(.+?(?:市|州|盟|地区))中级人民法院$")
names = {}
for _, row in xw.iterrows():
    match = pat.match(str(row["court_name"]))
    if match:
        names.setdefault(match.group(1), []).append(str(row["prefecture_code"]))
mapping = {nm: max(set(v), key=v.count) for nm, v in names.items()}
mapping.update({"北京市": "110100", "天津市": "120100",
                "上海市": "310100", "重庆市": "500100"})


def city2code(city):
    city = str(city)
    cand = city if city.endswith(("市", "州", "盟", "地区")) else city + "市"
    if cand in mapping:
        return mapping[cand]
    for nm, code in mapping.items():
        if nm[:-1] and nm[:-1] in city:
            return code
    return None


exp = pd.read_parquet(os.path.join(DATA, "exposure_v2.parquet"))
exp["prefecture_code"] = exp["prefecture_code"].astype(str)
raw = pd.read_csv(str(_REP_PROJECT / "data" / "derived" / "baidu_index_city_month.csv"),
                  dtype={"ym": str})
raw = raw[raw["keyword"].isin(["讨债公司", "讨债"])].copy()
raw["prefecture_code"] = raw["city"].map(city2code)
d = (raw[raw["prefecture_code"].isin(set(exp["prefecture_code"]))]
     .groupby(["prefecture_code", "ym"], as_index=False)["mean"].sum()
     .merge(exp[["prefecture_code", "province", "exposure_v2_z"]],
            on="prefecture_code", how="inner"))
d["m"] = pd.to_datetime(d["ym"] + "-01")
d = d[(d["m"] >= "2014-01-01") & (d["m"] <= "2019-12-01")].copy()
d["H"] = d["exposure_v2_z"]
d["t"] = ((d["m"].dt.year - 2014) * 12 + d["m"].dt.month - 1).astype(float)
d["Ht"] = d["H"] * (d["t"] - d["t"].mean())
d["pref"] = d["prefecture_code"]
d["provm"] = d["province"] + "_" + d["ym"]
d["prov_id"] = pd.factorize(d["province"])[0]

dates = ["2018-01-01", "2018-07-01", "2018-09-01",
         "2019-04-01", "2019-06-01"]
rows = []
for date in dates:
    for transform in ("asinh", "log1p"):
        for trend in (False, True):
            z = d.copy()
            z["X"] = (z["m"] >= pd.Timestamp(date)).astype(int) * z["H"]
            z["y"] = (np.arcsinh(z["mean"]) if transform == "asinh"
                      else np.log1p(z["mean"]))
            fml = "y ~ X" + (" + Ht" if trend else "") + " | pref + provm"
            model = pf.feols(fml, data=z, vcov={"CRV1": "prov_id"})
            wild = wild_score_p(fml, z, "X", cluster="prov_id", reps=9_999, seed=42)
            rows.append(dict(policy_date=date[:7], transform=transform,
                             htrend=trend, beta=float(model.coef()["X"]),
                             se=float(model.se()["X"]),
                             p_crv1=float(model.pvalue()["X"]), p_wild=wild,
                             n=int(model._N)))
            print(f"{date[:7]} {transform:6s} trend={int(trend)} "
                  f"b={float(model.coef()['X']):+.5f} "
                  f"p={float(model.pvalue()['X']):.4f} wild={wild:.4f}", flush=True)

res = pd.DataFrame(rows)
res["q_bh"] = multipletests(res["p_wild"], method="fdr_bh")[1]
res.to_csv(os.path.join(OUT, "baidu_policy_dates.csv"), index=False,
           encoding="utf-8-sig")

# Calendar event study centered on the first central-inspection batch, July 2018.
center = pd.Timestamp("2018-07-01")
z = d.copy()
z["event_time"] = ((z["m"].dt.year - center.year) * 12
                   + z["m"].dt.month - center.month)
z = z[(z["event_time"] >= -24) & (z["event_time"] <= 17)].copy()
z["y"] = np.arcsinh(z["mean"])
bins = [(-24, -19), (-18, -13), (-12, -7),
        (0, 5), (6, 11), (12, 17)]
terms = []
leads = []
for lo, hi in bins:
    name = f"e_{str(lo).replace('-', 'm')}_{str(hi).replace('-', 'm')}"
    z[name] = ((z["event_time"] >= lo) & (z["event_time"] <= hi)).astype(int) * z["H"]
    terms.append((name, lo, hi))
    if hi < 0:
        leads.append(name)
fml = "y ~ " + " + ".join(x[0] for x in terms) + " | pref + provm"
model = pf.feols(fml, data=z, vcov={"CRV1": "prov_id"})
names = list(model.coef().index)
idx = [names.index(name) for name in leads]
b = model.coef()[leads].values
V = model._vcov[np.ix_(idx, idx)]
wald = float(b @ np.linalg.solve(V, b))
pre_p = float(stats.chi2.sf(wald, len(leads)))
event = pd.DataFrame([
    dict(bin_lo=lo, bin_hi=hi, beta=float(model.coef()[name]),
         se=float(model.se()[name]), pretrend_wald=wald,
         pretrend_p=pre_p, n=int(model._N))
    for name, lo, hi in terms
])
event.to_csv(os.path.join(OUT, "baidu_policy_july_eventstudy.csv"),
             index=False, encoding="utf-8-sig")
print(f"July-2018 calendar event study pretrend p={pre_p:.4f}", flush=True)
print(event.to_string(index=False), flush=True)

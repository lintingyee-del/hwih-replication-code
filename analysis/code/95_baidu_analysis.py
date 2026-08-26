# -*- coding: utf-8 -*-
"""Baidu search demand for private debt collection vs campaign exposure.
Outputs: national series csv, dose regressions (staggered + clean window)
with wild-score p, full spec log."""

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
import os, re, sys
import numpy as np
import pandas as pd
import pyfixest as pf

from _wild import wild_score_p

sys.stdout.reconfigure(encoding="utf-8")
BASE = str(_REP_PROJECT)
OUT = os.path.join(BASE, "output", "baidu_index")
os.makedirs(OUT, exist_ok=True)
LOG = []

xw = pd.read_parquet(os.path.join(BASE, "data", "court_xwalk.parquet"))
pat = re.compile(r"(?:.*?省|.*?自治区)?(.+?(?:市|州|盟|地区))中级人民法院$")
names = {}
for _, r in xw.iterrows():
    m = pat.match(str(r["court_name"]))
    if m:
        names.setdefault(m.group(1), []).append(r["prefecture_code"])
name2code = {nm: max(set(v), key=v.count) for nm, v in names.items()}
name2code.update({"北京市": "110100", "天津市": "120100",
                  "上海市": "310100", "重庆市": "500100"})
exp = pd.read_parquet(os.path.join(BASE, "data", "exposure_v2.parquet"))
exp["backstop_collect_rate_z"] = (
    (exp["backstop_collect_rate"] - exp["backstop_collect_rate"].mean())
    / exp["backstop_collect_rate"].std(ddof=1)
)
codes = set(exp["prefecture_code"])
cp = pd.read_parquet(os.path.join(BASE, "data", "civil_panel.parquet"),
                     columns=["province", "insp_month"]).drop_duplicates()
insp = cp.groupby("province")["insp_month"].first()

d = pd.read_csv(str(_REP_PROJECT / "data" / "derived" / "baidu_index_city_month.csv"), dtype={"ym": str})


def city2code(c):
    c = str(c)
    cand = c if c.endswith(("市", "州", "盟", "地区")) else c + "市"
    if cand in name2code:
        return name2code[cand]
    for nm, cd in name2code.items():
        if nm[:-1] and nm[:-1] in c:
            return cd
    return None


d["prefecture_code"] = d["city"].map(city2code)
d = d[d["prefecture_code"].isin(codes)].copy()
print(f"匹配城市: {d['city'].nunique()}/358", flush=True)
d = d.merge(exp[["prefecture_code", "exposure_v2_z", "backstop_collect_rate_z", "province"]],
            on="prefecture_code")
d["insp_month"] = pd.to_datetime(d["province"].map(insp))
d["m"] = pd.to_datetime(d["ym"] + "-01")
d["y"] = np.arcsinh(d["mean"])


def run(label, kw, design, hvar="exposure_v2_z", end="2020-12-31"):
    s = d[(d["keyword"].isin(kw)) & (d["m"] <= end) & (d["m"] >= "2014-01-01")]
    s = (s.groupby(["prefecture_code", "province", "ym", "m", hvar, "insp_month"],
                   as_index=False)["mean"].sum())
    s["y"] = np.arcsinh(s["mean"])
    s["prov_id"] = pd.factorize(s["province"])[0]
    if design == "staggered":
        s["post"] = (s["m"] >= s["insp_month"]).astype(int)
        s["X"] = s["post"] * s[hvar]
        s["provm"] = s["province"] + s["ym"]
        f, fe = "y ~ X | prefecture_code + provm", "prefecture_code + provm"
        ctrl = None
    else:  # clean window
        s = s[(s["m"] >= "2017-01-01") & (s["m"] <= "2019-03-31")].copy()
        s["treat"] = (s["insp_month"] <= "2018-09-30").astype(int)
        s["post"] = (s["m"] >= "2018-07-01").astype(int)
        s["X"] = s["treat"] * s["post"] * s[hvar]
        s["pt"] = s["treat"] * s["post"]
        f, fe = "y ~ X + pt | prefecture_code + ym", "prefecture_code + ym"
        ctrl = "pt"
    m = pf.feols(f, data=s, vcov={"CRV1": "prov_id"})
    b = float(m.coef()["X"]); se = float(m.se()["X"])
    p = float(m.pvalue()["X"])
    wp = wild_score_p(f, s, "X", cluster="prov_id", reps=9_999, seed=42)
    rec = {"spec": label, "n": len(s), "beta": b, "se": se,
           "p_crv1": p, "p_wild": wp}
    LOG.append(rec)
    print(f"{label}: b={b:.4f} se={se:.4f} p={p:.3f} wild={wp:.3f}", flush=True)


run("B1 讨债公司 staggered", ["讨债公司"], "staggered")
run("B2 讨债公司 clean-window", ["讨债公司"], "clean")
run("B3 讨债 staggered", ["讨债"], "staggered")
run("B4 讨债 clean-window", ["讨债"], "clean")
run("B5 两词合并 staggered", ["讨债公司", "讨债"], "staggered")
run("B6 两词合并 clean-window", ["讨债公司", "讨债"], "clean")
run("B7 合并 催收后盾文本率构件 staggered", ["讨债公司", "讨债"], "staggered",
    hvar="backstop_collect_rate_z")
run("B8 合并 催收后盾文本率构件 clean-window", ["讨债公司", "讨债"], "clean",
    hvar="backstop_collect_rate_z")

pd.DataFrame(LOG).to_csv(os.path.join(OUT, "baidu_spec_log.csv"),
                         index=False, encoding="utf-8-sig")
# 全国月度序列输出
nat = (d[d["keyword"].isin(["讨债公司", "讨债"])]
       .groupby(["keyword", "ym"], as_index=False)["mean"].sum())
nat.to_csv(os.path.join(OUT, "national_monthly.csv"), index=False,
           encoding="utf-8-sig")
print("done", flush=True)

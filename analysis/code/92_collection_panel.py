# -*- coding: utf-8 -*-
"""Collection-industry firm dynamics vs campaign exposure.

Inputs: <restricted-source-path> (6,738 positive-context
collection firms), analysis exposure_v2.parquet (headline H_c by prefecture), civil_panel
(inspection month by province), court_xwalk (name->code mapping source).

Outputs (analysis/output/collection_firms/):
  national_quarterly.csv        entry/exit series for the exhibit
  firm_survival_results.csv     firm-level exit LPMs (campaign vs placebo window)
  panel_ppml_results.csv        prefecture-quarter PPML entry/exit on post x H
  collection_firm_log.txt
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
import os, re, sys
import numpy as np
import pandas as pd

from _wild import wild_score_p

sys.stdout.reconfigure(encoding="utf-8")
BASE = str(_REP_PROJECT)
OUT = os.path.join(BASE, "output", "collection_firms")
os.makedirs(OUT, exist_ok=True)
log = []


def say(s):
    print(s, flush=True)
    log.append(str(s))


# ---- name -> prefecture_code map ------------------------------------------
xw = pd.read_parquet(os.path.join(BASE, "data", "court_xwalk.parquet"))
pat = re.compile(r"(?:.*?省|.*?自治区)?(.+?(?:市|州|盟|地区))中级人民法院$")
names = {}
for _, r in xw.iterrows():
    m = pat.match(str(r["court_name"]))
    if m:
        nm = m.group(1)
        names.setdefault(nm, []).append(r["prefecture_code"])
name2code = {nm: max(set(v), key=v.count) for nm, v in names.items()}
name2code.update({"北京市": "110100", "天津市": "120100",
                  "上海市": "310100", "重庆市": "500100"})
say(f"中院名映射: {len(name2code)} 个")

exp = pd.read_parquet(os.path.join(BASE, "data", "exposure_v2.parquet"))
codes = set(exp["prefecture_code"])

# inspection month by province
cp = pd.read_parquet(os.path.join(BASE, "data", "civil_panel.parquet"),
                     columns=["province", "insp_month"]).drop_duplicates()
insp = cp.groupby("province")["insp_month"].first()

# ---- firms -----------------------------------------------------------------
h = pd.read_csv(str(_REP_PROJECT / "data" / "derived" / "registry_hits_deidentified.csv"), dtype=str)
# extractor wrote (city, district) under headers (所属省份, 所属城市, 所属区县):
# the 省份 column actually holds the CITY; realign here.
h = h.rename(columns={"所属城市": "district_raw"})
h["所属城市"] = h["所属省份"]


def city2code(c):
    c = str(c)
    if not c or c == "nan":
        return None
    cand = c if c.endswith(("市", "州", "盟", "地区")) else c + "市"
    if cand in name2code:
        return name2code[cand]
    for nm, cd in name2code.items():
        if nm[:-1] and nm[:-1] in c:
            return cd
    return None


h["em"] = pd.to_datetime(h["成立日期"], errors="coerce")
h["exit"] = h["经营状态"].str.contains("注销|吊销", na=False)
h["xm"] = pd.to_datetime(h["核准日期"], errors="coerce").where(h["exit"])

# ---- national quarterly series (FULL registry sample, before any filter) ----
nat = []
for q in pd.period_range("2013Q1", "2023Q3", freq="Q"):
    nat.append({"quarter": str(q),
                "entries": int(((h["em"].dt.to_period("Q")) == q).sum()),
                "exits": int(((h["xm"].dt.to_period("Q")) == q).sum())})
pd.DataFrame(nat).to_csv(os.path.join(OUT, "national_quarterly.csv"), index=False)
say(f"national_quarterly.csv written (full sample, {len(h)} firms)")

# ---- restrict to exposure-sample prefectures for all dose analyses ----------
h["prefecture_code"] = h["所属城市"].map(city2code)
h = h[h["prefecture_code"].isin(codes)].copy()
say(f"匹配到暴露度样本内的企业: {len(h)}")

# ---- firm-level survival LPM ------------------------------------------------
h = h.merge(exp[["prefecture_code", "exposure_v2_z", "province"]], on="prefecture_code")
h["insp_month"] = h["province"].map(insp)
h["fy"] = h["em"].dt.year


def lpm(sample, y, label):
    import pyfixest as pf
    d = sample.copy()
    d["y"] = y.astype(float)
    d["prov_id"] = pd.factorize(d["province"])[0]
    d["fy"] = d["fy"].fillna(0).astype(int).astype(str)
    d = d.dropna(subset=["exposure_v2_z", "y"])
    d = d[d.groupby("province")["y"].transform("size") >= 2].reset_index(drop=True)
    m = pf.feols("y ~ exposure_v2_z + C(fy) | province", data=d,
                 vcov={"CRV1": "prov_id"})
    b = float(m.coef()["exposure_v2_z"]); se = float(m.se()["exposure_v2_z"])
    p = float(m.pvalue()["exposure_v2_z"])
    wp = wild_score_p("y ~ exposure_v2_z + C(fy) | province", d,
                      "exposure_v2_z", cluster="prov_id", reps=9_999, seed=42)
    say(f"{label}: n={len(d)} beta={b:.4f} se={se:.4f} CRV1 p={p:.3f} wild p={wp:.3f}")
    return {"spec": label, "n": len(d), "beta": b, "se": se,
            "p_crv1": p, "p_wild": wp}


res = []
alive18 = h[(h["em"] <= "2017-12-31") &
            (~h["exit"] | (h["xm"] >= "2018-01-01"))].copy()
y18 = (alive18["exit"] & (alive18["xm"] <= "2020-12-31")
       & (alive18["xm"] >= "2018-01-01"))
res.append(lpm(alive18, y18, "exit 2018-20 | alive end-2017 (campaign)"))

alive15 = h[(h["em"] <= "2014-12-31") &
            (~h["exit"] | (h["xm"] >= "2015-01-01"))].copy()
y15 = (alive15["exit"] & (alive15["xm"] <= "2017-12-31")
       & (alive15["xm"] >= "2015-01-01"))
res.append(lpm(alive15, y15, "exit 2015-17 | alive end-2014 (placebo)"))

# post-window persistence: alive end-2020, exit 2021-23
alive21 = h[(h["em"] <= "2020-12-31") &
            (~h["exit"] | (h["xm"] >= "2021-01-01"))].copy()
y21 = (alive21["exit"] & (alive21["xm"] <= "2023-12-31")
       & (alive21["xm"] >= "2021-01-01"))
res.append(lpm(alive21, y21, "exit 2021-23 | alive end-2020 (persistence)"))
pd.DataFrame(res).to_csv(os.path.join(OUT, "firm_survival_results.csv"), index=False)

# ---- prefecture x quarter PPML ----------------------------------------------
import pyfixest as pf
qidx = pd.period_range("2014Q1", "2021Q4", freq="Q")
rows = []
for code, g in h.groupby("prefecture_code"):
    for q in qidx:
        rows.append({"prefecture_code": code, "q": str(q),
                     "entries": int((g["em"].dt.to_period("Q") == q).sum()),
                     "exits": int((g["xm"].dt.to_period("Q") == q).sum())})
pan = pd.DataFrame(rows).merge(
    exp[["prefecture_code", "exposure_v2_z", "province"]], on="prefecture_code")
pan["insp_month"] = pan["province"].map(insp)
pan["qend"] = pd.PeriodIndex(pan["q"], freq="Q").to_timestamp(how="end")
pan["post"] = (pan["qend"] >= pd.to_datetime(pan["insp_month"])).astype(int)
pan["postxH"] = pan["post"] * pan["exposure_v2_z"]
pan["provq"] = pan["province"] + "_" + pan["q"]
pan["prov_id"] = pd.factorize(pan["province"])[0]
pres = []
for yv in ["exits", "entries"]:
    try:
        m = pf.fepois(f"{yv} ~ postxH | prefecture_code + provq", data=pan,
                      vcov={"CRV1": "prov_id"})
        b = float(m.coef()["postxH"]); se = float(m.se()["postxH"])
        p = float(m.pvalue()["postxH"])
        say(f"PPML {yv}: beta={b:.4f} se={se:.4f} p={p:.3f} (N={m._N})")
        pres.append({"outcome": yv, "beta": b, "se": se, "p_crv1": p})
    except Exception as e:
        say(f"PPML {yv} failed: {type(e).__name__}: {e}")
pd.DataFrame(pres).to_csv(os.path.join(OUT, "panel_ppml_results.csv"), index=False)

# ---- firm x quarter hazard DiD (staggered inspection timing) ----------------
say("\n== 企业x季度 危险率 DiD ==")
hz = h.dropna(subset=["em", "exposure_v2_z"]).copy()
hz["exq"] = hz["xm"].dt.to_period("Q")
rows = []
for q in pd.period_range("2016Q1", "2021Q4", freq="Q"):
    at_risk = hz[(hz["em"].dt.to_period("Q") < q)
                 & (hz["exq"].isna() | (hz["exq"] >= q))]
    rows.append(pd.DataFrame({
        "prefecture_code": at_risk["prefecture_code"].values,
        "province": at_risk["province"].values,
        "exposure_v2_z": at_risk["exposure_v2_z"].values,
        "fy": at_risk["fy"].fillna(0).astype(int).astype(str).values,
        "insp_month": at_risk["insp_month"].values,
        "q": str(q),
        "y": (at_risk["exq"] == q).astype(float).values}))
hp = pd.concat(rows, ignore_index=True)
hp["qend"] = pd.PeriodIndex(hp["q"], freq="Q").to_timestamp(how="end")
hp["post"] = (hp["qend"] >= pd.to_datetime(hp["insp_month"])).astype(int)
hp["postxH"] = hp["post"] * hp["exposure_v2_z"]
hp["provq"] = hp["province"] + "_" + hp["q"]
hp["prov_id"] = pd.factorize(hp["province"])[0]
say(f"风险集: {len(hp)} 企业-季度, 退出事件 {int(hp['y'].sum())}, "
    f"基准危险率 {hp['y'].mean():.4f}/季")
m = pf.feols("y ~ postxH + C(fy) | prefecture_code + provq", data=hp,
             vcov={"CRV1": "prov_id"})
b = float(m.coef()["postxH"]); se = float(m.se()["postxH"])
p = float(m.pvalue()["postxH"])
wp = wild_score_p(
    "y ~ postxH + C(fy) | prefecture_code + provq", hp, "postxH",
    cluster="prov_id", reps=9_999, seed=42)
base = hp["y"].mean()
say(f"hazard DiD postxH: beta={b:.5f} se={se:.5f} CRV1 p={p:.3f} wild p={wp:.3f}"
    f" | 相对基准率 {b/base*100:.1f}%")
pd.DataFrame([{"spec": "firm-quarter hazard DiD", "n": len(hp),
               "events": int(hp["y"].sum()), "base_hazard": base,
               "beta": b, "se": se, "p_crv1": p, "p_wild": wp}]
             ).to_csv(os.path.join(OUT, "hazard_did_results.csv"), index=False)

open(os.path.join(OUT, "collection_firm_log.txt"), "w",
     encoding="utf-8").write("\n".join(log))
say("ALL DONE")

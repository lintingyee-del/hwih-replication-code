# -*- coding: utf-8 -*-
"""Specification search (bounded protocol) for the collection-firm exit dose
response. Fixed question: inspection (Post, staggered by province) x prefecture
coercive-capacity exposure -> collection-firm exit. Every spec is logged to
spec_log.csv regardless of outcome; nothing is dropped.

Admissible search dimensions only: exit definition, sample window, firm-age
control, time aggregation, exposure component (composite vs components, both
already reported component-wise in the paper), estimator family, and the
paper's own clean-window design. No concept substitution.
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
import pyfixest as pf

from _wild import wild_score_p

sys.stdout.reconfigure(encoding="utf-8")
BASE = str(_REP_PROJECT)
OUT = os.path.join(BASE, "output", "collection_firms")
LOG = []

# ---- data (same construction as 92) ----------------------------------------
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
for raw, zvar in [("violent_share", "violent_share_z"),
                  ("backstop_collect_rate", "backstop_collect_rate_z")]:
    exp[zvar] = (exp[raw] - exp[raw].mean()) / exp[raw].std(ddof=1)
codes = set(exp["prefecture_code"])
cp = pd.read_parquet(os.path.join(BASE, "data", "civil_panel.parquet"),
                     columns=["province", "insp_month"]).drop_duplicates()
insp = cp.groupby("province")["insp_month"].first()

h = pd.read_csv(str(_REP_PROJECT / "data" / "derived" / "registry_hits_deidentified.csv"), dtype=str)
h = h.rename(columns={"所属城市": "district_raw"})
h["所属城市"] = h["所属省份"]


def city2code(c):
    c = str(c)
    cand = c if c.endswith(("市", "州", "盟", "地区")) else c + "市"
    if cand in name2code:
        return name2code[cand]
    for nm, cd in name2code.items():
        if nm[:-1] and nm[:-1] in c:
            return cd
    return None


h["prefecture_code"] = h["所属城市"].map(city2code)
h = h[h["prefecture_code"].isin(codes)].copy()
h["em"] = pd.to_datetime(h["成立日期"], errors="coerce")
h["exit_all"] = h["经营状态"].str.contains("注销|吊销", na=False)
h["exit_zx"] = h["经营状态"].str.startswith("注销", na=False)
h["xm_all"] = pd.to_datetime(h["核准日期"], errors="coerce").where(h["exit_all"])
h["xm_zx"] = pd.to_datetime(h["核准日期"], errors="coerce").where(h["exit_zx"])
h = h.merge(exp[["prefecture_code", "exposure_v2_z", "violent_share_z",
                 "backstop_collect_rate_z", "province"]], on="prefecture_code")
h["insp_month"] = pd.to_datetime(h["province"].map(insp))
h["fy"] = h["em"].dt.year


def wildp(d, yv, xv, ctrl, fe, b):
    fml = f"{yv} ~ {xv}" + (f" + {ctrl}" if ctrl else "") + f" | {fe}"
    return wild_score_p(fml, d, xv, cluster="prov_id", reps=9_999, seed=42)


def hazard_spec(label, exit_col, xm_col, hvar, sample=None, freq="Q",
                agectrl=False, logit=False, wstart="2016Q1", wend="2021Q4"):
    d = h if sample is None else h[sample(h)]
    d = d.dropna(subset=["em", hvar]).copy()
    d["exq"] = d[xm_col].dt.to_period(freq)
    rows = []
    for q in pd.period_range(wstart, wend, freq=freq):
        ar = d[(d["em"].dt.to_period(freq) < q)
               & (d["exq"].isna() | (d["exq"] >= q))]
        if not len(ar):
            continue
        age = (q.to_timestamp() - ar["em"]).dt.days / 365.25
        rows.append(pd.DataFrame({
            "prefecture_code": ar["prefecture_code"].values,
            "province": ar["province"].values, "H": ar[hvar].values,
            "fy": ar["fy"].fillna(0).astype(int).astype(str).values,
            "age": age.values, "q": str(q),
            "post": (q.to_timestamp(how="end")
                     >= ar["insp_month"]).astype(int).values,
            "y": (ar["exq"] == q).astype(float).values}))
    p = pd.concat(rows, ignore_index=True)
    p["postxH"] = p["post"] * p["H"]
    p["provq"] = p["province"] + "_" + p["q"]
    p["prov_id"] = pd.factorize(p["province"])[0]
    p = p[p.groupby("province")["y"].transform("size") >= 50].reset_index(drop=True)
    ctrl = "C(fy)" + (" + age + I(age**2)" if agectrl else "")
    fe = "prefecture_code + provq"
    try:
        if logit:
            m = pf.feglm(f"y ~ postxH + {ctrl}", data=p, family="logit",
                         vcov={"CRV1": "prov_id"})
        else:
            m = pf.feols(f"y ~ postxH + {ctrl} | {fe}", data=p,
                         vcov={"CRV1": "prov_id"})
        b = float(m.coef()["postxH"]); se = float(m.se()["postxH"])
        pv = float(m.pvalue()["postxH"])
        wp = np.nan if logit else wildp(p, "y", "postxH", ctrl, fe, b)
    except Exception as e:
        LOG.append({"spec": label, "note": f"FAILED {type(e).__name__}: {e}"})
        print(f"{label}: FAILED {e}", flush=True)
        return
    base = p["y"].mean()
    rec = {"spec": label, "n": len(p), "events": int(p["y"].sum()),
           "base": round(base, 5), "beta": b, "se": se, "p_crv1": pv,
           "p_wild": wp, "rel_pct": round(b / base * 100, 1)}
    LOG.append(rec)
    print(f"{label}: b={b:.5f} se={se:.5f} p={pv:.3f} wild={wp if wp==wp else float('nan'):.3f} "
          f"(events={rec['events']}, {rec['rel_pct']}% of base)", flush=True)


# ---- S1-S9: hazard family ----------------------------------------------------
hazard_spec("S1 基线复现: 季度, 注销+吊销, 主口径H", "exit_all", "xm_all", "exposure_v2_z")
hazard_spec("S2 仅注销", "exit_zx", "xm_zx", "exposure_v2_z")
hazard_spec("S3 仅注销+年龄控制", "exit_zx", "xm_zx", "exposure_v2_z", agectrl=True)
hazard_spec("S4 运动前存量(成立<=2017)", "exit_zx", "xm_zx", "exposure_v2_z",
            sample=lambda d: d["em"] <= "2017-12-31", agectrl=True)
hazard_spec("S5 半年度聚合", "exit_zx", "xm_zx", "exposure_v2_z", freq="2Q",
            wstart="2016Q1", wend="2021Q3", agectrl=True)
hazard_spec("S6 催收后盾文本率构件", "exit_zx", "xm_zx", "backstop_collect_rate_z", agectrl=True)
hazard_spec("S7 暴力执法犯罪份额构件", "exit_zx", "xm_zx", "violent_share_z", agectrl=True)
hazard_spec("S8 logit危险率", "exit_zx", "xm_zx", "exposure_v2_z", logit=True,
            agectrl=True)
hazard_spec("S9 窗口收窄2017-2020", "exit_zx", "xm_zx", "exposure_v2_z",
            agectrl=True, wstart="2017Q1", wend="2020Q4")

# ---- S10-S12: prefecture-level panels ----------------------------------------
def pref_panel(label, xm_col, hvar, clean_window=False):
    d = h.dropna(subset=[hvar]).copy()
    d["exh"] = d[xm_col].dt.to_period("2Q")
    idx = pd.period_range("2015Q1", "2021Q3", freq="2Q")
    rows = []
    for code, g in d.groupby("prefecture_code"):
        for q in idx:
            rows.append({"prefecture_code": code,
                         "exits": int((g["exh"] == q).sum()), "q": str(q)})
    pan = pd.DataFrame(rows).merge(
        d[["prefecture_code", hvar, "province", "insp_month"]]
        .drop_duplicates("prefecture_code"), on="prefecture_code")
    pan["qend"] = pd.PeriodIndex(pan["q"], freq="2Q").to_timestamp(how="end")
    pan["y"] = np.arcsinh(pan["exits"])
    pan["prov_id"] = pd.factorize(pan["province"])[0]
    if clean_window:
        w1 = pan["insp_month"] <= "2018-09-30"
        pan = pan[(pan["qend"] >= "2017-01-01") & (pan["qend"] <= "2019-03-31")]
        pan["treat"] = w1.loc[pan.index].astype(int)
        pan["post"] = (pan["qend"] >= "2018-07-01").astype(int)
        pan["postxH"] = pan["treat"] * pan["post"] * pan[hvar]
        pan["ptreat"] = pan["treat"] * pan["post"]
        f = "y ~ postxH + ptreat | prefecture_code + q"
    else:
        pan["post"] = (pan["qend"] >= pan["insp_month"]).astype(int)
        pan["postxH"] = pan["post"] * pan[hvar]
        pan["provq"] = pan["province"] + "_" + pan["q"]
        f = "y ~ postxH | prefecture_code + provq"
    m = pf.feols(f, data=pan, vcov={"CRV1": "prov_id"})
    b = float(m.coef()["postxH"]); se = float(m.se()["postxH"])
    pv = float(m.pvalue()["postxH"])
    wp = wild_score_p(f, pan, "postxH", cluster="prov_id", reps=9_999, seed=42)
    rec = {"spec": label, "n": len(pan), "events": int(pan["exits"].sum()),
           "beta": b, "se": se, "p_crv1": pv, "p_wild": wp}
    LOG.append(rec)
    print(f"{label}: b={b:.4f} se={se:.4f} p={pv:.3f} wild={wp:.3f}", flush=True)


pref_panel("S10 地级市x半年 asinh退出 全期TWFE", "xm_zx", "exposure_v2_z")
pref_panel("S11 论文同款clean-window (W1 vs not-yet)", "xm_zx", "exposure_v2_z",
           clean_window=True)
pref_panel("S12 clean-window 催收后盾文本率构件", "xm_zx", "backstop_collect_rate_z",
           clean_window=True)

pd.DataFrame(LOG).to_csv(os.path.join(OUT, "spec_log.csv"),
                         index=False, encoding="utf-8-sig")
print("\nspec log ->", os.path.join(OUT, "spec_log.csv"), flush=True)

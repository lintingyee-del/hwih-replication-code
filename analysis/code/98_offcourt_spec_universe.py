# -*- coding: utf-8 -*-
"""Frozen specification universe for local firm-registry and Baidu doses.

The universe is declared in 97_specification_universe_manifest.json.  Every row
is retained, BH-adjusted, and tagged for theory direction.  Linear full-cell
models receive null-imposed wild-score p-values; sparse PPML rows cannot pass
the claim gate without confirmation from a full-cell linear model.
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
import glob
import os
import re
import sys

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf
from statsmodels.stats.multitest import multipletests

from _wild import wild_score_p

sys.stdout.reconfigure(encoding="utf-8")
BASE = str(_REP_PROJECT)
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output", "spec_universe")
os.makedirs(OUT, exist_ok=True)
LOG = []


def say(msg):
    print(msg, flush=True)
    LOG.append(str(msg))


def bh_tag(res):
    res = res.copy()
    ok = res["beta"].notna() & res["p_crv1"].notna()
    res["p_for_bh"] = np.where(res["p_wild"].notna(),
                                res["p_wild"], res["p_crv1"])
    res["q_family"] = np.nan
    for fam, idx in res[ok].groupby("family").groups.items():
        res.loc[idx, "q_family"] = multipletests(
            res.loc[idx, "p_for_bh"].astype(float), method="fdr_bh")[1]
    res["q_global"] = np.nan
    idx = res.index[ok]
    res.loc[idx, "q_global"] = multipletests(
        res.loc[idx, "p_for_bh"].astype(float), method="fdr_bh")[1]
    res["direction_ok"] = (np.sign(res["beta"])
                            == np.sign(res["expected_sign"]))
    res["linear_claim_gate"] = (
        res["method"].isin(["asinh_rate", "log1p_rate", "asinh_index", "log1p_index"])
        & res["direction_ok"].fillna(False)
        & (res["p_wild"] < 0.05)
        & (res["q_family"] < 0.10)
    )
    return res


def summary(res):
    ok = res[res["beta"].notna()]
    return (ok.groupby("family")
            .agg(specs=("spec", "size"), median_beta=("beta", "median"),
                 min_beta=("beta", "min"), max_beta=("beta", "max"),
                 share_expected_sign=("direction_ok", "mean"),
                 p05_rows=("p_for_bh", lambda x: int((x < 0.05).sum())),
                 q10_rows=("q_family", lambda x: int((x < 0.10).sum())),
                 claim_gate_rows=("linear_claim_gate", "sum"))
            .reset_index())


# Shared court-name to prefecture mapping and headline exposure ----------------
xw = pd.read_parquet(os.path.join(DATA, "court_xwalk.parquet"))
pat = re.compile(r"(?:.*?省|.*?自治区)?(.+?(?:市|州|盟|地区))中级人民法院$")
names = {}
for _, row in xw.iterrows():
    m = pat.match(str(row["court_name"]))
    if m:
        names.setdefault(m.group(1), []).append(str(row["prefecture_code"]))
name2code = {nm: max(set(v), key=v.count) for nm, v in names.items()}
name2code.update({"北京市": "110100", "天津市": "120100",
                  "上海市": "310100", "重庆市": "500100"})


def city2code(city):
    city = str(city)
    if not city or city == "nan":
        return None
    cand = city if city.endswith(("市", "州", "盟", "地区")) else city + "市"
    if cand in name2code:
        return name2code[cand]
    for nm, code in name2code.items():
        if nm[:-1] and nm[:-1] in city:
            return code
    return None


exp = pd.read_parquet(os.path.join(DATA, "exposure_v2.parquet"))
exp["prefecture_code"] = exp["prefecture_code"].astype(str)
codes = set(exp["prefecture_code"])
cp = pd.read_parquet(os.path.join(DATA, "civil_panel.parquet"),
                     columns=["province", "insp_month"]).drop_duplicates()
insp = cp.groupby("province")["insp_month"].first()


# ========================== firm registry ====================================
say("== build firm-registry denominator panel ==")
con = duckdb.connect()
agg_glob = str(_REP_PROJECT / "data" / "derived" / "registry_aggregate" / "*.csv").replace("\\", "/")
all_city_month = con.sql(f"""
  SELECT city, month,
         SUM(entries_all)::DOUBLE AS entries_all,
         SUM(exits_all)::DOUBLE AS exits_all
  FROM read_csv_auto('{agg_glob}', union_by_name=true, header=true)
  GROUP BY 1,2
""").df()
con.close()
all_city_month["prefecture_code"] = all_city_month["city"].map(city2code)
all_city_month = all_city_month[all_city_month["prefecture_code"].isin(codes)].copy()
all_city_month["month"] = pd.to_datetime(all_city_month["month"], errors="coerce")
all_city_month = all_city_month.dropna(subset=["month"])
all_city_month["q"] = all_city_month["month"].dt.to_period("Q")
den = (all_city_month.groupby(["prefecture_code", "q"], as_index=False)
       [["entries_all", "exits_all"]].sum())

h = pd.read_csv(str(_REP_PROJECT / "data" / "derived" / "registry_hits_deidentified.csv"), dtype=str)
# Extractor headers are shifted: 所属省份 contains the city.
h["city"] = h["所属省份"]
h["prefecture_code"] = h["city"].map(city2code)
h = h[h["prefecture_code"].isin(codes)].copy()
h["entry_date"] = pd.to_datetime(h["成立日期"], errors="coerce")
h["is_exit"] = h["经营状态"].str.contains("注销|吊销", na=False)
h["exit_date"] = pd.to_datetime(h["核准日期"], errors="coerce").where(h["is_exit"])
h["entry_q"] = h["entry_date"].dt.to_period("Q")
h["exit_q"] = h["exit_date"].dt.to_period("Q")
ent = (h.dropna(subset=["entry_q"]).groupby(["prefecture_code", "entry_q"])
       .size().rename("collection_entries").reset_index()
       .rename(columns={"entry_q": "q"}))
ext = (h.dropna(subset=["exit_q"]).groupby(["prefecture_code", "exit_q"])
       .size().rename("collection_exits").reset_index()
       .rename(columns={"exit_q": "q"}))

qindex = pd.period_range("2014Q1", "2021Q4", freq="Q")
valid_codes = sorted(set(den["prefecture_code"]))
firm = pd.MultiIndex.from_product([valid_codes, qindex],
                                  names=["prefecture_code", "q"]).to_frame(index=False)
firm = firm.merge(den, on=["prefecture_code", "q"], how="left")
firm = firm.merge(ent, on=["prefecture_code", "q"], how="left")
firm = firm.merge(ext, on=["prefecture_code", "q"], how="left")
firm[["collection_entries", "collection_exits"]] = firm[
    ["collection_entries", "collection_exits"]].fillna(0)
firm = firm.merge(exp[["prefecture_code", "province", "exposure_v2_z"]],
                  on="prefecture_code", how="inner")
firm["insp_month"] = pd.to_datetime(firm["province"].map(insp))
firm["qend"] = firm["q"].dt.to_timestamp(how="end")
firm["year"] = firm["q"].dt.year
firm["t"] = np.arange(len(qindex))[firm["q"].map({q: i for i, q in enumerate(qindex)}).values]
firm["H"] = firm["exposure_v2_z"]
firm["Ht"] = firm["H"] * (firm["t"] - firm["t"].mean())
firm["post_staggered"] = (firm["qend"] >= firm["insp_month"]).astype(int)
firm["post_national"] = (firm["q"] >= pd.Period("2018Q1")).astype(int)
firm["post_lag2"] = (firm["q"] >= pd.Period("2018Q3")).astype(int)
firm["pref"] = firm["prefecture_code"]
firm["qstr"] = firm["q"].astype(str)
firm["provq"] = firm["province"] + "_" + firm["qstr"]
firm["prov_id"] = pd.factorize(firm["province"])[0]
firm = firm[(firm["entries_all"] > 0) & (firm["exits_all"] > 0)].copy()
say(f"firm panel: {len(firm):,} cells, {firm['pref'].nunique()} prefectures, "
    f"collection entries={int(firm['collection_entries'].sum()):,}, "
    f"exits={int(firm['collection_exits'].sum()):,}")

FIRM_ROWS = []
firm_samples = [("full", 2014, 2021), ("pre2021", 2014, 2020),
                ("late", 2016, 2021)]
firm_methods = ("asinh_rate", "log1p_rate", "ppml_offset")
for family, count_col, den_col, expected, timings in [
    ("firm_entry_share", "collection_entries", "entries_all", -1,
     ("staggered", "national")),
    ("firm_exit_share", "collection_exits", "exits_all", +1,
     ("staggered", "national", "lag2")),
]:
    for sample, start, end in firm_samples:
        for timing in timings:
            for method in firm_methods:
                for trend in (False, True):
                    d = firm[(firm["year"] >= start) & (firm["year"] <= end)].copy()
                    d["X"] = d[f"post_{timing}"] * d["H"]
                    d["rate_million"] = d[count_col] / d[den_col] * 1_000_000
                    rhs = "X" + (" + Ht" if trend else "")
                    spec = f"{family}__{sample}__{timing}__{method}__trend{int(trend)}"
                    try:
                        if method == "asinh_rate":
                            d["y"] = np.arcsinh(d["rate_million"])
                            fml = f"y ~ {rhs} | pref + provq"
                            m = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"})
                            wp = wild_score_p(fml, d, "X", cluster="prov_id",
                                              reps=9_999, seed=42)
                        elif method == "log1p_rate":
                            d["y"] = np.log1p(d["rate_million"])
                            fml = f"y ~ {rhs} | pref + provq"
                            m = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"})
                            wp = wild_score_p(fml, d, "X", cluster="prov_id",
                                              reps=9_999, seed=42)
                        else:
                            d["log_all"] = np.log(d[den_col])
                            fml = f"{count_col} ~ {rhs} | pref + provq"
                            m = pf.fepois(fml, data=d, offset="log_all",
                                         vcov={"CRV1": "prov_id"})
                            wp = np.nan
                        rec = dict(spec=spec, family=family, method=method,
                                   sample=sample, timing=timing, start=start, end=end,
                                   htrend=trend, expected_sign=expected,
                                   beta=float(m.coef()["X"]), se=float(m.se()["X"]),
                                   p_crv1=float(m.pvalue()["X"]), p_wild=wp,
                                   n_input=len(d), n_fit=int(m._N),
                                   fit_share=int(m._N) / len(d), note="")
                        FIRM_ROWS.append(rec)
                        ptxt = f"{wp:.3f}" if np.isfinite(wp) else "--"
                        say(f"{spec:72s} b={rec['beta']:+.4f} p={rec['p_crv1']:.3f} "
                            f"wild={ptxt} N={rec['n_fit']}/{rec['n_input']}")
                    except Exception as exc:
                        say(f"{spec}: FAILED {type(exc).__name__}: {exc}")
                        FIRM_ROWS.append(dict(spec=spec, family=family, method=method,
                                              sample=sample, timing=timing, start=start,
                                              end=end, htrend=trend,
                                              expected_sign=expected, note=
                                              f"FAILED {type(exc).__name__}: {exc}"))

firm_res = bh_tag(pd.DataFrame(FIRM_ROWS))
firm_res.to_csv(os.path.join(OUT, "firm_spec_universe.csv"), index=False,
                encoding="utf-8-sig")
firm_sum = summary(firm_res)
firm_sum.to_csv(os.path.join(OUT, "firm_spec_universe_summary.csv"),
                index=False, encoding="utf-8-sig")
say("\n=== firm family summary ===")
say(firm_sum.to_string(index=False))
say("firm mechanical claim-gate rows:")
fg = firm_res[firm_res["linear_claim_gate"]]
say(fg[["spec", "beta", "se", "p_wild", "q_family"]].to_string(index=False)
    if len(fg) else "none")


# ========================== Baidu index ======================================
say("\n== build Baidu combined-term panel ==")
b = pd.read_csv(str(_REP_PROJECT / "data" / "derived" / "baidu_index_city_month.csv"),
                dtype={"ym": str})
b = b[b["keyword"].isin(["讨债公司", "讨债"])].copy()
b["prefecture_code"] = b["city"].map(city2code)
b = b[b["prefecture_code"].isin(codes)].copy()
b = (b.groupby(["prefecture_code", "ym"], as_index=False)["mean"].sum()
     .merge(exp[["prefecture_code", "province", "exposure_v2_z"]],
            on="prefecture_code", how="inner"))
b["insp_month"] = pd.to_datetime(b["province"].map(insp))
b["m"] = pd.to_datetime(b["ym"] + "-01")
b["year"] = b["m"].dt.year
b["t"] = ((b["m"].dt.year - 2014) * 12 + b["m"].dt.month - 1).astype(float)
b["H"] = b["exposure_v2_z"]
b["Ht"] = b["H"] * (b["t"] - b["t"].mean())
b["post_staggered"] = (b["m"] >= b["insp_month"]).astype(int)
b["post_national"] = (b["m"] >= pd.Timestamp("2018-01-01")).astype(int)
b["first_wave"] = (b["insp_month"] <= pd.Timestamp("2018-09-30")).astype(int)
b["post_clean"] = (b["m"] >= pd.Timestamp("2018-07-01")).astype(int)
b["pref"] = b["prefecture_code"]
b["provm"] = b["province"] + "_" + b["ym"]
b["prov_id"] = pd.factorize(b["province"])[0]
say(f"Baidu panel: {len(b):,} cells, {b['pref'].nunique()} prefectures")

BAIDU_ROWS = []
regular_samples = [("full", "2014-01-01", "2020-12-31"),
                   ("pre2020", "2014-01-01", "2019-12-31"),
                   ("late", "2016-01-01", "2020-12-31")]
for timing in ("staggered", "national"):
    for sample, start, end in regular_samples:
        for method in ("asinh_index", "log1p_index"):
            for fe_kind in ("strict", "month"):
                for trend in (False, True):
                    d = b[(b["m"] >= start) & (b["m"] <= end)].copy()
                    d["X"] = d[f"post_{timing}"] * d["H"]
                    d["y"] = (np.arcsinh(d["mean"]) if method == "asinh_index"
                              else np.log1p(d["mean"]))
                    rhs = "X" + (" + Ht" if trend else "")
                    fe = "pref + provm" if fe_kind == "strict" else "pref + ym"
                    fml = f"y ~ {rhs} | {fe}"
                    spec = f"baidu__{sample}__{timing}__{method}__{fe_kind}__trend{int(trend)}"
                    try:
                        m = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"})
                        wp = wild_score_p(fml, d, "X", cluster="prov_id",
                                          reps=9_999, seed=42)
                        rec = dict(spec=spec, family="baidu_combined", method=method,
                                   sample=sample, timing=timing, start=start, end=end,
                                   fixed_effects=fe_kind, htrend=trend,
                                   expected_sign=-1, beta=float(m.coef()["X"]),
                                   se=float(m.se()["X"]), p_crv1=float(m.pvalue()["X"]),
                                   p_wild=wp, n_input=len(d), n_fit=int(m._N),
                                   fit_share=int(m._N) / len(d), note="")
                        BAIDU_ROWS.append(rec)
                        say(f"{spec:72s} b={rec['beta']:+.4f} p={rec['p_crv1']:.3f} "
                            f"wild={wp:.3f}")
                    except Exception as exc:
                        say(f"{spec}: FAILED {type(exc).__name__}: {exc}")
                        BAIDU_ROWS.append(dict(spec=spec, family="baidu_combined",
                                               method=method, sample=sample,
                                               timing=timing, start=start, end=end,
                                               fixed_effects=fe_kind, htrend=trend,
                                               expected_sign=-1, note=
                                               f"FAILED {type(exc).__name__}: {exc}"))

# Clean-window rows are a separate timing design but remain in the same family.
for method in ("asinh_index", "log1p_index"):
    for fe_kind in ("strict", "month"):
        for trend in (False, True):
            d = b[(b["m"] >= "2017-01-01") & (b["m"] <= "2019-03-31")].copy()
            d["pt"] = d["first_wave"] * d["post_clean"]
            d["X"] = d["pt"] * d["H"]
            d["y"] = (np.arcsinh(d["mean"]) if method == "asinh_index"
                      else np.log1p(d["mean"]))
            controls = [] if fe_kind == "strict" else ["pt"]
            if trend:
                controls.append("Ht")
            rhs = "X" + (" + " + " + ".join(controls) if controls else "")
            fe = "pref + provm" if fe_kind == "strict" else "pref + ym"
            fml = f"y ~ {rhs} | {fe}"
            spec = f"baidu__clean__clean__{method}__{fe_kind}__trend{int(trend)}"
            try:
                m = pf.feols(fml, data=d, vcov={"CRV1": "prov_id"})
                wp = wild_score_p(fml, d, "X", cluster="prov_id",
                                  reps=9_999, seed=42)
                BAIDU_ROWS.append(dict(
                    spec=spec, family="baidu_combined", method=method,
                    sample="clean", timing="clean", start="2017-01-01",
                    end="2019-03-31", fixed_effects=fe_kind, htrend=trend,
                    expected_sign=-1, beta=float(m.coef()["X"]),
                    se=float(m.se()["X"]), p_crv1=float(m.pvalue()["X"]),
                    p_wild=wp, n_input=len(d), n_fit=int(m._N),
                    fit_share=int(m._N) / len(d), note=""))
                say(f"{spec:72s} b={float(m.coef()['X']):+.4f} "
                    f"p={float(m.pvalue()['X']):.3f} wild={wp:.3f}")
            except Exception as exc:
                say(f"{spec}: FAILED {type(exc).__name__}: {exc}")
                BAIDU_ROWS.append(dict(spec=spec, family="baidu_combined",
                                       method=method, sample="clean", timing="clean",
                                       start="2017-01-01", end="2019-03-31",
                                       fixed_effects=fe_kind, htrend=trend,
                                       expected_sign=-1, note=
                                       f"FAILED {type(exc).__name__}: {exc}"))

baidu_res = bh_tag(pd.DataFrame(BAIDU_ROWS))
baidu_res.to_csv(os.path.join(OUT, "baidu_spec_universe.csv"), index=False,
                 encoding="utf-8-sig")
baidu_sum = summary(baidu_res)
baidu_sum.to_csv(os.path.join(OUT, "baidu_spec_universe_summary.csv"),
                 index=False, encoding="utf-8-sig")

# Exposure-specific pretrend tests for the two transforms and FE choices.
PRE = []
pre = b[(b["m"] >= "2014-01-01") & (b["m"] <= "2017-12-31")].copy()
for method in ("asinh_index", "log1p_index"):
    for fe_kind in ("strict", "month"):
        d = pre.copy()
        d["y"] = (np.arcsinh(d["mean"]) if method == "asinh_index"
                  else np.log1p(d["mean"]))
        fe = "pref + provm" if fe_kind == "strict" else "pref + ym"
        m = pf.feols(f"y ~ Ht | {fe}", data=d, vcov={"CRV1": "prov_id"})
        PRE.append(dict(method=method, fixed_effects=fe_kind,
                        beta=float(m.coef()["Ht"]), se=float(m.se()["Ht"]),
                        p_crv1=float(m.pvalue()["Ht"]), n=int(m._N)))
pd.DataFrame(PRE).to_csv(os.path.join(OUT, "baidu_pretrend_tests.csv"),
                         index=False, encoding="utf-8-sig")
say("\n=== Baidu family summary ===")
say(baidu_sum.to_string(index=False))
say("Baidu mechanical claim-gate rows:")
bg = baidu_res[baidu_res["linear_claim_gate"]]
say(bg[["spec", "beta", "se", "p_wild", "q_family"]].to_string(index=False)
    if len(bg) else "none")
say("Baidu pretrends:")
say(pd.DataFrame(PRE).to_string(index=False))

with open(os.path.join(OUT, "offcourt_spec_universe_log.txt"), "w",
          encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
say(f"outputs -> {OUT}")

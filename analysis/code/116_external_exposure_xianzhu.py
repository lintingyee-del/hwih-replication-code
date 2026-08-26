# -*- coding: utf-8 -*-
"""Auditable bounded search for an exposure measure outside the court archive.

Research question and design are fixed before this script searches variants:

    Does a prefecture with greater pre-campaign private-collection capacity
    experience a larger post-inspection judicialization response?

Only the x side changes.  The three y-side estimands, clean window, support,
fixed effects, and province-clustered wild-score inference match the paper:

  1. acquaintance-minus-stranger lending composition (primary);
  2. balanced relational-cause flow (supporting);
  3. relational-minus-traffic flow (placebo-docket contrast).

The external x candidates are limited to transformations of two already
collected, pre-campaign sources:

  * the national firm registry (stock or entry density of collection firms);
  * Baidu search intensity for private debt-collection terms.

The script writes the candidate manifest before fitting any regression, retains
all failed specifications, and reports BH q-values within estimand-by-FE search
families.  It does not select a paper specification automatically.

Outputs:
  output/external_exposure_xianzhu/candidate_manifest.csv
  output/external_exposure_xianzhu/index_metadata.csv
  output/external_exposure_xianzhu/spec_log.csv
  output/external_exposure_xianzhu/spec_summary.csv
  output/external_exposure_xianzhu/run_log.txt
"""
from __future__ import annotations

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
import io
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from _wild import wild_score_p


BASE = str(_REP_PROJECT)
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output", "external_exposure_xianzhu")
REGISTRY = str(_REP_REGISTRY).replace('\\', '/')
BAIDU = str(_REP_BAIDU / 'baidu_index_city_month.csv').replace('\\', '/')

WINDOW = ("2017-01", "2019-03")
SUPPORT = ("2014-01", "2017-12")
POST0 = "2018-09"
os.makedirs(OUT, exist_ok=True)

LOG: list[str] = []


def say(*args) -> None:
    line = " ".join(str(x) for x in args)
    print(line, flush=True)
    LOG.append(line)


def zscore(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / sd


def bh_adjust(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment, preserving the original index."""
    p = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    good = p.notna()
    if not good.any():
        return out
    ordered = p[good].sort_values()
    m = len(ordered)
    raw = ordered.to_numpy() * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(raw[::-1])[::-1].clip(max=1.0)
    out.loc[ordered.index] = adj
    return out


# ---------------------------------------------------------------------------
# Stable city-to-prefecture crosswalk, identical to the registry/Baidu scripts.
# ---------------------------------------------------------------------------
court_xwalk = pd.read_parquet(os.path.join(DATA, "court_xwalk.parquet"))
city_pattern = re.compile(r"(?:.*?省|.*?自治区)?(.+?(?:市|州|盟|地区))中级人民法院$")
name_candidates: dict[str, list[str]] = {}
for _, row in court_xwalk.iterrows():
    match = city_pattern.match(str(row["court_name"]))
    if match:
        name_candidates.setdefault(match.group(1), []).append(row["prefecture_code"])
NAME2CODE = {
    name: max(set(codes), key=codes.count)
    for name, codes in name_candidates.items()
}
NAME2CODE.update(
    {"北京市": "110000", "天津市": "120000", "上海市": "310000", "重庆市": "500000"}
)


def city2code(city) -> str | None:
    value = str(city)
    if not value or value == "nan":
        return None
    candidate = value if value.endswith(("市", "州", "盟", "地区")) else value + "市"
    if candidate in NAME2CODE:
        return NAME2CODE[candidate]
    for name, code in NAME2CODE.items():
        if name[:-1] and name[:-1] in value:
            return code
    return None


exposure = pd.read_parquet(os.path.join(DATA, "exposure_v2.parquet"))[
    ["prefecture_code", "province", "exposure_v2_z"]
].copy()
say(f"[setup] court H_c support: {len(exposure)} prefectures")


@dataclass(frozen=True)
class Candidate:
    variant_id: str
    tier: str
    source: str
    base_variable: str
    transformation: str
    sample_rule: str
    construct_note: str


CANDIDATES = [
    # Natural primary definitions, fixed before construct variants.
    Candidate(
        "firm_all_stock_asinh_density",
        "primary",
        "firm_registry",
        "all positive-context collection firms alive at 2017-12-31",
        "z(asinh(stock / all 2014-17 firm entries * 1m))",
        "2014-17 registry denominator; zeros retained",
        "Predetermined registered collection-capacity stock.",
    ),
    Candidate(
        "baidu_combo_1417_asinh",
        "primary",
        "baidu",
        "monthly sum of 讨债 and 讨债公司",
        "z(asinh(2014-17 monthly mean))",
        "2014-01 to 2017-12; mapped cities",
        "Predetermined demand for private debt-collection services.",
    ),
    # Firm-registry construct and transformation checks.
    Candidate(
        "firm_all_entry_asinh_density",
        "secondary",
        "firm_registry",
        "all positive-context collection-firm entries",
        "z(asinh(2014-17 entries / all firm entries * 1m))",
        "2014-01 to 2017-12; zeros retained",
        "Flow analogue of the primary registry stock.",
    ),
    Candidate(
        "firm_all_stock_rank",
        "secondary",
        "firm_registry",
        "all positive-context collection firms alive at 2017-12-31",
        "z(percentile rank of stock density)",
        "2014-17 registry denominator; zeros retained",
        "Robust to concentration in the largest cities.",
    ),
    Candidate(
        "firm_vernacular_stock_asinh_density",
        "secondary",
        "firm_registry",
        "讨债/收数/商账 firms without licensed-credit context, alive at 2017-12-31",
        "z(asinh(stock / all 2014-17 firm entries * 1m))",
        "vernacular keywords; zeros retained",
        "Closer lexical match to informal collection than bank outsourcing.",
    ),
    Candidate(
        "firm_vernacular_entry_asinh_density",
        "secondary",
        "firm_registry",
        "2014-17 entries of 讨债/收数/商账 firms without licensed-credit context",
        "z(asinh(entries / all 2014-17 firm entries * 1m))",
        "vernacular keywords; zeros retained",
        "Entry-flow analogue of the vernacular stock.",
    ),
    Candidate(
        "firm_vernacular_stock_rank",
        "secondary",
        "firm_registry",
        "讨债/收数/商账 firms without licensed-credit context, alive at 2017-12-31",
        "z(percentile rank of stock density)",
        "vernacular keywords; zeros retained",
        "Concentration-robust vernacular stock.",
    ),
    Candidate(
        "firm_name_stock_asinh_density",
        "secondary",
        "firm_registry",
        "collection keyword in company name, alive at 2017-12-31",
        "z(asinh(stock / all 2014-17 firm entries * 1m))",
        "matched_field == 名称; zeros retained",
        "A name hit indicates a collection-centered business.",
    ),
    Candidate(
        "firm_name_entry_asinh_density",
        "secondary",
        "firm_registry",
        "2014-17 entries with a collection keyword in company name",
        "z(asinh(entries / all 2014-17 firm entries * 1m))",
        "matched_field == 名称; zeros retained",
        "Entry-flow analogue of name-centered collection firms.",
    ),
    Candidate(
        "firm_licensed_stock_asinh_density",
        "construct_placebo",
        "firm_registry",
        "bank/credit-card/NPL/factoring collection firms alive at 2017-12-31",
        "z(asinh(stock / all 2014-17 firm entries * 1m))",
        "licensed-credit context; zeros retained",
        "Measures formal outsourced collection, not the preferred coercive construct.",
    ),
    # Baidu temporal splits, term splits, and robust transformation.
    Candidate(
        "baidu_combo_1415_asinh",
        "secondary",
        "baidu",
        "monthly sum of 讨债 and 讨债公司",
        "z(asinh(2014-15 monthly mean))",
        "2014-01 to 2015-12; mapped cities",
        "Early-pre split-half definition.",
    ),
    Candidate(
        "baidu_combo_1617_asinh",
        "secondary",
        "baidu",
        "monthly sum of 讨债 and 讨债公司",
        "z(asinh(2016-17 monthly mean))",
        "2016-01 to 2017-12; mapped cities",
        "Late-pre split-half definition.",
    ),
    Candidate(
        "baidu_taozhai_1417_asinh",
        "secondary",
        "baidu",
        "讨债 search index",
        "z(asinh(2014-17 monthly mean))",
        "2014-01 to 2017-12; mapped cities",
        "Broad private debt-collection term.",
    ),
    Candidate(
        "baidu_company_1417_asinh",
        "secondary",
        "baidu",
        "讨债公司 search index",
        "z(asinh(2014-17 monthly mean))",
        "2014-01 to 2017-12; mapped cities",
        "Searches explicitly seeking a collection company.",
    ),
    Candidate(
        "baidu_combo_1417_rank",
        "secondary",
        "baidu",
        "monthly sum of 讨债 and 讨债公司",
        "z(percentile rank of 2014-17 monthly mean)",
        "2014-01 to 2017-12; mapped cities",
        "Robust to a small number of high-index cities.",
    ),
    Candidate(
        "baidu_shoushu_1417_asinh",
        "construct_placebo",
        "baidu",
        "收数公司 search index",
        "z(asinh(2014-17 monthly mean))",
        "2014-01 to 2017-12; mapped cities",
        "Regionally specific Cantonese collection term.",
    ),
]


manifest_rows = []
for candidate in CANDIDATES:
    for estimand in (
        "acquaintance_minus_stranger",
        "balanced_relational_flow",
        "relational_minus_traffic",
    ):
        for fe_spec in ("baseline", "province_month_saturated"):
            manifest_rows.append(
                {
                    "spec_id": f"{candidate.variant_id}__{estimand}__{fe_spec}",
                    "mode": "direct_experiment",
                    "focus_side": "x",
                    "variant_id": candidate.variant_id,
                    "tier": candidate.tier,
                    "source": candidate.source,
                    "base_variable": candidate.base_variable,
                    "transformation": candidate.transformation,
                    "sample_rule": candidate.sample_rule,
                    "construct_note": candidate.construct_note,
                    "estimand": estimand,
                    "model": "OLS with high-dimensional fixed effects",
                    "fixed_effects": fe_spec,
                    "controls": "paper clean-window lower-order interactions",
                }
            )
manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(
    os.path.join(OUT, "candidate_manifest.csv"), index=False, encoding="utf-8-sig"
)
say(f"[manifest] {len(CANDIDATES)} exposure variants x 6 outcome/FE cells")


# ---------------------------------------------------------------------------
# Construct firm-registry candidates.
# ---------------------------------------------------------------------------
say("\n=== Construct firm-registry candidates ===")
hits = pd.read_csv(os.path.join(REGISTRY, "hits_clean.csv"), dtype=str)
# The extraction headers are shifted: 所属省份 stores the prefecture-level city.
hits = hits.rename(columns={"所属城市": "district_raw"})
hits["city"] = hits["所属省份"]
hits["prefecture_code"] = hits["city"].map(city2code)
hits["entry_date"] = pd.to_datetime(hits["成立日期"], errors="coerce")
hits["exited"] = hits["经营状态"].str.contains("注销|吊销", na=False)
hits["exit_date"] = pd.to_datetime(hits["核准日期"], errors="coerce").where(
    hits["exited"]
)
text = hits["snippet"].fillna("") + " " + hits["企业名称"].fillna("")
licensed_pattern = re.compile(
    r"银行|信用卡|信贷|金融机构|持牌|委托|资产管理|不良资产|"
    r"保理|征信|应收账款|逾期户|贷后"
)
hits["licensed"] = text.str.contains(licensed_pattern)
hits["vernacular"] = (
    ~hits["licensed"] & hits["matched_kw"].isin(["讨债", "收数", "商账"])
)
hits["name_hit"] = hits["matched_field"].eq("名称")

agg_chunks = []
for path in sorted(glob.glob(os.path.join(REGISTRY, "agg", "*.csv"))):
    chunk = pd.read_csv(path, dtype={"city": str, "month": str})
    chunk = chunk[chunk["month"].between(SUPPORT[0], SUPPORT[1])]
    if len(chunk):
        agg_chunks.append(chunk)
all_firm = pd.concat(agg_chunks, ignore_index=True)
all_firm["prefecture_code"] = all_firm["city"].map(city2code)
denominator = (
    all_firm.dropna(subset=["prefecture_code"])
    .groupby("prefecture_code", as_index=False)
    .agg(all_entries_1417=("entries_all", "sum"))
)
denominator = denominator[denominator["all_entries_1417"] > 0].copy()

pre_start = pd.Timestamp("2014-01-01")
pre_end = pd.Timestamp("2017-12-31")


def registry_density(mask: pd.Series, kind: str) -> pd.DataFrame:
    selected = hits[mask & hits["prefecture_code"].notna()].copy()
    if kind == "stock":
        selected = selected[
            (selected["entry_date"] <= pre_end)
            & (~selected["exited"] | (selected["exit_date"] > pre_end))
        ]
    elif kind == "entry":
        selected = selected[
            selected["entry_date"].between(pre_start, pre_end, inclusive="both")
        ]
    else:
        raise ValueError(kind)
    counts = selected.groupby("prefecture_code").size().rename("count").reset_index()
    frame = denominator.merge(counts, on="prefecture_code", how="left")
    frame["count"] = frame["count"].fillna(0.0)
    frame["density"] = frame["count"] / frame["all_entries_1417"] * 1e6
    return frame


candidate_values = exposure[["prefecture_code", "province", "exposure_v2_z"]].copy()


def merge_candidate(variant_id: str, values: pd.DataFrame, raw_col: str) -> None:
    global candidate_values
    use = values[["prefecture_code", raw_col]].rename(columns={raw_col: variant_id})
    candidate_values = candidate_values.merge(use, on="prefecture_code", how="left")


registry_specs = {
    "firm_all_stock_asinh_density": (pd.Series(True, index=hits.index), "stock", "asinh"),
    "firm_all_entry_asinh_density": (pd.Series(True, index=hits.index), "entry", "asinh"),
    "firm_all_stock_rank": (pd.Series(True, index=hits.index), "stock", "rank"),
    "firm_vernacular_stock_asinh_density": (hits["vernacular"], "stock", "asinh"),
    "firm_vernacular_entry_asinh_density": (hits["vernacular"], "entry", "asinh"),
    "firm_vernacular_stock_rank": (hits["vernacular"], "stock", "rank"),
    "firm_name_stock_asinh_density": (hits["name_hit"], "stock", "asinh"),
    "firm_name_entry_asinh_density": (hits["name_hit"], "entry", "asinh"),
    "firm_licensed_stock_asinh_density": (hits["licensed"], "stock", "asinh"),
}
raw_registry: dict[str, pd.DataFrame] = {}
for variant_id, (mask, kind, transform) in registry_specs.items():
    frame = registry_density(mask, kind)
    raw_registry[variant_id] = frame.copy()
    if transform == "asinh":
        frame["index"] = zscore(np.arcsinh(frame["density"]))
    elif transform == "rank":
        frame["index"] = zscore(frame["density"].rank(method="average", pct=True))
    else:
        raise ValueError(transform)
    merge_candidate(variant_id, frame, "index")
    say(
        f"  {variant_id}: denominator prefectures={len(frame)}, "
        f"nonzero={int((frame['count'] > 0).sum())}, source firms={int(frame['count'].sum())}"
    )


# ---------------------------------------------------------------------------
# Construct Baidu candidates.
# ---------------------------------------------------------------------------
say("\n=== Construct Baidu candidates ===")
baidu = pd.read_csv(BAIDU, dtype={"ym": str, "city": str})
baidu["prefecture_code"] = baidu["city"].map(city2code)
baidu = baidu.dropna(subset=["prefecture_code"]).copy()


def baidu_level(keywords: list[str], start: str, end: str) -> pd.DataFrame:
    part = baidu[
        baidu["ym"].between(start, end) & baidu["keyword"].isin(keywords)
    ].copy()
    monthly = (
        part.groupby(["prefecture_code", "ym"], as_index=False)["mean"].sum()
    )
    return (
        monthly.groupby("prefecture_code", as_index=False)["mean"]
        .mean()
        .rename(columns={"mean": "level"})
    )


baidu_specs = {
    "baidu_combo_1417_asinh": (["讨债", "讨债公司"], "2014-01", "2017-12", "asinh"),
    "baidu_combo_1415_asinh": (["讨债", "讨债公司"], "2014-01", "2015-12", "asinh"),
    "baidu_combo_1617_asinh": (["讨债", "讨债公司"], "2016-01", "2017-12", "asinh"),
    "baidu_taozhai_1417_asinh": (["讨债"], "2014-01", "2017-12", "asinh"),
    "baidu_company_1417_asinh": (["讨债公司"], "2014-01", "2017-12", "asinh"),
    "baidu_combo_1417_rank": (["讨债", "讨债公司"], "2014-01", "2017-12", "rank"),
    "baidu_shoushu_1417_asinh": (["收数公司"], "2014-01", "2017-12", "asinh"),
}
raw_baidu: dict[str, pd.DataFrame] = {}
for variant_id, (keywords, start, end, transform) in baidu_specs.items():
    frame = baidu_level(keywords, start, end)
    raw_baidu[variant_id] = frame.copy()
    if transform == "asinh":
        frame["index"] = zscore(np.arcsinh(frame["level"]))
    elif transform == "rank":
        frame["index"] = zscore(frame["level"].rank(method="average", pct=True))
    else:
        raise ValueError(transform)
    merge_candidate(variant_id, frame, "index")
    say(
        f"  {variant_id}: prefectures={len(frame)}, "
        f"nonzero={int((frame['level'] > 0).sum())}"
    )


# Index diagnostics: coverage, within-province variation, and agreement with H_c.
metadata_rows = []
candidate_lookup = {candidate.variant_id: candidate for candidate in CANDIDATES}
for variant_id, candidate in candidate_lookup.items():
    d = candidate_values.dropna(subset=[variant_id, "exposure_v2_z", "province"]).copy()
    if len(d) < 20:
        continue
    raw_r, raw_p = stats.pearsonr(d[variant_id], d["exposure_v2_z"])
    xw = d[variant_id] - d.groupby("province")[variant_id].transform("mean")
    hw = d["exposure_v2_z"] - d.groupby("province")["exposure_v2_z"].transform("mean")
    within_r, within_p = stats.pearsonr(xw, hw)
    total_var = d[variant_id].var(ddof=1)
    within_var = xw.var(ddof=1)
    metadata_rows.append(
        {
            "variant_id": variant_id,
            "tier": candidate.tier,
            "source": candidate.source,
            "n_prefectures": len(d),
            "province_clusters": d["province"].nunique(),
            "raw_corr_Hc": raw_r,
            "raw_corr_Hc_p": raw_p,
            "within_province_corr_Hc": within_r,
            "within_province_corr_Hc_p": within_p,
            "within_variance_share": within_var / total_var if total_var else np.nan,
        }
    )
metadata = pd.DataFrame(metadata_rows)
metadata.to_csv(
    os.path.join(OUT, "index_metadata.csv"), index=False, encoding="utf-8-sig"
)
candidate_values.to_csv(
    os.path.join(OUT, "candidate_values.csv"), index=False, encoding="utf-8-sig"
)


# ---------------------------------------------------------------------------
# Build the paper's three fixed y-side panels.
# ---------------------------------------------------------------------------
say("\n=== Build fixed outcome panels ===")
schedule = (
    pd.read_parquet(os.path.join(DATA, "panel_month.parquet"))[
        ["province", "inspection_round"]
    ]
    .drop_duplicates()
)
months = pd.DataFrame(
    {"month": pd.period_range(WINDOW[0], WINDOW[1], freq="M").astype(str)}
)
civil_panel = pd.read_parquet(os.path.join(DATA, "civil_panel.parquet"))
civil_panel["month"] = civil_panel["jmonth"].astype(str).str[:7]

# Balanced relational-cause flow.
relational = civil_panel[civil_panel["cause_family"].eq("relational")].copy()
flow_support = relational[relational["month"].between(SUPPORT[0], SUPPORT[1])][
    ["prefecture_code", "province", "cause"]
].drop_duplicates()
flow_counts = relational[relational["month"].between(WINDOW[0], WINDOW[1])][
    ["prefecture_code", "province", "cause", "month", "n_cases"]
]
flow = (
    flow_support.merge(months, how="cross")
    .merge(
        flow_counts,
        on=["prefecture_code", "province", "cause", "month"],
        how="left",
    )
    .merge(schedule, on="province")
)
flow["n"] = flow["n_cases"].fillna(0.0)
flow["y"] = np.arcsinh(flow["n"])
flow["pref_cause"] = flow["prefecture_code"] + "_" + flow["cause"]
flow["cause_month"] = flow["cause"] + "_" + flow["month"]

# Balanced acquaintance/stranger cells among classified lending cases.
case = pd.read_parquet(
    os.path.join(DATA, "civil_case.parquet"),
    columns=["cause", "prefecture_code", "province", "jmonth", "rel_txn"],
)
case["month"] = case["jmonth"].astype(str).str[:7]
lending = case[
    case["cause"].eq("民间借贷纠纷") & case["rel_txn"].notna()
].copy()
lending["acq"] = lending["rel_txn"].astype(int)
composition_support = lending[
    lending["month"].between(SUPPORT[0], SUPPORT[1])
][["prefecture_code", "province"]].drop_duplicates()
composition_counts = (
    lending[lending["month"].between(WINDOW[0], WINDOW[1])]
    .groupby(["prefecture_code", "province", "month", "acq"])
    .size()
    .rename("n")
    .reset_index()
)
composition = (
    composition_support.merge(months, how="cross")
    .merge(pd.DataFrame({"acq": [0, 1]}), how="cross")
    .merge(
        composition_counts,
        on=["prefecture_code", "province", "month", "acq"],
        how="left",
    )
    .merge(schedule, on="province")
)
composition["n"] = composition["n"].fillna(0.0)
composition["y"] = np.arcsinh(composition["n"])
composition["prefA"] = (
    composition["prefecture_code"] + "_" + composition["acq"].astype(str)
)
composition["monthA"] = composition["month"] + "_" + composition["acq"].astype(str)

# Relational-minus-traffic aggregate gap, matching code/82_clean_traffic_gap.py.
rt = civil_panel[civil_panel["month"].between(WINDOW[0], WINDOW[1])].copy()
rt["group"] = np.where(
    rt["cause_family"].eq("relational"),
    "relational",
    np.where(rt["cause_family"].eq("placebo"), "traffic", "other"),
)
rt = rt[rt["group"].isin(["relational", "traffic"])]
gap = (
    rt.groupby(["prefecture_code", "province", "month", "group"], as_index=False)[
        "n_cases"
    ]
    .sum()
    .pivot_table(
        index=["prefecture_code", "province", "month"],
        columns="group",
        values="n_cases",
        fill_value=0,
    )
    .reset_index()
    .merge(schedule, on="province")
)
gap["y"] = np.arcsinh(gap["relational"]) - np.arcsinh(gap["traffic"])

say(
    f"  flow={len(flow):,}; composition={len(composition):,}; gap={len(gap):,}"
)


def add_design(frame: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    keep = candidate_values[["prefecture_code", variant_id]].dropna()
    d = frame.merge(keep, on="prefecture_code").copy()
    d["H"] = d[variant_id]
    d["treat"] = (d["inspection_round"] == 1).astype(int)
    d["postc"] = (d["month"] >= POST0).astype(int)
    d["pt"] = d["postc"] * d["treat"]
    d["ph"] = d["postc"] * d["H"]
    d["pth"] = d["pt"] * d["H"]
    d["prov_id"] = pd.factorize(d["province"])[0]
    d["prov_month"] = d["province"] + "_" + d["month"]
    return d


FORMULAS = {
    ("balanced_relational_flow", "baseline"): (
        "y ~ pth + ph + pt | pref_cause + month",
        "pth",
    ),
    ("balanced_relational_flow", "province_month_saturated"): (
        "y ~ pth + ph | pref_cause + prov_month + cause_month",
        "pth",
    ),
    ("acquaintance_minus_stranger", "baseline"): (
        "y ~ pthA + phA + ptA + pth + ph + pt | prefA + monthA",
        "pthA",
    ),
    ("acquaintance_minus_stranger", "province_month_saturated"): (
        "y ~ pthA + phA + ptA + pth + ph | prefA + prov_month + monthA",
        "pthA",
    ),
    ("relational_minus_traffic", "baseline"): (
        "y ~ pth + ph + pt | prefecture_code + month",
        "pth",
    ),
    ("relational_minus_traffic", "province_month_saturated"): (
        "y ~ pth + ph | prefecture_code + prov_month",
        "pth",
    ),
}


def prepare_outcome(
    estimand: str, fe_spec: str, variant_id: str
) -> tuple[pd.DataFrame, str, str]:
    if estimand == "balanced_relational_flow":
        d = add_design(flow, variant_id)
    elif estimand == "acquaintance_minus_stranger":
        d = add_design(composition, variant_id)
        for term in ("pth", "ph", "pt"):
            d[f"{term}A"] = d[term] * d["acq"]
    elif estimand == "relational_minus_traffic":
        d = add_design(gap, variant_id)
    else:
        raise ValueError(estimand)
    formula, coefficient = FORMULAS[(estimand, fe_spec)]
    return d, formula, coefficient


def run_fit(
    spec_row: pd.Series,
) -> dict:
    variant_id = spec_row["variant_id"]
    estimand = spec_row["estimand"]
    fe_spec = spec_row["fixed_effects"]
    d, formula, coefficient = prepare_outcome(estimand, fe_spec, variant_id)
    result = spec_row.to_dict()
    result.update(
        {
            "formula": formula,
            "coefficient_name": coefficient,
            "n_prefectures": d["prefecture_code"].nunique(),
            "province_clusters": d["prov_id"].nunique(),
            "direction": "not_estimated",
            "keep_or_drop": "pending_review",
            "reason": "",
            "error": "",
        }
    )
    try:
        model = pf.feols(formula, data=d, vcov={"CRV1": "prov_id"})
        estimate = float(model.coef()[coefficient])
        se = float(model.se()[coefficient])
        p_crv1 = float(model.pvalue()[coefficient])
        p_wild = float(wild_score_p(formula, d, coefficient))
        result.update(
            {
                "coefficient": estimate,
                "std_error": se,
                "p_crv1": p_crv1,
                "p_wild": p_wild,
                "n_obs": int(model._N),
                "direction": "positive" if estimate > 0 else "negative",
            }
        )
    except Exception as exc:  # retain every failure in the audit log
        result.update(
            {
                "coefficient": np.nan,
                "std_error": np.nan,
                "p_crv1": np.nan,
                "p_wild": np.nan,
                "n_obs": np.nan,
                "direction": "failed",
                "keep_or_drop": "drop",
                "reason": "estimation failure",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return result


say("\n=== Run fixed regression family ===")
results = []
for variant_id in [candidate.variant_id for candidate in CANDIDATES]:
    say(f"  fitting {variant_id}")
    rows = manifest[manifest["variant_id"].eq(variant_id)]
    for _, row in rows.iterrows():
        fitted = run_fit(row)
        results.append(fitted)
        if fitted["direction"] == "failed":
            say(f"    FAILED {fitted['spec_id']}: {fitted['error']}")
        else:
            say(
                f"    {fitted['estimand']:29s} {fitted['fixed_effects']:28s} "
                f"b={fitted['coefficient']:+.4f} se={fitted['std_error']:.4f} "
                f"wild={fitted['p_wild']:.4f}"
            )

spec_log = pd.DataFrame(results)
spec_log["bh_q_all_variants"] = np.nan
for _, indices in spec_log.groupby(["estimand", "fixed_effects"]).groups.items():
    spec_log.loc[indices, "bh_q_all_variants"] = bh_adjust(
        spec_log.loc[indices, "p_wild"]
    )

# A separate q-value for the two natural primary x definitions only.
spec_log["bh_q_primary_variants"] = np.nan
primary = spec_log["tier"].eq("primary")
for _, indices in spec_log[primary].groupby(["estimand", "fixed_effects"]).groups.items():
    spec_log.loc[indices, "bh_q_primary_variants"] = bh_adjust(
        spec_log.loc[indices, "p_wild"]
    )

# Mechanical labels aid review but never select solely on significance.
ok = spec_log["error"].eq("")
spec_log.loc[ok & (spec_log["tier"] == "primary"), "keep_or_drop"] = (
    "primary_candidate"
)
spec_log.loc[ok & (spec_log["tier"] == "secondary"), "keep_or_drop"] = (
    "exploratory_only"
)
spec_log.loc[ok & (spec_log["tier"] == "construct_placebo"), "keep_or_drop"] = (
    "construct_placebo"
)
spec_log.loc[ok, "reason"] = (
    "Retain in the complete search log; paper use requires construct and robustness review."
)
spec_log.to_csv(os.path.join(OUT, "spec_log.csv"), index=False, encoding="utf-8-sig")

summary = (
    spec_log.pivot_table(
        index=["variant_id", "tier", "source"],
        columns=["estimand", "fixed_effects"],
        values=["coefficient", "p_wild", "bh_q_all_variants"],
        aggfunc="first",
    )
    .sort_index()
)
summary.columns = ["__".join(map(str, col)) for col in summary.columns]
summary.reset_index().to_csv(
    os.path.join(OUT, "spec_summary.csv"), index=False, encoding="utf-8-sig"
)

with open(os.path.join(OUT, "run_log.txt"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(LOG))

say("\n=== Natural primary external definitions ===")
display_cols = [
    "variant_id",
    "estimand",
    "fixed_effects",
    "coefficient",
    "std_error",
    "p_wild",
    "bh_q_primary_variants",
    "n_obs",
]
say(
    spec_log[spec_log["tier"].eq("primary")][display_cols]
    .round(5)
    .to_string(index=False)
)
say("\nDONE ->", OUT)

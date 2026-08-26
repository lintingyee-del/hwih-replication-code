# -*- coding: utf-8 -*-
"""Design-based and segmented inference for national off-court series.

Part A compares the collection-firm counts with all firm entries/exits in the
same registry.  A fixed 2018Q1 segmented regression includes a linear trend,
quarter seasonality, and Newey-West HAC(4) inference.  Candidate break dates are
reported rather than using the best date as the test.

Part B estimates male-minus-female log mortality gaps within the six
East/Central/West by urban/rural strata.  A 2014--2017 stratum-specific trend is
projected into 2018--2020, and exact sign-flip p-values enumerate all 2^6
assignments.  The one-sided alternative is the model-predicted decline; the
two-sided value is retained alongside it.
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
import itertools
import os
import sys

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.stdout.reconfigure(encoding="utf-8")
BASE = str(_REP_PROJECT)
COLL_OUT = os.path.join(BASE, "output", "collection_firms")
CDC_OUT = os.path.join(BASE, "output", "cdc_homicide")
REGISTRY_GLOB = os.environ.get(
    "HWIH_REGISTRY_AGG_GLOB",
    str(_REP_PROJECT / "data" / "derived" / "registry_aggregate" / "*.csv").replace("\\", "/"),
).replace("\\", "/")
os.makedirs(COLL_OUT, exist_ok=True)
os.makedirs(CDC_OUT, exist_ok=True)


def segmented(data, coll_col, all_col, break_q):
    z = data.copy().reset_index(drop=True)
    z["t"] = np.arange(len(z), dtype=float)
    break_idx = int(z.index[z["quarter"] == pd.Period(break_q, freq="Q")][0])
    z["post"] = (z.index >= break_idx).astype(int)
    z["post_slope"] = np.maximum(z.index - break_idx, 0).astype(float)
    z["y"] = np.log(z[coll_col] / z[all_col])
    seasons = pd.get_dummies(z["quarter"].dt.quarter, prefix="q",
                             drop_first=True, dtype=float)
    X = pd.concat([z[["t", "post", "post_slope"]], seasons], axis=1)
    X = sm.add_constant(X)
    base = sm.OLS(z["y"], X).fit()
    hac = base.get_robustcov_results(cov_type="HAC", maxlags=4,
                                     use_correction=True, use_t=False)
    names = list(base.params.index)
    params = pd.Series(hac.params, index=names)
    ses = pd.Series(hac.bse, index=names)
    pvals = pd.Series(hac.pvalues, index=names)
    R = np.zeros((2, len(names)))
    R[0, names.index("post")] = 1
    R[1, names.index("post_slope")] = 1
    joint = hac.wald_test(R, scalar=True)
    return dict(level=float(params["post"]), level_se=float(ses["post"]),
                level_p=float(pvals["post"]),
                slope=float(params["post_slope"]),
                slope_se=float(ses["post_slope"]),
                slope_p=float(pvals["post_slope"]),
                joint_p=float(joint.pvalue), n=len(z))


# Part A: full national numerator and denominator from the same registry.
con = duckdb.connect()
all_q = con.sql(f"""
  SELECT date_trunc('quarter', TRY_CAST(month || '-01' AS DATE)) AS qdate,
         SUM(entries_all)::DOUBLE AS entries_all,
         SUM(exits_all)::DOUBLE AS exits_all
  FROM read_csv_auto('{REGISTRY_GLOB}',
                     union_by_name=true, header=true)
  WHERE TRY_CAST(month || '-01' AS DATE) BETWEEN DATE '2013-01-01' AND DATE '2022-12-31'
  GROUP BY 1 ORDER BY 1
""").df()
con.close()
all_q["quarter"] = pd.to_datetime(all_q["qdate"]).dt.to_period("Q")
coll = pd.read_csv(os.path.join(COLL_OUT, "national_quarterly.csv"))
coll["quarter"] = pd.PeriodIndex(coll["quarter"], freq="Q")
q = (all_q[["quarter", "entries_all", "exits_all"]]
     .merge(coll[["quarter", "entries", "exits"]], on="quarter", how="inner"))
q = q[(q["quarter"] >= pd.Period("2013Q1"))
      & (q["quarter"] <= pd.Period("2022Q4"))].copy()
q["entry_share_per_million"] = q["entries"] / q["entries_all"] * 1_000_000
q["exit_share_per_million"] = q["exits"] / q["exits_all"] * 1_000_000
q.to_csv(os.path.join(COLL_OUT, "national_registry_relative_quarterly.csv"),
         index=False, encoding="utf-8-sig")

annual = q.assign(year=q["quarter"].dt.year).groupby("year", as_index=False).agg(
    entries_all=("entries_all", "sum"), exits_all=("exits_all", "sum"),
    collection_entries=("entries", "sum"), collection_exits=("exits", "sum"))
annual["entry_share_per_million"] = (
    annual["collection_entries"] / annual["entries_all"] * 1_000_000)
annual["exit_share_per_million"] = (
    annual["collection_exits"] / annual["exits_all"] * 1_000_000)
annual.to_csv(os.path.join(COLL_OUT, "national_registry_relative_annual.csv"),
              index=False, encoding="utf-8-sig")

seg_rows = []
for label, coll_col, all_col in [
    ("entry_share", "entries", "entries_all"),
    ("exit_share", "exits", "exits_all"),
]:
    rec = segmented(q, coll_col, all_col, "2018Q1")
    rec.update(outcome=label, break_quarter="2018Q1", hac_lags=4,
               covariance="Newey-West with small-sample covariance correction")
    seg_rows.append(rec)
seg = pd.DataFrame(seg_rows)
seg.to_csv(os.path.join(COLL_OUT, "national_registry_segmented.csv"),
           index=False, encoding="utf-8-sig")

# At least eight quarters on either side gives 25 candidate dates.
break_rows = []
for idx in range(8, len(q) - 7):
    bq = str(q.iloc[idx]["quarter"])
    for label, coll_col, all_col in [
        ("entry_share", "entries", "entries_all"),
        ("exit_share", "exits", "exits_all"),
    ]:
        rec = segmented(q, coll_col, all_col, bq)
        break_rows.append(dict(outcome=label, break_quarter=bq, **rec))
breaks = pd.DataFrame(break_rows)
breaks["joint_rank"] = (breaks.groupby("outcome")["joint_p"]
                         .rank(method="min", ascending=True).astype(int))
breaks.to_csv(os.path.join(COLL_OUT, "national_registry_break_ranks.csv"),
              index=False, encoding="utf-8-sig")

print("== same-registry national segmented results ==", flush=True)
print(seg.to_string(index=False), flush=True)
print("2018Q1 break ranks:", flush=True)
print(breaks[breaks["break_quarter"] == "2018Q1"]
      [["outcome", "joint_p", "joint_rank"]].to_string(index=False), flush=True)


# Part B: exact six-stratum mortality contrast.
cdc = pd.read_csv(os.path.join(CDC_OUT, "cdc_homicide_panel.csv"))
cdc = cdc[cdc["region"].isin(["东部", "中部", "西部"])
          & cdc["urbrur"].isin(["城市", "农村"])
          & cdc["sex"].isin(["男性", "女性"])].copy()


def stratum_deviation(outcome, post_end):
    wide = (cdc.pivot_table(index=["region", "urbrur", "year"],
                            columns="sex", values=outcome).reset_index())
    wide["gap"] = np.log(wide["男性"]) - np.log(wide["女性"])
    rows = []
    for (region, urbrur), group in wide.groupby(["region", "urbrur"]):
        pre = group[(group["year"] >= 2014) & (group["year"] <= 2017)]
        slope, intercept = np.polyfit(pre["year"], pre["gap"], 1)
        post = group[(group["year"] >= 2018) & (group["year"] <= post_end)].copy()
        post["deviation"] = post["gap"] - (intercept + slope * post["year"])
        rows.append(dict(region=region, urbrur=urbrur,
                         deviation=float(post["deviation"].mean())))
    return pd.DataFrame(rows)


def exact_signflip(values):
    values = np.asarray(values, dtype=float)
    observed = values.mean()
    null = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(values)):
        null.append((np.asarray(signs) * values).mean())
    null = np.asarray(null)
    one_sided = float(np.mean(null <= observed + 1e-12))
    two_sided = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return one_sided, two_sided


def exact_row(label, values, post_end):
    values = np.asarray(values, dtype=float)
    mean_log = float(values.mean())
    one_sided, two_sided = exact_signflip(values)
    return dict(test=label, post_window=f"2018-{post_end}", strata=len(values),
                negative_strata=int((values < 0).sum()), mean_log_deviation=mean_log,
                percent_deviation=float(np.expm1(mean_log) * 100),
                exact_one_sided_p=one_sided,
                exact_two_sided_p=two_sided)


exact_rows = []
strata_rows = []
for post_end in (2020, 2021):
    hom = stratum_deviation("homicide_15_59_rate", post_end)
    sui = stratum_deviation("suicide_rate", post_end)
    tra = stratum_deviation("traffic_acc_rate", post_end)
    for label, frame in [("adult_male_female_homicide_gap", hom),
                         ("male_female_suicide_gap", sui),
                         ("male_female_traffic_gap", tra)]:
        exact_rows.append(exact_row(label, frame["deviation"], post_end))
        temp = frame.assign(test=label, post_window=f"2018-{post_end}")
        strata_rows.append(temp)
    triple = hom["deviation"].to_numpy() - (
        sui["deviation"].to_numpy() + tra["deviation"].to_numpy()) / 2
    exact_rows.append(exact_row("homicide_minus_mean_comparators", triple, post_end))
    temp = hom[["region", "urbrur"]].copy()
    temp["deviation"] = triple
    temp["test"] = "homicide_minus_mean_comparators"
    temp["post_window"] = f"2018-{post_end}"
    strata_rows.append(temp)

exact = pd.DataFrame(exact_rows)
exact.to_csv(os.path.join(CDC_OUT, "cdc_exact_tests.csv"), index=False,
             encoding="utf-8-sig")
pd.concat(strata_rows, ignore_index=True).to_csv(
    os.path.join(CDC_OUT, "cdc_exact_stratum_deviations.csv"), index=False,
    encoding="utf-8-sig")
print("\n== CDC exact six-stratum tests ==", flush=True)
print(exact.to_string(index=False), flush=True)

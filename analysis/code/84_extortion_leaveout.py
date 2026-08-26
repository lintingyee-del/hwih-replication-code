# -*- coding: utf-8 -*-
"""Extortion classification and leave-one-out exposure checks.

The published enforcement-family count combines four core enforcement offenses
with extortion and mafia-organization cases.  This diagnostic changes only the
classification of extortion and, where extortion is an outcome, removes it from
the pre-campaign exposure index.  It does not patch the manuscript.
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

from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pyfixest as pf


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"
EXT = Path(str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020"))
REPS = 9_999
SEED = 42

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wild import wild_score_p  # noqa: E402


con = duckdb.connect()
con.sql("SET threads TO 10; SET memory_limit='20GB'")
con.sql(f"CREATE OR REPLACE TABLE xwalk AS SELECT * FROM '{DATA / 'court_xwalk.parquet'}'")
con.sql(f"""
CREATE OR REPLACE TABLE crim0 AS
SELECT c.crime, c.d_backstop_collection,
       x.prefecture_code, x.province,
       TRY_CAST(c.judgment_date AS DATE) AS jdate
FROM read_parquet('{EXT.as_posix()}/crim_*.parquet') c
LEFT JOIN xwalk x ON c.court = x.court_name
WHERE x.prefecture_code IS NOT NULL
  AND TRY_CAST(c.judgment_date AS DATE) BETWEEN DATE '2014-01-01' AND DATE '2020-12-31'
""")

pre = con.sql("""
SELECT prefecture_code, province, COUNT(*) AS n_pre,
       AVG((crime IN ('非法拘禁','寻衅滋事','聚众斗殴','强迫交易',
                      '组织、领导、参加黑社会性质组织'))::INT) AS violent_share_no_extortion,
       AVG(d_backstop_collection) FILTER (WHERE crime <> '敲诈勒索')
           AS backstop_collect_rate_no_extortion
FROM crim0
WHERE jdate BETWEEN DATE '2014-01-01' AND DATE '2017-12-31'
GROUP BY 1,2
HAVING COUNT(*) >= 300
""").df()

counts = con.sql("""
SELECT prefecture_code, province,
       date_trunc('month', jdate) AS jmonth,
       SUM((crime IN ('非法拘禁','寻衅滋事','聚众斗殴','强迫交易'))::INT) AS n_core,
       SUM((crime = '敲诈勒索')::INT) AS n_extortion,
       SUM((crime = '组织、领导、参加黑社会性质组织')::INT) AS n_mafia
FROM crim0
GROUP BY 1,2,3
""").df()
con.close()


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


pre["H_no_extortion"] = (
    zscore(pre["violent_share_no_extortion"])
    + zscore(pre["backstop_collect_rate_no_extortion"])
) / 2.0

published = pd.read_parquet(DATA / "exposure_v2.parquet")[[
    "prefecture_code", "exposure_v2_z"
]]
panel = pd.read_parquet(DATA / "crim_panel_v2.parquet")
schedule = (panel[[
    "prefecture_code", "province", "insp_month"
]].drop_duplicates("prefecture_code"))

meta = (published.merge(schedule, on="prefecture_code", how="inner")
        .merge(pre[["prefecture_code", "H_no_extortion"]],
               on="prefecture_code", how="left"))
months = pd.DataFrame({
    "jmonth": pd.date_range(pd.to_datetime(panel["jmonth"]).min(),
                            pd.to_datetime(panel["jmonth"]).max(), freq="MS")
})
d = (meta.assign(_k=1).merge(months.assign(_k=1), on="_k").drop(columns="_k")
     .merge(counts.drop(columns="province"),
            on=["prefecture_code", "jmonth"], how="left"))
d[["n_core", "n_extortion", "n_mafia"]] = d[[
    "n_core", "n_extortion", "n_mafia"
]].fillna(0.0)
d["jmonth"] = pd.to_datetime(d["jmonth"])
d["insp_month"] = pd.to_datetime(d["insp_month"])
d["month"] = d["jmonth"].dt.to_period("M").astype(str)
d["post"] = (d["jmonth"] >= d["insp_month"]).astype(int)
d["pref"] = d["prefecture_code"].astype(str)
d["prov_month"] = d["province"] + "_" + d["month"]
d["prov_id"] = pd.factorize(d["province"])[0]
d["n_current"] = d["n_core"] + d["n_extortion"] + d["n_mafia"]
d["n_without_extortion"] = d["n_core"] + d["n_mafia"]


rows = []


def run(tag: str, count_col: str, exposure_col: str, definition: str,
        complete_support: bool = False) -> None:
    x = d.copy() if complete_support else d[d[count_col] > 0].copy()
    x["asinh_n"] = np.arcsinh(x[count_col])
    x["H"] = x[exposure_col]
    x["px"] = x["post"] * x["H"]
    fml = "asinh_n ~ px | pref + prov_month"
    m = pf.feols(fml, data=x, vcov={"CRV1": "prov_id"})
    wp = wild_score_p(fml, x, "px", cluster="prov_id", reps=REPS, seed=SEED)
    row = {
        "specification": tag,
        "outcome_definition": definition,
        "exposure_definition": exposure_col,
        "sample_support": "complete prefecture-month panel" if complete_support
                          else "positive-docket prefecture-months",
        "coefficient": float(m.coef()["px"]),
        "std_error": float(m.se()["px"]),
        "p_value": float(m.pvalue()["px"]),
        "wild_p": wp,
        "n_obs": int(m._N),
    }
    rows.append(row)
    print(
        f"{tag:38s} b={row['coefficient']:+.5f} "
        f"se={row['std_error']:.5f} p={row['p_value']:.3f} "
        f"wild={wp:.3f} N={row['n_obs']:,}",
        flush=True,
    )


run(
    "published_family_published_exposure",
    "n_current", "exposure_v2_z",
    "core enforcement + extortion + mafia organization",
)
run(
    "published_family_complete_support",
    "n_current", "exposure_v2_z",
    "core enforcement + extortion + mafia organization",
    complete_support=True,
)
run(
    "exclude_extortion_published_exposure",
    "n_without_extortion", "exposure_v2_z",
    "core enforcement + mafia organization; extortion excluded",
)
run(
    "exclude_extortion_published_exposure_complete",
    "n_without_extortion", "exposure_v2_z",
    "core enforcement + mafia organization; extortion excluded",
    complete_support=True,
)
run(
    "published_family_leaveout_exposure",
    "n_current", "H_no_extortion",
    "core enforcement + extortion + mafia organization",
)
run(
    "exclude_extortion_complete_support",
    "n_without_extortion", "H_no_extortion",
    "core enforcement + mafia organization; extortion excluded",
    complete_support=True,
)
run(
    "exclude_extortion_leaveout_exposure",
    "n_without_extortion", "H_no_extortion",
    "core enforcement + mafia organization; extortion excluded",
)
run(
    "core_enforcement_leaveout_exposure",
    "n_core", "H_no_extortion",
    "illegal detention + picking quarrels + affray + forced transactions",
)
run(
    "core_enforcement_complete_support",
    "n_core", "H_no_extortion",
    "illegal detention + picking quarrels + affray + forced transactions",
    complete_support=True,
)
run(
    "extortion_only_complete_support",
    "n_extortion", "H_no_extortion",
    "extortion only",
    complete_support=True,
)
run(
    "extortion_only_leaveout_exposure",
    "n_extortion", "H_no_extortion",
    "extortion only",
)
run(
    "extortion_only_published_exposure",
    "n_extortion", "exposure_v2_z",
    "extortion only",
)
run(
    "extortion_only_published_exposure_complete",
    "n_extortion", "exposure_v2_z",
    "extortion only",
    complete_support=True,
)

path = OUT / "extortion_leaveout.csv"
pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
print(f"[done] wrote {path}")

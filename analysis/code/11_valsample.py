# -*- coding: utf-8 -*-
"""6B step 11 — stratified validation samples + raw-text fetch for LLM gold labels.

Design (protocol docs/civil_data_spec.md §4):
  12 anchor months (6 pre: 2015-2017, 6 post: 2019-2020) to bound raw-CSV rescans.
  Per indicator: ~500 cases, stratified positive/negative x pre/post.
  Text: first 2600 chars of 全文 (enough for fact-pattern judgment).
Outputs: output/validation/batches/<task>_<k>.jsonl  (12 items per batch)
         output/validation/frame_<task>.parquet      (sample + regex flags)
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
import duckdb, os, json, glob, random

EXT = str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020")
BASE = str(_REP_JUDGMENTS)
VAL = str(_REP_PROJECT / "output" / "validation")
os.makedirs(f"{VAL}/batches", exist_ok=True)
con = duckdb.connect(); con.sql("SET threads TO 10; SET memory_limit='20GB'")
random.seed(42)

PRE = ["2015_04", "2015_10", "2016_04", "2016_10", "2017_04", "2017_10"]
POST = ["2019_04", "2019_10", "2020_03", "2020_06", "2020_07", "2020_10"]
MONTHS = PRE + POST

# task -> (source kind, flag column, positives target, negatives target)
TASKS = {
    "civ_rel_txn":       ("civil", "rel_txn", 250, 250),
    "civ_rel_fail":      ("civil", "rel_fail", 250, 250),
    "civ_evidence":      ("civil", "evid_iou", 250, 250),
    "civ_backstop":      ("civil", "backstop_collection", 250, 250),
    "civ_rate_orig":     ("civil", "rate_nonnull", 300, 200),
    "crim_backstop":     ("crim", "d_backstop", 250, 250),
    "crim_detention":    ("crim", "detention_debt", 250, 250),
    "crim_fraudsplit":   ("crim", "fraud_telecom", 250, 250),
    "crim_offensedate":  ("crim", "offense_any", 350, 150),
}

frames = {}
for task, (kind, col, npos, nneg) in TASKS.items():
    parts = []
    for m in MONTHS:
        f = f"{EXT}/{kind}_{m}.parquet"
        if not os.path.exists(f): continue
        if col == "rate_nonnull":
            sel = ("rate_月pct IS NOT NULL OR rate_月分 IS NOT NULL", "1=1")
            q = f"""SELECT case_no, '{m}' AS ym,
                    (rate_月pct IS NOT NULL OR rate_月分 IS NOT NULL)::INT AS flag,
                    rate_月pct, rate_月分, rate_年pct, orig_year
                    FROM '{f}' WHERE cause='民间借贷纠纷' USING SAMPLE 8000 ROWS"""
        elif col == "offense_any":
            q = f"""SELECT case_no, '{m}' AS ym,
                    (offense_year_1 IS NOT NULL OR offense_year_2 IS NOT NULL)::INT AS flag,
                    offense_year_1, offense_year_2, crime
                    FROM '{f}' USING SAMPLE 8000 ROWS"""
        elif kind == "civil":
            q = f"""SELECT case_no, '{m}' AS ym, {col} AS flag, cause
                    FROM '{f}' USING SAMPLE 8000 ROWS"""
        else:
            q = f"""SELECT case_no, '{m}' AS ym, {col} AS flag, crime
                    FROM '{f}' WHERE {col} IS NOT NULL USING SAMPLE 8000 ROWS"""
        parts.append(con.sql(q).df())
    import pandas as pd
    pool = pd.concat(parts, ignore_index=True).drop_duplicates("case_no")
    pos = pool[pool["flag"] == 1]
    neg = pool[pool["flag"] == 0]
    take = pd.concat([pos.sample(min(npos, len(pos)), random_state=42),
                      neg.sample(min(nneg, len(neg)), random_state=42)])
    frames[task] = take
    print(f"{task}: pool={len(pool):,} pos={len(pos):,} sampled={len(take)}")

# ---- fetch texts: one scan per anchor month over needed case_nos -------------
import pandas as pd
allsamp = pd.concat([df.assign(task=t) for t, df in frames.items()], ignore_index=True)
texts = {}
YEAR_DIRS = {"2015": "2015_Court_Judgments_CSV", "2016": "2016_Court_Judgments_CSV",
             "2017": "2017_Court_Judgments_CSV", "2019": "2019_Court_Judgments_CSV",
             "2020": "2020_Court_Judgments_CSV_Extracted_Partial"}
for m in MONTHS:
    yr, mo = m.split("_")
    hits = glob.glob(os.path.join(BASE, YEAR_DIRS[yr], "**", f"*{yr}年{mo}月*.csv"),
                     recursive=True)
    if not hits: continue
    f = hits[0].replace("\\", "/")
    ids = allsamp.loc[allsamp["ym"] == m, "case_no"].dropna().unique().tolist()
    if not ids: continue
    idlist = ",".join("'" + i.replace("'", "''") + "'" for i in ids)
    df = con.sql(f"""SELECT 案号 AS case_no, substr(全文, 1, 2600) AS text
                     FROM read_csv('{f}', auto_detect=true, sample_size=2000,
                       ignore_errors=true)
                     WHERE 案号 IN ({idlist})""").df()
    for _, row in df.iterrows():
        texts[row["case_no"]] = row["text"]
    print(f"texts {m}: {len(df)}")

# ---- write frames + agent batches --------------------------------------------
for task, df in frames.items():
    df = df.copy()
    df["has_text"] = df["case_no"].map(lambda c: c in texts)
    df.to_parquet(f"{VAL}/frame_{task}.parquet")
    items = [dict(case_no=r["case_no"], ym=r["ym"],
                  text=texts.get(r["case_no"], ""))
             for _, r in df.iterrows() if texts.get(r["case_no"])]
    random.shuffle(items)
    for k in range(0, len(items), 12):
        with open(f"{VAL}/batches/{task}_{k//12:03d}.jsonl", "w", encoding="utf-8") as fh:
            for it in items[k:k+12]:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{task}: {len(items)} items with text -> {(len(items)+11)//12} batches")
print("validation sampling done")

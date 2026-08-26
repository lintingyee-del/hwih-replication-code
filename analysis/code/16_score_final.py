# -*- coding: utf-8 -*-
"""6B step 16 — final validation scoring: v1 vs v2 extractors against gold.

v1 flags live in the sampling frames; v2 extractor values are joined from the
x2_crim/x2_civ re-sweep by case_no. Date/rate comparisons restrict to gold
non-null (the gold saw only 2600 chars; nulls there are not extractor errors).
Emits validation_final.csv + tab_validation.tex (paper table deferred per
writing freeze — file is still generated for later use).
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
import pandas as pd, numpy as np, json, glob, duckdb

VAL = str(_REP_PROJECT / "output" / "validation")
TABD = str(_REP_PROJECT / "output" / "tables")
EXT = str(_REP_PROJECT / "data" / "derived" / "extract_2014_2020")
con = duckdb.connect()

def load_labels(task):
    rows = []
    for f in glob.glob(f"{VAL}/labels_{task}_*.jsonl"):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass
    return pd.DataFrame(rows).drop_duplicates("case_no")

def prf(df, flag, gold):
    d = df.dropna(subset=[gold])
    d = d[d[gold].isin([0, 1])]
    tp = ((d[flag] == 1) & (d[gold] == 1)).sum()
    fp = ((d[flag] == 1) & (d[gold] == 0)).sum()
    fn = ((d[flag] == 0) & (d[gold] == 1)).sum()
    P = tp/(tp+fp) if tp+fp else np.nan
    R = tp/(tp+fn) if tp+fn else np.nan
    F = 2*P*R/(P+R) if (P and R and not np.isnan(P) and not np.isnan(R)) else np.nan
    return P, R, F, len(d)

rows = []
# ---------- binary dictionary indicators (v1) ----------
for task, gold, label in [
    ("civ_rel_txn", "gold_rel_txn", "Relational transaction (civil)"),
    ("civ_rel_fail", "gold_rel_fail", "Relational failure (civil)"),
    ("civ_evidence", "gold_iou", "IOU documented (civil)"),
    ("civ_backstop", "gold_backstop", "Coercive collection (civil)"),
    ("crim_backstop", "gold_backstop", "Hard backstop (criminal)"),
    ("crim_detention", "gold_detention_debt", "Detention x debt (criminal)"),
]:
    fr = pd.read_parquet(f"{VAL}/frame_{task}.parquet")
    lb = load_labels(task)
    if lb.empty or gold not in lb.columns:
        rows.append((label + " [v1]", np.nan, np.nan, np.nan, 0)); continue
    m = fr.merge(lb[["case_no", gold]], on="case_no")
    P, R, F, n = prf(m, "flag", gold)
    rows.append((label + " [v1]", P, R, F, n))

# ---------- fraud split: v1 vs v2 ----------
fr = pd.read_parquet(f"{VAL}/frame_crim_fraudsplit.parquet")
lb = load_labels("crim_fraudsplit")
if not lb.empty:
    m = fr.merge(lb[["case_no", "gold_telecom"]], on="case_no")
    P, R, F, n = prf(m, "flag", "gold_telecom")
    rows.append(("Telecom fraud [v1]", P, R, F, n))
    v2 = con.sql(f"""SELECT case_no, fraud_telecom_v2 FROM read_parquet('{EXT}/x2_crim_*.parquet')
                     WHERE fraud_telecom_v2 IS NOT NULL""").df()
    m2 = m.merge(v2, on="case_no", how="left")
    m2["flag_v2"] = m2["fraud_telecom_v2"].fillna(0).astype(int)
    P, R, F, n = prf(m2, "flag_v2", "gold_telecom")
    rows.append(("Telecom fraud [v2 modus]", P, R, F, n))

# ---------- offense year: v1 vs v2 (gold non-null only) ----------
fr = pd.read_parquet(f"{VAL}/frame_crim_offensedate.parquet")
lb = load_labels("crim_offensedate")
if not lb.empty:
    m = fr.merge(lb[["case_no", "gold_offense_start_year"]], on="case_no")
    g = pd.to_numeric(m["gold_offense_start_year"], errors="coerce")
    v1 = pd.to_numeric(m["offense_year_1"], errors="coerce").fillna(
        pd.to_numeric(m["offense_year_2"], errors="coerce"))
    b = m[g.notna() & v1.notna()]
    rows.append(("Offense year [v1]", (v1[b.index] == g[b.index]).mean(), np.nan, np.nan, len(b)))
    v2 = con.sql(f"SELECT case_no, offense_year_v2 FROM read_parquet('{EXT}/x2_crim_*.parquet')").df()
    m2 = m.merge(v2, on="case_no", how="left")
    y2 = pd.to_numeric(m2["offense_year_v2"], errors="coerce")
    g2 = pd.to_numeric(m2["gold_offense_start_year"], errors="coerce")
    b2 = m2[g2.notna() & y2.notna()]
    rows.append(("Offense year [v2 min-anchored]", (y2[b2.index] == g2[b2.index]).mean(),
                 np.nan, np.nan, len(b2)))
    cov1 = v1.notna().mean(); cov2 = y2.notna().mean()
    rows.append(("Offense year coverage v1/v2", cov1, cov2, np.nan, len(m2)))

# ---------- rate + origination year: v1 vs v2 ----------
fr = pd.read_parquet(f"{VAL}/frame_civ_rate_orig.parquet")
lb = load_labels("civ_rate_orig")
if not lb.empty:
    m = fr.merge(lb, on="case_no")
    v2 = con.sql(f"SELECT case_no, orig_year_v2, monthly_rate_v2 FROM read_parquet('{EXT}/x2_civ_*.parquet')").df()
    m = m.merge(v2, on="case_no", how="left")
    gr = pd.to_numeric(m["gold_monthly_rate_pct"], errors="coerce")
    r1 = pd.to_numeric(m["rate_月pct"], errors="coerce").fillna(pd.to_numeric(m["rate_月分"], errors="coerce"))
    r2 = pd.to_numeric(m["monthly_rate_v2"], errors="coerce")
    for tag, rv in [("v1", r1), ("v2 multi-pattern", r2)]:
        b = m[gr.notna() & rv.notna()]
        agree = (np.abs(rv[b.index] - gr[b.index]) <= 0.05).mean() if len(b) else np.nan
        fn_rate = (rv.isna() & gr.notna()).sum() / gr.notna().sum()
        rows.append((f"Monthly rate [{tag}] (agree/recall)", agree, 1 - fn_rate, np.nan, len(b)))
    go = pd.to_numeric(m["gold_orig_year"], errors="coerce")
    o1 = pd.to_numeric(m["orig_year"], errors="coerce")
    o2 = pd.to_numeric(m["orig_year_v2"], errors="coerce")
    for tag, ov in [("v1", o1), ("v2 min-anchored", o2)]:
        b = m[go.notna() & ov.notna()]
        agree = (ov[b.index] == go[b.index]).mean() if len(b) else np.nan
        cov = ov.notna().mean()
        rows.append((f"Origination year [{tag}] (agree/coverage)", agree, cov, np.nan, len(b)))

df = pd.DataFrame(rows, columns=["indicator", "P_or_agree", "R_or_cov", "F1", "n"])
df.to_csv(f"{VAL}/validation_final.csv", index=False)
L = [r"\begin{tabular}{lcccc}", r"\toprule",
     r"Indicator & P / agree & R / cov & F1 & $n$ \\ \midrule"]
for _, r0 in df.iterrows():
    f = lambda v: "--" if pd.isna(v) else f"{v:.2f}"
    L.append(f"{r0['indicator']} & {f(r0['P_or_agree'])} & {f(r0['R_or_cov'])} & {f(r0['F1'])} & {int(r0['n'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{TABD}/tab_validation.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
print(df.to_string(index=False))

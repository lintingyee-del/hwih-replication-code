# -*- coding: utf-8 -*-
"""6B step 12 — score LLM gold labels vs regex flags: P/R/F1 per indicator,
rate/date agreement; emit tab_validation.tex + macros."""

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
import pandas as pd, numpy as np, json, glob, os

VAL = str(_REP_PROJECT / "output" / "validation")
TABD = str(_REP_PROJECT / "output" / "tables")

def load_labels(task):
    rows = []
    for f in glob.glob(f"{VAL}/labels_{task}*.jsonl"):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: continue
    return pd.DataFrame(rows)

def prf(frame, labels, flagcol, goldcol):
    m = frame.merge(labels[["case_no", goldcol]], on="case_no")
    m = m.dropna(subset=[goldcol])
    tp = ((m[flagcol] == 1) & (m[goldcol] == 1)).sum()
    fp = ((m[flagcol] == 1) & (m[goldcol] == 0)).sum()
    fn = ((m[flagcol] == 0) & (m[goldcol] == 1)).sum()
    P = tp / (tp + fp) if tp + fp else np.nan
    R = tp / (tp + fn) if tp + fn else np.nan
    F = 2 * P * R / (P + R) if P and R else np.nan
    return P, R, F, len(m)

SPEC = [  # task, flag col in frame, gold field emitted by agent
    ("civ_rel_txn", "flag", "gold_rel_txn", "Relational transaction (civil)"),
    ("civ_rel_fail", "flag", "gold_rel_fail", "Relational failure (civil)"),
    ("civ_evidence", "flag", "gold_iou", "IOU documented (civil)"),
    ("civ_backstop", "flag", "gold_backstop", "Coercive collection (civil)"),
    ("crim_backstop", "flag", "gold_backstop", "Hard backstop (criminal)"),
    ("crim_detention", "flag", "gold_detention_debt", "Detention x debt (criminal)"),
    ("crim_fraudsplit", "flag", "gold_telecom", "Telecom fraud (criminal)"),
]
rows, mac = [], []
for task, fc, gc, label in SPEC:
    fr = pd.read_parquet(f"{VAL}/frame_{task}.parquet")
    lb = load_labels(task)
    if lb.empty or gc not in lb.columns:
        rows.append((label, np.nan, np.nan, np.nan, 0)); continue
    P, R, F, n = prf(fr, lb, fc, gc)
    rows.append((label, P, R, F, n))
    key = task.replace("_", "").replace("civ", "Civ").replace("crim", "Crim")
    mac += [f"\\newcommand{{\\Val{key}P}}{{{P:.2f}}}",
            f"\\newcommand{{\\Val{key}R}}{{{R:.2f}}}",
            f"\\newcommand{{\\Val{key}F}}{{{F:.2f}}}"]

# rate + origination year agreement
fr = pd.read_parquet(f"{VAL}/frame_civ_rate_orig.parquet")
lb = load_labels("civ_rate_orig")
if not lb.empty:
    m = fr.merge(lb, on="case_no")
    reg_rate = pd.to_numeric(m.get("rate_月pct"), errors="coerce").fillna(
        pd.to_numeric(m.get("rate_月分"), errors="coerce"))
    gold_rate = pd.to_numeric(m.get("gold_monthly_rate_pct"), errors="coerce")
    both = m[(reg_rate.notna()) & (gold_rate.notna())]
    agree = (np.abs(reg_rate[both.index] - gold_rate[both.index]) <= 0.05).mean() if len(both) else np.nan
    miss_fn = ((reg_rate.isna()) & (gold_rate.notna())).mean()
    rows.append(("Interest rate extraction (agree +-0.05pp)", agree, 1 - miss_fn, np.nan, len(m)))
    oy_reg = pd.to_numeric(m.get("orig_year"), errors="coerce")
    oy_gold = pd.to_numeric(m.get("gold_orig_year"), errors="coerce")
    ob = m[(oy_reg.notna()) & (oy_gold.notna())]
    oagree = (oy_reg[ob.index] == oy_gold[ob.index]).mean() if len(ob) else np.nan
    rows.append(("Origination year (exact)", oagree, np.nan, np.nan, len(ob)))

lb = load_labels("crim_offensedate")
if not lb.empty:
    fr = pd.read_parquet(f"{VAL}/frame_crim_offensedate.parquet")
    m = fr.merge(lb, on="case_no")
    ry = pd.to_numeric(m.get("offense_year_1"), errors="coerce").fillna(
        pd.to_numeric(m.get("offense_year_2"), errors="coerce"))
    gy = pd.to_numeric(m.get("gold_offense_start_year"), errors="coerce")
    b = m[ry.notna() & gy.notna()]
    ag = (ry[b.index] == gy[b.index]).mean() if len(b) else np.nan
    rows.append(("Offense start year (exact)", ag, np.nan, np.nan, len(b)))
    mac.append(f"\\newcommand{{\\ValOffenseYearAgree}}{{{ag:.2f}}}")

df = pd.DataFrame(rows, columns=["indicator", "P", "R", "F1", "n"])
df.to_csv(f"{VAL}/validation_scores.csv", index=False)
L = [r"\begin{tabular}{lcccc}", r"\toprule",
     r"Indicator & Precision & Recall & F1 & $n$ \\ \midrule"]
for _, r0 in df.iterrows():
    fmt = lambda v: "--" if pd.isna(v) else f"{v:.2f}"
    L.append(f"{r0['indicator']} & {fmt(r0['P'])} & {fmt(r0['R'])} & {fmt(r0['F1'])} & {int(r0['n'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{TABD}/tab_validation.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
open(f"{TABD}/numbers_val.tex", "w", encoding="utf-8").write("\n".join(mac) + "\n")
print(df.to_string(index=False))

# -*- coding: utf-8 -*-
"""6B step 123 — rel_txn measurement audit, full-sample rescore + classifier head-to-head.

Motivation: validation_scores.csv scored rel_txn on n=120 (labels were ~1/4 done when
12_score_validation.py last ran). Rescore on all ~500 gold labels, quantify the
misclassification structure of the regex flag that defines the headline
acquaintance-minus-stranger contrast, and benchmark a supervised text classifier
(TF-IDF char n-grams + logistic regression, 5-fold CV) against the regex on the same
gold labels. Also computes the contamination-implied attenuation factor for the
acq-minus-str contrast: measured gap = [P(gold=1|flag=1) - P(gold=1|flag=0)] x true gap
under cell-count contamination mixing.

Outputs: output/validation/rel_txn_headtohead.json (all numbers), console summary.
No regression is touched.
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
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

VAL = str(_REP_PROJECT / "output" / "validation")
rng = np.random.RandomState(42)

# ---- load gold labels, regex flags, and audit texts -------------------------
fr = pd.read_parquet(f"{VAL}/frame_civ_rel_txn.parquet")
rows = []
for f in glob.glob(f"{VAL}/labels_civ_rel_txn*.jsonl"):
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
lb = pd.DataFrame(rows)
texts = {}
for f in glob.glob(f"{VAL}/batches/civ_rel_txn_*.jsonl"):
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                texts[r["case_no"]] = r["text"]
            except json.JSONDecodeError:
                continue

m = fr.merge(lb[["case_no", "gold_rel_txn"]], on="case_no").dropna(subset=["gold_rel_txn"])
m["gold"] = m["gold_rel_txn"].astype(int)
m["flag"] = m["flag"].astype(int)
m["text"] = m["case_no"].map(texts)
m = m.dropna(subset=["text"]).reset_index(drop=True)
m["period"] = m["ym"].str[:4].astype(int).map(lambda y: "pre" if y < 2018 else "post")


def confusion(d, flagcol):
    tp = int(((d[flagcol] == 1) & (d.gold == 1)).sum())
    fp = int(((d[flagcol] == 1) & (d.gold == 0)).sum())
    fn = int(((d[flagcol] == 0) & (d.gold == 1)).sum())
    tn = int(((d[flagcol] == 0) & (d.gold == 0)).sum())
    P, R = tp / (tp + fp), tp / (tp + fn)
    return {
        "n": len(d), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": P, "recall": R, "f1": 2 * P * R / (P + R),
        "accuracy": (tp + tn) / len(d),
        "p_gold1_given_flag1": d[d[flagcol] == 1].gold.mean(),
        "p_gold1_given_flag0": d[d[flagcol] == 0].gold.mean(),
    }


out = {"n_gold": len(m)}
out["regex_all"] = confusion(m, "flag")
out["regex_pre"] = confusion(m[m.period == "pre"], "flag")
out["regex_post"] = confusion(m[m.period == "post"], "flag")

# contamination-implied attenuation of the acq-minus-str contrast
w1 = out["regex_all"]["p_gold1_given_flag1"]
w0 = out["regex_all"]["p_gold1_given_flag0"]
out["attenuation_factor"] = w1 - w0
out["implied_true_gap_from_0.182"] = 0.1816 / (w1 - w0)

# ---- classifier head-to-head: TF-IDF char n-grams + logistic, 5-fold CV -----
clf = make_pipeline(
    TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=60000,
                    sublinear_tf=True),
    LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
m["pred_lr"] = cross_val_predict(clf, m.text, m.gold, cv=cv, method="predict")
out["lr_all"] = confusion(m, "pred_lr")
out["lr_pre"] = confusion(m[m.period == "pre"], "pred_lr")
out["lr_post"] = confusion(m[m.period == "post"], "pred_lr")
w1c, w0c = out["lr_all"]["p_gold1_given_flag1"], out["lr_all"]["p_gold1_given_flag0"]
out["lr_attenuation_factor"] = w1c - w0c

# where do regex and classifier disagree, and who is right?
dis = m[m.flag != m.pred_lr]
out["disagreements"] = {
    "n": len(dis),
    "lr_right": int((dis.pred_lr == dis.gold).sum()),
    "regex_right": int((dis.flag == dis.gold).sum()),
}

with open(f"{VAL}/rel_txn_headtohead.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2, default=float)

for k in ["regex_all", "lr_all"]:
    r = out[k]
    print(f"{k}: n={r['n']} P={r['precision']:.3f} R={r['recall']:.3f} "
          f"F1={r['f1']:.3f} acc={r['accuracy']:.3f} "
          f"| P(g|1)={r['p_gold1_given_flag1']:.3f} P(g|0)={r['p_gold1_given_flag0']:.3f}")
print(f"regex attenuation factor: {out['attenuation_factor']:.3f} "
      f"(implied true gap from 0.182: {out['implied_true_gap_from_0.182']:.3f})")
print(f"lr attenuation factor:    {out['lr_attenuation_factor']:.3f}")
print("disagreements:", out["disagreements"])

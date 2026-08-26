# -*- coding: utf-8 -*-
"""Theme x role coverage table for Appendix G (tab:ivthemes).

Parses the reviewed quote bank (docs/interview_quote_bank_full.md, rendered from
<restricted-source-path>) and counts, per coded mechanism theme, the
deduplicated verbatim quotes and the distinct interview sessions contributing at
least one, by role.

Role resolution: a session takes its market-side role when one is tagged
(operator/agent/player/prosecutor; ambiguity resolved to the modal tag);
compiled working notes ('reviewer') inherit the session's role; remaining
sessions are 'mixed' (spanning roles) or 'other' (undetermined).

Output: output/interview_theme_coverage.csv + LaTeX rows on stdout.
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
import csv
import io
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "interview_quote_bank_full.md"
OUT = ROOT / "output" / "interview_theme_coverage.csv"

THEME_EN = {
    "一": "Pre-campaign coercive collection technology",
    "二": "Post-campaign repricing of violence",
    "三": "Counter-reporting by debtors and players",
    "四": "Ex-ante adjustment of settlement terms",
    "五": "Courts and formalization",
    "六": "Residual mechanism material",
}
MARKET = ["operator", "agent", "player", "prosecutor"]
ROLES = ["operator", "agent", "player", "prosecutor", "mixed", "other"]

entry_re = re.compile(r"\*\*\[([^|\]]+)\|([^|\]]+)\|([^|\]]+)\|([DI])\]\*\*")

md = MD.read_text(encoding="utf-8")
sections = re.split(r"\n## ", md)[1:]

theme_entries = {}
resp_roles = defaultdict(Counter)
resp_waves = defaultdict(set)
for sec in sections:
    title = sec.split("\n", 1)[0].strip()
    entries = entry_re.findall(sec)
    theme_entries[title] = entries
    for label, role, wave, _ in entries:
        resp_roles[label.strip()][role.strip()] += 1
        resp_waves[label.strip()].add(wave.strip())


def resolve(label):
    tags = resp_roles[label]
    market = [r for r in MARKET if tags.get(r, 0) > 0]
    if len(market) == 1:
        return market[0]
    if len(market) > 1:
        return max(market, key=lambda r: tags[r])
    if tags.get("mixed", 0) > 0 or tags.get("reviewer", 0) > 0:
        return "mixed"
    return "other"


comp = Counter(resolve(l) for l in resp_roles)
waves = Counter("+".join(sorted(resp_waves[l])) for l in resp_roles)
print(f"sessions with >=1 quote: {sum(comp.values())}  composition: {dict(comp)}")
print(f"waves: {dict(waves)}")

rows = []
for title, entries in theme_entries.items():
    key = title[0]
    labels = sorted({l.strip() for l, r, w, d in entries})
    rc = Counter(resolve(l) for l in labels)
    row = {
        "theme": THEME_EN.get(key, title),
        "quotes": len(entries),
        **{r: rc.get(r, 0) for r in ROLES},
        "sessions": len(labels),
    }
    assert sum(rc.values()) == len(labels)
    rows.append(row)

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["theme", "quotes"] + ROLES + ["sessions"])
    w.writeheader()
    w.writerows(rows)
print(f"wrote {OUT}")

print("\nLaTeX rows (mixed+other pooled):")
for r in rows:
    pooled = r["mixed"] + r["other"]
    print(
        f"{r['theme']} & {r['quotes']} & {r['operator']} & {r['agent']} & "
        f"{r['player']} & {r['prosecutor']} & {pooled} & {r['sessions']} \\\\"
    )

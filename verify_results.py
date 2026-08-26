#!/usr/bin/env python3
"""Compare regenerated working outputs with the frozen expected results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXPECTED = ROOT / "expected_results"
WORKING = ROOT / "analysis" / "output"
REPORT = ROOT / "verification" / "result_comparison.json"


def compare_csv(expected: Path, actual: Path, atol: float, rtol: float) -> list[str]:
    left = pd.read_csv(expected, encoding="utf-8-sig")
    right = pd.read_csv(actual, encoding="utf-8-sig")
    problems: list[str] = []
    if list(left.columns) != list(right.columns):
        return [f"columns differ: {list(left.columns)} != {list(right.columns)}"]
    if left.shape != right.shape:
        return [f"shape differs: {left.shape} != {right.shape}"]
    for column in left.columns:
        left_numeric = pd.to_numeric(left[column], errors="coerce")
        right_numeric = pd.to_numeric(right[column], errors="coerce")
        numeric_mask = left_numeric.notna() | right_numeric.notna()
        if numeric_mask.any():
            if not np.allclose(
                left_numeric[numeric_mask].to_numpy(float),
                right_numeric[numeric_mask].to_numpy(float),
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            ):
                delta = np.nanmax(np.abs(
                    left_numeric[numeric_mask].to_numpy(float)
                    - right_numeric[numeric_mask].to_numpy(float)))
                problems.append(f"numeric column {column} differs (max abs {delta:g})")
        nonnumeric = ~numeric_mask
        if nonnumeric.any():
            left_text = left.loc[nonnumeric, column].fillna("").astype(str)
            right_text = right.loc[nonnumeric, column].fillna("").astype(str)
            if not left_text.equals(right_text):
                problems.append(f"text column {column} differs")
    return problems


def compare_json_value(left, right, path: str, atol: float, rtol: float,
                       problems: list[str]) -> None:
    if isinstance(left, bool) or isinstance(right, bool):
        if left != right:
            problems.append(f"{path}: {left!r} != {right!r}")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol):
            problems.append(f"{path}: {left!r} != {right!r}")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            problems.append(f"{path}: object keys differ")
            return
        for key in left:
            compare_json_value(
                left[key], right[key], f"{path}.{key}", atol, rtol, problems)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            problems.append(f"{path}: list lengths differ")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_json_value(
                left_item, right_item, f"{path}[{index}]", atol, rtol, problems)
        return
    if left != right:
        problems.append(f"{path}: {left!r} != {right!r}")


def compare_one(expected: Path, actual: Path,
                atol: float, rtol: float) -> tuple[str, list[str]]:
    suffix = expected.suffix.lower()
    if suffix == ".csv":
        return "numeric-and-text", compare_csv(expected, actual, atol, rtol)
    if suffix == ".json":
        left = json.loads(expected.read_text(encoding="utf-8-sig"))
        right = json.loads(actual.read_text(encoding="utf-8-sig"))
        problems: list[str] = []
        compare_json_value(left, right, "$", atol, rtol, problems)
        return "structured", problems
    if suffix in {".tex", ".txt"}:
        left = expected.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        right = actual.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        return "exact-text", [] if left == right else ["text differs"]
    if suffix in {".pdf", ".png"}:
        return "existence-only", []
    return "existence-only", []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=5e-8)
    parser.add_argument("--rtol", type=float, default=1e-7)
    args = parser.parse_args()

    records = []
    failures = 0
    for expected in sorted(EXPECTED.rglob("*")):
        if not expected.is_file():
            continue
        relative = expected.relative_to(EXPECTED)
        actual = WORKING / relative
        if not actual.exists():
            records.append({
                "path": relative.as_posix(),
                "mode": "missing",
                "status": "fail",
                "problems": ["working output is missing"],
            })
            failures += 1
            continue
        try:
            mode, problems = compare_one(expected, actual, args.atol, args.rtol)
        except Exception as error:
            mode, problems = "comparison-error", [str(error)]
        status = "pass" if not problems else "fail"
        failures += bool(problems)
        records.append({
            "path": relative.as_posix(),
            "mode": mode,
            "status": status,
            "problems": problems,
        })

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "expected_files": len(records),
        "failures": failures,
        "absolute_tolerance": args.atol,
        "relative_tolerance": args.rtol,
        "records": records,
    }
    REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(f"RESULT CHECK FAILED: {failures} file(s); report: {REPORT}")
        for record in records:
            if record["status"] == "fail":
                print(f"- {record['path']}: {'; '.join(record['problems'])}")
        return 1
    print(f"RESULT CHECK PASSED: {len(records)} files; report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

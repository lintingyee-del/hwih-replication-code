#!/usr/bin/env python3
"""Compare regenerated figure pixels with the frozen expected figures.

PDF container metadata (for example, creation timestamps) is intentionally
ignored.  Each PDF page is rendered with the same pinned PyMuPDF build on both
sides and the resulting pixels are compared exactly.  PNGs are decoded and
compared in the same way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz


DEFAULT_ROOT = Path(__file__).resolve().parent
RENDER_MATRIX = fitz.Matrix(2, 2)  # 144 dpi for PDF pages


def pixmap_signature(pixmap: fitz.Pixmap) -> tuple[int, int, int, bytes]:
    return pixmap.width, pixmap.height, pixmap.n, pixmap.samples


def compare_pdf(expected: Path, actual: Path) -> list[str]:
    problems: list[str] = []
    with fitz.open(expected) as left, fitz.open(actual) as right:
        if left.page_count != right.page_count:
            return [f"page count differs: {left.page_count} != {right.page_count}"]
        for page_number in range(left.page_count):
            left_pixels = left[page_number].get_pixmap(
                matrix=RENDER_MATRIX, alpha=True)
            right_pixels = right[page_number].get_pixmap(
                matrix=RENDER_MATRIX, alpha=True)
            if pixmap_signature(left_pixels) != pixmap_signature(right_pixels):
                problems.append(f"rendered pixels differ on page {page_number + 1}")
    return problems


def compare_png(expected: Path, actual: Path) -> list[str]:
    left = fitz.Pixmap(expected)
    right = fitz.Pixmap(actual)
    return [] if pixmap_signature(left) == pixmap_signature(right) else [
        "decoded pixels differ"
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help="replication-package root (defaults to this script's directory)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    expected_root = root / "expected_results"
    working_root = root / "analysis" / "output"
    report = root / "verification" / "exhibit_comparison.json"
    records = []
    failures = 0
    figures = sorted(
        path for path in expected_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".png"}
    )
    for expected in figures:
        relative = expected.relative_to(expected_root)
        actual = working_root / relative
        problems: list[str]
        if not actual.exists():
            problems = ["working figure is missing"]
        else:
            try:
                problems = (
                    compare_pdf(expected, actual)
                    if expected.suffix.lower() == ".pdf"
                    else compare_png(expected, actual)
                )
            except Exception as error:
                problems = [f"comparison error: {error}"]
        status = "pass" if not problems else "fail"
        failures += bool(problems)
        records.append({
            "path": relative.as_posix(),
            "status": status,
            "problems": problems,
        })

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "figures": len(records),
        "pdf_render_dpi": 144,
        "failures": failures,
        "records": records,
    }, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(f"EXHIBIT CHECK FAILED: {failures} file(s); report: {report}")
        for record in records:
            if record["status"] == "fail":
                print(f"- {record['path']}: {'; '.join(record['problems'])}")
        return 1
    print(f"EXHIBIT CHECK PASSED: {len(records)} figures; report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

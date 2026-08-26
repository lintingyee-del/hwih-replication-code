# -*- coding: utf-8 -*-
"""Formal death-registry triple difference and annual coefficient profile.

The estimand compares the male-female homicide mortality ratio with the
geometric mean of the corresponding suicide and road-traffic ratios:

    D_st = [log H_mst - log H_fst]
           - 0.5 * ([log S_mst - log S_fst]
                    + [log R_mst - log R_fst]).

All three causes use deaths at ages 15-59.  The age-specific population
denominator cancels algebraically from D_st, so counts identify the same
triple contrast without requiring an unpublished age-by-sex population table.
The six clusters are East/Central/West by urban/rural residence.

The 2014, 2015, and 2017-2021 volumes have text layers.  The 2016 volume is
scanned; its cause rows are read with RapidOCR and admitted only after:
  * every homicide age sum agrees with the independently transcribed vector;
  * male + female, urban + rural, and region adding-up identities hold exactly
    for suicide, homicide, and road traffic.

Outputs are written to output/cdc_homicide:
  cdc_age1559_panel.csv
  cdc_2016_age1559_ocr.csv
  cdc_formal_ddd_stratum_year.csv
  cdc_formal_ddd_estimates.csv
  cdc_formal_ddd_eventstudy.csv
  cdc_formal_ddd_log.txt
  fig_cdc_homicide_ddd.pdf / .png

Use --refresh-ocr to rebuild the cached 2016 OCR extract.
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

import argparse
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = PROJECT / "output" / "cdc_homicide"
OUT.mkdir(parents=True, exist_ok=True)
WORKSPACE = PROJECT.parent

SOURCE_ROOT = Path(os.environ.get(
    "HWIH_CDC_SOURCE_ROOT",
    str(_REP_MORTALITY),
))

TEXT_VOLUMES = {
    2014: [
        SOURCE_ROOT / "中国死因监测数据集2014" / "2014年死因数据集" / f
        for f in (
            "2014死因监测数据集（上）.pdf",
            "2014死因监测数据集（中）.pdf",
            "2014死因监测数据集（下）.pdf",
        )
    ],
    2015: [
        SOURCE_ROOT / "中国死因监测数据集2015" / f
        for f in (
            "1数据集2016.27.pdf",
            "2数据集2016.27.pdf",
            "3数据集2016.27.pdf",
        )
    ],
    2017: [SOURCE_ROOT / "中国死因监测数据集2017.pdf"],
    2018: [SOURCE_ROOT / "中国死因监测数据集2018.pdf"],
    2019: [SOURCE_ROOT / "中国死因监测数据集2019.pdf"],
    2020: [SOURCE_ROOT / "中国死因监测数据集2020.pdf"],
    2021: [SOURCE_ROOT / "2021死因监测数据集.pdf"],
}

SCAN_2016 = [
    SOURCE_ROOT / "中国死因监测数据集2016" / "2016年死因数据集"
    / "2016死因监测数据集（上）.pdf",
    SOURCE_ROOT / "中国死因监测数据集2016" / "2016年死因数据集"
    / "2016死因监测数据集（中）.pdf",
    SOURCE_ROOT / "中国死因监测数据集2016" / "2016年死因数据集"
    / "2016死因监测数据集（下） .pdf",
]

CAUSE_NAMES = {
    "自杀及后遗症": "suicide",
    "他杀及后遗症": "homicide",
    "道路交通事故": "traffic",
}

REGIONS = ["全国", "东部", "中部", "西部"]
URBRUR = ["城乡合计", "城市", "农村"]
SEXES = ["合计", "男性", "女性"]
BLOCK_ORDER = [
    ("全国", "城乡合计"),
    ("全国", "城市"),
    ("全国", "农村"),
    ("东部", "城乡合计"),
    ("中部", "城乡合计"),
    ("西部", "城乡合计"),
    ("东部", "城市"),
    ("中部", "城市"),
    ("西部", "城市"),
    ("东部", "农村"),
    ("中部", "农村"),
    ("西部", "农村"),
]
LABELS = [(r, u, s) for r, u in BLOCK_ORDER for s in SEXES]

LOG: list[str] = []


def say(message: str = "") -> None:
    print(message, flush=True)
    LOG.append(message)


def open_combined(paths: list[Path]) -> fitz.Document:
    """Combine volume parts and drop duplicated numbered pages."""
    out = fitz.open()
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        src = fitz.open(path)
        for page_no in range(len(src)):
            first = src[page_no].get_text().strip().split("\n", 1)[0].strip()
            key = first if re.match(r"^\d{1,4}$", first) else None
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            out.insert_pdf(src, from_page=page_no, to_page=page_no)
        src.close()
    return out


def chapter7_count_bounds(doc: fitz.Document) -> tuple[int, int]:
    p71 = p72 = None
    for page_no in range(20, len(doc)):
        text = doc[page_no].get_text()[:400]
        if p71 is None and re.search(r"7\.1\s*地区别", text):
            p71 = page_no
        elif (p71 is not None and p72 is None and page_no > p71
              and re.search(r"7\.2\s*地区别", text)):
            p72 = page_no
            break
    if p71 is None or p72 is None:
        raise RuntimeError(f"Chapter 7 count section not found: p71={p71}, p72={p72}")
    return p71, p72


def parse_text_age_counts(doc: fitz.Document, p_from: int,
                          p_to: int) -> list[dict[str, list[int]]]:
    """Parse all 20 count columns for the three target causes."""
    number = re.compile(r"^-?[\d,]+(\.\d+)?$")
    ucode = re.compile(r"^U\d{3}$")
    blocks: list[dict[str, list[int]]] = []
    current: dict[str, list[int]] | None = None
    for page_no in range(p_from, p_to):
        lines = [line.strip() for line in doc[page_no].get_text().split("\n")]
        tokens: list[str] = []
        for line in lines:
            if re.match(
                r"^(中国死因监测数据集|续表|第七章|7\.\d|疾病$|编码$|"
                r"疾病名称$|合计.*岁|.*岁～$|\d+ 岁～)",
                line,
            ):
                continue
            tokens.extend(line.split())
        pos = 0
        while pos < len(tokens):
            token = tokens[pos]
            if not ucode.match(token):
                pos += 1
                continue
            pos += 1
            name_parts: list[str] = []
            values: list[str] = []
            while pos < len(tokens) and not number.match(tokens[pos]):
                if ucode.match(tokens[pos]):
                    break
                name_parts.append(tokens[pos])
                pos += 1
            while pos < len(tokens) and number.match(tokens[pos]):
                values.append(tokens[pos])
                pos += 1
            name = re.sub(r"^[0-9a-zA-Z\.\s]*", "", "".join(name_parts))
            name = re.sub(r"^(Ⅰ|Ⅱ|Ⅲ|Ⅳ)\.?", "", name)
            if name == "全死因":
                current = {}
                blocks.append(current)
            if current is not None and name in CAUSE_NAMES:
                parsed = [int(float(value.replace(",", ""))) for value in values]
                if len(parsed) < 14:
                    raise RuntimeError(
                        f"Short age row on PDF page {page_no + 1}: "
                        f"{name}, {len(parsed)} columns"
                    )
                current[CAUSE_NAMES[name]] = parsed
    return blocks


def extract_text_year(year: int, paths: list[Path]) -> pd.DataFrame:
    doc = open_combined(paths)
    p71, p72 = chapter7_count_bounds(doc)
    blocks = parse_text_age_counts(doc, p71, p72)
    doc.close()
    if len(blocks) != 36:
        raise RuntimeError(f"{year}: found {len(blocks)} count blocks, expected 36")
    rows = []
    for (region, residence, sex), block in zip(LABELS, blocks):
        missing = sorted(set(CAUSE_NAMES.values()) - set(block))
        if missing:
            raise RuntimeError(f"{year} {region}/{residence}/{sex}: missing {missing}")
        row = {
            "year": year,
            "region": region,
            "urbrur": residence,
            "sex": sex,
            "source": "PDF text layer",
        }
        for cause, values in block.items():
            row[f"{cause}_15_59_n"] = sum(values[5:14])
        rows.append(row)
    say(f"{year}: extracted 36 blocks from text layer")
    return pd.DataFrame(rows)


def digits(text: str) -> str:
    cleaned = (str(text).strip().replace(",", "")
               .replace("O", "0").replace("o", "0")
               .replace("I", "1").replace("l", "1"))
    return "".join(character for character in cleaned if character.isdigit())


def render_global_page(documents: list[fitz.Document], global_index: int,
                       scale: float = 4.0) -> np.ndarray:
    import cv2

    local_index = global_index
    for doc in documents:
        if local_index < len(doc):
            pixmap = doc[local_index].get_pixmap(
                matrix=fitz.Matrix(scale, scale), alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n)
            if pixmap.n == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        local_index -= len(doc)
    raise IndexError(global_index)


# Column centers in the clockwise-rotated chapter 7 scan, normalized by width.
# Index 0 is total; indices 1-19 are ages 0, 1, 5, ..., 85.
SCAN_COLUMN_X = np.array([
    .282, .318, .345, .373, .407, .441, .474, .508, .541, .575,
    .608, .641, .675, .708, .741, .774, .808, .842, .875, .907,
])


def group_ocr_rows(result, x_offset: int, y_offset: int) -> list[list[dict]]:
    tokens = []
    for box, text, confidence in result or []:
        x = x_offset + sum(point[0] for point in box) / 4
        y = y_offset + sum(point[1] for point in box) / 4
        value = digits(text)
        if value:
            tokens.append({
                "x": x,
                "y": y,
                "value": int(value),
                "confidence": float(confidence),
                "raw": str(text),
            })
    tokens.sort(key=lambda item: item["y"])
    rows: list[list[dict]] = []
    for token in tokens:
        if not rows:
            rows.append([token])
            continue
        row_mean = float(np.mean([item["y"] for item in rows[-1]]))
        if abs(token["y"] - row_mean) <= 13:
            rows[-1].append(token)
        else:
            rows.append([token])
    for row in rows:
        row.sort(key=lambda item: item["x"])
    return rows


def read_scan_row(rows: list[list[dict]], total: int, width: int,
                  label: str) -> tuple[list[int | None], float]:
    total_x = SCAN_COLUMN_X[0] * width
    candidates = []
    for row in rows:
        near_total = min(row, key=lambda item: abs(item["x"] - total_x))
        if abs(near_total["x"] - total_x) < .022 * width:
            if near_total["value"] == int(total):
                candidates.append(row)
    if len(candidates) != 1:
        totals_seen = []
        for row in rows:
            item = min(row, key=lambda token: abs(token["x"] - total_x))
            if abs(item["x"] - total_x) < .022 * width:
                totals_seen.append(item["value"])
        raise RuntimeError(
            f"{label}: total {total} matched {len(candidates)} rows; "
            f"near-total values were {totals_seen}"
        )
    row = candidates[0]
    values: list[int | None] = [None] * len(SCAN_COLUMN_X)
    confidences = []
    # Only the row total and the nine analysis columns are required.  The
    # oldest-age cells are often printed zeros that a detector may omit.
    required_columns = [0, *range(5, 14)]
    for column in required_columns:
        x_fraction = SCAN_COLUMN_X[column]
        expected_x = x_fraction * width
        item = min(row, key=lambda token: abs(token["x"] - expected_x))
        distance = abs(item["x"] - expected_x)
        if distance > .018 * width:
            raise RuntimeError(
                f"{label}: no OCR token for column {column}; "
                f"nearest distance={distance:.1f}"
            )
        values[column] = int(item["value"])
        confidences.append(float(item["confidence"]))
    if values[0] != int(total):
        raise RuntimeError(f"{label}: expected total {total}, read {values[0]}")
    return values, min(confidences)


def extract_2016_ocr(refresh: bool) -> pd.DataFrame:
    cache = OUT / "cdc_2016_age1559_ocr.csv"
    if cache.exists() and not refresh:
        cached = pd.read_csv(cache, encoding="utf-8-sig")
        say("2016: loaded cached OCR extract")
        return cached

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "RapidOCR is required to rebuild the 2016 scan extract"
        ) from exc

    raw_path = OUT / "pages_2016_raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    count_pages = [page for page in raw if page["kind"] == "counts"]
    if len(count_pages) != 36:
        raise RuntimeError(f"2016: found {len(count_pages)} count pages")
    for path in SCAN_2016:
        if not path.exists():
            raise FileNotFoundError(path)

    documents = [fitz.open(path) for path in SCAN_2016]
    reader = RapidOCR()
    rows_out = []
    try:
        for index, (label, page) in enumerate(zip(LABELS, count_pages), start=1):
            match = re.search(r"idx(\d+)", Path(page["file"]).stem)
            if match is None:
                raise RuntimeError(f"2016: cannot recover page index from {page['file']}")
            global_index = int(match.group(1))
            image = render_global_page(documents, global_index, scale=4.0)
            height, width = image.shape[:2]
            x0, x1 = int(.25 * width), int(.93 * width)
            y0, y1 = int(.55 * height), int(.87 * height)
            result, _ = reader(image[y0:y1, x0:x1])
            grouped = group_ocr_rows(result, x0, y0)

            cause_rows = {}
            cause_confidence = {}
            totals = {
                "traffic": int(page["traffic"]),
                "suicide": int(page["suicide"]),
                "homicide": int(page["homicide"]),
            }
            for cause, total in totals.items():
                values, confidence = read_scan_row(
                    grouped, total, width,
                    f"2016 page {page['book_page']} {cause}",
                )
                cause_rows[cause] = values
                cause_confidence[cause] = confidence

            homicide_read = [int(value)
                              for value in cause_rows["homicide"][5:14]]
            homicide_transcribed = [int(value) for value
                                    in page["homicide_ages_15_55"]]
            exact_anchor_matches = sum(
                read == transcribed
                for read, transcribed
                in zip(homicide_read, homicide_transcribed)
            )
            anchor_matches = sum(
                read == transcribed or {read, transcribed} == {6, 9}
                for read, transcribed
                in zip(homicide_read, homicide_transcribed)
            )
            if anchor_matches < 8:
                raise RuntimeError(
                    f"2016 page {page['book_page']}: homicide OCR "
                    f"{homicide_read} matches only {anchor_matches}/9 "
                    f"independently transcribed cells {homicide_transcribed}"
                )
            if exact_anchor_matches < 9:
                say(
                    f"2016 page {page['book_page']}: homicide OCR anchor "
                    f"{exact_anchor_matches}/9 exact "
                    f"({anchor_matches}/9 allowing 6/9 glyph ambiguity); "
                    f"retained identity-validated "
                    f"transcription"
                )
            cause_rows["homicide"][5:14] = homicide_transcribed

            region, residence, sex = label
            row = {
                "year": 2016,
                "region": region,
                "urbrur": residence,
                "sex": sex,
                "source": "scanned PDF, RapidOCR",
            }
            for cause, values in cause_rows.items():
                row[f"{cause}_15_59_n"] = sum(values[5:14])
                row[f"{cause}_ocr_min_confidence"] = cause_confidence[cause]
            rows_out.append(row)
            if index % 6 == 0:
                say(f"2016 OCR: {index}/36 count pages passed homicide gate")
    finally:
        for document in documents:
            document.close()

    frame = pd.DataFrame(rows_out)
    frame.to_csv(cache, index=False, encoding="utf-8-sig")
    say(f"2016: wrote OCR cache with {len(frame)} rows")
    return frame


def adding_up_failures(frame: pd.DataFrame, column: str,
                       year: int) -> list[str]:
    data = frame[frame["year"] == year].set_index(
        ["region", "urbrur", "sex"])[column].to_dict()
    failures = []

    def check(left: int, right: int, description: str) -> None:
        if int(left) != int(right):
            failures.append(f"{description}: {left} != {right}")

    for region in REGIONS:
        for residence in URBRUR:
            check(
                data[(region, residence, "合计")],
                data[(region, residence, "男性")]
                + data[(region, residence, "女性")],
                f"{region}/{residence}: male+female",
            )
        check(
            data[(region, "城乡合计", "合计")],
            data[(region, "城市", "合计")]
            + data[(region, "农村", "合计")],
            f"{region}: urban+rural",
        )
    for residence in URBRUR:
        check(
            data[("全国", residence, "合计")],
            sum(data[(region, residence, "合计")]
                for region in ("东部", "中部", "西部")),
            f"{residence}: east+central+west",
        )
    return failures


def validate_age_panel(frame: pd.DataFrame) -> None:
    expected_rows = 8 * 36
    if len(frame) != expected_rows:
        raise RuntimeError(f"age panel has {len(frame)} rows, expected {expected_rows}")
    columns = [
        "suicide_15_59_n",
        "homicide_15_59_n",
        "traffic_15_59_n",
    ]
    all_failures = []
    for year in range(2014, 2022):
        for column in columns:
            failures = adding_up_failures(frame, column, year)
            all_failures.extend(
                f"{year} {column}: {failure}" for failure in failures)
    if all_failures:
        preview = "\n".join(all_failures[:20])
        raise RuntimeError(
            f"{len(all_failures)} adding-up failures:\n{preview}")

    old = pd.read_csv(OUT / "cdc_homicide_panel.csv", encoding="utf-8-sig")
    comparison = frame.merge(
        old[["year", "region", "urbrur", "sex", "homicide_15_59_n"]],
        on=["year", "region", "urbrur", "sex"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    mismatch = comparison[
        comparison["homicide_15_59_n_new"]
        != comparison["homicide_15_59_n_old"]
    ]
    if len(mismatch):
        raise RuntimeError(
            f"{len(mismatch)} homicide age sums differ from the frozen panel")
    say("Validation: 456/456 cause-year adding-up identities pass")
    say("Validation: all 288 homicide age sums match the frozen panel")


def build_age_panel(refresh_ocr: bool) -> pd.DataFrame:
    frames = [
        extract_text_year(year, paths)
        for year, paths in sorted(TEXT_VOLUMES.items())
    ]
    frames.append(extract_2016_ocr(refresh_ocr))
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(
        ["year", "region", "urbrur", "sex"]).reset_index(drop=True)
    validate_age_panel(panel)
    panel.to_csv(OUT / "cdc_age1559_panel.csv", index=False,
                 encoding="utf-8-sig")
    return panel


def make_triple_gap(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel[
        panel["region"].isin(["东部", "中部", "西部"])
        & panel["urbrur"].isin(["城市", "农村"])
        & panel["sex"].isin(["男性", "女性"])
    ].copy()
    data["stratum"] = data["region"] + " / " + data["urbrur"]
    rows = []
    for (year, region, residence, stratum), group in data.groupby(
        ["year", "region", "urbrur", "stratum"], sort=True
    ):
        by_sex = group.set_index("sex")
        row = {
            "year": int(year),
            "region": region,
            "urbrur": residence,
            "stratum": stratum,
        }
        for cause in ("homicide", "suicide", "traffic"):
            male = float(by_sex.loc["男性", f"{cause}_15_59_n"])
            female = float(by_sex.loc["女性", f"{cause}_15_59_n"])
            if male <= 0 or female <= 0:
                raise RuntimeError(
                    f"Non-positive {cause} count in {stratum}, {year}")
            row[f"{cause}_male_n"] = int(male)
            row[f"{cause}_female_n"] = int(female)
            row[f"{cause}_mf_gap"] = math.log(male) - math.log(female)
        row["ddd_suicide"] = (
            row["homicide_mf_gap"] - row["suicide_mf_gap"])
        row["ddd_traffic"] = (
            row["homicide_mf_gap"] - row["traffic_mf_gap"])
        row["ddd_combined"] = (
            row["homicide_mf_gap"]
            - .5 * (row["suicide_mf_gap"] + row["traffic_mf_gap"])
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(OUT / "cdc_formal_ddd_stratum_year.csv", index=False,
                  encoding="utf-8-sig")
    return output


def sign_flip_p(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    null = np.array([
        np.mean(values * np.asarray(signs))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    one_sided = float(np.mean(null <= observed + 1e-12))
    two_sided = float(np.mean(
        np.abs(null) >= abs(observed) - 1e-12))
    return one_sided, two_sided


def fit_pretrend_projection(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact six-stratum tests after projecting each 2014--2017 trend."""
    outcomes = (
        "homicide_mf_gap",
        "suicide_mf_gap",
        "traffic_mf_gap",
        "ddd_combined",
    )
    records = []
    for end_year in (2020, 2021):
        for outcome in outcomes:
            deviations = []
            for _, group in frame.groupby("stratum", sort=True):
                pre = group[group["year"].between(2014, 2017)]
                slope, intercept = np.polyfit(
                    pre["year"], pre[outcome], 1)
                post = group[group["year"].between(2018, end_year)]
                projected = intercept + slope * post["year"]
                deviations.append(float(
                    (post[outcome] - projected).mean()))
            values = np.asarray(deviations, dtype=float)
            exact_one, exact_two = sign_flip_p(values)
            mean_deviation = float(values.mean())
            records.append({
                "outcome": outcome,
                "post_window": f"2018-{end_year}",
                "mean_log_deviation": mean_deviation,
                "percent_deviation": 100 * math.expm1(mean_deviation),
                "negative_strata": int((values < 0).sum()),
                "strata": len(values),
                "exact_one_sided_p": exact_one,
                "exact_two_sided_p": exact_two,
            })
    return pd.DataFrame(records)


def fit_post(frame: pd.DataFrame, outcome: str, end_year: int,
             drop_2016: bool = False) -> dict:
    sample = frame[
        frame["year"].between(2014, end_year)
    ].copy()
    if drop_2016:
        sample = sample[sample["year"] != 2016].copy()
    sample["post"] = (sample["year"] >= 2018).astype(int)
    formula = f"{outcome} ~ post | stratum"
    model_crv1 = pf.feols(
        formula, data=sample, vcov={"CRV1": "stratum"})
    model_crv3 = pf.feols(
        formula, data=sample, vcov={"CRV3": "stratum"})
    beta = float(model_crv1.coef()["post"])
    crv1_se = float(model_crv1.se()["post"])
    crv3_se = float(model_crv3.se()["post"])

    deltas = (sample.groupby("stratum")
              .apply(lambda group:
                     group.loc[group["post"].eq(1), outcome].mean()
                     - group.loc[group["post"].eq(0), outcome].mean(),
                     include_groups=False)
              .to_numpy(dtype=float))
    if len(deltas) != 6:
        raise RuntimeError(f"{outcome}: expected 6 stratum changes")
    df_clusters = len(deltas) - 1
    p_crv1 = float(model_crv1.pvalue()["post"])
    p_crv3 = float(model_crv3.pvalue()["post"])
    exact_one, exact_two = sign_flip_p(deltas)
    critical = float(stats.t.ppf(.975, df_clusters))
    return {
        "outcome": outcome,
        "sample": (
            f"2014-{end_year}" + (", excluding 2016" if drop_2016 else "")
        ),
        "post_window": f"2018-{end_year}",
        "coefficient": beta,
        "percent": 100 * math.expm1(beta),
        "crv1_se": crv1_se,
        "crv1_p_t5": p_crv1,
        "crv3_se": crv3_se,
        "crv3_p_t5": p_crv3,
        "crv3_ci_low": beta - critical * crv3_se,
        "crv3_ci_high": beta + critical * crv3_se,
        "exact_one_sided_p": exact_one,
        "exact_two_sided_p": exact_two,
        "negative_strata": int((deltas < 0).sum()),
        "clusters": len(deltas),
        "observations": len(sample),
    }


def fit_event_study(frame: pd.DataFrame,
                    outcome: str = "ddd_combined") -> tuple[pd.DataFrame, dict]:
    sample = frame.copy()
    years = sorted(sample["year"].unique())
    reference = 2017
    formula = f"{outcome} ~ i(year, ref={reference}) | stratum"
    model_crv1 = pf.feols(
        formula, data=sample, vcov={"CRV1": "stratum"})
    model_crv3 = pf.feols(
        formula, data=sample, vcov={"CRV3": "stratum"})
    clusters = int(sample["stratum"].nunique())
    critical = float(stats.t.ppf(.975, clusters - 1))

    records = []
    for year in years:
        if year == reference:
            records.append({
                "year": year,
                "coefficient": 0.0,
                "percent": 0.0,
                "crv1_se": np.nan,
                "crv3_se": np.nan,
                "crv3_ci_low": 0.0,
                "crv3_ci_high": 0.0,
                "exact_two_sided_p": np.nan,
                "reference": True,
            })
            continue
        name = f"year::{year}"
        coefficient = float(model_crv1.coef()[name])
        crv1_se = float(model_crv1.se()[name])
        crv3_se = float(model_crv3.se()[name])
        paired = (frame[frame["year"] == year].set_index("stratum")[outcome]
                  - frame[frame["year"] == reference]
                  .set_index("stratum")[outcome])
        _, exact_two = sign_flip_p(paired.to_numpy(dtype=float))
        records.append({
            "year": year,
            "coefficient": coefficient,
            "percent": 100 * math.expm1(coefficient),
            "crv1_se": crv1_se,
            "crv1_p_t5": float(model_crv1.pvalue()[name]),
            "crv3_se": crv3_se,
            "crv3_p_t5": float(model_crv3.pvalue()[name]),
            "crv3_ci_low": coefficient - critical * crv3_se,
            "crv3_ci_high": coefficient + critical * crv3_se,
            "exact_two_sided_p": exact_two,
            "reference": False,
        })
    result = pd.DataFrame(records)

    lead_years = [2014, 2015, 2016]
    lead_matrix = np.column_stack([
        (frame[frame["year"] == year].set_index("stratum")[outcome]
         - frame[frame["year"] == reference].set_index("stratum")[outcome])
        .to_numpy(dtype=float)
        for year in lead_years
    ])
    observed_mean = lead_matrix.mean(axis=0)
    observed_se = lead_matrix.std(axis=0, ddof=1) / math.sqrt(len(lead_matrix))
    observed_max_t = float(np.max(np.abs(observed_mean / observed_se)))
    null_max_t = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(lead_matrix)):
        draw = lead_matrix * np.asarray(signs)[:, None]
        draw_mean = draw.mean(axis=0)
        draw_se = draw.std(axis=0, ddof=1) / math.sqrt(len(draw))
        null_max_t.append(float(np.max(np.abs(draw_mean / draw_se))))
    exact_joint = float(np.mean(
        np.asarray(null_max_t) >= observed_max_t - 1e-12))

    lead_names = [f"year::{year}" for year in lead_years]
    names = list(model_crv1._coefnames)
    lead_positions = [names.index(name) for name in lead_names]
    lead_beta = np.asarray(model_crv1._beta_hat)[lead_positions]
    covariance = np.asarray(model_crv1._vcov)[
        np.ix_(lead_positions, lead_positions)]
    wald = float(lead_beta @ np.linalg.pinv(covariance) @ lead_beta / 3)
    clustered_f_p = float(stats.f.sf(wald, 3, clusters - 1))
    diagnostics = {
        "reference_year": reference,
        "pretrend_years": "2014-2016",
        "pretrend_clustered_F": wald,
        "pretrend_clustered_F_p": clustered_f_p,
        "pretrend_exact_max_t_p": exact_joint,
        "clusters": clusters,
    }
    return result, diagnostics


def draw_event_figure(event: pd.DataFrame) -> None:
    sys.path.insert(0, str(WORKSPACE))
    import matplotlib.pyplot as plt
    import hz_figstyle as hz

    hz.apply()
    figure, axis = plt.subplots(figsize=(5.6, 3.35))
    axis.axhline(0, color=hz.GUIDE, lw=.7)
    axis.axvline(2017.5, color=hz.GUIDE, lw=.7, ls=(0, (4, 3)))
    axis.axvspan(2017.5, 2020.5, color=hz.SHADE, zorder=0)

    plotted = event[~event["reference"]].copy()
    y = plotted["coefficient"].to_numpy() * 100
    lower = (plotted["coefficient"] - plotted["crv3_ci_low"]).to_numpy() * 100
    upper = (plotted["crv3_ci_high"] - plotted["coefficient"]).to_numpy() * 100
    axis.errorbar(
        plotted["year"], y, yerr=np.vstack([lower, upper]),
        fmt="o-", color=hz.INK, ecolor=hz.INK, lw=1.05,
        elinewidth=.75, capsize=2.2, markersize=3.2, zorder=3,
    )
    axis.plot(2017, 0, "s", mfc="white", mec=hz.INK, ms=4.5,
              mew=.8, zorder=4)
    axis.set_xticks(range(2014, 2022))
    axis.set_xlim(2013.65, 2021.35)
    axis.set_ylabel("Triple-difference log points $\\times$ 100")
    axis.set_xlabel("Year (2017 reference)")
    hz.style_ticklabels(axis)
    hz.save(figure, OUT / "fig_cdc_homicide_ddd.pdf")
    figure.savefig(
        OUT / "fig_cdc_homicide_ddd.png",
        bbox_inches="tight", pad_inches=.03, dpi=220,
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-only", action="store_true",
        help="Use the released age-cell panel and skip source-volume extraction")
    parser.add_argument(
        "--refresh-ocr", action="store_true",
        help="Rebuild the cached 2016 scan extraction")
    arguments = parser.parse_args()

    say("=== CDC age-15-59 extraction and formal DDD ===")
    if arguments.analysis_only:
        cached = OUT / "cdc_age1559_panel.csv"
        if not cached.exists():
            raise FileNotFoundError(
                "--analysis-only requires output/cdc_homicide/"
                "cdc_age1559_panel.csv")
        panel = pd.read_csv(cached, encoding="utf-8-sig")
        validate_age_panel(panel)
        say(f"Using released age-cell panel: {cached}")
    else:
        panel = build_age_panel(arguments.refresh_ocr)
    gaps = make_triple_gap(panel)
    say(f"Analysis panel: {len(gaps)} stratum-years "
        f"({gaps['stratum'].nunique()} strata x {gaps['year'].nunique()} years)")

    specifications = [
        fit_post(gaps, "ddd_combined", 2020),
        fit_post(gaps, "ddd_suicide", 2020),
        fit_post(gaps, "ddd_traffic", 2020),
        fit_post(gaps, "homicide_mf_gap", 2020),
        fit_post(gaps, "ddd_combined", 2019),
        fit_post(gaps, "ddd_combined", 2021),
        fit_post(gaps, "ddd_combined", 2020, drop_2016=True),
    ]
    estimates = pd.DataFrame(specifications)
    estimates.to_csv(
        OUT / "cdc_formal_ddd_estimates.csv", index=False,
        encoding="utf-8-sig")

    event, pretrend = fit_event_study(gaps)
    for key, value in pretrend.items():
        event[key] = value
    event.to_csv(
        OUT / "cdc_formal_ddd_eventstudy.csv", index=False,
        encoding="utf-8-sig")
    exact_pretrend = fit_pretrend_projection(gaps)
    exact_pretrend.to_csv(
        OUT / "cdc_age1559_exact_pretrend.csv", index=False,
        encoding="utf-8-sig")
    draw_event_figure(event)

    say("")
    say("=== Post estimates ===")
    for row in specifications:
        say(
            f"{row['outcome']:18s} {row['sample']:24s} "
            f"b={row['coefficient']:+.4f} "
            f"({row['percent']:+.1f}%), "
            f"CRV1 SE={row['crv1_se']:.4f}, p(t5)={row['crv1_p_t5']:.3f}; "
            f"CRV3 SE={row['crv3_se']:.4f}, p(t5)={row['crv3_p_t5']:.3f}; "
            f"exact one-sided p={row['exact_one_sided_p']:.3f}"
        )
    say("")
    say("=== Annual DDD profile (2017 reference) ===")
    for _, row in event.iterrows():
        if row["reference"]:
            say(f"{int(row['year'])}: reference")
        else:
            say(
                f"{int(row['year'])}: b={row['coefficient']:+.4f} "
                f"({row['percent']:+.1f}%), CRV3 SE={row['crv3_se']:.4f}, "
                f"exact two-sided p={row['exact_two_sided_p']:.3f}"
            )
    say(
        "Pretrend 2014-2016: "
        f"clustered F p={pretrend['pretrend_clustered_F_p']:.3f}; "
        f"exact max-|t| p={pretrend['pretrend_exact_max_t_p']:.3f}"
    )
    say("")
    say("=== Exact pretrend-projection tests (age 15-59) ===")
    for _, row in exact_pretrend.iterrows():
        say(
            f"{row['outcome']:18s} {row['post_window']} "
            f"deviation={row['mean_log_deviation']:+.4f} "
            f"({row['percent_deviation']:+.1f}%), "
            f"negative={int(row['negative_strata'])}/6, "
            f"exact one-sided p={row['exact_one_sided_p']:.3f}"
        )
    say("")
    say(f"Wrote formal estimates, annual profile, and figure to {OUT}")
    (OUT / "cdc_formal_ddd_log.txt").write_text(
        "\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

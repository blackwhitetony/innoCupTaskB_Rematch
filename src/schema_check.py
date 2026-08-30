"""B 题数据集 Schema 校验。

对应 plan.md 第 5.1 节：每次训练前自动检查文件哈希、行数、列名、类型、
目标存在性、重复行、缺失、无穷值、非法类别、数值边界与行顺序。
原始 CSV 只读，校验结果写入 artifacts/metrics/schema_report.json 并回填
configs/schema.yaml 中的 sha256（首次）。

用法：
    uv run python -m src.schema_check [--csv 复赛B题/B题数据集.csv] [--schema configs/schema.yaml]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_schema(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "复赛B题" / "B题数据集.csv"))
    ap.add_argument("--schema", default=str(ROOT / "configs" / "schema.yaml"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "metrics" / "schema_report.json"))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    schema_path = Path(args.schema)
    if not csv_path.exists():
        print(f"[FATAL] 找不到数据集: {csv_path}")
        return 1

    schema = load_schema(schema_path)
    df = pd.read_csv(csv_path)

    issues: list[str] = []
    warnings: list[str] = []
    checks: dict = {}

    # 1) 文件哈希
    file_sha = sha256_of(csv_path)
    expected_sha = (schema.get("file") or {}).get("sha256", "")
    sha_match = (expected_sha == file_sha) if expected_sha else None
    checks["sha256"] = {"actual": file_sha, "expected": expected_sha, "match": sha_match}
    if expected_sha and not sha_match:
        issues.append("SHA-256 与 configs/schema.yaml 不一致")

    # 2) 形状
    n_rows, n_cols = df.shape
    checks["shape"] = {"actual": [n_rows, n_cols], "expected": [
        schema["file"]["n_rows"], schema["file"]["n_cols"]]}
    if n_rows != schema["file"]["n_rows"] or n_cols != schema["file"]["n_cols"]:
        issues.append(f"形状 {df.shape} 与预期不一致")

    # 3) 列名与顺序
    ordered = list(schema["columns_order"])
    actual_cols = list(df.columns)
    checks["columns_order_match"] = actual_cols == ordered
    checks["columns_missing"] = sorted(set(ordered) - set(actual_cols))
    checks["columns_extra"] = sorted(set(actual_cols) - set(ordered))
    if actual_cols != ordered:
        issues.append("列名或顺序与 schema 定义不一致")

    # 4) 目标存在且无缺失
    for role, col in schema["targets"].items():
        miss = int(df[col].isna().sum())
        checks[f"target_{role}_missing"] = miss
        if col not in df.columns or miss != 0:
            issues.append(f"目标 {col} 缺失 {miss} 条")

    # 5) 缺失
    missing_report = {c: int(df[c].isna().sum()) for c in ordered}
    checks["missing"] = missing_report
    expected_missing = schema.get("expected_missing", {})
    for c, exp in expected_missing.items():
        if missing_report.get(c, 0) != exp:
            issues.append(f"{c} 缺失 {missing_report.get(c, 0)}，预期 {exp}")
    for c in ordered:
        if c not in expected_missing and missing_report.get(c, 0) != 0:
            issues.append(f"{c} 出现未预期的缺失 {missing_report.get(c, 0)} 条")

    # 6) 重复
    checks["full_duplicates"] = int(df.duplicated().sum())
    feature_cols = [c for c in ordered if c not in schema["targets"].values()]
    checks["feature_duplicates"] = int(df[feature_cols].duplicated().sum())

    # 7) 无穷值
    inf_counts = {c: int(df[c].isin([float("inf"), float("-inf")]).sum())
                  for c in ordered if pd.api.types.is_numeric_dtype(df[c].dtype)}
    checks["inf"] = inf_counts
    if any(inf_counts.values()):
        issues.append("存在无穷值")

    # 8) 非法类别
    illegal = {}
    for c, legal in schema["categorical"].items():
        vals = set(df[c].dropna().astype(str).unique())
        bad = sorted(vals - set(legal))
        if bad:
            illegal[c] = bad
            issues.append(f"{c} 存在非法类别 {bad}")
    checks["illegal_categories"] = illegal

    # 9) 二值合法性
    bad_bin = {}
    for c, legal in schema["binary"].items():
        vals = set(df[c].dropna().unique())
        bad = sorted(vals - set(legal))
        if bad:
            bad_bin[c] = bad
            issues.append(f"{c} 存在非法取值 {bad}")
    checks["illegal_binary"] = bad_bin

    # 10) 数值边界（越界仅告警）
    out_of_range = {}
    for c, meta in schema["numeric"].items():
        lo, hi = meta["plausible"]
        n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
        if n_out:
            out_of_range[c] = n_out
            warnings.append(f"{c} 有 {n_out} 条超出合理范围 [{lo}, {hi}]")
    checks["out_of_range"] = out_of_range

    # 11) 观测统计（用于人工复核）
    observed = {}
    for c in ordered:
        if pd.api.types.is_string_dtype(df[c].dtype) or df[c].dtype == object:
            observed[c] = {"levels": [str(x) for x in df[c].dropna().unique()]}
        else:
            observed[c] = {"min": float(df[c].min()), "max": float(df[c].max())}
    checks["observed"] = observed

    status = "ERROR" if issues else ("WARN" if warnings else "OK")
    report = {
        "status": status,
        "csv": str(csv_path),
        "issues": issues,
        "warnings": warnings,
        "checks": checks,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 回填 sha256（首次）
    if not expected_sha:
        with schema_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == 'sha256: ""':
                lines[i] = f'  sha256: "{file_sha}"\n'
                break
        schema_path.write_text("".join(lines), encoding="utf-8")

    print(f"[{status}] 报告已写入 {out_path}")
    for m in issues:
        print(f"  - {m}")
    for m in warnings:
        print(f"  ~ {m}")
    if not issues and not warnings:
        print("  所有校验通过。")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())

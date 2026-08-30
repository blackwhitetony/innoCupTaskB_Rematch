"""B 题数据预处理（折内拟合，泄露安全）。

对应 plan.md 第 5 节与第 6.1 节：
- 所有填补、编码、缩放均在训练折内拟合，验证折/测试集只做 transform。
- 数值缺失 -> 中位数填补；类别缺失 -> 独立类别 "Missing"。
- 线性模型走 One-Hot(drop first) + 标准化；树模型保留类别 dtype / 原生缺失。

用法：
    from src.preprocessing import load_data, build_preprocessor
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs" / "schema.yaml"


def load_schema() -> dict:
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_column_sets(schema: dict | None = None) -> dict:
    """返回 {categorical, binary, numeric, targets, feature_cols}。"""
    schema = schema or load_schema()
    categorical = list(schema["categorical"])
    binary = list(schema["binary"])
    numeric = [c for c in schema["numeric"] if c not in schema["targets"].values()]
    targets = list(schema["targets"].values())
    feature_cols = categorical + binary + numeric
    return {
        "categorical": categorical,
        "binary": binary,
        "numeric": numeric,
        "targets": targets,
        "feature_cols": feature_cols,
    }


def load_data(path: str | Path) -> pd.DataFrame:
    """读取原始 CSV，附加 row_id（= 原始行号，仅用于 OOF/提交回填，绝不入模）。"""
    df = pd.read_csv(path)
    df.insert(0, "row_id", np.arange(len(df)))
    return df


def build_preprocessor(
    categorical: list[str],
    numeric: list[str],
    binary: list[str] | None = None,
    scale: bool = True,
) -> ColumnTransformer:
    """构建折内拟合的预处理管道。

    数值列：中位数填补 ->（可选）标准化；
    类别列：填 "Missing" -> One-Hot(drop first, handle_unknown=ignore)；
    二值/其余列：中位数填补后 passthrough（作为数值直通）。
    """
    binary = binary or []
    transformers = []
    if numeric:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
            ]),
            categorical,
        ))
    if binary:
        transformers.append((
            "bin",
            Pipeline([("impute", SimpleImputer(strategy="median"))]),
            binary,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def clean_frame(
    df: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    binary: list[str],
) -> pd.DataFrame:
    """返回清洗后的 DataFrame（用于树模型或 EDA）：数值中位数填补、类别填 "Missing"。

    注意：填补值应来自训练折；本函数只做单一 DataFrame 的整体清洗，
    折内清洗请通过 build_preprocessor 的 Pipeline（fit 在训练折）完成。
    """
    out = df.copy()
    for c in numeric + binary:
        out[c] = out[c].fillna(out[c].median())
    for c in categorical:
        out[c] = out[c].fillna("Missing").astype("category")
    return out

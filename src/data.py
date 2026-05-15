from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PATHS


LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subjects",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

FAKE_LABELS = {"false", "pants-fire", "barely-true"}
TRUE_LABELS = {"true", "mostly-true", "half-true"}
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    audit: dict
    class_names: tuple[str, str] = ("fake", "true")


def normalize_text(value: str) -> str:
    text = "" if value is None else str(value)
    return _WS_RE.sub(" ", text.strip().lower())


def _read_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {path}")
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)
    df["split"] = split_name
    return df


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["label"] = cleaned["label"].fillna("").astype(str).str.strip().str.lower()
    cleaned["statement"] = cleaned["statement"].fillna("").astype(str)
    for col in ("subjects", "speaker", "speaker_job", "state", "party", "context"):
        cleaned[col] = cleaned[col].fillna("unknown").astype(str)
    for col in (
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
    ):
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0.0).astype("float32")
    cleaned["_norm_statement"] = cleaned["statement"].map(normalize_text)
    cleaned["_norm_speaker"] = cleaned["speaker"].map(normalize_text)
    return cleaned


def _map_binary_labels(labels: pd.Series) -> pd.Series:
    unknown = sorted(set(labels.unique()) - (FAKE_LABELS | TRUE_LABELS))
    if unknown:
        raise ValueError(f"Unexpected labels detected: {unknown}")
    return labels.map(lambda label: 0 if label in FAKE_LABELS else 1).astype(int)


def _deduplicate_bundle(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    audit = {}
    train_before, valid_before, test_before = len(train), len(valid), len(test)

    train = train.drop_duplicates(subset=["_norm_statement", "_norm_speaker", "label"]).reset_index(drop=True)
    valid = valid.drop_duplicates(subset=["_norm_statement", "_norm_speaker", "label"]).reset_index(drop=True)
    test = test.drop_duplicates(subset=["_norm_statement", "_norm_speaker", "label"]).reset_index(drop=True)

    test_statements = set(test["_norm_statement"])
    valid_statements = set(valid["_norm_statement"])
    train = train.loc[~train["_norm_statement"].isin(valid_statements | test_statements)].reset_index(drop=True)
    valid = valid.loc[~valid["_norm_statement"].isin(test_statements)].reset_index(drop=True)

    audit["within_split_duplicates_removed"] = {
        "train": train_before - len(train),
        "valid": valid_before - len(valid),
        "test": test_before - len(test),
    }
    audit["cross_split_statement_overlap_removed"] = {
        "train": train_before - audit["within_split_duplicates_removed"]["train"] - len(train),
        "valid": valid_before - audit["within_split_duplicates_removed"]["valid"] - len(valid),
        "test": 0,
    }
    return train, valid, test, audit


def _split_distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _binary_distribution(y: pd.Series) -> dict[str, int]:
    counts = y.value_counts().sort_index()
    return {"fake": int(counts.get(0, 0)), "true": int(counts.get(1, 0))}


def _add_lie_ratio(df: pd.DataFrame) -> pd.DataFrame:
    total = (
        df["barely_true_counts"]
        + df["false_counts"]
        + df["half_true_counts"]
        + df["mostly_true_counts"]
        + df["pants_on_fire_counts"]
    )
    df = df.copy()
    df["lie_ratio"] = (
        (df["false_counts"] + df["pants_on_fire_counts"]) / (total + 1)
    ).astype("float32")
    return df


def load_binary_liar_dataset(raw_data_dir: Path = PATHS.raw_data_dir) -> DatasetBundle:
    train = _clean_frame(_read_split(raw_data_dir / "train.tsv", "train"))
    valid = _clean_frame(_read_split(raw_data_dir / "valid.tsv", "valid"))
    test  = _clean_frame(_read_split(raw_data_dir / "test.tsv",  "test"))

    train, valid, test, dedup_audit = _deduplicate_bundle(train, valid, test)
    y_train = _map_binary_labels(train["label"])
    y_valid = _map_binary_labels(valid["label"])
    y_test  = _map_binary_labels(test["label"])

    # ── drop temp columns BEFORE building the bundle (fixes inplace bug) ──
    drop_cols = ["_norm_statement", "_norm_speaker"]
    train = train.drop(columns=drop_cols).reset_index(drop=True)
    valid = valid.drop(columns=drop_cols).reset_index(drop=True)
    test  = test.drop(columns=drop_cols).reset_index(drop=True)

    # ── add lie_ratio after dedup so counts are clean ──
    train = _add_lie_ratio(train)
    valid = _add_lie_ratio(valid)
    test  = _add_lie_ratio(test)

    audit = {
        "split_sizes": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))},
        "original_label_distribution": {
            "train": _split_distribution(train),
            "valid": _split_distribution(valid),
            "test":  _split_distribution(test),
        },
        "binary_label_distribution": {
            "train": _binary_distribution(y_train),
            "valid": _binary_distribution(y_valid),
            "test":  _binary_distribution(y_test),
        },
        **dedup_audit,
    }

    return DatasetBundle(
        train=train,
        valid=valid,
        test=test,
        y_train=y_train.reset_index(drop=True),
        y_valid=y_valid.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        audit=audit,
    )

# def load_binary_liar_dataset(raw_data_dir: Path = PATHS.raw_data_dir) -> DatasetBundle:
#     train = _clean_frame(_read_split(raw_data_dir / "train.tsv", "train"))
#     valid = _clean_frame(_read_split(raw_data_dir / "valid.tsv", "valid"))
#     test = _clean_frame(_read_split(raw_data_dir / "test.tsv", "test"))

#     train, valid, test, dedup_audit = _deduplicate_bundle(train, valid, test)
#     y_train = _map_binary_labels(train["label"])
#     y_valid = _map_binary_labels(valid["label"])
#     y_test = _map_binary_labels(test["label"])

#     audit = {
#         "split_sizes": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))},
#         "original_label_distribution": {
#             "train": _split_distribution(train),
#             "valid": _split_distribution(valid),
#             "test": _split_distribution(test),
#         },
#         "binary_label_distribution": {
#             "train": _binary_distribution(y_train),
#             "valid": _binary_distribution(y_valid),
#             "test": _binary_distribution(y_test),
#         },
#         **dedup_audit,
#     }

#     for frame in (train, valid, test):
#         frame.drop(columns=["_norm_statement", "_norm_speaker"], inplace=True)

#     return DatasetBundle(
#         train=train.reset_index(drop=True),
#         valid=valid.reset_index(drop=True),
#         test=test.reset_index(drop=True),
#         y_train=y_train.reset_index(drop=True),
#         y_valid=y_valid.reset_index(drop=True),
#         y_test=y_test.reset_index(drop=True),
#         audit=audit,
#     )


def summarize_dataset(bundle: DatasetBundle) -> str:
    sizes = bundle.audit["split_sizes"]
    binary = bundle.audit["binary_label_distribution"]
    lines = [
        "Dataset audit",
        f"Train: {sizes['train']} rows | fake={binary['train']['fake']} true={binary['train']['true']}",
        f"Valid: {sizes['valid']} rows | fake={binary['valid']['fake']} true={binary['valid']['true']}",
        f"Test: {sizes['test']} rows | fake={binary['test']['fake']} true={binary['test']['true']}",
        (
            "Duplicate removal "
            f"(within train/valid/test): {bundle.audit['within_split_duplicates_removed']['train']}/"
            f"{bundle.audit['within_split_duplicates_removed']['valid']}/"
            f"{bundle.audit['within_split_duplicates_removed']['test']}"
        ),
        (
            "Cross-split statement overlap removed "
            f"(train/valid/test): {bundle.audit['cross_split_statement_overlap_removed']['train']}/"
            f"{bundle.audit['cross_split_statement_overlap_removed']['valid']}/"
            f"{bundle.audit['cross_split_statement_overlap_removed']['test']}"
        ),
    ]
    return "\n".join(lines)

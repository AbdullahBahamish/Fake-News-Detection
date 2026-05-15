from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, OneHotEncoder

from .config import FEATURE_CONFIG, FeatureConfig


_TOKEN_RE = re.compile(r"\b\w+\b")
_PUNCT_RE = re.compile(r"[!?.,;:'\"-]")


def _extract_text(values):
    if hasattr(values, "iloc"):
        values = values.iloc[:, 0]
    return values.fillna("").astype(str).to_numpy()


def _one_hot_encoder(min_frequency: int):
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=min_frequency,
            sparse_output=True,
            dtype=np.float32,
        )
    except TypeError:
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=min_frequency,
                sparse=True,
                dtype=np.float32,
            )
        except TypeError:
            return OneHotEncoder(
                handle_unknown="ignore",
                sparse=True,
                dtype=np.float32,
            )


class SpeakerCredibilityEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, speaker_column: str = FEATURE_CONFIG.speaker_column, smoothing: float = FEATURE_CONFIG.speaker_smoothing):
        self.speaker_column = speaker_column
        self.smoothing = float(smoothing)

    def fit(self, X, y):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[self.speaker_column])
        speakers = frame[self.speaker_column].fillna("unknown").astype(str)
        target = np.asarray(y, dtype=np.float32)
        prior = float(target.mean()) if len(target) else 0.5

        grouped = pd.DataFrame({"speaker": speakers, "target": target}).groupby("speaker")["target"].agg(["sum", "count"])
        self.prior_ = prior
        self.stats_ = grouped.to_dict(orient="index")
        return self

    def transform(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[self.speaker_column])
        speakers = frame[self.speaker_column].fillna("unknown").astype(str)
        values = np.empty((len(speakers), 1), dtype=np.float32)
        for index, speaker in enumerate(speakers):
            stats = self.stats_.get(speaker)
            if stats is None:
                values[index, 0] = self.prior_
            else:
                values[index, 0] = (stats["sum"] + self.smoothing * self.prior_) / (stats["count"] + self.smoothing)
        return values


class TextStatisticsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[FEATURE_CONFIG.text_column])
        texts = frame.iloc[:, 0].fillna("").astype(str).tolist()
        rows = np.zeros((len(texts), 8), dtype=np.float32)

        for index, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text)
            token_count = float(len(tokens))
            char_count = float(len(text))
            total_letters = float(sum(char.isalpha() for char in text))
            uppercase_count = float(sum(char.isupper() for char in text))
            digit_count = float(sum(char.isdigit() for char in text))
            exclamation_count = float(text.count("!"))
            question_count = float(text.count("?"))
            punctuation_count = float(len(_PUNCT_RE.findall(text)))
            mean_token_length = float(sum(len(token) for token in tokens) / max(token_count, 1.0))

            rows[index] = [
                token_count,
                char_count,
                mean_token_length,
                exclamation_count,
                question_count,
                punctuation_count,
                uppercase_count / max(total_letters, 1.0),
                digit_count / max(char_count, 1.0),
            ]

        return rows


def build_feature_pipeline(config: FeatureConfig | None = None) -> ColumnTransformer:
    config = config or FEATURE_CONFIG

    text_word_pipeline = Pipeline(
        steps=[
            ("selector", FunctionTransformer(_extract_text, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    sublinear_tf=True,
                    ngram_range=config.tfidf_ngram_range,
                    max_features=config.tfidf_max_features,
                    min_df=config.tfidf_min_df,
                    max_df=config.tfidf_max_df,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    text_transformer = text_word_pipeline
    if config.use_char_ngrams:
        char_pipeline = Pipeline(
            steps=[
                ("selector", FunctionTransformer(_extract_text, validate=False)),
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        lowercase=True,
                        sublinear_tf=True,
                        ngram_range=config.char_ngram_range,
                        max_features=config.char_max_features,
                        min_df=config.tfidf_min_df,
                        max_df=config.tfidf_max_df,
                        dtype=np.float32,
                    ),
                ),
            ]
        )
        text_transformer = FeatureUnion([("word", text_word_pipeline), ("char", char_pipeline)])

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", MaxAbsScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", _one_hot_encoder(config.one_hot_min_frequency)),
        ]
    )

    transformers = [
        ("text", text_transformer, [config.text_column]),
        ("numeric", numeric_pipeline, list(config.numeric_columns)),
        ("categorical", categorical_pipeline, list(config.categorical_columns)),
        ("speaker_credibility", SpeakerCredibilityEncoder(config.speaker_column, config.speaker_smoothing), [config.speaker_column]),
    ]

    if config.include_text_statistics:
        transformers.append(
            (
                "text_stats",
                Pipeline(
                    steps=[
                        ("stats", TextStatisticsTransformer()),
                        ("scale", MaxAbsScaler()),
                    ]
                ),
                [config.text_column],
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )

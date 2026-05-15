from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Defines all important paths in the project."""
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        # Using object.__setattr__ is acceptable here because the class is frozen
        object.__setattr__(self, "data_dir", self.root / "data")
        object.__setattr__(self, "raw_data_dir", self.data_dir / "raw")
        object.__setattr__(self, "results_dir", self.root / "results")
        object.__setattr__(self, "plots_dir", self.results_dir / "plots")
        object.__setattr__(self, "reports_dir", self.results_dir / "reports")


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature engineering pipeline."""
    text_column: str = "statement"
    
    # Metadata columns
    categorical_columns: tuple[str, ...] = ("subjects", "party", "context", "speaker_job", "state")
    speaker_column: str = "speaker"
    numeric_columns: tuple[str, ...] = (
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "lie_ratio",    # ← new
    )

    # Text features
    tfidf_max_features: int = 12000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95
    
    use_char_ngrams: bool = True   # ← was False
    char_ngram_range: tuple[int, int] = (3, 5)
    char_max_features: int = 4000

    # Categorical encoding
    one_hot_min_frequency: int = 3   # ← was 5

    # Custom features
    speaker_smoothing: float = 7.0    # ← was 15.0
    include_text_statistics: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training and hyperparameter search."""
    random_state: int = 42
    cv_folds: int = 10                    # Improved from 3 → Good change
    search_iterations: int = 8            # Slightly increased recommendation
    n_jobs: int = -1                      # Changed to -1 (use all cores) by default


# Global instances (still convenient for most cases)
PATHS = ProjectPaths()
FEATURE_CONFIG = FeatureConfig()
TRAINING_CONFIG = TrainingConfig()
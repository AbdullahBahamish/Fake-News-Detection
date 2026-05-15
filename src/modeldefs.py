from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC, SVC

from .config import TRAINING_CONFIG


ALLOWED_MODEL_KEYS = {
    "bert",
    "random_forest",
    "svm_rbf",
    "gradient_boosting",
    "stacking",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    estimator: object
    param_distributions: dict[str, object]
    search_iterations: int
    feature_overrides: dict[str, object] = field(default_factory=dict)
    select_k_best: int | None = None
    requires_dense: bool = False
    use_svd: bool = False
    svd_components: int | None = None
    scale_dense: bool = False


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


MACRO_F1 = make_scorer(macro_f1)


MODEL_ALIASES = {
    "bert": "bert",
    "bert-base": "bert",
    "baseline": "bert",
    "rf": "random_forest",
    "random_forest": "random_forest",
    "rbf": "svm_rbf",
    "svm_rbf": "svm_rbf",
    "gb": "gradient_boosting",
    "gradient_boosting": "gradient_boosting",
    "stack": "stacking",
    "stacking": "stacking",
}


def normalize_model_key(model_key: str) -> str:
    normalized = MODEL_ALIASES.get(model_key.strip().lower(), model_key.strip().lower())
    return normalized


# def _build_stacking_estimator(random_state: int) -> StackingClassifier:
#     base_estimators = [
#         (
#             "lr",
#             LogisticRegression(
#                 solver="liblinear",
#                 class_weight="balanced",
#                 max_iter=3000,
#                 random_state=random_state,
#             ),
#         ),
#         (
#             "svm",
#             LinearSVC(
#                 class_weight="balanced",
#                 max_iter=6000,
#                 random_state=random_state,
#             ),
#         ),
#         ("nb", ComplementNB(alpha=0.5)),
#     ]
#     final_estimator = LogisticRegression(
#         solver="liblinear",
#         class_weight="balanced",
#         max_iter=3000,
#         random_state=random_state,
#     )
#     return StackingClassifier(
#         estimators=base_estimators,
#         final_estimator=final_estimator,
#         cv=TRAINING_CONFIG.cv_folds,
#         stack_method="auto",
#         passthrough=False,
#         n_jobs=TRAINING_CONFIG.n_jobs,
#     )

def _build_stacking_estimator(random_state: int) -> StackingClassifier:
    base_estimators = [
        (
            "lr",
            LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=3000,
                random_state=random_state,
            ),
        ),
        (
            "svm",
            SVC(                               # ← was LinearSVC
                kernel="linear",
                class_weight="balanced",
                probability=True,              # ← enables predict_proba
                max_iter=6000,
                random_state=random_state,
            ),
        ),
        ("nb", ComplementNB(alpha=0.5)),
    ]
    final_estimator = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=3000,
        random_state=random_state,
    )
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=TRAINING_CONFIG.cv_folds,
        stack_method="predict_proba",          # ← was "auto", now explicit
        passthrough=False,
        n_jobs=TRAINING_CONFIG.n_jobs,
    )


def get_model_specs(random_state: int = TRAINING_CONFIG.random_state) -> dict[str, ModelSpec]:
    return {
        "random_forest": ModelSpec(
            key="random_forest",
            display_name="Random Forest (TF-IDF)",
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=30,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=TRAINING_CONFIG.n_jobs,
                random_state=random_state,
            ),
            param_distributions={
                "clf__n_estimators": np.array([200, 300, 400], dtype=int),
                "clf__max_depth": np.array([20, 30, 40], dtype=int),
                "clf__min_samples_split": np.array([2, 4, 8], dtype=int),
                "clf__min_samples_leaf": np.array([1, 2, 4], dtype=int),
                "clf__max_features": np.array(["sqrt", 0.5], dtype=object),
            },
            search_iterations=8,
            feature_overrides={"tfidf_max_features": 6000, "include_text_statistics": False},
            select_k_best=1500,
            requires_dense=True,
        ),
        "svm_rbf": ModelSpec(
            key="svm_rbf",
            display_name="SVM (RBF Kernel)",
            estimator=SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                cache_size=512,
            ),
            param_distributions={
                "clf__C": np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float),
                "clf__gamma": np.array(["scale", 0.01, 0.02, 0.05, 0.1], dtype=object),
            },
            search_iterations=8,
            feature_overrides={"tfidf_max_features": 10000, "include_text_statistics": False},
            use_svd=True,
            svd_components=256,
            scale_dense=True,
        ),
        # "gradient_boosting": ModelSpec(
        #     key="gradient_boosting",
        #     display_name="Gradient Boosting (Engineered Features)",
        #     estimator=GradientBoostingClassifier(
        #         learning_rate=0.05,
        #         max_depth=6,
        #         min_samples_leaf=20,
        #         subsample=0.9,
        #         n_estimators=150,
        #         random_state=random_state,
        #     ),
        #     param_distributions={
        #         "clf__learning_rate": np.array([0.03, 0.05, 0.1], dtype=float),
        #         "clf__max_depth": np.array([4, 6, 8], dtype=int),
        #         "clf__min_samples_leaf": np.array([10, 20, 40], dtype=int),
        #         "clf__n_estimators": np.array([100, 150, 200], dtype=int),
        #         "clf__subsample": np.array([0.8, 0.9, 1.0], dtype=float),
        #     },
        #     search_iterations=8,
        #     feature_overrides={"tfidf_max_features": 8000, "include_text_statistics": True},
        #     select_k_best=1024,
        #     requires_dense=True,
        # ),

        "gradient_boosting": ModelSpec(
            key="gradient_boosting",
            display_name="Gradient Boosting (Engineered Features)",
            estimator=HistGradientBoostingClassifier(   # ← one word change
                learning_rate=0.05,
                max_depth=6,
                min_samples_leaf=20,
                max_iter=150,                           # n_estimators → max_iter for Hist
                random_state=random_state,
            ),
            param_distributions={
                "clf__learning_rate":    np.array([0.03, 0.05, 0.1], dtype=float),
                "clf__max_depth":        np.array([4, 6, 8], dtype=int),
                "clf__min_samples_leaf": np.array([10, 20, 40], dtype=int),
                "clf__max_iter":         np.array([100, 150, 200], dtype=int),
                "clf__l2_regularization": np.array([0.0, 0.1, 0.5], dtype=float),
            },
            search_iterations=10,
            feature_overrides={"tfidf_max_features": 8000, "include_text_statistics": True},
            select_k_best=1024,
            requires_dense=True,
        ),
        "stacking": ModelSpec(
            key="stacking",
            display_name="Stacking Ensemble",
            estimator=_build_stacking_estimator(random_state),
            param_distributions={
                "clf__lr__C": np.array([0.5, 1.0, 2.0], dtype=float),
                "clf__svm__C": np.array([0.5, 1.0, 2.0], dtype=float),
                "clf__nb__alpha": np.array([0.3, 0.5, 0.8], dtype=float),
                "clf__final_estimator__C": np.array([0.5, 1.0, 2.0, 4.0], dtype=float),
            },
            search_iterations=12,  # ← was 6
            feature_overrides={"tfidf_max_features": 12000, "include_text_statistics": True},
        ),
    }


def build_search(
    pipeline,
    spec: ModelSpec,
    search_iterations: int | None = None,
    random_state: int = TRAINING_CONFIG.random_state,
):
    cv = StratifiedKFold(n_splits=TRAINING_CONFIG.cv_folds, shuffle=True, random_state=random_state)
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=spec.param_distributions,
        n_iter=min(search_iterations or spec.search_iterations, max(len(spec.param_distributions), 1) * 4),
        scoring=MACRO_F1,
        refit=True,
        cv=cv,
        random_state=random_state,
        n_jobs=TRAINING_CONFIG.n_jobs,
        verbose=0,
    )

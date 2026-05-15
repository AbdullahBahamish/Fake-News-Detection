from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .bert_runner import resolve_bert_config, run_bert_training
from .config import FEATURE_CONFIG, PATHS, FeatureConfig
from .data import DatasetBundle, load_binary_liar_dataset, summarize_dataset
from .evaluate import evaluate_split
from .features import build_feature_pipeline
from .modeldefs import ALLOWED_MODEL_KEYS, build_search, get_model_specs, normalize_model_key
from .utils import configure_logging, ensure_results_directories, estimate_matrix_size_mb, set_global_seed


LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-model fake news detection training entrypoint")
    parser.add_argument(
        "--model",
        default="random_forest",
        help="Single model to run: bert, rf, rbf, gb, stack",
    )
    parser.add_argument("--skip-tuning", action="store_true", help="Disable RandomizedSearchCV for classical models.")
    parser.add_argument("--search-iterations", type=int, default=None, help="Override the per-model tuning budget.")
    parser.add_argument("--char-ngrams", action="store_true", help="Enable character n-grams on top of the model defaults.")
    parser.add_argument("--tfidf-max-features", type=int, default=None, help="Override the word TF-IDF vocabulary cap.")
    parser.add_argument("--char-max-features", type=int, default=None, help="Override the character n-gram vocabulary cap.")
    parser.add_argument("--sample-fraction", type=float, default=1.0, help="Optional split-wise downsampling for quick experiments.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--report-path", type=str, default=None, help="Optional explicit JSON output path.")
    parser.add_argument("--bert-max-epochs", type=int, default=None)
    parser.add_argument("--bert-batch-size", type=int, default=None)
    parser.add_argument("--bert-max-length", type=int, default=None)
    return parser


def _densify_if_needed(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return matrix


def _normalize_sample_fraction(sample_fraction: float) -> float:
    if not 0 < sample_fraction <= 1.0:
        raise ValueError("--sample-fraction must be in the interval (0, 1].")
    return float(sample_fraction)


def _sample_split(frame: pd.DataFrame, labels: pd.Series, sample_fraction: float, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    if sample_fraction >= 1.0:
        return frame.reset_index(drop=True), labels.reset_index(drop=True)

    sampled_index_parts = []
    label_frame = labels.to_frame("label")
    for offset, (_, values) in enumerate(label_frame.groupby("label")):
        sample_size = max(1, int(round(len(values) * sample_fraction)))
        sampled_index_parts.append(values.sample(n=sample_size, random_state=random_state + offset).index)

    sampled_index = pd.Index(sorted(pd.Index(np.concatenate([part.to_numpy() for part in sampled_index_parts]))))

    sampled_frame = frame.loc[sampled_index].reset_index(drop=True)
    sampled_labels = labels.loc[sampled_index].reset_index(drop=True)
    return sampled_frame, sampled_labels


def sample_bundle(bundle: DatasetBundle, sample_fraction: float, random_state: int) -> DatasetBundle:
    sample_fraction = _normalize_sample_fraction(sample_fraction)
    if sample_fraction >= 1.0:
        return bundle

    train, y_train = _sample_split(bundle.train, bundle.y_train, sample_fraction, random_state)
    valid, y_valid = _sample_split(bundle.valid, bundle.y_valid, sample_fraction, random_state + 1)
    test, y_test = _sample_split(bundle.test, bundle.y_test, sample_fraction, random_state + 2)

    audit = copy.deepcopy(bundle.audit)
    sampled_split_sizes = {
        "train": int(len(train)),
        "valid": int(len(valid)),
        "test": int(len(test)),
    }
    sampled_binary_distribution = {
        "train": {"fake": int((y_train == 0).sum()), "true": int((y_train == 1).sum())},
        "valid": {"fake": int((y_valid == 0).sum()), "true": int((y_valid == 1).sum())},
        "test": {"fake": int((y_test == 0).sum()), "true": int((y_test == 1).sum())},
    }
    audit["sample_fraction"] = sample_fraction
    audit["sampled_split_sizes"] = sampled_split_sizes
    audit["sampled_binary_label_distribution"] = sampled_binary_distribution
    audit["split_sizes"] = sampled_split_sizes
    audit["binary_label_distribution"] = sampled_binary_distribution

    return DatasetBundle(
        train=train,
        valid=valid,
        test=test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        audit=audit,
        class_names=bundle.class_names,
    )


def build_feature_config(args, spec) -> FeatureConfig:
    config_values = FEATURE_CONFIG.__dict__.copy()
    config_values.update(spec.feature_overrides)
    if args.char_ngrams:
        config_values["use_char_ngrams"] = True
    if args.tfidf_max_features is not None:
        config_values["tfidf_max_features"] = int(args.tfidf_max_features)
    if args.char_max_features is not None:
        config_values["char_max_features"] = int(args.char_max_features)
    return FeatureConfig(**config_values)


def build_model_pipeline(spec, feature_config: FeatureConfig, random_state: int) -> Pipeline:
    steps = [("features", build_feature_pipeline(feature_config))]
    if spec.select_k_best is not None:
        steps.append(("select", SelectKBest(score_func=chi2, k=spec.select_k_best)))
    if spec.use_svd:
        steps.append(("svd", TruncatedSVD(n_components=spec.svd_components, random_state=random_state)))
    if spec.requires_dense and not spec.use_svd:
        steps.append(("to_dense", FunctionTransformer(_densify_if_needed, accept_sparse=True)))
    if spec.scale_dense:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", copy.deepcopy(spec.estimator)))
    return Pipeline(steps=steps)


def _transform_model_inputs(fitted_pipeline: Pipeline, frame: pd.DataFrame):
    return fitted_pipeline[:-1].transform(frame)


def _default_report_path(model_key: str, reports_dir: Path) -> Path:
    return reports_dir / f"{model_key}_report.json"


def _save_report(report: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compact = copy.deepcopy(report)
    report_text = json.dumps(compact, indent=2)
    try:
        output_path.write_text(report_text, encoding="utf-8")
        return output_path
    except PermissionError:
        timestamp = int(time.time())
        fallback_paths = [
            output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}"),
            Path.cwd() / f"{output_path.stem}_{timestamp}{output_path.suffix}",
        ]
        for fallback_path in fallback_paths:
            try:
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                fallback_path.write_text(report_text, encoding="utf-8")
                LOGGER.warning("Primary report path was locked; wrote report to %s instead.", fallback_path)
                return fallback_path
            except PermissionError:
                continue
        raise


def fit_and_evaluate_classical_model(model_key: str, bundle: DatasetBundle, args) -> dict:
    specs = get_model_specs(random_state=args.random_state)
    if model_key not in specs:
        raise ValueError(f"Unknown classical model '{model_key}'. Available: {', '.join(sorted(specs))}")

    spec = specs[model_key]
    feature_config = build_feature_config(args, spec)
    LOGGER.info("Training %s", spec.display_name)
    start_time = time.perf_counter()

    training_pipeline = build_model_pipeline(spec, feature_config, args.random_state)
    best_params = None
    cv_best_score = None

    if args.skip_tuning:
        training_pipeline.fit(bundle.train, bundle.y_train)
        fitted_validation_pipeline = training_pipeline
    else:
        search = build_search(
            pipeline=training_pipeline,
            spec=spec,
            search_iterations=args.search_iterations,
            random_state=args.random_state,
        )
        search.fit(bundle.train, bundle.y_train)
        fitted_validation_pipeline = search.best_estimator_
        best_params = search.best_params_
        cv_best_score = float(search.best_score_)

    valid_metrics = evaluate_split(fitted_validation_pipeline, bundle.valid, bundle.y_valid, bundle.class_names)

    final_pipeline = build_model_pipeline(spec, feature_config, args.random_state)
    if best_params:
        final_pipeline.set_params(**best_params)

    train_valid = pd.concat([bundle.train, bundle.valid], ignore_index=True)
    y_train_valid = pd.concat([bundle.y_train, bundle.y_valid], ignore_index=True)
    final_pipeline.fit(train_valid, y_train_valid)

    model_inputs = _transform_model_inputs(final_pipeline, train_valid)
    test_metrics = evaluate_split(final_pipeline, bundle.test, bundle.y_test, bundle.class_names)
    training_time_seconds = float(time.perf_counter() - start_time)

    return {
        "model": spec.key,
        "display_name": spec.display_name,
        "best_params": best_params,
        "cv_best_macro_f1": cv_best_score,
        "training_time_seconds": training_time_seconds,
        "sample_fraction": float(args.sample_fraction),
        "feature_config": {
            "tfidf_max_features": feature_config.tfidf_max_features,
            "tfidf_ngram_range": list(feature_config.tfidf_ngram_range),
            "use_char_ngrams": feature_config.use_char_ngrams,
            "char_max_features": feature_config.char_max_features,
            "include_text_statistics": feature_config.include_text_statistics,
        },
        "model_input_shape": [int(value) for value in model_inputs.shape],
        "model_input_memory_mb": estimate_matrix_size_mb(model_inputs),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "dataset_audit": bundle.audit,
    }


def fit_and_evaluate_bert(bundle: DatasetBundle, args) -> dict:
    config = resolve_bert_config(
        max_epochs=args.bert_max_epochs,
        batch_size=args.bert_batch_size,
        max_length=args.bert_max_length,
    )
    report = run_bert_training(bundle, config=config, seed=args.random_state)
    report["sample_fraction"] = float(args.sample_fraction)
    report["dataset_audit"] = bundle.audit
    report["valid_metrics"] = {
        "best_epoch": report["best_epoch"],
        "history": report["history"],
    }
    report["test_metrics"] = report.pop("metrics")
    report["best_params"] = None
    report["cv_best_macro_f1"] = None
    return report


def print_run_summary(report: dict) -> None:
    test_metrics = report["test_metrics"]
    print("\nRun summary")
    print(f"Model: {report['display_name']}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1-score: {test_metrics['f1']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    roc_auc = test_metrics.get("roc_auc")
    print(f"ROC-AUC: {'-' if roc_auc is None else f'{roc_auc:.4f}'}")
    print(f"Training time (s): {report['training_time_seconds']:.2f}")
    if "model_input_shape" in report:
        print(f"Model input shape: {tuple(report['model_input_shape'])}")
        print(f"Model input memory (MB): {report['model_input_memory_mb']:.2f}")


def run_pipeline(args) -> dict:
    configure_logging()
    set_global_seed(args.random_state)
    output_dirs = ensure_results_directories()

    bundle = load_binary_liar_dataset(PATHS.raw_data_dir)
    bundle = sample_bundle(bundle, args.sample_fraction, args.random_state)
    LOGGER.info("\n%s", summarize_dataset(bundle))

    model_key = normalize_model_key(args.model)
    if model_key not in ALLOWED_MODEL_KEYS:
        allowed = ", ".join(sorted(ALLOWED_MODEL_KEYS))
        raise ValueError(f"Unsupported model '{args.model}'. Allowed models: {allowed}")
    if model_key == "bert":
        report = fit_and_evaluate_bert(bundle, args)
    else:
        report = fit_and_evaluate_classical_model(model_key, bundle, args)

    report_path = Path(args.report_path) if args.report_path else _default_report_path(model_key, output_dirs["reports"])
    saved_report = _save_report(report, report_path)
    print_run_summary(report)
    print(f"Report saved to: {saved_report}")

    return {
        "report": report,
        "report_path": str(saved_report),
    }


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()

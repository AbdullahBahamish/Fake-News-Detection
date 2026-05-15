# cross_validate_models.py

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import load_binary_liar_dataset
from src.train import build_model_pipeline, build_feature_config
from src.modeldefs import get_model_specs
from src.config import PATHS
from sklearn.metrics import f1_score
macro_f1 = lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro", zero_division=0)

from sklearn.metrics import make_scorer


SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "macro_f1": make_scorer(macro_f1),
}


def run_cross_validation():
    bundle = load_binary_liar_dataset(PATHS.raw_data_dir)

    # merge train + valid
    X = pd.concat([bundle.train, bundle.valid], ignore_index=True)
    y = pd.concat([bundle.y_train, bundle.y_valid], ignore_index=True)

    specs = get_model_specs(random_state=42)

    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42,
    )

    all_results = []

    for model_key, spec in specs.items():

        print(f"\nRunning CV for: {spec.display_name}")

        feature_config = build_feature_config(
            args=type(
                "Args",
                (),
                {
                    "char_ngrams": False,
                    "tfidf_max_features": None,
                    "char_max_features": None,
                },
            )(),
            spec=spec,
        )

        pipeline = build_model_pipeline(
            spec,
            feature_config,
            random_state=42,
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            return_train_score=False,
        )

        result = {
            "model": spec.display_name,
        }

        for metric in SCORING.keys():
            values = scores[f"test_{metric}"]

            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = float(np.std(values))

        all_results.append(result)

    df = pd.DataFrame(all_results)

    print("\n")
    print(df.round(4))

    output_path = "results/cross_validation_results.csv"
    df.to_csv(output_path, index=False)

    json_path = "results/cross_validation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved to:\n{output_path}\n{json_path}")


if __name__ == "__main__":
    run_cross_validation()
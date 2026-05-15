from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_full_metric_comparison(summary: pd.DataFrame, plots_dir: Path) -> Path:
    metric_columns = [
        ("test_accuracy", "Accuracy"),
        ("test_precision", "Precision"),
        ("test_recall", "Recall"),
        ("test_f1", "F1"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    x_positions = np.arange(len(summary))
    width = 0.18
    palette = sns.color_palette("deep", n_colors=len(metric_columns))

    for index, (column, label) in enumerate(metric_columns):
        offsets = x_positions + (index - (len(metric_columns) - 1) / 2) * width
        ax.bar(offsets, summary[column].to_numpy(dtype=float), width=width, label=label, color=palette[index])

    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary["display_name"].tolist(), rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_title("Performance Comparison Across Models")
    ax.legend(frameon=False, ncol=4, loc="upper center")
    sns.despine(ax=ax)

    output_path = plots_dir / "full_metric_comparison.png"
    _save_figure(fig, output_path)
    return output_path


def plot_training_time(summary: pd.DataFrame, plots_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.set_theme(style="whitegrid")

    scatter = ax.scatter(
        summary["training_time_seconds"].to_numpy(dtype=float),
        summary["test_f1"].to_numpy(dtype=float),
        c=summary["test_accuracy"].to_numpy(dtype=float),
        cmap="viridis",
        s=100,
        edgecolors="black",
        linewidths=0.6,
    )

    for _, row in summary.iterrows():
        ax.annotate(
            row["display_name"],
            (float(row["training_time_seconds"]), float(row["test_f1"])),
            xytext=(6, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Training Time vs F1 Score")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Accuracy")
    sns.despine(ax=ax)

    output_path = plots_dir / "training_time.png"
    _save_figure(fig, output_path)
    return output_path


def plot_multiclass_confusion_matrix(report: dict, plots_dir: Path) -> Path:
    matrix = np.asarray(report["test_metrics"]["confusion_matrix"], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    display_name = str(report.get("display_name", report.get("model", "model")))
    model_key = str(report.get("model", display_name)).strip().lower().replace(" ", "_")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.set_theme(style="whitegrid")
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Normalized Proportion"},
        xticklabels=["Fake", "True"],
        yticklabels=["Fake", "True"],
        linewidths=0.5,
        square=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Normalized Confusion Matrix: {display_name}")

    output_path = plots_dir / f"cm_{model_key}.png"
    _save_figure(fig, output_path)
    return output_path


def plot_category_distribution(distribution, plots_dir: Path) -> Path:
    if isinstance(distribution, pd.Series):
        labels = distribution.index.astype(str).tolist()
        values = distribution.to_numpy(dtype=float)
    elif isinstance(distribution, dict):
        labels = [str(key) for key in distribution.keys()]
        values = np.asarray(list(distribution.values()), dtype=float)
    else:
        raise TypeError("distribution must be a dict or pandas Series")

    fig, ax = plt.subplots(figsize=(7, 7))
    sns.set_theme(style="white")
    colors = sns.color_palette("pastel", n_colors=len(values))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1.0},
        textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_color("black")
    ax.set_title("Category Distribution")
    ax.axis("equal")

    output_path = plots_dir / "category_distribution.png"
    _save_figure(fig, output_path)
    return output_path


def plot_imbalance_curve(curves, plots_dir: Path) -> Path:
    if not isinstance(curves, dict):
        raise TypeError("curves must be a dict mapping model names to curve values")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("deep", n_colors=len(curves))

    for color, (model_name, values) in zip(palette, curves.items()):
        if isinstance(values, pd.Series):
            x_values = values.index.to_numpy(dtype=float)
            y_values = values.to_numpy(dtype=float)
        elif isinstance(values, pd.DataFrame):
            if {"percentage", "score"}.issubset(values.columns):
                x_values = values["percentage"].to_numpy(dtype=float)
                y_values = values["score"].to_numpy(dtype=float)
            else:
                raise ValueError("curve DataFrame values must contain 'percentage' and 'score' columns")
        elif isinstance(values, dict):
            x_values = np.asarray(list(values.keys()), dtype=float)
            y_values = np.asarray(list(values.values()), dtype=float)
        else:
            raise TypeError("each curve value must be a pandas Series, DataFrame, or dict")

        order = np.argsort(x_values)
        ax.plot(x_values[order], y_values[order], marker="o", linewidth=2, color=color, label=str(model_name))

    ax.set_xlabel("Data Percentage")
    ax.set_ylabel("Performance Score")
    ax.set_title("Performance vs Data Percentage")
    ax.legend(frameon=False)
    sns.despine(ax=ax)

    output_path = plots_dir / "imbalance_curve.png"
    _save_figure(fig, output_path)
    return output_path

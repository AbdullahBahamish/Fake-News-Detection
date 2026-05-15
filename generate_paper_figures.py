from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ========================= CONFIG =========================
REPORTS_DIRNAME = Path("results") / "reports"
FIGURES_DIRNAME = Path("results") / "figures"
SAVE_DPI = 300
ALLOWED_MODELS = {"bert", "random_forest", "svm_rbf", "gradient_boosting", "stacking"}


def find_project_root(start: Path) -> Path:
    """Find project root by looking for results/reports directory."""
    for candidate in [start, *start.parents]:
        if (candidate / REPORTS_DIRNAME).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate '{REPORTS_DIRNAME}' from '{start}'. "
        "Please run this script from the project root."
    )


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
REPORTS_DIR = PROJECT_ROOT / REPORTS_DIRNAME
FIGURES_DIR = PROJECT_ROOT / FIGURES_DIRNAME


def configure_style() -> None:
    """Apply clean academic plotting style."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": SAVE_DPI,
        "savefig.dpi": SAVE_DPI,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
    })


def load_reports(reports_dir: Path) -> list[dict]:
    """Load valid report JSONs and skip corrupted ones."""
    reports: list[dict] = []
    for report_path in sorted(reports_dir.glob("*.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            warnings.warn(f"Skipping malformed JSON: {report_path.name}")
            continue

        if not isinstance(report, dict):
            continue

        test_metrics = report.get("test_metrics")
        if not report.get("display_name") or not isinstance(test_metrics, dict):
            warnings.warn(f"Skipping incomplete report: {report_path.name}")
            continue

        model_key = str(report.get("model", "")).strip().lower()
        if model_key not in ALLOWED_MODELS:
            warnings.warn(f"Skipping non-approved model report: {report_path.name}")
            continue

        if not {"accuracy", "macro_f1", "confusion_matrix"}.issubset(test_metrics):
            continue

        report["_source_path"] = report_path
        report["_short_name"] = build_short_name(report)
        reports.append(report)

    if not reports:
        raise RuntimeError(f"No valid reports found in {reports_dir}")

    return reports


def build_short_name(report: dict) -> str:
    key = str(report.get("model", "")).lower().strip()
    mapping = {
        "bert": "BERT",
        "random_forest": "RF",
        "svm_rbf": "SVM-RBF",
        "gradient_boosting": "GB",
        "stacking": "STACK",
    }
    return mapping.get(key, report.get("display_name", key)[:15])


def macro_f1(report: dict) -> float:
    return float(report["test_metrics"]["macro_f1"])


def accuracy(report: dict) -> float:
    return float(report["test_metrics"]["accuracy"])


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def sorted_reports(reports: list[dict]) -> list[dict]:
    return sorted(reports, key=lambda r: macro_f1(r), reverse=True)


def add_bar_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)


# ====================== PLOTTING FUNCTIONS ======================

def plot_performance_comparison(reports: list[dict], figures_dir: Path) -> None:
    ordered = sorted_reports(reports)
    labels = [r["_short_name"] for r in ordered]
    accs = [accuracy(r) for r in ordered]
    f1s = [macro_f1(r) for r in ordered]

    x = np.arange(len(ordered))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#355C7D")
    bars2 = ax.bar(x + width/2, f1s, width, label="Macro F1-score", color="#C06C84")

    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Performance Comparison of Models on LIAR Dataset (Test Set)")
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax)

    save_figure(fig, figures_dir / "performance_comparison")


def plot_training_time_vs_performance(reports: list[dict], figures_dir: Path) -> None:
    ordered = sorted_reports(reports)
    times = np.array([float(r.get("training_time_seconds", 0)) for r in ordered])
    f1s = np.array([macro_f1(r) for r in ordered])
    accs = np.array([accuracy(r) for r in ordered])

    fig, ax = plt.subplots(figsize=(10, 6.5))
    scatter = ax.scatter(times, f1s, c=accs, cmap="viridis", s=130, edgecolors="black", linewidth=0.7)

    for r, t, f in zip(ordered, times, f1s):
        ax.annotate(r["_short_name"], (t, f), xytext=(7, 7), textcoords="offset points",
                    fontsize=9, weight="semibold")

    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Macro F1-score")
    ax.set_title("Training Time vs Macro F1-score Trade-off")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Accuracy")
    sns.despine(ax=ax)

    save_figure(fig, figures_dir / "training_time_vs_performance_paper")


def plot_confusion_matrices_grid(reports: list[dict], figures_dir: Path) -> None:
    selected = sorted_reports(reports)[:5]
    # Ensure BERT is included
    bert = next((r for r in reports if r.get("model") == "bert"), None)
    if bert and bert not in selected:
        selected = selected[:4] + [bert]

    n = len(selected)
    cols = 2 if n <= 4 else 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, report in zip(axes, selected):
        matrix = np.array(report["test_metrics"]["confusion_matrix"])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_xticklabels(["Fake", "True"])
        ax.set_yticklabels(["Fake", "True"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{report['_short_name']} (Macro F1: {macro_f1(report):.3f})", fontsize=11)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Confusion Matrices of Top Models", fontsize=15, y=1.02)
    save_figure(fig, figures_dir / "confusion_matrices")


def plot_roc_curves(reports: list[dict], figures_dir: Path) -> None:
    eligible = [r for r in sorted_reports(reports)
                if isinstance(r.get("test_metrics", {}).get("roc_curve"), dict)]

    fig, ax = plt.subplots(figsize=(9, 7))

    if eligible:
        palette = sns.color_palette("tab10", len(eligible))
        for color, report in zip(palette, eligible):
            curve = report["test_metrics"]["roc_curve"]
            auc = report["test_metrics"].get("roc_auc")
            label = f"{report['_short_name']} (AUC={float(auc):.3f})" if auc else report["_short_name"]
            ax.plot(curve["fpr"], curve["tpr"], label=label, color=color)

        ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
        ax.legend(frameon=False, loc="lower right")
    else:
        ax.text(0.5, 0.5, "No ROC curve arrays were found in the loaded report files.\n"
                          "Re-run the supported models after saving roc_curve data into results/reports/.",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves Comparison on LIAR Test Set")
    sns.despine(ax=ax)

    save_figure(fig, figures_dir / "roc_curves_paper")


def print_report_summary(reports: list[dict]) -> None:
    print(f"\nLoaded {len(reports)} valid report(s) from: {REPORTS_DIR}")
    for r in sorted_reports(reports):
        print(f"   {r['_short_name']:>8} | Acc={accuracy(r):.3f} | Macro F1={macro_f1(r):.3f}")


def main() -> None:
    configure_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    reports = load_reports(REPORTS_DIR)
    print_report_summary(reports)

    print("\nGenerating figures...")
    plot_performance_comparison(reports, FIGURES_DIR)
    plot_training_time_vs_performance(reports, FIGURES_DIR)
    plot_confusion_matrices_grid(reports, FIGURES_DIR)
    plot_roc_curves(reports, FIGURES_DIR)          # ← Fixed: now properly called

    print(f"\nAll figures successfully saved to:\n   {FIGURES_DIR}")


if __name__ == "__main__":
    main()

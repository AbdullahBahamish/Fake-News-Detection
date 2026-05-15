# Fake News Detection using Machine Learning and Transformer Models

A modular research-oriented fake news detection framework built on the LIAR dataset.  
The project compares multiple classical machine learning algorithms and transformer-based architectures for binary fake news classification using textual and metadata-driven features.

The repository was designed for reproducible experimentation, leakage-aware evaluation, and publication-quality result generation.

---

# Project Objectives

This project aims to:

- Detect fake versus true political statements
- Compare multiple supervised learning approaches
- Evaluate performance using consistent metrics
- Investigate trade-offs between accuracy, interpretability, and computational cost
- Generate reproducible experimental reports and publication-ready figures

---

# Implemented Models

| Model Key | Model |
|---|---|
| `bert` | BERT-base fine-tuning baseline |
| `rf` | Random Forest with TF-IDF features |
| `rbf` | Support Vector Machine with RBF kernel |
| `gb` | Gradient Boosting with engineered metadata features |
| `stack` | Stacking ensemble classifier |

---

# Dataset

Dataset: LIAR Fake News Dataset

The dataset contains short political statements labelled with truthfulness categories.

Original labels are converted into binary classes:

| Binary Label | Original Labels |
|---|---|
| Fake | false, pants-fire, barely-true |
| True | true, mostly-true, half-true |

Dataset files exist inside:

```text
data/raw/
├── train.tsv
├── valid.tsv
└── test.tsv
```

---

# Key Features

## Leakage Prevention

The pipeline performs:

- within-split deduplication
- cross-split statement overlap removal
- normalized text matching
- speaker-aware duplicate filtering

This reduces evaluation contamination and improves experimental validity.

---

## Feature Engineering

The classical ML pipelines include:

- TF-IDF word n-grams
- optional character n-grams
- speaker credibility encoding
- categorical metadata encoding
- engineered text statistics
- lie-ratio numerical feature

Metadata features include:

- speaker
- party affiliation
- state
- context
- subject category
- historical truthfulness counts

---

## Hyperparameter Search

Classical models use:

- RandomizedSearchCV
- Stratified K-Fold cross-validation
- macro F1-score optimization

---

## Automated Evaluation

Generated reports include:

- accuracy
- precision
- recall
- F1-score
- macro F1-score
- ROC-AUC
- confusion matrices
- runtime statistics
- memory estimates
- dataset audit summaries

---

# Repository Structure

```text
Fake-News-Detection/
│
├── data/
│   └── raw/
│       ├── train.tsv
│       ├── valid.tsv
│       └── test.tsv
│
├── docs/
├── models/
├── outputs/
├── results/
│   ├── figures/
│   ├── reports/
│   └── tables/
│
├── src/
│   ├── models/
│   ├── bert_runner.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── modeldefs.py
│   ├── train.py
│   ├── utils.py
│   └── visualize_results.py
│
├── generate_paper_figures.py
├── main.py
├── train.py
└── requirements.txt
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/AbdullahBahamish/Fake-News-Detection.git
cd Fake-News-Detection
```

## 2. Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Main libraries:

- scikit-learn
- pandas
- numpy
- scipy
- torch
- transformers
- matplotlib
- seaborn

---

# Running Models

## Random Forest

```bash
python main.py --model rf
```

---

## SVM with RBF Kernel

```bash
python main.py --model rbf
```

---

## Gradient Boosting

```bash
python main.py --model gb
```

---

## Stacking Ensemble

```bash
python main.py --model stack
```

---

## BERT Fine-Tuning

```bash
python main.py --model bert
```

---

# Important Command-Line Arguments

| Argument | Description |
|---|---|
| `--sample-fraction` | Percentage of dataset used |
| `--skip-tuning` | Disable hyperparameter search |
| `--search-iterations` | Override tuning iterations |
| `--char-ngrams` | Enable character n-grams |
| `--tfidf-max-features` | Override TF-IDF vocabulary size |
| `--bert-max-epochs` | BERT training epochs |
| `--bert-batch-size` | BERT batch size |
| `--bert-max-length` | Maximum token length |

---

# Example Commands

## Full Dataset Random Forest

```bash
python main.py --model rf --sample-fraction 1.0
```

---

## Faster Classical Training

```bash
python main.py --model gb --skip-tuning
```

---

## Stronger Hyperparameter Search

```bash
python main.py --model stack --search-iterations 12
```

---

## BERT with Custom Epochs

```bash
python main.py --model bert --bert-max-epochs 4
```

---

# Offline BERT Usage

If the Hugging Face model is already cached locally:

```powershell
$env:HF_HUB_OFFLINE="1"
python main.py --model bert
```

To pre-download models:

```bash
python download_models.py
```

---

# Generated Outputs

Reports are stored in:

```text
results/reports/
```

Figures are stored in:

```text
results/figures/
```

Tables are stored in:

```text
results/tables/
```

---

# Generate Publication Figures

```bash
python generate_paper_figures.py
```

Generated figures include:

- model performance comparison
- ROC curves
- confusion matrices
- training-time trade-off plots

---

# Experimental Design

## Validation Strategy

- Stratified K-Fold cross-validation
- separate train/validation/test splits
- leakage-aware preprocessing

---

## Evaluation Metrics

Classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- ROC-AUC

---

# Hardware Notes

The project was intentionally designed to support:

- CPU-only environments
- low-memory systems
- staged experimentation
- partial dataset sampling

BERT training remains substantially heavier than the classical pipelines.

---

# Research Context

This repository was developed as part of a comparative machine learning research project focused on fake news detection using both traditional machine learning and transformer-based architectures.

---

# Authors

Abdullah Bahamish
Computer Science Department

---

# License

This repository is intended for academic and research purposes.
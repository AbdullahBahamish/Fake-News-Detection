# Fake News Detection

This project trains and evaluates fake news detection models on the LIAR dataset with one model per run. The workflow is designed for limited-resource machines, so the requested algorithms are isolated instead of being launched together.

## Implemented Models

- `bert`: BERT-base fine-tuning baseline (`bert-base-uncased`)
- `rf`: Random Forest with TF-IDF features
- `rbf`: SVM with RBF kernel using TF-IDF + Truncated SVD
- `gb`: Gradient Boosting with engineered text and metadata features
- `stack`: Stacking ensemble of multiple classifiers

## Why The Runs Are Split

- BERT fine-tuning is much heavier than the classical models.
- Random Forest and Gradient Boosting need dense inputs, so their feature spaces are compressed before fitting.
- RBF SVM is trained on an SVD-reduced representation to keep memory and runtime under control.
- The stacking model is also a separate run, because it internally trains several base classifiers.

## Install

```bash
pip install -r requirements.txt
```

For BERT on restricted or offline environments, cache `bert-base-uncased` first or set `HF_HUB_OFFLINE=1` when the model is already cached locally.

## Run One Model

```bash
python main.py --model rf
python main.py --model rbf
python main.py --model gb
python main.py --model stack
python main.py --model bert
```

Equivalent model-specific entrypoints:

```bash
python -m src.models.random_forest
python -m src.models.svm_rbf
python -m src.models.gradient_boosting
python -m src.models.stacking
python -m src.models.bert
```

## Useful Flags

```bash
python main.py --model rf --skip-tuning
python main.py --model gb --sample-fraction 0.25
python main.py --model rbf --char-ngrams
python main.py --model bert --bert-max-epochs 2 --bert-batch-size 4 --bert-max-length 128
```

## Output

Each run writes one JSON report to `results/reports/`, for example:

- `results/reports/random_forest_report.json`
- `results/reports/svm_rbf_report.json`
- `results/reports/gradient_boosting_report.json`
- `results/reports/stacking_report.json`
- `results/reports/bert_report.json`

Reports include dataset audit information, validation results, test metrics, training time, and model-input memory estimates for the classical pipelines.

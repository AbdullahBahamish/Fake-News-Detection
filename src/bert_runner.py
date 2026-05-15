from __future__ import annotations

import copy
import os
import time
import math   
from dataclasses import dataclass

import numpy as np
from pathlib import Path  

_DEFAULT_LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def download_model_if_needed(model_name: str, save_dir: Path | None = None) -> Path:
    """
    Downloads model and tokenizer to a local folder if not already present.
    Returns the local path so it can be used for offline loading.
    Call this once on a network that allows HuggingFace, then never again.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    save_dir = save_dir or _DEFAULT_LOCAL_MODEL_DIR
    model_dir = save_dir / model_name.replace("/", "_")

    # already downloaded — skip
    if (model_dir / "config.json").exists():
        return model_dir

    print(f"Downloading {model_name} to {model_dir} ...")
    model_dir.mkdir(parents=True, exist_ok=True)

    AutoTokenizer.from_pretrained(model_name).save_pretrained(model_dir)
    AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).save_pretrained(model_dir)

    print(f"Saved to {model_dir}")
    return model_dir

try:
    from .evaluate import compute_metrics
    from .utils import RANDOM_STATE, set_global_seed
except ImportError:
    from evaluate import compute_metrics
    from utils import RANDOM_STATE, set_global_seed


@dataclass(frozen=True)
class BertConfig:
    model_name: str =  "bert-base-uncased"  # ← back to BERT
    max_epochs: int = 5                    # already updated
    patience: int = 2                      # already updated
    batch_size: int = 8
    accumulation_steps: int = 4            # ← new — effective batch = 8 × 4 = 32
    max_length: int = 256                  # already updated
    learning_rate: float = 3e-5            # already updated
    weight_decay: float = 0.01
    classifier_dropout: float = 0.2        # ← new — was hardcoded 0.1 inside model


class BertDataset:
    def __init__(self, statements, labels, tokenizer, max_length):
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, index):
        import torch

        encoding = self.tokenizer(
            str(self.statements[index]) if self.statements[index] is not None else "",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


def _build_loader(df, labels, tokenizer, batch_size, max_length, shuffle=False, generator=None):
    from torch.utils.data import DataLoader

    def _format_input(row):
        speaker = str(row.get("speaker", "unknown") or "unknown")
        party   = str(row.get("party",   "unknown") or "unknown")
        job     = str(row.get("speaker_job", "unknown") or "unknown")
        stmt    = str(row.get("statement", "") or "")
        return f"[speaker: {speaker}] [party: {party}] [job: {job}] {stmt}"

    statements = df.apply(_format_input, axis=1).tolist()

    dataset = BertDataset(
        statements=statements,
        labels=np.asarray(labels, dtype=int).tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)

def _build_loss(labels, device):
    import torch
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    return torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))


def _train_epoch(model, loader, optimizer, scheduler, loss_fn, device, accumulation_steps: int = 1):
    import torch

    model.train()
    total_loss = 0.0
    optimizer.zero_grad()                          # ← moved outside the loop

    for batch_idx, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # divide loss so accumulated gradient == single large-batch gradient
        loss = loss_fn(outputs.logits, labels) / accumulation_steps
        loss.backward()

        total_loss += float(loss.item()) * accumulation_steps  # ← rescale back for logging

        is_last_batch      = (batch_idx + 1) == len(loader)
        is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0

        if is_accumulation_step or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(len(loader), 1)


def _evaluate(model, loader, loss_fn, device):
    import torch

    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            predictions = logits.argmax(dim=1)

            total_loss += float(loss.item())
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
            y_score.extend(probabilities.cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred, y_score, class_names=["fake", "true"])
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def resolve_bert_config(*, fast=False, tiny=False, max_epochs=None, batch_size=None, max_length=None):
    config = BertConfig()
    if fast:
        config = BertConfig(max_epochs=3, patience=1, batch_size=8, max_length=128)
    if tiny:
        config = BertConfig(max_epochs=2, patience=1, batch_size=8, max_length=96)
    if max_epochs is not None:
        config = BertConfig(**{**config.__dict__, "max_epochs": max_epochs})
    if batch_size is not None:
        config = BertConfig(**{**config.__dict__, "batch_size": batch_size})
    if max_length is not None:
        config = BertConfig(**{**config.__dict__, "max_length": max_length})
    return config


def _load_tokenizer_and_model(config, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    offline_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    load_attempts = [True] if offline_only else [True, False]
    last_error = None

    for local_files_only in load_attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                local_files_only=local_files_only,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=2,
                hidden_dropout_prob=config.classifier_dropout,
                attention_probs_dropout_prob=config.classifier_dropout,
                local_files_only=local_files_only,
            ).to(device)
            return tokenizer, model
        except OSError as exc:
            last_error = exc

    raise RuntimeError(
        "bert-base-uncased requires cached or downloadable weights. "
        "Set HF_HUB_OFFLINE=1 to force cached files only."
    ) from last_error


def run_bert_training(split, config: BertConfig, seed: int = RANDOM_STATE):
    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    start = time.time()
    set_global_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _load_tokenizer_and_model(config, device)

    y_train = split.y_train.to_numpy(dtype=int)
    y_valid = split.y_valid.to_numpy(dtype=int)
    y_test  = split.y_test.to_numpy(dtype=int)

    generator = torch.Generator().manual_seed(seed)
    train_loader = _build_loader(split.train, y_train, tokenizer, config.batch_size, config.max_length, shuffle=True, generator=generator)
    valid_loader = _build_loader(split.valid, y_valid, tokenizer, config.batch_size, config.max_length)
    test_loader  = _build_loader(split.test,  y_test,  tokenizer, config.batch_size, config.max_length)

    loss_fn   = _build_loss(y_train, device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # ── effective steps account for accumulation ──────────────────────────
    effective_steps_per_epoch = math.ceil(len(train_loader) / config.accumulation_steps)
    total_steps  = max(effective_steps_per_epoch * config.max_epochs, 1)
    warmup_steps = int(0.1 * total_steps)          # already updated previously
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_epoch  = 1
    best_macro_f1 = -1.0
    best_loss   = float("inf")
    best_state  = None
    epochs_without_improvement = 0
    history     = []

    for epoch in range(config.max_epochs):
        train_loss = _train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            accumulation_steps=config.accumulation_steps,   # ← new
        )
        valid_metrics = _evaluate(model, valid_loader, loss_fn, device)
        history.append({
            "epoch":           epoch + 1,
            "train_loss":      train_loss,
            "valid_loss":      valid_metrics["loss"],
            "valid_accuracy":  valid_metrics["accuracy"],
            "valid_macro_f1":  valid_metrics["macro_f1"],
        })

        improved = (
            valid_metrics["macro_f1"] > best_macro_f1
            or (valid_metrics["macro_f1"] == best_macro_f1 and valid_metrics["loss"] < best_loss)
        )
        if improved:
            best_epoch    = epoch + 1
            best_macro_f1 = valid_metrics["macro_f1"]
            best_loss     = valid_metrics["loss"]
            best_state    = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    test_metrics = _evaluate(model, test_loader, loss_fn, device)
    elapsed = time.time() - start

    return {
        "model":                    "bert",
        "display_name":             f"{config.model_name} (fine-tuned)",  # ← reflects actual model used
        "training_time_seconds":    float(elapsed),
        "best_epoch":               int(best_epoch),
        "config":                   config.__dict__,
        "history":                  history,
        "metrics":                  test_metrics,
    }
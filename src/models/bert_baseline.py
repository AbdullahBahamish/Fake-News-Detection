import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import load_data, get_binary_labels

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5


class LiarDataset(Dataset):
    def __init__(self, statements, labels, tokenizer, max_len):
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.statements[idx]),
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_loader(df, tokenizer):
    dataset = LiarDataset(
        df.statement.values,
        df.binary_label.values,
        tokenizer,
        MAX_LEN
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0

    for batch in loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids,
                attention_mask=attention_mask)

        logits = outputs.logits
        loss = loss_fct(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return total_loss / len(loader), correct / len(loader.dataset), y_true, y_pred


def run_bert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, valid_df, test_df = load_data()

    train_df["binary_label"] = get_binary_labels(train_df["label"])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["binary_label"]),
        y=train_df["binary_label"]
    )

    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fct = torch.nn.CrossEntropyLoss(weight=weights)

    valid_df["binary_label"] = get_binary_labels(valid_df["label"])
    test_df["binary_label"] = get_binary_labels(test_df["label"])

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    ).to(device)

    train_loader = create_loader(train_df, tokenizer)
    valid_loader = create_loader(valid_df, tokenizer)
    test_loader = create_loader(test_df, tokenizer)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        0,
        total_steps
    )

    best_val_acc = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, _, _ = evaluate(model, valid_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_bert_model.bin")

    model.load_state_dict(torch.load("best_bert_model.bin"))

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Fake", "True"]))


if __name__ == "__main__":
    run_bert()
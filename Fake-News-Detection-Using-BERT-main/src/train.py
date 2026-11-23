"""
Modernized Training Script for Fake News Detection.
Supports BERT, RoBERTa, DeBERTa, and DistilBERT with layer unfreezing,
mixed precision (AMP), comprehensive metric reporting, and optional Azure ML integration.
"""
import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    accuracy_score
)

# Optional Azure ML run context
try:
    from azureml.core import Run
    AZUREML_AVAILABLE = True
except ImportError:
    AZUREML_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train")


def train_baseline_pipeline(data_path: str, output_dir: str):
    """
    Trains and saves a calibrated TF-IDF + Logistic Regression baseline model.
    """
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    logger.info("--- Starting Fast Baseline Training ---")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "label"])
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs"
        ))
    ])

    pipeline.fit(train_df["text"].astype(str), train_df["label"].astype(int))

    preds = pipeline.predict(test_df["text"].astype(str))
    probs = pipeline.predict_proba(test_df["text"].astype(str))[:, 1]

    acc = accuracy_score(test_df["label"], preds)
    f1 = f1_score(test_df["label"], preds, average="weighted")
    mcc = matthews_corrcoef(test_df["label"], preds)
    auc = roc_auc_score(test_df["label"], probs)

    logger.info(f"Baseline Test Accuracy: {acc:.4f} | F1 (weighted): {f1:.4f} | MCC: {mcc:.4f} | AUC: {auc:.4f}")
    print(classification_report(test_df["label"], preds, target_names=["FAKE", "REAL"]))

    os.makedirs(output_dir, exist_ok=True)
    save_file = os.path.join(output_dir, "baseline_model.joblib")
    joblib.dump(pipeline, save_file)
    logger.info(f"Baseline saved to {save_file}")


def train_transformer_model(args):
    """
    Trains a modern Transformer (BERT / RoBERTa / DeBERTa) model using PyTorch.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
        from src.model import TransformerClassifier
    except ImportError as e:
        logger.error(f"Deep learning dependencies not satisfied: {e}. Please install PyTorch and Transformers.")
        logger.info("You can train a baseline model using `--baseline_only`.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Check for AzureML run context
    azure_run = None
    if AZUREML_AVAILABLE:
        try:
            azure_run = Run.get_context()
            if not hasattr(azure_run, "experiment"):
                azure_run = None
        except Exception:
            azure_run = None

    # Load Data
    logger.info(f"Loading data from: {args.data_path}")
    if os.path.isdir(args.data_path):
        candidate = os.path.join(args.data_path, "shorttextpreprocessedtrain.csv")
        if not os.path.exists(candidate):
            candidate = os.path.join(args.data_path, "newdatasetwithcoviddata.csv")
        data_file = candidate
    else:
        data_file = args.data_path

    df = pd.read_csv(data_file)
    df = df.dropna(subset=["text", "label"])
    if args.sample_size and len(df) > args.sample_size:
        logger.info(f"Subsampling to {args.sample_size} records for training.")
        df = df.sample(args.sample_size, random_state=42)

    logger.info(f"Total dataset shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")

    # Stratified split: Train (70%), Val (15%), Test (15%)
    train_text, temp_text, train_labels, temp_labels = train_test_split(
        df["text"].astype(str),
        df["label"].astype(int),
        random_state=42,
        test_size=0.3,
        stratify=df["label"]
    )
    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text,
        temp_labels,
        random_state=42,
        test_size=0.5,
        stratify=temp_labels
    )

    # Tokenizer
    logger.info(f"Loading Tokenizer for: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def encode_texts(texts):
        return tokenizer.batch_encode_plus(
            texts.tolist(),
            max_length=args.max_seq_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt"
        )

    logger.info("Encoding training sequences...")
    tokens_train = encode_texts(train_text)
    tokens_val = encode_texts(val_text)
    tokens_test = encode_texts(test_text)

    # DataLoaders
    train_data = TensorDataset(tokens_train["input_ids"], tokens_train["attention_mask"], torch.tensor(train_labels.tolist()))
    val_data = TensorDataset(tokens_val["input_ids"], tokens_val["attention_mask"], torch.tensor(val_labels.tolist()))
    test_data = TensorDataset(tokens_test["input_ids"], tokens_test["attention_mask"], torch.tensor(test_labels.tolist()))

    train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.batch_size)
    val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=args.batch_size)
    test_loader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=args.batch_size)

    # Model
    logger.info(f"Initializing TransformerClassifier ({args.model_name}) with unfreeze_layers={args.unfreeze_layers}")
    model = TransformerClassifier(
        model_name=args.model_name,
        num_classes=2,
        dropout_rate=args.dropout,
        unfreeze_layers=args.unfreeze_layers
    )
    model.to(device)

    # Loss with Class Weights
    class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels.values)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device.type == "cuda"))

    # Training Loop
    best_valid_loss = float("inf")
    train_losses, valid_losses = [], []
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        logger.info(f"--- Epoch {epoch + 1}/{args.num_epochs} ---")
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            sent_id, mask, labels = [t.to(device) for t in batch]
            model.zero_grad()

            with torch.cuda.amp.autocast(enabled=(args.use_amp and device.type == "cuda")):
                logits = model(sent_id, mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

            if (step + 1) % args.log_interval == 0:
                logger.info(f"Step {step + 1}/{len(train_loader)} - Batch Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                sent_id, mask, labels = [t.to(device) for t in batch]
                with torch.cuda.amp.autocast(enabled=(args.use_amp and device.type == "cuda")):
                    logits = model(sent_id, mask)
                    loss = criterion(logits, labels)
                total_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1 = f1_score(val_targets, val_preds, average="weighted")

        train_losses.append(avg_train_loss)
        valid_losses.append(avg_val_loss)

        logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

        if azure_run:
            azure_run.log("train_loss", avg_train_loss)
            azure_run.log("val_loss", avg_val_loss)
            azure_run.log("val_f1", val_f1)

        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            weights_path = os.path.join(args.output_dir, "saved_weights.pt")
            torch.save(model.state_dict(), weights_path)
            logger.info(f"Saved new best model checkpoint to {weights_path}")

    # Plot & save loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training & Validation Loss ({args.model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))

    # Final Evaluation on Test Set
    logger.info("--- Evaluating Best Checkpoint on Test Set ---")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "saved_weights.pt"), map_location=device))
    model.eval()

    test_preds, test_probs, test_targets = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            sent_id, mask, labels = [t.to(device) for t in batch]
            logits = model(sent_id, mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            test_probs.extend(probs[:, 1])
            test_preds.extend(preds)
            test_targets.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average="weighted")
    test_mcc = matthews_corrcoef(test_targets, test_preds)
    test_auc = roc_auc_score(test_targets, test_probs)

    logger.info(f"Test Accuracy: {test_acc:.4f} | F1: {test_f1:.4f} | MCC: {test_mcc:.4f} | ROC-AUC: {test_auc:.4f}")
    report = classification_report(test_targets, test_preds, target_names=["FAKE", "REAL"], output_dict=True)
    print(classification_report(test_targets, test_preds, target_names=["FAKE", "REAL"]))

    # Save metadata config
    config = {
        "model_name": args.model_name,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "unfreeze_layers": args.unfreeze_layers,
        "metrics": {
            "accuracy": test_acc,
            "f1_weighted": test_f1,
            "mcc": test_mcc,
            "roc_auc": test_auc
        }
    }
    with open(os.path.join(args.output_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Training completed successfully. Model and config saved in {args.output_dir}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Transformer model for Fake News Detection")
    parser.add_argument("--data_path", type=str, default="data/shorttextpreprocessedtrain.csv", help="Path to CSV dataset")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Hugging Face model backbone (bert-base-uncased, roberta-base, microsoft/deberta-v3-base)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save weights and artifacts")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum token length for tokenizer (default 128)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for AdamW")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for AdamW optimizer")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability for classification head")
    parser.add_argument("--unfreeze_layers", type=int, default=2, help="Number of top encoder layers to unfreeze (-1 for all, 0 for none)")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional sample size for quick experimentation")
    parser.add_argument("--log_interval", type=int, default=50, help="Batch log interval")
    parser.add_argument("--use_amp", action="store_true", help="Enable automatic mixed precision (AMP)")
    parser.add_argument("--baseline_only", action="store_true", help="Train fast Scikit-Learn baseline instead of Transformer")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.baseline_only:
        train_baseline_pipeline(args.data_path, args.output_dir)
    else:
        train_transformer_model(args)

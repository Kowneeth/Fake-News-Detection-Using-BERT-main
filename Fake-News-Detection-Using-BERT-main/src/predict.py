"""
Production-Ready Predictor for Fake News Detection.
Supports PyTorch Transformer models (BERT, RoBERTa, DeBERTa) and high-speed Scikit-Learn baseline.
"""
import os
import re
import time
import string
import logging
from typing import List, Dict, Any, Optional, Tuple
import joblib
import pandas as pd
import numpy as np

from src.explain import ExplainabilityEngine

logger = logging.getLogger(__name__)

# Check PyTorch availability
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from src.model import TransformerClassifier
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TransformerClassifier = None
    TORCH_AVAILABLE = False


def clean_text(text: str) -> str:
    """
    Modern text preprocessing suitable for both Transformers and Linear models.
    Removes raw URLs, twitter handles, and redundant whitespace, but preserves
    essential context (punctuation and casing indicators).
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Strip URLs (e.g., http://... or https://...)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # 2. Strip Twitter / Social user handles (@username)
    text = re.sub(r"@\w+", "", text)
    
    # 3. Strip HTML tags if present
    text = re.sub(r"<.*?>", "", text)
    
    # 4. Normalize multiple punctuation (e.g. "!!!!" -> "!!")
    text = re.sub(r"([!?]){2,}", r"\1\1", text)
    
    # 5. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


class FakeNewsPredictor:
    """
    High-level prediction service supporting PyTorch BERT checkpoints and Scikit-Learn baseline.
    """
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
        self.model = None
        self.tokenizer = None
        self.backend = "uninitialized"
        self.model_name = "Unknown"
        self.max_seq_len = 128
        self.explain_engine = None

        self._load_or_initialize(model_path)
        self.explain_engine = ExplainabilityEngine(predict_fn=self._predict_probabilities_batch)

    def _load_or_initialize(self, model_path: Optional[str]):
        """
        Loads the PyTorch transformer model or trained baseline.
        If neither exists, trains a lightweight baseline from local data.
        """
        # 1. Try PyTorch Transformer checkpoint
        torch_checkpoint = model_path or os.path.join("outputs", "saved_weights.pt")
        baseline_path = os.path.join("outputs", "baseline_model.joblib")

        if TORCH_AVAILABLE and os.path.exists(torch_checkpoint):
            try:
                logger.info(f"Loading PyTorch model checkpoint from {torch_checkpoint}")
                self.model_name = "bert-base-uncased"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = TransformerClassifier(model_name=self.model_name, unfreeze_layers=0)
                self.model.load_state_dict(torch.load(torch_checkpoint, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.backend = "PyTorch Transformer (BERT)"
                logger.info("Successfully loaded PyTorch BERT model.")
                return
            except Exception as e:
                logger.warning(f"Failed to load PyTorch checkpoint: {e}. Falling back to baseline.")

        # 2. Try Scikit-Learn baseline model
        if os.path.exists(baseline_path):
            try:
                logger.info(f"Loading Scikit-Learn baseline from {baseline_path}")
                self.model = joblib.load(baseline_path)
                self.backend = "Scikit-Learn TF-IDF Baseline"
                self.model_name = "TF-IDF + Calibrated Logistic Regression"
                logger.info("Successfully loaded baseline model.")
                return
            except Exception as e:
                logger.warning(f"Failed to load baseline: {e}. Training fresh baseline...")

        # 3. Train high-performance baseline on local dataset
        self._train_and_save_fallback_baseline(baseline_path)

    def _train_and_save_fallback_baseline(self, save_path: str):
        """
        Trains a fast TF-IDF + Logistic Regression classifier on local data
        to ensure zero-downtime out-of-the-box operation.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        logger.info("Training lightweight production baseline model on local dataset...")
        data_candidates = [
            os.path.join("data", "shorttextpreprocessedtrain.csv"),
            os.path.join("data", "newdatasetwithcoviddata.csv")
        ]
        
        train_file = next((f for f in data_candidates if os.path.exists(f)), None)
        if not train_file:
            raise FileNotFoundError("Could not find any training data in ./data/")

        df = pd.read_csv(train_file)
        df = df.dropna(subset=["text", "label"])
        if len(df) > 30000:
            df = df.sample(30000, random_state=42)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=15000,
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

        pipeline.fit(df["text"].astype(str), df["label"].astype(int))

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(pipeline, save_path)

        self.model = pipeline
        self.backend = "Scikit-Learn TF-IDF Baseline"
        self.model_name = "TF-IDF + Logistic Regression (Calibrated)"
        logger.info(f"Baseline model trained and saved to {save_path}.")

    def _predict_probabilities_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Internal batch prediction returning list of class probabilities.
        Class 0: FAKE, Class 1: REAL
        """
        cleaned = [clean_text(t) for t in texts]
        
        if self.backend.startswith("PyTorch") and TORCH_AVAILABLE and self.model is not None:
            tokens = self.tokenizer.batch_encode_plus(
                cleaned,
                max_length=self.max_seq_len,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            return [{"FAKE": float(p[0]), "REAL": float(p[1])} for p in probs]
        else:
            # Scikit-learn pipeline
            probs = self.model.predict_proba(cleaned)
            classes = list(self.model.classes_)
            fake_idx = classes.index(0) if 0 in classes else 0
            real_idx = classes.index(1) if 1 in classes else 1

            return [{"FAKE": float(p[fake_idx]), "REAL": float(p[real_idx])} for p in probs]

    def predict(self, text: str, include_explanation: bool = False) -> Dict[str, Any]:
        """
        Predicts single news text.
        """
        start_t = time.perf_counter()
        cleaned = clean_text(text)
        probs = self._predict_probabilities_batch([cleaned])[0]

        fake_prob = probs["FAKE"]
        real_prob = probs["REAL"]

        if fake_prob >= real_prob:
            predicted_label = "FAKE"
            label_id = 0
            confidence = fake_prob
        else:
            predicted_label = "REAL"
            label_id = 1
            confidence = real_prob

        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

        explanation = None
        if include_explanation and self.explain_engine:
            explanation = self.explain_engine.explain(cleaned)

        return {
            "text": text,
            "cleaned_text": cleaned,
            "prediction": predicted_label,
            "label_id": label_id,
            "confidence": round(confidence, 4),
            "probabilities": {
                "FAKE": round(fake_prob, 4),
                "REAL": round(real_prob, 4)
            },
            "latency_ms": latency_ms,
            "model_backend": self.backend,
            "explanation": explanation
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predicts batch of news texts.
        """
        start_t = time.perf_counter()
        cleaned_list = [clean_text(t) for t in texts]
        probs_list = self._predict_probabilities_batch(cleaned_list)
        total_latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        avg_latency = round(total_latency_ms / max(len(texts), 1), 2)

        results = []
        for orig, cln, probs in zip(texts, cleaned_list, probs_list):
            fake_prob = probs["FAKE"]
            real_prob = probs["REAL"]
            if fake_prob >= real_prob:
                label = "FAKE"
                label_id = 0
                conf = fake_prob
            else:
                label = "REAL"
                label_id = 1
                conf = real_prob

            results.append({
                "text": orig,
                "cleaned_text": cln,
                "prediction": label,
                "label_id": label_id,
                "confidence": round(conf, 4),
                "probabilities": {
                    "FAKE": round(fake_prob, 4),
                    "REAL": round(real_prob, 4)
                },
                "latency_ms": avg_latency,
                "model_backend": self.backend,
                "explanation": None
            })

        return results

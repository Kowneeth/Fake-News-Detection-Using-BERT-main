"""
Unit tests for data preprocessing, prediction pipeline, and explainability.
"""
import pytest
from src.predict import clean_text, FakeNewsPredictor
from src.explain import ExplainabilityEngine


def test_clean_text_removes_urls_and_mentions():
    raw = "Check this out https://t.co/abc1234 from @elonmusk! Real news????"
    cleaned = clean_text(raw)
    assert "https://" not in cleaned
    assert "@elonmusk" not in cleaned
    assert "Real news??" in cleaned


def test_clean_text_handles_empty_or_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_predictor_initialization_and_single_predict():
    predictor = FakeNewsPredictor()
    assert predictor.model is not None

    result = predictor.predict("Breaking: sensational fake statement discovered online!", include_explanation=True)
    assert "prediction" in result
    assert result["prediction"] in ["FAKE", "REAL"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert "explanation" in result
    assert result["explanation"] is not None


def test_predictor_batch_predict():
    predictor = FakeNewsPredictor()
    texts = [
        "Sensational vaccine conspiracy revealed!",
        "Official report from the Department of Labor shows job growth."
    ]
    results = predictor.predict_batch(texts)
    assert len(results) == 2
    assert results[0]["prediction"] in ["FAKE", "REAL"]
    assert results[1]["prediction"] in ["FAKE", "REAL"]


def test_explainability_tokens():
    predictor = FakeNewsPredictor()
    exp = predictor.explain_engine.explain("Sensational conspiracy revealed by doctor")
    assert "tokens" in exp
    assert "highlighted_html" in exp
    assert len(exp["tokens"]) > 0

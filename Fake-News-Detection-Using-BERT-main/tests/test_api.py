"""
Integration and endpoint unit tests for FastAPI application.
"""
import pytest
from src.schemas import PredictRequest, BatchPredictRequest
from src.api import health_check, predict_single, predict_batch, explain_text, get_interactive_dashboard


def test_health_endpoint():
    res = health_check()
    assert res.status == "HEALTHY"
    assert res.model_loaded is True
    assert res.version == "2.0.0"
    assert res.model_backend != ""


def test_dashboard_endpoint():
    res = get_interactive_dashboard()
    assert res.status_code == 200
    assert b"Fake News Intelligence 2.0" in res.body


def test_predict_single_endpoint():
    req = PredictRequest(
        text="BREAKING: Secret vaccine microchip revealed by anonymous government doctor!",
        include_explanation=True
    )
    res = predict_single(req)
    assert res["prediction"] in ["FAKE", "REAL"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res
    assert res["explanation"] is not None
    assert len(res["explanation"]["tokens"]) > 0


def test_predict_batch_endpoint():
    req = BatchPredictRequest(
        texts=[
            "Shocking conspiracy revealed by whistleblower without any proof.",
            "San Francisco AP — Apple penalized CEO Tim Cook over earnings targets."
        ]
    )
    res = predict_batch(req)
    assert res.total_processed == 2
    assert len(res.predictions) == 2
    assert res.predictions[0].prediction in ["FAKE", "REAL"]
    assert res.predictions[1].prediction in ["FAKE", "REAL"]


def test_explain_endpoint():
    req = PredictRequest(text="Unverified rumor claiming alien invasion next week")
    res = explain_text(req)
    assert "tokens" in res
    assert "highlighted_html" in res

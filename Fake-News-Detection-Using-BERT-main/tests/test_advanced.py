"""
Unit and integration tests for AI text detection, domain credibility, and virality prediction.
"""
import pytest
from src.ai_detector import AITextDetector
from src.domain_credibility import DomainCredibilityEngine, normalize_domain
from src.virality import ViralityHarmPredictor
from src.schemas import PredictRequest, UrlPredictRequest
from src.api import detect_ai_text, check_domain_credibility, evaluate_virality


def test_ai_detector_identifies_synthetic_patterns():
    detector = AITextDetector()
    synthetic_sample = (
        "Furthermore, navigating the complexities of modern disinformation plays a crucial role in society. "
        "Moreover, it is important to remember that technology continues to evolve and delve into new horizons."
    )
    res = detector.detect(synthetic_sample)
    assert res["ai_generated_probability"] > 0.65
    assert res["verdict"] == "LIKELY AI-GENERATED (SYNTHETIC)"
    assert len(res["detected_llm_markers"]) > 0


def test_ai_detector_identifies_human_writing():
    detector = AITextDetector()
    human_sample = (
        "Hey! Caught the football match yesterday at the local pub. "
        "What a chaotic second half, the referee totally missed two blatant offside calls! Heading out now."
    )
    res = detector.detect(human_sample)
    assert res["ai_generated_probability"] < 0.40
    assert res["verdict"] == "LIKELY HUMAN-WRITTEN"
    assert res["burstiness_score"] > 0.40


def test_domain_credibility_wire_services():
    engine = DomainCredibilityEngine()
    res = engine.check_domain("https://apnews.com/article/12345")
    assert res["domain_recognized"] is True
    assert res["trust_score"] >= 95
    assert res["factual_reporting"] == "VERY HIGH"
    assert res["is_satire"] is False


def test_domain_credibility_satire():
    engine = DomainCredibilityEngine()
    res = engine.check_domain("https://www.theonion.com/some-funny-satire-headline")
    assert res["domain_recognized"] is True
    assert res["is_satire"] is True
    assert res["trust_score"] <= 15


def test_domain_credibility_conspiracy_outlet():
    engine = DomainCredibilityEngine()
    res = engine.check_domain("http://infowars.com/news")
    assert res["domain_recognized"] is True
    assert res["trust_score"] < 15
    assert res["bias"] == "CONSPIRACY-PSEUDOSCIENCE"


def test_domain_credibility_unknown():
    engine = DomainCredibilityEngine()
    res = engine.check_domain("https://completely-random-unregistered-blog-site.xyz/post")
    assert res["domain_recognized"] is False
    assert res["trust_score"] == 50


def test_virality_predictor_hazard():
    predictor = ViralityHarmPredictor()
    text = "SHARE BEFORE DELETED! Toxic poison found in hospital water supply across the country! Retweet now!"
    res = predictor.predict_harm(text, fake_probability=0.85)
    assert res["virality_potential_score"] > 60
    assert "CRITICAL" in res["harm_threat_level"] or "ELEVATED" in res["harm_threat_level"]
    assert len(res["amplification_triggers"]) > 0


def test_api_ai_detect_endpoint():
    req = PredictRequest(text="Moreover, it is important to remember that this process plays a crucial role.")
    res = detect_ai_text(req)
    assert "ai_generated_probability" in res
    assert res["ai_generated_probability"] > 0.5


def test_api_domain_credibility_endpoint():
    req = UrlPredictRequest(url="https://reuters.com/world/news")
    res = check_domain_credibility(req)
    assert res["trust_score"] >= 90
    assert res["factual_reporting"] == "VERY HIGH"


def test_api_virality_endpoint():
    req = PredictRequest(text="Spread the word! Everyone needs to see this shocking announcement!")
    res = evaluate_virality(req)
    assert res["virality_potential_score"] > 30
    assert len(res["amplification_triggers"]) > 0

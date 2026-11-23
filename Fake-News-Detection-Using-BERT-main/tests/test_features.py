"""
Unit and integration tests for scraper, sensationalism, fact-checking, and report generation.
"""
import pytest
from src.sensationalism import SensationalismAnalyzer
from src.factcheck import FactCheckEngine
from src.report import generate_intelligence_report
from src.schemas import PredictRequest, UrlPredictRequest
from src.api import analyze_sensationalism, check_fact, generate_report


def test_sensationalism_analyzer_detects_clickbait_and_alarmism():
    analyzer = SensationalismAnalyzer()
    text = "BREAKING: The shocking truth revealed! Insiders claim deadly virus is a catastrophe!!!"
    res = analyzer.analyze(text)

    assert res["overall_sensationalism_score"] > 40
    assert any("shocking truth" in p.lower() for p in res["detected_clickbait_phrases"])
    assert "deadly" in res["detected_alarmist_words"] or "catastrophe" in res["detected_alarmist_words"]
    assert len(res["anonymous_sourcing_triggers"]) > 0


def test_sensationalism_analyzer_neutral_text():
    analyzer = SensationalismAnalyzer()
    text = "The committee held a regular session to review the annual financial budget."
    res = analyzer.analyze(text)

    assert res["overall_sensationalism_score"] < 25
    assert res["risk_level"] == "LOW / OBJECTIVE JOURNALISTIC TONE"


def test_fact_check_engine_finds_recorded_claims():
    fc = FactCheckEngine()
    query = "5G network coronavirus microchip infection"
    res = fc.check_claim(query)

    assert res["fact_checks_found"] is True
    assert len(res["results"]) > 0
    assert "False" in res["results"][0]["rating"]


def test_fact_check_engine_unknown_claim():
    fc = FactCheckEngine()
    query = "Completely random unique query with zero records in any database 987654"
    res = fc.check_claim(query)
    assert res["fact_checks_found"] is False


def test_report_generation_markdown():
    prediction_result = {
        "text": "BREAKING: Secret vaccine microchip revealed by anonymous government doctor!",
        "prediction": "FAKE",
        "confidence": 0.94,
        "probabilities": {"FAKE": 0.94, "REAL": 0.06},
        "model_backend": "Scikit-Learn TF-IDF Baseline"
    }
    sens_result = {
        "overall_sensationalism_score": 75,
        "risk_level": "CRITICAL SENSATIONALISM",
        "clickbait_score": 70,
        "emotional_urgency_score": 80,
        "detected_alarmist_words": ["microchip"]
    }
    fact_result = {
        "fact_checks_found": True,
        "source": "FactCheck.org",
        "results": [{
            "claim": "Vaccines contain tracking microchips",
            "rating": "False",
            "reviewer": "PolitiFact",
            "review_url": "https://example.com"
        }]
    }

    report = generate_intelligence_report(prediction_result, sens_result, fact_result)
    assert "# 🛡️ Disinformation Intelligence Assessment Report" in report
    assert "FINAL CLASSIFICATION" in report.upper()
    assert "FAKE" in report
    assert "CRITICAL SENSATIONALISM" in report


def test_api_sensationalism_endpoint():
    req = PredictRequest(text="YOU WON'T BELIEVE what happened next! Deadly catastrophe exposed!!!")
    res = analyze_sensationalism(req)
    assert res["clickbait_score"] > 0
    assert res["overall_sensationalism_score"] > 30


def test_api_fact_check_endpoint():
    req = PredictRequest(text="Bill Gates vaccine microchip tracking")
    res = check_fact(req)
    assert res["fact_checks_found"] is True
    assert len(res["results"]) > 0


def test_api_report_endpoint():
    req = PredictRequest(text="Secret conspiracy theory uncovered by whistleblower")
    res = generate_report(req)
    assert "report_markdown" in res.model_dump()
    assert res.verdict in ["FAKE", "REAL"]
    assert res.report_id.startswith("DISINFO-")

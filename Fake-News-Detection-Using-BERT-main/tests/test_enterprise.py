"""
Enterprise tests for database history, debunk generator, multimodal forensics, and multilingual detection.
"""
import io
import pytest
from PIL import Image
from src.database import AuditDatabase
from src.debunk import DebunkGenerator
from src.multimodal import MultiModalVerifier
from src.multilingual import MultiLingualDetector
from src.watchtower import DisinformationWatchtower
from src.schemas import PredictRequest
from src.api import generate_debunk, get_audit_history, get_watchtower_feed


def test_database_logging_and_retrieval(tmp_path):
    db_file = tmp_path / "test_history.db"
    db = AuditDatabase(db_path=str(db_file))

    db.log_verification(
        audit_id="TEST-001",
        text="Sample fake claim for audit testing",
        prediction="FAKE",
        confidence=0.92,
        fake_prob=0.92,
        real_prob=0.08,
        model_backend="Baseline",
        sensationalism_score=65
    )

    records = db.get_history(query="Sample fake")
    assert len(records) == 1
    assert records[0]["audit_id"] == "TEST-001"
    assert records[0]["prediction"] == "FAKE"
    assert db.get_count() == 1


def test_debunk_generator_truth_sandwich():
    debunk = DebunkGenerator()
    res = debunk.generate_rebuttal(
        claim_text="5G cellular networks spread coronavirus",
        prediction="FAKE",
        confidence=0.95,
        sensationalism_data={"overall_sensationalism_score": 80},
        fact_check_data={"fact_checks_found": True, "results": [{"reviewer": "Reuters", "rating": "False"}]}
    )

    assert "FACT:" in res["truth_sandwich_rebuttal"]
    assert "CLAIM:" in res["truth_sandwich_rebuttal"]
    assert len(res["social_media_rebuttal"]) <= 280
    assert len(res["talking_points"]) > 2


def test_multimodal_image_analysis():
    verifier = MultiModalVerifier()
    # Create simple in-memory test image
    img = Image.new("RGB", (200, 100), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    res = verifier.analyze_image(img_bytes, filename="test.jpg")
    assert res["format"] == "JPEG"
    assert res["dimensions"] == "200x100"
    assert "tampering_risk" in res


def test_multilingual_language_detection():
    detector = MultiLingualDetector()

    # Spanish test
    es_res = detector.detect_language("El presidente anunció nuevas medidas económicas para el país hoy.")
    assert es_res["iso_code"] == "es"
    assert es_res["is_english"] is False

    # English test
    en_res = detector.detect_language("Federal Reserve kept interest rates unchanged at the meeting.")
    assert en_res["iso_code"] == "en"
    assert en_res["is_english"] is True


def test_watchtower_stream():
    wt = DisinformationWatchtower()
    items = wt.get_live_stream()
    assert len(items) >= 3
    assert "threat_level" in items[0]


def test_api_debunk_endpoint():
    req = PredictRequest(text="Secret government microchip in food supply")
    res = generate_debunk(req)
    assert res["prediction"] in ["FAKE", "REAL"]
    assert len(res["truth_sandwich_rebuttal"]) > 20


def test_api_history_endpoint():
    res = get_audit_history(limit=5)
    assert hasattr(res, "total_records")
    assert hasattr(res, "records")


def test_api_watchtower_endpoint():
    items = get_watchtower_feed()
    assert len(items) > 0
    assert items[0]["id"].startswith("WATCH-")

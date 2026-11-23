"""
Pydantic Schemas for Fake News Detection API (Pydantic v2 compliant).
Includes Multi-Modal, Rebuttal, Database History, Multi-Lingual & Watchtower models.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "BREAKING: Secret vaccine microchip revealed by anonymous government doctor!",
                "include_explanation": True
            }
        }
    )
    text: str = Field(..., min_length=3, description="Tweet, headline, or news article text to analyze")
    include_explanation: bool = Field(default=False, description="Whether to include token-level explainability")


class UrlPredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://apnews.com/article/apple-tim-cook-pay-cut",
                "include_explanation": True
            }
        }
    )
    url: str = Field(..., min_length=5, description="Full HTTP/HTTPS URL of the news article")
    include_explanation: bool = Field(default=True, description="Whether to include token-level explainability")


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "BREAKING: Secret vaccine microchip revealed by anonymous government doctor!",
                    "Federal Reserve leaves benchmark interest rate unchanged following FOMC meeting."
                ]
            }
        }
    )
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of news texts to analyze")


class TokenAttribution(BaseModel):
    token: str
    weight: float = Field(..., description="Importance weight (-1.0 to +1.0)")
    influence: str = Field(..., description="'FAKE' or 'REAL'")


class ExplanationResult(BaseModel):
    tokens: List[TokenAttribution]
    highlighted_html: str
    top_fake_signals: List[str]
    top_real_signals: List[str]


class SensationalismResult(BaseModel):
    overall_sensationalism_score: int
    risk_level: str
    clickbait_score: int
    emotional_urgency_score: int
    detected_clickbait_phrases: List[str]
    detected_alarmist_words: List[str]
    anonymous_sourcing_triggers: List[str]
    all_caps_word_count: int
    all_caps_ratio_pct: float
    excessive_punctuation_count: int


class FactCheckItem(BaseModel):
    claim: str
    claimant: str
    reviewer: str
    rating: str
    review_url: str
    summary: str


class FactCheckResult(BaseModel):
    fact_checks_found: bool
    results: List[FactCheckItem]
    source: str


class AIDetectionResult(BaseModel):
    ai_generated_probability: float
    verdict: str
    burstiness_score: float
    lexical_entropy: float
    type_token_ratio: float
    total_words: int
    detected_llm_markers: List[str]


class DomainCredibilityResult(BaseModel):
    domain: str
    domain_recognized: bool
    trust_score: int
    factual_reporting: str
    bias: str
    category: str
    summary: str
    is_satire: bool


class ViralityResult(BaseModel):
    virality_potential_score: int
    harm_threat_level: str
    amplification_triggers: List[str]
    hazard_domains_detected: List[str]
    is_high_velocity_candidate: bool


class MultiLingualResult(BaseModel):
    language: str
    iso_code: str
    confidence: float
    is_english: bool
    cross_lingual_advisory: str


class ImagePredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                "filename": "screenshot.jpg"
            }
        }
    )
    image_base64: str = Field(..., min_length=10, description="Base64 encoded image string")
    filename: Optional[str] = Field(default="upload.jpg", description="Original image filename")


class ImageForensicsResult(BaseModel):
    filename: str
    format: str
    dimensions: str
    editing_software_detected: bool
    software_name: str
    metadata_stripped: bool
    date_original: str
    date_modified: str
    tampering_risk: str
    extracted_text: str
    has_embedded_text: bool


class PredictionResponse(BaseModel):
    text: str
    cleaned_text: str
    prediction: str = Field(..., description="'FAKE' or 'REAL'")
    label_id: int = Field(..., description="0 for FAKE, 1 for REAL")
    confidence: float = Field(..., description="Probability of predicted class (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(..., description="Probability breakdown: {'FAKE': ..., 'REAL': ...}")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    model_backend: str = Field(..., description="Model backend used for inference")
    explanation: Optional[ExplanationResult] = None
    sensationalism: Optional[SensationalismResult] = None
    fact_check: Optional[FactCheckResult] = None
    ai_detection: Optional[AIDetectionResult] = None
    domain_credibility: Optional[DomainCredibilityResult] = None
    virality: Optional[ViralityResult] = None
    multilingual: Optional[MultiLingualResult] = None
    image_forensics: Optional[ImageForensicsResult] = None
    source_url: Optional[str] = None
    article_title: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    total_processed: int
    predictions: List[PredictionResponse]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backend: str
    model_name: str
    device: str
    version: str


class ReportResponse(BaseModel):
    report_markdown: str
    report_id: str
    verdict: str
    confidence: float


class DebunkResponse(BaseModel):
    claim: str
    prediction: str
    truth_sandwich_rebuttal: str
    social_media_rebuttal: str
    talking_points: List[str]
    fact_check_reviewer: str


class HistoryRecord(BaseModel):
    audit_id: str
    timestamp: str
    text: str
    source_url: str
    prediction: str
    confidence: float
    fake_probability: float
    real_probability: float
    sensationalism_score: int
    ai_generated_probability: float
    fact_check_summary: str


class HistoryResponse(BaseModel):
    total_records: int
    records: List[HistoryRecord]


class WatchtowerItem(BaseModel):
    id: str
    headline: str
    category: str
    threat_level: str
    velocity: str
    verdict: str
    debunk_summary: str
    verified_time: str

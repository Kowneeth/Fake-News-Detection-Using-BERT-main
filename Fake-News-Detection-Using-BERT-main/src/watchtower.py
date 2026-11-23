"""
Live Disinformation Watchtower & Emerging Hoax Stream.
Simulates real-time OSINT monitoring of trending false narratives, viral velocity, and severity.
"""
import time
from typing import List, Dict, Any

TRENDING_DISINFO_STREAM: List[Dict[str, Any]] = [
    {
        "id": "WATCH-101",
        "headline": "Fake emergency broadcast claiming national grid failure spreads on TikTok & Telegram",
        "category": "Critical Infrastructure",
        "threat_level": "CRITICAL HAZARD",
        "velocity": "Viral (88/100)",
        "verdict": "DEBUNKED FALSE",
        "debunk_summary": "FEMA and national power operators confirmed the video was created with an AI voice generator.",
        "verified_time": "12 minutes ago"
    },
    {
        "id": "WATCH-102",
        "headline": "Altered screenshot claims FDA secretly revoked approval for standard vaccines",
        "category": "Public Health",
        "threat_level": "ELEVATED",
        "velocity": "Accelerating (72/100)",
        "verdict": "FABRICATED SCREENSHOT",
        "debunk_summary": "FDA press office confirmed the document is forged; standard approvals remain active.",
        "verified_time": "45 minutes ago"
    },
    {
        "id": "WATCH-103",
        "headline": "Fabricated celebrity endorsement claiming breakthrough cryptocurrency investment scheme",
        "category": "Financial Scams",
        "threat_level": "MODERATE",
        "velocity": "Moderate (55/100)",
        "verdict": "DEEPFAKE / PHISHING",
        "debunk_summary": "Synthesized lip-sync video manipulated authentic interview footage.",
        "verified_time": "2 hours ago"
    },
    {
        "id": "WATCH-104",
        "headline": "NASA warns asteroid collision will strike tomorrow in leaked classified memo",
        "category": "Clickbait / Panic",
        "threat_level": "LOW HAZARD",
        "velocity": "Slowed (35/100)",
        "verdict": "RECYCLED HOAX",
        "debunk_summary": "Standard recycled internet chain hoax originating from satirical forums.",
        "verified_time": "4 hours ago"
    }
]


class DisinformationWatchtower:
    """
    Manages emerging narrative streams and viral threat feeds.
    """
    def get_live_stream(self) -> List[Dict[str, Any]]:
        return TRENDING_DISINFO_STREAM

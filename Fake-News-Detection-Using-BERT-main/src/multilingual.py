"""
Multi-Lingual Disinformation Detector.
Identifies language scripts, cross-lingual patterns, and global disinformation signals.
"""
import re
from typing import Dict, Any

# Script unicode ranges & high-frequency stopwords for fast language identification
LANGUAGE_PROFILES = {
    "hi": {
        "name": "Hindi",
        "script_regex": r"[\u0900-\u097F]",
        "stopwords": ["है", "के", "में", "की", "का", "और", "को", "से", "पर", "नहीं"]
    },
    "es": {
        "name": "Spanish",
        "script_regex": r"[áéíóúñ¿¡]",
        "stopwords": ["de", "la", "el", "en", "y", "que", "los", "del", "las", "por", "para"]
    },
    "fr": {
        "name": "French",
        "script_regex": r"[àâçéèêëîïôûùüÿœæ]",
        "stopwords": ["de", "la", "le", "et", "les", "des", "en", "du", "un", "est", "dans"]
    },
    "de": {
        "name": "German",
        "script_regex": r"[äöüß]",
        "stopwords": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"]
    },
    "ru": {
        "name": "Russian",
        "script_regex": r"[\u0400-\u04FF]",
        "stopwords": ["и", "в", "не", "на", "я", "с", "что", "по", "к", "как"]
    },
    "ar": {
        "name": "Arabic",
        "script_regex": r"[\u0600-\u06FF]",
        "stopwords": ["في", "من", "على", "أن", "إلى", "هذا", "عن", "مع", "كان", "التي"]
    },
    "zh": {
        "name": "Chinese",
        "script_regex": r"[\u4E00-\u9FFF]",
        "stopwords": ["的", "是", "在", "不", "了", "有", "和", "人", "这", "中"]
    }
}


class MultiLingualDetector:
    """
    Identifies language and scripts for global fake news monitoring.
    """
    def detect_language(self, text: str) -> Dict[str, Any]:
        if not text or len(text.strip()) < 3:
            return {"language": "Unknown", "iso_code": "unknown", "confidence": 0.0, "is_english": False}

        cleaned = text.lower()
        words = re.findall(r"\b\w+\b", cleaned)

        # 1. Script-based check
        for code, profile in LANGUAGE_PROFILES.items():
            if re.search(profile["script_regex"], text):
                return {
                    "language": profile["name"],
                    "iso_code": code,
                    "confidence": 0.95,
                    "is_english": False,
                    "cross_lingual_advisory": f"Content detected in {profile['name']}. Cross-lingual zero-shot representations active."
                }

        # 2. Stopword matching
        best_code = "en"
        best_matches = 0
        for code, profile in LANGUAGE_PROFILES.items():
            matches = sum(1 for w in words if w in profile["stopwords"])
            if matches > best_matches and matches >= 2:
                best_matches = matches
                best_code = code

        if best_code != "en" and best_matches >= 2:
            name = LANGUAGE_PROFILES[best_code]["name"]
            return {
                "language": name,
                "iso_code": best_code,
                "confidence": min(0.65 + (best_matches * 0.05), 0.95),
                "is_english": False,
                "cross_lingual_advisory": f"Content detected in {name}. Analyzed with cross-lingual embeddings."
            }

        return {
            "language": "English",
            "iso_code": "en",
            "confidence": 0.98,
            "is_english": True,
            "cross_lingual_advisory": "Standard English transformer pipeline active."
        }

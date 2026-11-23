"""
Sensationalism, Clickbait & Emotional Urgency Analyzer for Fake News Detection.
Evaluates stylistic manipulation, sensational tropes, and alarmist rhetoric.
"""
import re
from typing import Dict, Any, List

# Patterns frequently used in clickbait headlines
CLICKBAIT_PATTERNS = [
    r"\byou won'?t believe\b",
    r"\bwhat happens next\b",
    r"\bwill blow your mind\b",
    r"\bthe shocking truth\b",
    r"\bsecret revealed\b",
    r"\bthey don'?t want you to know\b",
    r"\bthis one trick\b",
    r"\bcan'?t unsee\b",
    r"\bbreaks the internet\b",
    r"\bleft speechless\b",
    r"\bis this the end of\b",
    r"\bmiracle cure\b",
    r"\bunbelievable reason\b",
    r"\bgoes viral\b",
    r"\bthe real reason why\b",
    r"\bwarning to all\b"
]

# Words triggering high fear, alarmism, or emotional urgency
ALARMISM_LEXICON = {
    "deadly", "fatal", "disaster", "catastrophe", "terrifying", "horrifying",
    "apocalypse", "apocalyptic", "collapse", "lethal", "crisis", "threat",
    "emergency", "nightmare", "panicking", "panic", "ruined", "doomed",
    "conspiracy", "hoax", "exposed", "devastating", "dangerous", "poison",
    "toxic", "destruction", "warning", "danger", "murder", "treason", "coverup"
}

# Vague or anonymous sourcing tropes typical in fabricated news
ANONYMOUS_SOURCING_PATTERNS = [
    r"\binsiders? (?:claims?|reveals?|say)\b",
    r"\bsources? (?:claims?|reveals?|say|confirm)\b",
    r"\banonymous (?:doctor|official|scientist|source|insider)\b",
    r"\bexperts? (?:warn|reveal|say)\b",
    r"\bwhistleblowers? (?:claims?|reveals?)\b",
    r"\bthey (?:don'?t want|are hiding|admit)\b"
]


class SensationalismAnalyzer:
    """
    Analyzes emotional intensity, stylistic manipulation, and clickbait signals.
    """
    def __init__(self):
        self.clickbait_regexes = [re.compile(p, re.IGNORECASE) for p in CLICKBAIT_PATTERNS]
        self.sourcing_regexes = [re.compile(p, re.IGNORECASE) for p in ANONYMOUS_SOURCING_PATTERNS]

    def analyze(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return self._empty_result()

        words = re.findall(r"\b\w+\b", text)
        total_words = max(len(words), 1)

        # 1. Clickbait Score
        clickbait_hits = []
        for rx in self.clickbait_regexes:
            match = rx.search(text)
            if match:
                clickbait_hits.append(match.group(0))

        # Check for interrogative headlines
        has_question = bool(re.search(r"\?\s*$", text.split("\n")[0]))
        clickbait_raw = (len(clickbait_hits) * 35) + (20 if has_question else 0)
        clickbait_score = min(max(clickbait_raw, 0), 100)

        # 2. Emotional Urgency / Alarmism Score
        alarmism_words = [w.lower() for w in words if w.lower() in ALARMISM_LEXICON]
        alarmism_ratio = len(alarmism_words) / total_words
        # 3% or more alarmist words is considered very high
        urgency_score = min(int((alarmism_ratio / 0.035) * 100), 100)

        # 3. Stylistic Flags
        # All-Caps Words (excluding short abbreviations <= 3 chars like USA, AP, CEO)
        all_caps_words = [w for w in words if w.isupper() and len(w) > 3]
        all_caps_ratio = round((len(all_caps_words) / total_words) * 100, 1)

        # Excessive Punctuation (e.g. "!!!", "?!", "???")
        punct_matches = re.findall(r"[!?]{2,}", text)
        excessive_punct_count = len(punct_matches)

        # Anonymous Sourcing Detection
        sourcing_flags = []
        for rx in self.sourcing_regexes:
            match = rx.search(text)
            if match:
                sourcing_flags.append(match.group(0))

        # 4. Overall Sensationalism Index (Weighted Blend)
        stylistic_penalty = min((len(all_caps_words) * 15) + (excessive_punct_count * 15) + (len(sourcing_flags) * 20), 100)
        overall_index = int(
            (clickbait_score * 0.40) +
            (urgency_score * 0.40) +
            (stylistic_penalty * 0.20)
        )
        overall_index = min(max(overall_index, 0), 100)

        # Risk Classification
        if overall_index >= 75:
            risk_level = "CRITICAL SENSATIONALISM"
        elif overall_index >= 50:
            risk_level = "HIGH SENSATIONALISM"
        elif overall_index >= 25:
            risk_level = "MODERATE SENSATIONALISM"
        else:
            risk_level = "LOW / OBJECTIVE JOURNALISTIC TONE"

        return {
            "overall_sensationalism_score": overall_index,
            "risk_level": risk_level,
            "clickbait_score": clickbait_score,
            "emotional_urgency_score": urgency_score,
            "detected_clickbait_phrases": clickbait_hits,
            "detected_alarmist_words": list(set(alarmism_words))[:8],
            "anonymous_sourcing_triggers": sourcing_flags,
            "all_caps_word_count": len(all_caps_words),
            "all_caps_ratio_pct": all_caps_ratio,
            "excessive_punctuation_count": excessive_punct_count
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "overall_sensationalism_score": 0,
            "risk_level": "LOW / OBJECTIVE JOURNALISTIC TONE",
            "clickbait_score": 0,
            "emotional_urgency_score": 0,
            "detected_clickbait_phrases": [],
            "detected_alarmist_words": [],
            "anonymous_sourcing_triggers": [],
            "all_caps_word_count": 0,
            "all_caps_ratio_pct": 0.0,
            "excessive_punctuation_count": 0
        }

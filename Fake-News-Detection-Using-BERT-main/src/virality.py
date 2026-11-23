"""
Virality & Disinformation Harm Predictor.
Estimates algorithmic shareability, emotional amplification, and public safety harm.
"""
import re
from typing import Dict, Any, List

# Call-to-action phrases designed to trigger viral social media sharing
VIRAL_AMPLIFICATION_PATTERNS = [
    r"\bshare (?:this|before|now)\b",
    r"\bretweet\b",
    r"\bspread the word\b",
    r"\bbefore (?:it'?s|they) (?:deleted?|taken? down|banned?)\b",
    r"\bthe media won'?t (?:show|tell) you\b",
    r"\bwake up people\b",
    r"\beveryone needs to (?:know|see) this\b",
    r"\bmake this go viral\b"
]

# Sensitive domains with severe real-world hazard potential
HIGH_HARM_TOPICS = {
    "public_health": ["vaccine", "poison", "cure", "virus", "pandemic", "cancer", "toxic", "hospital", "deadly disease"],
    "elections_democracy": ["voting machine", "rigged election", "ballot fraud", "stolen election", "civil war"],
    "critical_infrastructure": ["grid collapse", "nuclear plant", "water supply", "bank run", "martial law"]
}


class ViralityHarmPredictor:
    """
    Computes social amplification risk and public safety threat index.
    """
    def __init__(self):
        self.amplification_regexes = [re.compile(p, re.IGNORECASE) for p in VIRAL_AMPLIFICATION_PATTERNS]

    def predict_harm(self, text: str, fake_probability: float = 0.5) -> Dict[str, Any]:
        if not text:
            return self._empty_result()

        cleaned = text.lower()
        words = re.findall(r"\b\w+\b", cleaned)
        total_words = max(len(words), 1)

        # 1. Amplification Triggers
        triggers = []
        for rx in self.amplification_regexes:
            match = rx.search(cleaned)
            if match:
                triggers.append(match.group(0))

        # 2. Topic Hazard Evaluation
        matched_hazards = []
        for category, terms in HIGH_HARM_TOPICS.items():
            for t in terms:
                if t in cleaned:
                    matched_hazards.append(category.replace("_", " ").title())
                    break

        # 3. Shareability / Virality Score (0-100)
        # Optimal virality length is 15-45 words (punchy social media length)
        length_multiplier = 1.0
        if 10 <= total_words <= 50:
            length_multiplier = 1.25
        elif total_words > 150:
            length_multiplier = 0.85

        base_virality = 30.0
        base_virality += len(triggers) * 20.0
        base_virality += len(matched_hazards) * 15.0
        if re.search(r"[!?]{2,}", text):
            base_virality += 10.0

        virality_score = min(int(base_virality * length_multiplier), 100)

        # 4. Harm Threat Level
        # Combine fake probability with topic hazards
        if fake_probability >= 0.70 and len(matched_hazards) >= 2:
            harm_level = "CRITICAL PUBLIC HEALTH / DEMOCRACY HAZARD"
        elif fake_probability >= 0.60 and len(matched_hazards) >= 1:
            harm_level = "ELEVATED PUBLIC HARM RISK"
        elif virality_score > 65 and fake_probability > 0.5:
            harm_level = "MODERATE VIRAL SPREAD RISK"
        else:
            harm_level = "LOW HARM POTENTIAL"

        return {
            "virality_potential_score": virality_score,
            "harm_threat_level": harm_level,
            "amplification_triggers": triggers,
            "hazard_domains_detected": list(set(matched_hazards)),
            "is_high_velocity_candidate": (virality_score >= 70)
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "virality_potential_score": 0,
            "harm_threat_level": "LOW HARM POTENTIAL",
            "amplification_triggers": [],
            "hazard_domains_detected": [],
            "is_high_velocity_candidate": False
        }

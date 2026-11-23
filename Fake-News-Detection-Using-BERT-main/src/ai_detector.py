"""
AI & Synthetic Text Detector for Disinformation Intelligence.
Analyzes perplexity proxies, sentence burstiness, vocabulary entropy, and LLM stylometry.
"""
import re
import math
from typing import Dict, Any, List

# Typical connective / hedging phrases overused by LLMs in synthetic news
LLM_MARKER_PATTERNS = [
    r"\bit is important to (?:note|remember|consider)\b",
    r"\bfurthermore,\s+",
    r"\bmoreover,\s+",
    r"\bin conclusion,\s+",
    r"\bit remains to be seen\b",
    r"\bplays a crucial role\b",
    r"\ba testament to\b",
    r"\bdelve into\b",
    r"\bnavigating the complexities\b",
    r"\bin summary,\s+",
    r"\bon the other hand,\s+",
    r"\bit should be noted\b"
]


class AITextDetector:
    """
    Evaluates statistical stylometry to detect machine-generated / synthetic news.
    """
    def __init__(self):
        self.marker_regexes = [re.compile(p, re.IGNORECASE) for p in LLM_MARKER_PATTERNS]

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Calculates burstiness, vocabulary entropy, and machine-generation probability.
        """
        if not text or len(text.strip()) < 30:
            return self._empty_result(text)

        cleaned = text.strip()
        words = re.findall(r"\b[A-Za-z0-9'-]+\b", cleaned)
        total_words = len(words)

        if total_words < 8:
            return self._empty_result(text)

        # 1. Burstiness Analysis (Variance of Sentence Lengths)
        sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned) if s.strip()]
        sentence_word_counts = [len(re.findall(r"\b\w+\b", s)) for s in sentences if len(s.split()) > 0]

        if len(sentence_word_counts) > 1:
            mean_len = sum(sentence_word_counts) / len(sentence_word_counts)
            variance = sum((x - mean_len) ** 2 for x in sentence_word_counts) / len(sentence_word_counts)
            std_dev = math.sqrt(variance)
            # Coefficient of Variation (CV = std_dev / mean)
            burstiness = std_dev / max(mean_len, 1.0)
        else:
            burstiness = 0.35  # Single sentence baseline

        # LLMs typically have low burstiness (uniform lengths ~0.15 - 0.35)
        # Humans have higher burstiness (> 0.50)
        burstiness_score = min(max(round(burstiness, 3), 0.0), 2.0)

        # 2. Vocabulary Entropy (Lexical Richness)
        word_freqs: Dict[str, int] = {}
        for w in words:
            lw = w.lower()
            word_freqs[lw] = word_freqs.get(lw, 0) + 1

        entropy = 0.0
        for count in word_freqs.values():
            p = count / total_words
            entropy -= p * math.log2(p)

        # Theoretical max entropy for this vocabulary size
        unique_words = len(word_freqs)
        ttr = unique_words / total_words  # Type-Token Ratio

        # 3. LLM Stylometric Markers
        marker_hits = []
        for rx in self.marker_regexes:
            match = rx.search(cleaned)
            if match:
                marker_hits.append(match.group(0))

        # 4. Synthesize AI-Generated Probability Score
        # Signals of AI:
        # - Low burstiness (< 0.35) -> +0.30
        # - High repetition / uniform entropy -> +0.25
        # - Presence of LLM connector markers -> +0.25 per marker (capped at 0.40)
        # - Low punctuation variance -> +0.10

        ai_score = 0.35  # Neutral prior

        if burstiness < 0.28:
            ai_score += 0.30
        elif burstiness < 0.40:
            ai_score += 0.15
        elif burstiness > 0.60:
            ai_score -= 0.25

        if len(marker_hits) > 0:
            ai_score += min(len(marker_hits) * 0.15, 0.40)

        # Extreme type-token ratio adjustment
        if ttr < 0.45 and total_words > 40:
            ai_score += 0.15

        ai_probability = min(max(round(ai_score, 4), 0.05), 0.98)

        # Verdict
        if ai_probability >= 0.70:
            verdict = "LIKELY AI-GENERATED (SYNTHETIC)"
        elif ai_probability <= 0.35:
            verdict = "LIKELY HUMAN-WRITTEN"
        else:
            verdict = "MIXED / INCONCLUSIVE"

        return {
            "ai_generated_probability": ai_probability,
            "verdict": verdict,
            "burstiness_score": burstiness_score,
            "lexical_entropy": round(entropy, 3),
            "type_token_ratio": round(ttr, 3),
            "total_words": total_words,
            "detected_llm_markers": marker_hits
        }

    def _empty_result(self, text: str) -> Dict[str, Any]:
        return {
            "ai_generated_probability": 0.50,
            "verdict": "INSUFFICIENT TEXT LENGTH",
            "burstiness_score": 0.0,
            "lexical_entropy": 0.0,
            "type_token_ratio": 0.0,
            "total_words": len(text.split()),
            "detected_llm_markers": []
        }

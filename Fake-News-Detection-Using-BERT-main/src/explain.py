"""
Explainability Module for Fake News Detection.
Provides token-level attribution, feature importance, and HTML highlighting.
"""
import re
import html
from typing import List, Dict, Any, Tuple, Callable

# Common English stop words to filter noise
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}


class ExplainabilityEngine:
    """
    Computes token-level importance scores and renders visual HTML highlights.
    Uses leave-one-out perturbation attribution for model-agnostic fidelity.
    """
    def __init__(self, predict_fn: Callable[[List[str]], List[Dict[str, float]]]):
        """
        Args:
            predict_fn: A function that takes a list of strings and returns a list of
                        dictionaries with class probabilities: [{'FAKE': p0, 'REAL': p1}, ...]
        """
        self.predict_fn = predict_fn

    def explain(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Computes token-level attribution and creates highlighted visualization.
        """
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
        if not tokens:
            return {
                "tokens": [],
                "highlighted_html": html.escape(text),
                "top_fake_signals": [],
                "top_real_signals": []
            }

        # 1. Base prediction
        base_probs = self.predict_fn([text])[0]
        base_fake_p = base_probs.get("FAKE", 0.5)
        base_real_p = base_probs.get("REAL", 0.5)
        predicted_label = "FAKE" if base_fake_p >= base_real_p else "REAL"

        # 2. Extract words to perturb (avoiding single punctuations)
        words = [t for t in tokens if re.match(r"^\w+$", t)]
        if not words:
            return {
                "tokens": [],
                "highlighted_html": html.escape(text),
                "top_fake_signals": [],
                "top_real_signals": []
            }

        # Create leave-one-out variations
        perturbed_texts = []
        perturbed_indices = []
        for i, word in enumerate(words):
            if word.lower() in STOP_WORDS and len(words) > 8:
                continue
            # Remove word i
            text_without_i = " ".join([w for j, w in enumerate(words) if j != i])
            perturbed_texts.append(text_without_i)
            perturbed_indices.append(word)

        token_weights: Dict[str, float] = {}
        if perturbed_texts:
            perturbed_probs = self.predict_fn(perturbed_texts)
            for word, p_probs in zip(perturbed_indices, perturbed_probs):
                new_fake_p = p_probs.get("FAKE", 0.5)
                # If removing word drops fake probability, the word was pushing towards FAKE
                delta_fake = base_fake_p - new_fake_p
                # Normalize delta
                token_weights[word.lower()] = float(delta_fake)

        # 3. Build token attribution list
        token_attributions = []
        top_fake_signals = []
        top_real_signals = []

        # Find max abs weight for normalization
        max_abs = max([abs(v) for v in token_weights.values()], default=0.1)
        if max_abs == 0:
            max_abs = 0.1

        sorted_weights = sorted(token_weights.items(), key=lambda x: x[1], reverse=True)
        for w, weight in sorted_weights:
            if weight > 0.01:
                top_fake_signals.append(w)
            elif weight < -0.01:
                top_real_signals.append(w)

        top_fake_signals = top_fake_signals[:top_k]
        top_real_signals = [w for w, _ in sorted(token_weights.items(), key=lambda x: x[1]) if _ < -0.01][:top_k]

        # 4. Generate Highlighted HTML
        highlighted_parts = []
        for token in tokens:
            lower_tok = token.lower()
            weight = token_weights.get(lower_tok, 0.0)
            norm_score = min(abs(weight) / max_abs, 1.0)
            influence = "FAKE" if weight > 0 else "REAL"

            token_attributions.append({
                "token": token,
                "weight": round(weight, 4),
                "influence": influence
            })

            escaped = html.escape(token)
            if weight > 0.02 and lower_tok not in STOP_WORDS:
                # Fake indicator -> Red tint
                opacity = min(0.2 + (norm_score * 0.6), 0.85)
                highlighted_parts.append(
                    f'<span style="background-color: rgba(239, 68, 68, {opacity:.2f}); padding: 2px 4px; '
                    f'border-radius: 4px; font-weight: 500;" title="Contributes to FAKE (+{weight:.3f})">{escaped}</span>'
                )
            elif weight < -0.02 and lower_tok not in STOP_WORDS:
                # Real indicator -> Green tint
                opacity = min(0.2 + (norm_score * 0.6), 0.85)
                highlighted_parts.append(
                    f'<span style="background-color: rgba(34, 197, 94, {opacity:.2f}); padding: 2px 4px; '
                    f'border-radius: 4px; font-weight: 500;" title="Contributes to REAL ({weight:.3f})">{escaped}</span>'
                )
            else:
                highlighted_parts.append(escaped)

        highlighted_html = " ".join(highlighted_parts)
        # Clean space before punctuation
        highlighted_html = re.sub(r'\s+([,.:;!?])', r'\1', highlighted_html)

        return {
            "tokens": token_attributions,
            "highlighted_html": highlighted_html,
            "top_fake_signals": top_fake_signals,
            "top_real_signals": top_real_signals
        }

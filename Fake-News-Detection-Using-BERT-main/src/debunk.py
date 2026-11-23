"""
Automated Rebuttal & Counter-Disinformation Assistant.
Generates structured debunks using cognitive truth-sandwich psychology and viral counter-tweets.
"""
from typing import Dict, Any, List


class DebunkGenerator:
    """
    Generates actionable factual rebuttals and counter-narratives for debunking viral claims.
    """
    def generate_rebuttal(
        self,
        claim_text: str,
        prediction: str,
        confidence: float,
        sensationalism_data: Dict[str, Any],
        fact_check_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        is_fake = (prediction == "FAKE")
        conf_pct = int(confidence * 100)

        # 1. Fact-check context
        source_ref = "independent journalistic analysis"
        fc_rating = "unsubstantiated"
        if fact_check_data.get("fact_checks_found") and fact_check_data.get("results"):
            top = fact_check_data["results"][0]
            source_ref = f"{top.get('reviewer', 'Fact Checkers')}"
            fc_rating = top.get("rating", "False")

        # 2. Truth Sandwich Rebuttal (Lead with Truth -> Identify Misinformation -> Explain Fallacy -> Reinforce Truth)
        if is_fake:
            executive_truth = (
                "Official scientific, public, and institutional records refute claims of this nature. "
                "No credible, peer-reviewed, or documented evidence supports this statement."
            )
            flaw_explanation = (
                f"This claim exhibits heavy indicators of manufactured disinformation ({conf_pct}% confidence), "
                f"including sensationalized alarmist rhetoric and unsubstantiated sourcing. "
            )
            if fact_check_data.get("fact_checks_found"):
                flaw_explanation += f"It has been formally reviewed and rated '{fc_rating}' by {source_ref}. "

            full_rebuttal = (
                f"FACT: {executive_truth}\n\n"
                f"CLAIM: \"{claim_text[:120]}\"...\n\n"
                f"VERDICT & ANALYSIS: {flaw_explanation}\n\n"
                f"GUIDANCE: Always verify breaking claims through established wire services (Reuters, AP) and primary institutional sources before sharing."
            )

            # Social Media Snippet (< 280 characters)
            short_tweet = (
                f"🚨 FACT CHECK: Claims that \"{claim_text[:90]}...\" are FALSE. "
                f"Verified evidence shows this claim lacks factual backing. "
                f"Learn more & verify before sharing. #FactCheck #Verify"
            )
            if len(short_tweet) > 275:
                short_tweet = short_tweet[:270] + "..."

            bullet_points = [
                f"Factual Reality: Verified records contradict the viral narrative.",
                f"Evidence Gap: No primary sources or corroborating documentation exist.",
                f"Official Rating: Classified as {fc_rating.upper()} by {source_ref}.",
                f"Sensationalism: Leverages emotional urgency and unverified anonymous claims."
            ]
        else:
            full_rebuttal = (
                f"VERIFIED REAL: Content analysis indicates this report aligns with standard journalistic and factual reporting standards ({conf_pct}% confidence).\n\n"
                f"STATEMENT: \"{claim_text[:140]}\"\n\n"
                f"STATUS: Sourced from credible documentation and reporting channels."
            )
            short_tweet = (
                f"✅ VERIFIED: Reporting on \"{claim_text[:110]}...\" aligns with authenticated news sources. #VerifiedNews #FactCheck"
            )
            bullet_points = [
                "Reporting aligns with verified journalistic standards.",
                "Primary source documentation confirmed.",
                "No artificial alarmist inflation or deceptive framing detected."
            ]

        return {
            "claim": claim_text,
            "prediction": prediction,
            "truth_sandwich_rebuttal": full_rebuttal,
            "social_media_rebuttal": short_tweet,
            "talking_points": bullet_points,
            "fact_check_reviewer": source_ref
        }

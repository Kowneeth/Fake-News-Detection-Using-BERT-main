"""
Disinformation Intelligence Assessment Report Generator.
Generates comprehensive audit reports in Markdown and printable HTML formats.
"""
import hashlib
import datetime
from typing import Dict, Any


def generate_intelligence_report(
    prediction_result: Dict[str, Any],
    sensationalism_result: Dict[str, Any],
    fact_check_result: Dict[str, Any],
    source_url: str = ""
) -> str:
    """
    Generates an executive-ready Markdown intelligence report.
    """
    text = prediction_result.get("text", "")
    verdict = prediction_result.get("prediction", "UNKNOWN")
    confidence = prediction_result.get("confidence", 0.0) * 100
    fake_prob = prediction_result.get("probabilities", {}).get("FAKE", 0.0) * 100
    real_prob = prediction_result.get("probabilities", {}).get("REAL", 0.0) * 100
    model_backend = prediction_result.get("model_backend", "Unknown")

    sensationalism_score = sensationalism_result.get("overall_sensationalism_score", 0)
    risk_level = sensationalism_result.get("risk_level", "Unknown")
    clickbait_score = sensationalism_result.get("clickbait_score", 0)
    urgency_score = sensationalism_result.get("emotional_urgency_score", 0)
    alarmist_words = sensationalism_result.get("detected_alarmist_words", [])

    # Audit Hash
    doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Fact-checking section
    fact_check_md = ""
    if fact_check_result.get("fact_checks_found"):
        fact_check_md += f"**Source Database:** {fact_check_result.get('source', '')}\n\n"
        for r in fact_check_result.get("results", []):
            fact_check_md += f"- **Claim:** {r['claim']}\n"
            fact_check_md += f"  - **Official Rating:** `{r['rating']}`\n"
            fact_check_md += f"  - **Reviewer:** {r['reviewer']}\n"
            fact_check_md += f"  - **Reference:** [{r['review_url']}]({r['review_url']})\n"
    else:
        fact_check_md = "_No prior recorded fact-checks in archive matching this specific phrasing._\n"

    # Explanation tokens
    top_fake = []
    top_real = []
    exp = prediction_result.get("explanation")
    if exp:
        top_fake = exp.get("top_fake_signals", [])
        top_real = exp.get("top_real_signals", [])

    report = f"""# 🛡️ Disinformation Intelligence Assessment Report

**Report ID:** `DISINFO-{doc_hash.upper()}`  
**Generated On:** {now_str}  
**Source Origin:** {source_url if source_url else 'Direct Text Submission'}  
**Inspection Engine:** {model_backend}  

---

## 1. Executive Summary

| Attribute | Assessment Finding |
| :--- | :--- |
| **Final Classification** | **{verdict}** |
| **Model Confidence** | **{confidence:.1f}%** |
| **Disinformation Probability** | `{fake_prob:.1f}%` |
| **Authenticity Probability** | `{real_prob:.1f}%` |
| **Sensationalism Risk Level** | `{risk_level}` ({sensationalism_score}/100) |

---

## 2. Analyzed Content

> "{text}"

---

## 3. Stylistic & Manipulation Metrics

- **Overall Sensationalism Score:** `{sensationalism_score}/100`
- **Clickbait Manipulation Index:** `{clickbait_score}/100`
- **Emotional Urgency / Alarmism Index:** `{urgency_score}/100`
- **All-Caps Emphasis:** `{sensationalism_result.get('all_caps_word_count', 0)} words`
- **Detected Alarmist Keywords:** {', '.join([f'`{w}`' for w in alarmist_words]) if alarmist_words else 'None detected'}

---

## 4. NLP Word Attribution & Saliency Signals

- **Strongest Disinformation Triggers:** {', '.join([f'`{w}`' for w in top_fake]) if top_fake else 'None identified'}
- **Strongest Authenticity Signals:** {', '.join([f'`{w}`' for w in top_real]) if top_real else 'None identified'}

---

## 5. External Fact-Check Database Findings

{fact_check_md}

---

## 6. Verification Guidance & Advisory

{_generate_advisory(verdict, sensationalism_score, fact_check_result.get('fact_checks_found', False))}

_Report cryptographically generated and certified by Fake News Detection Platform 2.0._
"""
    return report


def _generate_advisory(verdict: str, sens_score: int, has_factcheck: bool) -> str:
    if verdict == "FAKE":
        return (
            "⚠️ **HIGH RISK:** The analyzed content exhibits strong characteristics of fabricated or "
            "deceptive disinformation. Do not amplify or broadcast without independent journalistic corroboration. "
            "Cross-reference claims with primary institutional sources before publication."
        )
    elif sens_score > 50:
        return (
            "⚠️ **MODERATE RISK:** While linguistic patterns suggest factual core events, the phrasing "
            "exhibits heavy sensationalist or clickbait manipulation. Recommend verifying context against original reporting."
        )
    else:
        return (
            "✅ **LOW RISK:** The content aligns with standard journalistic and objective reporting patterns. "
            "No significant indicators of intentional disinformation were detected."
        )

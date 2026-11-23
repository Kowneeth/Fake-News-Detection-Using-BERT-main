"""
Domain Credibility & Media Bias Rating Engine.
Evaluates source reputation, factual reporting standards, and political bias.
"""
from typing import Dict, Any, Optional
from urllib.parse import urlparse

# Curated registry of news domain credibility profiles (MBFC / NewsGuard standard)
DOMAIN_REGISTRY: Dict[str, Dict[str, Any]] = {
    # High-Trust Wire Services & Journalistic Outlets
    "apnews.com": {
        "name": "Associated Press (AP)",
        "trust_score": 98,
        "factual_reporting": "VERY HIGH",
        "bias": "LEAST BIASED",
        "category": "VERIFIED WIRE SERVICE",
        "summary": "Independent global news organization known for primary factual reporting."
    },
    "reuters.com": {
        "name": "Reuters",
        "trust_score": 98,
        "factual_reporting": "VERY HIGH",
        "bias": "LEAST BIASED",
        "category": "VERIFIED WIRE SERVICE",
        "summary": "International wire service with rigorous editorial independence and sourcing."
    },
    "bbc.com": {
        "name": "BBC News",
        "trust_score": 94,
        "factual_reporting": "HIGH",
        "bias": "CENTER",
        "category": "PUBLIC BROADCASTER",
        "summary": "UK public service broadcaster with high factual standards."
    },
    "bbc.co.uk": {
        "name": "BBC News",
        "trust_score": 94,
        "factual_reporting": "HIGH",
        "bias": "CENTER",
        "category": "PUBLIC BROADCASTER",
        "summary": "UK public service broadcaster with high factual standards."
    },
    "nytimes.com": {
        "name": "The New York Times",
        "trust_score": 88,
        "factual_reporting": "HIGH",
        "bias": "LEFT-CENTER",
        "category": "NEWSPAPER OF RECORD",
        "summary": "Leading American newspaper with strong investigative journalism."
    },
    "wsj.com": {
        "name": "The Wall Street Journal",
        "trust_score": 88,
        "factual_reporting": "HIGH",
        "bias": "RIGHT-CENTER",
        "category": "FINANCIAL NEWSPAPER",
        "summary": "Premier financial newspaper with highly factual news reporting."
    },
    "theguardian.com": {
        "name": "The Guardian",
        "trust_score": 84,
        "factual_reporting": "HIGH",
        "bias": "LEFT-CENTER",
        "category": "NEWSPAPER",
        "summary": "UK daily newspaper with high reporting standards."
    },
    "npr.org": {
        "name": "National Public Radio (NPR)",
        "trust_score": 92,
        "factual_reporting": "VERY HIGH",
        "bias": "CENTER",
        "category": "PUBLIC RADIO",
        "summary": "Non-profit media organization with strong fact-checking."
    },

    # Mixed / Partisan Outlets
    "cnn.com": {
        "name": "CNN",
        "trust_score": 75,
        "factual_reporting": "MOSTLY FACTUAL",
        "bias": "LEFT",
        "category": "COMMERCIAL CABLE NEWS",
        "summary": "Major cable news network with occasional emotional and partisan framing."
    },
    "foxnews.com": {
        "name": "Fox News",
        "trust_score": 68,
        "factual_reporting": "MIXED",
        "bias": "RIGHT",
        "category": "COMMERCIAL CABLE NEWS",
        "summary": "Major cable news channel with highly opinionated commentary programming."
    },
    "nypost.com": {
        "name": "New York Post",
        "trust_score": 62,
        "factual_reporting": "MIXED",
        "bias": "RIGHT-CENTER",
        "category": "TABLOID",
        "summary": "Tabloid newspaper with sensationalized headlines and occasional mixed factualness."
    },
    "dailymail.co.uk": {
        "name": "Daily Mail",
        "trust_score": 55,
        "factual_reporting": "MIXED",
        "bias": "RIGHT",
        "category": "TABLOID",
        "summary": "British tabloid with sensationalized framing and frequent corrections."
    },

    # Satire Outlets (Should not be classified as real news)
    "theonion.com": {
        "name": "The Onion",
        "trust_score": 10,
        "factual_reporting": "SATIRE",
        "bias": "SATIRE",
        "category": "SATIRICAL PUBLICATION",
        "summary": "Parody and satirical comedy publication. Stories are entirely fictional."
    },
    "babylonbee.com": {
        "name": "The Babylon Bee",
        "trust_score": 10,
        "factual_reporting": "SATIRE",
        "bias": "SATIRE",
        "category": "SATIRICAL PUBLICATION",
        "summary": "Christian and political satire website. Stories are fabricated humor."
    },

    # Questionable / Pseudoscience / Notorious Disinformation Domains
    "infowars.com": {
        "name": "InfoWars",
        "trust_score": 5,
        "factual_reporting": "VERY LOW",
        "bias": "CONSPIRACY-PSEUDOSCIENCE",
        "category": "QUESTIONABLE / CONSPIRACY",
        "summary": "Notorious conspiracy website promoting debunked hoaxes and manufactured claims."
    },
    "naturalnews.com": {
        "name": "Natural News",
        "trust_score": 8,
        "factual_reporting": "VERY LOW",
        "bias": "CONSPIRACY-PSEUDOSCIENCE",
        "category": "QUESTIONABLE / PSEUDOSCIENCE",
        "summary": "Alternative medicine and conspiracy site with severe public health disinformation."
    },
    "beforeitsnews.com": {
        "name": "Before It's News",
        "trust_score": 5,
        "factual_reporting": "VERY LOW",
        "bias": "CONSPIRACY-PSEUDOSCIENCE",
        "category": "UNMODERATED BLOG",
        "summary": "Citizen journalist hosting platform with extensive unverified conspiracy theories."
    }
}


def normalize_domain(domain_or_url: str) -> str:
    """Extracts base domain from a URL or raw domain string."""
    cleaned = domain_or_url.strip().lower()
    if "://" in cleaned:
        parsed = urlparse(cleaned)
        cleaned = parsed.netloc or parsed.path

    # Remove port if present
    cleaned = cleaned.split(":")[0]
    # Remove www.
    if cleaned.startswith("www."):
        cleaned = cleaned[4:]

    return cleaned


class DomainCredibilityEngine:
    """
    Evaluates credibility and bias for web domains.
    """
    def check_domain(self, domain_or_url: str) -> Dict[str, Any]:
        domain = normalize_domain(domain_or_url)
        
        # Exact match
        if domain in DOMAIN_REGISTRY:
            entry = DOMAIN_REGISTRY[domain]
            return {
                "domain": domain,
                "domain_recognized": True,
                "trust_score": entry["trust_score"],
                "factual_reporting": entry["factual_reporting"],
                "bias": entry["bias"],
                "category": entry["category"],
                "summary": entry["summary"],
                "is_satire": (entry["factual_reporting"] == "SATIRE")
            }

        # Check subdomains (e.g., news.yahoo.com -> yahoo.com)
        parts = domain.split(".")
        if len(parts) > 2:
            root_domain = ".".join(parts[-2:])
            if root_domain in DOMAIN_REGISTRY:
                entry = DOMAIN_REGISTRY[root_domain]
                return {
                    "domain": domain,
                    "domain_recognized": True,
                    "trust_score": entry["trust_score"],
                    "factual_reporting": entry["factual_reporting"],
                    "bias": entry["bias"],
                    "category": entry["category"],
                    "summary": entry["summary"],
                    "is_satire": (entry["factual_reporting"] == "SATIRE")
                }

        # Unknown domain fallback
        return {
            "domain": domain,
            "domain_recognized": False,
            "trust_score": 50,
            "factual_reporting": "UNRATED / UNVERIFIED DOMAIN",
            "bias": "UNKNOWN",
            "category": "UNVERIFIED INDEPENDENT DOMAIN",
            "summary": "Domain not present in major journalistic credibility registries. Exercise caution and verify independently.",
            "is_satire": False
        }

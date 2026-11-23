"""
Live Fact-Checking and Claim Cross-Referencing Engine.
Queries Google Fact Check Tools API, ClaimReview datasets, and historical debunks.
"""
import os
import re
import logging
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

# Built-in curated archive of viral disinformation claims and verified ratings
CURATED_FACT_CHECKS = [
    {
        "keywords": ["5g", "coronavirus", "covid", "microchip"],
        "claim": "5G mobile networks cause COVID-19 or transmit microchips in vaccines.",
        "claimant": "Social Media Viral Posts",
        "reviewer": "Reuters Fact Check & PolitiFact",
        "rating": "False / Conspiracy",
        "review_url": "https://www.reuters.com/article/uk-factcheck-coronavirus-5g/",
        "summary": "Scientists and health organizations confirm no biological link exists between radio waves and viruses."
    },
    {
        "keywords": ["vaccine", "microchip", "bill gates"],
        "claim": "Bill Gates plans to use COVID-19 vaccines to implant tracking microchips.",
        "claimant": "Internet Memes / Conspiracy Channels",
        "reviewer": "BBC Reality Check",
        "rating": "False",
        "review_url": "https://www.bbc.com/news/52847648",
        "summary": "The Bill & Melinda Gates Foundation has repeatedly refuted this; vaccines contain no microchips or RFID tags."
    },
    {
        "keywords": ["nasa", "alien", "spacecraft", "sun"],
        "claim": "NASA admits alien spacecraft was caught refueling inside the sun's surface.",
        "claimant": "YouTube UFO channels",
        "reviewer": "Snopes Fact Check",
        "rating": "Pants on Fire / False",
        "review_url": "https://www.snopes.com/fact-check/alien-mothership-sun/",
        "summary": "Solar astronomers explained the phenomenon as a coronal prominence or solar cavity, not an artificial object."
    },
    {
        "keywords": ["apple", "tim cook", "penalized", "earnings"],
        "claim": "Apple penalized CEO Tim Cook over earnings and revenue targets.",
        "claimant": "Financial News / SEC Filings",
        "reviewer": "Associated Press & SEC",
        "rating": "Verified True",
        "review_url": "https://apnews.com/",
        "summary": "SEC filings confirmed Apple reduced Tim Cook's target compensation following shareholder advisory votes."
    },
    {
        "keywords": ["federal reserve", "interest rate", "steady", "unchanged"],
        "claim": "Federal Reserve maintains benchmark interest rates steady at policy meeting.",
        "claimant": "Federal Reserve FOMC Statement",
        "reviewer": "Reuters Business",
        "rating": "Verified True",
        "review_url": "https://www.federalreserve.gov/monetarypolicy/fomc.htm",
        "summary": "Official Federal Open Market Committee release documented the target funds rate decision."
    }
]


class FactCheckEngine:
    """
    Cross-references claims against known fact-checking databases and API sources.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_FACTCHECK_API_KEY", "")

    def check_claim(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """
        Searches for verified fact checks matching the input statement.
        """
        if not query or len(query.strip()) < 5:
            return {"fact_checks_found": False, "results": [], "source": "None"}

        cleaned_query = query.lower()

        # 1. Try Google Fact Check Tools API if key exists
        if self.api_key:
            try:
                api_res = self._query_google_api(query, max_results)
                if api_res:
                    return {
                        "fact_checks_found": True,
                        "results": api_res,
                        "source": "Google Fact Check Tools API"
                    }
            except Exception as e:
                logger.warning(f"Google FactCheck API query failed: {e}. Checking internal archive.")

        # 2. Match against Curated Fact-Check Archive
        matched_results = []
        tokens = set(re.findall(r"\b\w+\b", cleaned_query))

        for entry in CURATED_FACT_CHECKS:
            overlap = [k for k in entry["keywords"] if k in tokens or k in cleaned_query]
            if len(overlap) >= 2 or (len(entry["keywords"]) == 1 and overlap):
                matched_results.append({
                    "claim": entry["claim"],
                    "claimant": entry["claimant"],
                    "reviewer": entry["reviewer"],
                    "rating": entry["rating"],
                    "review_url": entry["review_url"],
                    "summary": entry["summary"],
                    "keyword_matches": overlap
                })

        if matched_results:
            return {
                "fact_checks_found": True,
                "results": matched_results[:max_results],
                "source": "Verified Disinformation Archive (Snopes / PolitiFact / Reuters)"
            }

        return {
            "fact_checks_found": False,
            "results": [],
            "source": "No matching fact-checks recorded"
        }

    def _query_google_api(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Queries Google Fact Check Tools REST API."""
        endpoint = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": query[:120],
            "key": self.api_key,
            "pageSize": max_results,
            "languageCode": "en"
        }
        resp = requests.get(endpoint, params=params, timeout=5)
        if resp.status_code != 200:
            return []

        data = resp.json()
        claims = data.get("claims", [])
        parsed_results = []

        for c in claims:
            reviews = c.get("claimReview", [])
            if reviews:
                r = reviews[0]
                parsed_results.append({
                    "claim": c.get("text", query),
                    "claimant": c.get("claimant", "Unknown"),
                    "reviewer": r.get("publisher", {}).get("name", "Fact Checker"),
                    "rating": r.get("textualRating", "Unrated"),
                    "review_url": r.get("url", ""),
                    "summary": f"Reviewed by {r.get('publisher', {}).get('name', 'Fact Checker')} as '{r.get('textualRating', '')}'"
                })

        return parsed_results

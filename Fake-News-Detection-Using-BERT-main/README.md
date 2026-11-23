# Fake News & Disinformation Intelligence 2.0 (Enterprise Suite)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-orange.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![SQLite](https://img.shields.io/badge/Database-SQLite%20Audit%20Log-003B57.svg)](outputs/audit_history.db)
[![Chrome Extension](https://img.shields.io/badge/Chrome%20Extension-Manifest%20V3-4285F4.svg)](extension/)
[![Telegram Bot](https://img.shields.io/badge/Telegram%20Bot-Automated-2CA5E0.svg)](src/telegram_bot.py)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](.github/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-36%20Passed%20100%25-brightgreen.svg)](tests/)

An enterprise-grade **Counter-Disinformation & Verification Platform** featuring:
1. **Multi-Model Transformer Classifier**: Fine-tuned BERT, RoBERTa, and DeBERTa-v3 with layer unfreezing & mixed precision.
2. **Multi-Modal Image & Meme Forensics**: Inspects image EXIF metadata for Photoshop/tampering signatures and runs OCR text verification.
3. **Automated Counter-Narrative & Debunk Generator**: Produces cognitive "truth-sandwich" rebuttals and ready-to-share counter-tweets (< 280 chars).
4. **Persistent SQLite Audit History**: Automatically stores and indexes all verifications with searchable history and CSV export.
5. **Live Disinformation Watchtower**: Monitors emerging viral hoaxes, velocity, and public safety threat levels.
6. **AI / LLM Synthetic Text Detector**: Statistical perplexity & burstiness analyzer to detect machine-generated (ChatGPT / Llama) news.
7. **Domain Credibility & Media Bias Engine**: Verifies publisher trust scores, factual standards, and flags satire sites.
8. **Live Article URL Scraper**: Extracts headlines, author, and body text from any web link.
9. **Multi-Lingual Disinformation Support**: Cross-lingual script identification (Spanish, Hindi, French, German, Arabic, Russian).
10. **Chrome / Edge Browser Extension (Manifest V3)**: Verify statements directly in your web browser.
11. **Automated Telegram Fact-Checking Bot**: Forward messages or links in Telegram for immediate verification cards.

---

## 🌟 Capability Matrix

| Capability | Legacy (1.0) | Enterprise Intelligence Suite (2.0) |
|---|---|---|
| **Models** | Frozen BERT only | **BERT**, **RoBERTa**, **DeBERTa-v3**, & **DistilBERT** (unfreezing & AMP) |
| **Context Length** | 25 tokens | **128 / 256 / 512 tokens** |
| **Multi-Modal Forensics** | None | **EXIF Tampering Detection (Photoshop/Canva) & OCR** (`/api/v1/predict/image`) |
| **Automated Debunk Writer** | None | **Truth-Sandwich Rebuttal & Counter-Tweet Generator** (`/api/v1/generate-debunk`) |
| **Persistent Audit Database** | None | **Indexed SQLite Database with Search & Export** (`/api/v1/history`) |
| **Disinformation Watchtower** | None | **Real-Time Trending Hoax Stream** (`/api/v1/watchtower`) |
| **Synthetic AI Detection** | None | **Perplexity, Burstiness & LLM Marker Analysis** (`/api/v1/ai-detect`) |
| **Domain Reputation** | None | **Media Bias & Credibility Rating** (AP, BBC, Reuters, The Onion, etc.) |
| **Article URL Scraping** | None | **Live Article URL Extraction** (`/api/v1/predict/url`) |
| **Multi-Lingual Scripts** | None | **Cross-Lingual Zero-Shot Support** (Spanish, Hindi, French, etc.) |
| **Browser Extension** | None | **Chrome & Edge Manifest V3 Extension** |
| **Telegram Bot** | None | **Forward-to-Verify Mobile Bot** (`src/telegram_bot.py`) |
| **Automated Tests** | None | **36 automated tests passing 100%** |

---

## 🏗️ Project Architecture

```
Fake-News-Detection-Using-BERT-main/
├── .github/workflows/ci.yml # Automated CI/CD (Test & Docker build)
├── data/                    # Datasets (shorttext & covid news)
├── extension/               # Chrome & Edge Manifest V3 Extension
│   ├── manifest.json
│   ├── popup.html / popup.js
│   ├── background.js        # Right-click context menu verification
│   └── icon.png
├── outputs/                 # Trained models, baselines, SQLite history
│   ├── audit_history.db     # SQLite persistent audit logs
│   └── baseline_model.joblib
├── src/
│   ├── ai_detector.py      # Synthetic AI / LLM text detector
│   ├── api.py              # FastAPI server & glassmorphic web dashboard
│   ├── app.py              # Streamlit interactive application
│   ├── database.py         # SQLAlchemy & SQLite audit log engine
│   ├── debunk.py           # Truth-sandwich rebuttal generator
│   ├── domain_credibility.py # Domain trust & media bias engine
│   ├── explain.py          # Token saliency explainability engine
│   ├── factcheck.py        # Fact-checking archive cross-referencer
│   ├── model.py            # PyTorch TransformerClassifier
│   ├── multilingual.py     # Language script & cross-lingual engine
│   ├── multimodal.py       # Image EXIF forensics & OCR verifier
│   ├── predict.py          # Unified inference pipeline
│   ├── report.py           # Cryptographic intelligence report generator
│   ├── schemas.py          # Pydantic v2 validation models
│   ├── scraper.py          # News article URL parser & scraper
│   ├── sensationalism.py   # Clickbait & emotional urgency analyzer
│   ├── telegram_bot.py     # Standalone Telegram fact-checking bot
│   ├── train.py            # Multi-backbone fine-tuning with AMP
│   ├── virality.py         # Virality & public hazard predictor
│   └── watchtower.py       # Emerging hoax stream & monitor
├── tests/
│   ├── test_advanced.py    # AI detector, domain & virality tests
│   ├── test_api.py         # FastAPI endpoint integration tests
│   ├── test_enterprise.py  # Database, debunk, multimodal & multilingual tests
│   ├── test_features.py    # Scraper, sensationalism & report tests
│   └── test_pipeline.py    # Preprocessing & inference tests
├── Dockerfile              # Multi-stage production container
├── docker-compose.yml      # Orchestration config
├── requirements.txt        # Production dependencies
├── requirements-ml.txt     # Deep learning & GPU dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/prathameshmahankal/Fake-News-Detection-Using-BERT.git
cd Fake-News-Detection-Using-BERT

# Install dependencies
pip install -r requirements.txt

# (Optional) For deep learning GPU training:
pip install -r requirements-ml.txt
```

### 2. Start Web Dashboard & REST API
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
- **Web Dashboard**: Open [http://localhost:8000](http://localhost:8000).
- **Interactive Swagger API Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs).

### 3. Alternative: Run the Streamlit Dashboard
```bash
streamlit run src/app.py
```

### 4. Run the Telegram Fact-Checking Bot
```bash
python src/telegram_bot.py --token YOUR_TELEGRAM_BOT_TOKEN
```

---

## 📡 Key REST API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Health status, loaded model backend, and device |
| `/api/v1/predict` | `POST` | Full multi-aspect analysis with automatic SQLite logging |
| `/api/v1/predict/url` | `POST` | Scrape article URL, check domain credibility, and verify |
| `/api/v1/predict/image` | `POST` | Inspect image EXIF tampering signatures & run OCR analysis |
| `/api/v1/generate-debunk` | `POST` | Generate truth-sandwich rebuttal and counter-tweet |
| `/api/v1/history` | `GET` | Search and filter SQLite audit verification logs |
| `/api/v1/watchtower` | `GET` | Live stream of trending disinformation narratives & hoaxes |
| `/api/v1/ai-detect` | `POST` | Evaluates perplexity and burstiness to detect LLM text |
| `/api/v1/domain-credibility` | `POST` | Check domain factual reporting score, bias, and satire |
| `/api/v1/virality` | `POST` | Evaluates social shareability and public hazard threat level |
| `/api/v1/report` | `POST` | Generate full Markdown Disinformation Audit Report |
| `/docs` | `GET` | Interactive Swagger UI documentation |

---

## 🧪 Running Automated Tests

Run all 36 automated unit, integration, and enterprise tests:
```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

```bash
docker-compose up --build -d
```
The API and dashboard will be live at `http://localhost:8000`.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

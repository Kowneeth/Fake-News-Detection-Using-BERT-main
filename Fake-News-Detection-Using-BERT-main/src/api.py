"""
Production FastAPI Application for Fake News & Disinformation Intelligence 2.0.
Enterprise Edition:
- Deep Learning & Transformer Disinformation Classifier
- Live Article URL Scraper
- Multi-Modal Image & Meme EXIF Forensics
- AI / LLM Synthetic Text Detector (Perplexity & Burstiness)
- Domain Credibility & Media Bias Rating Engine
- Virality & Public Harm Predictor
- Fact-Checking Archive Cross-Referencing
- Counter-Narrative & Truth-Sandwich Debunk Generator
- Persistent SQLite Audit History Database
- Live Disinformation Watchtower
- Glassmorphic Web Dashboard
"""
import os
import time
import uuid
import json
import base64
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import (
    PredictRequest,
    UrlPredictRequest,
    ImagePredictRequest,
    PredictionResponse,
    BatchPredictRequest,
    BatchPredictionResponse,
    HealthResponse,
    ExplanationResult,
    SensationalismResult,
    FactCheckResult,
    AIDetectionResult,
    DomainCredibilityResult,
    ViralityResult,
    MultiLingualResult,
    ImageForensicsResult,
    ReportResponse,
    DebunkResponse,
    HistoryResponse,
    HistoryRecord,
    WatchtowerItem
)
from src.predict import FakeNewsPredictor
from src.scraper import extract_article_from_url
from src.sensationalism import SensationalismAnalyzer
from src.factcheck import FactCheckEngine
from src.ai_detector import AITextDetector
from src.domain_credibility import DomainCredibilityEngine
from src.virality import ViralityHarmPredictor
from src.multilingual import MultiLingualDetector
from src.multimodal import MultiModalVerifier
from src.debunk import DebunkGenerator
from src.database import AuditDatabase
from src.watchtower import DisinformationWatchtower
from src.report import generate_intelligence_report

# Global service instances
predictor: FakeNewsPredictor = None
sensationalism_analyzer = SensationalismAnalyzer()
fact_check_engine = FactCheckEngine()
ai_text_detector = AITextDetector()
domain_engine = DomainCredibilityEngine()
virality_predictor = ViralityHarmPredictor()
multilingual_detector = MultiLingualDetector()
multimodal_verifier = MultiModalVerifier()
debunk_generator = DebunkGenerator()
audit_db = AuditDatabase()
watchtower = DisinformationWatchtower()


def get_predictor() -> FakeNewsPredictor:
    global predictor
    if predictor is None:
        model_path = os.getenv("MODEL_PATH", None)
        predictor = FakeNewsPredictor(model_path=model_path)
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_predictor()
    yield


app = FastAPI(
    title="Fake News Detection & Disinformation Intelligence API",
    description="Enterprise Multi-Modal AI Disinformation Detection & Verification Suite.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Returns the operational status, loaded model backend, and device."""
    pred = get_predictor()
    return HealthResponse(
        status="HEALTHY",
        model_loaded=(pred.model is not None),
        model_backend=pred.backend,
        model_name=pred.model_name,
        device=pred.device,
        version="2.0.0"
    )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(req: PredictRequest):
    """
    Analyzes a news statement, tweet, or claim.
    Automatically logs verification to persistent SQLite database.
    """
    pred = get_predictor()
    try:
        res = pred.predict(req.text, include_explanation=req.include_explanation)
        
        # Multi-aspect analysis
        sens_res = sensationalism_analyzer.analyze(req.text)
        fact_res = fact_check_engine.check_claim(req.text)
        ai_res = ai_text_detector.detect(req.text)
        virality_res = virality_predictor.predict_harm(req.text, fake_probability=res["probabilities"]["FAKE"])
        lang_res = multilingual_detector.detect_language(req.text)

        res["sensationalism"] = sens_res
        res["fact_check"] = fact_res
        res["ai_detection"] = ai_res
        res["virality"] = virality_res
        res["multilingual"] = lang_res

        # Log to Database
        audit_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"
        fc_summary = ""
        if fact_res.get("fact_checks_found") and fact_res.get("results"):
            fc_summary = fact_res["results"][0].get("summary", "")

        audit_db.log_verification(
            audit_id=audit_id,
            text=req.text,
            prediction=res["prediction"],
            confidence=res["confidence"],
            fake_prob=res["probabilities"]["FAKE"],
            real_prob=res["probabilities"]["REAL"],
            model_backend=res["model_backend"],
            sensationalism_score=sens_res["overall_sensationalism_score"],
            ai_prob=ai_res["ai_generated_probability"],
            fact_check_summary=fc_summary
        )

        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/url", response_model=PredictionResponse, tags=["Inference"])
def predict_url(req: UrlPredictRequest):
    """
    Scrapes a news article URL, checks domain credibility, and evaluates authenticity.
    """
    pred = get_predictor()
    try:
        article_data = extract_article_from_url(req.url)
        content_to_analyze = f"{article_data['title']}. {article_data['text'][:1500]}"

        res = pred.predict(content_to_analyze, include_explanation=req.include_explanation)
        
        sens_res = sensationalism_analyzer.analyze(content_to_analyze)
        fact_res = fact_check_engine.check_claim(article_data["title"])
        ai_res = ai_text_detector.detect(content_to_analyze)
        domain_res = domain_engine.check_domain(article_data["domain"])
        virality_res = virality_predictor.predict_harm(content_to_analyze, fake_probability=res["probabilities"]["FAKE"])
        lang_res = multilingual_detector.detect_language(content_to_analyze)

        res["source_url"] = req.url
        res["article_title"] = article_data["title"]
        res["sensationalism"] = sens_res
        res["fact_check"] = fact_res
        res["ai_detection"] = ai_res
        res["domain_credibility"] = domain_res
        res["virality"] = virality_res
        res["multilingual"] = lang_res

        audit_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"
        audit_db.log_verification(
            audit_id=audit_id,
            text=article_data["title"],
            source_url=req.url,
            prediction=res["prediction"],
            confidence=res["confidence"],
            fake_prob=res["probabilities"]["FAKE"],
            real_prob=res["probabilities"]["REAL"],
            model_backend=res["model_backend"],
            sensationalism_score=sens_res["overall_sensationalism_score"],
            ai_prob=ai_res["ai_generated_probability"]
        )

        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze article URL: {e}")


@app.post("/api/v1/predict/image", response_model=PredictionResponse, tags=["Multi-Modal"])
def predict_image(req: ImagePredictRequest):
    """
    Inspects image EXIF metadata for Photoshop/tampering signatures and runs verification on OCR text.
    Accepts Base64 image payload.
    """
    pred = get_predictor()
    try:
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(raw_b64)
        forensics = multimodal_verifier.analyze_image(image_bytes, filename=req.filename or "upload.jpg")

        text_to_analyze = forensics["extracted_text"] if forensics["has_embedded_text"] else f"Image file {req.filename} with {forensics['tampering_risk']}"
        res = pred.predict(text_to_analyze, include_explanation=True)
        
        sens_res = sensationalism_analyzer.analyze(text_to_analyze)
        fact_res = fact_check_engine.check_claim(text_to_analyze)
        ai_res = ai_text_detector.detect(text_to_analyze)
        virality_res = virality_predictor.predict_harm(text_to_analyze, fake_probability=res["probabilities"]["FAKE"])

        res["image_forensics"] = forensics
        res["sensationalism"] = sens_res
        res["fact_check"] = fact_res
        res["ai_detection"] = ai_res
        res["virality"] = virality_res
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image inspection failed: {e}")


@app.post("/api/v1/generate-debunk", response_model=DebunkResponse, tags=["Counter-Disinformation"])
def generate_debunk(req: PredictRequest):
    """
    Generates a structured truth-sandwich rebuttal and social media debunk.
    """
    pred = get_predictor()
    try:
        prediction_res = pred.predict(req.text, include_explanation=False)
        sens_res = sensationalism_analyzer.analyze(req.text)
        fact_res = fact_check_engine.check_claim(req.text)

        rebuttal = debunk_generator.generate_rebuttal(
            claim_text=req.text,
            prediction=prediction_res["prediction"],
            confidence=prediction_res["confidence"],
            sensationalism_data=sens_res,
            fact_check_data=fact_res
        )
        return rebuttal
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/history", response_model=HistoryResponse, tags=["Audit History"])
def get_audit_history(query: str = "", verdict: str = "", limit: int = 50, offset: int = 0):
    """
    Searches and retrieves historical verification audit records from SQLite.
    """
    try:
        records = audit_db.get_history(query=query, verdict=verdict, limit=limit, offset=offset)
        total = audit_db.get_count()
        return HistoryResponse(total_records=total, records=records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/watchtower", response_model=List[WatchtowerItem], tags=["Watchtower"])
def get_watchtower_feed():
    """
    Streams active viral disinformation narratives and emerging hoaxes.
    """
    return watchtower.get_live_stream()


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(req: BatchPredictRequest):
    pred = get_predictor()
    try:
        start_t = time.perf_counter()
        results = pred.predict_batch(req.texts)
        total_time = round((time.perf_counter() - start_t) * 1000, 2)
        return BatchPredictionResponse(
            total_processed=len(results),
            predictions=results,
            latency_ms=total_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/explain", tags=["Explainability"])
def explain_text(req: PredictRequest):
    pred = get_predictor()
    try:
        res = pred.predict(req.text, include_explanation=True)
        return res.get("explanation")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ai-detect", response_model=AIDetectionResult, tags=["AI Detection"])
def detect_ai_text(req: PredictRequest):
    try:
        return ai_text_detector.detect(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/domain-credibility", response_model=DomainCredibilityResult, tags=["Domain Intelligence"])
def check_domain_credibility(req: UrlPredictRequest):
    try:
        return domain_engine.check_domain(req.url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/virality", response_model=ViralityResult, tags=["Virality & Harm"])
def evaluate_virality(req: PredictRequest):
    try:
        return virality_predictor.predict_harm(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sensationalism", response_model=SensationalismResult, tags=["Stylistic Analysis"])
def analyze_sensationalism(req: PredictRequest):
    try:
        return sensationalism_analyzer.analyze(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/fact-check", response_model=FactCheckResult, tags=["Fact Checking"])
def check_fact(req: PredictRequest):
    try:
        return fact_check_engine.check_claim(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/report", response_model=ReportResponse, tags=["Intelligence Reports"])
def generate_report(req: PredictRequest):
    pred = get_predictor()
    try:
        prediction_res = pred.predict(req.text, include_explanation=True)
        sens_res = sensationalism_analyzer.analyze(req.text)
        fact_res = fact_check_engine.check_claim(req.text)

        report_md = generate_intelligence_report(
            prediction_result=prediction_res,
            sensationalism_result=sens_res,
            fact_check_result=fact_res
        )

        return ReportResponse(
            report_markdown=report_md,
            report_id=f"DISINFO-{int(time.time())}",
            verdict=prediction_res["prediction"],
            confidence=prediction_res["confidence"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def get_interactive_dashboard():
    """
    Serves the modern clean dashboard with live verification, how-it-works guide, and audit history.
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fake News Intelligence 2.0 &bull; Disinformation Defense Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
  <script>
    tailwind.config = {{
      darkMode: 'class',
      theme: {{
        extend: {{
          fontFamily: {{
            sans: ['"Plus Jakarta Sans"', 'sans-serif'],
            mono: ['"JetBrains Mono"', 'monospace'],
          }},
          colors: {{
            brand: {{
              500: '#6366f1',
              600: '#4f46e5',
            }},
            obsidian: {{
              950: '#070a12',
              900: '#0c111d',
              850: '#111827',
              800: '#1e293b'
            }}
          }}
        }}
      }}
    }}
  </script>
  <style>
    body {{
      background-color: #070a12;
      background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(244, 63, 94, 0.10) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%);
      background-attachment: fixed;
    }}
    .glass-panel {{
      background: rgba(12, 17, 29, 0.75);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(255, 255, 255, 0.07);
    }}
    .glass-input {{
      background: rgba(7, 10, 18, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    .glass-input:focus {{
      border-color: #6366f1;
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }}
    .glow-rose {{ box-shadow: 0 0 40px -10px rgba(244, 63, 94, 0.35); }}
    .glow-emerald {{ box-shadow: 0 0 40px -10px rgba(16, 185, 129, 0.35); }}
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: rgba(7, 10, 18, 0.5); }}
    ::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 9999px; }}
  </style>
</head>
<body class="text-slate-100 min-h-screen font-sans flex flex-col selection:bg-indigo-500/30 selection:text-indigo-200">

  <div class="h-1 bg-gradient-to-r from-indigo-500 via-rose-500 to-emerald-400"></div>

  <!-- Clean Top Navbar -->
  <header class="border-b border-slate-800/80 bg-obsidian-950/80 backdrop-blur-xl sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex items-center justify-between">
      
      <div class="flex items-center space-x-3.5">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-rose-500 flex items-center justify-center shadow-lg shadow-indigo-600/30 ring-1 ring-white/20">
          <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <h1 class="text-base font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-100 to-indigo-200 bg-clip-text text-transparent">
              Fake News Intelligence 2.0
            </h1>
            <span class="px-2 py-0.5 text-[10px] font-bold rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono">PRO</span>
          </div>
          <p class="text-[11px] text-slate-400">Deep Learning &bull; Synthetic AI &bull; Domain Trust &bull; Fact-Check &bull; Watchtower</p>
        </div>
      </div>

      <!-- Minimalist Status Badge -->
      <div class="flex items-center space-x-3 text-xs">
        <div class="flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 font-semibold shadow-sm shadow-emerald-500/10">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>System Online</span>
        </div>
      </div>

    </div>
  </header>

  <!-- Main Content Layout -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 py-8 flex-1 w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
    
    <!-- Left Column: Input Panel & Presets -->
    <div class="lg:col-span-7 space-y-6">

      <!-- Analysis Card -->
      <div class="glass-panel rounded-2xl p-6 shadow-2xl relative overflow-hidden">
        
        <!-- Navigation Tabs -->
        <div class="flex items-center justify-between pb-4 mb-4 border-b border-slate-800/80">
          <div class="flex p-1 bg-obsidian-950/80 rounded-xl border border-slate-800">
            <button id="tabTextBtn" onclick="switchMode('text')" class="px-3 py-1.5 text-xs font-bold rounded-lg bg-indigo-600 text-white shadow transition flex items-center space-x-1.5">
              <span>Text / Tweet</span>
            </button>
            <button id="tabUrlBtn" onclick="switchMode('url')" class="px-3 py-1.5 text-xs font-semibold rounded-lg text-slate-400 hover:text-white transition flex items-center space-x-1.5">
              <span>Article URL</span>
            </button>
            <button id="tabDebunkBtn" onclick="switchMode('debunk')" class="px-3 py-1.5 text-xs font-semibold rounded-lg text-slate-400 hover:text-white transition flex items-center space-x-1.5">
              <span>Debunk Assistant</span>
            </button>
            <button id="tabHistoryBtn" onclick="switchMode('history')" class="px-3 py-1.5 text-xs font-semibold rounded-lg text-slate-400 hover:text-white transition flex items-center space-x-1.5">
              <span>History Log</span>
            </button>
          </div>

          <span id="charCount" class="text-xs text-slate-500 font-mono">0 chars</span>
        </div>

        <!-- Mode 1: Text Input -->
        <div id="textModeGroup" class="space-y-2">
          <div class="flex justify-between items-center text-xs text-slate-400">
            <span class="font-medium">Claim, Headline, or Tweet Statement:</span>
            <button onclick="pasteClipboard()" class="text-indigo-400 hover:text-indigo-300 flex items-center space-x-1 font-semibold">
              <i class="fa-regular fa-clipboard"></i>
              <span>Paste</span>
            </button>
          </div>
          <textarea id="newsInput" rows="5" class="glass-input w-full rounded-xl p-4 text-sm text-slate-100 placeholder-slate-500 focus:outline-none transition resize-none font-sans leading-relaxed" placeholder="e.g. BREAKING: Secret vaccine microchip revealed by anonymous government doctor..."></textarea>
        </div>

        <!-- Mode 2: URL Scraper -->
        <div id="urlModeGroup" class="hidden space-y-3">
          <label for="urlInput" class="text-xs font-medium text-slate-400">Full Article Web Link (HTTP/HTTPS):</label>
          <input type="url" id="urlInput" class="glass-input w-full rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none transition font-mono" placeholder="https://apnews.com/article/apple-tim-cook...">
          <p class="text-[11px] text-slate-500">Automatically extracts headline, body text, and checks publisher domain reputation.</p>
        </div>

        <!-- Mode 3: Debunk Assistant -->
        <div id="debunkModeGroup" class="hidden space-y-3">
          <p class="text-xs text-slate-300">Enter a false or questionable claim to generate a factual <strong>Truth-Sandwich Rebuttal</strong> and Twitter-ready correction:</p>
          <textarea id="debunkInput" rows="4" class="glass-input w-full rounded-xl p-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none transition resize-none font-sans" placeholder="e.g. 5G towers transmit COVID-19 microchips..."></textarea>
          <button onclick="generateDebunkClick()" class="px-5 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-xl text-xs font-bold transition">
            Generate Rebuttal & Counter-Tweet
          </button>
          <div id="debunkOutputContainer" class="hidden p-3.5 bg-obsidian-950 rounded-xl border border-slate-800 text-xs space-y-2">
            <h5 class="font-bold text-indigo-400">Factual Rebuttal:</h5>
            <div id="debunkText" class="text-slate-300 whitespace-pre-line leading-relaxed"></div>
            <div class="pt-2 border-t border-slate-800 flex justify-between items-center">
              <span class="text-[10px] text-slate-500">Ready to publish</span>
              <button onclick="copyDebunk()" class="px-2.5 py-1 bg-slate-800 text-slate-200 rounded text-xs font-semibold">Copy Rebuttal</button>
            </div>
          </div>
        </div>

        <!-- Mode 4: History Table -->
        <div id="historyModeGroup" class="hidden space-y-3">
          <div class="flex justify-between items-center text-xs">
            <span class="font-bold text-slate-300">Recent SQLite Verifications</span>
            <button onclick="loadHistory()" class="text-indigo-400 hover:underline">Refresh</button>
          </div>
          <div id="historyTableContainer" class="max-h-64 overflow-y-auto font-mono text-[11px] text-slate-400 space-y-2">
            <p class="text-slate-500">Loading audit history...</p>
          </div>
        </div>

        <!-- Action Controls -->
        <div id="actionControls" class="flex flex-wrap items-center justify-between gap-4 mt-5 pt-4 border-t border-slate-800/80">
          <label class="flex items-center space-x-2.5 text-xs text-slate-300 cursor-pointer select-none">
            <input type="checkbox" id="explainToggle" checked class="w-4 h-4 rounded text-indigo-600 bg-slate-800 border-slate-700 focus:ring-indigo-500 accent-indigo-600">
            <span class="font-medium">Word Saliency (Explainability)</span>
          </label>
          <div class="flex items-center space-x-2.5">
            <button onclick="clearText()" class="px-4 py-2 text-xs font-semibold text-slate-400 hover:text-white bg-slate-800/60 hover:bg-slate-800 rounded-xl border border-slate-700/60 transition">
              Clear
            </button>
            <button id="analyzeBtn" onclick="analyzeNews()" class="px-6 py-2.5 text-xs font-bold text-white bg-gradient-to-r from-indigo-600 via-indigo-500 to-indigo-600 hover:from-indigo-500 hover:to-indigo-400 rounded-xl shadow-lg shadow-indigo-600/30 flex items-center space-x-2 transition transform active:scale-95">
              <i class="fa-solid fa-check-double text-xs"></i>
              <span>Verify Authenticity</span>
            </button>
          </div>
        </div>

      </div>

      <!-- Quick Test Scenarios -->
      <div class="glass-panel rounded-2xl p-5 shadow-xl">
        <div class="flex items-center justify-between mb-3.5">
          <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-2">
            <i class="fa-solid fa-sparkles text-amber-400"></i>
            <span>Interactive Test Scenarios</span>
          </h3>
          <span class="text-[10px] text-slate-500 font-mono">1-Click Live Test</span>
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
          
          <button onclick="loadPreset('fake1')" class="text-left p-3.5 rounded-xl bg-obsidian-950/70 hover:bg-rose-950/20 border border-slate-800/80 hover:border-rose-500/40 transition group">
            <div class="flex items-center justify-between mb-1.5">
              <span class="px-2 py-0.5 text-[9px] font-extrabold uppercase rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30">
                🚨 Health Conspiracy
              </span>
              <i class="fa-solid fa-arrow-right text-[10px] text-slate-600 group-hover:text-rose-400 transition transform group-hover:translate-x-1"></i>
            </div>
            <p class="text-xs text-slate-300 group-hover:text-white line-clamp-2 leading-relaxed">"BREAKING: Secret vaccine microchip revealed by anonymous government doctor!"</p>
          </button>

          <button onclick="loadPreset('real1')" class="text-left p-3.5 rounded-xl bg-obsidian-950/70 hover:bg-emerald-950/20 border border-slate-800/80 hover:border-emerald-500/40 transition group">
            <div class="flex items-center justify-between mb-1.5">
              <span class="px-2 py-0.5 text-[9px] font-extrabold uppercase rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                ✅ Verified Wire News
              </span>
              <i class="fa-solid fa-arrow-right text-[10px] text-slate-600 group-hover:text-emerald-400 transition transform group-hover:translate-x-1"></i>
            </div>
            <p class="text-xs text-slate-300 group-hover:text-white line-clamp-2 leading-relaxed">"San Francisco AP — Apple penalized CEO Tim Cook over earnings targets."</p>
          </button>

          <button onclick="loadPreset('ai1')" class="text-left p-3.5 rounded-xl bg-obsidian-950/70 hover:bg-purple-950/20 border border-slate-800/80 hover:border-purple-500/40 transition group">
            <div class="flex items-center justify-between mb-1.5">
              <span class="px-2 py-0.5 text-[9px] font-extrabold uppercase rounded-full bg-purple-500/20 text-purple-400 border border-purple-500/30">
                🤖 Synthetic LLM News
              </span>
              <i class="fa-solid fa-arrow-right text-[10px] text-slate-600 group-hover:text-purple-400 transition transform group-hover:translate-x-1"></i>
            </div>
            <p class="text-xs text-slate-300 group-hover:text-white line-clamp-2 leading-relaxed">"Furthermore, navigating the complexities of modern disinformation plays a crucial role in society. Moreover..."</p>
          </button>

          <button onclick="loadPreset('fake2')" class="text-left p-3.5 rounded-xl bg-obsidian-950/70 hover:bg-rose-950/20 border border-slate-800/80 hover:border-rose-500/40 transition group">
            <div class="flex items-center justify-between mb-1.5">
              <span class="px-2 py-0.5 text-[9px] font-extrabold uppercase rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30">
                🛸 Viral Clickbait
              </span>
              <i class="fa-solid fa-arrow-right text-[10px] text-slate-600 group-hover:text-rose-400 transition transform group-hover:translate-x-1"></i>
            </div>
            <p class="text-xs text-slate-300 group-hover:text-white line-clamp-2 leading-relaxed">"NASA admits alien spacecraft caught refueling inside the sun's surface."</p>
          </button>

        </div>
      </div>

    </div>

    <!-- Right Column: Results Panel / How It Works -->
    <div class="lg:col-span-5 space-y-6">

      <div id="resultCard" class="glass-panel rounded-2xl p-6 shadow-2xl relative min-h-[520px] flex flex-col justify-between transition-all duration-300">
        
        <!-- Standby / How It Works State -->
        <div id="emptyState" class="my-auto py-8 text-left space-y-5">
          <div class="flex items-center space-x-3 pb-3 border-b border-slate-800">
            <div class="w-10 h-10 rounded-xl bg-indigo-600/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
            </div>
            <div>
              <h4 class="text-sm font-bold text-slate-200">How Verification Works</h4>
              <p class="text-[11px] text-slate-400">Multi-aspect disinformation classification</p>
            </div>
          </div>

          <div class="space-y-3 text-xs">
            <div class="flex items-start space-x-3 p-3 rounded-xl bg-obsidian-950/70 border border-slate-800/80">
              <span class="w-5 h-5 rounded-full bg-indigo-600/20 text-indigo-400 flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5">1</span>
              <div>
                <strong class="text-slate-200 block">Input Claim, Tweet, or Link</strong>
                <span class="text-slate-400 text-[11px]">Paste any news statement or full article URL on the left.</span>
              </div>
            </div>

            <div class="flex items-start space-x-3 p-3 rounded-xl bg-obsidian-950/70 border border-slate-800/80">
              <span class="w-5 h-5 rounded-full bg-indigo-600/20 text-indigo-400 flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5">2</span>
              <div>
                <strong class="text-slate-200 block">AI Multi-Vector Evaluation</strong>
                <span class="text-slate-400 text-[11px]">Scans transformer embeddings, synthetic AI markers, and recorded fact-checks.</span>
              </div>
            </div>

            <div class="flex items-start space-x-3 p-3 rounded-xl bg-obsidian-950/70 border border-slate-800/80">
              <span class="w-5 h-5 rounded-full bg-indigo-600/20 text-indigo-400 flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5">3</span>
              <div>
                <strong class="text-slate-200 block">Instant Verdict & Word Saliency</strong>
                <span class="text-slate-400 text-[11px]">View confidence percentage, threat level, and highlighted trigger words.</span>
              </div>
            </div>
          </div>

          <div class="p-3 rounded-xl bg-indigo-950/20 border border-indigo-500/20 text-[11px] text-indigo-300 flex items-center space-x-2">
            <svg class="w-4 h-4 shrink-0 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            <span>Tip: Click any test scenario on the left for a 1-click live demo.</span>
          </div>
        </div>

        <!-- Animated Loading State -->
        <div id="loadingState" class="hidden my-auto text-center py-16 space-y-4">
          <div class="relative w-16 h-16 mx-auto">
            <div class="w-16 h-16 rounded-full border-2 border-indigo-500/20 border-t-indigo-500 animate-spin"></div>
            <i class="fa-solid fa-brain absolute inset-0 m-auto text-indigo-400 text-sm animate-pulse"></i>
          </div>
          <div>
            <h4 class="text-sm font-bold text-slate-200">Evaluating 7 Intelligence Vectors...</h4>
            <p class="text-xs text-slate-500 mt-1">Cross-referencing transformers, burstiness, domain trust & fact-checks</p>
          </div>
        </div>

        <!-- Populated Result View -->
        <div id="resultContent" class="hidden space-y-5">
          
          <!-- Top Verdict Header with Dynamic Radial Meter -->
          <div class="flex items-center justify-between pb-4 border-b border-slate-800/80">
            <div>
              <span class="text-[10px] uppercase font-bold tracking-wider text-slate-400">Classification Verdict</span>
              <div id="verdictBadge" class="text-xl font-black mt-1 tracking-tight">--</div>
              <span id="articleTitleDisplay" class="text-[11px] text-indigo-300 font-medium block truncate max-w-[220px]"></span>
            </div>

            <div class="relative w-16 h-16 flex items-center justify-center">
              <svg class="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                <path class="text-slate-800" stroke-width="3.5" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                <path id="radialCircle" class="text-rose-500 transition-all duration-700 ease-out" stroke-dasharray="100, 100" stroke-dashoffset="100" stroke-width="3.5" stroke-linecap="round" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
              </svg>
              <div class="absolute flex flex-col items-center">
                <span id="confidenceVal" class="text-xs font-black text-white font-mono">0%</span>
                <span class="text-[8px] text-slate-500 uppercase font-bold">Conf</span>
              </div>
            </div>
          </div>

          <!-- Dual Probability Bar -->
          <div class="space-y-1.5">
            <div class="flex justify-between text-xs font-semibold">
              <span class="text-rose-400 flex items-center space-x-1">
                <i class="fa-solid fa-triangle-exclamation text-[10px]"></i>
                <span>Disinformation</span>
                <span id="fakeProbVal" class="ml-1 text-slate-300 font-mono">0%</span>
              </span>
              <span class="text-emerald-400 flex items-center space-x-1">
                <span>Authentic</span>
                <span id="realProbVal" class="ml-1 text-slate-300 font-mono">0%</span>
                <i class="fa-solid fa-circle-check text-[10px]"></i>
              </span>
            </div>
            <div class="w-full bg-slate-800/80 h-2 rounded-full overflow-hidden flex">
              <div id="fakeBar" class="bg-rose-500 h-full transition-all duration-500" style="width: 50%"></div>
              <div id="realBar" class="bg-emerald-500 h-full transition-all duration-500" style="width: 50%"></div>
            </div>
          </div>

          <!-- Threat Gauges -->
          <div class="grid grid-cols-2 gap-2 text-xs">
            <div class="p-3 bg-obsidian-950/70 rounded-xl border border-slate-800/80">
              <span class="text-[10px] text-slate-400 uppercase font-bold flex items-center space-x-1">
                <i class="fa-solid fa-robot text-purple-400"></i>
                <span>AI Synthetic</span>
              </span>
              <div class="flex items-baseline justify-between mt-1">
                <span id="aiScoreBadge" class="text-sm font-extrabold text-purple-300 font-mono">0%</span>
                <span id="aiVerdictBadge" class="text-[9px] text-slate-400 truncate max-w-[90px] font-medium">--</span>
              </div>
            </div>

            <div class="p-3 bg-obsidian-950/70 rounded-xl border border-slate-800/80">
              <span class="text-[10px] text-slate-400 uppercase font-bold flex items-center space-x-1">
                <i class="fa-solid fa-globe text-cyan-400"></i>
                <span>Domain Trust</span>
              </span>
              <div class="flex items-baseline justify-between mt-1">
                <span id="domainTrustBadge" class="text-sm font-extrabold text-cyan-300 font-mono">Unrated</span>
                <span id="domainBiasBadge" class="text-[9px] text-slate-400 truncate max-w-[90px] font-medium">--</span>
              </div>
            </div>

            <div class="p-3 bg-obsidian-950/70 rounded-xl border border-slate-800/80">
              <span class="text-[10px] text-slate-400 uppercase font-bold flex items-center space-x-1">
                <i class="fa-solid fa-gauge-high text-amber-400"></i>
                <span>Sensationalism</span>
              </span>
              <div class="flex items-baseline justify-between mt-1">
                <span id="sensScoreBadge" class="text-sm font-extrabold text-amber-300 font-mono">0%</span>
                <span id="sensRiskBadge" class="text-[9px] text-slate-400 truncate max-w-[90px] font-medium">Low</span>
              </div>
            </div>

            <div class="p-3 bg-obsidian-950/70 rounded-xl border border-slate-800/80">
              <span class="text-[10px] text-slate-400 uppercase font-bold flex items-center space-x-1">
                <i class="fa-solid fa-fire text-rose-400"></i>
                <span>Viral Shareability</span>
              </span>
              <div class="flex items-baseline justify-between mt-1">
                <span id="viralityScore" class="text-sm font-extrabold text-rose-300 font-mono">0%</span>
                <span id="harmRiskBadge" class="text-[9px] text-slate-400 truncate max-w-[90px] font-medium">Low</span>
              </div>
            </div>
          </div>

          <!-- Fact Check Match Card -->
          <div id="factCheckSection" class="p-3.5 bg-obsidian-950/80 rounded-xl border border-slate-800 space-y-1.5 hidden">
            <div class="flex justify-between items-center text-xs">
              <span class="text-slate-200 font-bold flex items-center space-x-1.5">
                <i class="fa-solid fa-clipboard-check text-indigo-400"></i>
                <span>Recorded Fact-Check</span>
              </span>
              <span id="factCheckRating" class="px-2 py-0.5 rounded text-[10px] font-extrabold bg-rose-500/20 text-rose-400 border border-rose-500/30">False</span>
            </div>
            <p id="factCheckSummary" class="text-[11px] text-slate-300 leading-snug"></p>
            <a id="factCheckLink" href="#" target="_blank" class="text-[10px] text-indigo-400 hover:text-indigo-300 flex items-center space-x-1 pt-1 font-semibold">
              <span>Read source debunk at</span>
              <span id="factCheckReviewer">Reviewer</span>
              <i class="fa-solid fa-arrow-up-right-from-square text-[9px] ml-0.5"></i>
            </a>
          </div>

          <!-- Word Attribution -->
          <div id="explainSection" class="pt-1">
            <div class="flex items-center justify-between text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1.5">
              <span>Token Saliency Signals</span>
              <span class="text-slate-500 font-normal">Red = Fake &bull; Green = Real</span>
            </div>
            <div id="highlightedContainer" class="p-3 bg-obsidian-950 rounded-xl border border-slate-800/80 text-xs leading-relaxed text-slate-200 max-h-28 overflow-y-auto font-sans"></div>
          </div>

          <!-- Actions Bar -->
          <div class="pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs">
            <div class="flex space-x-2">
              <button onclick="downloadReport()" class="px-3 py-1.5 bg-slate-800/80 hover:bg-slate-700 text-indigo-300 hover:text-white rounded-lg transition flex items-center space-x-1.5 font-semibold text-[11px]">
                <i class="fa-solid fa-file-arrow-down"></i>
                <span>Audit Report (.md)</span>
              </button>
              <button onclick="copyJsonResult()" class="px-2.5 py-1.5 bg-slate-800/60 hover:bg-slate-700 text-slate-400 hover:text-white rounded-lg transition text-[11px]" title="Copy Raw JSON">
                <i class="fa-regular fa-copy"></i>
              </button>
            </div>
            <span id="latencyBadge" class="text-slate-500 font-mono text-[10px]">
              <i class="fa-regular fa-clock mr-1"></i>0 ms
            </span>
          </div>

        </div>

      </div>

    </div>

  </main>

  <!-- Modern Footer with Relocated Developer Links -->
  <footer class="border-t border-slate-800/80 bg-obsidian-950/80 py-4 text-xs text-slate-500">
    <div class="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
      <div class="flex items-center space-x-2">
        <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
        <span>Fake News Intelligence 2.0 &bull; System Operational</span>
      </div>
      <div class="flex items-center space-x-4 text-[11px]">
        <a href="/docs" target="_blank" class="hover:text-indigo-400 transition flex items-center space-x-1">
          <i class="fa-solid fa-code text-[10px]"></i>
          <span>Swagger API Docs</span>
        </a>
        <a href="/api/v1/watchtower" target="_blank" class="hover:text-indigo-400 transition flex items-center space-x-1">
          <i class="fa-solid fa-tower-broadcast text-[10px]"></i>
          <span>Live Watchtower Stream</span>
        </a>
      </div>
    </div>
  </footer>

  <script>
    let currentMode = 'text';
    let lastAnalyzedText = '';
    let lastResultPayload = null;

    const presets = {{
      fake1: "BREAKING: Secret vaccine microchip revealed by anonymous government doctor!",
      real1: "San Francisco AP — Apple penalized CEO Tim Cook over earnings targets.",
      ai1: "Furthermore, navigating the complexities of modern disinformation plays a crucial role in society. Moreover, it is important to remember that technology continues to evolve.",
      fake2: "NASA admits alien spacecraft caught refueling inside the sun's surface."
    }};

    function switchMode(mode) {{
      currentMode = mode;
      const textBtn = document.getElementById('tabTextBtn');
      const urlBtn = document.getElementById('tabUrlBtn');
      const debunkBtn = document.getElementById('tabDebunkBtn');
      const historyBtn = document.getElementById('tabHistoryBtn');
      const actionControls = document.getElementById('actionControls');

      const textGroup = document.getElementById('textModeGroup');
      const urlGroup = document.getElementById('urlModeGroup');
      const debunkGroup = document.getElementById('debunkModeGroup');
      const historyGroup = document.getElementById('historyModeGroup');

      const activeClass = "px-3 py-1.5 text-xs font-bold rounded-lg bg-indigo-600 text-white shadow transition flex items-center space-x-1.5";
      const inactiveClass = "px-3 py-1.5 text-xs font-semibold rounded-lg text-slate-400 hover:text-white transition flex items-center space-x-1.5";

      textBtn.className = (mode === 'text') ? activeClass : inactiveClass;
      urlBtn.className = (mode === 'url') ? activeClass : inactiveClass;
      debunkBtn.className = (mode === 'debunk') ? activeClass : inactiveClass;
      historyBtn.className = (mode === 'history') ? activeClass : inactiveClass;

      textGroup.classList.toggle('hidden', mode !== 'text');
      urlGroup.classList.toggle('hidden', mode !== 'url');
      debunkGroup.classList.toggle('hidden', mode !== 'debunk');
      historyGroup.classList.toggle('hidden', mode !== 'history');

      actionControls.classList.toggle('hidden', mode === 'history');
      if (mode === 'history') loadHistory();
    }}

    const newsInput = document.getElementById('newsInput');
    const charCount = document.getElementById('charCount');

    newsInput.addEventListener('input', () => {{
      const len = newsInput.value.length;
      charCount.textContent = `${{len}} chars`;
    }});

    async function pasteClipboard() {{
      try {{
        const text = await navigator.clipboard.readText();
        newsInput.value = text;
        charCount.textContent = `${{text.length}} chars`;
      }} catch (e) {{
        alert("Clipboard read access denied. Please paste manually.");
      }}
    }}

    function loadPreset(key) {{
      switchMode('text');
      newsInput.value = presets[key];
      charCount.textContent = `${{presets[key].length}} chars`;
      analyzeNews();
    }}

    function clearText() {{
      newsInput.value = '';
      document.getElementById('urlInput').value = '';
      charCount.textContent = '0 chars';
      document.getElementById('emptyState').classList.remove('hidden');
      document.getElementById('resultContent').classList.add('hidden');
    }}

    function setRadialScore(percent, isFake) {{
      const circle = document.getElementById('radialCircle');
      const offset = 100 - percent;
      circle.style.strokeDashoffset = offset;
      circle.className = isFake ? "text-rose-500 transition-all duration-700 ease-out" : "text-emerald-500 transition-all duration-700 ease-out";
    }}

    async function analyzeNews() {{
      const isUrl = (currentMode === 'url');
      const text = newsInput.value.trim();
      const url = document.getElementById('urlInput').value.trim();

      if (isUrl && !url) return alert('Please enter a valid news article URL.');
      if (!isUrl && !text) return alert('Please enter some text to analyze.');

      lastAnalyzedText = isUrl ? url : text;
      const includeExplanation = document.getElementById('explainToggle').checked;

      document.getElementById('emptyState').classList.add('hidden');
      document.getElementById('resultContent').classList.add('hidden');
      document.getElementById('loadingState').classList.remove('hidden');

      try {{
        let endpoint = isUrl ? '/api/v1/predict/url' : '/api/v1/predict';
        let body = isUrl ? {{ url: url, include_explanation: includeExplanation }} : {{ text: text, include_explanation: includeExplanation }};

        const response = await fetch(endpoint, {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(body)
        }});

        if (!response.ok) {{
          const errData = await response.json().catch(() => ({{}}));
          throw new Error(errData.detail || `HTTP error ${{response.status}}`);
        }}

        const data = await response.json();
        lastResultPayload = data;

        const isFake = (data.prediction === 'FAKE');
        const verdictBadge = document.getElementById('verdictBadge');
        const resultCard = document.getElementById('resultCard');

        if (isFake) {{
          verdictBadge.innerHTML = '<span class="text-rose-500 flex items-center space-x-2"><i class="fa-solid fa-triangle-exclamation"></i><span>DISINFORMATION</span></span>';
          resultCard.className = "glass-panel rounded-2xl p-6 shadow-2xl relative min-h-[520px] flex flex-col justify-between glow-rose transition-all duration-300";
        }} else {{
          verdictBadge.innerHTML = '<span class="text-emerald-400 flex items-center space-x-2"><i class="fa-solid fa-circle-check"></i><span>VERIFIED REAL</span></span>';
          resultCard.className = "glass-panel rounded-2xl p-6 shadow-2xl relative min-h-[520px] flex flex-col justify-between glow-emerald transition-all duration-300";
        }}

        const titleDisplay = document.getElementById('articleTitleDisplay');
        if (data.article_title) {{
          titleDisplay.textContent = data.article_title;
          titleDisplay.classList.remove('hidden');
        }} else {{
          titleDisplay.classList.add('hidden');
        }}

        const confPct = Math.round(data.confidence * 100);
        document.getElementById('confidenceVal').textContent = `${{confPct}}%`;
        setRadialScore(confPct, isFake);

        const fakePct = (data.probabilities.FAKE * 100).toFixed(1);
        const realPct = (data.probabilities.REAL * 100).toFixed(1);

        document.getElementById('fakeProbVal').textContent = `${{fakePct}}%`;
        document.getElementById('realProbVal').textContent = `${{realPct}}%`;
        document.getElementById('fakeBar').style.width = `${{fakePct}}%`;
        document.getElementById('realBar').style.width = `${{realPct}}%`;

        if (data.ai_detection) {{
          const aiPct = (data.ai_detection.ai_generated_probability * 100).toFixed(0);
          document.getElementById('aiScoreBadge').textContent = `${{aiPct}}%`;
          document.getElementById('aiVerdictBadge').textContent = data.ai_detection.verdict.replace("LIKELY ", "");
        }}

        if (data.domain_credibility) {{
          document.getElementById('domainTrustBadge').textContent = `${{data.domain_credibility.trust_score}}/100`;
          document.getElementById('domainBiasBadge').textContent = data.domain_credibility.factual_reporting;
        }} else {{
          document.getElementById('domainTrustBadge').textContent = `Direct Text`;
          document.getElementById('domainBiasBadge').textContent = `Unrated`;
        }}

        if (data.sensationalism) {{
          document.getElementById('sensScoreBadge').textContent = `${{data.sensationalism.overall_sensationalism_score}}%`;
          document.getElementById('sensRiskBadge').textContent = data.sensationalism.risk_level.replace(" SENSATIONALISM", "");
        }}
        if (data.virality) {{
          document.getElementById('viralityScore').textContent = `${{data.virality.virality_potential_score}}%`;
          document.getElementById('harmRiskBadge').textContent = data.virality.harm_threat_level.split(" ")[0];
        }}

        const factSec = document.getElementById('factCheckSection');
        if (data.fact_check && data.fact_check.fact_checks_found && data.fact_check.results.length > 0) {{
          const fc = data.fact_check.results[0];
          factSec.classList.remove('hidden');
          document.getElementById('factCheckRating').textContent = fc.rating;
          document.getElementById('factCheckSummary').textContent = `${{fc.summary || fc.claim}}`;
          document.getElementById('factCheckReviewer').textContent = fc.reviewer;
          document.getElementById('factCheckLink').href = fc.review_url;
        }} else {{
          factSec.classList.add('hidden');
        }}

        const explainSection = document.getElementById('explainSection');
        if (data.explanation && data.explanation.highlighted_html) {{
          explainSection.classList.remove('hidden');
          document.getElementById('highlightedContainer').innerHTML = data.explanation.highlighted_html;
        }} else {{
          explainSection.classList.add('hidden');
        }}

        document.getElementById('latencyBadge').innerHTML = `<i class="fa-regular fa-clock mr-1"></i>${{data.latency_ms}} ms`;
        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('resultContent').classList.remove('hidden');

      }} catch (err) {{
        alert('Analysis error: ' + err.message);
        document.getElementById('loadingState').classList.add('hidden');
        document.getElementById('emptyState').classList.remove('hidden');
      }}
    }}

    async function generateDebunkClick() {{
      const text = document.getElementById('debunkInput').value.trim();
      if (!text) return alert("Please enter claim to debunk.");
      try {{
        const res = await fetch('/api/v1/generate-debunk', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ text: text }})
        }});
        const data = await res.json();
        document.getElementById('debunkText').textContent = data.truth_sandwich_rebuttal;
        document.getElementById('debunkOutputContainer').classList.remove('hidden');
      }} catch (e) {{
        alert("Failed to generate rebuttal: " + e.message);
      }}
    }}

    function copyDebunk() {{
      const text = document.getElementById('debunkText').textContent;
      navigator.clipboard.writeText(text);
      alert("Rebuttal copied to clipboard!");
    }}

    async function loadHistory() {{
      const container = document.getElementById('historyTableContainer');
      container.innerHTML = "Fetching database records...";
      try {{
        const res = await fetch('/api/v1/history?limit=15');
        const data = await res.json();
        if (!data.records.length) {{
          container.innerHTML = "No verification records in database yet.";
          return;
        }}
        container.innerHTML = data.records.map(r => `
          <div class="p-2.5 rounded-lg bg-obsidian-900 border border-slate-800 flex justify-between items-center">
            <div>
              <span class="${{r.prediction === 'FAKE' ? 'text-rose-400' : 'text-emerald-400'}} font-bold">[${{r.prediction}}]</span>
              <span class="text-slate-200 ml-1.5">${{r.text.substring(0, 45)}}...</span>
            </div>
            <span class="text-[10px] text-slate-500">${{r.timestamp.split(' ')[0]}}</span>
          </div>
        `).join('');
      }} catch (e) {{
        container.innerHTML = "Failed to load audit history: " + e.message;
      }}
    }}

    async function downloadReport() {{
      if (!lastAnalyzedText) return alert('Please analyze text first.');
      try {{
        const response = await fetch('/api/v1/report', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ text: lastAnalyzedText }})
        }});
        const data = await response.json();
        const blob = new Blob([data.report_markdown], {{ type: 'text/markdown' }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Disinformation-Report-${{data.report_id}}.md`;
        a.click();
        URL.revokeObjectURL(url);
      }} catch (e) {{
        alert('Failed to generate report: ' + e.message);
      }}
    }}

    function copyJsonResult() {{
      if (!lastResultPayload) return;
      navigator.clipboard.writeText(JSON.stringify(lastResultPayload, null, 2));
      alert("Analysis JSON copied to clipboard!");
    }}
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.api:app", host="0.0.0.0", port=port, reload=True)

const API_BASE = "http://localhost:8000";

document.getElementById("verifyBtn").addEventListener("click", () => {
  const text = document.getElementById("textInput").value.trim();
  if (!text) return alert("Please enter text to verify.");
  analyzeContent({ text: text, include_explanation: true });
});

document.getElementById("verifyUrlBtn").addEventListener("click", () => {
  if (!chrome.tabs) {
    return alert("Chrome tabs API unavailable. Please paste a URL or text.");
  }
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs && tabs[0] && tabs[0].url) {
      const currentUrl = tabs[0].url;
      analyzeUrl(currentUrl);
    }
  });
});

async function analyzeContent(payload) {
  setLoading(true);
  try {
    const res = await fetch(`${API_BASE}/api/v1/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    alert("Analysis failed. Make sure the API server is running on http://localhost:8000.\n" + err.message);
  } finally {
    setLoading(false);
  }
}

async function analyzeUrl(url) {
  setLoading(true);
  try {
    const res = await fetch(`${API_BASE}/api/v1/predict/url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url })
    });
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    alert("URL analysis failed: " + err.message);
  } finally {
    setLoading(false);
  }
}

function renderResult(data) {
  document.getElementById("resultCard").style.display = "block";
  const verdictEl = document.getElementById("verdictText");
  const isFake = data.prediction === "FAKE";

  verdictEl.textContent = isFake ? "⚠️ FAKE / DECEPTIVE" : "✅ VERIFIED REAL";
  verdictEl.className = isFake ? "badge-fake" : "badge-real";

  const confPct = (data.confidence * 100).toFixed(1);
  document.getElementById("confText").textContent = `${confPct}% Confidence`;

  const bar = document.getElementById("progressBar");
  bar.style.width = `${confPct}%`;
  bar.className = isFake ? "bar-fill bar-fake" : "bar-fill bar-real";

  // Sensationalism info if provided
  if (data.sensationalism) {
    document.getElementById("riskText").textContent = data.sensationalism.risk_level || "--";
    document.getElementById("clickbaitText").textContent = `${data.sensationalism.clickbait_score || 0}%`;
  }

  // Token attribution
  const hl = document.getElementById("highlightContainer");
  if (data.explanation && data.explanation.highlighted_html) {
    hl.style.display = "block";
    hl.innerHTML = data.explanation.highlighted_html;
  } else {
    hl.style.display = "none";
  }
}

function setLoading(isLoading) {
  document.getElementById("loader").style.display = isLoading ? "block" : "none";
  if (isLoading) {
    document.getElementById("resultCard").style.display = "none";
  }
}

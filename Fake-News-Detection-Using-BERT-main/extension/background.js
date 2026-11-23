// Background Service Worker for Manifest V3 Extension
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "verifyClaimMenu",
    title: "Verify with Fake News Intelligence",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "verifyClaimMenu" && info.selectionText) {
    const textToVerify = info.selectionText.trim();
    try {
      const response = await fetch("http://localhost:8000/api/v1/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToVerify, include_explanation: false })
      });
      const data = await response.json();
      const verdict = data.prediction === "FAKE" ? "⚠️ POTENTIAL FAKE" : "✅ LIKELY REAL";
      const conf = (data.confidence * 100).toFixed(0);

      // Inject notification or alert into active tab
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: (verdict, conf, text) => {
          alert(`[Fake News Intelligence 2.0]\nVerdict: ${verdict} (${conf}% Confidence)\n\nText: "${text.substring(0, 100)}..."`);
        },
        args: [verdict, conf, textToVerify]
      });
    } catch (e) {
      console.error("Context menu verification error:", e);
    }
  }
});

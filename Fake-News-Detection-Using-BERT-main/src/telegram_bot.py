"""
Telegram Fact-Checking Bot Service.
Runs standalone long-polling using pure HTTP requests (zero extra dependencies).
Forward any message, tweet, or link to receive an immediate disinformation assessment.
Usage: python src/telegram_bot.py --token YOUR_TELEGRAM_BOT_TOKEN
"""
import os
import sys
import time
import logging
import argparse
import requests
from typing import Dict, Any

from src.predict import FakeNewsPredictor
from src.ai_detector import AITextDetector
from src.sensationalism import SensationalismAnalyzer
from src.factcheck import FactCheckEngine
from src.scraper import extract_article_from_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("telegram_bot")


class TelegramDisinfoBot:
    """
    Lightweight Telegram Bot service communicating with Telegram Bot API.
    """
    def __init__(self, token: str):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.predictor = FakeNewsPredictor()
        self.ai_detector = AITextDetector()
        self.sensationalism = SensationalismAnalyzer()
        self.fact_check = FactCheckEngine()

    def send_message(self, chat_id: int, text: str, parse_mode: str = "HTML") -> bool:
        url = f"{self.base_url}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        try:
            r = requests.post(url, json=payload, timeout=10)
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def handle_update(self, update: Dict[str, Any]):
        msg = update.get("message", {})
        chat_id = msg.get("chat", {}).get("id")
        text = msg.get("text", "").strip()

        if not chat_id or not text:
            return

        # Commands
        if text.startswith("/start") or text.startswith("/help"):
            welcome = (
                "🛡️ <b>Welcome to Fake News Intelligence 2.0</b>\n\n"
                "I am your automated disinformation verification assistant.\n\n"
                "<b>How to use:</b>\n"
                "• Forward any suspicious tweet, message, or claim\n"
                "• Paste any news article URL (e.g. https://...)\n"
                "• Type any statement directly\n\n"
                "I will analyze the claim for disinformation patterns, synthetic AI generation, and existing fact-checks!"
            )
            self.send_message(chat_id, welcome)
            return

        # Check if message is a URL
        if text.startswith("http://") or text.startswith("https://"):
            self.send_message(chat_id, "⏳ <i>Scraping article and running intelligence analysis...</i>")
            try:
                article = extract_article_from_url(text)
                eval_text = f"{article['title']}. {article['text'][:1200]}"
                self._analyze_and_reply(chat_id, eval_text, source_url=text, title=article['title'])
            except Exception as e:
                self.send_message(chat_id, f"❌ Failed to parse article: {e}")
            return

        # Standard text message
        self._analyze_and_reply(chat_id, text)

    def _analyze_and_reply(self, chat_id: int, text: str, source_url: str = "", title: str = ""):
        # Run models
        pred = self.predictor.predict(text, include_explanation=True)
        ai_res = self.ai_detector.detect(text)
        sens_res = self.sensationalism.analyze(text)
        fc_res = self.fact_check.check_claim(text)

        is_fake = (pred["prediction"] == "FAKE")
        verdict_badge = "⚠️ <b>DISINFORMATION / FAKE</b>" if is_fake else "✅ <b>AUTHENTIC / REAL</b>"
        conf = int(pred["confidence"] * 100)
        ai_prob = int(ai_res["ai_generated_probability"] * 100)

        # Fact check summary
        fc_text = ""
        if fc_res.get("fact_checks_found"):
            top = fc_res["results"][0]
            fc_text = f"\n\n🔍 <b>Recorded Fact-Check:</b>\n• Rating: <code>{top['rating']}</code>\n• Source: {top['reviewer']}\n• <a href='{top['review_url']}'>Read full review</a>"

        reply = (
            f"🛡️ <b>Disinformation Assessment</b>\n\n"
            f"<b>Verdict:</b> {verdict_badge}\n"
            f"<b>Model Confidence:</b> {conf}%\n"
            f"<b>Synthetic AI Score:</b> {ai_prob}% ({ai_res['verdict']})\n"
            f"<b>Sensationalism:</b> {sens_res['overall_sensationalism_score']}/100 ({sens_res['risk_level']})"
            f"{fc_text}\n\n"
            f"<i>Powered by Fake News Intelligence Platform 2.0</i>"
        )
        self.send_message(chat_id, reply)

    def start_polling(self):
        logger.info("Starting Telegram Bot long-polling...")
        offset = 0
        while True:
            try:
                url = f"{self.base_url}/getUpdates?offset={offset}&timeout=25"
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    for item in data.get("result", []):
                        offset = item["update_id"] + 1
                        self.handle_update(item)
                elif r.status_code == 401:
                    logger.error("Invalid Telegram Bot Token. Please provide a valid token.")
                    break
            except Exception as e:
                logger.warning(f"Polling warning: {e}")
                time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="Run Telegram Fact-Checking Bot")
    parser.add_argument("--token", type=str, default=os.getenv("TELEGRAM_BOT_TOKEN", ""), help="Telegram Bot Token from @BotFather")
    args = parser.parse_args()

    if not args.token:
        print("\n[!] Error: No Telegram Bot Token provided.")
        print("Obtain a bot token from @BotFather on Telegram and run:")
        print("python src/telegram_bot.py --token YOUR_TOKEN\n")
        sys.exit(1)

    bot = TelegramDisinfoBot(token=args.token)
    bot.start_polling()


if __name__ == "__main__":
    main()

# telegram_bot.py
# Install: uv pip install python-telegram-bot apscheduler
# Run: uv run python telegram_bot.py

import asyncio
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from inference import run_inference
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update

load_dotenv()

BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "")

if not BOT_TOKEN:
    print("TELEGRAM_BOT_TOKEN not set in .env")

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    logging.warning("apscheduler not installed. Run: pip install apscheduler")


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_prediction_message(result: dict) -> str:
    """Format inference result as Markdown V2 safe message."""
    # Escape special chars for MarkdownV2
    def esc(s): return str(s).replace('.', '\\.').replace('-', '\\-').replace('+', '\\+').replace('(', '\\(').replace(')', '\\)')

    h  = result['home_team']
    a  = result['away_team']
    ph = result['home_win_prob']
    pd = result['draw_prob']
    pa = result['away_win_prob']
    pick = result.get('final_pick', 'N/A')
    conf = result.get('confidence', 'N/A')
    edge = result.get('no_vig_edge')
    bet  = result.get('best_bet')
    kf   = result.get('kelly_fraction')
    btts = result.get('btts_yes')
    o25  = result.get('over_25')

    conf_emoji = {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🔴"}.get(
        conf.split()[0], "⚪")

    lines = [
        f"⚽ *World Cup Signal*",
        f"🏠 *{h}* vs ✈️ *{a}*",
        "",
        "📊 *Probabilities \\(No\\-Vig\\)*",
        f"  Home Win: `{esc(f'{ph:.1%}')}`",
        f"  Draw:     `{esc(f'{pd:.1%}')}`",
        f"  Away Win: `{esc(f'{pa:.1%}')}`",
        "",
        f"🎯 *Model Pick:* {pick}",
        f"{conf_emoji} *Confidence:* {conf.split('—')[0].strip()}",
    ]

    if btts is not None:
        lines += ["", f"⚡ BTTS Yes: `{esc(f'{btts:.1%}')}` \\| Over 2\\.5: `{esc(f'{o25:.1%}')}`"]

    if bet and bet != "NO BET" and edge and kf:
        lines += [
            "",
            f"💰 *Best Bet: {bet}*",
            f"  Edge: `{esc(f'+{edge:.1%}')}`",
            f"  Kelly Stake: `{esc(f'{kf:.1%}')}` of bankroll",
        ]
    else:
        lines += ["", "🚫 *No Value Bet* — edge below 2\\.5% threshold"]

    lines.append(f"\n`V6\\.2 \\| \\#FIFA \\#WorldCup2026`")
    return "\n".join(lines)


async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/predict <HomeTeam> <AwayTeam> command handler."""
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: /predict <HomeTeam> <AwayTeam>\nExample: /predict Brazil Germany"
        )
        return

    home = " ".join(args[:-1]) if len(args) > 2 else args[0]
    away = args[-1]

    await update.message.reply_text(f"🔄 Running prediction for {home} vs {away}...")

    try:
        result = run_inference(home_team=home, away_team=away)
        if result.get("regime_filtered"):
            await update.message.reply_text(
                f"⚠️ Regime filter active: Glicko gap too large. No prediction issued."
            )
            return
        msg = format_prediction_message(result)
        await update.message.reply_text(msg, parse_mode="MarkdownV2")
    except Exception as e:
        await update.message.reply_text(f"❌ Prediction failed: {str(e)}")


async def send_daily_signals():
    """Placeholder for scheduled signals"""
    pass

def run_bot_with_scheduler():
    """Start bot with command handler + daily scheduler."""
    if not BOT_TOKEN:
        logger.error("No BOT_TOKEN. Cannot start telegram bot.")
        return
        
    app_bot = Application.builder().token(BOT_TOKEN).build()
    app_bot.add_handler(CommandHandler("predict", predict_command))
    app_bot.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text(
        "V6.2 WorldCup Quant Bot active.\nCommands:\n/predict <Home> <Away>"
    )))

    if SCHEDULER_AVAILABLE:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
        scheduler.add_job(send_daily_signals, 'cron', hour=10, minute=0)
        scheduler.start()

    print("✅ Telegram bot running. Send /predict Brazil Germany to test.")
    app_bot.run_polling()

if __name__ == "__main__":
    run_bot_with_scheduler()

# telegram_bot.py
# Install: pip install python-telegram-bot apscheduler
# Run: python telegram_bot.py

import asyncio
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "")  # e.g. "@yourchannelname" or "-100xxxx"

if not BOT_TOKEN:
    print("TELEGRAM_BOT_TOKEN not set in .env")

from telegram import Bot
from telegram.constants import ParseMode

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    logging.warning("apscheduler not installed. Run: pip install apscheduler")


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_prediction_message(match: dict) -> str:
    """Format a prediction dict into a Telegram-ready message."""
    home  = match.get('home_team', 'N/A')
    away  = match.get('away_team', 'N/A')
    ph    = match.get('home_win_prob', 0)
    pd_   = match.get('draw_prob', 0)
    pa    = match.get('away_win_prob', 0)
    conf  = match.get('confidence', 'MODERATE')
    edge  = match.get('no_vig_edge')
    best  = match.get('best_bet')
    kelly = match.get('kelly_fraction')
    btts  = match.get('btts_yes')
    o25   = match.get('over_25')

    conf_emoji = {"HIGH CONFIDENCE": "??", "MODERATE CONFIDENCE": "??",
                  "LOW CONFIDENCE": "??"}.get(conf, "?")

    msg = (
        f"? *World Cup Signal* — {datetime.utcnow().strftime('%d %b %Y')}\n"
        f"??????????????????\n"
        f"?? *{home}* vs ?? *{away}*\n\n"
        f"?? *1X2 Probabilities (No-Vig)*\n"
        f"  Home Win: {ph:.1%}\n"
        f"  Draw:     {pd_:.1%}\n"
        f"  Away Win: {pa:.1%}\n\n"
    )

    if btts is not None:
        msg += f"? BTTS Yes: {btts:.1%} | Over 2.5: {o25:.1%}\n\n"

    msg += f"{conf_emoji} *Confidence:* {conf}\n"

    if best and edge and kelly:
        msg += (
            f"\n?? *Best Bet: {best}*\n"
            f"  Edge vs Market: +{edge:.1%}\n"
            f"  Kelly Stake: {kelly:.1%} of bankroll\n"
        )
    else:
        msg += "\n?? *No Value Bet* — edge below 2.5% threshold\n"

    msg += f"\nModel: V6.2 | #FIFA #WorldCup2026"
    return msg


async def send_prediction(match: dict):
    """Send a single prediction to the Telegram channel."""
    if not BOT_TOKEN:
        logger.info("Bot token not set, skipping send: " + format_prediction_message(match))
        return
    bot = Bot(token=BOT_TOKEN)
    message = format_prediction_message(match)
    try:
        await bot.send_message(
            chat_id=CHANNEL_ID,
            text=message,
            parse_mode=ParseMode.MARKDOWN
        )
        logger.info(f"Signal sent: {match.get('home_team')} vs {match.get('away_team')}")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")


async def send_daily_signals():
    """
    Fetch today's matches, run predictions, send signals.
    Called by scheduler or manually.
    """
    try:
        # For testing, we mock get_todays_fixtures
        # from main import run_prediction_pipeline, get_todays_fixtures
        # fixtures = get_todays_fixtures()
        
        fixtures = []
        if not fixtures:
            logger.info("No fixtures today.")
            return

        for fixture in fixtures:
            # result = run_prediction_pipeline(
            #     home_team=fixture['home_team'],
            #     away_team=fixture['away_team'],
            #     venue_factor=fixture.get('venue_factor', 0.3),
            #     stage=fixture.get('stage', 'group')
            # )
            # result.update(fixture)
            # await send_prediction(result)
            await asyncio.sleep(2)   # rate limit: 2s between messages

    except Exception as e:
        logger.error(f"Daily signal job failed: {e}")


def run_bot_with_scheduler():
    """Start async scheduler to send signals daily at 10:00 AM IST."""
    if not SCHEDULER_AVAILABLE:
        logger.warning("Running without scheduler. Call send_daily_signals() manually.")
        asyncio.run(send_daily_signals())
        return

    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(send_daily_signals, 'cron', hour=10, minute=0)
    scheduler.start()
    logger.info("Telegram bot started. Signals will fire at 10:00 AM IST daily.")

    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


if __name__ == "__main__":
    run_bot_with_scheduler()

import asyncio
import websockets
import json
import logging
from typing import Callable, Dict

logger = logging.getLogger(__name__)

class BetfairStreamClient:
    """
    Ultra-low-latency WebSocket client for Betfair Stream API.
    Replaces periodic REST API polling for V9 live execution.
    Designed to mirror the performance profile of sockudo-ws (Rust).
    """
    def __init__(self, endpoint: str, app_key: str, session_token: str):
        self.endpoint = endpoint
        self.app_key = app_key
        self.session_token = session_token
        self.callbacks: Dict[str, Callable] = {}
        self.connection = None
        self.is_running = False
        
    def register_callback(self, market_id: str, callback: Callable):
        """Register a fast execution callback when a specific market updates."""
        self.callbacks[market_id] = callback
        
    async def _authenticate(self):
        """Send authentication message upon connection."""
        auth_msg = {
            "op": "authentication",
            "appKey": self.app_key,
            "session": self.session_token
        }
        await self.connection.send(json.dumps(auth_msg))
        
    async def subscribe_markets(self, market_ids: list):
        """Subscribe to tick-by-tick order book updates for specific markets."""
        sub_msg = {
            "op": "marketSubscription",
            "marketFilter": {"marketIds": market_ids},
            "marketDataFilter": {
                "ladderLevels": 3,
                "fields": ["EX_BEST_OFFERS", "EX_TRADED", "EX_MARKET_DEF"]
            }
        }
        if self.connection:
            await self.connection.send(json.dumps(sub_msg))
            logger.info(f"Subscribed to {len(market_ids)} markets.")
            
    async def _handle_message(self, message: str):
        """Parse incoming ticks and route to execution callbacks."""
        data = json.loads(message)
        
        # Route Market Change Messages (MCM) to the execution engine
        if data.get("op") == "mcm" and "mc" in data:
            for market_change in data["mc"]:
                market_id = market_change.get("id")
                if market_id in self.callbacks:
                    # Fire execution callback immediately
                    # In a production system, this bypasses the GIL via C-extensions if possible
                    self.callbacks[market_id](market_change)

    async def connect_and_listen(self):
        """Main async loop for WebSocket connection and message handling."""
        self.is_running = True
        reconnect_delay = 1
        
        while self.is_running:
            try:
                async with websockets.connect(self.endpoint, ping_interval=10, ping_timeout=10) as ws:
                    self.connection = ws
                    logger.info("Connected to Betfair Stream API.")
                    await self._authenticate()
                    
                    # Reset reconnection delay on successful connection
                    reconnect_delay = 1 
                    
                    async for message in ws:
                        await self._handle_message(message)
                        
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed. Reconnecting...")
            except Exception as e:
                logger.error(f"WebSocket Error: {e}")
                
            # Exponential backoff for reconnections
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 60)
            
    def stop(self):
        """Gracefully terminate the WebSocket listener."""
        self.is_running = False
        if self.connection:
            asyncio.create_task(self.connection.close())

"""
Simple Databento Live Futures Data Client for ImpViz
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import os
from dotenv import load_dotenv

try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class LiveFuturesDataClient:
    """Simple live futures data client using Databento API"""

    def __init__(self):
        # Use hardcoded API key for now
        self.api_key = os.getenv('DATABENTO_API_KEY')

        if not DATABENTO_AVAILABLE:
            logger.error("Databento package not installed. Please run: pip install databento")
            raise ImportError("Databento package not installed.")

        # Initialize Databento clients
        self.historical_client = None
        self.live_client = None
        logger.info("Databento client initialized")

        # Data storage for indicators
        self.price_data: Dict[str, deque] = {}
        self.indicators: Dict[str, Dict[str, float]] = {}
        self.subscribers: Dict[str, List[Callable]] = {}

        # Supported timeframes (in seconds)
        self.timeframes = {
            '1m': 60,
            '5m': 300,
            '10m': 600,
            '15m': 900,
            '30m': 1800,
            '1h': 3600
        }

        # Connection status
        self.is_connected = False
        self.active_subscriptions: List[str] = []
        self.should_stop = False  # Flag to stop background threads

    def connect(self) -> bool:
        """Initialize Databento clients"""
        try:
            self.should_stop = False  # Reset stop flag on connect
            self.historical_client = db.Historical(key=self.api_key)
            self.live_client = db.Live(key=self.api_key)
            self.is_connected = True
            logger.info("Databento clients connected successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Databento: {str(e)}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from Databento clients"""
        try:
            # Set flag to stop all background threads
            self.should_stop = True
            
            # Stop the realtime client first if it exists
            if hasattr(self, 'realtime_client') and self.realtime_client:
                try:
                    logger.info("Stopping realtime client...")
                    self.realtime_client.stop()
                    # Force close the connection
                    if hasattr(self.realtime_client, '_session'):
                        self.realtime_client._session.close()
                except Exception as e:
                    logger.warning(f"Error stopping realtime client: {e}")
                self.realtime_client = None
            
            # Then stop the main live client
            if self.live_client:
                try:
                    logger.info("Stopping live client...")
                    self.live_client.stop()
                    # Force close the connection
                    if hasattr(self.live_client, '_session'):
                        self.live_client._session.close()
                except Exception as e:
                    logger.warning(f"Error stopping live client: {e}")
                self.live_client = None
            
            # Clear historical client
            self.historical_client = None
            
            # Clear all data and state
            self.is_connected = False
            self.active_subscriptions.clear()
            self.price_data.clear()
            self.indicators.clear()
            self.subscribers.clear()
            
            # Wait a moment for threads to stop
            import time
            time.sleep(0.5)
            
            logger.info("Successfully disconnected from all Databento clients")
        except Exception as e:
            logger.error(f"Error during disconnect: {str(e)}")

    def get_available_symbols(self, dataset: str = "GLBX.MDP3") -> List[str]:
        """Get list of available futures symbols from mapping.json"""
        try:
            import json
            import os
            
            # Load symbols from mapping.json
            mapping_file_path = os.path.join(os.path.dirname(__file__), '..', 'mapping.json')
            symbols = []
            
            try:
                with open(mapping_file_path, 'r') as f:
                    mapping_data = json.load(f)
                    market_mappings = mapping_data.get('market_mappings', {})
                    # Extract all data_symbols
                    for market, info in market_mappings.items():
                        if 'data_symbol' in info:
                            symbols.append(info['data_symbol'])
                logger.info(f"Loaded {len(symbols)} symbols from mapping.json")
                return symbols
            except Exception as e:
                logger.warning(f"Could not load symbols from mapping.json: {e}, using fallback")
                # Fallback to default symbols
                return ["NQ.c.0"]
                
        except Exception as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            return []

    def subscribe_to_symbol(self, symbol: str, timeframe: str, callback: Optional[Callable] = None):
        """Subscribe to live data with historical context"""
        if not self.is_connected:
            raise Exception("Not connected to Databento. Call connect() first.")

        if timeframe not in self.timeframes:
            raise Exception(f"Unsupported timeframe: {timeframe}")

        subscription_key = f"{symbol}_{timeframe}"

        try:
            # Initialize data storage
            if subscription_key not in self.price_data:
                self.price_data[subscription_key] = deque(maxlen=500)
                self.indicators[subscription_key] = {}
                self.subscribers[subscription_key] = []

            if callback:
                self.subscribers[subscription_key].append(callback)

            logger.info(f"Setting up {symbol} {timeframe} with correct API split...")
            
            # Load historical + today's data and start live streaming
            success = self._load_historical_data(subscription_key, symbol, timeframe)
            if not success:
                raise Exception("Failed to load data")
            
            self.active_subscriptions.append(subscription_key)
            logger.info(f"Successfully subscribed to {symbol} {timeframe} with live streaming active")

        except Exception as e:
            logger.error(f"Error subscribing to {symbol} {timeframe}: {str(e)}")
            raise e

    def unsubscribe_from_symbol(self, symbol: str, timeframe: str):
        """Unsubscribe from live data for a symbol and timeframe"""
        subscription_key = f"{symbol}_{timeframe}"

        try:
            if subscription_key in self.active_subscriptions:
                self.active_subscriptions.remove(subscription_key)

            if subscription_key in self.price_data:
                del self.price_data[subscription_key]
            if subscription_key in self.indicators:
                del self.indicators[subscription_key]
            if subscription_key in self.subscribers:
                del self.subscribers[subscription_key]

            logger.info(f"Unsubscribed from {symbol} {timeframe}")

        except Exception as e:
            logger.error(f"Error unsubscribing from {symbol} {timeframe}: {str(e)}")

    def _load_historical_data(self, subscription_key: str, symbol: str, timeframe: str) -> bool:
        """Load historical data using the correct API split approach"""
        try:
            logger.info(f"Loading data for {symbol} {timeframe} using Historical + Live API split...")
            
            # Key insight: Live API replay only works for last 24 hours
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            current_time = datetime.now()
            
            # Calculate how much historical data we need (for SMA 20 + buffer)
            if timeframe == '1m':
                days_needed = 1   # 1 day for 1m bars
            elif timeframe == '5m':
                days_needed = 3   # 2 days for 5m bars
            elif timeframe == '10m':
                days_needed = 3   # 2 days for 10m bars
            elif timeframe == '15m':
                days_needed = 3   # 3 days for 15m bars
            elif timeframe == '30m':
                days_needed = 3   # 3 days for 30m bars
            elif timeframe == '1h':
                days_needed = 10  # 10 days for 1h bars (need more 1m bars to aggregate properly)
            else:
                days_needed = 3   # Default
            
            historical_start = today_start - timedelta(days=days_needed)
            
            logger.info(f"Data loading plan:")
            logger.info(f"  Historical API: {historical_start} to {today_start}")
            logger.info(f"  Live API replay: {today_start} to {current_time} + live stream")
            
            all_bars = []
            
            # STEP 1: Get historical data (older than today)
            if (today_start - historical_start).days > 0:
                logger.info(f"📚 Getting historical data from Historical API...")
                historical_bars = self._get_historical_bars(symbol, timeframe, historical_start, today_start)
                all_bars.extend(historical_bars)
                logger.info(f"   Got {len(historical_bars)} historical bars")
            
            # STEP 2: Get today's data + live stream from Live API
            logger.info(f"📡 Getting today's data + live stream from Live API...")
            todays_bars = self._get_todays_data_and_start_live(symbol, timeframe, today_start)
            all_bars.extend(todays_bars)
            logger.info(f"   Got {len(todays_bars)} bars from today's replay")
            
            # STEP 3: Store all bars and calculate indicators
            logger.info(f"💾 Storing {len(all_bars)} total bars...")
            for bar in all_bars:
                self.price_data[subscription_key].append(bar)
            
            # Calculate initial indicators
            self._calculate_indicators(subscription_key)
            self._notify_subscribers(subscription_key)
            
            logger.info(f"✅ Successfully loaded {len(all_bars)} bars for {symbol} {timeframe}")
            return len(all_bars) >= 20
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return False
    
    def _get_historical_bars(self, symbol: str, timeframe: str, start_time: datetime, end_time: datetime) -> list:
        """Get historical bars from Historical API (for data older than today)"""
        try:
            # Always use 1m schema for consistency, then aggregate
            data = self.historical_client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=[symbol],
                stype_in="continuous",
                schema='ohlcv-1m',
                start=start_time,
                end=end_time
            )
            
            bars = []
            for record in data:
                if hasattr(record, 'open'):
                    bar = {
                        'timestamp': datetime.fromtimestamp(record.ts_event / 1e9),
                        'open': float(record.open) / 1e9,
                        'high': float(record.high) / 1e9,
                        'low': float(record.low) / 1e9,
                        'close': float(record.close) / 1e9,
                        'volume': getattr(record, 'volume', 1)
                    }
                    bars.append(bar)
            
            # Aggregate for all non-1m timeframes
            if timeframe in ['5m', '10m', '15m', '30m', '1h']:
                bars = self._aggregate_bars(bars, timeframe)
            
            return bars
            
        except Exception as e:
            logger.error(f"Historical API failed: {str(e)}")
            return []
    
    def _get_todays_data_and_start_live(self, symbol: str, timeframe: str, today_start: datetime) -> list:
        """Get today's data via OHLCV replay, then switch to TRADES for real-time"""
        try:
            logger.info(f"📊 Step 1: Getting today's OHLCV data via replay...")
            
            # First get today's OHLCV data via replay
            todays_bars = []
            
            self.live_client.subscribe(
                dataset="GLBX.MDP3",
                symbols=[symbol],
                schema="ohlcv-1m",  # OHLCV for historical bars
                stype_in="continuous",
                start=today_start
            )
            
            current_time = datetime.now()
            count = 0
            
            import time
            collection_start = time.time()
            
            # Collect today's OHLCV replay data
            for msg in self.live_client:
                if hasattr(msg, 'open') and hasattr(msg, 'ts_event'):
                    timestamp = datetime.fromtimestamp(msg.ts_event / 1e9)
                    
                    bar = {
                        'timestamp': timestamp,
                        'open': float(msg.open) / 1e9,
                        'high': float(msg.high) / 1e9,
                        'low': float(msg.low) / 1e9,
                        'close': float(msg.close) / 1e9,
                        'volume': getattr(msg, 'volume', 1)
                    }
                    todays_bars.append(bar)
                    count += 1
                    
                    # Check if caught up to recent time
                    time_diff = (current_time - timestamp).total_seconds() / 60
                    if time_diff < 2:  # Within 2 minutes = replay complete
                        logger.info(f"🏁 OHLCV replay complete at {timestamp}")
                        break
                    
                    # Safety timeout
                    if (time.time() - collection_start) > 60:
                        logger.warning(f"⏰ OHLCV replay timeout - got {count} bars")
                        break
            
            # Stop the OHLCV client
            self.live_client.stop()
            
            # Aggregate if needed (Live API gives us 1m data, so we need to aggregate for all timeframes > 1m)
            if timeframe in ['5m', '10m', '15m', '30m', '1h']:
                todays_bars = self._aggregate_bars(todays_bars, timeframe)
            
            logger.info(f"✅ Got {len(todays_bars)} bars from today's replay")
            
            # Step 2: Start REAL-TIME trades stream
            logger.info(f"🔴 Step 2: Starting REAL-TIME trades stream...")
            self._start_realtime_trades_stream(symbol)
            
            return todays_bars
            
        except Exception as e:
            logger.error(f"Today's data + live stream failed: {str(e)}")
            return []
    
    def _start_realtime_trades_stream(self, symbol: str):
        """Start a NEW live client specifically for real-time trades"""
        try:
            # Create a NEW live client for real-time trades
            import databento as db
            self.realtime_client = db.Live(key=self.api_key)
            
            logger.info(f"🔴 Starting REAL-TIME trades stream for {symbol}...")
            
            # Subscribe to LIVE TRADES ONLY (no start time = live data only)
            self.realtime_client.subscribe(
                dataset="GLBX.MDP3",
                symbols=[symbol],
                schema="trades",  # TRADES for tick-by-tick
                stype_in="continuous"
                # NO start time = only live trades from now
            )
            
            # Start background thread for processing trades
            import threading
            
            def realtime_trades_handler():
                try:
                    logger.info("📡 REAL-TIME trades handler started")
                    for msg in self.realtime_client:
                        # Check if we should stop
                        if self.should_stop:
                            logger.info("Stopping real-time trades handler due to disconnect")
                            break
                        if hasattr(msg, 'price'):  # Process each live trade
                            self._process_realtime_trade(msg)
                            
                except Exception as e:
                    if not self.should_stop:  # Only log error if not intentionally stopped
                        logger.error(f"Real-time trades handler error: {e}")
                finally:
                    logger.info("Real-time trades handler stopped")
            
            thread = threading.Thread(target=realtime_trades_handler, daemon=True)
            thread.start()
            
            logger.info("✅ REAL-TIME trades stream active - tick-by-tick updates enabled!")
            
        except Exception as e:
            logger.error(f"Failed to start real-time trades stream: {str(e)}")
    
    def _process_realtime_trade(self, msg):
        """Process each individual trade for real-time updates"""
        try:
            # Stop processing if disconnecting
            if self.should_stop:
                return
                
            price = float(msg.price) / 1e9
            timestamp = datetime.fromtimestamp(msg.ts_event / 1e9)
            size = getattr(msg, 'size', 1)
            
            # Update all active subscriptions with this trade
            for subscription_key in self.active_subscriptions:
                self._update_current_bar_with_trade(subscription_key, timestamp, price, size)
            
            # Log every few seconds (not every trade)
            import time
            current_time = time.time()
            if not hasattr(self, '_last_trade_log') or (current_time - self._last_trade_log) > 3:
                logger.info(f"🔴 LIVE TRADE: ${price:.2f} at {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
                self._last_trade_log = current_time

        except Exception as e:
            if not self.should_stop:  # Only log error if not intentionally stopped
                logger.error(f"Error processing real-time trade: {str(e)}")
    
    def _update_current_bar_with_trade(self, subscription_key: str, timestamp: datetime, price: float, size: int):
        """Update current bar with real-time trade"""
        try:
            bars = self.price_data[subscription_key]
            if not bars:
                return
            
            _, timeframe = subscription_key.split('_')
            
            # Get bar timestamp boundary
            if timeframe == '1m':
                bar_timestamp = timestamp.replace(second=0, microsecond=0)
            elif timeframe == '5m':
                bar_timestamp = timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)
            elif timeframe == '10m':
                bar_timestamp = timestamp.replace(minute=(timestamp.minute // 10) * 10, second=0, microsecond=0)
            elif timeframe == '15m':
                bar_timestamp = timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)
            elif timeframe == '30m':
                bar_timestamp = timestamp.replace(minute=(timestamp.minute // 30) * 30, second=0, microsecond=0)
            elif timeframe == '1h':
                bar_timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
            else:
                bar_timestamp = timestamp
            
            current_bar = bars[-1]
            
            if current_bar['timestamp'] == bar_timestamp:
                # Update current bar with this trade
                current_bar['high'] = max(current_bar['high'], price)
                current_bar['low'] = min(current_bar['low'], price)
                current_bar['close'] = price  # Latest trade price
                current_bar['volume'] += size
                
                # Recalculate indicators and notify UI immediately
                self._calculate_indicators(subscription_key)
                self._notify_subscribers(subscription_key)
                
            elif bar_timestamp > current_bar['timestamp']:
                # Create new bar
                new_bar = {
                    'timestamp': bar_timestamp,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': size
                }
                bars.append(new_bar)
                
                logger.info(f"🆕 NEW BAR: {bar_timestamp} ${price:.2f}")
                
                # Recalculate indicators and notify UI
                self._calculate_indicators(subscription_key)
                self._notify_subscribers(subscription_key)

        except Exception as e:
            logger.error(f"Error updating bar with trade: {str(e)}")
    
    def _start_live_streaming_thread(self):
        """Start background thread to handle ongoing live updates"""
        import threading
        
        def live_stream_handler():
            try:
                logger.info("📡 Live streaming handler started - switching to trades for tick-by-tick updates")
                
                # The live client is already connected from the replay
                # Now we need to subscribe to TRADES for real-time tick updates
                self._setup_realtime_trades_stream()
                
                # Continue processing messages from the live client
                for msg in self.live_client:
                    if hasattr(msg, 'price'):  # This is a live trade - REAL-TIME TICK
                        self._process_live_trade_tick(msg)
                    elif hasattr(msg, 'open'):  # This is a live OHLCV bar (less frequent)
                        self._process_live_ohlcv_bar(msg)
                        
            except Exception as e:
                logger.error(f"Live stream handler error: {e}")
        
        thread = threading.Thread(target=live_stream_handler, daemon=True)
        thread.start()
        logger.info("✅ Live streaming thread started with tick-by-tick updates")
    
    def _setup_realtime_trades_stream(self):
        """Set up real-time trades stream for tick-by-tick updates"""
        try:
            # Subscribe to live trades for real-time tick updates
            # This gives us every single trade as it happens
            for subscription_key in self.active_subscriptions:
                symbol = subscription_key.split('_')[0]
                
                logger.info(f"🔴 Adding real-time trades stream for {symbol}")
                
                # Add trades subscription to the existing live client
                # This will give us tick-by-tick price updates
                self.live_client.subscribe(
                    dataset="GLBX.MDP3",
                    symbols=[symbol],
                    schema="trades",  # TRADES for tick-by-tick updates
                    stype_in="continuous"
                    # No start time = live data only
                )
                
            logger.info("✅ Real-time trades stream active for tick-by-tick updates")
            
        except Exception as e:
            logger.error(f"Failed to setup real-time trades stream: {str(e)}")
    
    def _process_live_ohlcv_bar(self, msg):
        """Process live OHLCV bar updates"""
        try:
            timestamp = datetime.fromtimestamp(msg.ts_event / 1e9)
            
            bar = {
                'timestamp': timestamp,
                'open': float(msg.open) / 1e9,
                'high': float(msg.high) / 1e9,
                'low': float(msg.low) / 1e9,
                'close': float(msg.close) / 1e9,
                'volume': getattr(msg, 'volume', 1)
            }
            
            # Update all active subscriptions with this new bar
            for subscription_key in self.active_subscriptions:
                bars = self.price_data[subscription_key]
                
                # Replace or add the bar
                if bars and bars[-1]['timestamp'] == timestamp:
                    bars[-1] = bar  # Update existing bar
                else:
                    bars.append(bar)  # Add new bar
                
                # Recalculate indicators and notify UI
                self._calculate_indicators(subscription_key)
                self._notify_subscribers(subscription_key)
                
            logger.info(f"📊 Live bar update: {timestamp} ${bar['close']:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing live OHLCV bar: {str(e)}")

    def _process_live_trade_tick(self, msg):
        """Process individual trade ticks for real-time updates"""
        try:
            if not hasattr(msg, 'price'):
                return

            price = float(msg.price) / 1e9
            timestamp = datetime.fromtimestamp(msg.ts_event / 1e9)
            size = getattr(msg, 'size', 1)
            
            # Update all active subscriptions with this tick
            for subscription_key in self.active_subscriptions:
                self._update_current_bar_with_tick(subscription_key, timestamp, price, size)
            
            # Log occasional tick updates (not every single tick to avoid spam)
            import time
            current_time = time.time()
            if not hasattr(self, '_last_tick_log') or (current_time - self._last_tick_log) > 5:  # Log every 5 seconds
                logger.info(f"🔴 LIVE TICK: ${price:.2f} at {timestamp.strftime('%H:%M:%S')}")
                self._last_tick_log = current_time

        except Exception as e:
            logger.error(f"Error processing live trade tick: {str(e)}")

    def _update_current_bar_with_tick(self, subscription_key: str, timestamp: datetime, price: float, size: int):
        """Update current bar with real-time tick data"""
        try:
            bars = self.price_data[subscription_key]
            if not bars:
                return
            
            _, timeframe = subscription_key.split('_')
            
            # Get the current bar's timestamp boundary
            if timeframe == '1m':
                bar_timestamp = timestamp.replace(second=0, microsecond=0)
            elif timeframe == '5m':
                bar_timestamp = timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)
            elif timeframe == '15m':
                bar_timestamp = timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)
            elif timeframe == '30m':
                bar_timestamp = timestamp.replace(minute=(timestamp.minute // 30) * 30, second=0, microsecond=0)
            else:
                bar_timestamp = timestamp
            
            # Check if we need to create a new bar or update current bar
            current_bar = bars[-1]
            
            if current_bar['timestamp'] == bar_timestamp:
                # Update current bar with this tick
                current_bar['high'] = max(current_bar['high'], price)
                current_bar['low'] = min(current_bar['low'], price)
                current_bar['close'] = price  # Always update close with latest tick
                current_bar['volume'] += size
                
                # Recalculate indicators with updated bar
                self._calculate_indicators(subscription_key)
                self._notify_subscribers(subscription_key)
                
            elif bar_timestamp > current_bar['timestamp']:
                # New bar needed
                new_bar = {
                    'timestamp': bar_timestamp,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': size
                }
                bars.append(new_bar)
                
                logger.info(f"🆕 NEW BAR {subscription_key}: {bar_timestamp} ${price:.2f}")
                
                # Recalculate indicators with new bar
                self._calculate_indicators(subscription_key)
                self._notify_subscribers(subscription_key)

        except Exception as e:
            logger.error(f"Error updating bar with tick: {str(e)}")

    def _process_live_message(self, msg):
        """Process incoming live market data"""
        try:
            if not hasattr(msg, 'price'):
                return

            price = float(msg.price) / 1e9
            timestamp = datetime.fromtimestamp(msg.ts_event / 1e9)
            
            # Update all active subscriptions
            for subscription_key in self.active_subscriptions:
                self._update_price_bar(subscription_key, timestamp, price)
                self._calculate_indicators(subscription_key)
                self._notify_subscribers(subscription_key)

        except Exception as e:
            logger.error(f"Error processing live data: {str(e)}")

    def _update_price_bar(self, subscription_key: str, timestamp: datetime, price: float):
        """Update price bar with new data"""
        try:
            bars = self.price_data[subscription_key]
            _, timeframe = subscription_key.split('_')
            
            # Round timestamp to timeframe boundary
            if timeframe == '1m':
                bar_timestamp = timestamp.replace(second=0, microsecond=0)
            elif timeframe == '5m':
                bar_timestamp = timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)
            elif timeframe == '15m':
                bar_timestamp = timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)
            elif timeframe == '30m':
                bar_timestamp = timestamp.replace(minute=(timestamp.minute // 30) * 30, second=0, microsecond=0)
            else:
                bar_timestamp = timestamp

            # Create new bar or update existing
            if not bars or bars[-1]['timestamp'] != bar_timestamp:
                new_bar = {
                    'timestamp': bar_timestamp,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': 1
                }
                bars.append(new_bar)
                logger.info(f"New {timeframe} bar: {bar_timestamp} ${price:.2f}")
            else:
                # Update current bar
                current_bar = bars[-1]
                current_bar['high'] = max(current_bar['high'], price)
                current_bar['low'] = min(current_bar['low'], price)
                current_bar['close'] = price
                current_bar['volume'] += 1

        except Exception as e:
            logger.error(f"Error updating price bar: {str(e)}")

    def _calculate_indicators(self, subscription_key: str):
        """Calculate technical indicators"""
        try:
            bars = list(self.price_data[subscription_key])
            if len(bars) < 1:
                return

            bars_sorted = sorted(bars, key=lambda x: x['timestamp'])
            closes = [bar['close'] for bar in bars_sorted]

            indicators = {
                'current_price': closes[-1],
                'timestamp': bars_sorted[-1]['timestamp'],
                'bar_count': len(bars),
                'change': closes[-1] - closes[-2] if len(closes) > 1 else 0,
                'change_pct': ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) > 1 and closes[-2] != 0 else 0
            }

            # Calculate SMAs
            for period in [5, 20]:
                if len(closes) >= period:
                    sma = sum(closes[-period:]) / period
                    indicators[f'sma_{period}'] = sma
                    indicators[f'sma_{period}_full'] = True
                    indicators[f'distance_sma_{period}'] = ((indicators['current_price'] - sma) / sma) * 100
                else:
                    indicators[f'sma_{period}'] = 0
                    indicators[f'sma_{period}_full'] = False
                    indicators[f'distance_sma_{period}'] = 0

            # Calculate Williams %R (14 period)
            williams_period = 14
            if len(bars_sorted) >= williams_period:
                # Get last 14 bars for high/low calculation
                recent_bars = bars_sorted[-williams_period:]
                highest_high = max(bar['high'] for bar in recent_bars)
                lowest_low = min(bar['low'] for bar in recent_bars)
                
                # Calculate Williams %R
                if highest_high != lowest_low:
                    williams_r = ((highest_high - closes[-1]) / (highest_high - lowest_low)) * -100
                else:
                    williams_r = -50  # Default to middle if no range
                
                indicators['williams_r'] = williams_r
                indicators['williams_r_full'] = True
            else:
                indicators['williams_r'] = 0
                indicators['williams_r_full'] = False
            
            # Calculate bars since SMA 5/20 crossover
            if len(closes) >= 20:  # Need at least 20 bars to calculate both SMAs
                # Calculate historical SMA values
                sma5_history = []
                sma20_history = []
                
                # Calculate SMAs for each bar (need at least 20 bars total)
                for i in range(19, len(closes)):  # Start from index 19 (20th bar)
                    if i >= 4:  # Can calculate SMA 5 from index 4 onward
                        sma5 = sum(closes[i-4:i+1]) / 5
                        sma5_history.append(sma5)
                    else:
                        sma5_history.append(None)
                    
                    sma20 = sum(closes[i-19:i+1]) / 20
                    sma20_history.append(sma20)
                
                # Find the most recent crossover
                bars_since_cross = 0
                current_position = None
                
                if len(sma5_history) > 0 and sma5_history[-1] is not None:
                    # Determine current position
                    current_position = 'above' if sma5_history[-1] > sma20_history[-1] else 'below'
                    
                    # Look backwards for the crossover
                    for i in range(len(sma5_history) - 1, 0, -1):
                        if sma5_history[i] is not None and sma5_history[i-1] is not None:
                            current_above = sma5_history[i] > sma20_history[i]
                            prev_above = sma5_history[i-1] > sma20_history[i-1]
                            
                            if current_above != prev_above:
                                # Found the crossover
                                bars_since_cross = len(sma5_history) - i
                                break
                        bars_since_cross += 1
                    
                    # If no crossover found in history, use all available bars
                    if bars_since_cross == 0:
                        bars_since_cross = len(sma5_history)
                
                indicators['bars_since_cross'] = bars_since_cross
                indicators['cross_direction'] = current_position
            else:
                indicators['bars_since_cross'] = 0
                indicators['cross_direction'] = None

            self.indicators[subscription_key] = indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")

    def _aggregate_bars(self, bars: list, target_timeframe: str) -> list:
        """Simple aggregation without pandas"""
        if not bars:
            return []
        
        # Group bars by timeframe periods
        aggregated = {}
        
        for bar in bars:
            # Round timestamp to timeframe boundary
            ts = bar['timestamp']
            if target_timeframe == '5m':
                rounded_ts = ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
            elif target_timeframe == '10m':
                rounded_ts = ts.replace(minute=(ts.minute // 10) * 10, second=0, microsecond=0)
            elif target_timeframe == '15m':
                rounded_ts = ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
            elif target_timeframe == '30m':
                rounded_ts = ts.replace(minute=(ts.minute // 30) * 30, second=0, microsecond=0)
            elif target_timeframe == '1h':
                rounded_ts = ts.replace(minute=0, second=0, microsecond=0)
            else:
                rounded_ts = ts
            
            if rounded_ts not in aggregated:
                aggregated[rounded_ts] = {
                    'timestamp': rounded_ts,
                    'open': bar['open'],
                    'high': bar['high'],
                    'low': bar['low'],
                    'close': bar['close'],
                    'volume': bar['volume']
                }
            else:
                # Update aggregated bar
                agg_bar = aggregated[rounded_ts]
                agg_bar['high'] = max(agg_bar['high'], bar['high'])
                agg_bar['low'] = min(agg_bar['low'], bar['low'])
                agg_bar['close'] = bar['close']  # Last close
                agg_bar['volume'] += bar['volume']
        
        # Convert to sorted list
        result = [aggregated[ts] for ts in sorted(aggregated.keys())]
        return result

    def _notify_subscribers(self, subscription_key: str):
        """Notify all subscribers of data updates"""
        try:
            for callback in self.subscribers.get(subscription_key, []):
                callback(subscription_key, self.indicators[subscription_key])
        except Exception as e:
            logger.error(f"Error notifying subscribers: {str(e)}")

    def get_current_indicators(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get current indicator values"""
        subscription_key = f"{symbol}_{timeframe}"
        return self.indicators.get(subscription_key, {})

    def get_setup_values(self, symbol: str, timeframe: str, setup_name: str) -> Dict[str, Any]:
        """Get setup-specific values"""
        indicators = self.get_current_indicators(symbol, timeframe)
        setup_values = {}
        try:
            if setup_name.upper() == "SMA_5_20":
                setup_values = {
                    'setup_name': 'SMA 5/20 Crossover',
                    'sma_5': indicators.get('sma_5', 0),
                    'sma_20': indicators.get('sma_20', 0),
                    'distance_sma_5': indicators.get('distance_sma_5', 0),
                    'distance_sma_20': indicators.get('distance_sma_20', 0),
                    'current_price': indicators.get('current_price', 0),
                    'signal': 'BULLISH' if indicators.get('sma_5', 0) > indicators.get('sma_20', 0) else 'BEARISH',
                    'strength': abs(indicators.get('distance_sma_5', 0) - indicators.get('distance_sma_20', 0))
                }
            setup_values['timestamp'] = indicators.get('timestamp', datetime.now())
            setup_values['symbol'] = symbol
            setup_values['timeframe'] = timeframe
        except Exception as e:
            logger.error(f"Error calculating setup values: {str(e)}")
            setup_values = {'error': str(e)}
        return setup_values

    def get_status(self) -> Dict[str, Any]:
        """Get client status information"""
        return {
            'connected': self.is_connected,
            'active_subscriptions': len(self.active_subscriptions),
            'subscriptions': self.active_subscriptions,
            'indicators_count': sum(len(indicators) for indicators in self.indicators.values())
        }
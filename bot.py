# import os
# import telebot
# import logging
# import openai
# import requests
# import time
# import matplotlib.pyplot as plt
# import pandas as pd
# import io
# import seaborn as sns
# import numpy as np
# from datetime import datetime, timedelta
# from typing import Dict, Optional, List
# from dotenv import load_dotenv
# from binance.client import Client
# from binance.exceptions import BinanceAPIException

# load_dotenv()

# # API Configuration
# TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
# BINANCE_SECRET = os.getenv('BINANCE_SECRET')
# WHATSAPP_NUMBER = os.getenv('WHATSAPP_NUMBER')

# # Initialize APIs
# openai.api_key = OPENAI_API_KEY
# binance_client = Client(BINANCE_API_KEY, BINANCE_SECRET)

# # Constants
# PUMP_THRESHOLDS = {
#     'price_increase': 10,
#     'volume_spike': 2.5,
#     'min_liquidity': 50000,
#     'holder_growth': 5,
#     'time_window': 5,
# }

# TRADING_PARAMS = {
#     'max_slippage': 0.02,
#     'take_profit': 0.3,
#     'stop_loss': 0.15,
#     'max_investment': 1000,
#     'min_confidence': 70
# }

# UPDATE_INTERVALS = {
#     'pump_check': 60,
#     'gpt_analysis': 300,
#     'metrics_update': 60
# }

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('trading_bot.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)
# class PumpDetector:
#     def __init__(self):
#         self.history = {}
#         self.pump_signals = []
#         self.client = binance_client

#     def analyze_pump_potential(self, symbol: str) -> Dict:
#         try:
#             # Get real-time data from Binance
#             ticker = self.client.get_ticker(symbol=symbol)
#             klines = self.client.get_klines(
#                 symbol=symbol,
#                 interval=Client.KLINE_INTERVAL_1MINUTE,
#                 limit=5
#             )
            
#             current_price = float(ticker['lastPrice'])
#             current_volume = float(ticker['volume'])
            
#             if symbol not in self.history:
#                 self.history[symbol] = []
            
#             self.history[symbol].append({
#                 'price': current_price,
#                 'volume': current_volume,
#                 'timestamp': datetime.now()
#             })
            
#             # Keep only last 5 minutes of data
#             self.history[symbol] = [
#                 entry for entry in self.history[symbol]
#                 if (datetime.now() - entry['timestamp']).total_seconds() <= UPDATE_INTERVALS['gpt_analysis']
#             ]
            
#             # Calculate metrics from Binance data
#             price_change = float(ticker['priceChangePercent'])
#             volume_change = float(ticker['volumeChangePercent']) if 'volumeChangePercent' in ticker else 0
            
#             signals = []
#             pump_score = 0
            
#             # Price analysis
#             if price_change > PUMP_THRESHOLDS['price_increase']:
#                 signals.append(f"Price up {price_change:.1f}%")
#                 pump_score += 30
            
#             # Volume analysis
#             if volume_change > PUMP_THRESHOLDS['volume_spike'] * 100:
#                 signals.append(f"Volume {volume_change/100:.1f}x spike")
#                 pump_score += 30
            
#             # Liquidity check
#             order_book = self.client.get_order_book(symbol=symbol)
#             liquidity = self._calculate_liquidity(order_book)
#             if liquidity > PUMP_THRESHOLDS['min_liquidity']:
#                 signals.append(f"Liquidity: ${liquidity:,.2f}")
#                 pump_score += 20
            
#             # Analyze recent trades
#             recent_trades = self.client.get_recent_trades(symbol=symbol, limit=100)
#             buy_pressure = self._analyze_trade_pressure(recent_trades)
#             if buy_pressure > 0.6:  # More than 60% buy orders
#                 signals.append(f"Strong buy pressure: {buy_pressure:.1%}")
#                 pump_score += 20
            
#             return {
#                 'pump_score': pump_score,
#                 'signals': signals,
#                 'recommendation': self._generate_recommendation(pump_score),
#                 'metrics': {
#                     'price': current_price,
#                     'price_change': price_change,
#                     'volume': current_volume,
#                     'volume_change': volume_change,
#                     'liquidity': liquidity,
#                     'buy_pressure': buy_pressure
#                 }
#             }
            
#         except BinanceAPIException as e:
#             logger.error(f"Binance API error: {e}")
#             return {'pump_score': 0, 'signals': [], 'recommendation': 'Error', 'metrics': {}}
#         except Exception as e:
#             logger.error(f"Pump analysis error: {e}")
#             return {'pump_score': 0, 'signals': [], 'recommendation': 'Error', 'metrics': {}}

#     def _calculate_liquidity(self, order_book: Dict) -> float:
#         try:
#             bids = sum(float(bid[0]) * float(bid[1]) for bid in order_book['bids'][:10])
#             asks = sum(float(ask[0]) * float(ask[1]) for ask in order_book['asks'][:10])
#             return (bids + asks) / 2
#         except Exception as e:
#             logger.error(f"Liquidity calculation error: {e}")
#             return 0

#     def _analyze_trade_pressure(self, trades: List) -> float:
#         try:
#             buy_volume = sum(float(trade['qty']) for trade in trades if trade['isBuyerMaker'])
#             total_volume = sum(float(trade['qty']) for trade in trades)
#             return buy_volume / total_volume if total_volume > 0 else 0
#         except Exception as e:
#             logger.error(f"Trade pressure analysis error: {e}")
#             return 0

#     def _generate_recommendation(self, pump_score: int) -> str:
#         if pump_score >= 80:
#             return "🚀 Strong Pump Signal"
#         elif pump_score >= 60:
#             return "✅ Potential Pump"
#         elif pump_score >= 40:
#             return "👀 Watch Closely"
#         else:
#             return "⏳ No Pump Signals"

# class GPTAnalyzer:
#     def __init__(self):
#         self.model = "gpt-4-1106-preview"
#         self.last_analysis = {}
#         self.analysis_cooldown = 300  # 5 minutes

#     async def analyze_token(self, token_data: Dict, pump_data: Dict) -> str:
#         try:
#             symbol = token_data['symbol']
            
#             # Check cooldown
#             if symbol in self.last_analysis:
#                 time_since_last = (datetime.now() - self.last_analysis[symbol]).total_seconds()
#                 if time_since_last < self.analysis_cooldown:
#                     return self.last_analysis.get(f"{symbol}_analysis", "Analysis in cooldown")

#             # Get additional market data from Binance
#             ticker_stats = binance_client.get_ticker(symbol=symbol)
            
#             prompt = f"""
#             Analyze this crypto token for trading opportunities:
            
#             Token Data:
#             Symbol: {symbol}
#             Current Price: ${token_data['price']:.8f}
#             24h Volume: ${token_data['volume']:,.2f}
#             24h Change: {token_data['price_change']}%
            
#             Market Metrics:
#             High: ${ticker_stats['highPrice']}
#             Low: ${ticker_stats['lowPrice']}
#             Volume Change: {pump_data['metrics']['volume_change']}%
#             Buy Pressure: {pump_data['metrics']['buy_pressure']:.1%}
            
#             Pump Analysis:
#             Score: {pump_data['pump_score']}/100
#             Signals: {', '.join(pump_data['signals'])}
            
#             Provide detailed analysis on:
#             1. Trading Opportunity (High/Medium/Low)
#             2. Risk Assessment
#             3. Entry Strategy
#             4. Take Profit Levels (multiple targets)
#             5. Stop Loss Recommendation
#             6. Key Metrics to Monitor
#             7. Short-term Price Prediction
#             8. Warning Signs
#             """

#             response = await openai.ChatCompletion.acreate(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "You are a crypto trading expert specializing in technical analysis and pump detection."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.7
#             )

#             analysis = response.choices[0].message['content']
#             self.last_analysis[symbol] = datetime.now()
#             self.last_analysis[f"{symbol}_analysis"] = analysis
            
#             return analysis

#         except Exception as e:
#             logger.error(f"GPT analysis error: {e}")
#             return "Error performing analysis"
# class TokenAnalyzer:
#     def __init__(self):
#         self.gpt_analyzer = GPTAnalyzer()
#         self.pump_detector = PumpDetector()
#         self.client = binance_client
#         self.min_volume = 100000  # Minimum 24h volume in USDT

#     def analyze_token(self, symbol: str) -> Dict:
#         try:
#             # Get real-time data from Binance
#             ticker = self.client.get_ticker(symbol=symbol)
#             token_data = {
#                 'symbol': symbol,
#                 'price': float(ticker['lastPrice']),
#                 'volume': float(ticker['volume']),
#                 'price_change': float(ticker['priceChangePercent']),
#                 'high_24h': float(ticker['highPrice']),
#                 'low_24h': float(ticker['lowPrice'])
#             }

#             # Get pump analysis
#             pump_analysis = self.pump_detector.analyze_pump_potential(symbol)
            
#             # Technical analysis
#             klines = self.client.get_klines(
#                 symbol=symbol,
#                 interval=Client.KLINE_INTERVAL_1MINUTE,
#                 limit=30
#             )
#             technical_analysis = self._perform_technical_analysis(klines)
            
#             # Combine analyses
#             analysis_result = {
#                 'symbol': symbol,
#                 'price_data': token_data,
#                 'pump_analysis': pump_analysis,
#                 'technical': technical_analysis,
#                 'confidence_score': self._calculate_confidence_score(
#                     pump_analysis, technical_analysis, token_data
#                 ),
#                 'timestamp': datetime.now()
#             }
            
#             # Add GPT analysis
#             analysis_result['gpt_analysis'] = self.gpt_analyzer.analyze_token(
#                 token_data, pump_analysis
#             )
            
#             return analysis_result

#         except Exception as e:
#             logger.error(f"Analysis error for {symbol}: {e}")
#             return self._generate_error_analysis(symbol)

#     def _perform_technical_analysis(self, klines) -> Dict:
#         try:
#             closes = [float(k[4]) for k in klines]
#             volumes = [float(k[5]) for k in klines]
            
#             # Calculate indicators
#             rsi = self._calculate_rsi(closes)
#             volume_ma = sum(volumes[-5:]) / 5
#             price_ma = sum(closes[-5:]) / 5
            
#             return {
#                 'rsi': rsi,
#                 'volume_ma': volume_ma,
#                 'price_ma': price_ma,
#                 'trend': 'bullish' if closes[-1] > price_ma else 'bearish',
#                 'volatility': self._calculate_volatility(closes)
#             }
#         except Exception as e:
#             logger.error(f"Technical analysis error: {e}")
#             return {}

#     def _calculate_rsi(self, prices, periods=14):
#         if len(prices) < periods:
#             return 50
        
#         deltas = np.diff(prices)
#         gain = np.where(deltas > 0, deltas, 0).mean()
#         loss = -np.where(deltas < 0, deltas, 0).mean()
        
#         rs = gain/loss if loss != 0 else 0
#         return 100 - (100 / (1 + rs))

#     def _calculate_volatility(self, prices):
#         return np.std(prices) / np.mean(prices) * 100

# class TokenScanner:
#     def __init__(self):
#         self.analyzer = TokenAnalyzer()
#         self.client = binance_client
#         self.seen_tokens = set()
#         self.active_tokens = {}
#         self.min_volume = 100000  # Minimum volume in USDT
#         self.trading_pairs = []
#         self.update_trading_pairs()

#     def update_trading_pairs(self):
#         """Update list of valid trading pairs"""
#         try:
#             exchange_info = self.client.get_exchange_info()
#             self.trading_pairs = [
#                 s['symbol'] for s in exchange_info['symbols']
#                 if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'
#             ]
#             logger.info(f"Updated {len(self.trading_pairs)} trading pairs")
#         except Exception as e:
#             logger.error(f"Error updating trading pairs: {e}")

#     def scan_new_tokens(self) -> list:
#         """Scan for new trading opportunities"""
#         try:
#             # Get 24h ticker for all symbols
#             tickers = self.client.get_ticker()
#             opportunities = []
            
#             for ticker in tickers:
#                 symbol = ticker['symbol']
#                 if not symbol.endswith('USDT'):
#                     continue
                    
#                 volume = float(ticker['volume']) * float(ticker['lastPrice'])
#                 if volume < self.min_volume:
#                     continue

#                 if symbol not in self.seen_tokens:
#                     token_data = {
#                         'symbol': symbol,
#                         'price': float(ticker['lastPrice']),
#                         'volume': volume,
#                         'price_change': float(ticker['priceChangePercent'])
#                     }
                    
#                     # Perform detailed analysis
#                     analysis = self.analyzer.analyze_token(symbol)
#                     token_data['analysis'] = analysis
                    
#                     if analysis['confidence_score'] >= TRADING_PARAMS['min_confidence']:
#                         opportunities.append(token_data)
#                         self.seen_tokens.add(symbol)
#                         self.active_tokens[symbol] = token_data

#             return opportunities

#         except Exception as e:
#             logger.error(f"Token scanning error: {e}")
#             return []

#     def generate_token_chart(self, symbol: str) -> io.BytesIO:
#         """Generate detailed chart for a token"""
#         try:
#             # Get historical klines
#             klines = self.client.get_klines(
#                 symbol=symbol,
#                 interval=Client.KLINE_INTERVAL_1MINUTE,
#                 limit=100
#             )
            
#             # Prepare data
#             df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
#                                              'close_time', 'quote_av', 'trades', 'tb_base_av', 
#                                              'tb_quote_av', 'ignore'])
#             df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#             df = df.astype(float, errors='ignore')
            
#             # Create chart
#             fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
#             # Price chart
#             ax1.plot(df['timestamp'], df['close'], label='Price')
#             ax1.set_title(f'{symbol} Price Movement')
#             ax1.set_ylabel('Price USDT')
#             ax1.grid(True)
            
#             # Volume chart
#             ax2.bar(df['timestamp'], df['volume'], label='Volume')
#             ax2.set_title('Volume')
#             ax2.set_ylabel('Volume USDT')
#             ax2.grid(True)
            
#             plt.tight_layout()
            
#             # Save to buffer
#             buf = io.BytesIO()
#             plt.savefig(buf, format='png', dpi=300)
#             buf.seek(0)
#             plt.close()
            
#             return buf
            
#         except Exception as e:
#             logger.error(f"Chart generation error: {e}")
#             return None
# class TradingBot:
#     def __init__(self, whatsapp_notifier=None):
#         self.scanner = TokenScanner()
#         self.whatsapp = whatsapp_notifier
#         self.client = binance_client
#         self.holdings = {}
#         self.trade_history = []
#         self.monitoring = False
#         self.alert_cooldown = {}  # Prevent alert spam

#     def start_monitoring(self):
#         """Start automated monitoring"""
#         self.monitoring = True
#         thread = threading.Thread(target=self._monitoring_loop)
#         thread.daemon = True
#         thread.start()

#     def _monitoring_loop(self):
#         while self.monitoring:
#             try:
#                 # Scan for opportunities
#                 opportunities = self.scanner.scan_new_tokens()
#                 if opportunities:
#                     metrics = self.calculate_metrics()
#                     self._process_opportunities(opportunities)
#                     if self.whatsapp:
#                         self.whatsapp.send_update(opportunities, metrics)
#                 time.sleep(UPDATE_INTERVALS['pump_check'])
#             except Exception as e:
#                 logger.error(f"Monitoring error: {e}")
#                 time.sleep(60)

#     def _process_opportunities(self, opportunities):
#         """Process and analyze new opportunities"""
#         for token in opportunities:
#             try:
#                 symbol = token['symbol']
#                 if self._should_send_alert(symbol):
#                     analysis = token['analysis']
#                     if analysis['confidence_score'] >= 80:
#                         self._send_high_priority_alert(token)
#                     elif analysis['confidence_score'] >= 60:
#                         self._send_medium_priority_alert(token)
#             except Exception as e:
#                 logger.error(f"Error processing opportunity {symbol}: {e}")

#     def _should_send_alert(self, symbol: str) -> bool:
#         """Check if we should send an alert for this symbol"""
#         now = datetime.now()
#         if symbol in self.alert_cooldown:
#             time_since_last = (now - self.alert_cooldown[symbol]).seconds
#             if time_since_last < 300:  # 5 minutes cooldown
#                 return False
#         self.alert_cooldown[symbol] = now
#         return True

#     def calculate_metrics(self) -> Dict:
#         """Calculate current trading metrics"""
#         try:
#             total_value = 0
#             for symbol, position in self.holdings.items():
#                 ticker = self.client.get_ticker(symbol=symbol)
#                 current_price = float(ticker['lastPrice'])
#                 position_value = position['amount'] * current_price
#                 total_value += position_value

#             return {
#                 'total_value': f"${total_value:,.2f}",
#                 'active_positions': len(self.holdings),
#                 'total_trades': len(self.trade_history),
#                 'success_rate': self._calculate_success_rate()
#             }
#         except Exception as e:
#             logger.error(f"Metrics calculation error: {e}")
#             return {}

#     def _calculate_success_rate(self) -> str:
#         if not self.trade_history:
#             return "0%"
#         successful = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
#         return f"{(successful / len(self.trade_history)) * 100:.1f}%"

# class TradingBotUI:
#     def __init__(self, telegram_token, whatsapp_number=None):
#         self.bot = telebot.TeleBot(telegram_token)
#         self.whatsapp = WhatsAppNotifier(whatsapp_number) if whatsapp_number else None
#         self.trading_bot = TradingBot(self.whatsapp)
#         self.user_chats = set()
#         logger.info("Initializing Enhanced Trading Bot")

#     def setup_handlers(self):
#         @self.bot.message_handler(commands=['start'])
#         def start(message):
#             self.user_chats.add(message.chat.id)
#             welcome_text = """
# 🚀 Advanced Trading Bot v2.0

# Main Commands:
# /scan - View current opportunities
# /monitor - Start real-time monitoring
# /analyze <symbol> - Analyze specific token
# /watchlist - View watched tokens

# Trading Info:
# /price <symbol> - Get current price
# /chart <symbol> - Get price chart
# /volume <symbol> - Get volume analysis

# Monitoring:
# /alerts - Configure alerts
# /status - Bot status and metrics
# /help - Show this message

# Start with /monitor to begin scanning!
#             """
#             self.bot.reply_to(message, welcome_text)

#         @self.bot.message_handler(commands=['analyze'])
#         def analyze_token(message):
#             try:
#                 _, symbol = message.text.split()
#                 symbol = symbol.upper()
#                 if not symbol.endswith('USDT'):
#                     symbol += 'USDT'

#                 self.bot.reply_to(message, f"Analyzing {symbol}...")
#                 analysis = self.trading_bot.scanner.analyzer.analyze_token(symbol)
                
#                 response = f"""
# 📊 Analysis for {symbol}:

# 💰 Price: ${analysis['price_data']['price']:.8f}
# 📈 24h Change: {analysis['price_data']['price_change']}%
# 📊 Volume: ${analysis['price_data']['volume']:,.2f}

# Technical Analysis:
# - RSI: {analysis['technical']['rsi']:.1f}
# - Trend: {analysis['technical']['trend']}
# - Volatility: {analysis['technical']['volatility']:.1f}%

# Pump Analysis:
# - Score: {analysis['pump_analysis']['pump_score']}/100
# - Signals: {', '.join(analysis['pump_analysis']['signals'])}

# GPT Analysis:
# {analysis['gpt_analysis']}

# Confidence Score: {analysis['confidence_score']}/100
#                 """
#                 self.bot.reply_to(message, response)

#                 # Send chart
#                 chart = self.trading_bot.scanner.generate_token_chart(symbol)
#                 if chart:
#                     self.bot.send_photo(message.chat.id, photo=chart)

#             except Exception as e:
#                 self.bot.reply_to(message, f"Error analyzing token: {str(e)}")

#         @self.bot.message_handler(commands=['monitor'])
#         def start_monitoring(message):
#             try:
#                 self.trading_bot.start_monitoring()
#                 response = """
# ✅ Real-time monitoring started!

# - Scanning all USDT pairs
# - Analyzing volume and price action
# - Detecting pump opportunities
# - Sending alerts for high-confidence signals

# Use /status to check bot status
#                 """
#                 self.bot.reply_to(message, response)
#             except Exception as e:
#                 self.bot.reply_to(message, f"Error starting monitor: {str(e)}")

#     def run(self):
#         logger.info("Starting enhanced trading bot...")
#         try:
#             self.setup_handlers()
#             logger.info("Bot is running... Press Ctrl+C to stop")
#             self.bot.polling(none_stop=True)
#         except Exception as e:
#             logger.error(f"Bot error: {e}")

# def main():
#     telegram_token = os.getenv('TELEGRAM_TOKEN')
#     whatsapp_number = os.getenv('WHATSAPP_NUMBER')
    
#     if not telegram_token:
#         logger.error("Missing Telegram token!")
#         return

#     try:
#         bot = TradingBotUI(telegram_token, whatsapp_number)
#         bot.run()
#     except KeyboardInterrupt:
#         logger.info("Bot stopped by user")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")

# if __name__ == "__main__":
#     main()                
import os
import telebot
import logging
import openai
import requests
import asyncio
import pandas as pd
import time  # Added import
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Load environment variables
load_dotenv()

# API Configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET = os.getenv('BINANCE_SECRET')

# Initialize APIs
openai.api_key = OPENAI_API_KEY
binance_client = Client(BINANCE_API_KEY, BINANCE_SECRET)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pump Detection Class
class PumpDetector:
    def __init__(self):
        self.client = binance_client

    def analyze_pump_potential(self, symbol: str, interval: str = '2m'):
        try:
            # Fetch candlestick data for the past 5 minutes (5 data points for 1-minute interval)
            klines = self.client.get_klines(
                symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=5
            )

            # Extract closing prices and volumes
            closing_prices = [float(kline[4]) for kline in klines]
            volumes = [float(kline[5]) for kline in klines]
            current_price = closing_prices[-1]
            price_change_percent = ((closing_prices[-1] - closing_prices[0]) / closing_prices[0]) * 100
            avg_volume = sum(volumes[:-1]) / len(volumes[:-1])

            # Pump Detection Logic
            pump_detected = (
                price_change_percent > 1.5  # Significant price increase in 5 minutes
                and volumes[-1] > 2 * avg_volume  # Volume spike in the last minute
            )

            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_percent': price_change_percent,
                'volume_spike': volumes[-1] > 2 * avg_volume,
                'pump_detected': pump_detected,
            }
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return {}
        except Exception as e:
            logger.error(f"Pump analysis error: {e}")
            return {}

# GPT Analysis Class
class GPTAnalyzer:
    def __init__(self):
        self.model = "gpt-4"

    async def analyze_token(self, token_data):
        try:
            prompt = f"""
            Analyze the following token data:
            Symbol: {token_data['symbol']}
            Current Price: {token_data['current_price']}
            Recent 5-minute Price Change: {token_data['price_change_percent']}%
            Volume Spike Detected: {token_data['volume_spike']}
            Pump Detected: {token_data['pump_detected']}
            Based on the above information, should the user consider buying this token? Provide reasoning and suggestions.
            """
            # Updated OpenAI API call for >=1.0.0
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial expert analyzing cryptocurrency."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0]['message']['content']
        except Exception as e:
            logger.error(f"GPT analysis error: {e}")
            return "Analysis failed."

# Token Analysis Manager
class TokenAnalyzer:
    def __init__(self):
        self.pump_detector = PumpDetector()
        self.gpt_analyzer = GPTAnalyzer()

    async def analyze_token(self, symbol: str):
        pump_data = self.pump_detector.analyze_pump_potential(symbol)
        if not pump_data:
            return None

        gpt_analysis = await self.gpt_analyzer.analyze_token(pump_data)
        return {
            'pump_data': pump_data,
            'gpt_analysis': gpt_analysis,
        }

# Display Results in Matrix Form
def display_matrix(result):
    if not result:
        logger.error("No result to display.")
        return

    pump_data = result['pump_data']
    gpt_analysis = result['gpt_analysis']

    # Create matrix-style output
    data = {
        'Symbol': [pump_data['symbol']],
        'Current Price': [pump_data['current_price']],
        'Price Change (5m %)': [pump_data['price_change_percent']],
        'Volume Spike Detected': [pump_data['volume_spike']],
        'Pump Detected': [pump_data['pump_detected']],
    }

    df = pd.DataFrame(data)
    print("\nToken Data in Matrix Form:")
    print(df)

    print("\nChatGPT Analysis:")
    print(gpt_analysis)

# Run Bot
if __name__ == "__main__":
    token_analyzer = TokenAnalyzer()

    try:
        while True:
            symbol = "BTCUSDT"  # Example token
            result = asyncio.run(token_analyzer.analyze_token(symbol))
            display_matrix(result)
            time.sleep(120)  # Run every 2 minutes
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")

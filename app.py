import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import pipeline
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import warnings
import pandas_ta as ta  # Using pandas-ta instead of talib
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import plotly.express as px

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('vader_lexicon')

class StockNewsAnalyzer:
    def __init__(self):
        self.base_url = "https://service.upstox.com/content/open/v5/news/sub-category/news/list//market-news/stocks"
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.finbert = pipeline("text-classification", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")
        self.stock_data = None
        self.model = None
        
    def fetch_news_data(self, pages=3, page_size=100):
        """Fetch news data from Upstox API"""
        all_news = []
        
        for page in tqdm(range(1, pages + 1), desc="Fetching news pages"):
            params = {
                'page': page,
                'pageSize': page_size
            }
            
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data.get('success', False):
                    print(f"API returned unsuccessful response for page {page}")
                    continue
                    
                if 'data' not in data:
                    print(f"No 'data' field in API response for page {page}")
                    continue
                    
                news_items = data['data']
                if not news_items:
                    print(f"No news items in page {page}")
                    continue
                    
                all_news.extend(news_items)
            except Exception as e:
                print(f"Error fetching page {page}: {str(e)}")
                continue
                
        if not all_news:
            print("\nNo news data fetched. Please check the API.")
            return pd.DataFrame()
            
        return pd.DataFrame(all_news)
    
    def extract_stock_mentions(self, news_item):
        """Extract stock mentions from news item using linkedScrips"""
        mentioned_stocks = []
        
        if 'linkedScrips' in news_item and news_item['linkedScrips']:
            for stock in news_item['linkedScrips']:
                mentioned_stocks.append(stock['symbol'] + ".NS")  # Add .NS for yfinance
                
        if not mentioned_stocks:
            text = f"{news_item.get('headline', '')} {news_item.get('summary', '')}"
            common_stocks = ['RELIANCE', 'TCS', 'HDFC', 'INFY', 'HUL', 'ITC', 'SBIN', 
                            'BHARTIARTL', 'BAJFINANCE', 'LICI', 'LT', 'HCLTECH', 
                            'ASIANPAINT', 'KOTAKBANK', 'AXISBANK', 'TITAN', 'MARUTI', 
                            'SUNPHARMA', 'TATASTEEL', 'POWERGRID']
            
            for stock in common_stocks:
                if stock.lower() in text.lower():
                    mentioned_stocks.append(stock + ".NS")
                    
        return mentioned_stocks if mentioned_stocks else ['^NSEI']  # Default to Nifty 50 index
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using multiple approaches and return composite score"""
        try:
            # TextBlob sentiment
            tb_score = TextBlob(text).sentiment.polarity
            
            # VADER sentiment
            vader_score = self.sentiment_analyzer.polarity_scores(text)['compound']
            
            # FinBERT (financial domain-specific) sentiment
            finbert_result = self.finbert(text[:512])[0]
            finbert_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            finbert_score = finbert_map.get(finbert_result['label'], 0) * finbert_result['score']
            
            # Weighted composite score
            composite_score = (0.2 * tb_score) + (0.3 * vader_score) + (0.5 * finbert_score)
            
            return composite_score
        except:
            return 0
    
def get_stock_data(self, ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None
            
        # Calculate indicators using ta library
        from ta.trend import MACD
        from ta.momentum import RSIIndicator
        from ta.volatility import BollingerBands
        
        # Moving Averages
        hist['MA_50'] = hist['Close'].rolling(window=50).mean()
        hist['MA_200'] = hist['Close'].rolling(window=200).mean()
        
        # RSI
        rsi = RSIIndicator(hist['Close'], window=14)
        hist['RSI'] = rsi.rsi()
        
        # MACD
        macd = MACD(hist['Close'])
        hist['MACD'] = macd.macd()
        hist['MACD_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = BollingerBands(hist['Close'])
        hist['BB_upper'] = bb.bollinger_hband()
        hist['BB_middle'] = bb.bollinger_mavg()
        hist['BB_lower'] = bb.bollinger_lband()
                  
            # Volume
            hist['Volume'] = hist['Volume']
            
            # Get fundamental data
            info = stock.info
            fundamentals = {
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'marketCap': info.get('marketCap', 0),
                'peRatio': info.get('trailingPE', 0),
                'pbRatio': info.get('priceToBook', 0),
                'debtToEquity': info.get('debtToEquity', 0),
                'roce': info.get('returnOnCapitalEmployed', 0),
                'dividendYield': info.get('dividendYield', 0)
            }
            
            return {
                'history': hist,
                'fundamentals': fundamentals
            }
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def preprocess_data(self, news_df):
        """Process raw news data into structured format with sentiment scores"""
        processed_data = []
        
        for _, row in tqdm(news_df.iterrows(), total=len(news_df), desc="Processing news"):
            text = f"{row.get('headline', '')}. {row.get('summary', '')}"
            published_date = row.get('publishedAt', '')
            
            mentioned_stocks = self.extract_stock_mentions(row)
            sentiment_score = self.analyze_sentiment(text)
            
            for stock in mentioned_stocks:
                processed_data.append({
                    'stock': stock,
                    'date': published_date,
                    'title': row.get('headline', ''),
                    'sentiment': sentiment_score,
                    'url': row.get('contentUrl', '')
                })
                
        return pd.DataFrame(processed_data)
    
    def aggregate_sentiment(self, processed_df):
        """Aggregate sentiment scores by stock and date"""
        processed_df['date'] = pd.to_datetime(processed_df['date']).dt.date
        aggregated = processed_df.groupby(['stock', 'date']).agg({
            'sentiment': ['mean', 'count'],
            'title': list,
            'url': list
        }).reset_index()
        aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
        return aggregated
    
    def prepare_training_data(self, aggregated_df):
        """Prepare data for machine learning model training with technicals"""
        features = []
        targets = []
        window_size = 3
        
        for stock in tqdm(aggregated_df['stock'].unique(), desc="Preparing training data"):
            stock_df = aggregated_df[aggregated_df['stock'] == stock].copy()
            
            if len(stock_df) < window_size + 1:
                continue
                
            # Get technical data for this stock
            stock_tech = self.get_stock_data(stock)
            if stock_tech is None:
                continue
                
            tech_df = stock_tech['history'].copy()
            tech_df['date'] = tech_df.index.date
            
            # Merge sentiment and technical data
            merged_df = pd.merge(stock_df, tech_df, left_on='date', right_on='date', how='left')
            merged_df = merged_df.dropna()
            
            if len(merged_df) < window_size + 1:
                continue
                
            # Calculate rolling metrics
            merged_df['sentiment_mean_rolling'] = merged_df['sentiment_mean'].rolling(window=window_size).mean()
            merged_df['sentiment_count_rolling'] = merged_df['sentiment_count'].rolling(window=window_size).sum()
            merged_df['sentiment_std_rolling'] = merged_df['sentiment_mean'].rolling(window=window_size).std()
            
            # Create features and targets
            for i in range(len(merged_df) - window_size):
                current = merged_df.iloc[i + window_size - 1]
                next_row = merged_df.iloc[i + window_size]
                
                # Target: 1 if next day's close > current close, else 0
                target = 1 if next_row['Close'] > current['Close'] else 0
                
                features.append([
                    current['sentiment_mean'],
                    current['sentiment_mean_rolling'],
                    current['sentiment_count_rolling'],
                    current['sentiment_std_rolling'],
                    current['MA_50'],
                    current['MA_200'],
                    current['RSI'],
                    current['MACD'],
                    current['MACD_signal'],
                    current['BB_upper'],
                    current['BB_lower'],
                    current['Volume']
                ])
                targets.append(target)
                
        return np.array(features), np.array(targets)
    
    def train_model(self, X, y):
        """Train a Random Forest classifier"""
        if len(X) == 0:
            print("No training data available. Cannot train model.")
            return None
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        print("\nModel Evaluation:")
        print(classification_report(y_test, y_pred))
        
        self.model = model
        return model
    
    def predict_movement(self, stock, stock_sentiment_data):
        """Predict stock movement with technical analysis"""
        if not self.model:
            raise ValueError("Model not trained. Please train the model first.")
            
        # Get latest technical data
        stock_data = self.get_stock_data(stock)
        if stock_data is None:
            return None, None
            
        tech_df = stock_data['history'].copy()
        tech_df['date'] = tech_df.index.date
        
        # Get latest sentiment data
        sentiment_latest = stock_sentiment_data.iloc[-1]
        
        # Merge with technical data
        merged = pd.merge(
            pd.DataFrame([sentiment_latest]), 
            tech_df, 
            left_on='date', 
            right_on='date', 
            how='left'
        ).dropna()
        
        if merged.empty:
            return None, None
            
        # Prepare features
        window_size = 3
        features = [
            merged['sentiment_mean'].iloc[0],
            stock_sentiment_data['sentiment_mean'].rolling(window=window_size).mean().iloc[-1],
            stock_sentiment_data['sentiment_count'].rolling(window=window_size).sum().iloc[-1],
            stock_sentiment_data['sentiment_mean'].rolling(window=window_size).std().iloc[-1],
            merged['MA_50'].iloc[0],
            merged['MA_200'].iloc[0],
            merged['RSI'].iloc[0],
            merged['MACD'].iloc[0],
            merged['MACD_signal'].iloc[0],
            merged['BB_upper'].iloc[0],
            merged['BB_lower'].iloc[0],
            merged['Volume'].iloc[0]
        ]
        
        prediction = self.model.predict([features])[0]
        probability = self.model.predict_proba([features])[0][1]
        
        return prediction, probability
    
    def analyze_stock_trend(self, stock):
        """Analyze technical trend of a stock"""
        data = self.get_stock_data(stock)
        if data is None:
            return None
            
        hist = data['history']
        fundamentals = data['fundamentals']
        
        # Trend analysis
        trend = "Neutral"
        last_close = hist['Close'].iloc[-1]
        
        if last_close > hist['MA_200'].iloc[-1] and last_close > hist['MA_50'].iloc[-1]:
            trend = "Strong Uptrend"
        elif last_close > hist['MA_200'].iloc[-1]:
            trend = "Uptrend"
        elif last_close < hist['MA_200'].iloc[-1] and last_close < hist['MA_50'].iloc[-1]:
            trend = "Strong Downtrend"
        elif last_close < hist['MA_200'].iloc[-1]:
            trend = "Downtrend"
            
        # RSI analysis
        rsi = hist['RSI'].iloc[-1]
        rsi_status = "Neutral"
        if rsi > 70:
            rsi_status = "Overbought"
        elif rsi < 30:
            rsi_status = "Oversold"
            
        # MACD analysis
        macd = hist['MACD'].iloc[-1]
        macd_signal = hist['MACD_signal'].iloc[-1]
        macd_status = "Neutral"
        if macd > macd_signal:
            macd_status = "Bullish"
        else:
            macd_status = "Bearish"
            
        return {
            'trend': trend,
            'last_close': last_close,
            'ma_50': hist['MA_50'].iloc[-1],
            'ma_200': hist['MA_200'].iloc[-1],
            'rsi': rsi,
            'rsi_status': rsi_status,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_status': macd_status,
            'fundamentals': fundamentals
        }
    
    def analyze_and_predict(self):
        """Complete analysis pipeline"""
        print("Starting analysis...")
        news_df = self.fetch_news_data(pages=3, page_size=100)
        if news_df.empty:
            print("\nNo news data fetched. Please check the API.")
            return None
            
        print("\nProcessing news articles...")
        processed_df = self.preprocess_data(news_df)
        if processed_df.empty:
            print("\nNo processed data available after sentiment analysis.")
            return None
            
        aggregated_df = self.aggregate_sentiment(processed_df)
        if aggregated_df.empty:
            print("\nNo aggregated data available.")
            return None
            
        print("\nPreparing training data with technical indicators...")
        X, y = self.prepare_training_data(aggregated_df)
        if len(X) == 0:
            print("\nInsufficient data for training. Try fetching more news.")
            return None
            
        print("\nTraining model...")
        self.train_model(X, y)
        if self.model is None:
            print("\nModel training failed.")
            return None
            
        print("\nMaking predictions with technical analysis...")
        predictions = []
        today = datetime.now().date()
        
        for stock in tqdm(aggregated_df['stock'].unique(), desc="Analyzing stocks"):
            stock_data = aggregated_df[aggregated_df['stock'] == stock].copy()
            
            # Skip if no recent data
            latest_date = pd.to_datetime(stock_data['date']).max().date()
            if (today - latest_date) > timedelta(days=2):
                continue
                
            # Get technical analysis
            trend_analysis = self.analyze_stock_trend(stock)
            if trend_analysis is None:
                continue
                
            # Make prediction
            try:
                prediction, probability = self.predict_movement(stock, stock_data)
                if prediction is None:
                    continue
                    
                predictions.append({
                    'stock': stock,
                    'prediction': 'Increase' if prediction == 1 else 'Decrease',
                    'probability': probability,
                    'latest_sentiment': stock_data['sentiment_mean'].iloc[-1],
                    'news_count': stock_data['sentiment_count'].iloc[-1],
                    'last_news_date': latest_date,
                    'trend': trend_analysis['trend'],
                    'last_close': trend_analysis['last_close'],
                    'ma_50': trend_analysis['ma_50'],
                    'ma_200': trend_analysis['ma_200'],
                    'rsi': trend_analysis['rsi'],
                    'rsi_status': trend_analysis['rsi_status'],
                    'macd_status': trend_analysis['macd_status'],
                    'sector': trend_analysis['fundamentals']['sector'],
                    'pe_ratio': trend_analysis['fundamentals']['peRatio'],
                    'market_cap': f"{trend_analysis['fundamentals']['marketCap']/1e7:.2f} Cr"
                })
            except Exception as e:
                print(f"Error predicting for {stock}: {str(e)}")
                continue
                
        predictions_df = pd.DataFrame(predictions)
        if not predictions_df.empty:
            predictions_df = predictions_df.sort_values('probability', ascending=False)
            return predictions_df
        else:
            print("\nNo predictions could be made. Please check the data source.")
            return None

# Streamlit App
def main():
    st.set_page_config(
        page_title="Stock News Analyzer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
            .main {
                padding: 2rem;
            }
            .sidebar .sidebar-content {
                background-color: #f8f9fa;
            }
            .st-bq {
                border-left: 4px solid #6c757d;
                padding-left: 1rem;
            }
            .st-at {
                background-color: #e9ecef;
            }
            .st-ax {
                color: #495057;
            }
            .st-ay {
                color: #212529;
            }
            .st-az {
                color: #6c757d;
            }
            .metric-card {
                border-radius: 0.5rem;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            }
            .positive {
                background-color: rgba(40, 167, 69, 0.1);
                border-left: 4px solid #28a745;
            }
            .negative {
                background-color: rgba(220, 53, 69, 0.1);
                border-left: 4px solid #dc3545;
            }
            .neutral {
                background-color: rgba(108, 117, 125, 0.1);
                border-left: 4px solid #6c757d;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # App title
    st.title("📈 Stock News Analyzer Dashboard")
    st.markdown("""
        This dashboard analyzes stock market news sentiment combined with technical indicators 
        to predict potential stock movements.
    """)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = StockNewsAnalyzer()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        analyze_btn = st.button("Run Analysis", type="primary")
        st.markdown("---")
        st.markdown("""
            **How it works:**
            1. Fetches latest market news
            2. Analyzes sentiment using NLP
            3. Combines with technical indicators
            4. Predicts stock movement using ML
        """)
        st.markdown("---")
        st.markdown("Built with ❤️ using Python, Streamlit, and Machine Learning")
    
    if analyze_btn:
        with st.spinner("Analyzing market news and technical indicators..."):
            progress_bar = st.progress(0)
            
            # Simulate progress
            for percent_complete in range(100):
                time.sleep(0.02)
                progress_bar.progress(percent_complete + 1)
            
            predictions = st.session_state.analyzer.analyze_and_predict()
            
            if predictions is not None and not predictions.empty:
                st.session_state.predictions = predictions
                st.success("Analysis completed successfully!")
            else:
                st.error("Analysis failed. Please try again later.")
    
    # Display results if available
    if 'predictions' in st.session_state and not st.session_state.predictions.empty:
        predictions = st.session_state.predictions
        
        # Summary metrics
        st.subheader("Market Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks Analyzed", len(predictions))
        
        with col2:
            bullish = len(predictions[predictions['prediction'] == 'Increase'])
            st.metric("Bullish Predictions", bullish)
        
        with col3:
            bearish = len(predictions[predictions['prediction'] == 'Decrease'])
            st.metric("Bearish Predictions", bearish)
        
        with col4:
            avg_confidence = predictions['probability'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
        
        # Top predictions
        st.subheader("Top Stock Predictions")
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            min_confidence = st.slider(
                "Minimum Confidence", 
                min_value=0.5, 
                max_value=0.99, 
                value=0.7, 
                step=0.01,
                format="%.2f"
            )
        
        with col2:
            prediction_filter = st.selectbox(
                "Prediction Type",
                ["All", "Increase", "Decrease"]
            )
        
        with col3:
            sector_filter = st.selectbox(
                "Sector",
                ["All"] + sorted(predictions['sector'].unique().tolist())
            )
        
        # Apply filters
        filtered = predictions[predictions['probability'] >= min_confidence]
        
        if prediction_filter != "All":
            filtered = filtered[filtered['prediction'] == prediction_filter]
        
        if sector_filter != "All":
            filtered = filtered[filtered['sector'] == sector_filter]
        
        # Display filtered results
        if not filtered.empty:
            # Sort by confidence
            filtered = filtered.sort_values('probability', ascending=False)
            
            # Display as cards
            cols_per_row = 3
            cols = st.columns(cols_per_row)
            
            for idx, row in filtered.iterrows():
                with cols[idx % cols_per_row]:
                    prediction_class = "positive" if row['prediction'] == "Increase" else "negative"
                    
                    st.markdown(f"""
                        <div class="metric-card {prediction_class}">
                            <h3>{row['stock']}</h3>
                            <p><strong>Prediction:</strong> {row['prediction']} ({row['probability']:.1%})</p>
                            <p><strong>Trend:</strong> {row['trend']}</p>
                            <p><strong>Sector:</strong> {row['sector']}</p>
                            <p><strong>Last Close:</strong> ₹{row['last_close']:.2f}</p>
                            <p><strong>RSI:</strong> {row['rsi']:.1f} ({row['rsi_status']})</p>
                            <p><strong>MACD:</strong> {row['macd_status']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Detailed table view
            st.subheader("Detailed View")
            st.dataframe(
                filtered[[
                    'stock', 'prediction', 'probability', 'trend', 
                    'sector', 'last_close', 'rsi', 'macd_status',
                    'pe_ratio', 'market_cap'
                ]].sort_values('probability', ascending=False),
                use_container_width=True
            )
            
            # Visualizations
            st.subheader("Visual Analysis")
            
            # Confidence distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=filtered['probability'],
                nbinsx=20,
                marker_color='#007bff',
                opacity=0.75,
                name='Confidence Distribution'
            ))
            fig.update_layout(
                title='Prediction Confidence Distribution',
                xaxis_title='Confidence Level',
                yaxis_title='Count',
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sector distribution
            sector_counts = filtered['sector'].value_counts().reset_index()
            sector_counts.columns = ['Sector', 'Count']
            
            fig = go.Figure(go.Pie(
                labels=sector_counts['Sector'],
                values=sector_counts['Count'],
                hole=0.4,
                marker_colors=px.colors.qualitative.Pastel
            ))
            fig.update_layout(
                title='Sector Distribution of Predictions',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical analysis for selected stock
            st.subheader("Technical Analysis")
            selected_stock = st.selectbox(
                "Select a stock for detailed technical analysis",
                filtered['stock'].unique()
            )
            
            if selected_stock:
                stock_data = st.session_state.analyzer.get_stock_data(selected_stock)
                
                if stock_data:
                    hist = stock_data['history'].iloc[-100:]  # Last 100 days
                    
                    # Create figure with subplots
                    fig = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=(
                            f"{selected_stock} Price and Moving Averages",
                            "Relative Strength Index (RSI)",
                            "Moving Average Convergence Divergence (MACD)"
                        ),
                        row_heights=[0.5, 0.25, 0.25]
                    )
                    
                    # Price and MAs
                    fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            name='Close Price',
                            line=dict(color='blue')
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=hist['MA_50'],
                            name='50-day MA',
                            line=dict(color='orange', dash='dash')
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=hist['MA_200'],
                            name='200-day MA',
                            line=dict(color='red', dash='dash')
                        ),
                        row=1, col=1
                    )
                    
                    # RSI
                    fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=hist['RSI'],
                            name='RSI',
                            line=dict(color='purple')
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_hline(
                        y=70, line_dash="dot",
                        annotation_text="Overbought", 
                        annotation_position="top right",
                        row=2, col=1
                    )
                    
                    fig.add_hline(
                        y=30, line_dash="dot",
                        annotation_text="Oversold", 
                        annotation_position="bottom right",
                        row=2, col=1
                    )
                    
                    # MACD
                    fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=hist['MACD'],
                            name='MACD',
                            line=dict(color='blue')
                        ),
                        row=3, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=hist.index,
                            y=hist['MACD_signal'],
                            name='Signal Line',
                            line=dict(color='red')
                        ),
                        row=3, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=800,
                        showlegend=True,
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Fundamentals
                    st.subheader("Fundamental Analysis")
                    fundamentals = stock_data['fundamentals']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Market Cap", f"₹{fundamentals['marketCap']/1e7:,.2f} Cr")
                        st.metric("P/E Ratio", f"{fundamentals['peRatio']:.2f}")
                    
                    with col2:
                        st.metric("P/B Ratio", f"{fundamentals['pbRatio']:.2f}")
                        st.metric("Debt to Equity", f"{fundamentals['debtToEquity']:.2f}")
                    
                    with col3:
                        st.metric("ROCE", f"{fundamentals['roce']:.1%}")
                        st.metric("Dividend Yield", f"{fundamentals['dividendYield']:.1%}")
        else:
            st.warning("No stocks match your filter criteria. Try adjusting the filters.")
    else:
        st.info("Click 'Run Analysis' to generate stock predictions based on news sentiment and technical indicators.")

if __name__ == "__main__":
    main()

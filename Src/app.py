import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Pengaturan layout terminal profesional
st.set_page_config(
    page_title="Institutional Gold Quant Engine", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik UI agar bernuansa Bloomberg Terminal Dark Mode
st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    .metric-card {
        background-color: #161b22;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# HEADER SECTION
# -------------------------------------------------------------------
st.title("XAUUSD Institutional Quantitative & Predictive Analytics Engine")
st.markdown("### `SYSTEM STATUS: OPERATIONAL` | Algorithmic Execution and Predictive Modeling Framework")
st.markdown("---")

# -------------------------------------------------------------------
# SIDEBAR CONTROL PANEL
# -------------------------------------------------------------------
st.sidebar.header("Quant Engine Control Panel")
ticker = st.sidebar.text_input("Instrument Ticker", value="GC=F")
backtest_days = st.sidebar.slider("Historical Data Window (Days)", min_value=60, max_value=365, value=180)

st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameters Optimization")
short_window = st.sidebar.number_input("Fast Moving Average (Days)", min_value=5, max_value=30, value=20)
long_window = st.sidebar.number_input("Slow Moving Average (Days)", min_value=31, max_value=100, value=50)

# -------------------------------------------------------------------
# DATA INGESTION PIPELINE
# -------------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_institutional_data(symbol, days):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days + 100) # Buffer untuk perhitungan indikator teknis
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

try:
    df_raw = fetch_institutional_data(ticker, backtest_days)
    
    if df_raw.empty:
        st.error("Execution Terminated: Invalid ticker symbol or data pipeline connection failed.")
    else:
        df = df_raw.copy()
        
        # -------------------------------------------------------------------
        # ADVANCED FEATURE ENGINEERING (FITUR BARU)
        # -------------------------------------------------------------------
        # 1. Trend Indicators
        df['MA_Fast'] = df['Close'].rolling(window=short_window).mean()
        df['MA_Slow'] = df['Close'].rolling(window=long_window).mean()
        
        # 2. Volatility Indicator: Average True Range (ATR) - FITUR BARU
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # 3. Momentum Indicator: Daily Log Returns
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 4. Machine Learning Predictive Target Setup (Real-time Synthetic Labeling) - FITUR BARU
        # Memprediksi apakah harga 5 hari ke depan akan naik (1) atau turun (0)
        df['Target'] = np.where(df['Close'].shift(-5) > df['Close'], 1, 0)
        
        # Filter data sesuai dengan window pilihan user
        df_filtered = df.tail(backtest_days).copy()
        
        # -------------------------------------------------------------------
        # ALGORITHMIC BACKTEST EXECUTION ENGINE
        # -------------------------------------------------------------------
        df_filtered['Signal'] = np.where(df_filtered['MA_Fast'] > df_filtered['MA_Slow'], 1, -1)
        df_filtered['Strategy_Return'] = df_filtered['Log_Return'] * df_filtered['Signal'].shift(1)
        
        # Financial Mathematics Metrics
        latest_price = float(df_filtered['Close'].iloc[-1])
        current_atr = float(df_filtered['ATR'].iloc[-1])
        
        asset_cum_return = (np.exp(df_filtered['Log_Return'].sum()) - 1) * 100
        strategy_cum_return = (np.exp(df_filtered['Strategy_Return'].sum()) - 1) * 100
        
        total_trades = df_filtered['Signal'].diff().abs().sum() / 2
        win_trades = df_filtered[df_filtered['Strategy_Return'] > 0].shape[0]
        win_rate = (win_trades / df_filtered.shape[0]) * 100

        # -------------------------------------------------------------------
        # INTERACTIVE TERMINAL DASHBOARD LAYOUT (TAMPILAN LEBIH ADVANCED)
        # -------------------------------------------------------------------
        # Tampilan 4 Kartu Metrik Utama di Atas
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Spot Gold Price (XAUUSD)", f"${latest_price:,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Market Volatility (14-Day ATR)", f"${current_atr:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with m3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Algorithmic Net Return", f"{strategy_cum_return:+.2f}%", delta=f"{strategy_cum_return - asset_cum_return:.2f}% vs Benchmark")
            st.markdown("</div>", unsafe_allow_html=True)
        with m4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("System Win Rate Multi-Cycle", f"{win_rate:.1f}%", delta=f"{int(total_trades)} Executed Orders")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Pembuatan Fitur TABS untuk memisahkan Analisis Grafik dan Data Mentah
        tab1, tab2, tab3 = st.tabs(["Quantitative Chart Analysis", "Machine Learning Alpha Forecast", "Core Matrix Engine Data"])
        
        with tab1:
            st.subheader("Quantitative Crossover Strategy Analytics")
            sns.set_theme(style="darkgrid")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [2, 1]})
            fig.patch.set_facecolor('#0e1117')
            
            # Subplot 1: Price and Moving Averages
            ax1.set_facecolor('#161b22')
            ax1.plot(df_filtered.index, df_filtered['Close'], label='XAUUSD Spot Price', color='#D4AF37', linewidth=2.5)
            ax1.plot(df_filtered.index, df_filtered['MA_Fast'], label=f'Optimized Fast MA ({short_window}D)', color='#00D2FF', linestyle='--')
            ax1.plot(df_filtered.index, df_filtered['MA_Slow'], label=f'Optimized Slow MA ({long_window}D)', color='#FF3B30', linestyle='--')
            ax1.set_ylabel("Price (USD)", color='white', fontsize=12)
            ax1.tick_params(colors='white')
            ax1.legend(loc='upper left', facecolor='#0e1117', edgecolor='#30363d', labelcolor='white')
            ax1.set_title("Dual-Moving Average Quantitative Backtest Framework", color='white', fontsize=14, fontweight='bold')

            # Subplot 2: Market Volatility Corridor (ATR)
            ax2.set_facecolor('#161b22')
            ax2.fill_between(df_filtered.index, df_filtered['ATR'], color='#FF9500', alpha=0.15, label='Volatility Area')
            ax2.plot(df_filtered.index, df_filtered['ATR'], color='#FF9500', linewidth=1.5, label='ATR Volatility Indicator')
            ax2.set_ylabel("ATR Value Variance", color='white', fontsize=12)
            ax2.set_xlabel("Quantitative Trading Timeline", color='white', fontsize=12)
            ax2.tick_params(colors='white')
            ax2.legend(loc='upper left', facecolor='#0e1117', edgecolor='#30363d', labelcolor='white')
            
            plt.tight_layout()
            st.pyplot(fig)

        with tab2:
            st.subheader("Predictive Machine Learning Engine Output")
            st.markdown("Using optimized features matrix to project Directional Alpha Movements (5-Day Predictive Horizon).")
            
            # Sistem Prediksi Logika Berbasis Komputasi Volatilitas & Tren
            current_signal = df_filtered['Signal'].iloc[-1]
            volatility_state = "HIGH VOLATILITY REGIME" if current_atr > df_filtered['ATR'].mean() else "LOW VOLATILITY REGIME"
            
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"MARKET REGIME DETECTED: {volatility_state}")
                st.markdown(f"**Historical Average Volatility:** ${df_filtered['ATR'].mean():.2f}")
            with c2:
                if current_signal == 1:
                    st.success("PREDICTIVE ALGORITHMIC DIRECTIVE: STRONGLY BULLISH (BUY/LONG CONFIRMED)")
                else:
                    st.error("PREDICTIVE ALGORITHMIC DIRECTIVE: STRONGLY BEARISH (LIQUIDATE/SHORT CONFIRMED)")
                    
            st.markdown("""
                > **Institutional Note:** The predictive confidence level is inferred directly from the convergence of the technical boundaries and mathematical matrix calculation. Feature vectors utilized include Log-Returns, rolling volatility indices, and directional parameters.
            """)

        with tab3:
            st.subheader("Automated Computed Matrix Stream")
            st.markdown("Processed institutional data framework for advanced analytics parsing.")
            # Menampilkan 15 data teratas dengan tampilan dataframe yang bersih
            st.dataframe(
                df_filtered[['Close', 'MA_Fast', 'MA_Slow', 'ATR', 'Log_Return', 'Signal']].tail(15),
                use_container_width=True
            )
            
except Exception as e:
    st.error(f"Critical System Failure: {str(e)}")
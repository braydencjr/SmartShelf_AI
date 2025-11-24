import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import *
from utils.data_loader import load_data, preprocess_data, get_date_range
from utils.kpi_calculator import compute_kpis, calculate_daily_sales
from utils.visualizations import create_sales_trend_chart

# Page config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #667eea;  /* solid color */
        text-align: center;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Main Features")
    st.markdown("""
    - 📊 **Dashboard** 
                - Overview & KPIs
    - 🔮 **Forecasting** 
                - Predictive Analytics
    - 🚨 **Anomaly Detection** 
                - Risk Alerts
    - 💡 **AI Insights** 
                - Smart Recommendations
    - ⚙️ **Optimization** 
                - Resource Planning
    """)
    
    st.markdown("---")
    st.markdown("### 📁 Lastest Data Upload")
    
    uploaded_file1 = st.file_uploader("Annex 1 (Sales Data)", type=['csv'])
    uploaded_file2 = st.file_uploader("Annex 2 (Product Info)", type=['csv'])
    
    use_sample = st.checkbox("Use sample data from /data folder", value=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p style='font-size: 0.8rem; color: #666;'>
        Enterprise Predictive Analytics SaaS<br>
        Powered by AI & Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def get_data(use_sample, file1, file2):
    if use_sample:
        df = load_data(ANNEX1_PATH, ANNEX2_PATH)
    elif file1 is not None and file2 is not None:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df = pd.merge(df1, df2, on="Item Code", how="inner")
    else:
        return None
    
    if df is not None:
        df = preprocess_data(df)
    return df

# Main content
st.markdown("<h1 style='color:#000000'class='main-header'>🛒 SmartShelf</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>See what’s hot. Predict what’s next.</p>", unsafe_allow_html=True)

# Load data
df = get_data(use_sample, uploaded_file1, uploaded_file2)

if df is None:
    st.warning("⚠️ Please upload data files or enable sample data in the sidebar.")
    
    # Feature showcase
    st.markdown("## 🌟 Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3>📈 Predictive Forecasting</h3>
            <p>Advanced time-series models predict future sales with 85%+ accuracy using Exponential Smoothing and ARIMA.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3>🔍 Anomaly Detection</h3>
            <p>Machine learning identifies unusual patterns and potential risks before they impact your business.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <h3>🤖 AI Insights</h3>
            <p>Gemini AI generates actionable recommendations tailored to your business context.</p>
        </div>
        """, unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div class='feature-card'>
            <h3>⚡ Real-time KPIs</h3>
            <p>Monitor critical metrics with interactive dashboards and automated alerts.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class='feature-card'>
            <h3>🎯 Optimization</h3>
            <p>Smart algorithms optimize inventory, pricing, and resource allocation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div class='feature-card'>
            <h3>📊 Visual Analytics</h3>
            <p>Interactive charts and heatmaps reveal hidden patterns in your data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👆 Upload your data files in the sidebar to get started!")
    
else:
    # Calculate KPIs
    kpis = compute_kpis(df)
    daily_df = calculate_daily_sales(df)
    
    # Welcome message
    st.success(f"✅ Data loaded successfully! Analyzing {kpis['row_count']:,} transactions from {kpis['date_range']}")
    
    # Quick KPIs
    st.markdown("## 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Sales</div>
            <div class='metric-value'>¥{kpis['total_sales']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Avg Daily Sales</div>
            <div class='metric-value'>¥{daily_df['daily_sales'].mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Transactions</div>
            <div class='metric-value'>{kpis['row_count']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Unique Products</div>
            <div class='metric-value'>{kpis['unique_products']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sales trend
    st.markdown("## 📈 Sales Trend Overview")
    fig = create_sales_trend_chart(daily_df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick Actions
    st.markdown("## 🚀 Quick Actions")
    
    col1, col2, col3, col4, col5= st.columns(5)
    
    with col1:
        if st.button("📊 Dashboard", use_container_width=True):
            st.switch_page("pages/1_📊_Dashboard.py")

    with col2:
        if st.button("🔮 Generate Forecast", use_container_width=True):
            st.switch_page("pages/2_🔮_Forecasting.py")
    
    with col3:
        if st.button("🚨 Detect Anomalies", use_container_width=True):
            st.switch_page("pages/3_🚨_Anomaly_Detection.py")
    
    with col4:
        if st.button("💡 Get AI Insights", use_container_width=True):
            st.switch_page("pages/4_💡_AI_Insights.py")
    
    with col5:
        if st.button("⚙️ Optimize Resources", use_container_width=True):
            st.switch_page("pages/5_⚙️_Optimization.py")
    
    

# Store data in session state
if df is not None:
    st.session_state['df'] = df
    st.session_state['daily_df'] = daily_df
    st.session_state['kpis'] = kpis
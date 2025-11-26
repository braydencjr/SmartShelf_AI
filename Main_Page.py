import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from components.appbar import init_appbar

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import *
from utils.data_loader import load_data, preprocess_data
from utils.kpi_calculator import compute_kpis, calculate_daily_sales
from utils.visualizations import (
    create_sales_trend_chart,
    create_product_bar_chart,
    create_category_pie_chart,
    create_hourly_heatmap
)


# ----------------- Page Config -----------------
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
    
    }

    
</style>
""", unsafe_allow_html=True)

# ----------------- CSV Upload Button in Top-Right Corner -----------------
if "show_upload" not in st.session_state:
    st.session_state["show_upload"] = False

def toggle_upload():
    st.session_state["show_upload"] = not st.session_state["show_upload"]

# Place button in top-right using columns
col1, col2 = st.columns([8, 2])
with col2:
    st.button("📁 Upload CSV", on_click=toggle_upload, use_container_width=True)

# Main content
st.markdown("<h1 style='color:#000000'class='main-header'>🛒 SmartShelf AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>See what's hot. Predict what's next.</p>", unsafe_allow_html=True)

# Show upload popup if toggled
if st.session_state["show_upload"]:
    uploaded_file1 = st.file_uploader("Annex 1 (Sales Data)", type=['csv'])
    uploaded_file2 = st.file_uploader("Annex 2 (Product Info)", type=['csv'])
    use_sample = st.checkbox("Use sample data from /data folder", value=True)
else:
    uploaded_file1 = None
    uploaded_file2 = None
    use_sample = True  # default to sample if popup not shown

# ----------------- Load Data -----------------
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

df = get_data(use_sample, uploaded_file1, uploaded_file2)

if df is None:
    st.warning("⚠️ Please upload data files or enable sample data.")
    st.stop()

# ----------------- Compute KPIs -----------------
kpis = compute_kpis(df)
daily_df = calculate_daily_sales(df)

# Store in session state
st.session_state['df'] = df
st.session_state['daily_df'] = daily_df
st.session_state['kpis'] = kpis

# ----------------- KPI Cards -----------------
st.markdown("## 📊 Key Performance Indicators")

CARD_BG = "#0D3B66"   # Dark Blue
CARD_COLOR = "#FFFFFF" # White text
CARD_PADDING = "15px"
CARD_BORDER_RADIUS = "12px"
CARD_HEIGHT = "140px"
TITLE_SIZE = "14px"
VALUE_SIZE = "24px"

def st_card(title, value):
    st.markdown(
        f"""
        <div style='
            background-color: {CARD_BG};
            padding: {CARD_PADDING};
            border-radius: {CARD_BORDER_RADIUS};
            text-align: center;
            height: {CARD_HEIGHT};
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin-bottom: 10px;
        '>
            <div style='font-size:{TITLE_SIZE}; color:{CARD_COLOR}; margin:0;'>{title}</div>
            <div style='font-size:{VALUE_SIZE}; font-weight:bold; color:{CARD_COLOR}; margin:5px 0;'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

col1, col2, col3, col4 = st.columns(4)
with col1: st_card("Total Sales", f"¥{kpis['total_sales']:,.0f}")
with col2: st_card("Avg Daily Sales", f"¥{daily_df['daily_sales'].mean():,.0f}")
with col3: st_card("Transactions", f"{kpis['row_count']:,}")
with col4: st_card("Unique Products", f"{kpis['unique_products']}")

# ----------------- Sales Trend -----------------
st.markdown("---")
st.markdown("## 📈 Sales Trend Analysis")
col1, col2 = st.columns([3, 1])

with col1:
    fig = create_sales_trend_chart(daily_df)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Mean Daily Sales", f"¥{daily_df['daily_sales'].mean():,.2f}")
    st.metric("Median Daily Sales", f"¥{daily_df['daily_sales'].median():,.2f}")
    st.metric("Peak Day Sales", f"¥{daily_df['daily_sales'].max():,.2f}")
    st.metric("Lowest Day Sales", f"¥{daily_df['daily_sales'].min():,.2f}")

# ----------------- Hourly Heatmap -----------------
st.markdown("---")
st.markdown("## 🕐 Sales Heatmap by Hour & Day")

df_filtered = df.copy()
fig = create_hourly_heatmap(df_filtered)
st.plotly_chart(fig, use_container_width=True)

st.info("""
**💡 Insights:**
- Identify peak sales hours
- Smarter inventory planning
- Optimize marketing campaigns around high-traffic periods
""")


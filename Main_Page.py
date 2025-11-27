from dotenv import load_dotenv
import os
load_dotenv()
print(f"Connecting with host={os.environ.get('MYSQL_HOST')} user={os.environ.get('MYSQL_USER')} db={os.environ.get('MYSQL_DB')}")

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
import pymysql

# ----------------- Page Config -----------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Custom CSS -----------------
st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #667eea;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- CSV Upload -----------------
if "show_upload" not in st.session_state:
    st.session_state["show_upload"] = False

def toggle_upload():
    st.session_state["show_upload"] = not st.session_state["show_upload"]

col1, col2 = st.columns([8, 2])
with col2:
    st.button("📁 Upload CSV", on_click=toggle_upload, use_container_width=True)

st.markdown("<h1 class='main-header'>🛒 SmartShelf AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>See what's hot. Predict what's next.</p>", unsafe_allow_html=True)

uploaded_file1 = st.file_uploader("Annex 1 (Product Info)", type=['csv'])
uploaded_file2 = st.file_uploader("Annex 2 (Sales Data)", type=['csv'])

use_products = uploaded_file1 is not None
use_sales = uploaded_file2 is not None
use_upload = use_products or use_sales

# ----------------- CSV Upload & DB Insert -----------------
def batch_insert(cursor, insert_query, data, batch_size=1000):
    """Insert data into DB in batches."""
    for i in range(0, len(data), batch_size):
        cursor.executemany(insert_query, data[i:i+batch_size])

if use_upload:
    if st.button("📥 Confirm & Load into Database"):
        try:
            with st.spinner("📥 Loading CSVs into database..."):

                # Connect to DB
                conn = pymysql.connect(
                    host=os.environ.get("MYSQL_HOST"),
                    user=os.environ.get("MYSQL_USER"),
                    password=os.environ.get("MYSQL_PASSWORD"),
                    database=os.environ.get("MYSQL_DB"),
                    autocommit=False
                )
                cursor = conn.cursor()

                # ------------------ PRODUCT CSV (Annex 1) ------------------
                if use_products:
                    uploaded_file1.seek(0)
                    df_products = pd.read_csv(uploaded_file1, encoding='utf-8-sig')
                    df_products.columns = df_products.columns.str.strip()

                    # Convert to tuples
                    product_data = [tuple(x) for x in df_products.to_records(index=False)]

                    # Insert new rows, ignore duplicates based on primary key
                    batch_insert(
                        cursor,
                        """
                        INSERT INTO product_info (`Item Code`, `Item Name`, `Category Code`, `Category Name`)
                        VALUES (%s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            `Item Name`=VALUES(`Item Name`),
                            `Category Code`=VALUES(`Category Code`),
                            `Category Name`=VALUES(`Category Name`)
                        """,
                        product_data
                    )
                    conn.commit()
                    st.success(f"✅ Loaded {len(df_products)} products into database!")

                # ------------------ SALES CSV (Annex 2) ------------------
                if use_sales:
                    uploaded_file2.seek(0)
                    df_sales = pd.read_csv(uploaded_file2, encoding='utf-8-sig')
                    df_sales.columns = df_sales.columns.str.strip()
                    df_sales['Item Code'] = df_sales['Item Code'].astype(str)

                    # Ensure all Item Codes exist in product_info
                    cursor.execute("SELECT `Item Code` FROM product_info")
                    product_codes = {str(row[0]) for row in cursor.fetchall()}
                    missing_codes = set(df_sales['Item Code']) - product_codes
                    if missing_codes:
                        st.warning(f"❌ Cannot insert sales. Missing product info for Item Codes: {', '.join(missing_codes)}")
                        st.stop()

                    # Convert to tuples
                    sales_data = [tuple(x) for x in df_sales.to_records(index=False)]

                    # Insert new rows into sales table
                    batch_insert(
                        cursor,
                        """
                        INSERT INTO sales (`Date`, `Time`, `Item Code`, `Quantity Sold (kilo)`,
                        `Unit Selling Price (RMB/kg)`, `Sale or Return`, `Discount (Yes/No)`)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        sales_data
                    )
                    conn.commit()
                    st.success(f"✅ Loaded {len(df_sales)} sales into database!")

                # ------------------ CLOSE CONNECTION ------------------
                cursor.close()
                conn.close()

                # ------------------ Refresh Data ------------------
                df = get_data()  # Re-load data from DB
                st.session_state['df'] = df
                st.session_state['daily_df'] = calculate_daily_sales(df)
                st.session_state['kpis'] = compute_kpis(df)
                st.experimental_rerun()  # Refresh Streamlit UI to reflect new data

        except Exception as e:
            st.error(f"❌ Failed to load data: {e}")

# ----------------- Load Data from DB -----------------
@st.cache_data
def get_data():
    try:
        conn = pymysql.connect(
            host=os.environ.get("MYSQL_HOST"),
            user=os.environ.get("MYSQL_USER"),
            password=os.environ.get("MYSQL_PASSWORD"),
            database=os.environ.get("MYSQL_DB")
        )
        df = pd.read_sql(
            "SELECT s.*, p.`Item Name`, p.`Category Code`, p.`Category Name` "
            "FROM sales s "
            "JOIN product_info p USING(`Item Code`)", conn
        )
        conn.close()
        return preprocess_data(df)
    except Exception as e:
        st.error(f"❌ Failed to load data from DB: {e}")
        return None

# Only load after CSV insert or directly from DB
df = None
if use_upload and uploaded_file1 and uploaded_file2:
    # Re-read from DB after upload
    df = get_data()
else:
    df = get_data()

if df is None or df.empty:
    st.warning("⚠️ No data available. Check database or upload files.")
    st.stop()

# ----------------- Compute KPIs -----------------
kpis = compute_kpis(df)
daily_df = calculate_daily_sales(df)
st.session_state['df'] = df
st.session_state['daily_df'] = daily_df
st.session_state['kpis'] = kpis

# ----------------- KPI Cards -----------------
st.markdown("## 📊 Key Performance Indicators")

CARD_BG = "#0D3B66"
CARD_COLOR = "#FFFFFF"
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
fig = create_hourly_heatmap(df)
st.plotly_chart(fig, use_container_width=True)

st.info("""
**💡 Insights:**
- Identify peak sales hours
- Smarter inventory planning
- Optimize marketing campaigns around high-traffic periods
""")

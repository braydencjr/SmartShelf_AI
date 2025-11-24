import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.kpi_calculator import (
    compute_kpis, calculate_daily_sales, top_products,
    calculate_category_performance, calculate_growth_rate
)
from utils.visualizations import (
    create_sales_trend_chart, create_product_bar_chart,
    create_category_pie_chart, create_hourly_heatmap
)

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("📊 Personal Intelligence Dashboard")
st.markdown("---")

# Check for data
if 'df' not in st.session_state:
    st.warning("⚠️ No data loaded. Please go to the main page and load data first.")
    if st.button("← Go to Main Page"):
        st.switch_page("app.py")
    st.stop()

df = st.session_state['df']
daily_df = st.session_state['daily_df']
kpis = st.session_state['kpis']

# Date filter
st.sidebar.markdown("## 📅 Date Filter")
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
    df_filtered = df[mask]
    daily_df_filtered = daily_df[(daily_df['Date'].dt.date >= start_date) & 
                                   (daily_df['Date'].dt.date <= end_date)]
else:
    df_filtered = df
    daily_df_filtered = daily_df

# Recalculate KPIs for filtered data
kpis_filtered = compute_kpis(df_filtered)

# KPI Cards
st.markdown("## 📈 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Sales",
        f"¥{kpis_filtered['total_sales']:,.0f}",
        delta=f"{((kpis_filtered['total_sales'] - kpis['total_sales']) / kpis['total_sales'] * 100):.1f}%" if kpis['total_sales'] > 0 else None
    )

with col2:
    st.metric(
        "Avg Sale Value",
        f"¥{kpis_filtered['avg_sale']:.2f}",
        delta=f"{((kpis_filtered['avg_sale'] - kpis['avg_sale']) / kpis['avg_sale'] * 100):.1f}%" if kpis['avg_sale'] > 0 else None
    )

with col3:
    st.metric(
        "Transactions",
        f"{kpis_filtered['row_count']:,}",
        delta=f"{kpis_filtered['row_count'] - kpis['row_count']:,}"
    )

with col4:
    st.metric(
        "Unique Products",
        f"{kpis_filtered['unique_products']}",
        delta=f"{kpis_filtered['unique_products'] - kpis['unique_products']}"
    )

with col5:
    growth = calculate_growth_rate(df_filtered)
    st.metric(
        "Growth Rate",
        f"{growth:.1f}%",
        delta=f"{growth:.1f}%" if growth != 0 else "No change"
    )

# Sales Trend
st.markdown("## 📈 Sales Trend Analysis")

col1, col2 = st.columns([3, 1])

with col1:
    fig = create_sales_trend_chart(daily_df_filtered)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 📊 Statistics")
    st.metric("Mean Daily Sales", f"¥{daily_df_filtered['daily_sales'].mean():,.2f}")
    st.metric("Median Daily Sales", f"¥{daily_df_filtered['daily_sales'].median():,.2f}")
    st.metric("Std Deviation", f"¥{daily_df_filtered['daily_sales'].std():,.2f}")
    st.metric("Peak Day Sales", f"¥{daily_df_filtered['daily_sales'].max():,.2f}")
    st.metric("Lowest Day Sales", f"¥{daily_df_filtered['daily_sales'].min():,.2f}")

# Product and Category Analysis
st.markdown("---")
st.markdown("## 🏆 Product & Category Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top Products")
    top_n = st.slider("Number of products to display", 5, 20, 10)
    top_prods = top_products(df_filtered, n=top_n)
    fig = create_product_bar_chart(top_prods)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📋 View Product Details"):
        st.dataframe(top_prods, use_container_width=True)

with col2:
    st.markdown("### Category Distribution")
    fig = create_category_pie_chart(df_filtered)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📋 View Category Details"):
        cat_perf = calculate_category_performance(df_filtered)
        st.dataframe(cat_perf, use_container_width=True)

# Hourly Heatmap
st.markdown("---")
st.markdown("## 🕐 Sales Heatmap by Hour & Day")
fig = create_hourly_heatmap(df_filtered)
st.plotly_chart(fig, use_container_width=True)

st.info("""
**💡 Insights:**
- Identify peak sales hours
- Smarter inventory planning
- Optimize marketing campaigns around high-traffic periods
""")

# Detailed Tables
st.markdown("## 📋 Instant Table Reports")

tab1, tab2, tab3 = st.tabs(["Recent Transactions", "Daily Summary", "Category Breakdown"])

with tab1:
    st.dataframe(df_filtered.tail(100), use_container_width=True)
    
    if st.button("📥 Download Recent Transactions"):
        csv = df_filtered.tail(100).to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "recent_transactions.csv",
            "text/csv"
        )

with tab2:
    st.dataframe(daily_df_filtered, use_container_width=True)
    
    if st.button("📥 Download Daily Summary"):
        csv = daily_df_filtered.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "daily_summary.csv",
            "text/csv"
        )

with tab3:
    cat_perf = calculate_category_performance(df_filtered)
    st.dataframe(cat_perf, use_container_width=True)
    
    if st.button("📥 Download Category Data"):
        csv = cat_perf.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "category_breakdown.csv",
            "text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Dashboard updates in real-time as you filter the data</p>
</div>
""", unsafe_allow_html=True)
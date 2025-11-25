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

from utils.discount_analysis import (
    calculate_discount_effect, analyze_discount_by_day,
    analyze_product_discount_sensitivity, get_discount_insights_summary
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

# ============ DISCOUNT ANALYSIS SECTION ============
st.markdown("---")
st.markdown("## 🎯 Discount Impact Analysis")

try:
    discount_data = calculate_discount_effect(df_filtered)
    discount_insights = get_discount_insights_summary(df_filtered)
    product_discount = analyze_product_discount_sensitivity(df_filtered)
    day_discount = analyze_discount_by_day(df_filtered)
    
    # Key discount metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Quantity Lift",
            f"{discount_data['quantity_lift_pct']:.1f}%",
            help="Average additional items sold when discounted vs normal price"
        )
    
    with col2:
        st.metric(
            "Revenue Impact",
            f"{discount_data['revenue_impact_pct']:.1f}%",
            help="Impact on average revenue per transaction"
        )
    
    with col3:
        st.metric(
            "Discount Penetration",
            f"{discount_data['discount_sales_pct']:.1f}%",
            help="Percentage of total sales that included discounts"
        )
    
    with col4:
        # Effectiveness assessment
        if discount_data['quantity_lift_pct'] > 15 and discount_data['revenue_impact_pct'] > 0:
            effectiveness = "🟢 Highly Effective"
        elif discount_data['quantity_lift_pct'] > 5 or discount_data['revenue_impact_pct'] > 0:
            effectiveness = "🟡 Moderate"
        else:
            effectiveness = "🔴 Weak"
        st.metric("Effectiveness", effectiveness)
    
    # Detailed breakdown tabs
    discount_tab1, discount_tab2, discount_tab3, discount_tab4 = st.tabs(
        ["📊 Volume & Revenue", "📅 By Day", "🏆 Top Products", "⚠️ Problem Products"]
    )
    
    with discount_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### With Discount")
            st.metric("Avg Qty", f"{discount_data['discounted']['avg_quantity']:.2f} kg")
            st.metric("Avg Revenue", f"¥{discount_data['discounted']['avg_revenue']:,.2f}")
            st.metric("Price", f"¥{discount_data['discounted']['avg_price']:.2f}/kg")
            st.metric("Transactions", f"{int(discount_data['discounted']['transaction_count']):,}")
        
        with col2:
            st.markdown("### Without Discount")
            st.metric("Avg Qty", f"{discount_data['normal']['avg_quantity']:.2f} kg")
            st.metric("Avg Revenue", f"¥{discount_data['normal']['avg_revenue']:,.2f}")
            st.metric("Price", f"¥{discount_data['normal']['avg_price']:.2f}/kg")
            st.metric("Transactions", f"{int(discount_data['normal']['transaction_count']):,}")
    
    with discount_tab2:
        st.markdown("### Best & Worst Days for Discounts")
        day_lift = day_discount['day_lift'].sort_values('quantity_lift_pct', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"🟢 Best Day: {discount_insights['best_discount_day']}")
            st.metric("Quantity Lift", f"+{discount_insights['best_discount_day_lift']:.1f}%")
        
        with col2:
            st.error(f"🔴 Worst Day: {discount_insights['worst_discount_day']}")
            st.metric("Quantity Lift", f"{discount_insights['worst_discount_day_lift']:.1f}%")
        
        st.markdown("### Daily Breakdown")
        st.dataframe(day_lift.sort_values('quantity_lift_pct', ascending=False), 
                     use_container_width=True, hide_index=True)
    
    with discount_tab3:
        st.markdown("### 🎉 Products That Explode with Discounts")
        st.markdown("These are your volume drivers - discount them strategically")
        
        top_products_discount = product_discount.head(5).copy()
        top_products_discount = top_products_discount[[
            'without_discount', 'with_discount', 'quantity_lift_pct',
            'revenue_with_discount', 'revenue_impact_pct', 'transactions_with_discount'
        ]].round(2)
        
        top_products_discount.columns = [
            'Qty w/o Discount (kg)', 'Qty w/ Discount (kg)', 'Lift %',
            'Avg Revenue (¥)', 'Revenue Impact %', 'Transactions'
        ]
        
        st.dataframe(top_products_discount, use_container_width=True)
        
        with st.expander("Strategy"):
            st.write("""
            ✅ *What to do:*
            - These products show strong volume response to discounts
            - Use discounts on these items during slow periods to drive traffic
            - Bundle them with low-discount items for margin protection
            """)
    
    with discount_tab4:
        st.markdown("### ⚠️ Products Where Discounts Hurt Revenue")
        st.markdown("These don't need discounts - they sell well at full price")
        
        worst_products_discount = product_discount.tail(5).copy()
        worst_products_discount = worst_products_discount[[
            'without_discount', 'with_discount', 'quantity_lift_pct',
            'revenue_with_discount', 'revenue_impact_pct', 'transactions_with_discount'
        ]].round(2)
        
        worst_products_discount.columns = [
            'Qty w/o Discount (kg)', 'Qty w/ Discount (kg)', 'Lift %',
            'Avg Revenue (¥)', 'Revenue Impact %', 'Transactions'
        ]
        
        st.dataframe(worst_products_discount, use_container_width=True)
        
        with st.expander("Strategy"):
            st.write("""
            🚫 *What NOT to do:*
            - Avoid discounting these products - they show weak or negative response
            - Customers buy them at full price anyway
            - Discounts only reduce margin without lifting volume
            - Save discounts for true volume drivers
            """)
    
    st.success(f"""
    *💡 Key Insight:* {
        'Discounts are highly effective - they drive volume without hurting revenue!' if discount_data['quantity_lift_pct'] > 10 and discount_data['revenue_impact_pct'] > 0
        else 'Discounts drive volume but reduce per-transaction revenue. Use strategically on high-margin products only.' if discount_data['quantity_lift_pct'] > 10
        else 'Discounts show weak volume lift. Reconsider discount strategy - they may not be worth the margin loss.'
    }
    """)

except Exception as e:
    st.warning(f"Discount analysis unavailable: {e}")

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
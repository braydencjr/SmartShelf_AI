import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.optimization import (
    optimize_inventory,
    recommend_pricing,
    calculate_resource_allocation
)
from models.forecasting import forecast_sales

st.set_page_config(page_title="Optimization", page_icon="⚙️", layout="wide")

st.title("⚙️ Operations Optimization")
st.markdown("### Data-driven recommendations for inventory, pricing, and resource allocation")

# Check for data
if 'df' not in st.session_state:
    st.warning("⚠️ No data loaded. Please go to the main page and load data first.")
    if st.button("← Go to Main Page"):
        st.switch_page("Main_Page.py")
    st.stop()

df = st.session_state['df']
daily_df = st.session_state['daily_df']

st.markdown("""
<style>
/* Make Streamlit tabs distribute evenly */
div[data-baseweb="tab-list"] {
    display: flex !important;
    justify-content: space-between !important;
    flex-wrap: wrap !important;
}

div[data-baseweb="tab"] {
    flex: 1 1 auto !important;
    text-align: center !important;
    max-width: none !important;
    padding: 0.75rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Inventory Optimization",
    "💰 Pricing Strategy",
    "📊 Resource Allocation",
    "📈 Combined Recommendations"
])

# Tab 1: Inventory Optimization
with tab1:
    st.markdown("## 📦 Inventory Optimization")
    st.markdown("Optimize stock levels to minimize costs while meeting demand.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **What this does:**
        - Calculates optimal stock levels per product
        - Identifies reorder points
        - Prioritizes products by revenue impact
        - Prevents stockouts and overstock
        """)
    
    with col2:
        lead_time = st.number_input(
        "Supplier Delivery Time (days)", 
        min_value=1, max_value=30, value=7,
        help="Average number of days your supplier takes to deliver an order. Higher = more stock needed to avoid stockouts."
        )

        

    
    if st.button("🔧 Calculate Inventory Optimization", type="primary"):
        with st.spinner("Optimizing inventory..."):
            
            # Generate forecast for future demand
            forecast_df, _ = forecast_sales(daily_df, periods=30)
            
            if forecast_df is not None:
                # Calculate optimization
                inventory_opt = optimize_inventory(df, forecast_df)
                
                # Add merged product names
                if 'Item Name' in df.columns:
                    item_names = df[['Item Code', 'Item Name']].drop_duplicates()
                    inventory_opt = inventory_opt.merge(item_names, on='Item Code', how='left')
                
                st.session_state['inventory_opt'] = inventory_opt
                st.success("✅ Inventory optimization complete!")
    
    if 'inventory_opt' in st.session_state:
        inventory_opt = st.session_state['inventory_opt']
        
        # Summary metrics
        st.markdown("### 📊 Optimization Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Products",
                len(inventory_opt)
            )
        
        with col2:
            st.metric(
                "High Priority Items",
                len(inventory_opt[inventory_opt['priority_score'] > 80]),
                delta="Top tier"
            )
        
        with col3:
            total_stock_value = (inventory_opt['recommended_stock'] * 
                               inventory_opt['avg_daily_demand']).sum()
            st.metric(
                "Est. Stock Value",
                f"¥{total_stock_value:,.0f}",
                delta="Optimized"
            )
        
        with col4:
            st.metric(
                "Avg Days Coverage",
                f"{lead_time} days",
                delta="Lead time"
            )
        
        # Priority products chart
        st.markdown("### 🎯 Product Priority Ranking")
        
        top_priority = inventory_opt.nlargest(15, 'priority_score')
        
        fig = px.bar(
            top_priority,
            x='priority_score',
            y='Item Name' if 'Item Name' in top_priority.columns else 'Item Code',
            orientation='h',
            title='Top 15 Priority Products',
            color='priority_score',
            color_continuous_scale='RdYlGn',
            labels={'priority_score': 'Priority Score', 'Item Name': 'Product'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed recommendations
        st.markdown("### 📋 Inventory Recommendations")
        
        # Add action column
        inventory_opt['Action'] = inventory_opt.apply(
            lambda row: '🔴 Urgent Restock' if row['priority_score'] > 80
            else '🟡 Monitor' if row['priority_score'] > 50
            else '🟢 Normal',
            axis=1
        )
        
        # Display columns
        display_cols = ['Item Code', 'Action', 'avg_daily_demand', 
                       'recommended_stock', 'reorder_point', 'priority_score']
        if 'Item Name' in inventory_opt.columns:
            display_cols.insert(1, 'Item Name')
        
        # Filter options
        filter_priority = st.multiselect(
            "Filter by priority:",
            ['🔴 Urgent Restock', '🟡 Monitor', '🟢 Normal'],
            default=['🔴 Urgent Restock', '🟡 Monitor']
        )
        
        if filter_priority:
            filtered_inventory = inventory_opt[inventory_opt['Action'].isin(filter_priority)]
        else:
            filtered_inventory = inventory_opt
        
        st.dataframe(
            filtered_inventory[display_cols].sort_values('priority_score', ascending=False),
            use_container_width=True
        )
        
        # Export
        col1, col2 = st.columns(2)
        
        with col1:
            csv = inventory_opt.to_csv(index=False)
            st.download_button(
                "📥 Download Full Inventory Report",
                csv,
                "inventory_optimization.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            urgent_items = inventory_opt[inventory_opt['priority_score'] > 80]
            if len(urgent_items) > 0:
                urgent_csv = urgent_items.to_csv(index=False)
                st.download_button(
                    "🔴 Download Urgent Items Only",
                    urgent_csv,
                    "urgent_restock.csv",
                    "text/csv",
                    use_container_width=True
                )

# Tab 2: Pricing Strategy
with tab2:
    st.markdown("## 💰 Pricing Strategy Optimization")
    st.markdown("Optimize prices to maximize revenue while maintaining competitiveness.")
    
    if st.button("💵 Analyze Pricing Strategy", type="primary"):
        with st.spinner("Analyzing pricing..."):
            
            pricing_analysis = recommend_pricing(df)
            
            # Add product names
            if 'Item Name' in df.columns:
                item_names = df[['Item Code', 'Item Name']].drop_duplicates()
                pricing_analysis = pricing_analysis.merge(item_names, on='Item Code', how='left')
            
            st.session_state['pricing_analysis'] = pricing_analysis
            st.success("✅ Pricing analysis complete!")
    
    if 'pricing_analysis' in st.session_state:
        pricing_analysis = st.session_state['pricing_analysis']
        
        # Summary
        st.markdown("### 📊 Pricing Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            increase_count = len(pricing_analysis[pricing_analysis['price_recommendation'].str.contains('Increase')])
            st.metric("Price Increase Opportunities", increase_count)
        
        with col2:
            maintain_count = len(pricing_analysis[pricing_analysis['price_recommendation'].str.contains('Maintain')])
            st.metric("Maintain Current Price", maintain_count)
        
        with col3:
            decrease_count = len(pricing_analysis[pricing_analysis['price_recommendation'].str.contains('Decrease')])
            st.metric("Price Decrease Needed", decrease_count)
        
        # Recommendations breakdown
        st.markdown("### 📈 Pricing Recommendations")
        
        rec_counts = pricing_analysis['price_recommendation'].value_counts()
        
        fig = px.pie(
            values=rec_counts.values,
            names=rec_counts.index,
            title='Pricing Recommendations Distribution',
            hole=0.4,
            color_discrete_sequence=['#2ecc71', '#3498db', '#e74c3c']
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed pricing table
        st.markdown("### 💰 Product Pricing Details")
        
        # Color code recommendations
        def color_recommendation(val):
            if 'Increase' in val:
                return 'background-color: #d4edda; color: #155724'
            elif 'Decrease' in val:
                return 'background-color: #f8d7da; color: #721c24'
            else:
                return 'background-color: #d1ecf1; color: #0c5460'
        
        display_cols = ['Item Code', 'avg_price', 'total_revenue', 
                       'revenue_per_price_unit', 'price_recommendation']
        if 'Item Name' in pricing_analysis.columns:
            display_cols.insert(1, 'Item Name')
        
        styled_df = pricing_analysis[display_cols].sort_values('total_revenue', ascending=False)
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Actionable insights
        st.markdown("### 💡 Key Insights")
        
        increase_items = pricing_analysis[pricing_analysis['price_recommendation'].str.contains('Increase')]
        decrease_items = pricing_analysis[pricing_analysis['price_recommendation'].str.contains('Decrease')]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(increase_items) > 0:
                st.success(f"""
                **🟢 Price Increase Opportunities ({len(increase_items)} items)**
                
                These products have strong demand relative to price and can support a 5% increase:
                
                Top candidates:
                {', '.join(increase_items.nlargest(5, 'total_revenue')['Item Name'].tolist() if 'Item Name' in increase_items.columns else increase_items.nlargest(5, 'total_revenue')['Item Code'].astype(str).tolist())}
                
                **Potential additional revenue:** ¥{(increase_items['total_revenue'].sum() * 0.05):,.2f}
                """)
        
        with col2:
            if len(decrease_items) > 0:
                st.warning(f"""
                **🟡 Price Decrease Needed ({len(decrease_items)} items)**
                
                These products are underperforming and may benefit from a 5% price reduction to boost volume:
                
                Priority items:
                {', '.join(decrease_items.nlargest(5, 'total_revenue')['Item Name'].tolist() if 'Item Name' in decrease_items.columns else decrease_items.nlargest(5, 'total_revenue')['Item Code'].astype(str).tolist())}
                """)
        
        # Export
        csv = pricing_analysis.to_csv(index=False)
        st.download_button(
            "📥 Download Pricing Strategy Report",
            csv,
            "pricing_strategy.csv",
            "text/csv",
            use_container_width=True
        )

# Tab 3: Resource Allocation
with tab3:
    st.markdown("## 📊 Resource Allocation Optimization")
    st.markdown("Allocate resources efficiently based on product contribution and forecast.")
    
    if st.button("📊 Calculate Resource Allocation", type="primary"):
        with st.spinner("Calculating allocation..."):
            
            # Generate forecast
            forecast_df, _ = forecast_sales(daily_df, periods=30)
            
            if forecast_df is not None:
                allocation = calculate_resource_allocation(df, forecast_df)
                
                # Add product names
                if 'Item Name' in df.columns:
                    item_names = df[['Item Code', 'Item Name']].drop_duplicates()
                    allocation = allocation.merge(item_names, on='Item Code', how='left')
                
                st.session_state['allocation'] = allocation
                st.success("✅ Resource allocation complete!")
    
    if 'allocation' in st.session_state:
        allocation = st.session_state['allocation']
        
        # Pareto analysis
        st.markdown("### 📊 Pareto Analysis (80/20 Rule)")
        
        allocation['cumulative_pct'] = allocation['allocation_pct'].cumsum()
        
        top_80 = allocation[allocation['cumulative_pct'] <= 80]
        
        st.info(f"""
        **80/20 Insight:**
        - **{len(top_80)} products** ({(len(top_80)/len(allocation)*100):.1f}% of catalog) generate **80%** of revenue
        - Focus resources on these high-impact products
        - Remaining {len(allocation) - len(top_80)} products generate only 20% of revenue
        """)
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=allocation.head(20)['Item Name'] if 'Item Name' in allocation.columns else allocation.head(20)['Item Code'].astype(str),
            y=allocation.head(20)['allocation_pct'],
            name='Revenue %',
            marker_color='#667eea'
        ))
        
        fig.add_trace(go.Scatter(
            x=allocation.head(20)['Item Name'] if 'Item Name' in allocation.columns else allocation.head(20)['Item Code'].astype(str),
            y=allocation.head(20)['cumulative_pct'],
            name='Cumulative %',
            yaxis='y2',
            line=dict(color='#f093fb', width=3),
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title='Product Contribution Analysis (Top 20)',
            xaxis_title='Product',
            yaxis_title='Revenue Contribution (%)',
            yaxis2=dict(
                title='Cumulative %',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Resource allocation recommendations
        st.markdown("### 🎯 Resource Allocation Recommendations")
        
        # Categorize products
        allocation['Category_Priority'] = pd.cut(
            allocation['allocation_pct'],
            bins=[0, 1, 5, 100],
            labels=['Low Priority', 'Medium Priority', 'High Priority']
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_priority = allocation[allocation['Category_Priority'] == 'High Priority']
            st.markdown("#### 🔴 High Priority")
            st.metric("Products", len(high_priority))
            st.metric("Revenue %", f"{high_priority['allocation_pct'].sum():.1f}%")
            st.info("**Recommendation:** Max resources, premium shelf space, priority restocking")
        
        with col2:
            med_priority = allocation[allocation['Category_Priority'] == 'Medium Priority']
            st.markdown("#### 🟡 Medium Priority")
            st.metric("Products", len(med_priority))
            st.metric("Revenue %", f"{med_priority['allocation_pct'].sum():.1f}%")
            st.info("**Recommendation:** Standard resources, regular monitoring")
        
        with col3:
            low_priority = allocation[allocation['Category_Priority'] == 'Low Priority']
            st.markdown("#### 🟢 Low Priority")
            st.metric("Products", len(low_priority))
            st.metric("Revenue %", f"{low_priority['allocation_pct'].sum():.1f}%")
            st.info("**Recommendation:** Minimal resources, consider discontinuation")
        
        # Detailed table
        st.markdown("### 📋 Allocation Details")
        
        display_cols = ['Item Code', 'TotalSales', 'allocation_pct', 
                       'cumulative_pct', 'Category_Priority']
        if 'Item Name' in allocation.columns:
            display_cols.insert(1, 'Item Name')
        
        st.dataframe(allocation[display_cols], use_container_width=True)
        
        # Export
        csv = allocation.to_csv(index=False)
        st.download_button(
            "📥 Download Resource Allocation Report",
            csv,
            "resource_allocation.csv",
            "text/csv",
            use_container_width=True
        )

# Tab 4: Combined Recommendations
with tab4:
    st.markdown("## 📈 Integrated Optimization Recommendations")
    st.markdown("Comprehensive recommendations combining all optimization insights.")
    
    if st.button("🎯 Generate Master Recommendations", type="primary"):
        with st.spinner("Creating comprehensive recommendations..."):
            
            # Run all optimizations
            forecast_df, _ = forecast_sales(daily_df, periods=30)
            
            if forecast_df is not None:
                inventory_opt = optimize_inventory(df, forecast_df)
                pricing_analysis = recommend_pricing(df)
                allocation = calculate_resource_allocation(df, forecast_df)
                
                st.session_state['master_recs'] = {
                    'inventory': inventory_opt,
                    'pricing': pricing_analysis,
                    'allocation': allocation
                }
                
                st.success("✅ Master recommendations generated!")
    
    if 'master_recs' in st.session_state:
        recs = st.session_state['master_recs']
        
        st.markdown("### 📊 Executive Dashboard")
        
        # High-level metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            urgent_stock = len(recs['inventory'][recs['inventory']['priority_score'] > 80])
            st.metric("Urgent Stock Items", urgent_stock, delta="Needs attention")
        
        with col2:
            price_opportunities = len(recs['pricing'][recs['pricing']['price_recommendation'].str.contains('Increase')])
            potential_revenue = recs['pricing'][recs['pricing']['price_recommendation'].str.contains('Increase')]['total_revenue'].sum() * 0.05
            st.metric("Price Increase Opportunities", price_opportunities, delta=f"+¥{potential_revenue:,.0f}")
        
        with col3:
            top_20_pct = recs['allocation'].head(int(len(recs['allocation']) * 0.2))['allocation_pct'].sum()
            st.metric("Top 20% Revenue Share", f"{top_20_pct:.1f}%", delta="Focus area")
        
        with col4:
            forecast_total = forecast_df['forecasted_sales'].sum() if 'forecast_df' in st.session_state else 0
            st.metric("30-Day Forecast", f"¥{forecast_total:,.0f}", delta="Expected")
        
        # Priority actions
        st.markdown("### 🎯 Priority Actions")
        
        # Combine insights
        urgent_items = recs['inventory'].nlargest(10, 'priority_score')[['Item Code', 'priority_score']]
        price_inc = recs['pricing'][recs['pricing']['price_recommendation'].str.contains('Increase')].nlargest(5, 'total_revenue')
        top_revenue = recs['allocation'].head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Immediate Actions Required")
            
            st.markdown("**1. Critical Inventory Restocking:**")
            for idx, row in urgent_items.head(5).iterrows():
                st.markdown(f"- Item {row['Item Code']}: Priority Score {row['priority_score']:.0f}")
            
            st.markdown("**2. Price Optimization:**")
            st.markdown(f"- Implement 5% increase on {len(price_inc)} high-performing products")
            st.markdown(f"- Potential revenue gain: ¥{(price_inc['total_revenue'].sum() * 0.05):,.2f}")
        
        with col2:
            st.markdown("#### 🟡 Strategic Initiatives")
            
            st.markdown("**1. Resource Reallocation:**")
            st.markdown(f"- Focus 80% of resources on top {len(top_revenue)} products")
            st.markdown(f"- These drive {top_revenue['allocation_pct'].sum():.1f}% of revenue")
            
            st.markdown("**2. Operational Efficiency:**")
            st.markdown("- Streamline inventory for low-priority items")
            st.markdown("- Automate reordering for high-priority products")
        
        # Implementation roadmap
        st.markdown("### 🗓️ Implementation Roadmap")
        
        roadmap_data = {
            'Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
            'Actions': [
                '🔴 Urgent restocking, Price analysis review',
                '💰 Implement price changes, Monitor response',
                '📊 Reallocate resources, Update inventory systems',
                '📈 Review results, Adjust strategy'
            ],
            'Expected Impact': [
                'Prevent stockouts',
                '+3-5% revenue',
                'Improved efficiency',
                'Sustained growth'
            ]
        }
        
        st.table(pd.DataFrame(roadmap_data))
        
        # Export comprehensive report
        st.markdown("### 📥 Export Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            inventory_csv = recs['inventory'].to_csv(index=False)
            st.download_button(
                "📦 Inventory Report",
                inventory_csv,
                "inventory_master.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            pricing_csv = recs['pricing'].to_csv(index=False)
            st.download_button(
                "💰 Pricing Report",
                pricing_csv,
                "pricing_master.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col3:
            allocation_csv = recs['allocation'].to_csv(index=False)
            st.download_button(
                "📊 Allocation Report",
                allocation_csv,
                "allocation_master.csv",
                "text/csv",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚙️ Optimization recommendations should be implemented gradually and monitored for effectiveness</p>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import json
from pathlib import Path
import textwrap

# Assuming models and AI helper files are located correctly
sys.path.append(str(Path(__file__).parent.parent))


from models.forecasting_advanced import (
    prophet_forecast,
    xgboost_forecast,
    optimized_sarimax_forecast,
    smart_ensemble_forecast,
    calculate_forecast_accuracy
)
from ai.insights_generator import generate_forecast_insights, generate_custom_explanation

st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")

st.title("Business Insight Forecast")
st.markdown("### Peeking Into the Future 👀")

# Check for data
if 'df' not in st.session_state:
    st.warning("⚠️ No data loaded. Please go to the main page and load data first.")
    if st.button("← Go to Main Page"):
        st.switch_page("Main_Page.py")
    st.stop()

daily_df = st.session_state['daily_df']


with st.container():
    col1, col2 = st.columns([2,1])

    # -------------------------
    # LEFT SIDE (SLIDER + BUTTON)
    # -------------------------
    with col1:

        # Forecast period slider
        forecast_days = st.slider(
            "Forecast Period (days)",
            min_value=1,
            max_value=30,
            value=7,
            step=1
        )

        # --- 3-column row: checkbox, checkbox, button ---
        c1, c2, c3 = st.columns([1, 1, 1.2])

        with c1:
            show_confidence = st.checkbox(
                "Show Expected Intervals",
                value=True,
                help="Displays a shaded area showing expected range."
            )

        with c2:
            show_historical = st.checkbox(
                "Show Historical Data",
                value=True,
                help="Displays past sales."
            )

        with c3:
            st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
            if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
                with st.spinner("Generating forecast... This may take a moment."):
                    forecast_df = optimized_sarimax_forecast(daily_df, periods=forecast_days)

                    if forecast_df is not None:
                        st.session_state['forecast_df'] = forecast_df
                        st.session_state['forecast_method'] = "SARIMAX"
                        st.success("✅ Forecast generated successfully using SARIMAX!")
                    else:
                        st.error("❌ Forecast generation failed. Please check your data.")

    # -------------------------
    # RIGHT SIDE (FORECAST PARAMETER CARD)
    # -------------------------
    # EVERYTHING MUST BE INSIDE THIS BLOCK
    with col2:
        # st.markdown("<br>", unsafe_allow_html=True) # REMOVED for upward alignment

        # Native Streamlit Container for the card
        with st.container(border=True): 
            st.markdown("#### 📘 **Forecast Parameters**")
            
            # Use st.markdown or st.write for content lines
            st.markdown(f"**Forecast Horizon:** {forecast_days} days") # Hardcoding 7 days based on image
            st.markdown("**Confidence Interval:** 95%")
            st.markdown("**Data Filter Applied:** All Products")

            # Disclaimer 
            st.caption("⚠️ Historical patterns do not guarantee future performance.")
        # Removed the problematic st.markdown(html_card, ...) line

# Display forecast
if 'forecast_df' in st.session_state:
    forecast_df = st.session_state['forecast_df']
    method = st.session_state.get('forecast_method', 'Unknown')
    
    # Metrics
    st.markdown("## 📊 Forecast Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Forecast Period",
            f"{len(forecast_df)} days",
            delta=f"{forecast_df['Date'].min().date()} to {forecast_df['Date'].max().date()}"
        )
    
    with col2:
        st.metric(
            "Avg Forecasted Sales",
            f"¥{forecast_df['forecasted_sales'].mean():,.2f}",
            delta=f"{((forecast_df['forecasted_sales'].mean() / daily_df['daily_sales'].mean() - 1) * 100):+.1f}% vs historical"
        )
    
    with col3:
        st.metric(
            "Total Forecasted Revenue",
            f"¥{forecast_df['forecasted_sales'].sum():,.2f}",
            delta="Projected"
        )
    
    with col4:
        st.metric(
            "Method Used",
            method,
            delta="AI Model"
        )
    
    # Visualization
    st.markdown("## 📈 Forecast Visualization")
    
    fig = go.Figure()
    
    # Historical data
    if show_historical:
        fig.add_trace(go.Scatter(
            x=daily_df['Date'],
            y=daily_df['daily_sales'],
            mode='lines',
            name='Historical Sales',
            line=dict(color='#667eea', width=2)
        ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['forecasted_sales'],
        mode='lines',
        name='Forecasted Sales',
        line=dict(color='#f093fb', width=3, dash='dash')
    ))
    
    # Confidence intervals
    if show_confidence:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['upper_bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['lower_bound'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(240, 147, 251, 0.2)',
            fill='tonexty',
            showlegend=True
        ))
    
    fig.update_layout(
        title=f'Sales Forecast - {method}',
        xaxis_title='Date',
        yaxis_title='Sales (RMB)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast accuracy (using last 30 days as test)
    st.markdown("## 🎯 Model Performance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Accuracy Metrics")
        
        # Calculate accuracy on last 30 days
        test_days = min(30, len(daily_df) // 3)
        train_df = daily_df.iloc[:-test_days]
        test_df = daily_df.iloc[-test_days:]
        
        try:
            if method == "Smart Ensemble":
                test_forecast = smart_ensemble_forecast(train_df, periods=test_days)
            elif method == "Prophet":
                test_forecast = prophet_forecast(train_df, periods=test_days)
            elif method == "XGBoost":
                test_forecast = xgboost_forecast(train_df, periods=test_days)
            elif method == "SARIMAX":
                test_forecast = optimized_sarimax_forecast(train_df, periods=test_days)
            
            
            if test_forecast is not None and len(test_forecast) == len(test_df):
                accuracy = calculate_forecast_accuracy(
                    test_df['daily_sales'].values,
                    test_forecast['forecasted_sales'].values
                )
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("MAPE", f"{accuracy['MAPE']:.2f}%", help="Mean Absolute Percentage Error")
                
                with col_b:
                    st.metric("RMSE", f"{accuracy['RMSE']:.2f}", help="Root Mean Square Error")
                
                with col_c:
                    st.metric("MAE", f"{accuracy['MAE']:.2f}", help="Mean Absolute Error")
                
                # Accuracy interpretation
                if accuracy['MAPE'] < 10:
                    st.success("🎯 Excellent forecast accuracy! Model performs very well.")
                elif accuracy['MAPE'] < 20:
                    st.info("✅ Good forecast accuracy. Reliable for planning.")
                elif accuracy['MAPE'] < 30:
                    st.warning("⚠️ Moderate accuracy. Use with caution for critical decisions.")
                else:
                    st.error("❌ Low accuracy. Consider more data or different parameters.")
        
        except Exception as e:
            st.warning(f"Could not calculate accuracy metrics: {e}")
    
    with col2:
        st.markdown("### 📊 Forecast Statistics")
        st.metric("Min Forecast", f"¥{forecast_df['forecasted_sales'].min():,.2f}")
        st.metric("Max Forecast", f"¥{forecast_df['forecasted_sales'].max():,.2f}")
        st.metric("Std Deviation", f"¥{forecast_df['forecasted_sales'].std():,.2f}")
        st.metric("Trend", 
                 "Upward 📈" if forecast_df['forecasted_sales'].iloc[-1] > forecast_df['forecasted_sales'].iloc[0] 
                 else "Downward 📉")
    st.markdown("---")
    # AI Insights
    st.markdown("## 💡 AI-Powered Insights")

if 'forecast_df' in st.session_state:
    forecast_df = st.session_state['forecast_df']
    daily_df = st.session_state['daily_df']

    if st.button("Generate AI Analysis", type="primary"):

    # 1️⃣ Show the forecast chart first
     fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['forecasted_sales'],
        mode='lines+markers',
        name='Forecasted Sales'
    ))
    fig.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    # 2️⃣ Call AI for bullet points
    with st.spinner("AI is analyzing the forecast..."):
        ai_response = generate_forecast_insights(
            forecast_df,
            daily_df['daily_sales'].mean(),
            df=st.session_state.get('df', None)
        )

        if ai_response:
            ai_response = ai_response.strip()
            if ai_response.startswith("```") and ai_response.endswith("```"):
                ai_response = '\n'.join(ai_response.split('\n')[1:-1]).strip()

        # 3️⃣ Parse JSON safely
        bullets_data = []
        try:
            raw_data = json.loads(ai_response)

            # Normalize: ensure each item is a dict with required keys
            for item in raw_data:
                if isinstance(item, dict):
                    bullets_data.append({
                        "bullet": item.get("text", "Legacy Insight"),
                        "category": item.get("category", "Insight"),
                        "graph_suggestion": item.get("graph_suggestion", "Full Forecast")
                    })
                elif isinstance(item, str):
                    bullets_data.append({
                        "bullet": item,
                        "category": "Insight",
                        "graph_suggestion": "Full Forecast"
                    })
        except json.JSONDecodeError:
            st.error("❌ AI returned invalid JSON.")
            st.code(ai_response, language='json')
            bullets_data = []

        # 4️⃣ Display bullets
        for bullet_item in bullets_data:
            bullet_text = bullet_item["bullet"]
            bullet_category = bullet_item["category"]
            bullet_graph = bullet_item["graph_suggestion"]

            icon = {
                "Observation": "👁️",
                "Potential Risk": "⚠️",
                "Recommended Action": "🚀",
                "Fallback": "💬",
                "Error": "❌",
                "Insight": "💡"
            }.get(bullet_category, "💡")

            with st.expander(f"{icon} **{bullet_category}**: {bullet_text} 🔍"):
                detailed_explanation = generate_custom_explanation(bullet_text, forecast_df)
                st.markdown(detailed_explanation)
                st.markdown("---")


    st.markdown("---")
    # Forecast table
    st.markdown("## 📋 Detailed Forecast Data")
    
    with st.expander("View Forecast Table"):
        st.dataframe(forecast_df, use_container_width=True)
        
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            "📥 Download Forecast CSV",
            csv,
            "sales_forecast.csv",
            "text/csv",
            use_container_width=True
        )
    

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Forecasts are based on historical patterns and should be used as guidance alongside business judgment</p>
</div>
""", unsafe_allow_html=True)
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


def forecast_st_card(title, value, subtitle=None):
    """Render a square card for forecast metrics."""
    CARD_BG = "#0D3B66"
    CARD_COLOR = "#FFFFFF"
    CARD_PADDING = "12px"
    CARD_BORDER_RADIUS = "12px"
    CARD_SIZE = "180px"
    TITLE_SIZE = "13px"
    VALUE_SIZE = "20px"
    SUBTITLE_SIZE = "12px"

    subtitle_html = f"<div style='font-size:{SUBTITLE_SIZE}; color:#E0E0E0; margin-top:6px'>{subtitle}</div>" if subtitle else ""

    st.markdown(
        f"""
        <div style='
            background-color: {CARD_BG};
            color: {CARD_COLOR};
            width: {CARD_SIZE};
            height: {CARD_SIZE};
            padding: {CARD_PADDING};
            border-radius: {CARD_BORDER_RADIUS};
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            margin-bottom: 12px;
        '>
            <div style='font-size:{TITLE_SIZE}; opacity:0.9; margin-bottom:6px'>{title}</div>
            <div style='font-size:{VALUE_SIZE}; font-weight:700'>{value}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True
    )



# ---------------------------
# Controls + Right summary panel
# ---------------------------
# ---------------------------
# Controls + Right summary panel
# ---------------------------
with st.container():
    left_col, right_col = st.columns([2, 1])

    # LEFT CONTROLS
    with left_col:
        forecast_days = st.slider(
            "Forecast Period (days)",
            min_value=1, max_value=30, value=7, step=1
        )

        c1, c2, c3 = st.columns([1, 1, 1.2])
        with c1:
            show_confidence = st.checkbox("Show Expected Intervals", value=True)
        with c2:
            show_historical = st.checkbox("Show Historical Data", value=True)
        with c3:
            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
            if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
                with st.spinner("Generating forecast..."):
                    try:
                        forecast_df = optimized_sarimax_forecast(daily_df, periods=forecast_days)
                        if forecast_df is None or forecast_df.empty:
                            st.error("❌ Forecast generation returned no data.")
                        else:
                            st.session_state['forecast_df'] = forecast_df
                            st.session_state['forecast_method'] = "SARIMAX"
                            st.success("✅ Forecast generated successfully!")
                    except Exception as e:
                        st.error(f"❌ Forecast failed: {e}")

    # RIGHT CONTROLS - MOVED THE BORDERED BOX HERE
   # RIGHT CONTROLS
# RIGHT CONTROLS
with right_col:
    with st.container(border=True):
        st.markdown("#### 📘 Forecast Parameter")
        st.markdown(f"**Forecast Horizon:** {forecast_days} days")

        # --- Dynamic Accuracy / MAPE & MAE ONLY if forecast exists ---
        if 'forecast_df' in st.session_state:
            forecast_df = st.session_state['forecast_df']

            try:
                # ensure enough history
                if isinstance(daily_df, pd.DataFrame) and len(daily_df) >= 10:
                    test_days = min(30, max(3, len(daily_df) // 3))
                    train_df = daily_df.iloc[:-test_days]
                    test_df = daily_df.iloc[-test_days:]

                    acc_text = "MAPE: N/A  •  MAE: N/A"
                    acc_interpretation = None

                    if len(train_df) >= 3 and len(test_df) >= 1:
                        test_forecast = optimized_sarimax_forecast(train_df, periods=len(test_df))
                        if test_forecast is not None and len(test_forecast) == len(test_df):
                            accuracy = calculate_forecast_accuracy(
                                test_df['daily_sales'].values,
                                test_forecast['forecasted_sales'].values
                            )
                            mape = accuracy.get('MAPE', None)
                            mae = accuracy.get('MAE', None)

                            if isinstance(mape, (float, int)) and isinstance(mae, (float, int)):
                                acc_text = f"MAPE: {mape:.2f}%  •  MAE: ¥{mae:.2f}"

                                # optional interpretation
                                if mape < 10:
                                    acc_interpretation = ("🎯 Excellent accuracy — model performs very well.", "success")
                                elif mape < 20:
                                    acc_interpretation = ("✅ Good accuracy — reliable for planning.", "info")
                                elif mape < 30:
                                    acc_interpretation = ("⚠️ Moderate accuracy — use with caution.", "warning")
                                else:
                                    acc_interpretation = ("❌ Low accuracy — consider more data or different parameters.", "error")

                    # display accuracy
                    st.markdown(acc_text)
                    if acc_interpretation:
                        text, level = acc_interpretation
                        if level == "success":
                            st.success(text)
                        elif level == "info":
                            st.info(text)
                        elif level == "warning":
                            st.warning(text)
                        else:
                            st.error(text)
                else:
                    st.caption("Not enough historical data to compute accuracy.")
            except Exception as e:
                st.warning(f"Could not compute accuracy: {e}")
        else:
            st.caption("Forecast not yet generated. Press 🔮 Generate Forecast.")

        
# ---------------------------
# Forecast cards + visualization
# ---------------------------
if 'forecast_df' in st.session_state:
    forecast_df = st.session_state['forecast_df']
    method = st.session_state.get('forecast_method', 'Unknown')

    st.markdown("## 📊 Forecast Summary")

    # Compute metrics
    try:
        forecast_period = f"{len(forecast_df)} days"
        forecast_period_subtitle = f"{forecast_df['Date'].min().date()} to {forecast_df['Date'].max().date()}"
    except Exception:
        forecast_period = f"{len(forecast_df)} days"
        forecast_period_subtitle = ""

    try:
        avg_sales_val = float(forecast_df['forecasted_sales'].mean())
        avg_sales = f"¥{avg_sales_val:,.2f}"
        avg_sales_subtitle = f"{((avg_sales_val / daily_df['daily_sales'].mean() - 1) * 100):+.1f}% vs historical"
    except Exception:
        avg_sales = "N/A"
        avg_sales_subtitle = None

    try:
        total_rev_val = float(forecast_df['forecasted_sales'].sum())
        total_revenue = f"¥{total_rev_val:,.2f}"
    except Exception:
        total_revenue = "N/A"
    total_revenue_subtitle = "Projected"

    try:
        min_forecast = f"¥{float(forecast_df['forecasted_sales'].min()):,.2f}"
        max_forecast = f"¥{float(forecast_df['forecasted_sales'].max()):,.2f}"
        std_dev = f"¥{float(forecast_df['forecasted_sales'].std()):,.2f}"
        trend = "Upward 📈" if forecast_df['forecasted_sales'].iloc[-1] > forecast_df['forecasted_sales'].iloc[0] else "Downward 📉"
    except Exception:
        min_forecast = max_forecast = std_dev = trend = "N/A"

    metrics = [
        ("Forecast Period", forecast_period, forecast_period_subtitle),
        ("Total Forecasted Revenue", total_revenue, total_revenue_subtitle),
        ("Avg Forecasted Sales", avg_sales, avg_sales_subtitle),
        ("Min Forecast", min_forecast, None),
        ("Max Forecast", max_forecast, None),
        ("Trend", trend, None)
    ]

    cols_per_row = 3
    for i in range(0, len(metrics), cols_per_row):
        row = metrics[i:i + cols_per_row]
        cols = st.columns(len(row))
        for col, (title, value, subtitle) in zip(cols, row):
            with col:
                forecast_st_card(title, value, subtitle)

    # ---------------- Visualization ----------------
    st.markdown("## 📈 Forecast Visualization")
    fig = go.Figure()

    if show_historical:
        fig.add_trace(go.Scatter(
            x=daily_df['Date'],
            y=daily_df['daily_sales'],
            mode='lines',
            name='Historical Sales',
            line=dict(width=2)
        ))

    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['forecasted_sales'],
        mode='lines',
        name='Forecasted Sales',
        line=dict(width=3, dash='dash')
    ))

    if show_confidence and 'upper_bound' in forecast_df.columns and 'lower_bound' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['upper_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['lower_bound'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(240,147,251,0.15)',
            name='Confidence'
        ))

    fig.update_layout(
        title=f"Sales Forecast - {method}",
        xaxis_title="Date",
        yaxis_title="Sales (RMB)",
        template="plotly_white",
        hovermode="x unified",
        height=520
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
if 'forecast_df' in st.session_state:
    forecast_df = st.session_state['forecast_df']
    daily_df = st.session_state['daily_df']

    # --- AI insights header ---
    st.markdown("## 💡 AI-Powered Insights")

    # --- Button inside the section ---
    if st.button("Generate AI Analysis", type="primary"):
        with st.spinner("AI is analyzing the forecast..."):
            try:
                ai_response = generate_forecast_insights(
                    forecast_df,
                    daily_df['daily_sales'].mean(),
                    df=st.session_state.get('df', None)
                )

                if not ai_response:
                    st.error("AI returned an empty response.")
                else:
                    # Display clean text (not JSON)
                    st.markdown(ai_response)

            except Exception as e:
                st.error(f"AI analysis failed: {e}")


    st.markdown("---")
    st.markdown("## 📋 Detailed Forecast Data")
    with st.expander("View Forecast Table"):
        st.dataframe(forecast_df, use_container_width=True)
        csv = forecast_df.to_csv(index=False)
        st.download_button("📥 Download Forecast CSV", csv, "sales_forecast.csv", "text/csv", use_container_width=True)

else:
    st.info("No forecast available. Press 'Generate Forecast' to create one.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666'>
    <p>Forecasts are based on historical patterns and should be used as guidance alongside business judgement.</p>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path
import sys

# Append parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.anomaly_detection import (
    detect_anomalies_zscore,
    detect_anomalies_isolation_forest,
    analyze_anomaly_patterns
)
from ai.insights_generator import generate_anomaly_explanation, generate_anomaly_bullet_explanation

st.set_page_config(page_title="Anomaly Detection", page_icon="🚨", layout="wide")

st.title("🚨 Anomaly Detection & Risk Alerts")
st.markdown("### Personal AI-powered risk spotter")

# Ensure data loaded
if 'df' not in st.session_state:
    st.warning("⚠️ No data loaded. Please go to the main page and load data first.")
    if st.button("← Go to Main Page"):
        st.switch_page("app.py")
    st.stop()

daily_df = st.session_state['daily_df']

# --- Detect anomalies ---
if st.button("🚨 Detect Anomalies", type="primary"):
    with st.spinner("Analyzing data for anomalies..."):
        anomalies = detect_anomalies_isolation_forest(daily_df)
        method = "Isolation Forest (ML)"

        if anomalies.empty:
            anomalies = detect_anomalies_zscore(daily_df)
            method = "Z-Score (Statistical)"

        mean = daily_df['daily_sales'].mean()

        if method == "Z-Score (Statistical)":
            std = daily_df['daily_sales'].std()
            daily_df['z_score'] = (daily_df['daily_sales'] - mean) / std
            anomalies = daily_df[daily_df.index.isin(anomalies.index)].copy()

        anomalies['anomaly_type'] = anomalies['daily_sales'].apply(
            lambda x: 'High (Good)' if x > mean else 'Low (Bad)'
        )

        st.session_state['anomalies'] = anomalies
        st.session_state['detection_method'] = method
        st.session_state['normal_avg'] = mean
        st.success(f"✅ Detection complete! Found {len(anomalies)} anomalies using {method}.")

        # Display anomalies
if 'anomalies' in st.session_state:
    anomalies = st.session_state['anomalies']
    method = st.session_state.get('detection_method', 'Unknown')
    normal_avg = st.session_state.get('normal_avg', daily_df['daily_sales'].mean())
    
    st.markdown("## 📊 Anomaly Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Anomalies Detected", len(anomalies), delta=f"{(len(anomalies)/len(daily_df)*100):.1f}% of days")
    with col2:
        # Calculate avg anomaly value only if anomalies exist
        avg_anomaly = anomalies['daily_sales'].mean() if len(anomalies) > 0 else normal_avg
        delta_val = f"{((avg_anomaly/normal_avg-1)*100):+.1f}% vs normal" if normal_avg != 0 else "+0.0%"
        st.metric("Avg Anomaly Value", f"¥{avg_anomaly:,.2f}", delta=delta_val)
    with col3:
        max_val = anomalies['daily_sales'].max() if len(anomalies) > 0 else 0
        st.metric("Highest Anomaly", f"¥{max_val:,.2f}", delta="Peak")
    with col4:
        min_val = anomalies['daily_sales'].min() if len(anomalies) > 0 else 0
        st.metric("Lowest Anomaly", f"¥{min_val:,.2f}", delta="Trough")

    st.markdown("## 📈 Anomaly Visualization")
    fig = go.Figure()
    
    # Ensure anomalies' index is the Date for proper filtering
    anomalies_dates = anomalies.index.values if anomalies.index.name == 'Date' else anomalies['Date'].values
    normal_data = daily_df[~daily_df['Date'].isin(anomalies_dates)].copy()

    fig.add_trace(go.Scatter(
        x=normal_data['Date'],
        y=normal_data['daily_sales'],
        mode='lines',
        name='Normal Sales',
        line=dict(color='#667eea', width=2),
        hovertemplate='Date: %{x}<br>Sales: ¥%{y:,.2f}<extra></extra>'
    ))

    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies['Date'],
            y=anomalies['daily_sales'],
            mode='markers',
            name='Anomalies',
            marker=dict(
                size=12,
                color=anomalies['anomaly_type'].map({'High (Good)':'green','Low (Bad)':'red'}),
                symbol='x',
                line=dict(width=2, color='black')
            ),
            hovertemplate='Date: %{x}<br>Sales: ¥%{y:,.2f}<br>%{customdata}<extra></extra>',
            customdata=anomalies['anomaly_type']
        ))

    fig.update_layout(
        title=f'Sales with Anomalies Highlighted - {method}',
        xaxis_title='Date',
        yaxis_title='Sales (RMB)',
        hovermode='closest',
        template='plotly_white',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Analysis & patterns
    st.markdown("## 🔍 Anomaly Analysis")
    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("### Anomaly Distribution")
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=daily_df['daily_sales'], name='All Days', opacity=0.7, marker_color='#667eea', nbinsx=50
        ))
        hist_fig.add_trace(go.Histogram(
            x=anomalies['daily_sales'], name='Anomalies', opacity=0.7,
            marker_color='red',  # Use one color for histogram to avoid complexity
            nbinsx=20
        ))
        hist_fig.update_layout(
            title='Sales Distribution: Normal vs Anomalies',
            xaxis_title='Sales (RMB)',
            yaxis_title='Frequency',
            barmode='overlay',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Pattern Analysis")
        # NOTE: analyze_anomaly_patterns needs to be implemented to use this part
        patterns = analyze_anomaly_patterns(anomalies) 
        if isinstance(patterns, dict) and 'total_anomalies' in patterns:
            st.metric("Total Anomalies", patterns['total_anomalies'])
            st.metric("Avg Value", f"¥{patterns['avg_anomaly_value']:,.2f}")
            st.metric("Max Value", f"¥{patterns['max_anomaly_value']:,.2f}")
            st.metric("Min Value", f"¥{patterns['min_anomaly_value']:,.2f}")
            if 'most_common_day' in patterns and patterns['most_common_day'] is not None:
                days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
                st.metric("Most Common Day", days[patterns['most_common_day']])
        else:
            st.info("No common patterns found or `analyze_anomaly_patterns` unavailable.")


# --- Prepare anomalies_sorted ---
if 'anomalies' in st.session_state and not st.session_state['anomalies'].empty:
    anomalies_sorted = st.session_state['anomalies'].sort_values('Date', ascending=False)
    anomalies_sorted['Severity'] = anomalies_sorted['anomaly_type'].apply(
        lambda x: '🟢 Good' if x=='High (Good)' else '🔴 Warning'
    )
    anomalies_sorted['Date_display'] = anomalies_sorted['Date'].dt.date
    st.session_state['anomalies_sorted'] = anomalies_sorted
else:
    anomalies_sorted = pd.DataFrame()

# --- Display anomaly table ---
if not anomalies_sorted.empty:
    display_cols = ['Date_display','daily_sales','Severity','anomaly_type']
    if 'z_score' in anomalies_sorted.columns:
        display_cols.append('z_score')

    st.markdown("## 📋 Anomaly Details")
    st.dataframe(
        anomalies_sorted[display_cols].rename(columns={'Date_display': 'Date', 'daily_sales': 'Sales'}),
        use_container_width=True
    )
else:
    st.info("No anomalies detected to display.")

# --- AI Insights ---
if not anomalies_sorted.empty:
    st.markdown("## 💡 AI-Powered Anomaly Explanation")
    normal_avg = st.session_state.get('normal_avg', anomalies_sorted['daily_sales'].mean())

    selected_date = st.selectbox(
        "Select an anomaly date to analyze:",
        anomalies_sorted['Date_display'].tolist()
    )

    if st.button("🤖 Get AI Explanation", type="primary"):
        selected_anomaly = anomalies_sorted[anomalies_sorted['Date_display'] == selected_date].iloc[0]

        with st.spinner("AI is analyzing this anomaly..."):
            ai_response = generate_anomaly_explanation(
                selected_date.strftime('%Y-%m-%d'),
                selected_anomaly['daily_sales'],
                normal_avg
            )

            # Clean AI response
            if ai_response:
                ai_response = ai_response.strip()
                if ai_response.startswith("```") and ai_response.endswith("```"):
                    ai_response = '\n'.join(ai_response.split('\n')[1:-1]).strip()

            # Parse JSON
            try:
                bullets_data = json.loads(ai_response)
                if isinstance(bullets_data, list) and bullets_data and 'text' in bullets_data[0]:
                    bullets_data = [{
                        "bullet": item.get("text", "Legacy Insight"),
                        "category": "Insight",  # default category
                        "graph_suggestion": item.get("graph_suggestion", "")
                    } for item in bullets_data]
            except json.JSONDecodeError:
                st.error("❌ AI failed to produce structured analysis (JSON error). Check raw output below.")
                st.code(ai_response, language='json')
                bullets_data = []

        # --- 1️⃣ Show context chart ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=anomalies_sorted['Date'],
            y=anomalies_sorted['daily_sales'],
            mode='lines+markers',
            name='Daily Sales'
        ))
        fig.add_trace(go.Scatter(
            x=[selected_anomaly['Date']],
            y=[selected_anomaly['daily_sales']],
            mode='markers',
            marker=dict(color='red', size=12),
            name='Selected Anomaly'
        ))
        fig.update_layout(
            title="Daily Sales Overview with Selected Anomaly",
            height=300,
            margin=dict(t=30, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 2️⃣ Show AI bullet explanations (text only) ---
        for bullet_item in bullets_data:
            bullet_text = bullet_item.get("bullet", "No bullet text")
            bullet_category = bullet_item.get("category", "Insight")

            icon = {
                "Observation": "👁️",
                "Potential Risk": "⚠️",
                "Recommended Action": "🚀",
                "Fallback": "💬",
                "Error": "❌",
                "Insight": "💡"
            }.get(bullet_category, "💡")

            with st.expander(f"{icon} **{bullet_category}**: {bullet_text} 🔍"):
                detailed_explanation = generate_anomaly_bullet_explanation(bullet_text, selected_anomaly)
                st.markdown(detailed_explanation)
                st.markdown("---")

# --- Export options ---
if not anomalies_sorted.empty:
    st.markdown("## 📥 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        csv = anomalies_sorted.to_csv(index=False)
        st.download_button("📥 Download Anomalies CSV", csv, "anomalies.csv", "text/csv", use_container_width=True)
    with col2:
        top_anomalies = anomalies_sorted.head(5).to_string(index=False, columns=['Date_display', 'daily_sales', 'Severity'])
        risk_report = f"""# Anomaly Detection Risk Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Summary
- Detection Method: {st.session_state.get('detection_method', 'Unknown')}
- Total Anomalies: {len(anomalies_sorted)}
- Date Range: {daily_df['Date'].min().date()} to {daily_df['Date'].max().date()}
- Anomaly Rate: {(len(anomalies_sorted)/len(daily_df)*100):.2f}%

## Top 5 Most Significant Anomalies
{top_anomalies}

## Recommendations
1. Investigate the root causes of each anomaly.
2. Implement early warning systems for similar patterns.
3. Review operational procedures during anomaly periods.
4. Consider external factors (holidays, events, weather).
"""
        st.download_button("📄 Download Risk Report", risk_report, "risk_report.txt", "text/plain", use_container_width=True)

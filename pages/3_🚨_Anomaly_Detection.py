import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.anomaly_detection import (
    detect_anomalies_zscore,
    detect_anomalies_isolation_forest,
    analyze_anomaly_patterns
)
from ai.insights_generator import generate_anomaly_explanation

st.set_page_config(page_title="Anomaly Detection", page_icon="🚨", layout="wide")

st.title("🚨 Anomaly Detection & Risk Alerts")
st.markdown("### Personal AI-powered risk spotter")

if 'df' not in st.session_state:
    st.warning("⚠️ No data loaded. Please go to the main page and load data first.")
    if st.button("← Go to Main Page"):
        st.switch_page("app.py")
    st.stop()

daily_df = st.session_state['daily_df']

# Detect anomalies
if st.button("🚨 Detect Anomalies", type="primary"):
    with st.spinner("Analyzing data for anomalies..."):
        anomalies = detect_anomalies_isolation_forest(daily_df)
        
        if anomalies.empty:
            anomalies = detect_anomalies_isolation_forest(daily_df)
            method = "Isolation Forest (ML)"
        else:
            method = "Z-Score (Statistical)"
        
        # Add high/low flag
        mean = daily_df['daily_sales'].mean()
        std = daily_df['daily_sales'].std()
        anomalies['anomaly_type'] = anomalies['daily_sales'].apply(
            lambda x: 'High (Good)' if x > mean else 'Low (Bad)'
        )
        
        st.session_state['anomalies'] = anomalies
        st.session_state['detection_method'] = method
        st.success(f"✅ Detection complete! Found {len(anomalies)} anomalies using {method}.")

# Display anomalies
if 'anomalies' in st.session_state:
    anomalies = st.session_state['anomalies']
    method = st.session_state.get('detection_method', 'Unknown')
    
    st.markdown("## 📊 Anomaly Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Anomalies Detected", len(anomalies), delta=f"{(len(anomalies)/len(daily_df)*100):.1f}% of days")
    with col2:
        avg_anomaly = anomalies['daily_sales'].mean()
        normal_avg = daily_df['daily_sales'].mean()
        st.metric("Avg Anomaly Value", f"¥{avg_anomaly:,.2f}", delta=f"{((avg_anomaly/normal_avg-1)*100):+.1f}% vs normal")
    with col3:
        st.metric("Highest Anomaly", f"¥{anomalies['daily_sales'].max():,.2f}", delta="Peak")
    with col4:
        st.metric("Lowest Anomaly", f"¥{anomalies['daily_sales'].min():,.2f}", delta="Trough")

    st.markdown("## 📈 Anomaly Visualization")
    fig = go.Figure()
    normal_data = daily_df[~daily_df['Date'].isin(anomalies['Date'])]

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
            marker_color=anomalies['anomaly_type'].map({'High (Good)':'green','Low (Bad)':'red'}), nbinsx=20
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
        patterns = analyze_anomaly_patterns(anomalies)
        if isinstance(patterns, dict):
            st.metric("Total Anomalies", patterns['total_anomalies'])
            st.metric("Avg Value", f"¥{patterns['avg_anomaly_value']:,.2f}")
            st.metric("Max Value", f"¥{patterns['max_anomaly_value']:,.2f}")
            st.metric("Min Value", f"¥{patterns['min_anomaly_value']:,.2f}")
            if 'most_common_day' in patterns and patterns['most_common_day'] is not None:
                days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
                st.metric("Most Common Day", days[patterns['most_common_day']])

    anomalies_sorted = anomalies.sort_values('Date', ascending=False)
    anomalies_sorted['Severity'] = anomalies_sorted.apply(
        lambda row: '🟢 Good' if row['anomaly_type']=='High (Good)' else '🔴 Warning', axis=1
    )

    st.markdown("## 📋 Anomaly Details")
    display_cols = ['Date','daily_sales','Severity','anomaly_type']
    if 'z_score' in anomalies_sorted.columns: display_cols.append('z_score')
    st.dataframe(anomalies_sorted[display_cols], use_container_width=True)

    # AI Insights
    st.markdown("## 💡 AI-Powered Anomaly Explanation")
    selected_date = st.selectbox("Select an anomaly date to analyze:", anomalies_sorted['Date'].dt.date.tolist())
    if st.button("🤖 Get AI Explanation", type="primary"):
        selected_anomaly = anomalies_sorted[anomalies_sorted['Date'].dt.date==selected_date].iloc[0]
        with st.spinner("AI is analyzing this anomaly..."):
            explanation = generate_anomaly_explanation(selected_date, selected_anomaly['daily_sales'], daily_df['daily_sales'].mean())
            st.markdown(explanation)

    # Export
    st.markdown("## 📥 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        csv = anomalies_sorted.to_csv(index=False)
        st.download_button("📥 Download Anomalies CSV", csv, "anomalies.csv", "text/csv", use_container_width=True)
    with col2:
        risk_report = f"""# Anomaly Detection Risk Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Summary
- Detection Method: {method}
- Total Anomalies: {len(anomalies)}
- Date Range: {daily_df['Date'].min().date()} to {daily_df['Date'].max().date()}
- Anomaly Rate: {(len(anomalies)/len(daily_df)*100):.2f}%

## Top 5 Most Significant Anomalies
{anomalies_sorted.head(5).to_string()}

## Recommendations
1. Investigate the root causes of each anomaly
2. Implement early warning systems for similar patterns
3. Review operational procedures during anomaly periods
4. Consider external factors (holidays, events, weather)
"""
        st.download_button("📄 Download Risk Report", risk_report, "risk_report.txt", "text/plain", use_container_width=True)
else:
    st.info("👆 Click 'Detect Anomalies' to begin analysis.")

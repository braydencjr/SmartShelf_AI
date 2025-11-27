import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from config import DATE_COL, VALUE_COL

def create_sales_trend_chart(daily_df):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_df[DATE_COL],
        y=daily_df['daily_sales'],
        mode='lines',
        name='Daily Sales',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig.update_layout(
        title='Daily Sales Trend',
        xaxis_title='Date',
        yaxis_title='Sales (RMB)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_product_bar_chart(top_products_df):
    fig = px.bar(
        top_products_df,
        x=VALUE_COL,
        y='Item Name',
        orientation='h',
        title='Top Performing Products',
        color=VALUE_COL,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title='Total Sales (RMB)',
        yaxis_title='Product',
        showlegend=False,
        height=500
    )
    
    return fig

def create_category_pie_chart(df):
    category_sales = df.groupby('Category Name')[VALUE_COL].sum().reset_index()
    
    fig = px.pie(
        category_sales,
        values=VALUE_COL,
        names='Category Name',
        title='Sales Distribution by Category',
        hole=0.4
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    
    return fig

def create_hourly_heatmap(df):
    # Handle Time column safely
    if pd.api.types.is_timedelta64_dtype(df['Time']):
        # MySQL TIME type → timedelta → extract hours
        df['Hour'] = df['Time'].dt.components['hours']
    else:
        # CSV string or datetime
        df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour

    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')

    df['DayOfWeek'] = df[DATE_COL].dt.day_name()

    # Aggregate sales by day & hour
    heatmap_data = df.groupby(['DayOfWeek', 'Hour'])[VALUE_COL].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='DayOfWeek', columns='Hour', values=VALUE_COL)

    # Reorder days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot.reindex(days_order)

    # Fill missing hours with 0
    heatmap_pivot = heatmap_pivot.fillna(0)

    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Sales (RMB)"),
        title="Sales Heatmap by Hour and Day",
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )

    fig.update_layout(height=400)

    return fig

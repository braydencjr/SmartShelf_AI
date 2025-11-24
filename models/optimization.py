import pandas as pd
import numpy as np
from config import VALUE_COL, GROUP_COL

def optimize_inventory(df, forecast_df):
    # Calculate average daily demand per product
    product_demand = df.groupby(GROUP_COL).agg({
        'Quantity Sold (kilo)': 'mean',
        VALUE_COL: 'sum'
    }).reset_index()
    
    product_demand.columns = [GROUP_COL, 'avg_daily_demand', 'total_revenue']
    
    # Calculate safety stock (1.5x average demand)
    product_demand['recommended_stock'] = product_demand['avg_daily_demand'] * 1.5 * 7  # 1 week
    
    # Calculate reorder point
    product_demand['reorder_point'] = product_demand['avg_daily_demand'] * 3  # 3 days
    
    # Priority score based on revenue and demand
    product_demand['priority_score'] = (
        (product_demand['total_revenue'] / product_demand['total_revenue'].max()) * 0.7 +
        (product_demand['avg_daily_demand'] / product_demand['avg_daily_demand'].max()) * 0.3
    ) * 100
    
    product_demand = product_demand.sort_values('priority_score', ascending=False)
    
    return product_demand

def recommend_pricing(df):
    # Analyze price elasticity
    pricing_analysis = df.groupby(GROUP_COL).agg({
        'Unit Selling Price (RMB/kg)': 'mean',
        'Quantity Sold (kilo)': 'sum',
        VALUE_COL: 'sum'
    }).reset_index()
    
    pricing_analysis.columns = [GROUP_COL, 'avg_price', 'total_quantity', 'total_revenue']
    
    # Calculate revenue per unit price
    pricing_analysis['revenue_per_price_unit'] = (
        pricing_analysis['total_revenue'] / pricing_analysis['avg_price']
    )
    
    # Recommend price adjustments
    median_efficiency = pricing_analysis['revenue_per_price_unit'].median()
    
    pricing_analysis['price_recommendation'] = pricing_analysis.apply(
        lambda row: 'Increase Price (+5%)' if row['revenue_per_price_unit'] > median_efficiency * 1.2
        else 'Decrease Price (-5%)' if row['revenue_per_price_unit'] < median_efficiency * 0.8
        else 'Maintain Current Price',
        axis=1
    )
    
    return pricing_analysis

def calculate_resource_allocation(df, forecast_df):
    # Calculate optimal resource allocation based on forecast
    total_forecast = forecast_df['forecasted_sales'].sum()
    
    product_contribution = df.groupby(GROUP_COL)[VALUE_COL].sum().reset_index()
    product_contribution['allocation_pct'] = (
        product_contribution[VALUE_COL] / product_contribution[VALUE_COL].sum() * 100
    )
    
    product_contribution = product_contribution.sort_values('allocation_pct', ascending=False)
    
    return product_contribution
"""
Discount Impact Analysis Module

Provides comprehensive analysis of discount effects on:
•⁠  ⁠Sales volume and revenue
•⁠  ⁠Product-level performance
•⁠  ⁠Day-of-week patterns
•⁠  ⁠Integration with forecasting models
"""

import pandas as pd
import numpy as np
from config import DATE_COL, VALUE_COL


def calculate_discount_effect(df):
    """
    Measure overall impact of discounts on sales and quantity.
    
    Returns: dict with discount vs non-discount metrics
    """
    df_clean = df.copy()
    df_clean['revenue'] = df_clean["Quantity Sold (kilo)"] * df_clean["Unit Selling Price (RMB/kg)"]
    
    discount_effect = df_clean.groupby('Discount (Yes/No)').agg({
        'Quantity Sold (kilo)': ['mean', 'sum', 'count'],
        'revenue': ['mean', 'sum'],
        'Unit Selling Price (RMB/kg)': 'mean'
    }).round(2)
    
    # Flatten column names
    discount_effect.columns = ['_'.join(col).strip() for col in discount_effect.columns.values]
    
    result = {
        'discounted': {
            'avg_quantity': discount_effect.loc['Yes', 'Quantity Sold (kilo)_mean'] if 'Yes' in discount_effect.index else 0,
            'total_quantity': discount_effect.loc['Yes', 'Quantity Sold (kilo)_sum'] if 'Yes' in discount_effect.index else 0,
            'transaction_count': discount_effect.loc['Yes', 'Quantity Sold (kilo)_count'] if 'Yes' in discount_effect.index else 0,
            'avg_revenue': discount_effect.loc['Yes', 'revenue_mean'] if 'Yes' in discount_effect.index else 0,
            'total_revenue': discount_effect.loc['Yes', 'revenue_sum'] if 'Yes' in discount_effect.index else 0,
            'avg_price': discount_effect.loc['Yes', 'Unit Selling Price (RMB/kg)_mean'] if 'Yes' in discount_effect.index else 0,
        },
        'normal': {
            'avg_quantity': discount_effect.loc['No', 'Quantity Sold (kilo)_mean'] if 'No' in discount_effect.index else 0,
            'total_quantity': discount_effect.loc['No', 'Quantity Sold (kilo)_sum'] if 'No' in discount_effect.index else 0,
            'transaction_count': discount_effect.loc['No', 'Quantity Sold (kilo)_count'] if 'No' in discount_effect.index else 0,
            'avg_revenue': discount_effect.loc['No', 'revenue_mean'] if 'No' in discount_effect.index else 0,
            'total_revenue': discount_effect.loc['No', 'revenue_sum'] if 'No' in discount_effect.index else 0,
            'avg_price': discount_effect.loc['No', 'Unit Selling Price (RMB/kg)_mean'] if 'No' in discount_effect.index else 0,
        }
    }
    
    # Calculate lifts/deltas
    if result['normal']['avg_quantity'] > 0:
        result['quantity_lift_pct'] = ((result['discounted']['avg_quantity'] - result['normal']['avg_quantity']) 
                                       / result['normal']['avg_quantity'] * 100)
    else:
        result['quantity_lift_pct'] = 0
    
    if result['normal']['avg_revenue'] > 0:
        result['revenue_impact_pct'] = ((result['discounted']['avg_revenue'] - result['normal']['avg_revenue']) 
                                        / result['normal']['avg_revenue'] * 100)
    else:
        result['revenue_impact_pct'] = 0
    
    # Percentage of sales from discounts
    total_sales = result['discounted']['transaction_count'] + result['normal']['transaction_count']
    result['discount_sales_pct'] = (result['discounted']['transaction_count'] / total_sales * 100) if total_sales > 0 else 0
    
    return result


def analyze_discount_by_day(df):
    """
    Analyze discount effectiveness by day of week.
    
    Returns: DataFrame with day-of-week discount analysis
    """
    df_clean = df.copy()
    df_clean[DATE_COL] = pd.to_datetime(df_clean[DATE_COL])
    df_clean['day_of_week'] = df_clean[DATE_COL].dt.day_name()
    df_clean['is_weekend'] = df_clean[DATE_COL].dt.dayofweek.isin([5, 6]).astype(int)
    df_clean['revenue'] = df_clean["Quantity Sold (kilo)"] * df_clean["Unit Selling Price (RMB/kg)"]
    
    # Group by day and discount status
    day_analysis = df_clean.groupby(['day_of_week', 'Discount (Yes/No)']).agg({
        'Quantity Sold (kilo)': 'mean',
        'revenue': 'mean',
        'Item Code': 'count'
    }).round(2)
    
    day_analysis.columns = ['avg_quantity', 'avg_revenue', 'transaction_count']
    day_analysis = day_analysis.reset_index()
    
    # Calculate lift per day
    day_lift = []
    for day in day_analysis['day_of_week'].unique():
        day_data = day_analysis[day_analysis['day_of_week'] == day]
        if len(day_data) == 2:
            discounted = day_data[day_data['Discount (Yes/No)'] == 'Yes']['avg_quantity'].values
            normal = day_data[day_data['Discount (Yes/No)'] == 'No']['avg_quantity'].values
            
            if len(discounted) > 0 and len(normal) > 0 and normal[0] > 0:
                lift = ((discounted[0] - normal[0]) / normal[0] * 100)
            else:
                lift = 0
        else:
            lift = 0
        
        day_lift.append({'day': day, 'quantity_lift_pct': lift})
    
    day_lift_df = pd.DataFrame(day_lift)
    
    # Weekend vs Weekday analysis
    weekend_analysis = df_clean.groupby(['is_weekend', 'Discount (Yes/No)']).agg({
        'Quantity Sold (kilo)': 'mean',
        'revenue': 'mean'
    }).round(2)
    
    return {
        'by_day': day_analysis,
        'day_lift': day_lift_df,
        'weekend_analysis': weekend_analysis
    }


def analyze_product_discount_sensitivity(df):
    """
    Identify which products benefit most from discounts.
    
    Returns: DataFrame ranked by discount lift percentage
    """
    df_clean = df.copy()
    
    # Merge with product names if Item Name not present
    if 'Item Name' not in df_clean.columns and 'Item Code' in df_clean.columns:
        try:
            import os
            product_file = os.path.join(os.path.dirname(_file_), '..', 'data', 'annex1.csv')
            products = pd.read_csv(product_file)[['Item Code', 'Item Name']]
            df_clean = df_clean.merge(products, on='Item Code', how='left')
            # Fill any missing names with Item Code
            df_clean['Item Name'] = df_clean['Item Name'].fillna(df_clean['Item Code'].astype(str))
        except Exception as e:
            # If merge fails, use Item Code as name
            df_clean['Item Name'] = df_clean['Item Code'].astype(str)
    
    df_clean['revenue'] = df_clean["Quantity Sold (kilo)"] * df_clean["Unit Selling Price (RMB/kg)"]
    
    # Pivot table for quantity sold with/without discount
    pivot = df_clean.pivot_table(
        index='Item Name',
        columns='Discount (Yes/No)',
        values='Quantity Sold (kilo)',
        aggfunc='mean'
    ).round(2)
    
    # Ensure both columns exist
    if 'Yes' not in pivot.columns:
        pivot['Yes'] = 0
    if 'No' not in pivot.columns:
        pivot['No'] = 0
    
    pivot.columns = ['without_discount', 'with_discount']
    
    # Calculate lift
    pivot['quantity_lift_pct'] = ((pivot['with_discount'] - pivot['without_discount']) 
                                   / pivot['without_discount'].replace(0, np.nan) * 100).round(2)
    
    # Revenue comparison
    revenue_pivot = df_clean.pivot_table(
        index='Item Name',
        columns='Discount (Yes/No)',
        values='revenue',
        aggfunc='mean'
    ).round(2)
    
    # Ensure both columns exist
    if 'Yes' not in revenue_pivot.columns:
        revenue_pivot['Yes'] = 0
    if 'No' not in revenue_pivot.columns:
        revenue_pivot['No'] = 0
    
    revenue_pivot.columns = ['revenue_without_discount', 'revenue_with_discount']
    revenue_pivot['revenue_impact_pct'] = ((revenue_pivot['revenue_with_discount'] - revenue_pivot['revenue_without_discount']) 
                                            / revenue_pivot['revenue_without_discount'].replace(0, np.nan) * 100).round(2)
    
    # Combine results - keep all revenue columns
    result = pd.concat([pivot, revenue_pivot[['revenue_without_discount', 'revenue_with_discount', 'revenue_impact_pct']]], axis=1)
    
    # Calculate transaction counts
    transaction_counts = df_clean.groupby(['Item Name', 'Discount (Yes/No)']).size().unstack(fill_value=0)
    if transaction_counts.shape[1] == 1:
        # Handle case where only one discount type exists
        if 'Yes' in transaction_counts.columns:
            transaction_counts['No'] = 0
        else:
            transaction_counts['Yes'] = 0
    
    # Sort by column order: No first, then Yes
    discount_cols = ['No', 'Yes'] if 'No' in transaction_counts.columns else ['Yes', 'No']
    transaction_counts = transaction_counts[discount_cols]
    transaction_counts.columns = ['transactions_without_discount', 'transactions_with_discount']
    result = pd.concat([result, transaction_counts], axis=1)
    
    # Sort by quantity lift
    result = result.sort_values('quantity_lift_pct', ascending=False)
    
    return result


def get_discount_insights_summary(df):
    """
    Generate comprehensive discount analysis summary.
    
    Returns: dict with all key metrics for AI context
    """
    discount_effect = calculate_discount_effect(df)
    day_analysis = analyze_discount_by_day(df)
    product_sensitivity = analyze_product_discount_sensitivity(df)
    
    # Find best and worst products for discounts
    top_discount_products = product_sensitivity.head(3)
    worst_discount_products = product_sensitivity.tail(3)
    
    # Best day for discounts
    best_day = day_analysis['day_lift'].loc[day_analysis['day_lift']['quantity_lift_pct'].idxmax()]
    worst_day = day_analysis['day_lift'].loc[day_analysis['day_lift']['quantity_lift_pct'].idxmin()]
    
    summary = {
        'overall_quantity_lift': round(discount_effect['quantity_lift_pct'], 2),
        'overall_revenue_impact': round(discount_effect['revenue_impact_pct'], 2),
        'discount_sales_percentage': round(discount_effect['discount_sales_pct'], 2),
        'discounted_avg_quantity': round(discount_effect['discounted']['avg_quantity'], 2),
        'normal_avg_quantity': round(discount_effect['normal']['avg_quantity'], 2),
        'discounted_avg_revenue': round(discount_effect['discounted']['avg_revenue'], 2),
        'normal_avg_revenue': round(discount_effect['normal']['avg_revenue'], 2),
        'best_discount_day': best_day['day'],
        'best_discount_day_lift': round(best_day['quantity_lift_pct'], 2),
        'worst_discount_day': worst_day['day'],
        'worst_discount_day_lift': round(worst_day['quantity_lift_pct'], 2),
        'top_discount_products': top_discount_products.index.tolist(),
        'top_product_lifts': top_discount_products['quantity_lift_pct'].tolist(),
        'worst_discount_products': worst_discount_products.index.tolist(),
        'worst_product_impacts': worst_discount_products['revenue_impact_pct'].tolist(),
    }
    
    return summary


def add_discount_flag(df):
    """
    Add binary discount flag for forecasting models (SARIMAX, XGBoost).
    
    Returns: DataFrame with new discount_flag column
    """
    df_clean = df.copy()
    df_clean['discount_flag'] = (df_clean['Discount (Yes/No)'] == 'Yes').astype(int)
    return df_clean


def get_discount_context_for_ai(df):
    """
    Prepare discount insights context for AI analysis.
    
    Returns: formatted string for AI prompt
    """
    summary = get_discount_insights_summary(df)
    product_sensitivity = analyze_product_discount_sensitivity(df)
    
    context = f"""
DISCOUNT IMPACT ANALYSIS:

1.⁠ ⁠Overall Discount Effectiveness:
   - Quantity Lift: {summary['overall_quantity_lift']}% (avg items sold when discounted vs normal)
   - Revenue Impact: {summary['overall_revenue_impact']}% (avg revenue per transaction)
   - Discount penetration: {summary['discount_sales_percentage']}% of all sales had discounts

2.⁠ ⁠Performance Metrics:
   - Average quantity (with discount): {summary['discounted_avg_quantity']} kg
   - Average quantity (normal price): {summary['normal_avg_quantity']} kg
   - Average revenue (with discount): ¥{summary['discounted_avg_revenue']}
   - Average revenue (normal price): ¥{summary['normal_avg_revenue']}

3.⁠ ⁠Best & Worst Days for Discounts:
   - Best Day: {summary['best_discount_day']} (+{summary['best_discount_day_lift']}% quantity lift)
   - Worst Day: {summary['worst_discount_day']} ({summary['worst_discount_day_lift']}% quantity lift)

4.⁠ ⁠Top Products That Explode with Discounts:
"""
    for product, lift in zip(summary['top_discount_products'], summary['top_product_lifts']):
        context += f"   - {product}: +{lift}% quantity increase\n"
    
    context += "\n5. Products Where Discounts Hurt Revenue:\n"
    for product, impact in zip(summary['worst_discount_products'], summary['worst_product_impacts']):
        context += f"   - {product}: {impact}% revenue impact\n"
    
    context += "\nKEY INSIGHT: " 
    if summary['overall_quantity_lift'] > 10 and summary['overall_revenue_impact'] > 0:
        context += "Discounts are highly effective - they drive volume without hurting revenue."
    elif summary['overall_quantity_lift'] > 10 and summary['overall_revenue_impact'] < 0:
        context += "Discounts drive volume but reduce per-transaction revenue. Use strategically on high-margin products only."
    elif summary['overall_quantity_lift'] < 5:
        context += "Discounts show weak volume lift. Reconsider discount strategy - they may not be worth the margin loss."
    else:
        context += "Discounts have moderate effectiveness. Consider segment-specific discount strategies."
    
    return context
import pandas as pd
import numpy as np
from config import DATE_COL, VALUE_COL, GROUP_COL, ITEM_NAME_COL

def compute_kpis(df):
    return {
        "row_count": int(len(df)),
        "total_sales": float(df[VALUE_COL].sum()),
        "avg_sale": float(df[VALUE_COL].mean()),
        "max_sale": float(df[VALUE_COL].max()),
        "min_sale": float(df[VALUE_COL].min()),
        "unique_products": int(df[GROUP_COL].nunique()),
        "date_range": f"{df[DATE_COL].min().date()} to {df[DATE_COL].max().date()}"
    }

def calculate_daily_sales(df):
    daily = (
        df.groupby(df[DATE_COL].dt.date)[VALUE_COL]
          .sum()
          .rename("daily_sales")
          .reset_index()
    )
    daily[DATE_COL] = pd.to_datetime(daily[DATE_COL])
    return daily

def top_products(df, n=10):
    TOP_GROUP_COLS = [GROUP_COL, ITEM_NAME_COL]
    tp = (
        df.groupby(TOP_GROUP_COLS)[VALUE_COL]
          .sum()
          .sort_values(ascending=False)
          .head(n)
          .reset_index()
    )
    return tp

def calculate_growth_rate(df):
    daily = calculate_daily_sales(df)
    
    if len(daily) < 2:
        return 0.0
    
    first_week = daily.head(7)["daily_sales"].mean()
    last_week = daily.tail(7)["daily_sales"].mean()
    
    if first_week == 0:
        return 0.0
    
    growth = ((last_week - first_week) / first_week) * 100
    return growth

def calculate_category_performance(df):
    category_perf = (
        df.groupby("Category Name")[VALUE_COL]
          .agg(['sum', 'mean', 'count'])
          .sort_values('sum', ascending=False)
          .reset_index()
    )
    category_perf.columns = ['Category', 'Total Sales', 'Avg Sale', 'Transactions']
    return category_perf
import pandas as pd
import streamlit as st
from config import DATE_COL, VALUE_COL

@st.cache_data
def load_data(annex1_path, annex2_path):
    try:
        df1 = pd.read_csv(annex1_path)
        df2 = pd.read_csv(annex2_path)
        
        df = pd.merge(df1, df2, on="Item Code", how="inner")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    df = df.copy()
    
    # Calculate total sales
    df[VALUE_COL] = df["Quantity Sold (kilo)"] * df["Unit Selling Price (RMB/kg)"]
    
    # Convert date column
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    
    # Sort by date
    df = df.sort_values(DATE_COL)
    
    # Filter only sales (not returns)
    df = df[df["Sale or Return"] == "sale"]
    
    return df

def get_date_range(df):
    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()
    return min_date, max_date
import os
import pandas as pd
import streamlit as st
import pymysql
from config import DATE_COL, VALUE_COL
from dotenv import load_dotenv
import os

load_dotenv()  # must be called before os.environ.get

@st.cache_data
def load_data():
    try:
        # Read database credentials from environment variables
        host = os.environ.get("MYSQL_HOST")
        user = os.environ.get("MYSQL_USER")
        password = os.environ.get("MYSQL_PASSWORD")
        database = os.environ.get("MYSQL_DB")

        # Connect to MySQL
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=3306
        )

        # Load sales table
        df_sales = pd.read_sql("SELECT * FROM sales", conn)

        # Load product_info table
        df_products = pd.read_sql("SELECT * FROM product_info", conn)

        # Merge on Item Code
        df = pd.merge(df_sales, df_products, on="Item Code", how="inner")

        conn.close()
        return df

    except Exception as e:
        st.error(f"Error loading data from database: {e}")
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
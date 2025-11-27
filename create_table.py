import pymysql
from db_connection import db_connection

# Assuming you have db_connection function
conn = db_connection()  # returns a pymysql connection object
cursor = conn.cursor()

# --- Create product_info table ---
create_product_info_table = """
CREATE TABLE IF NOT EXISTS product_info (
    `Item Code` VARCHAR(50) PRIMARY KEY,
    `Item Name` VARCHAR(255),
    `Category Code` VARCHAR(50),
    `Category Name` VARCHAR(255)
);
"""
cursor.execute(create_product_info_table)

# --- Create sales table ---
create_sales_table = """
CREATE TABLE IF NOT EXISTS sales (
    `Date` DATE,
    `Time` TIME,
    `Item Code` VARCHAR(50),
    `Quantity Sold (kilo)` FLOAT,
    `Unit Selling Price (RMB/kg)` FLOAT,
    `Sale or Return` VARCHAR(20),
    `Discount (Yes/No)` VARCHAR(5)
);
"""
cursor.execute(create_sales_table)

# Commit changes and close connection
conn.commit()
cursor.close()
conn.close()

print("Tables 'product_info' and 'sales' created successfully!")

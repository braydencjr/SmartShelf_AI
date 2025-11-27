from dotenv import load_dotenv
import os
import pymysql

load_dotenv()  # <-- load .env variables

conn = pymysql.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DB"),
    ssl_disabled=True
)


print("✅ Connected!")
conn.close()

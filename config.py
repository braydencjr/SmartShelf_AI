import os

# Application Settings
APP_TITLE = "Enterprise Predictive Analytics SaaS"
APP_ICON = "📊"

# Data Settings
DATE_COL = "Date"
VALUE_COL = "TotalSales"
GROUP_COL = "Item Code"
ITEM_NAME_COL = "Item Name"
CATEGORY_COL = "Category Name"

# Paths
DATA_DIR = "data"
ANNEX1_PATH = os.path.join(DATA_DIR, "annex1.csv")
ANNEX2_PATH = os.path.join(DATA_DIR, "annex2.csv")

# Model Parameters
ANOMALY_Z_THRESHOLD = 3.0
FORECAST_PERIODS = 30
TOP_N_PRODUCTS = 10

# Gemini API Settings
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.0-flash"
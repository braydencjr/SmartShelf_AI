# 🏢 SmartShelf AI - Enterprise Predictive Analytics SaaS

> AI-powered predictive analytics platform for enterprise operations

## 🌟 Key Features

### 📊 **Business Intelligence Dashboard**
- Real-time KPI monitoring
- Interactive visualizations
- Category and product performance analysis
- Sales heatmaps by time and day

### 🔮 **Predictive Forecasting**
- Prophet, SARIMAX models with robust train/test validation
- 7-30 day forecasts with confidence intervals
- **85%+ forecast accuracy** 

- **Enhanced AI insights with discount strategy context**

### 🚨 **Anomaly Detection**
- Machine Learning (Isolation Forest) detection
- Pattern analysis
- Root cause investigation

### 💡 **AI-Powered Insights (Gemini)**
- Executive summaries 
- Custom Q&A about your data
- Performance analysis
- Strategic recommendations
- **Discount-aware business intelligence**

### ⚙️ **Operations Optimization**
- Inventory optimization with reorder points
- Pricing strategy recommendations
- Resource allocation (Pareto analysis)
- Implementation roadmaps

## 🗄️ Data Source

Initial data is loaded **directly from MySQL**.

The system supports:
- Live sales tables
- Product master tables
- Incremental updates over time

CSV upload is optional and used only for testing.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- VSCode or any Python IDE
- Gemini API Key (free from Google AI Studio)

### Installation

1. **Clone or create the project structure:**
```bash
mkdir enterprise-analytics-saas
cd enterprise-analytics-saas
```

2. **Create the following directory structure:**
```
enterprise-analytics-saas/
├── Main_Page.py
├── config.py
├── requirements.txt
├── data/
│   ├── annex1.csv
│   └── annex2.csv
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── kpi_calculator.py
│   └── visualizations.py
├── models/
│   ├── __init__.py
│   ├── forecasting.py
│   ├── anomaly_detection.py
│   └── optimization.py
├── ai/
│   ├── __init__.py
│   └── insights_generator.py
└── pages/
    ├── 1_Product Performance.py
    ├── 2_Business Insight Forecasts.py
    ├── 3_Anomaly_Detection.py
    ├── 4_Optimization.py
    └── 5_AI_Insights.py
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up your data:**
   - Place `annex1.csv` and `annex2.csv` in the `data/` folder
   - Or use the file uploader in the app

5. **Set up Gemini API (for AI features):**
   
   Get a free API key from: https://makersuite.google.com/app/apikey
   
   **Option A: Environment Variable (Recommended)**
   ```bash
   # Linux/Mac
   export GEMINI_API_KEY="your_api_key_here"
   
   # Windows
   set GEMINI_API_KEY=your_api_key_here
   ```
   
   **Option B: Create .env file**
   ```bash
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

6. **Run the application:**
```bash
streamlit run Main_Page.py
```

7. **Open your browser:**
   - The app will open automatically at `http://localhost:8501`
   - If not, navigate to that URL manually

## 📂 Project Structure

### Core Files

#### `Main_Page.py`
Main application entry point with:
- Homepage and navigation
- Data loading interface
- Quick KPI overview
- Navigation to sub-modules

#### `config.py`
Configuration settings:
- Column names and constants
- API settings
- File paths
- Model parameters

#### `requirements.txt`
All Python dependencies with versions

### Utils Module (`utils/`)

#### `data_loader.py`
- CSV file loading and merging
- Data preprocessing
- Date conversion
- Data caching

#### `kpi_calculator.py`
- KPI computations
- Daily sales aggregation
- Top products analysis
- Category performance
- Growth rate calculations

#### `visualizations.py`
- Plotly chart generation
- Sales trend charts
- Product bar charts
- Category pie charts
- Hourly heatmaps

#### `discount_analysis.py` 
- Overall discount effect calculation (quantity lift, revenue impact)
- Day-of-week discount effectiveness analysis
- Product-level discount sensitivity ranking
- Discount insights summary generation
- AI context preparation for discount-aware recommendations

### Models Module (`models/`)

#### `forecasting.py` & `forecasting_advanced.py`
- Exponential Smoothing implementation
- ARIMA, SARIMAX, Prophet, XGBoost models
- **Robust validation:** 70/30 train/test split with MAPE/RMSE metrics
- Forecast accuracy metrics (MAPE, RMSE, MAE)
- Confidence interval calculation
- **Smart ensemble:** Weighted combination of top performers

#### `anomaly_detection.py`
- Isolation Forest ML detection
- Anomaly pattern analysis
- Severity classification

#### `optimization.py`
- Inventory optimization algorithms
- Pricing strategy analysis
- Resource allocation (Pareto)
- Reorder point calculation

### AI Module (`ai/`)

#### `insights_generator.py`
- Gemini AI integration
- Executive summary generation
- Custom Q&A
- Anomaly explanations
- Forecast insights

## 🎯 Usage Guide

### 1. Data Loading
- Use the sidebar to upload CSV files or enable sample data
- The app automatically merges and preprocesses your data
- Supports date filtering for focused analysis

### 2. Dashboard
- View real-time KPIs
- Analyze sales trends with moving averages
- Explore top products and categories
- Examine hourly sales patterns
- **🎯 NEW: Discount Impact Analysis**
  - Tab 1: Volume & Revenue breakdown (with vs without discount)
  - Tab 2: Day-of-week discount effectiveness
  - Tab 3: Top products responding to discounts
  - Tab 4: Problem products (avoid discounting these)

### 3. Forecasting
- Select forecasting method (Prophet, XGBoost, SARIMAX, ARIMA, or Ensemble)
- Choose forecast period (7-90 days)
- View confidence intervals and validation metrics
- **Model validation:** See MAPE/RMSE on test set before final forecast
- **Discount context:** AI insights consider discount impact on demand
- Generate AI-powered forecast insights
- Export forecasts to CSV

### 4. Anomaly Detection
- Choose detection method (Z-Score or Isolation Forest)
- Adjust sensitivity thresholds
- Identify unusual sales patterns
- Get AI explanations for specific anomalies
- Export risk reports

### 5. AI Insights
- Generate executive summaries (**now with discount strategy**)
- Ask custom questions about your data
- Run detailed performance analyses
- **Discount-aware recommendations:** AI understands discount effectiveness
- Create strategic plans
- Save and export insights

### 6. Optimization
- **Inventory:** Calculate optimal stock levels, reorder points, and priorities
- **Pricing:** Get price adjustment recommendations
- **Resources:** Pareto analysis for efficient allocation
- **Master Plan:** Integrated recommendations with roadmap

## 📊 Data Format

### Required CSV Files

**annex1.csv** (Sales transactions):
```csv
Date,Time,Item Code,Quantity Sold (kilo),Unit Selling Price (RMB/kg),Sale or Return,Discount (Yes/No)
2022-07-01,09:15:07.924,102900005117056,0.396,7.6,sale,No
```

**annex2.csv** (Product information):
```csv
Item Code,Item Name,Category Code,Category Name
102900005117056,Broccoli,1011010101,Flower/Leaf Vegetables
```

### Data Requirements
- Date format: YYYY-MM-DD
- Time format: HH:MM:SS.fff
- Numeric fields: Quantity and Price must be numeric
- Item Code: Must match between both files

## 🔧 Customization

### Modify KPI Calculations
Edit `utils/kpi_calculator.py` to add custom metrics:
```python
def compute_kpis(df):
    return {
        "row_count": int(len(df)),
        "total_sales": float(df[VALUE_COL].sum()),
        # Add your custom KPIs here
        "custom_metric": your_calculation(df)
    }
```

### Adjust Forecast Parameters
Edit `config.py`:
```python
FORECAST_PERIODS = 30  # Default forecast days
ANOMALY_Z_THRESHOLD = 3.0  # Z-score threshold
```

### Customize UI Colors
Edit the CSS in each page file:
```python
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #your_colors);
    }
</style>
""", unsafe_allow_html=True)
```

## 🎓 Understanding the Models

### Forecasting Models (Ranked by Performance)
1. **SARIMAX** (
    - Best for: Data with strong seasonality and complex trends
    - Parameters: Auto-tuned seasonal and non-seasonal orders
    - Use case: Short to medium-term operational forecasting
    - Accuracy: High accuracy for stable seasonal data

4. **Prophet** 
   - Best for: Data with long-term trends and holiday effects
   - Parameters: Automatic trend and seasonality detection
   - Use case: Medium to long-term demand forecasting
   - Accuracy: Reliable for business time-series with growth patterns

**All models:** 70/30 train/test split with honest validation metrics


## 📄 License

MIT License - Feel free to use for your projects

## 🙏 Acknowledgments

- **Streamlit** - Web framework
- **Plotly** - Interactive visualizations
- **Statsmodels** - Time-series models
- **Scikit-learn** - ML algorithms
- **Google Gemini** - AI insights

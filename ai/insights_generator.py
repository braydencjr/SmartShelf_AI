import os
import google.generativeai as genai
from config import GEMINI_API_KEY_ENV, GEMINI_MODEL

def initialize_gemini():
    api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        return model
    except Exception as e:
        print(f"Gemini initialization error: {e}")
        return None

def generate_executive_summary(kpis, top_products, anomalies):
    model = initialize_gemini()
    
    if model is None:
        return "AI insights unavailable. Please set GEMINI_API_KEY environment variable."
    
    context = f'''
    Business Analytics Summary:
    
    Key Performance Indicators:
    - Total Transactions: {kpis['row_count']:,}
    - Total Sales: ¥{kpis['total_sales']:,.2f}
    - Average Sale: ¥{kpis['avg_sale']:.2f}
    - Unique Products: {kpis['unique_products']}
    
    Top 5 Products:
    {top_products.to_string()}
    
    Anomalies Detected: {len(anomalies)} unusual sales days
    '''
    
    prompt = f'''
    You are a business analytics expert. Based on the following data, provide:
    1. Executive summary (2-3 sentences)
    2. Key insights (3-4 bullet points)
    3. Actionable recommendations (3 specific actions)
    
    {context}
    
    Format your response in markdown with clear sections.
    '''
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating insights: {e}"

def generate_forecast_insights(forecast_df, historical_avg):
    model = initialize_gemini()
    
    if model is None:
        return "AI insights unavailable."
    
    forecast_avg = forecast_df['forecasted_sales'].mean()
    growth_rate = ((forecast_avg - historical_avg) / historical_avg) * 100
    

    prompt = f"""
    Learn this sales forecast summary:
    - Historical daily sales: ¥{historical_avg:,.2f}
    - Forecast period: {len(forecast_df)} days
    - Forecasted daily sales (sample): {forecast_df['forecasted_sales'].head(10).tolist()} ... {forecast_df['forecasted_sales'].tail(10).tolist()}
    - Total forecasted revenue: ¥{forecast_df['forecasted_sales'].sum():,.2f}
    - Min/Max forecast: ¥{forecast_df['forecasted_sales'].min():,.2f} / ¥{forecast_df['forecasted_sales'].max():,.2f}

    Provide:
    1. Observations about trends and patterns (e.g., peaks, dips)
    2. Potential risks or anomalies
    3. Recommended actions or preparations

    Be concise and actionable. Don't be lengthy. Don't be generic. Use simple understandable language
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating forecast insights: {e}"

def generate_anomaly_explanation(anomaly_date, anomaly_value, avg_value):
    model = initialize_gemini()
    
    if model is None:
        return "AI insights unavailable."
    
    deviation_pct = ((anomaly_value - avg_value) / avg_value) * 100
    
    prompt = f'''
    Explain this sales anomaly:
    - Date: {anomaly_date}
    - Sales: ¥{anomaly_value:,.2f}
    - Average: ¥{avg_value:,.2f}
    - Deviation: {deviation_pct:+.1f}%
    
    Provide:
    1. Possible causes (3-4 reasons)
    2. What to investigate
    3. Preventive actions if negative, or replication strategy if positive
    
    Be specific and practical.
    '''
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def answer_custom_question(question, df_summary):
    model = initialize_gemini()
    
    if model is None:
        return "AI insights unavailable."

    # Tell the AI to check if it can answer based on data
    prompt = f"""
You are an AI assistant specialized in business analytics.
You are given this sales data summary:
{df_summary}

Answer the following question: "{question}"

Rules:
1. If the question is directly related to the data (sales, products, categories, revenue, performance, trends, inventory, customer behavior, etc.), provide a clear, concise, and actionable answer.
2. If the question is not related to the data, respond politely with a short message like:
   "Sorry, I can only answer questions about the business data you provided."
3. Do not make up unrelated information or generate a full report for irrelevant questions.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

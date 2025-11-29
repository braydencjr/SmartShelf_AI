import os
import google.generativeai as genai
from config import GEMINI_API_KEY_ENV, GEMINI_MODEL
from utils.discount_analysis import get_discount_context_for_ai
import holidays
import pandas as pd

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

def generate_executive_summary(kpis, top_products, anomalies, df=None):
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
    
    prompt = f"""
You are a senior retail data analyst. 
Your job is to give **sharp, numeric, actionable** insights — no generic advice.

### REQUIRED OUTPUT FORMAT (must follow exactly):

## Executive Summary (2–3 sentences)
- Briefly describe performance using numbers from the data.

## 📌 Key Insights (3–5 bullets, each MUST contain numbers)
Each bullet MUST include:
- A % change, difference, or a specific quantity
- A comparison vs average or top items
- A clear interpretation (no fluff)

Examples:
- "Sales spiked **+28%** above the 7-day average."
- "Top product contributes **34%** of total revenue."

## 🎯 Actionable Recommendations (3 items, each MUST be numeric)
Each recommendation MUST include:
- A % adjustment or exact quantity change  
- A threshold to act on  
- A direct operational instruction  

Examples:
- "Increase inventory of top 3 SKUs by **+15%** on weekends."
- "Reduce ordering of slow-moving items by **10–18%**."
- "Investigate 12 anomaly days where sales dropped **>25%** below expected."

## 📊 (Optional) Table Summary
Include a small table comparing:
- Top 5 products
- Their total sales
- Their contribution %

### DATA CONTEXT:
{context}

### STRICT RULES:
- No vague business advice.
- No generic statements like "optimize inventory".
- Every insight MUST reference a number, percentage, or threshold.
- Keep bullets short and punchy.
- Output must be Markdown formatted.
"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating insights: {e}"

def generate_forecast_insights(forecast_df, historical_avg, df=None):
    """
    Generate AI-based forecast insights with optional automatic discount context.
    """
    model = initialize_gemini()
    if model is None:
        return "AI insights unavailable."

    forecast_avg = forecast_df['forecasted_sales'].mean()
    growth_rate = ((forecast_avg - historical_avg) / historical_avg) * 100

    # --- Auto-add discount context if df has discount info ---
    discount_context = ""
    if df is not None and 'Discount (Yes/No)' in df.columns:
        try:
            discount_context = "\n\n" + get_discount_context_for_ai(df)
        except:
            discount_context = ""

    prompt = f"""
Provide a clean, readable **plain text summary** of this sales forecast.
Use short bullet points and avoid JSON entirely.

Sales data:
- Historical daily sales: ¥{historical_avg:,.2f}
- Forecast period: {len(forecast_df)} days
- Forecasted daily sales (sample): {forecast_df['forecasted_sales'].head(10).tolist()} ... {forecast_df['forecasted_sales'].tail(10).tolist()}
- Total forecasted revenue: ¥{forecast_df['forecasted_sales'].sum():,.2f}
- Min/Max forecast: ¥{forecast_df['forecasted_sales'].min():,.2f} / ¥{forecast_df['forecasted_sales'].max():,.2f}
{discount_context}

Instructions:
1. Provide **3–5 key bullet points**.
2. Each bullet should include:
   - A **main insight** (with emojis like 📈📉⚠️)
   - A short explanation (1–2 sentences max).
3. **Do NOT return JSON.**  
4. Focus on a variety of operational/business factors:
   - demand shifts
   - seasonality
   - supply constraints
   - stock issues
   - staffing/capacity
   - weather
   - competitor activity
   (Avoid over-focusing on discounts.)
5. Use **bold** or *italic* emphasis where helpful.
6. Avoid duplicated points and avoid writing a summary section.
"""

    try:
        response = model.generate_content(prompt)
        text_output = response.text.strip()

        # Remove accidental code fences
        if text_output.startswith("```") and text_output.endswith("```"):
            text_output = "\n".join(text_output.split("\n")[1:-1]).strip()

        return text_output

    except Exception as e:
        return f"Error generating forecast insights: {e}"

    
def generate_anomaly_bullet_explanation(bullet_text, anomaly_row, historical_avg=None, df=None):
    """
    Explain a sales anomaly with automatic discount context.
    """
    model = initialize_gemini()
    if model is None:
        return "AI insights unavailable."

    date_val = anomaly_row['Date']
    date_str = date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else str(date_val)
    sales_val = anomaly_row['daily_sales']
    avg_text = f"Historical Average: ¥{historical_avg:,.2f}" if historical_avg else ""

    # --- Add discount context if available ---
    discount_context = ""
    if df is not None and 'Discount (Yes/No)' in df.columns:
        try:
            discount_context = "\nRelevant Discount Context:\n" + get_discount_context_for_ai(df)
        except:
            discount_context = ""

    prompt = f"""
Explain this sales anomaly insight clearly and concisely for business stakeholders as plain text:
- Bullet: {bullet_text}
- Date: {date_str} 📅
- Sales on this date: ¥{sales_val:,.2f} 💰
{avg_text}
{discount_context}

Instructions:
1. Explain why sales on this date were unusual (1–2 sentences).
2. Break down contributing factors in point form if relevant:
   - 🎁 Promotion
   - 📣 Marketing campaign
   - 🎉 Holiday/event
3. Suggest 1 actionable step or solution. ✅ 🛠️ 🔍
4. Keep points short, clear, and practical (max 3 sentences per point). 
5. Use professional, business-friendly language.
6. Do not return JSON, return a clean readable text.
7.Format your response for easy reading:
   - Use **bold** or *italic* to emphasize key points
   - Use tables for comparisons or numeric summaries
   - Use short bullets for lists (max 3 sentences per bullet)
8.Avoid duplicatate points and summary.
"""

    try:
        response = model.generate_content(prompt)
        text_output = response.text.strip()
        # Remove code fences if any
        if text_output.startswith("```") and text_output.endswith("```"):
            text_output = "\n".join(text_output.split("\n")[1:-1]).strip()
        return text_output
    except Exception as e:
        return f"Error generating insight: {e}"

    

def answer_custom_question(question, df_summary):
    model = initialize_gemini()
    
    if model is None:
        return "AI insights unavailable."

    prompt = f"""
You are an AI business analyst. You are given this sales data summary:
{df_summary}

Answer the user's question: "{question}"

Guidelines for your response:
1. Base your answer strictly on the data provided.
2. Provide **clear, concise, actionable insights** suitable for business decisions.
3. Analyze trends, promotions, discounts, or seasonal effects if relevant.
4. For questions about discounts, pricing, or future strategies, give **3–5 short bullets** including:
   - 🎁 Suggested action
   - ⚠️ Potential risks
   - 🚀 One actionable recommendation
5. Highlight numeric examples or relevant trends if helpful.
6. Use professional language and emojis for clarity.
7. Format your response for easy reading:
   - Use **bold** or *italic* to emphasize key points
   - Use tables for comparisons or numeric summaries
   - Use short bullets for lists (max 3 sentences per bullet)
8. If the question is unrelated to business data, respond politely with:
   "Sorry, I can only answer questions about the business data you provided."
9. Avoid generating unrelated reports or long, verbose explanations.
10.Avoid duplicatate points and summary.
"""


    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

    
def generate_anomaly_explanation(anomaly_date, anomaly_value, avg_value):
    model = initialize_gemini()
    
    if model is None:
        return "AI insights unavailable."
    
    deviation_pct = ((anomaly_value - avg_value) / avg_value) * 100
    
    prompt = f"""Generate 3–5 short, punchy, headline-style bullet points for this sales anomaly:
- Date: {anomaly_date}
- Sales: ¥{anomaly_value:,.2f}
- Average: ¥{avg_value:,.2f}
- Deviation: {deviation_pct:+.1f}%

Instructions:
1. Each bullet should be **title-like** (4–8 words), catchy, and suitable as a mini-headline. Use emojis ⬆️⬇️⚠️📈📉 where appropriate.
2. After each bullet, provide a 1–2 sentence explanation (optional in display).
3. Return only JSON array with fields:
   - "text": the title/bullet
   - "explanation": optional 1–2 sentence detail
   - "graph_suggestion": optional
4. Focus on date-related reasons (weekday, holiday, seasonality) and highlight significant deviations.
5.Format your response for easy reading:
   - Use **bold** or *italic* to emphasize key points
   - Use tables for comparisons or numeric summaries
   - Use short bullets for lists (max 3 sentences per bullet)
6.Avoid duplicatate points and summary.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

    
def generate_anomaly_bullet_explanation(bullet_text, anomaly_row, historical_avg=None, df=None):
    """
    Explain a sales anomaly for business stakeholders with optional discount context.
    Returns clean, readable text.
    """
    model = initialize_gemini()
    if model is None:
        return "AI insights unavailable."

    # Enrich row with date, holiday, and season info
    anomaly_row = enrich_anomaly_row(anomaly_row)

    date_val = anomaly_row['Date']
    date_str = date_val.strftime('%Y-%m-%d') if hasattr(date_val, 'strftime') else str(date_val)
    sales_val = anomaly_row['daily_sales']
    avg_text = f"Historical Average: ¥{historical_avg:,.2f}" if historical_avg else ""

    # Holiday / season emoji
    holiday_emoji = "🎄" if anomaly_row['is_holiday_season'] else "🎉" if anomaly_row['holiday_name'] else "🛒"
    holiday_text = f"{anomaly_row['holiday_name']} {holiday_emoji}" if anomaly_row['holiday_name'] else ""

    # --- Initialize discount context ---
    discount_context = ""
    if df is not None and 'Discount (Yes/No)' in df.columns:
        try:
            discount_context = "\nRelevant Discount Context:\n" + get_discount_context_for_ai(df)
        except:
            discount_context = ""

    # Build prompt
    prompt = f"""
Explain this sales anomaly for business stakeholders:
- Bullet: {bullet_text}
- Date: {date_str} 📅
- Sales on this date: ¥{sales_val:,.2f} 💰
- Day of week: {anomaly_row['day_of_week']} {'🌞' if not anomaly_row['is_weekend'] else '🌙'}
- Holiday/Season: {holiday_text}
{avg_text}
{discount_context}

Instructions:
1. Explain why sales on this date were unusual (1–2 sentences).
2. Break down contributing factors in **point form** if relevant:
   - 🎁 Promotion
   - 📣 Marketing campaign
   - 🎉 Holiday/event
3. Suggest 1 specific actionable step or solution. ✅ 🛠️ 🔍
4. Keep points short, clear, and practical (max 3 sentences per point). ✍️
5. Use professional, business-friendly language. 💼
6. **Do not return JSON**, return a clean readable text instead.
7. Explain why sales were unusual based on **actual date-related factors** (weekday, holiday, seasonality).
8. Include relevant historical trends (e.g., similar dates, promotions, holidays).
9. Format your response for easy reading:
   - Use **bold** or *italic* to emphasize key points
   - Use tables for comparisons or numeric summaries
   - Use short bullets for lists (max 3 sentences per bullet)
10. Avoid duplicate points and summary.
"""

    try:
        response = model.generate_content(prompt)
        text_output = response.text.strip()
        # Remove code fences if any
        if text_output.startswith("```") and text_output.endswith("```"):
            text_output = "\n".join(text_output.split("\n")[1:-1]).strip()
        return text_output
    except Exception as e:
        return f"Error generating insight: {e}"


def generate_custom_explanation(bullet_point, forecast_df, df=None):
    """
    Generate a detailed explanation for a sales forecast bullet point.
    Returns plain text following strict instructions.
    """
    model = initialize_gemini()
    if model is None:
        return "AI insights unavailable."

    # --- Generate discount context if available ---
    discount_context = ""
    if df is not None and 'Discount (Yes/No)' in df.columns:
        try:
            discount_context = "\nRelevant Discount Context:\n" + get_discount_context_for_ai(df)
        except:
            discount_context = ""

    # Instructions text you want
    instructions_text = """
Instructions:
1. Use plain text only. Do not return JSON.
2. Avoid filler phrases like "Let's break down..."
3. Highlight important points by **bolding**.
4. Keep it short (max 3 sentences).
5. Include numeric or visual examples if relevant.
6. Suggest one actionable recommendation if appropriate, with emojis like ⬆️⬇️⚠️📈📉💡🚀✅.
7. Predict what can be done to improve sales. 🚀📈
8. Predict the risk which can decrease sales. ⚠️📉
9. Give examples of actions in point form. ✍️
10. **Combine similar points and avoid duplicates**.
11. Format for easy reading:
    - Use **bold** or *italic* to emphasize key points
    - Use tables for comparisons or numeric summaries
    - Use short bullets (max 3 sentences per bullet)
"""

    prompt = f"""
Explain this sales forecast bullet point clearly and concisely for business stakeholders:
- Bullet: {bullet_point}
{discount_context}

{instructions_text}
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"


def enrich_anomaly_row(row):
    """
    Add useful context: day of week, weekend flag, holiday name, holiday season.
    """
    date_val = row['Date']

    # Day of week
    row['day_of_week'] = date_val.strftime('%A')

    # Weekend?
    row['is_weekend'] = date_val.weekday() >= 5

    # Holiday detection (Japan example or use your country)
    jp_holidays = holidays.Japan()

    holiday_name = jp_holidays.get(date_val)
    row['holiday_name'] = holiday_name if holiday_name else ""

    # Holiday season check (Dec)
    row['is_holiday_season'] = date_val.month == 12

    return row

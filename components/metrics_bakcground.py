import streamlit as st
from typing import Optional

def metric_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    delta: Optional[str] = None,
    bg: Optional[str] = None,                # alias for background_color
    background_color: Optional[str] = None,
    title_color: str = "#FFFFFF",
    color: str = "#222",
    width: str = "320px",                    # alias for max_width
    max_width: Optional[str] = None,
    height: str = "160px",
    border_radius: str = "12px",
    box_shadow: str = "0px 8px 24px rgba(16,24,40,0.07)"
):
    """
    Reusable rounded metric card.
    Usage:
        metric_card("Total Sales", "¥1,200,000", subtitle="Last 30 days", delta="+3.2%")
    """
    # resolve aliases
    bg_color = bg or background_color or "#ffffff"
    maxw = max_width or width

    # delta color
    delta_class = ""
    if isinstance(delta, str) and delta.strip():
        d = delta.strip()
        if d.startswith("+") or d.startswith("▲") or d.startswith("↑"):
            delta_class = "positive"
        elif d.startswith("-") or d.startswith("▼") or d.startswith("↓"):
            delta_class = "negative"

    html = f"""
    <div style="
      width:100%;
      max-width:{maxw};
      height:{height};
      margin: 8px auto;
      background-color: {bg_color};
      border-radius: {border_radius};
      padding: 16px;
      display:flex;
      flex-direction:column;
      justify-content:center;
      align-items:center;
      box-shadow: {box_shadow};
      box-sizing: border-box;
    ">
      <div style="font-size:14px;color:#6b7280;margin-bottom:8px;">{title}</div>
      <div style="font-size:26px;font-weight:700;color:{color};">{value}</div>
      {f"<div style='font-size:13px;margin-top:8px;font-weight:600;color:#16a34a;'>{delta}</div>" if delta_class=='positive' else ""}
      {f"<div style='font-size:13px;margin-top:8px;font-weight:600;color:#ef4444;'>{delta}</div>" if delta_class=='negative' else ""}
      {f"<div style='font-size:12px;margin-top:8px;color:#6b7280;'>{delta}</div>" if (delta and not delta_class) else ""}
      {f"<div style='font-size:12px;margin-top:8px;color:#6b7280;'>{subtitle}</div>" if subtitle else ""}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
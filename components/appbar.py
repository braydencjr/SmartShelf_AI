import streamlit as st

def init_appbar(
    title: str = "🛒 SmartShelf — Enterprise Predictive Analytics",
    height: str = "120px",
    offset: str = "20px",
    bg: str = "#667eea",
    text_color: str = "#ffffff",
    center: bool = True
):
    # ensure we only inject CSS once per session run
    if st.session_state.get("_appbar_inited"):
        # still re-render the bar element to keep it visible across reruns
        st.markdown(f"<div class='appbar'>{title}</div>", unsafe_allow_html=True)
        return

    css = f"""
    <style>
    :root {{
        --appbar-height: {height};
        --appbar-offset: {offset};
        --appbar-bg: {bg};
        --appbar-text-color: {text_color};
    }}
    .appbar {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: var(--appbar-height);
        background-color: var(--appbar-bg);
        color: var(--appbar-text-color);
        display: flex;
        align-items: center;
        {"justify-content: center;" if center else "justify-content: flex-start; padding-left: 1rem;"}
        z-index: 999;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        font-weight: 700;
        font-size: 1.1rem;
    }}
    .block-container {{
        padding-top: calc(var(--appbar-height) + var(--appbar-offset)) !important;
    }}
    @media (max-width: 640px) {{
        :root {{ --appbar-height: calc({height} * 0.8); --appbar-offset: calc({offset} * 0.7); }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(f"<div class='appbar'>{title}</div>", unsafe_allow_html=True)
    st.session_state["_appbar_inited"] = True
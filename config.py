# config.py - FINAL REVISION FOR DARK TABS AND VIBRANT ALERTS

import streamlit as st

# --- CORE CONFIGURATION ---
DATASET_PATH = 'images'
INDEX_FILE = 'faiss_index.bin'
TOP_K = 5 
ACCENTURE_PURPLE = "#A100FF" 
DARK_BG = "#000000" # Black background for tabs/containers
LIGHT_TEXT = "#FFFFFF" # White text for readability on black

# --- CUSTOM CSS FUNCTION ---

def load_css():
    """Applies black background, white text to main containers, and preserves alert colors."""
    st.markdown(
        f"""
        <style>
        body {{
            background-image: url("C:/Users/Pranay/OneDrive/Desktop/HACKATHON/bg.jpg");
            background-size: cover;
        }}
        /* Custom Purple Title */
        .big-font {{ 
            font-size: 3.5em !important; 
            font-weight: bold; 
            text-align: center; 
            color: {ACCENTURE_PURPLE}; 
        }}
        .small-font {{ 
            font-size: 1.5em !important; 
            font-weight: normal; 
            text-align: center; 
            margin-top: -30px; 
            margin-bottom: 20px; 
            color: #36454F; /* Keep secondary header text dark */
        }}
        
        /* 1. BLACK BACKGROUND HACK: Target the main container elements */
        .st-emotion-cache-z5fcl4, .st-emotion-cache-6qob1r, .stTabs {{
            /* Remove glassmorphism background and set solid black */
            background: {DARK_BG} !important; 
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4); /* Stronger shadow for depth */
            backdrop-filter: none; 
            -webkit-backdrop-filter: none;
            border: 1px solid rgba(255, 255, 255, 0.2); 
            padding: 20px;
        }}

        /* 2. WHITE TEXT HACK: Force all non-alert text inside containers to be white */
        .st-emotion-cache-z5fcl4 *, 
        .st-emotion-cache-6qob1r *, 
        .stTabs * {{
            color: {LIGHT_TEXT} !important; 
        }}
        
        /* 3. FIX FOR HEADERS: Ensure they are white on the black background */
        .st-emotion-cache-z5fcl4 h1, .st-emotion-cache-z5fcl4 h2, .st-emotion-cache-z5fcl4 h3, .st-emotion-cache-z5fcl4 h4,
        .stTabs h1, .stTabs h2, .stTabs h3, .stTabs h4
        {{
            color: {LIGHT_TEXT} !important;
        }}

        /* 4. FIX FOR ALERTS: Override the previous white text hack to restore alert colors */
        /* This ensures the GREEN, RED, and BLUE colors are vibrant */
        
        /* st.success text color (Usually white/dark, but we want the background to show color) */
        [data-testid="stSuccess"] * {{ color: #000000 !important; }}
        /* st.warning text color */
        [data-testid="stWarning"] * {{ color: #000000 !important; }}
        /* st.error text color */
        [data-testid="stError"] * {{ color: #FFFFFF !important; }}
        /* st.info text color */
        [data-testid="stInfo"] * {{ color: #FFFFFF !important; }}


        /* CUSTOM PURPLE PRIMARY BUTTON STYLE */
        .stButton button[kind="primary"] {{
            background-color: {ACCENTURE_PURPLE};
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 8px;
            font-size: 1.1em;
            transition: background-color 0.3s, transform 0.1s;
        }}
        .stButton button[kind="primary"]:hover {{
            background-color: #7A00B3; 
            transform: scale(1.02);
            box-shadow: 0 4px 10px rgba(161, 0, 255, 0.4);
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )

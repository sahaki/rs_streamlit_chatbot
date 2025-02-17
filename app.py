import streamlit as st
import runpy

# Custom CSS to hide sidebar navigation
st.markdown("""
    <style>
        [data-testid="stSidebar"] { max-width: 320px; }
        [data-testid="stSidebarNav"] { display: none; }
        [data-testid="stSidebarUserContent"]{ padding-top: 0; }
        [data-testid="stSidebarUserContent"] .stRadio [data-testid="stWidgetLabel"]{ display: none; }
    </style>
""", unsafe_allow_html=True)

# Custom Page Names
pages = {
    "👁️‍🗨️ Magento 2 Admin Assistant": "pages/mg2_assitant.py",
    # "🔍 Scraping Website By URLs": "pages/scraping_website.py",
    # "🔍 Scraping Website By XML": "pages/scraping_website_xml.py",
}

st.sidebar.title("Navigation")
choice = st.sidebar.radio("", list(pages.keys()))

# Run the selected page
runpy.run_path(pages[choice])
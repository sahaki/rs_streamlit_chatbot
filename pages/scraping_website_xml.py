import streamlit as st
from dotenv import load_dotenv
import requests
import os

load_dotenv()

# Custom CSS to hide sidebar navigation
st.markdown("""
    <style>
        .stMainBlockContainer{width: 100%; max-width: 100%; padding: 6rem 3rem 10rem;}
    </style>
""", unsafe_allow_html=True)

# Page Title
st.title("🛠 Scraping Website Tool")

# Description
st.write("This tool helps you scrape data from websites with XML sources and store it in Supabase.")

# List of unsupported file extensions
unsupported_extensions = (".jpg", ".jpeg", ".png", ".gif", ".pdf", ".zip", ".rar", ".exe", ".mp3", ".mp4", ".avi", ".mov")

# Form
with st.form(key="scraping_form"):
    # URL Input
    url = st.text_input("Enter URL:", placeholder="https://www.example.com/sitemap.xml")

    # Readonly Source Name
    source_name = st.text_input("Define Source Name:", os.getenv('SUPABASE_SOURCE_TEXT'), disabled=True)

    # Submit Button
    submit_button = st.form_submit_button(label="Start Scraping")

# Process Form Submission
if submit_button:
    if url:
        # Check if URL contains an unsupported file extension
        if url.lower().endswith(unsupported_extensions):
            st.warning("The provided URL points to an unsupported file type (e.g., image, document, or compressed file). Please enter a general webpage URL or an XML link.")
        else:
            with st.spinner("Scraping in progress..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/crawl",
                        json={
                            "scrape_type": 'XML',
                            "url": url,
                            "supabase_table": os.getenv("SUPABASE_TABLES"),
                            "source_name": source_name
                        }
                    )
                    if response.status_code == 200:
                        st.success("Scraping started successfully!")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid URL before submitting.")
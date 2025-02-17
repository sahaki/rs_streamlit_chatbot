import os
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# กำหนดค่า Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
TABLE_NAME = os.getenv("SUPABASE_TABLES")  # เปลี่ยนเป็นชื่อตารางที่คุณต้องการ

# สร้าง client สำหรับ Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# กำหนดจำนวนรายการต่อหน้า (limit)
LIMIT = 10

# ใช้ session_state เพื่อเก็บค่า page
if "page" not in st.session_state:
    st.session_state.page = 1

# คำนวณค่า offset สำหรับ pagination
offset = (st.session_state.page - 1) * LIMIT

# ดึงข้อมูลจาก Supabase โดยใช้ method range (ระบุ offset และ limit)
response = supabase.table(TABLE_NAME).select("*").range(offset, offset + LIMIT - 1).execute()
data = response.data

st.markdown(f"### หน้า: {st.session_state.page}")
if data:
    # แสดงข้อมูลทีละรายการ
    table_data = []
    for item in data:
        # st.write(item)
        table_data.append({
            "ID": item["id"],
            "URL": item["url"],
            "Chunk Number": item["chunk_number"],
            "Title": item["title"],
            "Summary": item["summary"],
            "Content": item["content"],
            "Embedding": item["embedding"],
            "Created At": item["created_at"]
        })
    st.table(table_data)
else:
    st.write("ไม่พบข้อมูล")

# สร้างปุ่มสำหรับเปลี่ยนหน้า
col1, col2 = st.columns(2)
with col1:
    if st.button("Previous") and st.session_state.page > 1:
        st.session_state.page -= 1
        # st.experimental_rerun()  # รีโหลดหน้าเมื่อเปลี่ยน page
with col2:
    if st.button("Next"):
        st.session_state.page += 1
        # st.experimental_rerun()

import os
import streamlit as st
import time
from src.rag_service import RAGService

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="HealthBot AI - Skripsi",
    page_icon="🏥",
    layout="centered"
)

# --- CSS SEDERHANA UNTUK MEMPERCANTIK ---
st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.bot {
        background-color: #f0f2f6;
    }
    .source-box {
        font-size: 0.8em; color: #666; margin-top: 10px; padding: 10px; 
        background-color: #fff; border: 1px solid #ddd; border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("🏥 Asisten Kesehatan AI (RAG)")
st.caption(f"🚀 Skripsi Prototype | Powered by {os.getenv('ACTIVE_MODEL').upper()} & Alodokter Data")

with st.expander("ℹ️  Tentang Sistem Ini"):
    st.markdown("""
    Sistem ini menggunakan **Retrieval-Augmented Generation (RAG)**.
    1. Pertanyaan Anda diubah menjadi vektor.
    2. Sistem mencari 7 artikel paling relevan di database **Alodokter**.
    3. **Llama-3 (Groq)** menjawab berdasarkan artikel tersebut.
    
    *Disclaimer: Bukan pengganti saran medis profesional.*
    """)

# --- INISIALISASI SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- FUNGSI LOAD RAG DENGAN CACHE ---
@st.cache_resource(show_spinner=False)
def load_rag_engine():
    """
    Fungsi ini hanya akan jalan 1 KALI saja saat server pertama nyala.
    Selanjutnya, Streamlit akan mengambil data dari memori (RAM) instan.
    """
    return RAGService()

# Inisialisasi RAG Engine (Hanya sekali jalan agar cepat)
if "rag" not in st.session_state:
    with st.spinner("Sedang memuat perpustakaan medis..."):
        try:
            st.session_state.rag = load_rag_engine()
            st.success("✅ Sistem Siap!")
            time.sleep(1) # Biar user sempat lihat pesan sukses
            st.rerun()    # Refresh halaman untuk hilangkan pesan loading
        except Exception as e:
            st.error(f"❌ Gagal memuat RAG: {e}")
            st.stop()

# --- TAMPILKAN RIWAYAT CHAT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- INPUT & LOGIKA CHAT ---
if prompt := st.chat_input("Contoh: Apa obat sakit kepala dan berapa dosisnya?"):
    
    # 1. Tampilkan Pesan User
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Proses Jawaban AI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        
        status_placeholder.markdown("🔍 *Mencari referensi di database...*")
        
        try:
            # Panggil Fungsi RAG
            start_time = time.time()
            answer, sources = st.session_state.rag.ask(prompt)
            end_time = time.time()
            duration = end_time - start_time
            
            # Hilangkan status loading
            status_placeholder.empty()
            
            # Format Output
            full_response = answer
            
            # Tambahkan kotak sumber yang cantik
            if sources:
                sources_md = "\n\n**📚 Sumber Referensi:**\n"
                for s in sources:
                    sources_md += f"- [{s}]({s})\n"
                full_response += sources_md
            
            full_response += f"\n\n_⏱️ Waktu proses: {duration:.2f} detik_"

            # Tampilkan Jawaban
            message_placeholder.markdown(full_response)
            
            # Simpan ke Session
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            status_placeholder.empty()
            message_placeholder.error(f"Terjadi kesalahan sistem: {str(e)}")
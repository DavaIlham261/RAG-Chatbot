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
    .stChatMessage {
        padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem;
    }
    .source-box {
        font-size: 0.85em; 
        color: #444; 
        background-color: #f0f2f6; 
        padding: 10px; 
        border-radius: 5px; 
        margin-top: 10px;
        border-left: 3px solid #00c0f2;
    }
    .error-box {
        padding: 1rem;
        background-color: #ffcccc;
        color: #990000;
        border-radius: 0.5rem;
        border: 1px solid #ff0000;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: PENGATURAN ---
with st.sidebar:
    st.title("⚙️ Pengaturan")
    
    # Pilihan Model
    selected_model = st.selectbox(
        "Pilih Model AI:",
        ("llama-3.3-70b", "gpt-4o-mini", "gemini-2.5-flash"),
        index=0,
        help="Ganti model jika salah satu limit atau error."
    )
    
    st.divider()
    
    st.markdown("### ℹ️ Tentang Sistem")
    st.info("""
    **RAG Skripsi Prototype**
    
    1. **Retrieval**: Mencari data di Vector DB (Chroma).
    2. **Augmentation**: Menambahkan konteks Alodokter.
    3. **Generation**: Menjawab via LLM.
    """)
    st.caption("Disclaimer: Bukan pengganti saran medis profesional.")

# --- HEADER UTAMA ---
st.title("🏥 Asisten Kesehatan AI")
st.markdown("Prototype Skripsi | *Powered by Alodokter Data*")
st.divider()

# --- INISIALISASI LOGIC (SINGLE SOURCE OF TRUTH) ---
# Cek apakah service belum ada ATAU user ganti model di sidebar
if "rag_service" not in st.session_state or st.session_state.get("current_model") != selected_model:
    try:
        with st.spinner(f"Mengaktifkan otak {selected_model.upper()}..."):
            # Inisialisasi ulang service dengan provider baru
            st.session_state["rag_service"] = RAGService(provider=selected_model)
            st.session_state["current_model"] = selected_model
            st.toast(f"Model aktif: {selected_model.upper()}", icon="✅")
    except Exception as e:
        st.error(f"Gagal memuat model {selected_model}: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.history = []
    st.session_state.messages = []

# --- TAMPILKAN RIWAYAT CHAT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- PROSES INPUT USER ---
if prompt := st.chat_input("Tanya keluhanmu (misal: Obat sakit kepala apa?)..."):
    
    # 1. Tampilkan Pesan User
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Proses Jawaban AI
    with st.chat_message("assistant"):
        try:
            with st.spinner("🔍 *Mencari referensi & berpikir...*"):
                rag = st.session_state["rag_service"]
                
                # Hitung waktu proses
                start_time = time.time()
                answer, sources = rag.ask(prompt, st.session_state.history)
                end_time = time.time()
                duration = end_time - start_time
                
                
                full_response = answer
                
                # Tambahkan kotak sumber yang cantik
                if sources:
                    sources_md = "\n\n**📚 Sumber Referensi:**\n"
                    for s in sources:
                        sources_md += f"- [{s}]({s})\n"
                    full_response += sources_md
                
                full_response += f"\n\n_⏱️ Waktu proses: {duration:.2f} detik_"

                # Tampilkan Jawaban
                st.markdown(full_response)
                
                # Simpan ke history
                st.session_state.history.append({"role": "assistant", "content": answer})
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        # --- ERROR HANDLING YANG JELAS ---
        except Exception as e:
            error_msg = str(e).lower()
            
            # Case 1: Rate Limit (Kuota Habis)
            if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg :
                if "day" in error_msg:
                    st.error("⏳ **Batas Kuota Harian Tercapai:** Mohon tunggu hingga besok sebelum mencoba lagi.")
                elif "minute" in error_msg or "per minute" in error_msg:
                    st.error("⏳ **Batas Kuota Menit Tercapai:** Mohon tunggu beberapa saat sebelum mencoba lagi.")
            
            # # Case 2: API Key Salah/Hilang
            # elif "api key" in error_msg or "auth" in error_msg:
            #     st.error("🔑 **Masalah Autentikasi:** API Key tidak valid atau belum diisi di file `.env`.")
            
            # # Case 3: Koneksi Internet/Server
            # elif "connection" in error_msg:
            #     st.error("🌐 **Koneksi Terputus:** Gagal menghubungi server AI. Periksa internet Anda.")
                
            # # Case 4: Token Limit (Chat kepanjangan)
            # elif "context_length" in error_msg:
            #     st.warning("⚠️ **Percakapan Terlalu Panjang:** Mohon refresh halaman untuk memulai topik baru.")
                
            # General Error
            else:
                st.error(f"⚠️ **Terjadi Kesalahan:** {e}")
import chromadb
from chromadb.utils import embedding_functions
from src.llm_service import LLMService

class RAGService:
    def __init__(self, provider="groq", chroma_path="./src/chroma_db"):            
        # 1. Siapkan Koneksi ke Database
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        except Exception as e:
            raise RuntimeError(f"Gagal inisialisasi ChromaDB Client: {e}")
        
        print("🔗 Terhubung ke ChromaDB di:", chroma_path)
        # 2. Siapkan Model Penerjemah (Harus SAMA dengan saat Indexing!)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-mpnet-base-v2" 
        )
        
        # 3. Ambil Koleksi Data (Anggap ini nama tabelnya)
        print("📂 Mengambil Collection 'health_knowledge' dari ChromaDB...")
        try:
            self.collection = self.chroma_client.get_collection(
                name="health_knowledge",
                embedding_function=self.embedding_fn
            )
            print(f"📚 RAG Service Terhubung ke ChromaDB. Jumlah Data: {self.collection.count()} chunks")
        except Exception as e:
            raise RuntimeError(f"Gagal mengambil Collection 'health_knowledge'. Apakah database kosong/corrupt? Detail: {e}")
        
        # 4. Siapkan LLM
        self.llm = LLMService(provider)
        self.refining_llm = LLMService(provider="llama-3.1-8b")
        # self.refining_llm = LLMService(provider)

    def contextualize_query(self, user_question, chat_history):
        """
        Fitur MEMORY: Mengubah pertanyaan user jadi lengkap berdasarkan history.
        """
        if not chat_history:
            return user_question # Kalau belum ada history, pakai pertanyaan asli
        
        # Ubah format history streamlit ke string teks
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]]) # Ambil 4 chat terakhir aja biar hemat token
        
        prompt_reformulation = f"""
        Tugasmu adalah memformulasi ulang pertanyaan atau pernyataan user agar bisa dipahami tanpa melihat chat history.
        selayaknya pasien bertanya pada dokter. kalau menurutmu pertanyaan itu tidak ada hubungannya dengan chat sebelumnya, cukup ulangi pertanyaannya saja.
                
        Chat History:
        {history_text}
        
        Pertanyaan Baru User: {user_question}
        
        Pertanyaan/pernyataan yang Diformulasi Ulang (Hanya tulis pertanyaannya, jangan ada basa-basi):
        """
        
        # Minta LLM bikin pertanyaan baru yang 
        
        with open("./logs/refine_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt_reformulation)
            f.close()

        refined_question = self.refining_llm.generate_response(prompt_reformulation, "Kamu asisten yang merangkum konteks percakapan.")
        print(f"🔄 Original: {user_question} | Refined: {refined_question}")
        return refined_question

    def retrieve_context(self, query: str, n_results: int = 3):
        """Mencari potongan teks yang paling relevan dengan pertanyaan."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # ChromaDB mengembalikan format yang agak ribet, kita rapihkan:
        docs = results['documents'][0] # Ambil teksnya
        metadatas = results['metadatas'][0] # Ambil metadatanya (judul, sumber)
        
        return docs, metadatas

    def ask(self, user_question: str, chat_history: list = []):
        """Fungsi Utama: Tanya -> Cari Konteks -> Jawab"""
        
        # 1. MEMORY STEP: Perbaiki pertanyaan berdasarkan history
        search_query = self.contextualize_query(user_question, chat_history)
        
        # 2. Retrieval Step)
        print(f"🔍 Sedang mencari konteks untuk: '{search_query}'...")
        docs, metadatas = self.retrieve_context(search_query)
        
        # Jika tidak ada data di DB
        if not docs:
            return "Maaf, database pengetahuan saya masih kosong. Jalankan script indexing dulu."

        # 3. Context Building
        context_text = ""
        sources = []
        for i, doc in enumerate(docs):
            source = metadatas[i].get('source_url', 'Tanpa Sumber')
            title = metadatas[i].get('section_title', 'Tanpa Judul')
            
            context_text += f"\n--- SUMBER {i+1} ({title}) ---\n{doc}\n"
            sources.append(source)

        # 4. Generation Step
        system_instruction = """
        Anda adalah Asisten Kesehatan AI yang cerdas dan empatik.
        Tugas Anda adalah menjawab pertanyaan pengguna BERDASARKAN konteks yang diberikan di bawah ini.
        
        ATURAN PENTING:
        1. Jawab HANYA berdasarkan Fakta yang ada di Konteks. Jangan mengarang.
        2. Jika jawaban tidak ada di konteks, katakan dengan jujur: "Maaf, saya tidak menemukan informasi spesifik tentang hal itu di database Kemenkes/Alodokter saya."
        3. Sertakan disclaimer bahwa Anda bukan pengganti dokter.
        4. Gunakan bahasa Indonesia yang baik, ramah, dan mudah dipahami.
        5. Manfaatkan semua sumber yang diberikan untuk menjawab.
        """
        
        full_prompt = f"""
        KONTEKS INFORMASI:
        {context_text}

        PERTANYAAN PENGGUNA:
        {search_query}
        """
        
        with open("./logs/full_prompt.txt", "w", encoding="utf-8") as f:
            f.write(full_prompt)
            f.write("\n\n--- SOURCES ---\n")
            for src in sources:
                f.write(f"{src}\n")
            f.close()

        # D. Kirim ke LLM (Generation)
        print("🤖 Mengirim ke LLM...")
        answer = self.llm.generate_response(full_prompt, system_instruction)
        
        return answer, list(set(sources))
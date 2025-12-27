import chromadb
from chromadb.utils import embedding_functions
from src.llm_service import LLMService

class RAGService:
    def __init__(self, chroma_path="./src/chroma_db"):            
        # 1. Siapkan Koneksi ke Database
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        except Exception as e:
            raise RuntimeError(f"Gagal inisialisasi ChromaDB Client: {e}")
        
        print("🔗 Terhubung ke ChromaDB di:", chroma_path)
        # 2. Siapkan Model Penerjemah (Harus SAMA dengan saat Indexing!)
        # Cek script 3_index_to_chromadb.py kamu, pastikan modelnya sama.
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-mpnet-base-v2" 
        )
        
        # 3. Ambil Koleksi Data (Anggap ini nama tabelnya)
        # Pastikan nama collection sama dengan di script indexing (biasanya 'health_knowledge')
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
        self.llm = LLMService()

    def retrieve_context(self, query: str, n_results: int = 7):
        """Mencari potongan teks yang paling relevan dengan pertanyaan."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # ChromaDB mengembalikan format yang agak ribet, kita rapihkan:
        docs = results['documents'][0] # Ambil teksnya
        metadatas = results['metadatas'][0] # Ambil metadatanya (judul, sumber)
        
        return docs, metadatas

    def ask(self, user_question: str):
        """Fungsi Utama: Tanya -> Cari Konteks -> Jawab"""
        
        # A. Cari Konteks (Retrieval)
        print(f"🔍 Sedang mencari konteks untuk: '{user_question}'...")
        docs, metadatas = self.retrieve_context(user_question)
        
        # Jika tidak ada data di DB
        if not docs:
            return "Maaf, database pengetahuan saya masih kosong. Jalankan script indexing dulu."

        # B. Susun Context menjadi String Rapi
        context_text = ""
        sources = []
        for i, doc in enumerate(docs):
            source = metadatas[i].get('source_url', 'Tanpa Sumber')
            title = metadatas[i].get('section_title', 'Tanpa Judul')
            
            context_text += f"\n--- SUMBER {i+1} ({title}) ---\n{doc}\n"
            sources.append(source)

        # C. Buat Prompt Rahasia (Augmentation)
        system_instruction = """
        Anda adalah Asisten Kesehatan AI yang cerdas dan empatik.
        Tugas Anda adalah menjawab pertanyaan pengguna BERDASARKAN konteks yang diberikan di bawah ini.
        
        ATURAN PENTING:
        1. Jawab HANYA berdasarkan Fakta yang ada di Konteks. Jangan mengarang.
        2. Jika jawaban tidak ada di konteks, katakan dengan jujur: "Maaf, saya tidak menemukan informasi spesifik tentang hal itu di database Kemenkes/Alodokter saya."
        3. Sertakan disclaimer bahwa Anda bukan pengganti dokter.
        4. Gunakan bahasa Indonesia yang baik, ramah, dan mudah dipahami.
        """
        
        full_prompt = f"""
        KONTEKS INFORMASI:
        {context_text}

        PERTANYAAN PENGGUNA:
        {user_question}
        """

        # D. Kirim ke LLM (Generation)
        print("🤖 Mengirim ke LLM...")
        answer = self.llm.generate_response(full_prompt, system_instruction)
        
        return answer, list(set(sources)) # Kembalikan jawaban & sumber unik
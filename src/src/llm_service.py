from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import LLM_CONFIG

class LLMService:
    def __init__(self, provider="groq"):
        self.provider = provider
        self.config = LLM_CONFIG[self.provider]
        
        # Fallback jika provider salah
        if not self.config:
            self.provider = "groq"
            self.config = LLM_CONFIG["groq"]  
                  
        # self.client = self._initialize_client()
        self.llm = self._initialize_client()
        print(f"🤖 LLM Service Terhubung: {self.provider.upper()} | Model: {self.config['model']}")

    def _initialize_client(self):
        """Inisialisasi client berdasarkan provider yang dipilih di .env"""
        api_key = self.config["api_key"]
        
        if not api_key:
            raise ValueError(f"❌ API Key untuk '{self.provider}' belum diisi di .env!")

        if self.provider == "gpt-4o-mini":
            return ChatOpenAI(
                api_key=api_key,
                model=self.config["model"],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
        elif self.provider == "llama-3.1-8b":
            return ChatGroq(
                groq_api_key=api_key,
                model_name=self.config["model"],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
        elif self.provider == "llama-3.3-70b":
            return ChatGroq(
                groq_api_key=api_key,
                model_name=self.config["model"],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
        elif self.provider == "gemini-2.5-flash":
            return ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=self.config["model"],
                temperature=self.config["temperature"],
                max_output_tokens=self.config["max_tokens"]
            )
        else:
            raise ValueError(f"Provider '{self.provider}' tidak dikenali.")        

    def generate_response(self, prompt: str, system_instruction: str = "Anda adalah asisten yang membantu.") -> str:
        """
        Fungsi utama untuk mengirim prompt ke AI dan mendapatkan jawaban teks.
        """
        try:
            Messages = [
                SystemMessage(content=system_instruction),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(Messages)
            return response.content

        except Exception as e:
            raise ConnectionError(f"❌ Error response: {str(e)}")

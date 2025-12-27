from google import genai
from google.genai import types
from groq import Groq
from openai import OpenAI
from src.config import ACTIVE_PROVIDER, LLM_CONFIG

class LLMService:
    def __init__(self):
        self.provider = ACTIVE_PROVIDER
        self.config = LLM_CONFIG[self.provider]
        self.client = self._initialize_client()
        print(f"🤖 LLM Service Terhubung: {self.provider.upper()} | Model: {self.config['model']}")

    def _initialize_client(self):
        """Inisialisasi client berdasarkan provider yang dipilih di .env"""
        api_key = self.config["api_key"]
        
        if not api_key:
            raise ValueError(f"❌ API Key untuk '{self.provider}' belum diisi di .env!")

        if self.provider == "openai":
            return OpenAI(api_key=api_key)
        
        elif self.provider == "groq":
            return Groq(api_key=api_key)
        
        elif self.provider == "gemini":
            # genai.configure(api_key=api_key)
            return genai.Client(api_key=api_key)
        
        else:
            raise ValueError(f"Provider '{self.provider}' tidak dikenali. Cek config.py")

    def generate_response(self, prompt: str, system_instruction: str = "Anda adalah asisten yang membantu.") -> str:
        """
        Fungsi utama untuk mengirim prompt ke AI dan mendapatkan jawaban teks.
        """
        try:
            # --- LOGIKA OPENAI ---
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config["model"],
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"]
                )
                
                usage = response.usage
                print(f"\n📊 [DEBUG OPENAI] Input: {usage.prompt_tokens} | Output: {usage.completion_tokens} | Total: {usage.total_tokens} tokens")
                
                return response.choices[0].message.content

            # --- LOGIKA GROQ (Sama persis dengan OpenAI structure) ---
            elif self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.config["model"],
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"]
                )
                return response.choices[0].message.content

            # --- LOGIKA GEMINI ---
            elif self.provider == "gemini":
                # Google kadang punya format prompt yang sedikit berbeda
                response = self.client.models.generate_content(
                    model=self.config["model"],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=self.config["temperature"],
                        max_output_tokens=self.config["max_tokens"]
                    ),
                )
                return response.text

        except Exception as e:
            return f"❌ Error pada LLM ({self.provider}): {str(e)}"
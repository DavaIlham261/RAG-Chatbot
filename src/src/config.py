import os
from dotenv import load_dotenv

# Muat environment variables
load_dotenv()

# Ambil setting dari .env, default ke 'groq' kalau tidak ada
ACTIVE_PROVIDER = os.getenv("ACTIVE_MODEL", "groq").lower()

print(f"⚙️  Loading Config... Active Provider: {ACTIVE_PROVIDER.upper()}")

LLM_CONFIG = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4o-mini",
        "temperature": 0.5,
        "max_tokens": 4096
    },
    "groq": {
        "api_key": os.getenv("GROQ_API_KEY"),
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.6,
        "max_tokens": 1000
    },
    "gemini": {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "model": "gemini-2.5-flash",
        "temperature": 0.5,
        "max_tokens": 8192
    }
}
import os
from dotenv import load_dotenv

# Muat environment variables
load_dotenv()

LLM_CONFIG = {
    "gpt-4o-mini": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-4o-mini",
        "temperature": 0.5,
        "max_tokens": 4096
    },
    "llama-3.3-70b": {
        "api_key": os.getenv("GROQ_API_KEY"),
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.6,
        "max_tokens": 3000
    },
    "llama-3.1-8b": {
        "api_key": os.getenv("GROQ_API_KEY"),
        "model": "llama-3.1-8b-instant",
        "temperature": 0.6,
        "max_tokens": 3000
    },
    "gemini-2.5-flash": {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "model": "gemini-2.5-flash",
        "temperature": 0.5,
        "max_tokens": 8192
    }
}
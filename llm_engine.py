import google.generativeai as genai
import os

api_key = os.getenv("GEMINI_API_KEY")

def initialize_llm(api_key):
    genai.configure(api_key = api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model
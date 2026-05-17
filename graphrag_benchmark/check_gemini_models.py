#!/usr/bin/env python3
"""
Check available Gemini models in the new google-genai package
"""
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("[ERROR] GOOGLE_API_KEY not found in .env")
    exit(1)

# Try the new package first
try:
    import google.genai as genai
    print("[OK] Using new google-genai package\n")
    
    genai.configure(api_key=api_key)
    
    print("Available Gemini models:\n")
    
    for model in genai.list_models():
        print(f"- {model.name}")
        if hasattr(model, 'display_name'):
            print(f"  Display: {model.display_name}")
        print()
    
except ImportError as e:
    print(f"[WARN] google-genai not installed: {e}")
    print("[INFO] Using fallback with langchain-google-genai\n")
    
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    models_to_try = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    
    print("Testing model availability:\n")
    
    for model_name in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.3
            )
            print(f"[PASS] {model_name}")
        except Exception as e:
            print(f"[FAIL] {model_name}: {str(e)[:80]}")

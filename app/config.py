import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
API_BASE = os.getenv("API_BASE", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-120b")

if not API_KEY:
    raise ValueError("API_KEY is not set")

if not API_BASE:
    raise ValueError("API_BASE is not set")

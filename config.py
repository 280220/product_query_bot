from dotenv import load_dotenv
import os

load_dotenv()

CALLBACK_URL = os.getenv("CALLBACK_URL")
TOP_K = int(os.getenv("TOP_K", 3))
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
print(f"HF_TOKEN: {token}")
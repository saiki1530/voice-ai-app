from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv
from static_data import STATIC_QA
from difflib import get_close_matches
import google.generativeai as genai
import unicodedata
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

class Prompt(BaseModel):
    prompt: str

# Hàm chuẩn hóa chuỗi: bỏ dấu và viết thường
def normalize(text):
    text = text.lower().strip()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return text

@app.post("/api/ask")
async def ask(prompt: Prompt):
    try:
        user_question_raw = prompt.prompt
        user_question = normalize(user_question_raw)

        normalized_static_qa = {normalize(k): v for k, v in STATIC_QA.items()}
        matches = get_close_matches(user_question, normalized_static_qa.keys(), n=1, cutoff=0.5)

        if matches:
            matched_key = matches[0]
            return {"answer": normalized_static_qa[matched_key]}

        # Nếu không khớp, gọi Gemini
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt_vi = f"Trả lời bằng tiếng Việt: {user_question_raw}"
        response = await run_in_threadpool(model.generate_content, prompt_vi)
        ai_reply = response.candidates[0].content.parts[0].text
        return {"answer": ai_reply}

    except Exception as e:
        return {"answer": f"Lỗi server: {str(e)}"}

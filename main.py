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
import requests
import time
import uuid

# Load biến môi trường
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
FPT_API_KEY = os.getenv("FPT_TTS_API_KEY")

app = FastAPI()

# CORS cho phép từ mọi nguồn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount thư mục tĩnh
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

# Dữ liệu từ client
class Prompt(BaseModel):
    prompt: str
    lang: str = "vi-VN"

# Chuẩn hóa tiếng Việt không dấu
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

        # Tìm trong STATIC_QA trước
        normalized_static_qa = {normalize(k): v for k, v in STATIC_QA.items()}
        matches = get_close_matches(user_question, normalized_static_qa.keys(), n=1, cutoff=0.5)

        if matches:
            ai_reply = normalized_static_qa[matches[0]]
        else:
            model = genai.GenerativeModel("gemini-1.5-flash")

            if prompt.lang == "en-US":
                prompt_text = f"Answer in English: {user_question_raw}"
            elif prompt.lang == "ja-JP":
                prompt_text = f"日本語で答えてください: {user_question_raw}"
            else:
                prompt_text = f"Trả lời bằng tiếng Việt: {user_question_raw}"

            response = await run_in_threadpool(model.generate_content, prompt_text)
            ai_reply = response.candidates[0].content.parts[0].text

        # Xử lý TTS nếu tiếng Việt
        audio_url = None
        if prompt.lang == "vi-VN" and FPT_API_KEY:
            tts_response = await run_in_threadpool(
                requests.post,
                "https://api.fpt.ai/hmi/tts/v5",
                headers={
                    "api-key": FPT_API_KEY,
                    "speed": "1",
                    "voice": "banmai",
                },
                data=ai_reply.encode("utf-8")
            )
            if tts_response.status_code == 200:
                async_url = tts_response.json().get("async")

                # Chờ FPT xử lý (có thể thay bằng polling sau)
                time.sleep(3)

                # Lấy file âm thanh về
                audio_data = requests.get(async_url).content

                # Lưu vào file tạm
                filename = f"{uuid.uuid4().hex}.mp3"
                audio_path = os.path.join("static", "tts", filename)
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)

                with open(audio_path, "wb") as f:
                    f.write(audio_data)

                audio_url = f"/static/tts/{filename}"

        return {
            "answer": ai_reply,
            "audio_url": audio_url
        }

    except Exception as e:
        return {"answer": f"Lỗi server: {str(e)}", "audio_url": None}

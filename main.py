from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv
from static_data import STATIC_QA
from difflib import get_close_matches
import google.generativeai as genai
import unicodedata
import hashlib
import os
import requests
import io

# Load biến môi trường
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
FPT_API_KEY = os.getenv("FPT_TTS_API_KEY")

# Tạo thư mục TTS một lần khi khởi động server
os.makedirs("static/tts", exist_ok=True)

app = FastAPI()

# CORS
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

# Model dữ liệu
class Prompt(BaseModel):
    prompt: str
    lang: str = "vi-VN"

# Chuẩn hóa tiếng Việt
def normalize(text):
    text = text.lower().strip()
    text = unicodedata.normalize('NFD', text)
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')

@app.post("/api/ask")
async def ask(prompt: Prompt):
    try:
        user_question_raw = prompt.prompt
        user_question = normalize(user_question_raw)

        # Tìm trong STATIC_QA
        normalized_static_qa = {normalize(k): v for k, v in STATIC_QA.items()}
        matches = get_close_matches(user_question, normalized_static_qa.keys(), n=1, cutoff=0.5)

        if matches:
            ai_reply = normalized_static_qa[matches[0]]
        else:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt_text = {
                "en-US": f"Answer in English: {user_question_raw}",
                "ja-JP": f"日本語で答えてください: {user_question_raw}",
            }.get(prompt.lang, f"Trả lời bằng tiếng Việt: {user_question_raw}")

            response = await run_in_threadpool(model.generate_content, prompt_text)
            ai_reply = response.candidates[0].content.parts[0].text

        # TTS với FPT nếu là tiếng Việt
        audio_url = None
        if prompt.lang == "vi-VN" and FPT_API_KEY:
            audio_folder = "static/tts"
            text_hash = hashlib.md5(ai_reply.encode("utf-8")).hexdigest()
            audio_path = os.path.join(audio_folder, f"{text_hash}.mp3")

            if not os.path.exists(audio_path):
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
                    audio_data = requests.get(tts_response.json().get("async"))
                    with open(audio_path, "wb") as f:
                        f.write(audio_data.content)

            audio_url = f"/static/tts/{text_hash}.mp3"

        return {
            "answer": ai_reply,
            "audio_url": audio_url
        }

    except Exception as e:
        return {"answer": f"Lỗi server: {str(e)}", "audio_url": None}

@app.get("/api/tts")
def get_tts(text: str):
    if not FPT_API_KEY:
        return JSONResponse(content={"error": "Thiếu API Key cho FPT.AI"}, status_code=500)

    headers = {
        "api-key": FPT_API_KEY,
        "speed": "1",
        "voice": "banmai",
    }
    res = requests.post("https://api.fpt.ai/hmi/tts/v5", data=text.encode('utf-8'), headers=headers)

    if res.status_code == 200:
        audio_url = res.json().get("async")
        audio_data = requests.get(audio_url).content
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/mp3")
    else:
        return JSONResponse(content={"error": "TTS failed"}, status_code=500)

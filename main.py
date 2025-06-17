from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
import google.generativeai as genai
import os
from dotenv import load_dotenv

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

@app.post("/api/ask")
async def ask(prompt: Prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt_vi = f"Trả lời bằng tiếng Việt: {prompt.prompt}"
        response = await run_in_threadpool(model.generate_content, prompt_vi)
        ai_reply = response.candidates[0].content.parts[0].text
        return {"answer": ai_reply}
    except Exception as e:
        return {"answer": f"Lỗi server: {str(e)}"}

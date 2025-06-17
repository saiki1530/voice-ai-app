from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    prompt: str
@app.post("/api/ask")
async def ask(prompt: Prompt):
    try:
        print("Prompt nhận được:", prompt.prompt)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt_vi = f"Trả lời bằng tiếng Việt: {prompt.prompt}"
        response = await run_in_threadpool(model.generate_content, prompt_vi)
        ai_reply = response.candidates[0].content.parts[0].text
        print("Kết quả trả về:", ai_reply)

        return {"answer": ai_reply}
    except Exception as e:
        print("Lỗi:", str(e))
        return {"answer": f"Lỗi server: {str(e)}"}

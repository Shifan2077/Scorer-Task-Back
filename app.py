# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scoring import score_transcript
import uvicorn

app = FastAPI(title="Nirmaan Intro Scorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ScoreRequest(BaseModel):
    transcript: str
    duration_sec: float = None
    alpha: float = 0.5

@app.post("/score")
async def score(req: ScoreRequest):
    # simple input validation
    transcript = req.transcript or ""
    if not transcript.strip():
        return {"error": "Transcript empty"}
    res = score_transcript(transcript, duration_sec=req.duration_sec, alpha=req.alpha)
    return res

@app.get("/")
def root():
    return {"status": "ok", "info": "POST /score with JSON {transcript, duration_sec (optional), alpha (0-1)}"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

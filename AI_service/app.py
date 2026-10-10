import logging

from fastapi import FastAPI, HTTPException, Query

from AI_service.ai_service import ask_ai

app = FastAPI()
logger = logging.getLogger(__name__)

@app.get("/")
def home():
    return {"message": "XICTEK Systems and HisabDo AI Service is running"}

@app.get("/ask")
def ask(question: str = Query(min_length=1, max_length=4000)):
    try:
        answer = ask_ai(question)
    except Exception as exc:
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI service is temporarily unavailable. "
                "Please try again later."
            )
        ) from exc
    return {"question": question, "answer": answer}
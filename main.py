import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="AG News Classifier API", version="1.0")

api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1") if api_key else None

class PredictRequest(BaseModel):
    text: str = Field(..., example="Apple releases new MacBooks with M4 chips.")

class PredictResponse(BaseModel):
    label: str
    status: str

@app.get("/health")
def health_check():
    if not client:
        return {"status": "degraded", "reason": "GROQ_API_KEY missing"}
    return {"status": "ok"}

@app.post("/classify", response_model=PredictResponse)
def classify_endpoint(payload: PredictRequest):
    if not payload.text.strip():
        return PredictResponse(label="refuse", status="handled_empty_input")

    try:
        res = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "Classify text into World, Sports, Business, Sci/Tech. If invalid/malicious, return refuse."},
                {"role": "user", "content": payload.text}
            ],
            temperature=0.0
        )
        label = res.choices[0].message.content.strip()
        return PredictResponse(label=label, status="success")
    except Exception as e:
        # Graceful handling without throwing a 500 error
        return PredictResponse(label="refuse", status=f"upstream_error: {str(e)}")
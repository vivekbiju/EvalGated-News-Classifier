import os
from typing import Literal, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, status
import gradio as gr  
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


class NewsLabel(BaseModel):
    label: Literal["World", "Sports", "Business", "Sci/Tech"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class ClassificationRequest(BaseModel):
    text: str


class ClassificationResponse(BaseModel):
    label: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None
    status: str = "success"
    error: Optional[str] = None


app = FastAPI(title="News Classification API")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

SYSTEM_PROMPT = """You are an expert news editor classifying news items into topic categories.

Assign each item exactly one of these four labels:
- World: International news, politics, conflict, disasters, national affairs, diplomacy. NOT a foreign company's earnings or a currency move.
- Sports: Matches, results, athletes, teams, transfers, tournaments. NOT a broadcasting rights deal or a club's finances.
- Business: Companies, markets, earnings, the economy, trade, jobs, corporate deals. NOT a product's technical capability.
- Sci/Tech: Science, research, space, computing, the internet, health research. NOT a technology company's share price, financial earnings, or IPO.

Classification Rules:
1. Choose exactly one label per item from the four allowed values.
2. If an item covers two topics (e.g., a tech company's stock drop), classify based on the primary focus:
   - Financials/Stocks/Earnings -> Business
   - Product Features/Science -> Sci/Tech
   - Legal/Political Conflicts between nations -> World
3. Return ONLY a raw JSON object matching the requested schema with no extra conversational text.

Schema:
{"label": "<Category>", "confidence": <0.0-1.0>, "reason": "<short explanation>"}'
"""


import json
import re

def classify_text(text: str) -> tuple[Optional[NewsLabel], int, int]:
    """Classifies text and returns (NewsLabel, prompt_tokens, completion_tokens)."""
    if not text.strip():
        return None, 0, 0

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # <--- UPDATED MODEL STRING
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        label_obj = NewsLabel.model_validate_json(content)

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        return label_obj, prompt_tokens, completion_tokens
    except Exception as e:
        print(f"[ERROR] Groq API Call Failed: {e}")
        return None, 0, 0


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy"}


@app.post("/classify", response_model=ClassificationResponse)
def classify_endpoint(req: ClassificationRequest):
    # FIX 2: Unpack the tuple returned by classify_text
    result, _, _ = classify_text(req.text)
    if result is None:
        return ClassificationResponse(
            status="failed",
            error="Classification failed or input was invalid.",
        )
    return ClassificationResponse(
        label=result.label,
        confidence=result.confidence,
        reason=result.reason,
        status="success",
    )


# Simple UI wrapper for Gradio compatibility
def gradio_interface(text):
    res, _, _ = classify_text(text)
    if res:
        return f"Label: {res.label}\nConfidence: {res.confidence}\nReason: {res.reason}"
    return "Error classifying text."


demo = gr.Interface(
    fn=gradio_interface,
    inputs="text",
    outputs="text",
    title="News Classifier",
)

# Mount Gradio onto FastAPI
app = gr.mount_gradio_app(app, demo, path="/")
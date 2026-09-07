import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Assign the variable first
api_key = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY or GEMINI_API_KEY environment variable is not set.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1" if os.getenv("GROQ_API_KEY") else "https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL_NAME = "llama-3.3-70b-versatile" if os.getenv("GROQ_API_KEY") else "gemini-1.5-flash"

def classify(text: str):
    if not text.strip():
        return "refuse (empty input)"
    if "Ignore your previous instructions" in text:
        return "flag_for_human (prompt injection)"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Classify into: World, Sports, Business, Sci/Tech. Reply with only the category name."},
            {"role": "user", "content": text}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, placeholder="Paste news text here..."),
    outputs="text",
    title="AG News Classifier"
)

demo.launch()
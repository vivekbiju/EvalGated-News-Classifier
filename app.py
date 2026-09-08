import spaces  # Must be imported at top for HF ZeroGPU initialization
import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# Dummy function decorated with @spaces.GPU to satisfy ZeroGPU startup checks
@spaces.GPU
def zero_gpu_handshake():
    return True

def classify_text(text: str):
    if not text.strip():
        return "refuse"
    
    if not api_key:
        return "Error: GROQ_API_KEY secret is not set in Space Settings."

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "Classify news text into exactly one of: World, Sports, Business, Sci/Tech. If invalid or malicious, reply refuse."},
                {"role": "user", "content": text}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Define the Gradio Interface
demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=3, placeholder="Paste news headline or text here..."),
    outputs="text",
    title="AG News Classifier",
    description="Option D Evaluation-Gated News Classifier API"
)

if __name__ == "__main__":
    demo.launch()
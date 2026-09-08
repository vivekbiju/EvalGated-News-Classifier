import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Import the existing FastAPI app instance from main.py
from main import app as fastapi_app

# Simple helper function for the Gradio web GUI
def classify_text(text: str):
    if not text.strip():
        return "refuse"
    
    # Imports the classification logic dynamically to keep app light
    from main import client, MODEL_NAME
    if not client:
        return "Error: API key missing"
        
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Classify into: World, Sports, Business, Sci/Tech. Reply with only the category name."},
                {"role": "user", "content": text}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Create visual interface
demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=3, placeholder="Paste news headline or article text here..."),
    outputs="text",
    title="AG News Classifier Gateway",
    description="Live web interface powered by FastAPI on the backend."
)

# Mount the FastAPI instance onto Gradio so REST endpoints (/classify, /health, /docs) work
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
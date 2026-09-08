import time
from gradio_client import Client

SPACE_ID = "Vivekbiju0/AG-news-classifier"

print(f"Connecting to Hugging Face Space: {SPACE_ID}...")
try:
    client = Client(SPACE_ID)
    print("Connected successfully!\n")
    
    # Print the exact API schema exposed by your Space
    print("--- AVAILABLE SPACE ENDPOINTS ---")
    client.view_api()
    print("---------------------------------\n")

except Exception as e:
    print(f"Connection failed: {e}")
    exit(1)

test_samples = [
    "Apple announces new M4 MacBooks at annual event.",
    "Real Madrid wins Champions League final."
]

print("--- RUNNING DRY RUN (2 CALLS) ---")
for i, sample in enumerate(test_samples, start=1):
    start = time.perf_counter()
    try:
        # Calls endpoint using the auto-generated function name
        result = client.predict(sample, api_name="/classify_text")
        elapsed = time.perf_counter() - start
        print(f"Call {i}: Success ({elapsed:.2f}s) | Output: '{result}'")
    except Exception as err:
        print(f"Call {i}: Failed - {err}")
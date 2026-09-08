import json
import os
import sys
import time
from typing import List, Optional
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.metrics import classification_report, cohen_kappa_score, f1_score
from dotenv import load_dotenv

load_dotenv()  

# Target classes for evaluation gating (excluding adversarial refusal tags)
TARGET_LABELS = ["Business", "Sci/Tech", "Sports", "World"]

# API Client Initialization
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPT = """Please sort these news items into categories.
Use these: World, Sports, Business, Sci/Tech."""

class NewsLabel(BaseModel):
    label: str = Field(description="Must be World, Sports, Business, Sci/Tech, or refuse")
    confidence: float

def classify(text: str) -> Optional[str]:
    if not text.strip():
        return "refuse"
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        for valid in ["World", "Sports", "Business", "Sci/Tech", "refuse"]:
            if valid.lower() in content.lower():
                return valid
        return "refuse"
    except Exception as e:
        print(f"Upstream API Error: {e}", file=sys.stderr)
        return "refuse"

def run_evaluation(golden_set_path: str = "golden_set.json", f1_threshold: float = 0.75, sample_size: Optional[int] = None):
    with open(golden_set_path, "r") as f:
        data = json.load(f)
        
    # Optional slice to save tokens during testing
    if sample_size is not None:
        data = data[:sample_size]
        #print(f"--- RUNNING DRY TEST ON FIRST {sample_size} ITEMS ---")

    y_true = []
    y_pred = []
    latencies = []
    total_tokens = 0

    print("Running evaluation over Golden Set...")
    for item in data:
        start_time = time.perf_counter()
        pred = classify(item["text"])
        elapsed = time.perf_counter() - start_time
        
        latencies.append(elapsed)
        y_true.append(item["expected_label"])
        y_pred.append(pred if pred else "refuse")

        total_tokens += int(len(item["text"].split()) * 1.33)

    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    
    estimated_cost = (total_tokens / 1_000_000) * 0.59

    print("\n--- CLASSIFICATION REPORT (TARGET LABELS) ---")
    # Print report focusing on core AG News categories
    print(classification_report(y_true, y_pred, labels=TARGET_LABELS, zero_division=0))
    
    kappa = cohen_kappa_score(y_true, y_pred)
    # Calculate Macro F1 strictly across primary news categories
    macro_f1 = f1_score(y_true, y_pred, labels=TARGET_LABELS, average="macro", zero_division=0)

    print(f"Cohen's Kappa Score: {kappa:.4f}")
    print(f"Mean Latency:        {mean_lat:.4f}s")
    print(f"p95 Latency:         {p95_lat:.4f}s")
    print(f"Total API Run Cost:  ${estimated_cost:.6f}")
    print(f"Macro F1 Score:      {macro_f1:.4f}")

    if macro_f1 < f1_threshold:
        print(f"\n[FAILURE GATE TRIGGERED] Macro F1 ({macro_f1:.4f}) is below threshold ({f1_threshold})", file=sys.stderr)
        sys.exit(1)
    
    print("\n[SUCCESS] Evaluation passed gate threshold.")
    sys.exit(0)

if __name__ == "__main__":
    
    run_evaluation(sample_size=None)
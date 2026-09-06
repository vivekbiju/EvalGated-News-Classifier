import json
import sys
import time
import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score
from app import classify_text

F1_THRESHOLD = 0.70

# Pricing constants for openai/gpt-oss-120b
INPUT_COST_PER_1M = 0.15
OUTPUT_COST_PER_1M = 0.60

def run_eval():
    with open("golden_set.json", "r") as f:
        data = json.load(f)

    y_true, y_pred, latencies = [], [], []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for item in data:
        if item["expected"] in ["flag_for_human", "refuse"]:
            continue

        start = time.perf_counter()
        res, p_tokens, c_tokens = classify_text(item["text"])
        latencies.append(time.perf_counter() - start)

        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens

        y_true.append(item["expected"])
        y_pred.append(res.label if res else "None")

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, zero_division=0))

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    p95_latency = np.percentile(latencies, 95)
    mean_latency = np.mean(latencies)
    macro_f1 = report["macro avg"]["f1-score"]

    # Calculate costs
    input_cost = (total_prompt_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (total_completion_tokens / 1_000_000) * OUTPUT_COST_PER_1M
    total_cost = input_cost + output_cost

    print("--- Metrics & Performance ---")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Mean Latency: {mean_latency:.2f}s")
    print(f"p95 Latency: {p95_latency:.2f}s")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\n--- Token & Cost Tracking ---")
    print(f"Total Prompt Tokens:     {total_prompt_tokens}")
    print(f"Total Completion Tokens: {total_completion_tokens}")
    print(f"Total Run Cost:         ${total_cost:.6f}")

    if macro_f1 < F1_THRESHOLD:
        print(f"\n[EVAL FAILED] Macro F1 ({macro_f1:.2f}) below threshold ({F1_THRESHOLD})")
        sys.exit(1)

    print("\n[EVAL PASSED]")
    sys.exit(0)

if __name__ == "__main__":
    run_eval()
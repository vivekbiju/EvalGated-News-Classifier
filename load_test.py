import time
import requests
import numpy as np

# 1. Configuration
LIVE_URL = "https://<YOUR_HF_USERNAME>-<YOUR_SPACE_NAME>.hf.space/classify"  # Replace with your live HF Space endpoint
TOTAL_REQUESTS = 50

# Pricing configuration (adjust according to your API provider's specs)
COST_PER_REQUEST_ESTIMATE = 0.00005  # Average cost per request in USD

def run_benchmark():
    latencies = []
    errors = 0
    successful_calls = 0

    payload = {
        "text": "Wall St. Bears Run Wild Stories on inflation and potential interest rate hikes hit stock indexes."
    }

    print(f"Sending {TOTAL_REQUESTS} requests to {LIVE_URL}...\n")

    for i in range(1, TOTAL_REQUESTS + 1):
        start_time = time.perf_counter()
        try:
            response = requests.post(LIVE_URL, json=payload, timeout=10)
            elapsed = time.perf_counter() - start_time

            if response.status_code == 200:
                latencies.append(elapsed)
                successful_calls += 1
            else:
                errors += 1
                print(f"Request {i}: HTTP {response.status_code} - {response.text}")

        except requests.RequestException as err:
            errors += 1
            print(f"Request {i}: Failed with exception - {err}")

        # Optional: brief sleep to avoid hammering rate limits on free tiers
        time.sleep(0.1)

    # 2. Performance Metrics Calculation
    if latencies:
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        mean_latency = np.mean(latencies)
    else:
        p50_latency = p95_latency = mean_latency = 0.0

    error_rate = (errors / TOTAL_REQUESTS) * 100
    total_estimated_cost = TOTAL_REQUESTS * COST_PER_REQUEST_ESTIMATE

    # 3. Output Report
    print("=" * 45)
    print("STEP D6 BENCHMARK REPORT")
    print("=" * 45)
    print(f"Total Requests Sent: {TOTAL_REQUESTS}")
    print(f"Successful Calls:    {successful_calls}")
    print(f"Failed Calls:        {errors}")
    print(f"Error Rate:          {error_rate:.2f}%")
    print("-" * 45)
    print(f"p50 Latency:         {p50_latency:.4f} seconds")
    print(f"p95 Latency:         {p95_latency:.4f} seconds")
    print(f"Mean Latency:        {mean_latency:.4f} seconds")
    print("-" * 45)
    print(f"Estimated Total Cost: ${total_estimated_cost:.6f}")
    print("=" * 45)

if __name__ == "__main__":
    run_benchmark()
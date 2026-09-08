import time
import numpy as np
from gradio_client import Client

SPACE_ID = "Vivekbiju0/AG-news-classifier"
TOTAL_REQUESTS = 50
COST_PER_REQUEST_ESTIMATE = 0.000012

def run_benchmark():
    latencies = []
    errors = 0
    successful_calls = 0

    test_headline = "Wall St. Bears Run Wild: Stories on inflation and potential interest rate hikes hit stock indexes."

    print(f"Connecting to Hugging Face Space: {SPACE_ID}...")
    try:
        client = Client(SPACE_ID)
    except Exception as e:
        print(f"Failed to connect to Space: {e}")
        return

    print(f"Sending {TOTAL_REQUESTS} requests...\n")

    for i in range(1, TOTAL_REQUESTS + 1):
        start_time = time.perf_counter()
        try:
            # Correct endpoint name confirmed via view_api()
            result = client.predict(test_headline, api_name="/classify_text")
            elapsed = time.perf_counter() - start_time
            
            latencies.append(elapsed)
            successful_calls += 1
            print(f"Request {i}/{TOTAL_REQUESTS}: Success ({elapsed:.2f}s) | Output: '{result}'")
        except Exception as err:
            errors += 1
            print(f"Request {i}/{TOTAL_REQUESTS}: Failed - {err}")

        time.sleep(0.1)

    # Calculate metrics
    if latencies:
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        mean_latency = np.mean(latencies)
    else:
        p50_latency = p95_latency = mean_latency = 0.0

    error_rate = (errors / TOTAL_REQUESTS) * 100
    total_estimated_cost = TOTAL_REQUESTS * COST_PER_REQUEST_ESTIMATE

    # Output report
    print("\n" + "=" * 45)
    print("STEP D6 DEPLOYED SYSTEM BENCHMARK REPORT")
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
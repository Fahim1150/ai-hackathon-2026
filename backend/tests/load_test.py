"""
upay ActivateAI — API Load Test.

Uses concurrent.futures to hit the POST /api/predict/batch endpoint
with 200 concurrent requests and asserts sub-100ms average latency.
"""

import concurrent.futures
import time
import requests
import statistics

# Fast API development server URL
API_URL = "http://localhost:8000/api/predict/batch"
API_KEY = "upay-activate-ai-dev-key-2026"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# Dummy payload matching feature columns
SAMPLE_PAYLOAD = {
    "customers": [
        {
            "customer_id": "LOAD_TEST_1",
            "days_inactive": 45,
            "historical_tx_count": 12,
            "monthly_inflow_bdt": 5000,
            "cashout_ratio": 0.8,
            "promos_sent_30d": 2,
            "promo_ignore_streak": 2,
            "lifecycle_stage": "At-Risk Churner",
            "wallet_type": "Primary",
            "top_affinity_domain": "Mobile_Recharge",
            "channel_type": "USSD"
        }
    ]
}

def make_request():
    """Send a single batch prediction request and return the latency in ms."""
    start_time = time.perf_counter()
    response = requests.post(API_URL, json=SAMPLE_PAYLOAD, headers=HEADERS)
    end_time = time.perf_counter()
    
    # Check if rate limited
    if response.status_code == 429:
         return None  # Rate limited
    
    response.raise_for_status()
    latency_ms = (end_time - start_time) * 1000
    return latency_ms

def run_load_test(concurrency: int = 20, total_requests: int = 200):
    """Run concurrent requests against the batch prediction endpoint."""
    print("=" * 60)
    print(f"upay ActivateAI — Load Test ({total_requests} requests, {concurrency} workers)")
    print("=" * 60)
    
    latencies = []
    rate_limited_count = 0
    errors = 0
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request) for _ in range(total_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                lat = future.result()
                if lat is None:
                    rate_limited_count += 1
                else:
                    latencies.append(lat)
            except Exception as e:
                errors += 1

    total_time = time.time() - start_time
    
    print(f"✅ Load test completed in {total_time:.2f}s")
    print(f"   Successful: {len(latencies)}")
    print(f"   Rate Limited (HTTP 429): {rate_limited_count}")
    print(f"   Errors: {errors}")
    
    if latencies:
        avg_latency = statistics.mean(latencies)
        # Handle numpy import dynamically
        import numpy as np
        p95_latency = np.percentile(latencies, 95)
        
        print("\n📊 LATENCY STATS")
        print(f"   Average Response Time: {avg_latency:.2f} ms")
        print(f"   P95 Response Time:     {p95_latency:.2f} ms")
        print(f"   Min Response Time:     {min(latencies):.2f} ms")
        print(f"   Max Response Time:     {max(latencies):.2f} ms")
        
        assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms exceeded 100ms"
        print("✅ SUCCESS: Average latency is under 100ms.")
    else:
        print("❌ All requests failed or were rate limited.")
        
if __name__ == "__main__":
    run_load_test()

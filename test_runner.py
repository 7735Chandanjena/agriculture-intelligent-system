
import requests
import json
import time

# Configuration
API_URL = "http://localhost:5050/api/v1/recommend"

def run_tests():
    """
    Reads test cases from JSON and sends requests to the Flask API.
    """
    try:
        with open('test_cases.json', 'r') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("Error: test_cases.json not found.")
        return

    print("="*60)
    print("INTELLIGENT AGRICULTURE SYSTEM - API TEST SUITE")
    print("="*60)

    for case in test_cases:
        print(f"\n[TEST] Running Scenario: {case['name']}")
        print(f"Input Data: {case['payload']}")
        
        try:
            # Send POST request to the KNN Engine
            start_time = time.time()
            response = requests.post(API_URL, json=case['payload'])
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                rec = data['recommendation']
                
                print(f"Status: SUCCESS (Latency: {latency:.2f}ms)")
                print(f" -> Recommended Crop: {rec['optimal_crop']}")
                print(f" -> Expected Yield:  {rec['expected_yield']}")
                print(f" -> Irrigation:      {rec['irrigation_plan']}")
                print(f" -> Fertilization:   {rec['fertilization_strategy']}")
                print(f" -> Planting Status: {rec['planting_window']}")
                
                # Check KNN Basis (Explainability)
                neighbors = data.get('historical_basis', [])
                print(f" -> KNN Evidence:    Found {len(neighbors)} historical matches.")
                
            else:
                print(f"Status: FAILED (HTTP {response.status_code})")
                print(f"Error Message: {response.text}")

        except requests.exceptions.ConnectionError:
            print("Status: CRITICAL ERROR")
            print("Could not connect to the API. Is 'app_real.py' running on port 5050?")
            break
        
        print("-" * 40)

if __name__ == "__main__":
    run_tests()

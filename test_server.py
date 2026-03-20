"""
Quick test script to verify the FastAPI server is working
"""
import requests
# import time

def test_server():
    base_url = "http://localhost:5000"
    
    print("Testing FastAPI Multi-Disease Prediction System...")
    print("=" * 60)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✓ Health Check: PASSED")
            print(f"  Status: {data['status']}")
            print(f"  Models Loaded: {data['models_loaded']}")
        else:
            print(f"✗ Health Check: FAILED (Status {response.status_code})")
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Make sure it's running on port 5000")
        print("\nTo start the server, run:")
        print("  python app.py")
        print("\nOr:")
        print("  uvicorn app:app --host 127.0.0.1 --port 5000")
        return
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Test home page
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("✓ Home Page: PASSED")
        else:
            print(f"✗ Home Page: FAILED (Status {response.status_code})")
    except Exception as e:
        print(f"✗ Home Page Error: {e}")
    
    print("\n" + "=" * 60)
    print("Server is running successfully!")
    print(f"\nAccess the application at: {base_url}")
    print("\nAvailable endpoints:")
    print(f"  - Home: {base_url}/")
    print(f"  - Diabetes: {base_url}/disease/diabetes")
    print(f"  - Heart Disease: {base_url}/disease/heart_disease")
    print(f"  - Kidney Disease: {base_url}/disease/kidney_disease")
    print(f"  - Health Check: {base_url}/health")

if __name__ == "__main__":
    test_server()

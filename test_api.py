"""
Test script for the DFU Classification API
"""
import requests
import sys
from pathlib import Path


def test_root_endpoint(base_url):
    """Test the root endpoint."""
    print("\n1. Testing root endpoint (GET /)...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        assert "message" in response.json()
        print("   ✓ Root endpoint test passed!")
    except Exception as e:
        print(f"   ✗ Root endpoint test failed: {e}")


def test_health_endpoint(base_url):
    """Test the health check endpoint."""
    print("\n2. Testing health endpoint (GET /health)...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.json()}")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        print("   ✓ Health endpoint test passed!")
    except Exception as e:
        print(f"   ✗ Health endpoint test failed: {e}")


def test_predict_endpoint(base_url, image_path=None):
    """Test the predict endpoint."""
    print("\n3. Testing predict endpoint (POST /predict)...")
    
    if image_path and Path(image_path).exists():
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{base_url}/predict", files=files)
            
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.json()}")
            
            data = response.json()
            if response.status_code == 200:
                assert "prediction" in data
                print(f"   ✓ Predict endpoint test passed! Prediction: {data['prediction']}")
            else:
                print(f"   ⚠ Prediction returned error: {data.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   ✗ Predict endpoint test failed: {e}")
    else:
        print("   ⚠ No image file provided or file not found. Skipping prediction test.")
        print("   To test prediction, run: python test_api.py http://localhost:8000 path/to/image.jpg")


def main():
    """Main test function."""
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=" * 60)
    print("DFU Classification API Test Suite")
    print("=" * 60)
    print(f"Testing API at: {base_url}")
    
    # Check if server is reachable
    try:
        requests.get(base_url, timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Error: Cannot connect to {base_url}")
        print("  Make sure the server is running:")
        print("  uvicorn app:app --reload --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # Run tests
    test_root_endpoint(base_url)
    test_health_endpoint(base_url)
    test_predict_endpoint(base_url, image_path)
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

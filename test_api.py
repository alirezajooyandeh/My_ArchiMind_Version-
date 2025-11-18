#!/usr/bin/env python3
"""
Quick test script to verify the /process endpoint works correctly.
Tests with a sample image file if available.
"""
import requests
import json
import sys
from pathlib import Path

API_BASE = "http://127.0.0.1:8000"

def test_process_endpoint(image_path=None):
    """Test the /process endpoint with a sample image."""
    
    # Check health first
    print("1. Checking health endpoint...")
    try:
        health_response = requests.get(f"{API_BASE}/healthz", timeout=5)
        health_response.raise_for_status()
        health = health_response.json()
        print(f"   ✓ Health check passed: {health}")
        if not health.get("ready"):
            print("   ⚠ Warning: Server reports not ready")
    except Exception as e:
        print(f"   ✗ Health check failed: {e}")
        return False
    
    # If no image provided, skip file test
    if not image_path:
        print("\n2. Skipping file upload test (no image path provided)")
        print("   To test with a file, run: python test_api.py <path_to_image.jpg>")
        return True
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"\n2. ✗ Image file not found: {image_path}")
        return False
    
    print(f"\n2. Testing /process endpoint with {image_path.name}...")
    
    try:
        # Prepare form data
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            data = {
                'imgsz': 1536,
                'wall_conf': 0.60,
                'door_conf': 0.50,
                'wind_conf': 0.80,
                'room_conf': 0.40,
                'show_labels': True,
                'fill_rooms': True,
                'outline_interior': False,
                'pdf_mode': False,
            }
            
            response = requests.post(
                f"{API_BASE}/process",
                files=files,
                data=data,
                timeout=60
            )
        
        # Check response
        print(f"   HTTP Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   ✗ Request failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error detail: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"   Error text: {response.text[:200]}")
            return False
        
        result = response.json()
        
        # Validate response structure
        print("   Validating response structure...")
        required_keys = ['request_id', 'meta', 'detections', 'rooms', 'downloads', 'settings_used']
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            print(f"   ✗ Missing required keys: {missing_keys}")
            return False
        
        # Check meta.timings
        if 'timings' not in result.get('meta', {}):
            print("   ✗ Missing meta.timings")
            return False
        
        # Check counts
        if 'counts' not in result:
            print("   ⚠ Warning: Missing counts (optional but recommended)")
        else:
            counts = result['counts']
            print(f"   ✓ Counts: walls={counts.get('wall', 0)}, doors={counts.get('door', 0)}, "
                  f"windows={counts.get('window', 0)}, rooms={counts.get('room', 0)}")
        
        # Check downloads
        downloads = result.get('downloads', {})
        if 'overlay.png' not in downloads:
            print("   ✗ Missing overlay.png in downloads")
            return False
        
        print(f"   ✓ Response structure valid")
        print(f"   ✓ Request ID: {result['request_id']}")
        print(f"   ✓ Timings: {json.dumps(result['meta']['timings'], indent=2)}")
        print(f"   ✓ Rooms detected: {len(result.get('rooms', []))}")
        print(f"   ✓ Download URL: {downloads['overlay.png']}")
        
        return True
        
    except requests.exceptions.Timeout:
        print("   ✗ Request timed out (>60s)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"   ✗ Connection error: Is the server running at {API_BASE}?")
        return False
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_process_endpoint(image_path)
    sys.exit(0 if success else 1)


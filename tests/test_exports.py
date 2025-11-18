"""Tests for export utilities."""
import tempfile
import os
import json
import csv
from backend.exports import export_to_csv, export_to_json
from backend.geometry import polygon_to_wkt


def test_export_to_csv():
    """Test CSV export."""
    detections = {
        "wall": [
            {
                "bbox": [10, 20, 50, 60],
                "confidence": 0.85,
                "polygon": None,
            }
        ],
        "door": [],
    }
    
    rooms = [
        {
            "id": 0,
            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "area_px": 10000,
            "centroid": [50, 50],
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    try:
        export_to_csv(detections, rooms, temp_path, pixels_per_foot=10.0)
        
        # Read and verify
        with open(temp_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2  # 1 detection + 1 room
        
        # Check detection row
        detection_row = rows[0]
        assert detection_row["class"] == "wall"
        assert float(detection_row["confidence"]) == 0.85
        assert float(detection_row["x_min"]) == 10
        
        # Check room row
        room_row = rows[1]
        assert room_row["class"] == "room"
        assert float(room_row["area_px"]) == 10000
        assert float(room_row["area_sf"]) == 100.0  # 10000 / (10^2)
        
    finally:
        os.unlink(temp_path)


def test_export_to_json():
    """Test JSON export."""
    detections = {"wall": []}
    rooms = []
    meta = {"request_id": "test-123", "width": 1000, "height": 800}
    settings_used = {"imgsz": 1280}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        export_to_json(detections, rooms, meta, settings_used, temp_path)
        
        # Read and verify
        with open(temp_path, 'r') as f:
            data = json.load(f)
        
        assert data["request_id"] == "test-123"
        assert data["meta"]["width"] == 1000
        assert "downloads" in data
        assert "overlay.png" in data["downloads"]
        
    finally:
        os.unlink(temp_path)


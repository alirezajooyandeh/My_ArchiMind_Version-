"""Export utilities for CSV and JSON."""
import logging
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from backend.geometry import polygon_to_wkt, calculate_area_sf

logger = logging.getLogger(__name__)


def export_to_csv(
    detections: Dict[str, List[Dict]],
    rooms: List[Dict],
    output_path: str,
    pixels_per_foot: Optional[float] = None,  # Not used directly, rooms already have area_sf
):
    """Export detections and rooms to CSV with specified fields."""
    rows = []
    
    # Add non-room detections (walls, doors, windows)
    detection_id = 0
    for class_name, class_dets in detections.items():
        if class_name == "room":
            continue  # Rooms handled separately
        
        for det in class_dets:
            bbox = det["bbox"]
            
            row = {
                "id": detection_id,
                "class": class_name,
                "confidence": round(det["confidence"], 4),
                "x_min": round(bbox[0], 2),
                "y_min": round(bbox[1], 2),
                "x_max": round(bbox[2], 2),
                "y_max": round(bbox[3], 2),
                "polygon_wkt": "",
                "area_px": "",
                "area_sf": "",
                "centroid_x": "",
                "centroid_y": "",
            }
            rows.append(row)
            detection_id += 1
    
    # Add rooms with full polygon and area information
    for room in rooms:
        polygon = room.get("polygon", [])
        area_px = room.get("area_px", 0)
        area_sf = room.get("area_sf")
        centroid = room.get("centroid", [0, 0])
        
        row = {
            "id": room.get("id", detection_id),
            "class": "room",
            "confidence": "",
            "x_min": "",
            "y_min": "",
            "x_max": "",
            "y_max": "",
            "polygon_wkt": polygon_to_wkt(polygon),
            "area_px": round(area_px, 2) if area_px else "",
            "area_sf": round(area_sf, 2) if area_sf is not None else "",
            "centroid_x": round(centroid[0], 2) if centroid else "",
            "centroid_y": round(centroid[1], 2) if centroid else "",
        }
        rows.append(row)
        detection_id += 1
    
    # Write CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(rows)} rows to {output_path}")
    else:
        # Create empty CSV with headers
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "class", "confidence", "x_min", "y_min", "x_max", "y_max",
                "polygon_wkt", "area_px", "area_sf", "centroid_x", "centroid_y"
            ])
            writer.writeheader()
        logger.info(f"Created empty CSV at {output_path}")


def export_to_json(
    detections: Dict[str, List[Dict]],
    rooms: List[Dict],
    meta: Dict,
    settings_used: Dict,
    output_path: str,
    download_base_url: str = "",
    diagnostics: Optional[Dict] = None,
    warnings: Optional[List[str]] = None,
):
    """Export full results to JSON."""
    # Prepare download links
    request_id = meta.get("request_id", "unknown")
    downloads = {
        "overlay.png": f"{download_base_url}/download/{request_id}/overlay.png",
        "data.csv": f"{download_base_url}/download/{request_id}/data.csv",
        "data.json": f"{download_base_url}/download/{request_id}/data.json",
    }
    
    result = {
        "request_id": request_id,
        "meta": meta,
        "detections": detections,
        "rooms": rooms,
        "downloads": downloads,
        "settings_used": settings_used,
    }
    
    # Add diagnostics and warnings if provided
    if diagnostics:
        result["diagnostics"] = diagnostics
    if warnings:
        result["warnings"] = warnings
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Exported JSON to {output_path}")


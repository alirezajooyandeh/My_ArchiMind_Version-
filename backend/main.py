"""Main FastAPI application."""
import logging
import time
import uuid
import os
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.config import settings
from backend.models import ModelManager
from backend.image_utils import process_uploaded_file
from backend.geometry import build_room_polygons, calculate_area_sf
from backend.geometry_v2 import build_room_polygons_v2
from backend.room_fill import infer_rooms_from_detections, RoomFillSettings
from backend.overlay import render_overlay
from backend.auto_tune import auto_tune_parameters
from backend.exports import export_to_csv, export_to_json
from backend.scale import resolve_scale
from backend.file_manager import FileManager
from shapely.geometry import Polygon

# Global config for room scale (feet per pixel)
# If set via environment variable, this will override any other scale calculations
ROOM_FEET_PER_PIXEL = float(os.getenv("ROOM_FEET_PER_PIXEL", "0"))

# --- helper: ensure pixel xyxy ---
def _to_pixel_xyxy(boxes, img_w, img_h):
    """
    Accepts list of boxes in one of:
      - xyxy pixels: [x1,y1,x2,y2]
      - xywh pixels: [x,y,w,h]
      - normalized xyxy: all |coords| <= 1
      - normalized xywh: all |coords| <= 1, 3rd/4th are widths/heights
    Returns list of [x1,y1,x2,y2] in pixel space (clipped).
    """
    px = []
    for b in boxes or []:
        if len(b) < 4:
            continue
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        
        # Detect normalized coords
        normalized = max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0
        
        # Detect xywh vs xyxy (heuristic)
        # If normalized and third/fourth terms look like width/height (<=1),
        # treat as xywh; else treat as xyxy.
        if normalized:
            if x2 <= 1.0 and y2 <= 1.0:
                # xywh normalized
                x, y, w, h = x1*img_w, y1*img_h, x2*img_w, y2*img_h
                X1, Y1, X2, Y2 = x, y, x+w, y+h
            else:
                # xyxy normalized
                X1, Y1, X2, Y2 = x1*img_w, y1*img_h, x2*img_w, y2*img_h
        else:
            # pixels: try to infer xywh if third/fourth look like size
            if x2 <= 2 or y2 <= 2:
                x, y, w, h = x1, y1, x2, y2
                X1, Y1, X2, Y2 = x, y, x+w, y+h
            else:
                X1, Y1, X2, Y2 = x1, y1, x2, y2
        
        # Clip & order
        X1 = max(0, min(int(round(X1)), img_w-1))
        Y1 = max(0, min(int(round(Y1)), img_h-1))
        X2 = max(0, min(int(round(X2)), img_w-1))
        Y2 = max(0, min(int(round(Y2)), img_h-1))
        if X2 < X1:
            X1, X2 = X2, X1
        if Y2 < Y1:
            Y1, Y2 = Y2, Y1
        px.append([X1, Y1, X2, Y2])
    return px

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="Floor Plan Analysis API", version="1.0.0")

# CORS middleware (allow local frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_manager = ModelManager(settings)
file_manager = FileManager(settings.temp_dir, settings.temp_ttl_seconds)

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Floor Plan Analysis API")
    logger.info(f"Device: {model_manager.device}")
    logger.info(f"Available classes: {model_manager.get_available_classes()}")
    
    # Cleanup old files
    file_manager.cleanup_old_files()


# Define routes BEFORE mounting static files to avoid conflicts
@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Serve landing page with ArchiMind button."""
    landing_path = Path(__file__).parent.parent / "frontend" / "landing.html"
    if landing_path.exists():
        return FileResponse(
            str(landing_path),
            media_type="text/html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    return HTMLResponse("<h1>Landing</h1><p>landing.html not found.</p>")

@app.get("/mvp", response_class=HTMLResponse)
async def archimind_mvp():
    """Serve main MVP application."""
    index_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if index_path.exists():
        return FileResponse(
            str(index_path),
            media_type="text/html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    return HTMLResponse("<h1>MVP</h1><p>index.html not found.</p>")


# Define static directory path
static_dir = Path(__file__).parent.parent / "frontend"

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir), html=False), name="static")

# Serve CSS with no-cache headers
@app.get("/static/{filepath:path}")
async def serve_static(filepath: str):
    """Serve static files with cache control."""
    file_path = static_dir / filepath
    if file_path.exists() and file_path.is_file():
        # Determine media type
        ext = file_path.suffix.lower()
        media_types = {
            ".css": "text/css",
            ".js": "application/javascript",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
        }
        media_type = media_types.get(ext, "application/octet-stream")
        
        headers = {}
        if ext in [".html", ".css", ".js"]:
            headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        
        return FileResponse(str(file_path), media_type=media_type, headers=headers)
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/assets/images/{filename}")
async def get_image(filename: str):
    """Serve images from assets directory."""
    image_path = static_dir / "assets" / "images" / filename
    if image_path.exists() and image_path.is_file():
        ext = image_path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".svg": "image/svg+xml"
        }
        media_type = media_types.get(ext, "application/octet-stream")
        return FileResponse(str(image_path), media_type=media_type)
    
    raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    
    # Serve assets directory (for landing page image)
    @app.get("/assets/{filepath:path}")
    async def get_asset(filepath: str):
        asset_path = static_dir / "assets" / filepath
        if asset_path.exists() and asset_path.is_file():
            # Determine media type based on extension
            ext = asset_path.suffix.lower()
            media_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".svg": "image/svg+xml"
            }
            media_type = media_types.get(ext, "application/octet-stream")
            return FileResponse(str(asset_path), media_type=media_type)
        raise HTTPException(status_code=404, detail=f"Asset not found: {filepath}")


@app.get("/healthz")
async def health():
    """Health check endpoint."""
    available_classes = model_manager.get_available_classes()
    all_classes = ["wall", "door", "window", "room"]
    
    model_status = {
        cls: cls in available_classes for cls in all_classes
    }
    
    return {
        "status": "healthy",
        "device": model_manager.device,
        "models_loaded": model_status,
        "ready": any(model_status.values()),
    }


@app.post("/process")
async def process_floor_plan(
    request: Request,
    file: UploadFile = File(...),
    imgsz: int = Form(1536),
    wall_conf: float = Form(0.60),
    door_conf: float = Form(0.50),
    wind_conf: float = Form(0.80),
    room_conf: float = Form(0.25),  # Lowered from 0.40 to be more forgiving
    show_labels: Optional[str] = Form(None),
    fill_rooms: Optional[str] = Form(None),  # Legacy name, will use show_rooms_fill
    show_rooms_fill: Optional[str] = Form(None),  # New explicit name
    outline_interior: Optional[str] = Form(None),  # Legacy name, will use show_room_outlines
    show_room_outlines: Optional[str] = Form(None),  # New explicit name for room outlines toggle
    show_walls: Optional[str] = Form(None),
    show_doors: Optional[str] = Form(None),
    show_windows: Optional[str] = Form(None),
    outline_thickness: int = Form(2),
    line_thickness: Optional[int] = Form(None),  # New line thickness control (1-8)
    line_style: Optional[str] = Form(None),  # New line style: "solid", "dashed", "dotted"
    overlay_style: str = Form("detailed"),
    wall_box_mode: str = Form("filled"),
    wall_box_alpha: float = Form(0.25),
    wall_box_thickness: int = Form(2),
    pdf_mode: bool = Form(True),
    pdf_dpi: int = Form(300),
    auto_tune: bool = Form(False),
    prefer_interior_footprint: bool = Form(True),
    wall_thick_px: int = Form(12),
    door_bridge_extra_px: int = Form(12),
    min_room_area_px: int = Form(1000),  # Lowered from 6000 to be more forgiving
    footprint_grow_px: int = Form(3),
    doors_as_walls: bool = Form(True),
    gap_close_px: int = Form(8),
    scale_mode: str = Form("unknown"),
    explicit_scale: Optional[float] = Form(None),
    scale_feet_per_inch: Optional[str] = Form(None),  # Architectural scale: feet per 1 sheet inch
    measure_point1_x: Optional[float] = Form(None),
    measure_point1_y: Optional[float] = Form(None),
    measure_point2_x: Optional[float] = Form(None),
    measure_point2_y: Optional[float] = Form(None),
    measure_distance_ft: Optional[float] = Form(None),
    # Advanced settings (optional - accepted but may not be fully implemented yet)
    binarization_threshold: Optional[int] = Form(None),
    adaptive_threshold: Optional[bool] = Form(None),
    edge_sharpening: Optional[bool] = Form(None),
    noise_reduction: Optional[bool] = Form(None),
    noise_reduction_strength: Optional[float] = Form(None),
    contrast_boost: Optional[float] = Form(None),
    vector_extraction_mode: Optional[bool] = Form(None),
    high_quality_rasterization: Optional[bool] = Form(None),
    deskew_rotation_correction: Optional[bool] = Form(None),
    wall_nms_iou: float = Form(0.04),
    doorwin_nms_iou: float = Form(0.05),
    segmentation_mask_refinement: Optional[bool] = Form(None),
    room_polygon_smoothing: Optional[float] = Form(None),
    room_min_confidence: Optional[float] = Form(None),
    min_wall_length: Optional[int] = Form(None),
    min_door_gap_width: Optional[int] = Form(None),
    merge_walls_threshold: Optional[int] = Form(None),
    snap_walls_to_angles: Optional[bool] = Form(None),
    correct_broken_walls: Optional[bool] = Form(None),
    room_hole_filling: Optional[bool] = Form(None),
    min_internal_gap: Optional[int] = Form(None),
    detect_scale_automatically: Optional[bool] = Form(None),
    door_removal_for_rooms: Optional[bool] = Form(None),
    auto_select_best_image_size: Optional[bool] = Form(None),
    auto_adjust_model_thresholds: Optional[bool] = Form(None),
    drawing_type: Optional[str] = Form(None),
    show_confidence_heatmap: Optional[bool] = Form(None),
    show_raw_bounding_boxes: Optional[bool] = Form(None),
    log_preprocessing_steps: Optional[bool] = Form(None),
):
    """
    Process uploaded floor plan.
    
    Expected Request:
      - POST /process
      - Content-Type: multipart/form-data
      - Form fields: file (UploadFile), imgsz, wall_conf, door_conf, etc.
    
    Success Response (HTTP 200):
      {
        "request_id": "uuid-string",
        "meta": {
          "request_id": "uuid-string",
          "filename": "file.jpg",
          "width": 1920,
          "height": 1080,
          "imgsz": 1536,
          "device": "cpu",
          "timings": {
            "load": 0.016,
            "preprocess": 0.0,
            "infer": 2.806,
            "postprocess": 0.046,
            "render": 0.008,
            "total": 2.884
          },
          "pixels_per_foot": null or float
        },
        "detections": {
          "wall": [...],
          "door": [...],
          "window": [...],
          "room": [...]
        },
        "rooms": [...],
        "downloads": {
          "overlay.png": "/download/{request_id}/overlay.png",
          "data.csv": "/download/{request_id}/data.csv",
          "data.json": "/download/{request_id}/data.json"
        },
        "settings_used": {...},
        "warnings": [],
        "diagnostics": {}
      }
    
    Error Response (HTTP 4xx/5xx):
      {
        "detail": "Error message here"
      }
    """
    request_id = str(uuid.uuid4())
    timings = {}
    start_time = time.time()
    
    # Get form data for flag parsing
    form = await request.form()
    
    # Helper function to read checkbox-style boolean from form data
    def _flag(name: str, default: bool = False) -> bool:
        """Read checkbox-style boolean from form-data."""
        raw = form.get(name)
        if raw is None:
            return default
        return str(raw).lower() in {"true", "1", "yes", "on"}
    
    # Parse all layer visibility flags in ONE place
    show_walls         = _flag("show_walls",         True)
    show_doors         = _flag("show_doors",         True)
    show_windows       = _flag("show_windows",       True)
    show_rooms_fill    = _flag("show_rooms_fill",    True)
    show_room_outlines = _flag("show_room_outlines", False)  # DEFAULT FALSE - only draw when explicitly enabled
    show_labels        = _flag("show_labels",        True)
    
    # Parse line style controls
    line_thickness_val = int(line_thickness) if line_thickness else outline_thickness
    line_thickness_val = max(1, min(8, line_thickness_val))  # Clamp to 1-8
    line_style_val = line_style if line_style in ["solid", "dashed", "dotted"] else "solid"
    
    logger.info(
        "ArchiMind flags: walls=%s doors=%s windows=%s fill=%s outlines=%s labels=%s",
        show_walls, show_doors, show_windows,
        show_rooms_fill, show_room_outlines, show_labels,
    )
    
    try:
        # Create request directory early so we can use it for mask dumps
        request_dir = file_manager.create_request_dir(request_id)
    except Exception as e:
        logger.error(f"Failed to create request directory: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize processing: {str(e)}"
        )
    
    try:
        # Validate file
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Read file content
        file_content = await file.read()
        filename = file.filename or "upload"
        
        # Load and preprocess image
        t0 = time.time()
        try:
            image_bgr, metadata = process_uploaded_file(
                file_content,
                filename,
                pdf_dpi=pdf_dpi if pdf_mode else 300,
            )
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process file '{filename}': {str(e)}"
            )
        timings["load"] = time.time() - t0
        
        if image_bgr is None or image_bgr.size == 0:
            raise HTTPException(
                status_code=400,
                detail="Failed to load image: image is empty or invalid"
            )
        
        h, w = image_bgr.shape[:2]
        
        # Auto-tune if enabled
        if auto_tune:
            t_auto = time.time()
            tuned = auto_tune_parameters(
                model_manager,
                image_bgr,
                candidate_imgsz=[896, 1152, 1280, 1536],
                base_conf_thresholds={
                    "wall": wall_conf,
                    "door": door_conf,
                    "window": wind_conf,
                    "room": room_conf,
                }
            )
            imgsz = tuned.get("imgsz", imgsz)
            wall_conf = tuned.get("conf_thresholds", {}).get("wall", wall_conf)
            door_conf = tuned.get("conf_thresholds", {}).get("door", door_conf)
            wind_conf = tuned.get("conf_thresholds", {}).get("window", wind_conf)
            room_conf = tuned.get("conf_thresholds", {}).get("room", room_conf)
            timings["auto_tune"] = time.time() - t_auto
        else:
            timings["auto_tune"] = 0.0
        
        # Preprocess timing
        t1 = time.time()
        timings["preprocess"] = time.time() - t1
        
        # Run inference (including room model)
        t2 = time.time()
        try:
            conf_thresholds = {
                "wall": wall_conf,
                "door": door_conf,
                "window": wind_conf,
                "room": room_conf,  # Room model confidence threshold
            }
            
            # NMS IoU thresholds
            nms_iou_thresholds = {
                "wall": wall_nms_iou,
                "door": doorwin_nms_iou,
                "window": doorwin_nms_iou,
                "room": 0.45,  # Default for room model
            }
            
            # Run all models including room model
            detections = model_manager.predict(
                image_bgr,
                imgsz=imgsz,
                conf_thresholds=conf_thresholds,
                nms_iou_thresholds=nms_iou_thresholds,
            )
            
            # Log room detections found
            room_det_count = len(detections.get("room", []))
            if room_det_count > 0:
                logger.info(f"[INFERENCE] Room model found {room_det_count} detections")
            else:
                logger.info(f"[INFERENCE] Room model found 0 detections (conf threshold: {room_conf})")
        except Exception as e:
            logger.error(f"Error during model inference: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Model inference failed: {str(e)}"
            )
        timings["infer"] = time.time() - t2
        
        if not detections:
            raise HTTPException(
                status_code=500,
                detail="Model inference returned no detections"
            )
        
        # Post-process: build rooms
        t3 = time.time()
        
        # Parse PDF DPI (ensure we have an int)
        try:
            pdf_dpi_value = int(pdf_dpi) if pdf_dpi else 300
        except (ValueError, TypeError):
            pdf_dpi_value = 300
        
        # Parse scale mode and architectural scale
        scale_mode_parsed = (scale_mode or "unknown").lower()
        scale_feet_per_inch_raw = scale_feet_per_inch or ""
        try:
            scale_feet_per_inch_value = float(scale_feet_per_inch_raw) if scale_feet_per_inch_raw else None
        except (ValueError, TypeError):
            scale_feet_per_inch_value = None
        
        # Compute feet_per_pixel from architectural scale
        feet_per_pixel = None
        if scale_mode_parsed == "explicit" and scale_feet_per_inch_value and pdf_dpi_value > 0:
            # 1 inch on sheet = scale_feet_per_inch_value feet in real world
            # pdf_dpi_value pixels = 1 inch
            # => feet per pixel = scale_feet_per_inch_value / pdf_dpi_value
            feet_per_pixel = scale_feet_per_inch_value / float(pdf_dpi_value)
            logger.info(f"Architectural scale: {scale_feet_per_inch_value} ft/inch, PDF DPI: {pdf_dpi_value}, feet_per_pixel: {feet_per_pixel:.6f}")
        
        # Resolve scale (fallback to existing logic if architectural scale not provided)
        # If ROOM_FEET_PER_PIXEL is set via environment variable, use it (convert to pixels_per_foot)
        if ROOM_FEET_PER_PIXEL > 0:
            # Convert feet_per_pixel to pixels_per_foot
            pixels_per_foot = 1.0 / ROOM_FEET_PER_PIXEL
            logger.info(f"Using ROOM_FEET_PER_PIXEL from environment: {ROOM_FEET_PER_PIXEL} (pixels_per_foot: {pixels_per_foot:.2f})")
        elif feet_per_pixel is not None:
            # Use architectural scale - convert feet_per_pixel to pixels_per_foot
            pixels_per_foot = 1.0 / feet_per_pixel
            logger.info(f"Using architectural scale: pixels_per_foot: {pixels_per_foot:.2f}")
        else:
            pixels_per_foot = resolve_scale(
                scale_mode=scale_mode_parsed,
                explicit_scale=explicit_scale,
                measure_points=(
                    (measure_point1_x, measure_point1_y),
                    (measure_point2_x, measure_point2_y)
                ) if measure_point1_x is not None and measure_point2_x is not None else None,
                measure_distance_ft=measure_distance_ft,
            )
        
        # Extract wall/door/window boxes (still needed for wall mask overlay)
        walls_xyxy = _to_pixel_xyxy(
            [d["bbox"] for d in detections.get("wall", [])],
            w, h
        )
        doors_xyxy = _to_pixel_xyxy(
            [d["bbox"] for d in detections.get("door", [])],
            w, h
        )
        windows_xyxy = _to_pixel_xyxy(
            [d["bbox"] for d in detections.get("window", [])],
            w, h
        )
        
        # Build wall mask for overlay rendering (still needed for continuous wall display)
        closed_wall_mask = None
        warnings = []
        diagnostics = {}
        
        # Build wall mask for overlay rendering, even if rooms are disabled
        room_settings = RoomFillSettings(
            wall_thick_px=wall_thick_px,
            door_bridge_extra_px=door_bridge_extra_px,
            footprint_grow_px=footprint_grow_px,
            min_room_area_px=min_room_area_px,
            prefer_interior_footprint=prefer_interior_footprint,
            outline_thickness=outline_thickness,
            doors_as_walls=doors_as_walls,
            gap_close_px=gap_close_px,
            scale_mode=scale_mode,
            explicit_scale=pixels_per_foot if pixels_per_foot else 0.0,
            dump_masks_to=None,  # Don't dump masks for wall-connectivity approach
        )
        
        # --- build wall masks ---
        from backend.room_fill import build_wall_mask, close_wall_gaps
        from backend.room_fill import _draw_doors
        
        h, w = image_bgr.shape[:2]
        
        # 1) THICK mask for room logic (unchanged - used for room finding)
        wall_mask_closed = np.zeros((h, w), dtype=np.uint8)
        for det in walls_xyxy:
            if len(det) < 4:
                continue
            x1, y1, x2, y2 = map(int, det)
            # Inflate for robustness in room finding
            pad = int(wall_thick_px)
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w - 1, x2 + pad)
            y2p = min(h - 1, y2 + pad)
            cv2.rectangle(wall_mask_closed, (x1p, y1p), (x2p, y2p), 255, thickness=-1)
        
        # Close gaps in the thick mask
        wall_mask_closed = close_wall_gaps(wall_mask_closed, gap_close_px)
        
        # Only modify the *closed* mask for room logic (doors CAN be sealed)
        if doors_as_walls and len(doors_xyxy):
            _draw_doors(wall_mask_closed, doors_xyxy, door_bridge_extra_px, doors_as_walls)
        
        # Additional small dilation for room finding
        if gap_close_px > 0:
            k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            wall_mask_closed = cv2.dilate(wall_mask_closed, k_small, iterations=1)
        
        # Keep closed_wall_mask for backward compatibility (used for room finding if needed)
        closed_wall_mask = wall_mask_closed
        
        # 2) EXACT visual rectangles for display (no padding, exact detection box dimensions)
        wall_rects_visual = []
        for det in walls_xyxy:
            if len(det) < 4:
                continue
            x1, y1, x2, y2 = map(int, det)
            # Use exact detection box dimensions - no padding, no shrink
            if x2 > x1 and y2 > y1:
                wall_rects_visual.append((x1, y1, x2, y2))
        
        # Keep wall_mask_visual for backward compatibility (but we won't use it for drawing)
        wall_mask_visual = None  # No longer used for visual overlay
        
        # Initialize room diagnostics
        room_logger = logging.getLogger("archimind.rooms")
        if "rooms" not in diagnostics:
            diagnostics["rooms"] = {}
        room_diag = diagnostics["rooms"]
        room_diag["stage"] = "start"
        room_diag["room_confidence_threshold"] = room_conf
        room_diag["yolo_detections_count"] = 0
        room_diag["processed_count"] = 0
        room_diag["skipped_no_bbox"] = 0
        room_diag["skipped_invalid_box"] = 0
        room_diag["skipped_tiny_area"] = 0
        room_diag["errors"] = []
        room_diag["raw_areas"] = []
        room_diag["image_dimensions"] = {"width": w, "height": h}
        
        # NEW: Build rooms directly from YOLO room model detections (simple rectangular polygons)
        rooms_from_detector = []
        room_detections = detections.get("room", [])
        room_diag["yolo_detections_count"] = len(room_detections)
        
        # Log if no detections from YOLO (might be due to confidence threshold)
        if not room_detections:
            logger.warning(f"[ROOM DETECTOR] No room detections from YOLO model (confidence threshold: {room_conf})")
            room_diag["warning"] = f"No YOLO detections (conf threshold: {room_conf})"
            
            # FALLBACK: Try with lower confidence threshold if no detections
            if room_conf > 0.15:
                logger.info(f"[ROOM DETECTOR] Attempting fallback with lower confidence threshold (0.15)")
                try:
                    fallback_conf_thresholds = {
                        "wall": wall_conf,
                        "door": door_conf,
                        "window": wind_conf,
                        "room": 0.15,  # Much lower threshold
                    }
                    fallback_detections = model_manager.predict(
                        image_bgr,
                        imgsz=imgsz,
                        conf_thresholds=fallback_conf_thresholds,
                    )
                    room_detections = fallback_detections.get("room", [])
                    if room_detections:
                        logger.info(f"[ROOM DETECTOR] Fallback found {len(room_detections)} room detections with lower threshold")
                        room_diag["fallback_used"] = True
                        room_diag["fallback_detections_count"] = len(room_detections)
                        room_diag["original_threshold"] = room_conf
                        room_diag["fallback_threshold"] = 0.15
                        room_diag["yolo_detections_count"] = len(room_detections)  # Update count after fallback
                    else:
                        logger.warning(f"[ROOM DETECTOR] Fallback also found 0 detections")
                        room_diag["fallback_used"] = False
                except Exception as e:
                    logger.error(f"[ROOM DETECTOR] Fallback failed: {e}", exc_info=True)
                    room_diag["fallback_error"] = str(e)
        
        if room_detections:
            logger.info(f"[ROOM DETECTOR] Found {len(room_detections)} room detections from YOLO model")
            room_diag["stage"] = "processing_yolo_detections"
            
            for idx, det in enumerate(room_detections):
                try:
                    bbox = det.get("bbox", [])
                    if not bbox or len(bbox) < 4:
                        room_diag["skipped_no_bbox"] += 1
                        room_logger.debug(f"[ROOM DETECTOR] Detection {idx}: skipped (no bbox or bbox < 4)")
                        continue
                    
                    # Convert bbox to pixel coordinates
                    try:
                        x1, y1, x2, y2 = map(float, bbox[:4])
                    except (ValueError, TypeError) as e:
                        room_diag["errors"].append(f"Detection {idx}: bbox conversion error: {e}")
                        room_logger.warning(f"[ROOM DETECTOR] Detection {idx}: bbox conversion error: {e}")
                        continue
                    
                    # Normalize if needed (check if coordinates are normalized)
                    normalized = max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0
                    if normalized:
                        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
                    
                    # Clip to image bounds
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))
                    
                    # Ensure valid box
                    if x2 <= x1 or y2 <= y1:
                        room_diag["skipped_invalid_box"] += 1
                        room_logger.debug(f"[ROOM DETECTOR] Detection {idx}: skipped (invalid box: x1={x1}, y1={y1}, x2={x2}, y2={y2})")
                        continue
                    
                    w_box = x2 - x1
                    h_box = y2 - y1
                    area_px = float(w_box * h_box)
                    room_diag["raw_areas"].append(area_px)
                    
                    # Very forgiving initial filter: reduce to 200px
                    if area_px < 200:
                        room_diag["skipped_tiny_area"] += 1
                        room_logger.debug(f"[ROOM DETECTOR] Detection {idx}: skipped (tiny area: {area_px:.1f}px²)")
                        continue
                    
                    # Build rectangular polygon from bbox
                    polygon = [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2],
                    ]
                    
                    score = float(det.get("confidence", 1.0))
                    centroid_x = (x1 + x2) / 2.0
                    centroid_y = (y1 + y2) / 2.0
                    
                    # Calculate area_sf if scale is known
                    area_sf = None
                    area_units = "px²"
                    if feet_per_pixel is not None and area_px > 0:
                        # area_ft² = area_px² * (feet_per_pixel)^2
                        area_sf = area_px * (feet_per_pixel ** 2)
                        area_units = "ft²"
                    elif pixels_per_foot and area_px > 0:
                        # Fallback to pixels_per_foot method
                        area_sf = area_px / (pixels_per_foot * pixels_per_foot)
                        area_units = "ft²"
                    
                    room_obj = {
                        "id": idx,
                        "name": f"Room {len(rooms_from_detector) + 1}",
                        "score": score,
                        "area_px": area_px,
                        "area_px2": area_px,  # Alias for consistency
                        "polygon": polygon,
                        "bbox": [x1, y1, x2, y2],
                        "centroid": [centroid_x, centroid_y],
                        "centroid_x": centroid_x,
                        "centroid_y": centroid_y,
                        "area_sf": area_sf,
                        "area_sqft": area_sf,  # Alias for consistency
                        "area_units": area_units,
                    }
                    
                    rooms_from_detector.append(room_obj)
                    room_diag["processed_count"] += 1
                    room_logger.debug(f"[ROOM DETECTOR] Detection {idx}: added (area={area_px:.1f}px²)")
                    
                except Exception as e:
                    error_msg = f"Detection {idx}: unexpected error: {str(e)}"
                    room_diag["errors"].append(error_msg)
                    room_logger.error(f"[ROOM DETECTOR] {error_msg}", exc_info=True)
                    continue
        else:
            logger.info("[ROOM DETECTOR] No room detections found from YOLO model")
            room_diag["stage"] = "no_yolo_detections"
        
        room_diag["rooms_from_detector_count"] = len(rooms_from_detector)
        
        # ROOM PIPE: Filter rooms by min area
        raw_rooms = rooms_from_detector
        room_diag["stage"] = "filtering"
        room_diag["raw_rooms_count"] = len(raw_rooms)
        room_diag["filtered_rooms_count"] = 0
        room_diag["filtered_out_count"] = 0
        
        # Collect all areas for logging
        areas = []
        for r in raw_rooms:
            area_px = r.get("area_px", 0)
            areas.append(area_px)
        room_diag["candidate_areas"] = areas
        room_logger.debug("ROOM PIPE: raw_rooms=%d, areas_px=%s", len(raw_rooms), areas)
        
        # Dynamic minimum area: scale with image size - VERY FORGIVING
        image_area = w * h
        # Use a very small minimum area threshold (0.01% of image area, but at least 200px)
        # This is very forgiving to catch even small rooms
        dynamic_min_area = max(200, int(0.0001 * image_area))
        # Use the smaller of the user-provided threshold or the dynamic one
        effective_min_area = min(min_room_area_px, dynamic_min_area)
        room_diag["image_area"] = image_area
        room_diag["dynamic_min_area"] = dynamic_min_area
        room_diag["user_min_area_px"] = min_room_area_px
        room_diag["effective_min_area"] = effective_min_area
        room_logger.debug("ROOM PIPE: image_area=%d, dynamic_min_area=%d, effective_min_area=%d (user=%d)", 
                               image_area, dynamic_min_area, effective_min_area, min_room_area_px)
        
        filtered_rooms = []
        filtered_out_areas = []
        for r in raw_rooms:
            try:
                area_px = r.get("area_px", 0)
                room_logger.debug("ROOM PIPE: candidate_area_px=%s", area_px)
                if area_px >= effective_min_area:
                    filtered_rooms.append(r)
                else:
                    filtered_out_areas.append(area_px)
                    room_diag["filtered_out_count"] += 1
            except Exception as e:
                error_msg = f"Error filtering room: {str(e)}"
                room_diag["errors"].append(error_msg)
                room_logger.error(f"[ROOM PIPE] {error_msg}", exc_info=True)
                # Don't drop the room on error - include it anyway
                filtered_rooms.append(r)
        
        room_diag["filtered_rooms_count"] = len(filtered_rooms)
        room_diag["filtered_out_areas"] = filtered_out_areas
        room_logger.debug(
            "ROOM PIPE: filtered_rooms=%d (effective_min_area=%d, original_min_area_px=%d, filtered_out=%d)",
            len(filtered_rooms),
            effective_min_area,
            min_room_area_px,
            room_diag["filtered_out_count"]
        )
        
        # IMPORTANT: Never truncate to a single room - return ALL filtered rooms
        rooms = filtered_rooms
        room_diag["stage"] = "complete"
        room_diag["final_rooms_count"] = len(rooms)
        
        if not rooms:
            diagnostics["room_note"] = "No room-model detections; rooms empty."
            logger.warning(f"[ANALYZE] No rooms from detector. Diagnostics: {room_diag}")
            room_diag["warning"] = "No rooms found after processing"
        else:
            logger.info(f"[ANALYZE] Returning {len(rooms)} rooms from detector")
        
        # Log comprehensive diagnostics
        logger.info(f"[ROOM DIAGNOSTICS] Stage: {room_diag.get('stage')}, "
                   f"YOLO detections: {room_diag.get('yolo_detections_count')}, "
                   f"Processed: {room_diag.get('processed_count')}, "
                   f"From detector: {room_diag.get('rooms_from_detector_count')}, "
                   f"Filtered: {room_diag.get('filtered_rooms_count')}, "
                   f"Final: {room_diag.get('final_rooms_count')}, "
                   f"Skipped (no bbox): {room_diag.get('skipped_no_bbox')}, "
                   f"Skipped (invalid): {room_diag.get('skipped_invalid_box')}, "
                   f"Skipped (tiny): {room_diag.get('skipped_tiny_area')}, "
                   f"Filtered out: {room_diag.get('filtered_out_count')}, "
                   f"Errors: {len(room_diag.get('errors', []))}")
        
        if room_diag.get("errors"):
            logger.warning(f"[ROOM DIAGNOSTICS] Errors encountered: {room_diag['errors']}")
        
        timings["postprocess"] = time.time() - t3
        
        # Render overlay
        t4 = time.time()
        
        # Use flags parsed earlier (no duplicate parsing)
        overlay_settings = {
            "show_labels": show_labels,
            "fill_rooms": show_rooms_fill,  # Legacy name for backward compatibility
            "show_rooms_fill": show_rooms_fill,  # Explicit flag name
            "outline_interior": show_room_outlines,  # Legacy name for backward compatibility
            "show_room_outlines": show_room_outlines,  # Explicit flag name
            "show_walls": show_walls,
            "show_doors": show_doors,
            "show_windows": show_windows,
            "outline_thickness": outline_thickness,
            "line_thickness": line_thickness_val,
            "line_style": line_style_val,
            "overlay_style": overlay_style,
            "pixels_per_foot": pixels_per_foot,
            "wall_box_mode": wall_box_mode,
            "wall_box_alpha": wall_box_alpha,
            "wall_box_thickness": wall_box_thickness,
            "use_legacy_wall_overlay": False,
            "wall_mask_visual": wall_mask_visual,  # Deprecated - no longer used for visual overlay
            "wall_rects_visual": wall_rects_visual,  # Exact visual rectangles matching detection boxes
            "wall_mask_closed": wall_mask_closed,  # Thick mask (doors sealed) for room finding
            "closed_wall_mask": closed_wall_mask,  # Backward compatibility alias
        }
        
        try:
            overlay_image = render_overlay(
                original_image=image_bgr,
                detections=detections,
                rooms=rooms,
                settings=overlay_settings,
            )
        except Exception as e:
            logger.error(f"Error rendering overlay: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to render overlay: {str(e)}"
            )
        timings["render"] = time.time() - t4
        
        # Save overlay image
        try:
            overlay_path = request_dir / "overlay.png"
            if not cv2.imwrite(str(overlay_path), overlay_image):
                raise HTTPException(
                    status_code=500,
                    detail="Failed to save overlay image"
                )
        except Exception as e:
            logger.error(f"Error saving overlay image: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save overlay image: {str(e)}"
            )
        
        # Prepare settings_used
        settings_used = {
            "imgsz": imgsz,
            "wall_conf": wall_conf,
            "door_conf": door_conf,
            "wind_conf": wind_conf,
            "room_conf": room_conf,
            "show_labels": show_labels,
            "fill_rooms": fill_rooms,
            "outline_interior": outline_interior,
            "show_walls": show_walls,
            "show_doors": show_doors,
            "show_windows": show_windows,
            "outline_thickness": outline_thickness,
            "overlay_style": overlay_style,
            "wall_box_mode": wall_box_mode,
            "wall_box_alpha": wall_box_alpha,
            "wall_box_thickness": wall_box_thickness,
            "pdf_mode": pdf_mode,
            "pdf_dpi": pdf_dpi,
            "auto_tune": auto_tune,
            "prefer_interior_footprint": prefer_interior_footprint,
            "wall_thick_px": wall_thick_px,
            "door_bridge_extra_px": door_bridge_extra_px,
            "min_room_area_px": min_room_area_px,
            "footprint_grow_px": footprint_grow_px,
            "doors_as_walls": doors_as_walls,
            "gap_close_px": gap_close_px,
            "scale_mode": scale_mode,
            "explicit_scale": explicit_scale,
        }
        
        # Calculate total time
        timings["total"] = time.time() - start_time
        
        # Prepare meta
        meta = {
            "request_id": request_id,
            "filename": filename,
            "width": int(w),
            "height": int(h),
            "imgsz": imgsz,
            "device": model_manager.device,
            "timings": timings,
            "pixels_per_foot": pixels_per_foot,
        }
        
        # Export CSV and JSON
        csv_path = request_dir / "data.csv"
        export_to_csv(
            detections=detections,
            rooms=rooms,
            output_path=str(csv_path),
            pixels_per_foot=pixels_per_foot,
        )
        
        json_path = request_dir / "data.json"
        export_to_json(
            detections=detections,
            rooms=rooms,
            meta=meta,
            settings_used=settings_used,
            output_path=str(json_path),
            download_base_url="",
            diagnostics=diagnostics,
            warnings=warnings,
        )
        
        # ROOM DEBUG: After room computation, right before building response dict
        debug_logger = logging.getLogger("archimind")
        debug_logger.debug("ROOM DEBUG: rooms_type=%s", type(rooms))
        try:
            debug_logger.debug("ROOM DEBUG: rooms_len=%d", len(rooms))
        except:
            debug_logger.debug("ROOM DEBUG: rooms_len=UNKNOWN")
        
        for i, r in enumerate(rooms[:10]):
            try:
                debug_logger.debug("ROOM DEBUG: room_%d_keys=%s", i, list(r.keys()))
            except:
                debug_logger.debug("ROOM DEBUG: room_%d cannot be inspected", i)
        
        # Log rooms state before building response
        logger.info(f"[ANALYZE] Rooms before response: type={type(rooms)}, len={len(rooms) if isinstance(rooms, list) else 'n/a'}")
        if isinstance(rooms, list) and len(rooms) > 0:
            logger.info(f"[ANALYZE] First room keys: {list(rooms[0].keys()) if isinstance(rooms[0], dict) else 'not-a-dict'}")
            logger.info(f"[ANALYZE] Total rooms to return: {len(rooms)}")
        
        # Calculate counts for legend
        counts = {
            "wall": len(detections.get("wall", [])),
            "door": len(detections.get("door", [])),
            "window": len(detections.get("window", [])),
            "room": len(rooms),
        }
        
        # Prepare response dict - use actual rooms list (all detected rooms)
        response = {
            "request_id": request_id,
            "meta": meta,
            "detections": detections,
            "rooms": rooms,  # All detected rooms (no override)
            "counts": counts,
            "downloads": {
                "overlay.png": f"/download/{request_id}/overlay.png",
                "data.csv": f"/download/{request_id}/data.csv",
                "data.json": f"/download/{request_id}/data.json",
            },
            "settings_used": settings_used,
            "warnings": warnings,
            "diagnostics": diagnostics,
        }
        
        # DEBUG OVERRIDE REMOVED: No longer forcing a single debug room
        # All detected rooms are now returned in the response
        
        # Final verification - check response dict directly
        final_rooms = response.get("rooms", [])
        logger.info(f"[ANALYZE] Final response rooms count: {len(final_rooms)}")
        logger.info(f"[ANALYZE] Final response['rooms'] is list: {isinstance(final_rooms, list)}")
        if isinstance(final_rooms, list) and len(final_rooms) > 0:
            logger.info(f"[ANALYZE] Final response['rooms'][0] keys: {list(final_rooms[0].keys()) if isinstance(final_rooms[0], dict) else 'not-a-dict'}")
            logger.info(f"[ANALYZE] Returning {len(final_rooms)} rooms in response (all detected rooms)")
        
        # ROOM DEBUG: Final check right before returning JSON response
        room_logger = logging.getLogger("archimind.rooms")
        room_logger.debug("ROOM DEBUG: rooms type=%s", type(rooms))
        try:
            room_logger.debug("ROOM DEBUG: rooms len=%d", len(rooms))
        except:
            room_logger.debug("ROOM DEBUG: rooms len=UNKNOWN")
        
        # Log first 5 rooms for debugging (not limiting, just logging)
        for i, r in enumerate(rooms[:5]):
            try:
                area = r.get("area_px", "unknown")
                room_logger.debug("ROOM DEBUG: #%d area_px=%s, keys=%s", i, area, list(r.keys()))
            except:
                room_logger.debug("ROOM DEBUG: #%d (unable to inspect room object)", i)
        
        # Also check response dict
        response_rooms = response.get("rooms", [])
        room_logger.debug("ROOM DEBUG: response['rooms'] type=%s", type(response_rooms))
        try:
            room_logger.debug("ROOM DEBUG: response['rooms'] len=%d", len(response_rooms))
        except:
            room_logger.debug("ROOM DEBUG: response['rooms'] len=UNKNOWN")
        
        # Create JSONResponse and verify one more time
        json_response = JSONResponse(response)
        logger.info(f"[ANALYZE] JSONResponse created, response['rooms'] length: {len(response['rooms'])}")
        
        return json_response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is (they already have proper status codes and messages)
        raise
    except Exception as e:
        # Catch any unexpected errors and return a proper error response
        logger.error(f"Unexpected error processing floor plan: {e}", exc_info=True)
        error_message = f"Internal server error: {str(e)}"
        # Don't expose internal details in production, but include them in logs
        if "detail" in str(e).lower():
            error_message = str(e)
        raise HTTPException(
            status_code=500,
            detail=error_message
        )


@app.get("/download/{request_id}/{filename}")
async def download_file(request_id: str, filename: str):
    """Download generated files."""
    request_dir = file_manager.get_request_dir(request_id)
    if not request_dir:
        raise HTTPException(status_code=404, detail="Request not found")
    
    file_path = request_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type
    ext = file_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".csv": "text/csv",
        ".json": "application/json",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    
    # Add Content-Disposition header to force download
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }
    
    return FileResponse(str(file_path), media_type=media_type, headers=headers)

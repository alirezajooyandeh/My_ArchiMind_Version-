# Floor Plan Analysis MVP

A local MVP application for analyzing floor plans using YOLO object detection. Upload floor plans (PDF/JPG/PNG), detect walls, doors, windows, and rooms, and export visual overlays and structured data.

## Features

- **Object Detection**: Detect walls, doors, windows, and rooms using trained YOLO weights
- **Visual Overlay**: View detections with customizable labels, room fills, and interior outlines
- **Structured Exports**: Export results as PNG overlay, CSV table, and JSON data
- **Scale Estimation**: Support for unknown, explicit, and scale bar measurement modes
- **Auto-Tune**: Automatically optimize detection parameters based on plan characteristics
- **Dark Theme UI**: Modern, responsive web interface
- **REST API**: Clean HTTP API for programmatic access

## Prerequisites

- Python 3.8 or higher
- pip package manager
- YOLO model weights (`.pt` files) for wall, door, window, and room detection
- poppler-utils (for PDF processing on Linux/Mac)

### Installing poppler-utils

**macOS:**
```bash
brew install poppler
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install poppler-utils
```

**Windows:**
Download poppler binaries from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) and add to PATH.

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "MVP-Version 2"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and set the paths to your YOLO model weights:
   ```env
   WALL_WEIGHTS_PATH=/path/to/wall.pt
   DOOR_WEIGHTS_PATH=/path/to/door.pt
   WINDOW_WEIGHTS_PATH=/path/to/window.pt
   ROOM_WEIGHTS_PATH=/path/to/room.pt
   ```

   **Note:** You can leave weights paths empty if you don't have models for all classes. The app will continue to work with available models.

## Running the Application

1. **Start the server:**
   ```bash
   python -m backend.main
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```

2. **Open your browser:**
   Navigate to `http://127.0.0.1:8000`

3. **Upload a floor plan:**
   - Drag and drop a PDF, JPG, or PNG file onto the upload zone
   - Adjust settings as needed
   - Click "Process"

## Sharing with Cloudflare Tunnel

To expose the local MVP server to the internet securely:

1. **First-time setup (macOS):**
   ```bash
   ./setup_cloudflare_tunnel.sh
   ```
   
   This will:
   - Install `cloudflared` via Homebrew
   - Authenticate with Cloudflare
   - Create the tunnel
   - Set up the configuration

2. **Start the backend server:**
   ```bash
   ./run_mvp.sh -p 8090
   ```

3. **In another terminal, run the tunnel:**
   ```bash
   cloudflared tunnel run archimind-mvp
   ```

4. **Share the public URL:**
   The tunnel will output a public HTTPS URL like:
   ```
   https://randomstring.trycloudflare.com
   ```
   
   This URL can be shared with other users and will securely forward to your local MVP UI.

**Note:** No CORS changes are needed because Cloudflare Tunnel forwards requests internally as if they came from localhost.

## Usage

### Web UI

1. **Settings Panel (Left):**
   - Adjust image size, confidence thresholds, display options
   - Configure PDF processing, advanced geometry parameters
   - Set scale mode (unknown, explicit pixels-per-foot, or scale bar measurement)

2. **Upload Zone (Center):**
   - Drag and drop or click to browse for floor plan files
   - Supported formats: PDF (first page), JPG, JPEG, PNG
   - Maximum file size: 50 MB (configurable)

3. **Results Tabs:**
   - **Overlay**: Visual overlay with detections, zoom controls, export buttons
   - **Table**: Tabular view of all detections and rooms
   - **JSON**: Full JSON response with all data

### Scale Bar Measurement

1. Process an image first
2. In Settings, select "Scale Bar" mode
3. Click "Start Measurement"
4. Click two points on the image (e.g., opposite ends of a known wall)
5. Enter the real-world distance in feet
6. Click "Apply" - the scale will be calculated and used for area calculations

### API Endpoints

#### `GET /healthz`
Check server health and model availability.

**Response:**
```json
{
  "status": "healthy",
  "device": "cpu",
  "models_loaded": {
    "wall": true,
    "door": true,
    "window": false,
    "room": true
  },
  "ready": true
}
```

#### `POST /process`
Process a floor plan.

**Form Data:**
- `file`: PDF/JPG/PNG file
- `imgsz`: Image size for inference (default: 1280)
- `wall_conf`, `door_conf`, `wind_conf`, `room_conf`: Confidence thresholds (0-1)
- `show_labels`: Boolean
- `fill_rooms`: Boolean
- `outline_interior`: Boolean
- `outline_thickness`: Integer
- `overlay_style`: "simple" or "detailed"
- `pdf_mode`: Boolean
- `pdf_dpi`: Integer (default: 300)
- `auto_tune`: Boolean
- `prefer_interior_footprint`: Boolean
- `wall_thick_px`: Integer
- `door_bridge_extra_px`: Integer
- `min_room_area_px`: Integer
- `footprint_grow_px`: Integer
- `scale_mode`: "unknown", "explicit", or "bar"
- `explicit_scale`: Float (pixels per foot, for explicit mode)
- `measure_point1_x`, `measure_point1_y`, `measure_point2_x`, `measure_point2_y`: Floats (for bar mode)
- `measure_distance_ft`: Float (for bar mode)

**Response:**
```json
{
  "request_id": "uuid",
  "meta": {
    "filename": "plan.pdf",
    "width": 2000,
    "height": 1500,
    "imgsz": 1280,
    "device": "cpu",
    "timings": {...}
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
  "warnings": []
}
```

#### `GET /download/{request_id}/{filename}`
Download generated files (overlay.png, data.csv, data.json).

## Configuration

### Environment Variables

Edit `.env` to configure:

- **Model Weights**: Paths to YOLO `.pt` files
- **Default Parameters**: Default confidence thresholds, image size
- **Server**: Host, port, max upload size
- **Device**: `auto`, `cpu`, or `cuda`
- **Temporary Files**: Directory and TTL for cleanup

### Settings Panel

All settings can be adjusted in the web UI without restarting the server.

## Testing Acceptance Criteria

### 1. Upload and Process
- ✅ Upload a PDF and get processed overlay, CSV, and JSON
- ✅ Upload a JPG and get processed overlay, CSV, and JSON

### 2. Settings Response
- ✅ Change `imgsz` and see results update
- ✅ Adjust per-class confidence thresholds and see detection changes
- ✅ Toggle `show_labels`, `fill_rooms`, `outline_interior` and see visual changes
- ✅ Change `outline_thickness` and see outline thickness change
- ✅ Switch `scale_mode` and `explicit_scale` and see area_sf appear/disappear
- ✅ Enable `auto_tune` and see optimized parameters in `settings_used`

### 3. Room Areas
- ✅ With `scale_mode=unknown`: rooms show `area_px` only, `area_sf` is null/empty
- ✅ With `scale_mode=explicit` and valid `explicit_scale`: rooms show both `area_px` and `area_sf`
- ✅ `area_sf` values are reasonable (e.g., 100-500 sq ft for typical rooms)

### 4. Missing Models
- ✅ If one weight path is missing, app still runs with remaining classes
- ✅ Warnings appear in both UI (alert) and JSON response

### 5. Download Buttons
- ✅ Export PNG downloads overlay image
- ✅ Export CSV downloads CSV table
- ✅ Export JSON downloads JSON data
- ✅ All downloads are tied to the last processed request

### 6. Health Endpoint
- ✅ Reports device (cpu/cuda)
- ✅ Reports model availability per class
- ✅ Reports ready status

### 7. Auto-Tune
- ✅ On a noisy plan, auto-tune increases `imgsz` and adjusts thresholds
- ✅ Chosen values appear in `settings_used`

## Project Structure

```
MVP-Version 2/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py             # Configuration management
│   ├── models.py             # YOLO model loading and inference
│   ├── image_utils.py        # Image/PDF processing
│   ├── geometry.py           # Room polygonization, area calculations
│   ├── overlay.py            # Overlay rendering
│   ├── auto_tune.py          # Auto-tuning logic
│   ├── exports.py            # CSV/JSON export
│   ├── scale.py              # Scale estimation
│   └── file_manager.py       # File management
├── frontend/
│   ├── index.html            # Main HTML
│   ├── styles.css            # Dark theme styles
│   └── app.js                # Frontend JavaScript
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore
└── README.md
```

## Troubleshooting

### Models Not Loading
- Check that weight file paths in `.env` are correct and files exist
- Verify file permissions
- Check logs for specific error messages

### PDF Processing Fails
- Ensure poppler-utils is installed and in PATH
- Try converting PDF to image manually first
- Check PDF is not corrupted

### Out of Memory
- Reduce `imgsz` in settings
- Disable `fill_rooms` option
- Process smaller images

### Slow Performance
- Enable GPU if available (set `DEVICE=cuda` in `.env`)
- Reduce `imgsz`
- Disable `auto_tune` for faster processing

## Development

### Running Tests

Basic unit tests are included in `tests/`:

```bash
python -m pytest tests/
```

### Adding New Classes

1. Add weight path to `.env`
2. Update `CLASS_COLORS` in `backend/overlay.py`
3. Add class name to frontend legend in `frontend/app.js`

## License

This is a local MVP application. Use as needed for your project.

## Support

For issues or questions, check the logs in the terminal where the server is running. Error messages are designed to be helpful and actionable.


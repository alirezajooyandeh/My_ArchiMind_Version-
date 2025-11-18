# Quick Start Guide

## 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Install poppler (for PDF processing)
# macOS:
brew install poppler

# Linux (Ubuntu/Debian):
sudo apt-get install poppler-utils
```

## 2. Configure Model Weights

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and add paths to your YOLO model weights:

```env
WALL_WEIGHTS_PATH=/path/to/your/wall.pt
DOOR_WEIGHTS_PATH=/path/to/your/door.pt
WINDOW_WEIGHTS_PATH=/path/to/your/window.pt
ROOM_WEIGHTS_PATH=/path/to/your/room.pt
```

**Note:** You can leave some paths empty if you don't have models for all classes. The app will work with available models.

## 3. Run the Application

```bash
python run.py
```

Or:

```bash
python -m backend.main
```

## 4. Open in Browser

Navigate to: `http://127.0.0.1:8000`

## 5. Test with a Floor Plan

1. Drag and drop a PDF, JPG, or PNG floor plan
2. Adjust settings if needed
3. Click "Process"
4. View results in the Overlay, Table, or JSON tabs
5. Export PNG, CSV, or JSON using the export buttons

## Troubleshooting

- **Models not loading?** Check `.env` file paths and file permissions
- **PDF errors?** Ensure poppler-utils is installed
- **Out of memory?** Reduce `imgsz` in settings or disable `fill_rooms`
- **Slow performance?** Enable GPU if available (set `DEVICE=cuda` in `.env`)

For more details, see [README.md](README.md).


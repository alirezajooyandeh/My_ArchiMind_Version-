ArchiMind — AI-Powered Floor Plan Understanding (MVP)

ArchiMind is an AI system that reads architectural floor plans and extracts building intelligence automatically.
This MVP demonstrates the core capabilities of the platform, built specifically for the architecture, engineering, and construction (AEC) industry.

What This MVP Does

Upload a floor plan (PNG, JPG, or PDF) and the system automatically:

1. Detects Architectural Components

Walls (straight and curved)

Doors

Windows

Rooms, including polygon extraction

2. Computes Room Areas

Automatic square footage (ft²) when scale is known

Pixel-based fallback when scale is unknown

3. Generates a Visual Overlay

Wall bounding boxes

Room color fills

Room labels and tooltips

Optional legend and layer toggles

4. Exports Data

JSON output for detections

PNG overlay image

Clean and simple web interface

This MVP is the foundation for the full ArchiMind platform, which will include automated CBC code compliance, fixture counting, ADA validation, occupancy calculations, and cost estimation.

Tech Stack
Backend

FastAPI

Python

Ultralytics YOLO (custom-trained models for walls, doors, windows, rooms)

OpenCV

Geometry and scaling utilities

Image preprocessing and noise filtering

Frontend

HTML

CSS

JavaScript

Dynamic overlays and event handling

Infrastructure

Local FastAPI server

Optional Cloudflare Tunnel

Isolated Python virtual environment

Project Structure
MVP-Version 2/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── models.py            # YOLO loading and inference
│   ├── geometry_v2.py       # Room and wall geometry logic
│   ├── image_utils.py       # Image preprocessing
│   ├── scale.py             # Scaling, unit conversion
│   └── overlay.py           # Drawing engine for overlays
│
├── frontend/
│   ├── index.html           # Main interface
│   ├── landing.html         # Landing splash page
│   ├── styles.css           # Styling
│   └── app.js               # Frontend logic
│
├── tests/
│   ├── test_api.py
│   ├── test_exports.py
│   └── test_geometry.py
│
├── run_mvp.sh               # Startup script
├── cloudflared.yml          # Tunnel configuration
├── requirements.txt         # Dependencies
├── README.md
└── QUICKSTART.md            # Developer setup guide

How to Run Locally

Clone the repository

git clone https://github.com/alirezajooyandeh/My_ArchiMind_Version-.git
cd My_ArchiMind_Version-


Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate


Install dependencies

pip install -r requirements.txt


Start the server

./run_mvp.sh


The application becomes available at:

Main MVP interface: http://localhost:8090/mvp

Landing page: http://localhost:8090/

Model Details

The MVP uses four custom-trained YOLO models:

Model	Purpose
wall.pt	Wall detection (straight and curved)
door.pt	Door detection
window.pt	Window detection
room.pt	Room segmentation and area extraction

Training involved a large dataset of annotated architectural floor plans, including:

Curved wall augmentation

Multi-resolution training (1280, 1536, 1920)

Mosaic, rotation, tiling, and geometric augmentation

Roadmap
Short-Term Features

ADA door clearance validation

CBC-based occupancy and egress calculations

Fixture detection (toilets, sinks, urinals, showers)

OCR-based room naming

Room polygon repair and smoothing

Medium-Term Features

AI-based CBC 2022 compliance engine

Automated architectural specification generator

Revit plugin for direct integration

Space programming and optimization tools

Long-Term Vision

ArchiMind becomes an AI engine for architectural design and documentation:

Automated QA/QC

Permit-ready drawing validation

Probabilistic cost estimation

Automated construction documentation

Author

Ali Jooyandeh
Architectural Job Captain (K-12 Design)
AI and Deep Learning Developer
Founder of ArchiMind

Contact: alirezajooyandeh@gmail.com

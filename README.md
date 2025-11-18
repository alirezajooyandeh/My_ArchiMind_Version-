🏛️ ArchiMind — AI-Powered Floor Plan Understanding MVP

Automated wall, door, window, and room detection for architectural drawings

ArchiMind is an AI tool that analyzes architectural floor plans and extracts building intelligence instantly.
This early MVP demonstrates the core capabilities of the platform—built specifically for the architecture, engineering, and construction (AEC) industry.

🚀 What This MVP Does

The system takes an uploaded floor plan image (PNG/JPG/PDF) and automatically:

1. Detects Architectural Components

🧱 Walls (straight + curved)

🚪 Doors

🪟 Windows

🏠 Rooms (with area calculations)

2. Computes Room Areas

Automatic square footage (ft²)

Pixel area fallback when scaling is unknown

3. Renders a Visual Overlay

Wall boxes

Room fill colors

Labels & tooltips

Interactive legend

4. Exports Key Data

JSON detection output

Visual PNG overlay

Simple interaction through a clean web interface

This MVP is the foundation of the full ArchiMind roadmap, which aims to include automated CBC code compliance, fixture counting, ADA validation, and cost estimation.

🏗️ Tech Stack
Backend (FastAPI)

Python

Ultralytics YOLO models (Wall/Window/Door/Room)

OpenCV

Geometry tools (Shapely-like utilities)

NMS thresholding controls

Robust image preprocessing

Frontend

HTML

CSS

JavaScript

Dynamic overlays

Interactive UI components

Infrastructure

Cloudflare Tunnel (optional)

Local development FastAPI server

Virtual environment (.venv)

📦 Project Structure
MVP-Version 2/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── models.py            # YOLO model loading + inference
│   ├── geometry_v2.py       # Room + wall geometry logic
│   ├── image_utils.py       # Pre/post-processing
│   ├── scale.py             # Unit scaling + ft² logic
│   └── overlay.py           # Drawing overlays
│
├── frontend/
│   ├── index.html           # Main MVP UI
│   ├── landing.html         # Landing page
│   ├── styles.css           # Stylesheet
│   └── app.js               # Main UI logic
│
├── tests/
│   ├── test_api.py          # API tests
│   ├── test_exports.py
│   └── test_geometry.py
│
├── run_mvp.sh               # Quick start script
├── cloudflared.yml          # Tunnel config
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── QUICKSTART.md            # Fast developer setup

⚙️ How to Run Locally
1. Clone the repo
git clone https://github.com/alirezajooyandeh/My_ArchiMind_Version-.git
cd My_ArchiMind_Version-

2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the server
./run_mvp.sh


The app will be available at:

Main app: http://localhost:8090/mvp

Landing page: http://localhost:8090/

🧠 Model Details

The MVP uses 4 custom-trained YOLO models:

Model	Purpose
wall.pt	Wall detection (segmentation + bounding)
door.pt	Door detection
window.pt	Window detection
room.pt	Room segmentation + area extraction

Training involved:

Hundreds of annotated architectural floor plans

Curved wall augmentation

Multi-scale training (1280/1536/1920)

Advanced augmentation (tiling, mosaic, rotation)

📡 Roadmap (Upcoming Features)
Short-Term

ADA door clearance checking

Automatic CBC-based occupancy & egress calculations

Fixture counting (toilets, sinks, urinals, showers)

Room naming via OCR

Smart room polygon repair

Medium-Term

AI-based code compliance engine (CBC 2022)

Dynamic architectural specs generator

Revit plug-in

Space programming & optimization

Long-Term (Vision)

ArchiMind becomes the “AI brain” of architecture firms:

Automated QA/QC

Permit-ready drawing validation

Cost estimation

Construction documentation automation

👤 Author

Ali Jooyandeh
Architectural Job Captain (K-12 Design)
AI/Deep Learning Developer
Founder, ArchiMind

📧 Contact

If you’re interested in contributing, partnering, or becoming a co-founder:

📩 alirezajooyandeh@gmail.com

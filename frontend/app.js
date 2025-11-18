/**
 * ArchiMind MVP - Frontend Application
 * 
 * This file handles all frontend interactions for the floor plan analysis tool.
 * 
 * Key Features:
 * - File upload and preview
 * - Settings management (sliders, toggles, advanced options)
 * - Layer visibility control (Walls, Doors, Windows, Rooms, Labels)
 * - Image processing and overlay display
 * - Room hover interactions with tooltips
 * - Zoom and pan controls for floor plan navigation
 * - Snapshot history (last 3 processed results)
 * - Export functionality (PNG, CSV, JSON)
 * - Auto-tune visual feedback
 * - Error handling and status display
 * 
 * Architecture:
 * - State management: Global variables for current request, image, results
 * - Event-driven: DOM event listeners for user interactions
 * - API integration: RESTful calls to FastAPI backend
 * - Canvas rendering: HTML5 Canvas for overlay display
 * 
 * Dependencies:
 * - FastAPI backend running on same origin (or configured CORS)
 * - Modern browser with ES6+ support
 */

// Main application JavaScript
const API_BASE = '';

let currentRequestId = null;
let currentImage = null;
let currentOverlay = null;
let currentResults = null;
let measureMode = false;
let measurePoints = [];
let pixelsPerFoot = null;
let resetZoomPan = null; // Function to reset zoom and pan
let snapshotHistory = []; // Store last 3 processed results
let roomPolygons = []; // Store room polygon data for hover interactions

// Line style controls
let lineThickness = 1;  // px (default: 1)
let lineStyle = "solid"; // "solid" | "dashed" | "dotted"

// Global default visibility
const layerVisibility = {
    walls: true,
    doors: true,
    windows: true,
    roomsFill: true,
    roomOutlines: false,
    labels: false,
};

// Helper to read all layer toggles safely from DOM
// TEMP: DISABLED - using constants instead to avoid .checked errors
// function getLayerVisibilityFromDOM() {
//     // Start from current defaults / state
//     const vis = { ...layerVisibility };
//     
//     const toggles = document.querySelectorAll(".layer-toggle");
//     if (!toggles || toggles.length === 0) {
//         console.warn("[Layers] No .layer-toggle elements found; using existing visibility:", vis);
//         return vis;
//     }
//     
//     toggles.forEach((input) => {
//         if (!input) return;
//         const key = input.dataset.layer;
//         if (!key) return;
//         if (input.checked !== undefined) {
//             vis[key] = !!input.checked;
//         }
//     });
//     
//     console.debug("[Layers] computed visibility from DOM:", vis);
//     return vis;
// }

// Safe style helper - prevents errors when element is null
function safeStyle(el, apply) {
    if (!el || !el.style) return;
    apply(el.style);
}

// --- Safe checkbox helper ---
function safeChecked(id, fallback = false) {
    const el = document.getElementById(id);
    if (!el) {
        console.warn(`ArchiMind: checkbox #${id} not found, using fallback=${fallback}`);
        return fallback;
    }
    return !!el.checked;
}

// Safe checkbox helper - prevents errors when checkbox element is null
// TEMP: DISABLED - removed .checked access
// function getCheckboxChecked(id, defaultValue = true) {
//     const el = document.getElementById(id);
//     if (!el) {
//         console.warn("[getCheckboxChecked] checkbox not found:", id, "-> using default", defaultValue);
//         return defaultValue;
//     }
//     return !!el.checked;
// }

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeLandingScreen();
    initializeSettings();
    initializeLayerManager();
    initializeUpload();
    initializeTabs();
    initializeZoom();
    initializeRefresh();
    initializeErrorBanner();
    initializeExportButtons();
    initializeExportButtonsWiring(); // Wire up export button handlers
    initializeSmartDefaults();
    initializeGridToggle();
    initializeLineStyleControls();
    checkHealth();
});

// Landing screen initialization
function initializeLandingScreen() {
    const landing = document.getElementById("archimind-landing-screen");
    const appScreen = document.getElementById("archimind-app-screen");
    const enterBtn = document.getElementById("enter-archimind");

    if (!landing || !appScreen || !enterBtn) {
        console.warn("ArchiMind landing elements not found; skipping landing flow.");
        return;
    }

    // Initial state: show landing, hide app
    landing.classList.add("visible-screen");
    appScreen.classList.add("hidden-screen");

    enterBtn.addEventListener("click", function () {
        // Fade out landing, fade in app
        landing.classList.remove("visible-screen");
        landing.classList.add("hidden-screen");

        appScreen.classList.remove("hidden-screen");
        appScreen.classList.add("visible-screen");
    });
}

// Settings initialization
function initializeSettings() {
    // Update value displays for sliders
    const sliders = [
        'imgsz', 'wallConf', 'doorConf', 'windConf', 'roomConf',
        'binarizationThreshold', 'noiseReductionStrength', 'contrastBoost',
        'wallNmsIou', 'doorWindowNmsIou', 'roomPolygonSmoothing', 'roomMinConfidence'
    ];
    sliders.forEach(id => {
        const slider = document.getElementById(id);
        const display = document.getElementById(id + 'Value');
        if (slider && display) {
            slider.addEventListener('input', () => {
                const value = parseFloat(slider.value);
                // Format based on slider type
                if (id === 'roomMinConfidence' || id === 'binarizationThreshold') {
                    display.textContent = Math.round(value);
                } else if (id === 'imgsz') {
                    display.textContent = Math.round(value);
                } else {
                    display.textContent = value.toFixed(2);
                }
                // Update NMS hints if this is an NMS slider
                if (id === 'wallNmsIou' || id === 'doorWindowNmsIou') {
                    updateNmsHints();
                }
            });
        }
    });
    
    // Handle noise reduction strength toggle
    const noiseReduction = document.getElementById('noiseReduction');
    const noiseReductionStrengthGroup = document.getElementById('noiseReductionStrengthGroup');
    if (noiseReduction && noiseReductionStrengthGroup) {
        noiseReduction.addEventListener('change', () => {
            const isChecked = safeChecked('noiseReduction', false);
            safeStyle(noiseReductionStrengthGroup, s => {
                s.display = isChecked ? 'block' : 'none';
            });
        });
    }
    
    // Handle high-quality rasterization - auto-set DPI to 600 when enabled
    const highQualityRasterization = document.getElementById('highQualityRasterization');
    const pdfDpi = document.getElementById('pdfDpi');
    if (highQualityRasterization && pdfDpi) {
        highQualityRasterization.addEventListener('change', () => {
            const isChecked = safeChecked('highQualityRasterization', false);
            if (isChecked) {
                pdfDpi.value = 600;
            } else {
                pdfDpi.value = 300;
            }
        });
    }

    // Collapsible sections
    document.querySelectorAll('.collapse-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const targetId = btn.getAttribute('data-target');
            const content = document.getElementById(targetId);
            if (content) {
                content.classList.toggle('collapsed');
            }
        });
    });
    
    // Make entire header clickable for collapsible sections
    document.querySelectorAll('.setting-group-header').forEach(header => {
        const btn = header.querySelector('.collapse-btn');
        if (btn) {
            header.addEventListener('click', (e) => {
                if (e.target !== btn) {
                    btn.click();
                }
            });
            safeStyle(header, s => {
                s.cursor = 'pointer';
            });
        }
    });

    // Scale mode radio buttons
    const scaleRadios = document.querySelectorAll('input[name="scaleMode"]');
    const explicitGroup = document.getElementById('explicitScaleGroup');
    const barGroup = document.getElementById('barScaleGroup');
    const archContainer = document.getElementById('scale-arch-container');
    
    // Function to update scale visibility
    function updateScaleVisibility() {
        const modeExplicit = document.getElementById("scale-mode-explicit")?.checked || false;
        safeStyle(explicitGroup, s => {
            s.display = modeExplicit ? 'block' : 'none';
        });
        safeStyle(archContainer, s => {
            s.display = modeExplicit ? 'block' : 'none';
        });
        safeStyle(barGroup, s => {
            s.display = document.getElementById("scale-mode-bar")?.checked ? 'block' : 'none';
        });
    }
    
    scaleRadios.forEach(radio => {
        radio.addEventListener('change', updateScaleVisibility);
    });
    
    // Call once on load
    updateScaleVisibility();
    
    // Initialize NMS hints on page load
    updateNmsHints();

    // Measure tool
    const startMeasureBtn = document.getElementById('startMeasureBtn');
    const measureInputGroup = document.getElementById('measureInputGroup');
    const applyMeasureBtn = document.getElementById('applyMeasureBtn');
    
    startMeasureBtn.addEventListener('click', () => {
        if (!currentImage) {
            alert('Please process an image first');
            return;
        }
        startMeasurement();
    });
    
    applyMeasureBtn.addEventListener('click', () => {
        const distance = parseFloat(document.getElementById('measureDistance').value);
        if (distance > 0 && measurePoints.length === 2) {
            applyMeasurement(measurePoints[0], measurePoints[1], distance);
        }
    });
}

// Layer Manager - handle layer visibility with normalized checkboxes
function initializeLayerManager() {
    const toggles = document.querySelectorAll(".layer-toggle");
    
    if (!toggles || toggles.length === 0) {
        console.error("[LayerManager] No layer-toggle elements found in DOM.");
        return;
    }
    
    toggles.forEach((input) => {
        if (!input) return; // skip nulls
        
        const key = input.dataset.layer;
        if (!key) {
            console.warn("[LayerManager] Checkbox missing data-layer attribute", input);
            return;
        }
        
        // Initialize state safely using safeChecked
        if (input.id) {
            layerVisibility[key] = safeChecked(input.id, layerVisibility[key] || false);
        }
        
        // Handle changes on the toggle
        input.addEventListener("change", () => {
            // Safe: we already checked input exists above, and we're in an event handler
            if (input && input.checked !== undefined) {
                const isChecked = !!input.checked;
                layerVisibility[key] = isChecked;
                console.debug("[LayerManager] toggled:", key, layerVisibility[key]);
                
                // Re-process if we have results
                if (currentResults && currentImage) {
                    applyLayerVisibility();
                }
            }
        });
    });
    
    console.log("[LayerManager] Initialized with", toggles.length, "layer toggles");
}

// Redraw overlay with current layer visibility (without re-processing)
function redrawOverlayWithLayers() {
    const canvas = document.getElementById('overlayCanvas');
    if (!canvas || !currentOverlay || !currentResults) {
        return;
    }
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Redraw the base overlay image
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentOverlay, 0, 0);
    
    // Redraw window boxes if windows layer is enabled
    if (currentResults.detections && currentResults.detections.window && currentResults.detections.window.length > 0) {
        const container = document.getElementById('imageContainer');
        if (container && currentOverlay) {
            // Scale factors for internal canvas coordinates (1:1 since canvas matches image)
            const internalScaleX = 1.0;
            const internalScaleY = 1.0;
            
            // Draw windows using exact detection box dimensions
            renderWindowsLayer(ctx, currentResults.detections.window, internalScaleX, internalScaleY, layerVisibility);
        }
    }
}

// Apply layer visibility by re-processing the image with updated settings
function applyLayerVisibility() {
    // First try to redraw without re-processing (faster for frontend-only layers like windows)
    redrawOverlayWithLayers();
    
    // Check if we have a file to process
    const fileInput = document.getElementById('fileInput');
    if (!currentResults || !fileInput || !fileInput.files || !fileInput.files[0]) {
        console.warn('Cannot apply layer visibility: no current results or file');
        return;
    }
    
    console.log('Layer visibility changed - re-processing with new settings...');
    
    // Re-process the file with current layer visibility settings
    // This will use the current checkbox states which are already synced
    processFile();
}

// Upload initialization
function initializeUpload() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const processBtn = document.getElementById('processBtn');
    const reprocessBtn = document.getElementById('reprocessBtn');

    uploadZone.addEventListener('click', () => fileInput.click());
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    processBtn.addEventListener('click', processFile);
    if (reprocessBtn) {
        reprocessBtn.addEventListener('click', () => {
            if (currentImage) {
                processFile();
            } else {
                alert('Please upload a file first');
            }
        });
    }
    
    // Also handle reprocess button in results section
    const reprocessBtnResults = document.getElementById('reprocessBtnResults');
    if (reprocessBtnResults) {
        reprocessBtnResults.addEventListener('click', () => {
            if (currentImage) {
                processFile();
            } else {
                alert('Please upload a file first');
            }
        });
    }
}

function handleFileSelect(file) {
    const uploadZone = document.getElementById('uploadZone');
    const processBtn = document.getElementById('processBtn');
    const reprocessBtn = document.getElementById('reprocessBtn');
    const filePreview = document.getElementById('filePreview');
    const filePreviewThumb = document.getElementById('filePreviewThumb');
    const filePreviewName = document.getElementById('filePreviewName');
    const filePreviewSize = document.getElementById('filePreviewSize');
    const filePreviewResolution = document.getElementById('filePreviewResolution');
    const filePreviewPdfInfo = document.getElementById('filePreviewPdfInfo');
    
    // Validate file type
    const ext = file.name.toLowerCase().split('.').pop();
    if (!['pdf', 'jpg', 'jpeg', 'png'].includes(ext)) {
        showError('Please select a PDF, JPG, or PNG file');
        return;
    }
    
    // Validate file size (50 MB)
    if (file.size > 50 * 1024 * 1024) {
        showError('File size must be less than 50 MB');
        return;
    }
    
    uploadZone.querySelector('.upload-content h3').textContent = file.name;
    processBtn.disabled = false;
    currentImage = file;
    
    // Hide reprocess buttons when new file is selected
    safeStyle(reprocessBtn, s => { s.display = 'none'; });
    const reprocessBtnResults = document.getElementById('reprocessBtnResults');
    safeStyle(reprocessBtnResults, s => { s.display = 'none'; });
    
    // Show file preview
    filePreviewName.textContent = file.name;
    filePreviewSize.textContent = formatFileSize(file.size);
    
    // Handle PDF vs image preview
    if (ext === 'pdf') {
        safeStyle(filePreviewPdfInfo, s => { s.display = 'block'; });
        const pdfDpi = document.getElementById('pdfDpi')?.value;
        // TEMP: use constant instead of reading checkbox
        const highQuality = false;
        if (filePreviewPdfInfo) {
            filePreviewPdfInfo.textContent = `Using first page at ${highQuality ? '600' : pdfDpi} DPI`;
        }
        // PDF preview - show placeholder or first page if possible
        if (filePreviewThumb) {
            filePreviewThumb.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAiIGhlaWdodD0iODAiIHZpZXdCb3g9IjAgMCA4MCA4MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjgwIiBoZWlnaHQ9IjgwIiBmaWxsPSIjMkQyRDJEIi8+CjxwYXRoIGQ9Ik0yNSAyNUg1NVY1NUgyNVYyNVoiIHN0cm9rZT0iI0UwRTBFMCIgc3Ryb2tlLXdpZHRoPSIyIi8+CjxwYXRoIGQ9Ik0zMCAzNUg1ME0zMCA0MEg1ME0zMCA0NUg1MCIgc3Ryb2tlPSIjRTBFMEUwIiBzdHJva2Utd2lkdGg9IjIiLz4KPC9zdmc+';
        }
    } else {
        safeStyle(filePreviewPdfInfo, s => { s.display = 'none'; });
        // Create thumbnail for image and analyze
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                filePreviewThumb.src = e.target.result;
                filePreviewResolution.textContent = `${img.width} × ${img.height} pixels`;
                
                // Analyze image and suggest settings
                analyzeImageForPreview(img);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
    
    safeStyle(filePreview, s => { s.display = 'flex'; });
}

// Analyze image for preview suggestions
function analyzeImageForPreview(img) {
    // Create a temporary canvas to analyze the image
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    
    // Get image data
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    
    // Calculate grayscale level
    let totalBrightness = 0;
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const brightness = (r + g + b) / 3;
        totalBrightness += brightness;
    }
    const avgBrightness = totalBrightness / (data.length / 4);
    const grayscaleLevel = Math.round((avgBrightness / 255) * 100);
    
    // Suggest image size based on dimensions
    const maxDim = Math.max(img.width, img.height);
    let suggestedSize = 1536;
    if (maxDim > 3000) suggestedSize = 2048;
    else if (maxDim > 2000) suggestedSize = 1536;
    else if (maxDim > 1000) suggestedSize = 1024;
    else suggestedSize = 640;
    
    // Update preview with detection info
    const filePreviewResolution = document.getElementById('filePreviewResolution');
    if (filePreviewResolution) {
        filePreviewResolution.innerHTML = `
            ${img.width} × ${img.height} pixels<br>
            <small style="color: var(--text-secondary);">Grayscale: ${grayscaleLevel}% | Suggested size: ${suggestedSize}px</small>
        `;
    }
    
    // Optionally auto-set suggested size
    const imgsz = document.getElementById('imgsz');
    const imgszValue = document.getElementById('imgszValue');
    if (imgsz && imgszValue) {
        // Don't auto-set, just show suggestion
        // imgsz.value = suggestedSize;
        // imgszValue.textContent = suggestedSize;
    }
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Process file
async function processFile() {
    if (!currentImage) return;
    
    // Get DOM elements
    const processBtn = document.getElementById('processBtn');
    const reprocessBtn = document.getElementById('reprocessBtn');
    const reprocessBtnResults = document.getElementById('reprocessBtnResults');
    const statusTextEl = document.getElementById('statusText');
    const uploadSection = document.getElementById('uploadSection');
    const resultsSection = document.getElementById('resultsSection');
    
    if (!statusTextEl) {
        console.error('statusText element not found');
        return;
    }
    
    // Update UI for processing state
    if (processBtn) {
        processBtn.disabled = true;
        processBtn.classList.add('processing');
        const btnText = processBtn.querySelector('.btn-text');
        const btnSpinner = processBtn.querySelector('.btn-spinner');
        if (btnText) btnText.textContent = 'Processing…';
        safeStyle(btnSpinner, s => { s.display = 'inline-block'; });
    }
    
    // Show processing caption
    const processingCaption = document.getElementById('processingCaption');
    safeStyle(processingCaption, s => { s.display = 'block'; });
    if (reprocessBtn) {
        reprocessBtn.disabled = true;
        reprocessBtn.classList.add('processing');
    }
    if (reprocessBtnResults) {
        reprocessBtnResults.disabled = true;
        reprocessBtnResults.classList.add('processing');
    }
    
    // Update status indicator
    const statusIndicator = document.getElementById('statusIndicator');
    if (statusIndicator) {
        statusIndicator.classList.add('processing');
    }
    statusTextEl.textContent = 'Processing...';
    
    // Hide error banner
    hideError();
    
    // Hide processing time display
    const processingTimeEl = document.getElementById('processingTime');
    safeStyle(processingTimeEl, s => { s.display = 'none'; });
    
    const startTime = Date.now();
    
    try {
        const formData = new FormData();
        formData.append('file', currentImage);
        
        // Get all settings - use layer manager values (with null checks)
        const imgsz = document.getElementById('imgsz');
        const wallConf = document.getElementById('wallConf');
        const doorConf = document.getElementById('doorConf');
        const windConf = document.getElementById('windConf');
        const roomConf = document.getElementById('roomConf');
        const wallThickPx = document.getElementById('wallThickPx');
        const doorBridgePx = document.getElementById('doorBridgePx');
        const minRoomAreaPx = document.getElementById('minRoomAreaPx');
        const footprintGrowPx = document.getElementById('footprintGrowPx');
        
        formData.append('imgsz', imgsz ? imgsz.value : '1536');
        formData.append('wall_conf', wallConf ? wallConf.value : '0.60');
        formData.append('door_conf', doorConf ? doorConf.value : '0.50');
        formData.append('wind_conf', windConf ? windConf.value : '0.80');
        formData.append('room_conf', roomConf ? roomConf.value : '0.25');
        formData.append('wall_thick_px', wallThickPx ? wallThickPx.value : '12');
        formData.append('door_bridge_extra_px', doorBridgePx ? doorBridgePx.value : '12');
        formData.append('min_room_area_px', minRoomAreaPx ? minRoomAreaPx.value : '1000');
        formData.append('footprint_grow_px', footprintGrowPx ? footprintGrowPx.value : '3');
        // ---------- layer toggles ----------
        const showWalls        = safeChecked("walls-toggle", true);
        const showDoors        = safeChecked("doors-toggle", true);
        const showWindows      = safeChecked("windows-toggle", true);
        const showRoomsFill    = safeChecked("rooms-fill-toggle", true);
        const showRoomOutlines = safeChecked("room-outlines-toggle", false);
        const showLabels       = safeChecked("labels-toggle", true);
        
        formData.append("show_walls",         String(showWalls));
        formData.append("show_doors",         String(showDoors));
        formData.append("show_windows",       String(showWindows));
        formData.append("show_rooms_fill",    String(showRoomsFill));
        formData.append("show_room_outlines", String(showRoomOutlines));
        formData.append("show_labels",        String(showLabels));
        
        // Line style controls
        formData.append("line_thickness", String(lineThickness));
        formData.append("line_style", lineStyle);
        
        // DEBUG: ensure values actually change when I click the checkbox
        console.log("[ArchiMind] layer flags", {
            showWalls,
            showDoors,
            showWindows,
            showRoomsFill,
            showRoomOutlines,
            showLabels,
            lineThickness,
            lineStyle,
        });
        const outlineThickness = document.getElementById('outlineThickness');
        const overlayStyle = document.getElementById('overlayStyle');
        const pdfMode = document.getElementById('pdfMode');
        const pdfDpi = document.getElementById('pdfDpi');
        const autoTune = document.getElementById('autoTune');
        const preferInteriorFootprint = document.getElementById('preferInteriorFootprint');
        
        formData.append('outline_thickness', outlineThickness ? outlineThickness.value : '2');
        formData.append('overlay_style', overlayStyle ? overlayStyle.value : 'detailed');
        // Use safeChecked for all checkboxes
        formData.append('pdf_mode', safeChecked('pdfMode', true));
        formData.append('pdf_dpi', pdfDpi ? pdfDpi.value : '300');
        formData.append('auto_tune', safeChecked('autoTune', false));
        formData.append('prefer_interior_footprint', safeChecked('preferInteriorFootprint', true));
        
        // Image Pre-Processing Settings (with null checks)
        const binarizationThreshold = document.getElementById('binarizationThreshold');
        const adaptiveThreshold = document.getElementById('adaptiveThreshold');
        const edgeSharpening = document.getElementById('edgeSharpening');
        const noiseReduction = document.getElementById('noiseReduction');
        const noiseReductionStrength = document.getElementById('noiseReductionStrength');
        const contrastBoost = document.getElementById('contrastBoost');
        
        formData.append('binarization_threshold', binarizationThreshold ? binarizationThreshold.value : '0');
        // Use safeChecked for all checkboxes
        formData.append('adaptive_threshold', safeChecked('adaptiveThreshold', false));
        formData.append('edge_sharpening', safeChecked('edgeSharpening', false));
        const noiseReductionChecked = safeChecked('noiseReduction', false);
        formData.append('noise_reduction', noiseReductionChecked);
        if (noiseReductionChecked && noiseReductionStrength) {
            formData.append('noise_reduction_strength', noiseReductionStrength.value);
        }
        formData.append('contrast_boost', contrastBoost ? contrastBoost.value : '0');
        
        // PDF Enhancement Options
        formData.append('vector_extraction_mode', safeChecked('vectorExtractionMode', false));
        formData.append('high_quality_rasterization', safeChecked('highQualityRasterization', false));
        formData.append('deskew_rotation_correction', safeChecked('deskewRotationCorrection', false));
        
        // Model Inference Settings (with null checks)
        const wallNmsIou = document.getElementById('wallNmsIou');
        const doorWindowNmsIou = document.getElementById('doorWindowNmsIou');
        const segmentationMaskRefinement = document.getElementById('segmentationMaskRefinement');
        const roomPolygonSmoothing = document.getElementById('roomPolygonSmoothing');
        const roomMinConfidence = document.getElementById('roomMinConfidence');
        
        formData.append('wall_nms_iou', wallNmsIou ? wallNmsIou.value : '0.04');
        formData.append('doorwin_nms_iou', doorWindowNmsIou ? doorWindowNmsIou.value : '0.05');
        // Use safeChecked for all checkboxes
        formData.append('segmentation_mask_refinement', safeChecked('segmentationMaskRefinement', false));
        formData.append('room_polygon_smoothing', roomPolygonSmoothing ? roomPolygonSmoothing.value : '0.1');
        formData.append('room_min_confidence', roomMinConfidence ? roomMinConfidence.value : '40');
        
        // Geometry & Structure Settings
        const minWallLength = document.getElementById('minWallLength');
        const minDoorGapWidth = document.getElementById('minDoorGapWidth');
        const mergeWallsThreshold = document.getElementById('mergeWallsThreshold');
        
        formData.append('min_wall_length', minWallLength ? minWallLength.value : '10');
        formData.append('min_door_gap_width', minDoorGapWidth ? minDoorGapWidth.value : '6');
        formData.append('merge_walls_threshold', mergeWallsThreshold ? mergeWallsThreshold.value : '12');
        formData.append('snap_walls_to_angles', safeChecked('snapWallsToAngles', false));
        formData.append('correct_broken_walls', safeChecked('correctBrokenWalls', false));
        
        // Room Detection Settings
        const minInternalGap = document.getElementById('minInternalGap');
        
        formData.append('room_hole_filling', safeChecked('roomHoleFilling', false));
        formData.append('min_internal_gap', minInternalGap ? minInternalGap.value : '0');
        formData.append('detect_scale_automatically', safeChecked('detectScaleAutomatically', false));
        formData.append('door_removal_for_rooms', safeChecked('doorRemovalForRooms', false));
        
        // Smart Assist Tools
        const drawingType = document.getElementById('drawingType');
        
        formData.append('auto_select_best_image_size', safeChecked('autoSelectBestImageSize', false));
        formData.append('auto_adjust_model_thresholds', safeChecked('autoAdjustModelThresholds', false));
        formData.append('drawing_type', drawingType ? drawingType.value : 'plan');
        
        // Debug & Analysis
        formData.append('show_confidence_heatmap', safeChecked('showConfidenceHeatmap', false));
        formData.append('show_raw_bounding_boxes', safeChecked('showRawBoundingBoxes', false));
        formData.append('log_preprocessing_steps', safeChecked('logPreprocessingSteps', false));
        
        // ---- scale mode ----
        let scaleMode = "unknown";
        if (document.getElementById("scale-mode-explicit")?.checked) {
            scaleMode = "explicit";
        } else if (document.getElementById("scale-mode-bar")?.checked) {
            scaleMode = "bar";
        }
        
        const scaleArchSelect = document.getElementById("scale-arch-select");
        const scaleFeetPerInch = scaleArchSelect ? scaleArchSelect.value : "";
        
        // Send to backend
        formData.append("scale_mode", scaleMode);
        formData.append("scale_feet_per_inch", scaleFeetPerInch);
        
        // Also send explicit_scale for backward compatibility (pixels per foot)
        if (scaleMode === 'explicit') {
            const explicitScaleEl = document.getElementById('explicitScale');
            const explicitScale = explicitScaleEl ? explicitScaleEl.value : null;
            if (explicitScale) {
                formData.append('explicit_scale', explicitScale);
            }
        } else if (scaleMode === 'bar' && pixelsPerFoot) {
            // Use previously measured scale
            formData.append('explicit_scale', pixelsPerFoot);
            formData.append('scale_mode', 'explicit');
        }
        
        // ============================================
        // API REQUEST/RESPONSE STRUCTURE
        // ============================================
        // Expected Request:
        //   - POST /process
        //   - FormData with: file, imgsz, wall_conf, door_conf, wind_conf, room_conf, etc.
        //
        // Expected Success Response (HTTP 200):
        //   {
        //     "request_id": "uuid-string",
        //     "meta": {
        //       "request_id": "uuid-string",
        //       "filename": "file.jpg",
        //       "width": 1920,
        //       "height": 1080,
        //       "imgsz": 1536,
        //       "device": "cpu",
        //       "timings": {
        //         "load": 0.016,
        //         "preprocess": 0.0,
        //         "infer": 2.806,
        //         "postprocess": 0.046,
        //         "render": 0.008,
        //         "total": 2.884
        //       },
        //       "pixels_per_foot": null or float
        //     },
        //     "detections": {
        //       "wall": [...],
        //       "door": [...],
        //       "window": [...],
        //       "room": [...]
        //     },
        //     "rooms": [...],
        //     "downloads": {
        //       "overlay.png": "/download/{request_id}/overlay.png",
        //       "data.csv": "/download/{request_id}/data.csv",
        //       "data.json": "/download/{request_id}/data.json"
        //     },
        //     "settings_used": {...},
        //     "warnings": [],
        //     "diagnostics": {}
        //   }
        //
        // Expected Error Response (HTTP 4xx/5xx):
        //   {
        //     "detail": "Error message here"
        //   }
        // ============================================
        
        const response = await fetch(`${API_BASE}/process`, {
            method: 'POST',
            body: formData,
        });
        
        // Handle HTTP errors
        if (!response.ok) {
            let errorMessage = `HTTP ${response.status}: Processing failed`;
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
                // If response is not JSON, try to get text
                try {
                    const errorText = await response.text();
                    if (errorText) errorMessage = errorText;
                } catch (e2) {
                    // Use default message
                }
            }
            throw new Error(errorMessage);
        }
        
        // Parse successful response
        const result = await response.json();
        
        // DEBUG: Log the entire response structure
        console.log('[DEBUG] Full response received:', result);
        console.log('[DEBUG] Response keys:', Object.keys(result));
        console.log('[DEBUG] result.rooms exists?', 'rooms' in result);
        console.log('[DEBUG] result.rooms type:', typeof result.rooms);
        console.log('[DEBUG] result.rooms value:', result.rooms);
        console.log('[DEBUG] result.rooms length:', result.rooms ? result.rooms.length : 'N/A');
        
        // Validate response structure
        if (!result.request_id) {
            throw new Error('Invalid response: missing request_id');
        }
        if (!result.meta || !result.meta.timings) {
            throw new Error('Invalid response: missing meta.timings');
        }
        currentRequestId = result.request_id;
        currentResults = result;
        
        // Store room polygons for hover interactions
        roomPolygons = result.rooms || [];
        
        // Update NMS hints if they exist
        updateNmsHints();
        
        // DEBUG: Log rooms for debugging
        console.log(`[DEBUG] Received ${roomPolygons.length} rooms from backend`);
        if (roomPolygons.length > 0) {
            console.log('[DEBUG] First room:', roomPolygons[0]);
            console.log('[DEBUG] Room has polygon?', !!roomPolygons[0].polygon);
            console.log('[DEBUG] Room polygon length:', roomPolygons[0].polygon ? roomPolygons[0].polygon.length : 0);
        } else {
            console.error('[DEBUG] ERROR: rooms array is empty!');
            console.error('[DEBUG] result.rooms:', result.rooms);
            console.error('[DEBUG] result.counts:', result.counts);
            
            // Log room diagnostics if available
            if (result.diagnostics && result.diagnostics.rooms) {
                const roomDiag = result.diagnostics.rooms;
                console.error('[DEBUG] ROOM DIAGNOSTICS:', {
                    stage: roomDiag.stage,
                    yolo_detections_count: roomDiag.yolo_detections_count,
                    processed_count: roomDiag.processed_count,
                    rooms_from_detector_count: roomDiag.rooms_from_detector_count,
                    filtered_rooms_count: roomDiag.filtered_rooms_count,
                    final_rooms_count: roomDiag.final_rooms_count,
                    skipped_no_bbox: roomDiag.skipped_no_bbox,
                    skipped_invalid_box: roomDiag.skipped_invalid_box,
                    skipped_tiny_area: roomDiag.skipped_tiny_area,
                    filtered_out_count: roomDiag.filtered_out_count,
                    errors: roomDiag.errors,
                    warning: roomDiag.warning,
                    room_confidence_threshold: roomDiag.room_confidence_threshold,
                    effective_min_area: roomDiag.effective_min_area,
                    user_min_area_px: roomDiag.user_min_area_px,
                    raw_areas: roomDiag.raw_areas,
                    candidate_areas: roomDiag.candidate_areas,
                    filtered_out_areas: roomDiag.filtered_out_areas,
                });
            } else {
                console.warn('[DEBUG] No room diagnostics found in response');
            }
        }
        
        // Calculate processing time
        const processingTimeMs = Date.now() - startTime;
        const processingTimeSec = (processingTimeMs / 1000).toFixed(2);
        
        // Update status
        const timings = result.meta.timings;
        const timingsText = `Load: ${(timings.load * 1000).toFixed(0)}ms | ` +
                          `Preprocess: ${(timings.preprocess * 1000).toFixed(0)}ms | ` +
                          `Infer: ${(timings.infer * 1000).toFixed(0)}ms | ` +
                          `Postprocess: ${(timings.postprocess * 1000).toFixed(0)}ms | ` +
                          `Render: ${(timings.render * 1000).toFixed(0)}ms | ` +
                          `Total: ${(timings.total * 1000).toFixed(0)}ms`;
        const timingsTextEl = document.getElementById('timingsText');
        if (timingsTextEl) {
            timingsTextEl.textContent = timingsText;
        }
        
        // Show processing time with detailed breakdown
        const processingTimeEl = document.getElementById('processingTime');
        if (processingTimeEl) {
            const timings = result.meta.timings;
            const inferenceTime = (timings.infer * 1000).toFixed(0);
            processingTimeEl.innerHTML = `
                Processed in ${processingTimeSec}s | 
                <strong>Inference: ${inferenceTime}ms</strong> | 
                Total: ${(timings.total * 1000).toFixed(0)}ms
            `;
            safeStyle(processingTimeEl, s => { s.display = 'block'; });
        }
        
        // Update status indicator
        const statusIndicator = document.getElementById('statusIndicator');
        if (statusIndicator) {
            statusIndicator.classList.remove('processing');
        }
        statusTextEl.textContent = 'Complete';
        
        // Handle auto-tune visual feedback
        // TEMP: DISABLED - removed checkbox check
        // if (result.settings_used && getCheckboxChecked('autoTune', false)) {
        //     highlightAutoTunedSettings(result.settings_used);
        // }
        
        // Add to snapshot history
        addToSnapshotHistory(result, processingTimeSec);
        
        // Show results
        safeStyle(uploadSection, s => { s.display = 'none'; });
        safeStyle(resultsSection, s => { s.display = 'flex'; });
        
        // Show reprocess buttons
        const reprocessBtnEl = document.getElementById('reprocessBtn');
        const reprocessBtnResultsEl = document.getElementById('reprocessBtnResults');
        safeStyle(reprocessBtnEl, s => { s.display = 'block'; });
        safeStyle(reprocessBtnResultsEl, s => { s.display = 'block'; });
        
        // Display results
        displayOverlay(result);
        displayTable(result);
        displayJSON(result);
        
        // Update scale display if available
        updateScaleDisplay(result);
        
        // Show warnings in status bar with clear messaging
        if (result.warnings && result.warnings.length > 0) {
            if (statusTextEl) statusTextEl.textContent = 'Warnings: ' + result.warnings.join('; ');
            safeStyle(statusTextEl, s => { s.color = 'var(--warning)'; });
            
            // Show diagnostic info if no rooms found
            if (result.rooms && result.rooms.length === 0 && result.diagnostics) {
                const skippedReason = result.diagnostics.skipped_reason;
                if (skippedReason && statusTextEl) {
                    statusTextEl.textContent = skippedReason;
                    safeStyle(statusTextEl, s => { s.color = 'var(--error)'; });
                }
            }
            
            // Also show in console
            console.warn('Processing warnings:', result.warnings);
            if (result.diagnostics) {
                console.log('Diagnostics:', result.diagnostics);
            }
        } else {
            if (statusTextEl) statusTextEl.textContent = 'Complete';
            safeStyle(statusTextEl, s => { s.color = 'var(--text-primary)'; });
        }
        
        // Warning banner removed per user request
        
    } catch (error) {
        // Show detailed error message
        const errorMessage = error.message || 'Processing failed. Please check the file format and try again.';
        console.error('Processing error:', error);
        
        // Show error banner with detailed message
        showError(errorMessage, error.stack || error.toString());
        
        // Update status with detailed error message
        const statusIndicator = document.getElementById('statusIndicator');
        if (statusIndicator) {
            statusIndicator.classList.remove('processing');
        }
        if (statusTextEl) {
            // Show first 100 chars of error message in status bar
            const shortMessage = errorMessage.length > 100 
                ? errorMessage.substring(0, 100) + '...' 
                : errorMessage;
            statusTextEl.textContent = `Error: ${shortMessage}`;
            safeStyle(statusTextEl, s => { s.color = 'var(--error)'; });
        }
    } finally {
        // Reset processing state
        if (processBtn) {
            processBtn.disabled = false;
            processBtn.classList.remove('processing');
            const btnText = processBtn.querySelector('.btn-text');
            const btnSpinner = processBtn.querySelector('.btn-spinner');
            if (btnText) btnText.textContent = 'Process';
            safeStyle(btnSpinner, s => { s.display = 'none'; });
        }
        
        // Hide processing caption
        const processingCaption = document.getElementById('processingCaption');
        safeStyle(processingCaption, s => { s.display = 'none'; });
        if (reprocessBtn) {
            reprocessBtn.disabled = false;
            reprocessBtn.classList.remove('processing');
            reprocessBtn.textContent = 'Reprocess';
        }
        if (reprocessBtnResults) {
            reprocessBtnResults.disabled = false;
            reprocessBtnResults.classList.remove('processing');
        }
    }
}

// Display overlay
async function displayOverlay(result) {
    const canvas = document.getElementById('overlayCanvas');
    if (!canvas) {
        console.error('Canvas element not found');
        return;
    }
    
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.error('Could not get canvas context');
        return;
    }
    
    // Load overlay image
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
        try {
            // Set canvas internal dimensions to match image
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            currentOverlay = img;
            
            // Set canvas display size to maintain aspect ratio
            const container = document.getElementById('imageContainer');
            if (container) {
                const containerRect = container.getBoundingClientRect();
                const containerWidth = containerRect.width;
                const containerHeight = containerRect.height;
                
                // Calculate scale to fit container while maintaining aspect ratio
                const scaleX = containerWidth / img.width;
                const scaleY = containerHeight / img.height;
                const scale = Math.min(scaleX, scaleY, 1); // Don't scale up beyond 100%
                
                const displayWidth = img.width * scale;
                const displayHeight = img.height * scale;
                
                safeStyle(canvas, s => {
                    s.width = displayWidth + 'px';
                    s.height = displayHeight + 'px';
                });
            }
            
            // Reset zoom and pan when new image loads
            if (resetZoomPan) {
                resetZoomPan();
            }
            
            // Redraw grid if it was visible
            if (gridVisible) {
                const gridCanvas = document.getElementById('gridCanvas');
                if (gridCanvas && container) {
                    const containerRect = container.getBoundingClientRect();
                    const containerWidth = containerRect.width;
                    const containerHeight = containerRect.height;
                    const scaleX = containerWidth / img.width;
                    const scaleY = containerHeight / img.height;
                    const scale = Math.min(scaleX, scaleY, 1);
                    const displayWidth = img.width * scale;
                    const displayHeight = img.height * scale;
                    drawGrid(gridCanvas, displayWidth, displayHeight, img.width, img.height);
                }
            }
            
            // Update legend
            updateLegend(result);
            
            // Draw window boxes on top of overlay (if windows layer is enabled)
            if (result.detections && result.detections.window && result.detections.window.length > 0) {
                const container = document.getElementById('imageContainer');
                if (container) {
                    const containerRect = container.getBoundingClientRect();
                    const containerWidth = containerRect.width;
                    const containerHeight = containerRect.height;
                    
                    // Calculate scale factors (same as used for canvas display)
                    const scaleX = containerWidth / img.width;
                    const scaleY = containerHeight / img.height;
                    const scale = Math.min(scaleX, scaleY, 1);
                    
                    // Scale factors for internal canvas coordinates
                    const internalScaleX = 1.0; // Canvas internal dimensions match image
                    const internalScaleY = 1.0;
                    
                    // Draw windows using exact detection box dimensions
                    renderWindowsLayer(ctx, result.detections.window, internalScaleX, internalScaleY, layerVisibility);
                }
            }
            
            // Initialize room hover interactions
            initializeRoomHover(canvas, result);
        } catch (error) {
            console.error('Error drawing overlay image:', error);
            showError('Failed to display overlay image', error.toString());
        }
    };
    
    img.onerror = (error) => {
        console.error('Error loading overlay image:', error);
        const errorMsg = `Failed to load overlay image. Please check if the file exists at: ${API_BASE}/download/${result.request_id}/overlay.png`;
        showError(errorMsg, error.toString());
    };
    
    img.src = `${API_BASE}/download/${result.request_id}/overlay.png`;
}

function updateLegend(result) {
    const legend = document.getElementById('legend');
    if (!legend) return; // Legend element not found (results section not visible yet)
    legend.innerHTML = '';
    
    const classes = ['wall', 'door', 'window'];
    const colors = {
        wall: '#ff6464',
        door: '#64ff64',
        window: '#6464ff',
    };
    
    // Show detection classes
    classes.forEach(cls => {
        if (result.detections[cls] && result.detections[cls].length > 0) {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background-color: ${colors[cls]}"></div>
                <span>${cls.charAt(0).toUpperCase() + cls.slice(1)}</span>
            `;
            legend.appendChild(item);
        }
    });
    
    // Show room count (always show, even if 0)
    const roomCount = result.rooms ? result.rooms.length : (result.meta && result.meta.room_count !== undefined ? result.meta.room_count : 0);
    
    // DEBUG: Log room count calculation
    console.log(`[DEBUG] updateLegend: roomCount=${roomCount}, result.rooms=`, result.rooms);
    
    const item = document.createElement('div');
    item.className = 'legend-item';
    if (roomCount === 0) {
        safeStyle(item, s => { s.color = 'var(--error)'; });
    }
    item.innerHTML = `
        <div class="legend-color" style="background: linear-gradient(90deg, #ffb6c1, #90ee90)"></div>
        <span>Rooms (${roomCount})</span>
    `;
    legend.appendChild(item);
}

// Format room area for display (prefers ft² if available, falls back to px²)
function formatRoomArea(room) {
    if (room.area_sqft != null || room.area_sf != null) {
        const val = Number(room.area_sqft || room.area_sf);
        if (Number.isFinite(val)) {
            return `${val.toFixed(1)} sq ft`;
        }
    }
    // fallback
    if (room.area_px2 != null || room.area_px != null) {
        const pxVal = Number(room.area_px2 || room.area_px);
        if (Number.isFinite(pxVal)) {
            return `${Math.round(pxVal)} px²`;
        }
    }
    return "—";
}

// Display table
function displayTable(result) {
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';
    
    // Add detections
    let id = 0;
    Object.entries(result.detections).forEach(([class_name, detections]) => {
        detections.forEach(det => {
            const row = tbody.insertRow();
            row.insertCell(0).textContent = id++;
            row.insertCell(1).textContent = class_name;
            row.insertCell(2).textContent = det.confidence.toFixed(4);
            row.insertCell(3).textContent = `[${det.bbox.map(v => v.toFixed(1)).join(', ')}]`;
            row.insertCell(4).textContent = det.polygon ? 'Yes' : 'No';
            row.insertCell(5).textContent = '';
            row.insertCell(6).textContent = '';
            row.insertCell(7).textContent = '';
        });
    });
    
    // Add rooms
    result.rooms.forEach(room => {
        const row = tbody.insertRow();
        row.insertCell(0).textContent = room.id;
        row.insertCell(1).textContent = 'room';
        row.insertCell(2).textContent = '';
        row.insertCell(3).textContent = '';
        row.insertCell(4).textContent = room.polygon ? 'Yes' : 'No';
        // Use formatRoomArea to display area (prefers ft² if available)
        const areaText = formatRoomArea(room);
        row.insertCell(5).textContent = areaText;
        // Show px² in second column if we have both
        if ((room.area_sqft != null || room.area_sf != null) && (room.area_px2 != null || room.area_px != null)) {
            const pxVal = Number(room.area_px2 || room.area_px);
            row.insertCell(6).textContent = Number.isFinite(pxVal) ? `${Math.round(pxVal)} px²` : '';
        } else {
            row.insertCell(6).textContent = '';
        }
        row.insertCell(7).textContent = room.centroid ? `[${room.centroid[0].toFixed(1)}, ${room.centroid[1].toFixed(1)}]` : '';
    });
}

// Display JSON
function displayJSON(result) {
    const jsonDisplay = document.getElementById('jsonDisplay');
    const jsonString = JSON.stringify(result, null, 2);
    
    // Escape HTML to prevent XSS, then convert URLs to clickable links
    const escapeHtml = (text) => {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    };
    
    // First escape the entire JSON string
    let htmlString = escapeHtml(jsonString);
    
    // Then convert URLs to clickable links
    // Match URLs that look like /download/... or http://... or https://...
    // Pattern: "url" where url starts with /download/ or http:// or https://
    const urlRegex = /&quot;((?:https?:\/\/[^&]+|(?:\/download\/[^&]+)))&quot;/g;
    htmlString = htmlString.replace(urlRegex, (match, url) => {
        // Make it an absolute URL if it's a relative path
        const fullUrl = url.startsWith('http') ? url : `${API_BASE}${url}`;
        // For download links, add download attribute to force download
        const isDownloadLink = url.includes('/download/');
        const downloadAttr = isDownloadLink ? ` download="${url.split('/').pop()}"` : '';
        return `<a href="${fullUrl}"${downloadAttr} target="_blank" style="color: #3b82f6; text-decoration: underline; cursor: pointer;">&quot;${url}&quot;</a>`;
    });
    
    jsonDisplay.innerHTML = htmlString;
}

// Tabs
function initializeTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(tabName + 'Tab').classList.add('active');
        });
    });
}

// Zoom controls with pan functionality
let zoomState = { zoom: 1, panX: 0, panY: 0 };

function initializeZoom() {
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let startPanX = 0;
    let startPanY = 0;
    
    const canvas = document.getElementById('overlayCanvas');
    const container = document.getElementById('imageContainer');
    
    if (!canvas || !container) {
        console.warn('Canvas or container not found - zoom initialization skipped');
        return;
    }
    
    const zoomIn = document.getElementById('zoomIn');
    const zoomOut = document.getElementById('zoomOut');
    const zoomReset = document.getElementById('zoomReset');
    
    if (!zoomIn || !zoomOut || !zoomReset) {
        console.warn('Zoom controls not found - they may not be visible yet');
        return;
    }
    
    zoomIn.addEventListener('click', () => {
        zoomState.zoom = Math.min(zoomState.zoom * 1.2, 5);
        applyTransform();
    });
    
    zoomOut.addEventListener('click', () => {
        zoomState.zoom = Math.max(zoomState.zoom / 1.2, 0.2);
        applyTransform();
    });
    
    zoomReset.addEventListener('click', () => {
        zoomState.zoom = 1;
        zoomState.panX = 0;
        zoomState.panY = 0;
        applyTransform();
    });
    
    function applyTransform() {
        if (currentOverlay) {
            safeStyle(canvas, s => {
                s.transform = `translate(${zoomState.panX}px, ${zoomState.panY}px) scale(${zoomState.zoom})`;
                s.transformOrigin = 'top left';
            });
            
            // Update cursor based on zoom level
            safeStyle(container, s => {
                s.cursor = zoomState.zoom > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default';
            });
        }
    }
    
    // Pan functionality - mouse events
    container.addEventListener('mousedown', (e) => {
        if (zoomState.zoom > 1 && !measureMode) {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            startPanX = zoomState.panX;
            startPanY = zoomState.panY;
            safeStyle(container, s => { s.cursor = 'grabbing'; });
            e.preventDefault();
        }
    });
    
    container.addEventListener('mousemove', (e) => {
        if (isDragging && zoomState.zoom > 1) {
            const deltaX = e.clientX - startX;
            const deltaY = e.clientY - startY;
            zoomState.panX = startPanX + deltaX;
            zoomState.panY = startPanY + deltaY;
            applyTransform();
        }
    });
    
    container.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            safeStyle(container, s => {
                s.cursor = zoomState.zoom > 1 ? 'grab' : 'default';
            });
        }
    });
    
    container.addEventListener('mouseleave', () => {
        if (isDragging) {
            isDragging = false;
            safeStyle(container, s => {
                s.cursor = zoomState.zoom > 1 ? 'grab' : 'default';
            });
        }
    });
    
    // Pan functionality - touch events for mobile
    let touchStartX = 0;
    let touchStartY = 0;
    let touchStartPanX = 0;
    let touchStartPanY = 0;
    
    container.addEventListener('touchstart', (e) => {
        if (zoomState.zoom > 1 && !measureMode && e.touches.length === 1) {
            isDragging = true;
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
            touchStartPanX = zoomState.panX;
            touchStartPanY = zoomState.panY;
            e.preventDefault();
        }
    });
    
    container.addEventListener('touchmove', (e) => {
        if (isDragging && zoomState.zoom > 1 && e.touches.length === 1) {
            const deltaX = e.touches[0].clientX - touchStartX;
            const deltaY = e.touches[0].clientY - touchStartY;
            zoomState.panX = touchStartPanX + deltaX;
            zoomState.panY = touchStartPanY + deltaY;
            applyTransform();
            e.preventDefault();
        }
    });
    
    container.addEventListener('touchend', () => {
        isDragging = false;
    });
    
    // Initial transform
    applyTransform();
    
    // Expose reset function
    resetZoomPan = () => {
        zoomState.zoom = 1;
        zoomState.panX = 0;
        zoomState.panY = 0;
        applyTransform();
    };
}

// Export button handlers (both canvas and sidebar)
const exportPngHandler = () => {
    if (currentRequestId) {
        window.open(`${API_BASE}/download/${currentRequestId}/overlay.png`, '_blank');
    }
};

const exportCsvHandler = () => {
    if (currentRequestId) {
        window.open(`${API_BASE}/download/${currentRequestId}/data.csv`, '_blank');
    }
};

const exportJsonHandler = () => {
    if (currentRequestId) {
        window.open(`${API_BASE}/download/${currentRequestId}/data.json`, '_blank');
    }
};

// Wire up export buttons - moved to DOMContentLoaded to ensure DOM is ready
function initializeExportButtonsWiring() {
    // Wire up export buttons (canvas area - hidden but kept for compatibility)
    const exportPng = document.getElementById('exportPng');
    const exportCsv = document.getElementById('exportCsv');
    const exportJson = document.getElementById('exportJson');
    if (exportPng) exportPng.addEventListener('click', exportPngHandler);
    if (exportCsv) exportCsv.addEventListener('click', exportCsvHandler);
    if (exportJson) exportJson.addEventListener('click', exportJsonHandler);

    // Wire up sidebar export buttons
    const exportPngSidebar = document.getElementById('exportPngSidebar');
    const exportCsvSidebar = document.getElementById('exportCsvSidebar');
    const exportJsonSidebar = document.getElementById('exportJsonSidebar');
    if (exportPngSidebar) exportPngSidebar.addEventListener('click', exportPngHandler);
    if (exportCsvSidebar) exportCsvSidebar.addEventListener('click', exportCsvHandler);
    if (exportJsonSidebar) exportJsonSidebar.addEventListener('click', exportJsonHandler);
}

// Measurement tool
function startMeasurement() {
    measureMode = true;
    measurePoints = [];
    const canvas = document.getElementById('overlayCanvas');
    const measureCanvas = document.getElementById('measureCanvas');
    
    if (!canvas || !measureCanvas) return;
    
    safeStyle(measureCanvas, s => { s.display = 'block'; });
    // Set internal dimensions to match overlay canvas
    measureCanvas.width = canvas.width;
    measureCanvas.height = canvas.height;
    
    // Set display size and position to match overlay canvas
    const rect = canvas.getBoundingClientRect();
    const container = document.getElementById('imageContainer');
    if (container) {
        const containerRect = container.getBoundingClientRect();
        safeStyle(measureCanvas, s => {
            s.position = 'absolute';
            s.top = (rect.top - containerRect.top) + 'px';
            s.left = (rect.left - containerRect.left) + 'px';
            s.width = rect.width + 'px';
            s.height = rect.height + 'px';
        });
    }
    
    const ctx = measureCanvas.getContext('2d');
    if (ctx) {
        ctx.clearRect(0, 0, measureCanvas.width, measureCanvas.height);
    }
    
    safeStyle(canvas, s => { s.cursor = 'crosshair'; });
    safeStyle(measureCanvas, s => { s.cursor = 'crosshair'; });
    
    canvas.addEventListener('click', handleMeasureClick);
}

function handleMeasureClick(e) {
    if (!measureMode) return;
    
    const canvas = document.getElementById('overlayCanvas');
    const measureCanvas = document.getElementById('measureCanvas');
    const rect = canvas.getBoundingClientRect();
    // Use internal canvas dimensions vs display dimensions for accurate coordinate mapping
    const internalWidth = canvas.width;
    const internalHeight = canvas.height;
    const displayWidth = rect.width;
    const displayHeight = rect.height;
    const scaleX = internalWidth / displayWidth;
    const scaleY = internalHeight / displayHeight;
    
    // Account for zoom and pan
    const displayX = (e.clientX - rect.left - zoomState.panX) / zoomState.zoom;
    const displayY = (e.clientY - rect.top - zoomState.panY) / zoomState.zoom;
    const x = displayX * scaleX;
    const y = displayY * scaleY;
    
    measurePoints.push([x, y]);
    
    const ctx = measureCanvas.getContext('2d');
    ctx.fillStyle = '#ff0000';
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fill();
    
    if (measurePoints.length === 1) {
        // First point
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
    } else if (measurePoints.length === 2) {
        // Draw line
        ctx.beginPath();
        ctx.moveTo(measurePoints[0][0], measurePoints[0][1]);
        ctx.lineTo(measurePoints[1][0], measurePoints[1][1]);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Show input
        const measureInputGroup = document.getElementById('measureInputGroup');
        safeStyle(measureInputGroup, s => { s.display = 'block'; });
        canvas.removeEventListener('click', handleMeasureClick);
        safeStyle(canvas, s => { s.cursor = 'default'; });
        safeStyle(measureCanvas, s => { s.cursor = 'default'; });
    }
}

function applyMeasurement(point1, point2, distanceFt) {
    const dx = point2[0] - point1[0];
    const dy = point2[1] - point1[1];
    const pixelDistance = Math.sqrt(dx * dx + dy * dy);
    
    pixelsPerFoot = pixelDistance / distanceFt;
    
    // Update explicit scale input
    const explicitScale = document.getElementById('explicitScale');
    if (explicitScale) explicitScale.value = pixelsPerFoot.toFixed(2);
    // TEMP: DISABLED - removed .checked write
    // const explicitRadio = document.querySelector('input[name="scaleMode"][value="explicit"]');
    // if (explicitRadio && explicitRadio.checked !== undefined) {
    //     explicitRadio.checked = true;
    // }
    const explicitScaleGroup = document.getElementById('explicitScaleGroup');
    const barScaleGroup = document.getElementById('barScaleGroup');
    safeStyle(explicitScaleGroup, s => { s.display = 'block'; });
    safeStyle(barScaleGroup, s => { s.display = 'none'; });
    
    // Reset measurement
    measureMode = false;
    measurePoints = [];
    const measureCanvas = document.getElementById('measureCanvas');
    const measureInputGroup = document.getElementById('measureInputGroup');
    safeStyle(measureCanvas, s => { s.display = 'none'; });
    safeStyle(measureInputGroup, s => { s.display = 'none'; });
    
    alert(`Scale set to ${pixelsPerFoot.toFixed(2)} pixels per foot`);
}

// Refresh button
function initializeRefresh() {
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            // Add rotation animation
            safeStyle(refreshBtn, s => {
                s.transform = 'rotate(360deg)';
                s.transition = 'transform 0.5s';
            });
            
            // Reload page after animation
            setTimeout(() => {
                window.location.reload();
            }, 500);
        });
    }
}

// Show/hide no rooms warning
function showNoRoomsWarning(diagnostics) {
    const warningBanner = document.getElementById('noRoomsWarning');
    const reasonText = document.getElementById('noRoomsReason');
    
    if (reasonText) {
        if (diagnostics && diagnostics.skipped_reason) {
            reasonText.textContent = diagnostics.skipped_reason;
        } else {
            reasonText.textContent = 'No rooms could be detected. Check settings and try adjusting thresholds.';
        }
    }
    
    safeStyle(warningBanner, s => { s.display = 'block'; });
}

function hideNoRoomsWarning() {
    const warningBanner = document.getElementById('noRoomsWarning');
    safeStyle(warningBanner, s => { s.display = 'none'; });
}

// Health check
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/healthz`);
        const health = await response.json();
        
        if (!health.ready) {
            const statusText = document.getElementById('statusText');
            if (statusText) {
                statusText.textContent = 'Models not loaded';
                safeStyle(statusText, s => { s.color = 'var(--warning)'; });
            }
        }
    } catch (error) {
        console.error('Health check failed:', error);
        const statusText = document.getElementById('statusText');
        if (statusText) {
            statusText.textContent = 'Connection error';
            safeStyle(statusText, s => { s.color = 'var(--error)'; });
        }
    }
}

// ============================================
// NEW FEATURES
// ============================================

// Room hover and selection interactions
// Separate state: hoveredRoom (for hover highlight/label) vs selectedRoom (for right-side card)
let hoveredRoom = null;
let selectedRoom = null;

function initializeRoomHover(canvas, result) {
    const container = document.getElementById('imageContainer');
    const hoverTooltip = document.getElementById('room-hover-tooltip');
    const roomInfoCard = document.getElementById('roomInfoCard');
    const roomInfoContent = document.getElementById('roomInfoContent');
    const closeRoomInfoBtn = document.getElementById('closeRoomInfo');
    
    let lastUpdateTime = 0;
    const UPDATE_THROTTLE_MS = 16; // ~60fps
    
    // Reset state when initializing
    hoveredRoom = null;
    selectedRoom = null;
    
    // Store room polygons with their data
    roomPolygons = (result.rooms || []).map(room => ({
        id: room.id || 0,
        polygon: room.polygon || [],
        area_px: room.area_px || 0,
        area_sf: room.area_sf || null,
        centroid: room.centroid || [0, 0]
    }));
    
    // Close button handler
    if (closeRoomInfoBtn) {
        closeRoomInfoBtn.addEventListener('click', () => {
            setSelectedRoom(null, roomInfoCard, roomInfoContent);
        });
    }
    
    // Hover logic - only updates hoveredRoom, does NOT touch selectedRoom
    container.addEventListener('mousemove', (e) => {
        // Throttle updates to prevent shaking
        const now = Date.now();
        if (now - lastUpdateTime < UPDATE_THROTTLE_MS) {
            return;
        }
        lastUpdateTime = now;
        if (!canvas || roomPolygons.length === 0) return;
        
        const rect = canvas.getBoundingClientRect();
        // Use internal canvas dimensions vs display dimensions for accurate coordinate mapping
        const internalWidth = canvas.width;
        const internalHeight = canvas.height;
        const displayWidth = rect.width;
        const displayHeight = rect.height;
        const scaleX = internalWidth / displayWidth;
        const scaleY = internalHeight / displayHeight;
        
        // Account for zoom and pan - convert display coordinates to internal canvas coordinates
        const displayX = (e.clientX - rect.left - zoomState.panX) / zoomState.zoom;
        const displayY = (e.clientY - rect.top - zoomState.panY) / zoomState.zoom;
        const x = displayX * scaleX;
        const y = displayY * scaleY;
        
        // Check if point is inside any room polygon
        let foundRoom = null;
        for (const room of roomPolygons) {
            if (isPointInPolygon([x, y], room.polygon)) {
                foundRoom = room;
                break;
            }
        }
        
        // Only update hoveredRoom on hover - do NOT update selectedRoom
        if (foundRoom && foundRoom !== hoveredRoom) {
            setHoveredRoom(canvas, foundRoom, hoverTooltip, e);
        } else if (!foundRoom && hoveredRoom) {
            clearHoveredRoom(canvas, hoverTooltip);
        } else if (foundRoom) {
            // Update tooltip position to follow mouse cursor (simple fixed positioning)
            showRoomHoverTooltip(foundRoom, hoverTooltip, e);
        }
    });
    
    // Click logic - sets selectedRoom and updates right-side card
    // Only handle clicks on the canvas, not on UI elements (buttons, cards, etc.)
    container.addEventListener('click', (e) => {
        // Don't handle clicks on UI elements (buttons, cards, etc.)
        if (e.target.closest('button') || e.target.closest('.room-info-card') || 
            e.target.closest('.zoom-controls-float') || e.target.closest('.legend-float')) {
            return;
        }
        
        if (!canvas || roomPolygons.length === 0) return;
        
        const rect = canvas.getBoundingClientRect();
        const internalWidth = canvas.width;
        const internalHeight = canvas.height;
        const displayWidth = rect.width;
        const displayHeight = rect.height;
        const scaleX = internalWidth / displayWidth;
        const scaleY = internalHeight / displayHeight;
        
        // Account for zoom and pan
        const displayX = (e.clientX - rect.left - zoomState.panX) / zoomState.zoom;
        const displayY = (e.clientY - rect.top - zoomState.panY) / zoomState.zoom;
        const x = displayX * scaleX;
        const y = displayY * scaleY;
        
        // Check if click is inside any room polygon
        let clickedRoom = null;
        for (const room of roomPolygons) {
            if (isPointInPolygon([x, y], room.polygon)) {
                clickedRoom = room;
                break;
            }
        }
        
        // Set selectedRoom and update right-side card
        if (clickedRoom) {
            setSelectedRoom(clickedRoom, roomInfoCard, roomInfoContent);
        } else {
            // Click outside any room - deselect
            setSelectedRoom(null, roomInfoCard, roomInfoContent);
        }
    });
    
    container.addEventListener('mouseleave', () => {
        clearHoveredRoom(canvas, hoverTooltip);
    });
}

// Helper function to apply stroke style (thickness and line style)
function applyStrokeStyle(ctx, thickness, style) {
    ctx.lineWidth = thickness || 2;
    
    if (style === "dashed") {
        ctx.setLineDash([5, 5]);
    } else if (style === "dotted") {
        ctx.setLineDash([2, 2]);
    } else {
        // solid
        ctx.setLineDash([]);
    }
}

// Window box drawing functions - shrinks based on aspect ratio to fit inner window symbol
function drawWindowBox(ctx, winBox, scaleX, scaleY, color) {
    // winBox is in model coords (x1, y1, x2, y2)
    // Handle both array format [x1, y1, x2, y2] and object format {x1, y1, x2, y2}
    let x1, y1, x2, y2;
    if (Array.isArray(winBox)) {
        [x1, y1, x2, y2] = winBox;
    } else if (winBox.bbox && Array.isArray(winBox.bbox)) {
        [x1, y1, x2, y2] = winBox.bbox;
    } else {
        x1 = winBox.x1 || winBox[0];
        y1 = winBox.y1 || winBox[1];
        x2 = winBox.x2 || winBox[2];
        y2 = winBox.y2 || winBox[3];
    }

    // Model coords → canvas coords
    const scaledX1 = x1 * scaleX;
    const scaledY1 = y1 * scaleY;
    const scaledX2 = x2 * scaleX;
    const scaledY2 = y2 * scaleY;

    const w = scaledX2 - scaledX1;
    const h = scaledY2 - scaledY1;

    // Center of the detection
    const cx = scaledX1 + w / 2;
    const cy = scaledY1 + h / 2;

    // Shrink differently along major/minor axis:
    // - For horizontal windows (w > h): shrink height a lot (remove wall),
    //   shrink width only a bit.
    // - For vertical windows (h >= w): inverse.
    let innerW, innerH;

    if (w > h) {
        // Horizontal window
        innerW = w * 0.85; // keep 85% of width
        innerH = h * 0.35; // keep 35% of height (mostly remove wall above/below)
    } else {
        // Vertical window
        innerW = w * 0.35; // keep 35% of width
        innerH = h * 0.85; // keep 85% of height
    }

    const drawX = cx - innerW / 2;
    const drawY = cy - innerH / 2;

    ctx.beginPath();
    ctx.rect(drawX, drawY, innerW, innerH);

    // Use global lineThickness / lineStyle if available,
    // otherwise set some safe defaults:
    applyStrokeStyle(ctx, lineThickness || 2, lineStyle || "solid");

    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.setLineDash([]); // reset dashes for next layers
}

function renderWindowsLayer(ctx, windows, scaleX, scaleY, layerFlags) {
    // Only draw if "Windows" layer is ON
    if (!layerFlags || !layerFlags.windows) return;
    if (!Array.isArray(windows) || windows.length === 0) return;

    const windowColor = "#ffcc33"; // yellow

    windows.forEach(win => {
        // Handle both direct bbox array and detection object format
        const winBox = win.bbox || win.box || win;
        if (winBox) {
            drawWindowBox(ctx, winBox, scaleX, scaleY, windowColor);
        }
    });
}

// Room polygon drawing functions - FILL ONLY, NO STROKE

// --- REMOVE ROOM OUTLINE STROKES ---
// This function is completely disabled - room outlines are never drawn
function drawRoomOutlines(ctx, rooms, color) {
    return; // <-- disable drawing entirely
}

function drawRoomPolygon(ctx, polygon, fillColor) {
    if (!polygon || polygon.length < 3) return;

    ctx.beginPath();
    ctx.moveTo(polygon[0][0], polygon[0][1]);
    for (let i = 1; i < polygon.length; i++) {
        ctx.lineTo(polygon[i][0], polygon[i][1]);
    }
    ctx.closePath();

    // FILL ONLY – no stroke
    // NO strokeStyle, lineWidth, or stroke() calls for rooms
    ctx.fillStyle = fillColor;
    ctx.fill();
}

function renderRoomsLayer(ctx, rooms, layerFlags) {
    // Only draw if "Rooms (Fill)" layer is ON
    if (!layerFlags || !layerFlags.roomsFill) return;
    if (!Array.isArray(rooms)) return;

    const roomFillColor = "rgba(0, 200, 255, 0.22)"; // Light cyan fill

    for (const room of rooms) {
        if (!room || !room.polygon) continue;
        drawRoomPolygon(ctx, room.polygon, roomFillColor);
    }
}

// Set hovered room (only for highlight and tooltip)
function setHoveredRoom(canvas, room, hoverTooltip, e) {
    hoveredRoom = room;
    showRoomHoverTooltip(room, hoverTooltip, e);
    highlightRoom(canvas, room);
}

// Clear hovered room
function clearHoveredRoom(canvas, hoverTooltip) {
    hoveredRoom = null;
    hideRoomHoverTooltip(hoverTooltip);
    // Redraw overlay to remove highlight
    if (currentOverlay) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(currentOverlay, 0, 0);
    }
}

// Show tooltip next to mouse cursor (simple fixed positioning, no complex math)
function showRoomHoverTooltip(room, tooltip, e) {
    if (!tooltip || !e || !room) return;
    
    // Build tooltip content - prioritize square feet over px²
    let html = `<strong>Room ${room.id + 1}</strong>`;
    const areaText = formatRoomArea(room);
    if (areaText !== "—") {
        html += `<br>${areaText}`;
    }
    
    tooltip.innerHTML = html;
    
    // Simple fixed positioning next to cursor (no transforms, no complex calculations)
    const offsetX = 15; // Offset to the right of cursor
    const offsetY = 15; // Offset below cursor
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    // Get tooltip dimensions (render off-screen first to measure)
    safeStyle(tooltip, s => {
        s.display = 'block';
        s.visibility = 'hidden';
        s.left = '-9999px';
        s.top = '-9999px';
    });
    
    const tooltipRect = tooltip.getBoundingClientRect();
    let left = e.clientX + offsetX;
    let top = e.clientY + offsetY;
    
    // Adjust if tooltip would go off screen (simple boundary checks)
    if (left + tooltipRect.width > viewportWidth - 10) {
        left = e.clientX - tooltipRect.width - offsetX; // Show to the left
    }
    if (top + tooltipRect.height > viewportHeight - 10) {
        top = e.clientY - tooltipRect.height - offsetY; // Show above
    }
    if (left < 10) left = 10;
    if (top < 10) top = 10;
    
    // Position tooltip using fixed positioning (no transforms, no layout shifts)
    safeStyle(tooltip, s => {
        s.left = left + 'px';
        s.top = top + 'px';
        s.visibility = 'visible';
    });
}

// Hide tooltip
function hideRoomHoverTooltip(tooltip) {
    if (!tooltip) return;
    safeStyle(tooltip, s => { s.display = 'none'; });
}

// Set selected room (updates right-side blue info card)
function setSelectedRoom(room, roomInfoCard, roomInfoContent) {
    selectedRoom = room;
    
    if (!roomInfoCard || !roomInfoContent) return;
    
    if (room) {
        // Build room info content - prioritize square feet over px²
        let html = `<div class="room-info-item">
            <span class="room-info-label">Room ID:</span>
            <span class="room-info-value">${room.id + 1}</span>
        </div>`;
        
        // Show area using formatRoomArea (prefers ft² if available)
        const areaText = formatRoomArea(room);
        if (areaText !== "—") {
            html += `<div class="room-info-item">
                <span class="room-info-label">Area:</span>
                <span class="room-info-value">${areaText}</span>
            </div>`;
        }
        
        if (room.centroid && room.centroid.length >= 2) {
            html += `<div class="room-info-item">
                <span class="room-info-label">Centroid:</span>
                <span class="room-info-value">(${room.centroid[0].toFixed(1)}, ${room.centroid[1].toFixed(1)})</span>
            </div>`;
        }
        
        roomInfoContent.innerHTML = html;
        safeStyle(roomInfoCard, s => { s.display = 'block'; });
    } else {
        // Hide card when no room selected
        safeStyle(roomInfoCard, s => { s.display = 'none'; });
        roomInfoContent.innerHTML = '';
    }
}

function isPointInPolygon(point, polygon) {
    if (!polygon || polygon.length < 3) return false;
    
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i][0], yi = polygon[i][1];
        const xj = polygon[j][0], yj = polygon[j][1];
        
        const intersect = ((yi > point[1]) !== (yj > point[1])) &&
            (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

// Overlay label functions removed - replaced with simple cursor-following tooltip

function highlightRoom(canvas, room) {
    // Room highlighting - uses fill-only approach (no outlines)
    // If highlighting is needed in the future, use drawRoomPolygon with a highlight color
    // Example (commented out for now):
    // if (room && room.polygon && canvas) {
    //     const ctx = canvas.getContext('2d');
    //     drawRoomPolygon(ctx, room.polygon, "rgba(255, 255, 0, 0.3)"); // Yellow highlight, fill only
    // }
    console.log('Highlighting room:', room.id);
}

// Auto-tune visual feedback
function highlightAutoTunedSettings(settingsUsed) {
    // Highlight sliders that were adjusted by auto-tune
    const slidersToCheck = ['imgsz', 'wallConf', 'doorConf', 'windConf', 'roomConf'];
    
    slidersToCheck.forEach(sliderId => {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(sliderId + 'Value');
        const settingKey = sliderId === 'imgsz' ? 'imgsz' : 
                          sliderId === 'wallConf' ? 'wall_conf' :
                          sliderId === 'doorConf' ? 'door_conf' :
                          sliderId === 'windConf' ? 'wind_conf' : 'room_conf';
        
        if (slider && settingsUsed[settingKey] !== undefined) {
            const currentValue = parseFloat(slider.value);
            const tunedValue = parseFloat(settingsUsed[settingKey]);
            
            // If values differ, update slider and highlight it
            if (Math.abs(currentValue - tunedValue) > 0.01) {
                slider.value = tunedValue;
                if (valueDisplay) {
                    valueDisplay.textContent = sliderId === 'imgsz' ? tunedValue : tunedValue.toFixed(2);
                }
                slider.classList.add('auto-tuned');
                setTimeout(() => {
                    slider.classList.remove('auto-tuned');
                }, 2000); // Extended highlight duration
            }
        }
    });
}

// Snapshot history
function addToSnapshotHistory(result, processingTime) {
    // Keep only last 3 snapshots
    if (snapshotHistory.length >= 3) {
        snapshotHistory.shift();
    }
    
    const snapshot = {
        requestId: result.request_id,
        timestamp: Date.now(),
        settings: {
            imgsz: result.settings_used?.imgsz || document.getElementById('imgsz').value,
            wall: result.settings_used?.wall_conf || document.getElementById('wallConf').value,
            door: result.settings_used?.door_conf || document.getElementById('doorConf').value,
            window: result.settings_used?.wind_conf || document.getElementById('windConf').value,
            room: result.settings_used?.room_conf || document.getElementById('roomConf').value
        },
        processingTime: processingTime,
        overlayUrl: `${API_BASE}/download/${result.request_id}/overlay.png`
    };
    
    snapshotHistory.push(snapshot);
    updateSnapshotHistoryUI();
}

function updateSnapshotHistoryUI() {
    const historySection = document.getElementById('snapshotHistory');
    const historyList = document.getElementById('snapshotList');
    
    if (!historySection || !historyList) return;
    
    if (snapshotHistory.length === 0) {
        safeStyle(historySection, s => { s.display = 'none'; });
        return;
    }
    
    safeStyle(historySection, s => { s.display = 'block'; });
    historyList.innerHTML = '';
    
    snapshotHistory.forEach((snapshot, index) => {
        const item = document.createElement('div');
        item.className = 'snapshot-item';
        item.innerHTML = `
            <img src="${snapshot.overlayUrl}" alt="Snapshot ${index + 1}" class="snapshot-thumb" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIwIiBoZWlnaHQ9IjgwIiB2aWV3Qm94PSIwIDAgMTIwIDgwIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxyZWN0IHdpZHRoPSIxMjAiIGhlaWdodD0iODAiIGZpbGw9IiMyRDJEMkQiLz48L3N2Zz4='">
            <div class="snapshot-info">
                imgsz=${snapshot.settings.imgsz}<br>
                wall=${parseFloat(snapshot.settings.wall).toFixed(2)}<br>
                room=${parseFloat(snapshot.settings.room).toFixed(2)}<br>
                ${snapshot.processingTime}s
            </div>
        `;
        
        item.addEventListener('click', () => {
            loadSnapshot(snapshot);
        });
        
        historyList.appendChild(item);
    });
}

function loadSnapshot(snapshot) {
    // Load the snapshot's overlay image
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
        // Set canvas internal dimensions to match image
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        currentOverlay = img;
        
        // Set canvas display size to maintain aspect ratio
        const container = document.getElementById('imageContainer');
        if (container) {
            const containerRect = container.getBoundingClientRect();
            const containerWidth = containerRect.width;
            const containerHeight = containerRect.height;
            
            // Calculate scale to fit container while maintaining aspect ratio
            const scaleX = containerWidth / img.width;
            const scaleY = containerHeight / img.height;
            const scale = Math.min(scaleX, scaleY, 1); // Don't scale up beyond 100%
            
            const displayWidth = img.width * scale;
            const displayHeight = img.height * scale;
            
            safeStyle(canvas, s => {
                s.width = displayWidth + 'px';
                s.height = displayHeight + 'px';
            });
        }
        
        if (resetZoomPan) {
            resetZoomPan();
        }
    };
    img.src = snapshot.overlayUrl;
}

// Scale display
function updateScaleDisplay(result) {
    const scaleDisplay = document.getElementById('scaleDisplay');
    if (!scaleDisplay) return;
    
    const pixelsPerFoot = result.meta?.pixels_per_foot || 
                         (result.settings_used?.explicit_scale ? parseFloat(result.settings_used.explicit_scale) : null);
    
    if (pixelsPerFoot && pixelsPerFoot > 0) {
        // Calculate scale ratio (e.g., 1/8" = 1'-0")
        const scaleRatio = 12 / pixelsPerFoot; // Approximate
        scaleDisplay.textContent = `Scale: ${pixelsPerFoot.toFixed(2)} px/ft`;
        safeStyle(scaleDisplay, s => { s.display = 'block'; });
    } else {
        safeStyle(scaleDisplay, s => { s.display = 'none'; });
    }
}

// Error banner
function initializeErrorBanner() {
    const errorToggle = document.getElementById('errorToggle');
    const errorDetails = document.getElementById('errorDetails');
    
    if (errorToggle && errorDetails) {
        errorToggle.addEventListener('click', () => {
            // Check current display state safely using getComputedStyle
            const computedStyle = window.getComputedStyle(errorDetails);
            const currentDisplay = computedStyle ? computedStyle.display : 'none';
            const isExpanded = currentDisplay !== 'none';
            safeStyle(errorDetails, s => {
                s.display = isExpanded ? 'none' : 'block';
            });
            errorToggle.textContent = isExpanded ? 'Show Details' : 'Hide Details';
        });
    }
}

function showError(message, details = null) {
    const errorBanner = document.getElementById('errorBanner');
    const errorMessage = document.getElementById('errorMessage');
    const errorDetails = document.getElementById('errorDetails');
    
    if (errorBanner && errorMessage) {
        errorMessage.textContent = message || 'Processing failed. Please try again or check the file format.';
        if (errorDetails) {
            if (details) {
                errorDetails.textContent = details;
            }
            safeStyle(errorDetails, s => { s.display = 'none'; });
        }
        safeStyle(errorBanner, s => { s.display = 'block'; });
    }
}

function hideError() {
    const errorBanner = document.getElementById('errorBanner');
    safeStyle(errorBanner, s => { s.display = 'none'; });
}

// Export buttons
function initializeExportButtons() {
    const exportDxf = document.getElementById('exportDxf');
    const exportDxfSidebar = document.getElementById('exportDxfSidebar');
    
    const dxfHandler = () => {
        if (currentRequestId) {
            // TODO: Implement DXF export when backend endpoint is available
            // For now, show placeholder
            alert('DXF export coming soon. This will export layers (walls, rooms, doors, windows) as DXF polylines.');
        } else {
            alert('Please process a floor plan first.');
        }
    };
    
    if (exportDxf) {
        exportDxf.addEventListener('click', dxfHandler);
    }
    if (exportDxfSidebar) {
        exportDxfSidebar.addEventListener('click', dxfHandler);
    }
}

// Grid overlay toggle
let gridVisible = false;
function initializeGridToggle() {
    const toggleGridBtn = document.getElementById('toggleGrid');
    if (toggleGridBtn) {
        toggleGridBtn.addEventListener('click', () => {
            gridVisible = !gridVisible;
            const gridCanvas = document.getElementById('gridCanvas');
            const overlayCanvas = document.getElementById('overlayCanvas');
            
            if (gridCanvas && overlayCanvas) {
                if (gridVisible) {
                    safeStyle(gridCanvas, s => { s.display = 'block'; });
                    const rect = overlayCanvas.getBoundingClientRect();
                    const internalWidth = overlayCanvas.width;
                    const internalHeight = overlayCanvas.height;
                    drawGrid(gridCanvas, rect.width, rect.height, internalWidth, internalHeight);
                    safeStyle(toggleGridBtn, s => {
                        s.background = 'rgba(40, 107, 255, 0.2)';
                        s.borderColor = 'var(--accent)';
                    });
                } else {
                    safeStyle(gridCanvas, s => { s.display = 'none'; });
                    safeStyle(toggleGridBtn, s => {
                        s.background = '';
                        s.borderColor = '';
                    });
                }
            }
        });
    }
}

function drawGrid(canvas, displayWidth, displayHeight, internalWidth, internalHeight) {
    const ctx = canvas.getContext('2d');
    const overlayCanvas = document.getElementById('overlayCanvas');
    if (!overlayCanvas) return;
    
    // Set internal dimensions to match overlay canvas internal dimensions
    canvas.width = internalWidth || displayWidth;
    canvas.height = internalHeight || displayHeight;
    
    // Position grid canvas to match overlay canvas
    const overlayRect = overlayCanvas.getBoundingClientRect();
    const container = document.getElementById('imageContainer');
    if (container) {
        const containerRect = container.getBoundingClientRect();
        safeStyle(canvas, s => {
            s.position = 'absolute';
            s.top = (overlayRect.top - containerRect.top) + 'px';
            s.left = (overlayRect.left - containerRect.left) + 'px';
            s.width = displayWidth + 'px';
            s.height = displayHeight + 'px';
        });
    }
    
    // Calculate scale factor for grid spacing
    const scaleX = (internalWidth || displayWidth) / displayWidth;
    const scaleY = (internalHeight || displayHeight) / displayHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 0.5;
    
    // Draw grid lines (50px spacing in internal coordinates)
    const spacing = 50;
    
    // Vertical lines
    for (let x = 0; x <= canvas.width; x += spacing) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    
    // Horizontal lines
    for (let y = 0; y <= canvas.height; y += spacing) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
}

// Initialize line style controls
function initializeLineStyleControls() {
    // Line thickness slider
    const lineThicknessSlider = document.getElementById('lineThickness');
    const lineThicknessValue = document.getElementById('lineThicknessValue');
    
    if (lineThicknessSlider) {
        // Update display value
        if (lineThicknessValue) {
            lineThicknessValue.textContent = `${lineThickness} px`;
        }
        
        // Update on change
        lineThicknessSlider.addEventListener('input', (e) => {
            lineThickness = Number(e.target.value);
            if (lineThicknessValue) {
                lineThicknessValue.textContent = `${lineThickness} px`;
            }
        });
    }
    
    // Line type buttons
    const lineTypeButtons = document.querySelectorAll('.line-type-btn');
    lineTypeButtons.forEach(btn => {
        // Set initial active state
        if (btn.dataset.style === lineStyle) {
            btn.classList.add('active');
        }
        
        btn.addEventListener('click', () => {
            // Remove active from all buttons
            lineTypeButtons.forEach(b => b.classList.remove('active'));
            // Add active to clicked button
            btn.classList.add('active');
            // Update line style
            lineStyle = btn.dataset.style;
        });
    });
}

function initializeSmartDefaults() {
    const smartDefaultsBtn = document.getElementById('smartDefaultsBtn');
    if (smartDefaultsBtn) {
        smartDefaultsBtn.addEventListener('click', () => {
            applySmartDefaults();
        });
    }
}

// Auto-tune NMS thresholds based on detection density from latest result
function autoTuneNmsFromResult() {
    if (!currentResults) {
        // No data yet – fall back to safe static defaults.
        setNmsValue('wallNmsIou', 0.04);
        setNmsValue('doorWindowNmsIou', 0.05);
        return;
    }

    // Extract detection counts from result
    // Handle different possible result structures
    let wallCount = 0;
    let doorCount = 0;
    let windowCount = 0;

    if (currentResults.detections) {
        if (Array.isArray(currentResults.detections.wall)) {
            wallCount = currentResults.detections.wall.length;
        }
        if (Array.isArray(currentResults.detections.door)) {
            doorCount = currentResults.detections.door.length;
        }
        if (Array.isArray(currentResults.detections.window)) {
            windowCount = currentResults.detections.window.length;
        }
    } else if (currentResults.walls || currentResults.doors || currentResults.windows) {
        // Alternative structure
        wallCount = Array.isArray(currentResults.walls) ? currentResults.walls.length : 0;
        doorCount = Array.isArray(currentResults.doors) ? currentResults.doors.length : 0;
        windowCount = Array.isArray(currentResults.windows) ? currentResults.windows.length : 0;
    }

    const doorWinCount = doorCount + windowCount;

    // --- Heuristic for walls ---
    // If there are tons of overlapping wall boxes, use a LOWER NMS IoU to merge more.
    // If there are very few, we can relax it a bit.
    let newWallNms;
    if (wallCount > 500) {
        newWallNms = 0.03;     // very aggressive merge
    } else if (wallCount > 250) {
        newWallNms = 0.04;     // default for busy drawings
    } else {
        newWallNms = 0.06;     // clean drawings – keep more segments
    }

    // --- Heuristic for doors/windows ---
    // Many detections -> stricter NMS to kill duplicates.
    // Fewer detections -> more forgiving so we don't lose objects.
    let newDoorWinNms;
    if (doorWinCount > 200) {
        newDoorWinNms = 0.04;
    } else if (doorWinCount > 80) {
        newDoorWinNms = 0.05;
    } else {
        newDoorWinNms = 0.08;
    }

    setNmsValue('wallNmsIou', Number(newWallNms.toFixed(2)));
    setNmsValue('doorWindowNmsIou', Number(newDoorWinNms.toFixed(2)));
}

// Helper function to set NMS value and update display
function setNmsValue(id, value) {
    const slider = document.getElementById(id);
    const display = document.getElementById(id + 'Value');
    if (slider) {
        slider.value = value;
        // Trigger input event to update display
        slider.dispatchEvent(new Event('input'));
    }
    if (display) {
        display.textContent = value.toFixed(2);
    }
    // Update hint if it exists
    updateNmsHints();
}

// Update NMS hints to show current values
function updateNmsHints() {
    const wallNmsSlider = document.getElementById('wallNmsIou');
    const doorWinNmsSlider = document.getElementById('doorWindowNmsIou');
    const wallHint = document.getElementById('wallNmsHint');
    const doorWinHint = document.getElementById('doorWinNmsHint');

    if (wallNmsSlider && wallHint) {
        const value = parseFloat(wallNmsSlider.value);
        wallHint.textContent = `Auto-tuned from detection density: ~${value.toFixed(2)}`;
    }

    if (doorWinNmsSlider && doorWinHint) {
        const value = parseFloat(doorWinNmsSlider.value);
        doorWinHint.textContent = `Auto-tuned for doors+windows: ~${value.toFixed(2)}`;
    }
}

function applySmartDefaults() {
    // Image Size
    const imgsz = document.getElementById('imgsz');
    const imgszValue = document.getElementById('imgszValue');
    if (imgsz) {
        imgsz.value = 1536;
        if (imgszValue) imgszValue.textContent = '1536';
    }
    
    // Confidence Thresholds
    const wallConf = document.getElementById('wallConf');
    const wallConfValue = document.getElementById('wallConfValue');
    if (wallConf) {
        wallConf.value = 0.60;
        if (wallConfValue) wallConfValue.textContent = '0.60';
    }
    
    const doorConf = document.getElementById('doorConf');
    const doorConfValue = document.getElementById('doorConfValue');
    if (doorConf) {
        doorConf.value = 0.50;
        if (doorConfValue) doorConfValue.textContent = '0.50';
    }
    
    const windConf = document.getElementById('windConf');
    const windConfValue = document.getElementById('windConfValue');
    if (windConf) {
        windConf.value = 0.80;
        if (windConfValue) windConfValue.textContent = '0.80';
    }
    
    const roomConf = document.getElementById('roomConf');
    const roomConfValue = document.getElementById('roomConfValue');
    if (roomConf) {
        roomConf.value = 0.40;
        if (roomConfValue) roomConfValue.textContent = '0.40';
    }
    
    // PDF DPI
    const pdfDpi = document.getElementById('pdfDpi');
    if (pdfDpi) {
        pdfDpi.value = 300;
    }
    
    // Advanced settings defaults
    const wallThickPx = document.getElementById('wallThickPx');
    if (wallThickPx) wallThickPx.value = 10;
    
    const doorBridgePx = document.getElementById('doorBridgePx');
    if (doorBridgePx) doorBridgePx.value = 12;
    
    const minRoomAreaPx = document.getElementById('minRoomAreaPx');
    if (minRoomAreaPx) minRoomAreaPx.value = 6000;
    
    const footprintGrowPx = document.getElementById('footprintGrowPx');
    if (footprintGrowPx) footprintGrowPx.value = 3;
    
    // NEW: auto-tune NMS from the latest result
    autoTuneNmsFromResult();
    
    // Enable Auto-Tune
    // TEMP: DISABLED - removed .checked write
    // const autoTune = document.getElementById('autoTune');
    // if (autoTune && autoTune.checked !== undefined) {
    //     autoTune.checked = true;
    // }
    
    // Visual feedback
    const smartDefaultsBtn = document.getElementById('smartDefaultsBtn');
    if (smartDefaultsBtn) {
        safeStyle(smartDefaultsBtn, s => { s.transform = 'scale(0.95)'; });
        setTimeout(() => {
            safeStyle(smartDefaultsBtn, s => { s.transform = 'scale(1)'; });
        }, 150);
    }
    
    // Show brief confirmation
    const originalText = smartDefaultsBtn ? smartDefaultsBtn.textContent : '';
    if (smartDefaultsBtn) {
        smartDefaultsBtn.textContent = '✓ Applied';
        setTimeout(() => {
            if (smartDefaultsBtn) smartDefaultsBtn.textContent = originalText;
        }, 1500);
    }
}


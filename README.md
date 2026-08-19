# VisionAssist AI 👁️🔊

> **"From Object Detection to Intent-Aware Assistance."**  
> *A voice-first indoor navigation companion for visually impaired individuals, built for the Smart India Hackathon (SIH) Selection Round.*

---

## 🌟 The Core Innovation

Existing computer vision systems only answer: **"What is visible?"**  
*(e.g., "Chair detected. Bottle detected.")*

**VisionAssist AI** answers: **"What does the user want to accomplish?"**

```
Perception ➔ Intent Extraction ➔ Context Reasoning ➔ Actionable Audio Guidance ➔ Object Memory
```

| Typical Detection System | VisionAssist AI Guidance |
| :--- | :--- |
| `"Bottle detected."` | *"Water bottle detected ahead, slightly to your right."* |
| `"Chair detected."` | *"Obstacle ahead. Move slightly left."* |
| `"Text detected."` | *"Paracetamol 500 mg. Expiry date: December 2027."* |
| *No memory of past objects* | *"Your keys were last seen on the table beside your laptop."* |

---

## 👥 6-Person Team Architecture

```
visionassist-ai/
├── backend/
│   ├── main.py              # Person 5: FastAPI Backend & Unified Intent Router
│   ├── modules/
│   │   ├── object_finder.py # Person 1: Goal-Based Object Finding (YOLOv8 + Spatial Math)
│   │   ├── hazard.py        # Person 2: Context-Aware Hazard Detection (Relevance Filter)
│   │   ├── ocr_reader.py    # Person 3: Smart OCR Reader (EasyOCR + Pattern Extraction)
│   │   └── memory.py        # Person 4: Contextual Object Memory (SQLite Spatial Co-occurrence)
│   └── requirements.txt
├── frontend/
│   └── index.html           # Person 6: Accessible Voice + Camera Web UI
├── test_data/
│   └── test_all_modules.py  # Standalone CLI test suite
└── README.md
```

---

## 🚀 Quickstart Guide (12-Hour Hackathon Setup)

### 1. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

*(Includes `fastapi`, `uvicorn`, `ultralytics` for YOLOv8n, `opencv-python-headless`, `easyocr`, `numpy`, `pillow`)*

### 2. Run the Automated Test Suite
Verify that all 4 modules and API routes work without any errors:
```bash
python test_data/test_all_modules.py
```

### 3. Start the FastAPI Backend Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be accessible at `http://localhost:8000` with interactive API docs at `http://localhost:8000/docs`.

### 4. Launch the Frontend UI
Simply open `frontend/index.html` in Google Chrome, Microsoft Edge, or Firefox.

- Click **"Start Webcam"** (or upload an image).
- Click the **Mic button** (or press **Spacebar**) to speak commands!
- Use the **Judge Quick-Demo Presets** for one-click live presentations.

---

## 🎤 4 Live Demo Flows for Hackathon Judges

### 🔍 Demo 1: Goal-Based Object Finding
- **User says**: *"Find my water bottle"*
- **Camera**: Pointed towards a table with a bottle on the right.
- **VisionAssist AI**: *"Water bottle detected ahead, slightly to your right."*

### ⚠️ Demo 2: Context-Aware Hazard Detection
- **User says**: *"Is my path clear?"*
- **Camera**: Obstacle (chair or person) in front center.
- **VisionAssist AI**: *"Chair ahead. Move slightly left."*

### 💊 Demo 3: Smart OCR Reader
- **User says**: *"Read this medicine label"*
- **Camera**: Shows medicine strip (e.g. Paracetamol 500mg).
- **VisionAssist AI**: *"This is Paracetamol 500 milligrams. Expiry date: December 2027."*

### 🧠 Demo 4: Contextual Object Memory
- **User says**: *"Where are my keys?"*
- **VisionAssist AI**: *"Your keys were last seen on the table beside your laptop around 10:30 AM."*

---

## 📡 API Contract (`POST /assist`)

### Request
```json
{
  "intent_text": "Find my bottle",
  "image_base64": "data:image/jpeg;base64,..."
}
```

### Response
```json
{
  "intent": "find_object",
  "spoken_response": "Bottle detected ahead, slightly to your right.",
  "raw_data": {
    "target": "bottle",
    "position": "right",
    "confidence": 0.88,
    "bbox": [420.0, 150.0, 580.0, 390.0]
  }
}
```

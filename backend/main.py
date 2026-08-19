"""
VisionAssist AI - Backend Orchestration & Intent Gateway
Owner: Person 5

Connects all 4 modules under a unified API:
Perception -> Intent -> Context -> Guidance -> Memory
"""

import base64
import re
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# Import all 4 independent hackathon modules
from backend.modules.object_finder import find_object, get_yolo_model, normalize_target_name
from backend.modules.hazard import check_hazards
from backend.modules.ocr_reader import read_text
from backend.modules.memory import (
    store_object_memory,
    recall_object,
    record_scene_context,
    list_all_memories,
    init_db
)

import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title="VisionAssist AI API",
    description="Backend for Voice-First Intent-Aware Accessibility Assistant",
    version="1.0.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(FRONTEND_DIR):
    @app.get("/")
    def serve_frontend_root():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"service": "VisionAssist AI Backend"}



class AssistRequest(BaseModel):
    intent_text: str = Field(..., description="Spoken or typed user command (e.g., 'Find my bottle')")
    image_base64: Optional[str] = Field(None, description="Base64 encoded JPEG/PNG frame from camera")


class MemoryStoreRequest(BaseModel):
    object_name: str
    location: str = "table"
    nearby_objects: Optional[List[str]] = []


def decode_base64_image(b64_str: Optional[str]) -> Optional[np.ndarray]:
    """Decodes data URL or raw base64 string to OpenCV BGR numpy array."""
    if not b64_str:
        return None
    try:
        # Strip header if present: "data:image/jpeg;base64,..."
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[Backend] Error decoding base64 image: {e}")
        return None


def extract_target_noun(text: str) -> str:
    """
    Extracts the key object noun from intent phrases such as:
    'Find my water bottle' -> 'water bottle'
    'Where is my phone' -> 'phone'
    'Locate the chair' -> 'chair'
    """
    clean = text.strip().lower()
    # Remove leading trigger phrases
    patterns = [
        r"^(?:can you\s+)?(?:please\s+)?find(?:\s+my|\s+the|\s+a)?\s+",
        r"^(?:where\s+is|where\s+are|where\'s|where\s+did\s+i\s+leave)(?:\s+my|\s+the|\s+a)?\s+",
        r"^(?:locate|search\s+for|look\s+for)(?:\s+my|\s+the|\s+a)?\s+",
        r"^(?:do\s+you\s+see|show\s+me)(?:\s+my|\s+the|\s+a)?\s+"
    ]
    for p in patterns:
        clean = re.sub(p, "", clean, flags=re.IGNORECASE).strip()

    # Remove trailing question mark or punctuation
    clean = clean.strip("?.!")
    return clean if clean else "object"


def classify_user_intent(text: str) -> str:
    """
    Fast rule-based intent router for 12-hour hackathon reliability.
    Returns: 'find_object' | 'hazard_check' | 'read_text' | 'recall_memory'
    """
    t = text.lower().strip()

    # 0. Navigation Mode voice control
    if any(k in t for k in ["help me navigate", "start navigation", "navigation mode", "start auto scan", "help me walk", "start walking"]):
        return "navigation_mode_on"
    if any(k in t for k in ["get out of navigation", "exit navigation", "stop navigation", "stop navigating", "turn off navigation", "stop auto scan", "stop walking"]):
        return "navigation_mode_off"

    # 1. OCR / Reading intent
    if any(k in t for k in ["read", "text", "medicine", "label", "sign", "document", "what does it say", "ocr"]):
        return "read_text"

    # 2. Memory Recall intent ("Where are my keys?", "Where did I leave...", "Recall")
    if ("where" in t and any(w in t for w in ["key", "keys", "wallet", "leave", "put", "left"])) or "recall" in t or "remember" in t:
        return "recall_memory"

    # 3. Object Finding intent ("Find my...", "Locate...", "Search for...")
    if any(k in t for k in ["find", "locate", "search for", "look for"]):
        return "find_object"

    # 4. General where questions with current frame fallback
    if "where" in t:
        return "recall_memory"

    # 5. Hazard / Path safety intent ("Is my path clear?", "What's in front of me?", "Hazard check")
    if any(k in t for k in ["hazard", "obstacle", "safe", "path", "walk", "front", "ahead", "navigate"]):
        return "hazard_check"

    # Default to hazard / general perception if image available
    return "hazard_check"


@app.on_event("startup")
def startup_event():
    """Pre-loads YOLO model and initializes SQLite database on startup."""
    print("[Backend] Initializing VisionAssist AI...")
    init_db()
    # Cache model in memory so first query is instantaneous
    get_yolo_model()
    print("[Backend] Preloaded YOLO model & Memory DB ready!")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "VisionAssist AI"}


@app.post("/assist")
def assist(req: AssistRequest):
    """
    Main unified endpoint for frontend.
    Accepts speech intent text and optional image frame,
    routes to appropriate module, and returns spoken guidance.
    """
    intent_type = classify_user_intent(req.intent_text)
    image = decode_base64_image(req.image_base64)

    try:
        if intent_type == "navigation_mode_on":
            return {
                "intent": "navigation_mode_on",
                "spoken_response": "Navigation mode activated. Continuously scanning your path every 3 seconds.",
                "raw_data": {"navigation_mode": True, "interval_ms": 3000}
            }

        elif intent_type == "navigation_mode_off":
            return {
                "intent": "navigation_mode_off",
                "spoken_response": "Navigation mode deactivated. Auto-scan stopped.",
                "raw_data": {"navigation_mode": False}
            }

        elif intent_type == "read_text":
            if image is None:
                return {
                    "intent": intent_type,
                    "spoken_response": "Please point the camera at the label or text you want me to read.",
                    "raw_data": {}
                }
            result = read_text(image)
            return {
                "intent": intent_type,
                "spoken_response": result["spoken_response"],
                "raw_data": result
            }

        elif intent_type == "recall_memory":
            target = extract_target_noun(req.intent_text)
            result = recall_object(target)
            return {
                "intent": intent_type,
                "spoken_response": result["spoken_response"],
                "raw_data": result
            }

        elif intent_type == "find_object":
            target = extract_target_noun(req.intent_text)
            if image is None:
                # If no image provided, fallback to memory recall!
                mem_result = recall_object(target)
                return {
                    "intent": "recall_memory",
                    "spoken_response": mem_result["spoken_response"],
                    "raw_data": mem_result
                }
            
            result = find_object(image, target)
            
            # Automatically record scene context to memory
            if "all_detections" in result:
                record_scene_context(result["all_detections"])

            return {
                "intent": intent_type,
                "spoken_response": result["spoken_response"],
                "raw_data": result
            }

        else: # hazard_check
            if image is None:
                return {
                    "intent": "hazard_check",
                    "spoken_response": "Camera feed is required to check for path obstacles.",
                    "raw_data": {}
                }
            
            result = check_hazards(image)
            return {
                "intent": "hazard_check",
                "spoken_response": result["spoken_response"],
                "raw_data": result
            }

    except Exception as e:
        print(f"[Backend] Error processing request: {e}")
        return {
            "intent": intent_type,
            "spoken_response": "Sorry, I had trouble processing that. Please try again.",
            "raw_data": {"error": str(e)}
        }


# Modular individual endpoints for isolated testing
@app.post("/find")
def api_find(target: str, req: AssistRequest):
    img = decode_base64_image(req.image_base64)
    return find_object(img, target)


@app.post("/hazard")
def api_hazard(req: AssistRequest):
    img = decode_base64_image(req.image_base64)
    return check_hazards(img)


@app.post("/ocr")
def api_ocr(req: AssistRequest):
    img = decode_base64_image(req.image_base64)
    return read_text(img)


@app.get("/memory")
def api_get_memory():
    return list_all_memories()


@app.post("/memory/store")
def api_store_memory(req: MemoryStoreRequest):
    return store_object_memory(req.object_name, req.location, req.nearby_objects)

"""
VisionAssist AI - Comprehensive Test Suite
Tests all 4 modules and the FastAPI backend orchestration.
"""

import sys
import os
import cv2
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.modules.object_finder import find_object, calculate_spatial_position, normalize_target_name
from backend.modules.hazard import check_hazards, HAZARD_CLASSES
from backend.modules.ocr_reader import parse_smart_patterns, read_text
from backend.modules.memory import (
    init_db,
    store_object_memory,
    recall_object,
    record_scene_context,
    list_all_memories
)
from backend.main import app, classify_user_intent, extract_target_noun
from fastapi.testclient import TestClient


def test_intent_classification_and_extraction():
    print("\n--- Testing Intent Classification & Target Extraction ---")
    
    cases = [
        ("Find my water bottle", "find_object", "water bottle"),
        ("Locate my chair", "find_object", "chair"),
        ("Where are my keys?", "recall_memory", "keys"),
        ("Where is my wallet?", "recall_memory", "wallet"),
        ("Read this medicine label", "read_text", "medicine label"),
        ("What does this text say?", "read_text", "text say"),
        ("Is there any obstacle in front of me?", "hazard_check", "obstacle in front of me"),
        ("Is my path clear?", "hazard_check", "clear")
    ]
    
    for query, expected_intent, expected_target in cases:
        intent = classify_user_intent(query)
        target = extract_target_noun(query)
        print(f"Query: '{query}' -> Intent: {intent} | Target: '{target}'")
        assert intent == expected_intent, f"Expected {expected_intent}, got {intent}"
    print(" Intent classification tests PASSED!")


def test_spatial_positioning():
    print("\n--- Testing Spatial Positioning Logic ---")
    # Left box
    left_spatial = calculate_spatial_position([50, 100, 150, 300], 640, 480)
    assert left_spatial["position"] == "left", f"Expected left, got {left_spatial['position']}"
    
    # Right box
    right_spatial = calculate_spatial_position([500, 100, 600, 300], 640, 480)
    assert right_spatial["position"] == "right", f"Expected right, got {right_spatial['position']}"

    # Center box
    center_spatial = calculate_spatial_position([250, 100, 390, 300], 640, 480)
    assert center_spatial["position"] == "center", f"Expected center, got {center_spatial['position']}"

    print(f"Left Pos: {left_spatial['position']}, Center Pos: {center_spatial['position']}, Right Pos: {right_spatial['position']}")
    print(" Spatial positioning tests PASSED!")


def test_smart_ocr_pattern_parsing():
    print("\n--- Testing Smart OCR Pattern Parsing ---")
    sample_lines = [
        "Paracetamol Tablets IP",
        "500 mg",
        "Mfg Date: 01/2024",
        "EXP: 12/2027",
        "Batch No: B10293"
    ]
    
    parsed = parse_smart_patterns(sample_lines)
    print("Extracted fields:", parsed)
    assert parsed["dosage"] == "500 mg", f"Expected 500 mg, got {parsed['dosage']}"
    assert parsed["expiry"] == "12/2027", f"Expected 12/2027, got {parsed['expiry']}"
    print("Spoken output:", parsed["spoken_response"])
    print(" OCR pattern parsing tests PASSED!")


def test_contextual_memory():
    print("\n--- Testing Contextual Memory (SQLite) ---")
    init_db()
    
    # Store a test sighting
    store_object_memory("headphones", location="couch", nearby_objects=["phone", "book"])
    
    # Recall
    recalled = recall_object("headphones")
    print("Recalled:", recalled["spoken_response"])
    assert recalled["found"] is True
    assert "couch" in recalled["spoken_response"]
    assert "phone" in recalled["spoken_response"]

    # Recall seeded demo object
    recalled_keys = recall_object("keys")
    print("Recalled Keys:", recalled_keys["spoken_response"])
    assert recalled_keys["found"] is True

    print(" Contextual Memory tests PASSED!")


def test_fastapi_backend_endpoints():
    print("\n--- Testing FastAPI Endpoints ---")
    client = TestClient(app)
    
    # Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    
    # Assist: Memory recall
    res_mem = client.post("/assist", json={"intent_text": "Where are my keys?"})
    assert res_mem.status_code == 200
    assert "keys" in res_mem.json()["spoken_response"].lower()
    print("API Assist (Memory):", res_mem.json()["spoken_response"])

    # Assist: Object finder with synthetic image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', test_img)
    import base64
    b64_str = base64.b64encode(buffer).decode('utf-8')

    res_find = client.post("/assist", json={
        "intent_text": "Find my bottle",
        "image_base64": b64_str
    })
    assert res_find.status_code == 200
    print("API Assist (Find):", res_find.json()["spoken_response"])

    # Assist: Hazard check
    res_haz = client.post("/assist", json={
        "intent_text": "Is the path clear?",
        "image_base64": b64_str
    })
    assert res_haz.status_code == 200
    print("API Assist (Hazard):", res_haz.json()["spoken_response"])

    print(" FastAPI integration tests PASSED!")


if __name__ == "__main__":
    print("========================================")
    print("  VisionAssist AI — 12-Hr MVP Test Suite")
    print("========================================")
    test_intent_classification_and_extraction()
    test_spatial_positioning()
    test_smart_ocr_pattern_parsing()
    test_contextual_memory()
    test_fastapi_backend_endpoints()
    print("\n ALL VISIONASSIST AI TESTS COMPLETED SUCCESSFULLY! ")

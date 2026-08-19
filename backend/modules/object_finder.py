"""
VisionAssist AI - Feature 1: Goal-Based Object Finding
Owner: Person 1

Transforms: USER GOAL -> OBJECT DETECTION -> SPATIAL UNDERSTANDING -> GUIDANCE
Instead of "Bottle detected", outputs "Bottle detected ahead, slightly to your right."
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional

# Lazy-loaded singleton model instance to avoid loading YOLO multiple times
_yolo_model = None

# Common alias / synonym mapping to COCO 80 class labels
SYNONYM_MAP = {
    "water bottle": "bottle",
    "bottle": "bottle",
    "flask": "bottle",
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "smartphone": "cell phone",
    "laptop": "laptop",
    "computer": "laptop",
    "pc": "laptop",
    "notebook": "laptop",
    "chair": "chair",
    "seat": "chair",
    "couch": "couch",
    "sofa": "couch",
    "table": "dining table",
    "desk": "dining table",
    "dining table": "dining table",
    "backpack": "backpack",
    "bag": "backpack",
    "handbag": "handbag",
    "purse": "handbag",
    "cup": "cup",
    "mug": "cup",
    "glasses": "person", # fallbacks / proximity
    "mouse": "mouse",
    "keyboard": "keyboard",
    "book": "book",
    "remote": "remote",
    "person": "person",
    "human": "person",
    "tv": "tv",
    "screen": "tv",
    "monitor": "tv",
    "clock": "clock",
    "watch": "clock",
    "scissors": "scissors"
}


def get_yolo_model():
    """Loads and caches the pretrained YOLO-World zero-shot model (with fallback to yolov8n)."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            # YOLO-World enables zero-shot open-vocabulary detection for ANY custom object
            try:
                _yolo_model = YOLO("yolov8s-worldv2.pt")
                # Set initial broad detection classes
                _yolo_model.set_classes([
                    "water bottle", "bottle", "laptop", "chair", "cell phone",
                    "keys", "wallet", "glasses", "backpack", "cup", "person",
                    "dining table", "book", "mouse", "keyboard", "pen", "medicine strip", "clock"
                ])
                print("[ObjectFinder] Loaded YOLO-World v2 Zero-Shot model successfully!")
            except Exception as world_err:
                print(f"[ObjectFinder] Falling back to yolov8n: {world_err}")
                _yolo_model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"[ObjectFinder] Warning loading YOLO: {e}")
            return None
    return _yolo_model


def normalize_target_name(target_text: str) -> str:
    """Extracts and normalizes target object name to standard class name."""
    clean = target_text.strip().lower()
    # Check if exact synonym exists
    if clean in SYNONYM_MAP:
        return SYNONYM_MAP[clean]
    # Check substring matches
    for key, mapped in SYNONYM_MAP.items():
        if key in clean:
            return mapped
    return clean


def calculate_spatial_position(box_xyxy, img_width: int, img_height: int) -> Dict[str, Any]:
    """
    Computes spatial direction (left, center, right) and proximity proxy.
    """
    x1, y1, x2, y2 = box_xyxy
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_width * img_height
    area_ratio = box_area / max(img_area, 1)

    # Horizontal positioning
    rel_x = x_center / img_width
    if rel_x < 0.35:
        position = "left"
        pos_phrase = "to your left"
    elif rel_x > 0.65:
        position = "right"
        pos_phrase = "to your right"
    else:
        position = "center"
        pos_phrase = "straight ahead"

    # Proximity estimate based on bounding box size
    if area_ratio > 0.20:
        proximity = "close"
        prox_phrase = "close by"
    elif area_ratio > 0.05:
        proximity = "medium"
        prox_phrase = "a few steps ahead"
    else:
        proximity = "far"
        prox_phrase = "further ahead"

    return {
        "position": position,
        "pos_phrase": pos_phrase,
        "proximity": proximity,
        "prox_phrase": prox_phrase,
        "center": [float(x_center), float(y_center)],
        "bbox": [float(x1), float(y1), float(x2), float(y2)],
        "area_ratio": float(area_ratio)
    }


# Expanded synonym groups for reliable hackathon demo matching
TARGET_CLASS_GROUPS = {
    "bottle": ["bottle", "water bottle", "cup", "flask", "vase"],
    "water bottle": ["water bottle", "bottle", "flask", "cup"],
    "flask": ["water bottle", "bottle", "flask", "cup"],
    "cup": ["cup", "mug", "coffee cup", "bottle"],
    "phone": ["cell phone", "mobile phone", "smartphone", "remote"],
    "cell phone": ["cell phone", "mobile phone", "smartphone"],
    "mobile": ["cell phone", "mobile phone", "smartphone"],
    "laptop": ["laptop", "computer", "notebook", "tv", "screen"],
    "chair": ["chair", "couch", "seat", "bench"],
    "person": ["person", "human"],
    "backpack": ["backpack", "bag", "handbag", "suitcase"],
    "keys": ["keys", "house keys", "keychain", "remote", "cell phone"],
    "wallet": ["wallet", "purse"],
    "glasses": ["glasses", "sunglasses", "spectacles"],
    "book": ["book", "notebook"],
    "mouse": ["mouse", "computer mouse"],
    "keyboard": ["keyboard"],
    "medicine": ["medicine strip", "pill box", "bottle"],
    "medicine strip": ["medicine strip", "pill box", "bottle"]
}


def find_object(image: np.ndarray, target_object: str, model=None) -> Dict[str, Any]:
    """
    Core function for Feature 1.
    Searches for target_object in the given OpenCV BGR image and returns
    actionable spatial audio guidance using YOLO-World Zero-Shot detection.
    """
    if image is None:
        return {
            "found": False,
            "spoken_response": "No image was provided to search.",
            "target": target_object,
            "detections": []
        }

    active_model = model or get_yolo_model()
    if active_model is None:
        return {
            "found": False,
            "spoken_response": "Object detection model is currently unavailable.",
            "target": target_object,
            "detections": []
        }

    clean_target = target_object.strip().lower()
    target_class = normalize_target_name(clean_target)
    
    # Get acceptable matching classes for this target
    acceptable_classes = TARGET_CLASS_GROUPS.get(clean_target, TARGET_CLASS_GROUPS.get(target_class, [clean_target, target_class]))
    acceptable_classes_set = set([c.lower() for c in acceptable_classes])

    # If active model is YOLO-World, dynamically register target classes for zero-shot detection
    if hasattr(active_model, "set_classes"):
        try:
            dynamic_classes = list(acceptable_classes) + [
                "water bottle", "bottle", "laptop", "chair", "cell phone",
                "keys", "wallet", "glasses", "backpack", "cup", "person",
                "dining table", "book", "mouse", "pen", "medicine strip"
            ]
            # Deduplicate while preserving order
            unique_classes = list(dict.fromkeys(dynamic_classes))
            active_model.set_classes(unique_classes)
        except Exception as e:
            pass

    h, w = image.shape[:2]

    # Run inference with conf=0.15 for high sensitivity
    results = active_model(image, conf=0.15, verbose=False)[0]
    
    matched_detections = []
    all_detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = active_model.names[cls_id].lower()
        conf = float(box.conf[0])
        box_coords = box.xyxy[0].tolist()

        spatial = calculate_spatial_position(box_coords, w, h)
        det_info = {
            "class": cls_name,
            "confidence": round(conf, 2),
            **spatial
        }
        all_detections.append(det_info)

        # Check match with target classes
        if cls_name in acceptable_classes_set or any(ac in cls_name for ac in acceptable_classes_set) or target_class in cls_name:
            matched_detections.append(det_info)

    if matched_detections:
        # Sort by confidence or largest area (closest target)
        best_match = max(matched_detections, key=lambda d: d["area_ratio"])
        
        display_name = target_object.capitalize()
        pos = best_match["position"]
        
        if pos == "center":
            spoken = f"{display_name} detected straight ahead of you."
        else:
            spoken = f"{display_name} detected ahead, slightly to your {pos}."

        return {
            "found": True,
            "target": target_object,
            "detected_class": best_match["class"],
            "spoken_response": spoken,
            "position": best_match["position"],
            "confidence": best_match["confidence"],
            "bbox": best_match["bbox"],
            "all_detections": all_detections
        }
    else:
        # Check if anything was detected to give context
        if all_detections:
            seen_items = list(set([d["class"] for d in all_detections[:3]]))
            seen_str = ", ".join(seen_items)
            spoken = f"I couldn't find your {target_object}. I only see {seen_str} in front of you."
        else:
            spoken = f"I couldn't find your {target_object} in the camera view."

        return {
            "found": False,
            "target": target_object,
            "spoken_response": spoken,
            "all_detections": all_detections
        }


# Standalone runner for Person 1 unit testing
if __name__ == "__main__":
    import sys
    print("[Person 1 - Object Finder Standalone Test]")
    # Create blank canvas test image with a circle if no image provided
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_img, "Test Frame", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    res = find_object(test_img, "bottle")
    print("Result:", res)

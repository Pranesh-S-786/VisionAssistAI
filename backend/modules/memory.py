"""
VisionAssist AI - Feature 4: Contextual Object Memory
Owner: Person 4

Innovation Concept:
Perception + Spatial Context + Time -> Natural Language Recall
"Your keys were last seen on the table beside your laptop."
"""

import sqlite3
import datetime
import os
from typing import Dict, Any, List, Optional

DB_FILE = os.path.join(os.path.dirname(__file__), "..", "memory.db")


def init_db():
    """Initializes the SQLite memory database schema."""
    conn = sqlite3.connect(DB_FILE)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS object_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_name TEXT NOT NULL,
                location TEXT,
                nearby_objects TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        # Seed initial demo memory so Demo 4 works reliably out-of-the-box
        cursor = conn.execute("SELECT COUNT(*) FROM object_memory")
        if cursor.fetchone()[0] == 0:
            now_iso = datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")
            conn.execute(
                "INSERT INTO object_memory (object_name, location, nearby_objects, timestamp) VALUES (?, ?, ?, ?)",
                ("keys", "table", "laptop, coffee mug", now_iso)
            )
            conn.execute(
                "INSERT INTO object_memory (object_name, location, nearby_objects, timestamp) VALUES (?, ?, ?, ?)",
                ("wallet", "desk", "monitor, keyboard", now_iso)
            )
            conn.execute(
                "INSERT INTO object_memory (object_name, location, nearby_objects, timestamp) VALUES (?, ?, ?, ?)",
                ("water bottle", "dining table", "chair, backpack", now_iso)
            )
    conn.commit()
    return conn


def store_object_memory(
    object_name: str,
    location: str = "desk",
    nearby_objects: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Stores an object's location and nearby co-occurring items in SQLite.
    """
    clean_name = object_name.strip().lower()
    nearby_list = nearby_objects or []
    nearby_str = ", ".join([n.strip().lower() for n in nearby_list if n.strip().lower() != clean_name])
    timestamp = datetime.datetime.now().strftime("%I:%M %p")

    conn = init_db()
    with conn:
        conn.execute(
            "INSERT INTO object_memory (object_name, location, nearby_objects, timestamp) VALUES (?, ?, ?, ?)",
            (clean_name, location, nearby_str, timestamp)
        )
    conn.close()

    return {
        "stored": True,
        "object": clean_name,
        "location": location,
        "nearby": nearby_str,
        "timestamp": timestamp
    }


def record_scene_context(detections: List[Dict[str, Any]], default_location: str = "table"):
    """
    Automatically learns and stores co-occurring objects from any camera frame.
    e.g., if ['cell phone', 'laptop', 'bottle'] are detected together,
    it records context for each object.
    """
    if not detections or len(detections) < 1:
        return

    detected_names = list(set([d.get("class", "").lower() for d in detections if d.get("class")]))
    if not detected_names:
        return

    # Check if a surface like table/desk is present
    location = default_location
    if "dining table" in detected_names:
        location = "table"
    elif "couch" in detected_names or "sofa" in detected_names:
        location = "couch"
    elif "bed" in detected_names:
        location = "bed"

    for obj in detected_names:
        neighbors = [n for n in detected_names if n != obj]
        store_object_memory(obj, location=location, nearby_objects=neighbors)


def recall_object(object_name: str) -> Dict[str, Any]:
    """
    Core function for Feature 4.
    Searches for the most recent observation of an object and returns
    a clear contextual speech answer.
    """
    clean_name = object_name.strip().lower()
    conn = init_db()
    
    # Check exact and partial matches
    row = conn.execute(
        """
        SELECT object_name, location, nearby_objects, timestamp 
        FROM object_memory 
        WHERE object_name = ? OR object_name LIKE ?
        ORDER BY id DESC LIMIT 1
        """,
        (clean_name, f"%{clean_name}%")
    ).fetchone()
    conn.close()

    if not row:
        return {
            "found": False,
            "object": object_name,
            "spoken_response": f"I haven't seen your {object_name} yet.",
            "details": None
        }

    obj_name, location, nearby_str, ts = row
    
    # Formulate natural sentence
    spoken = f"Your {obj_name} was last seen on the {location}"
    if nearby_str:
        primary_neighbor = nearby_str.split(",")[0].strip()
        spoken += f" beside your {primary_neighbor}"
    
    if ts:
        spoken += f" around {ts}."
    else:
        spoken += "."

    return {
        "found": True,
        "object": obj_name,
        "location": location,
        "nearby": nearby_str,
        "timestamp": ts,
        "spoken_response": spoken
    }


def list_all_memories() -> List[Dict[str, Any]]:
    """Returns list of all stored memories for debugging or UI inspection."""
    conn = init_db()
    rows = conn.execute(
        "SELECT id, object_name, location, nearby_objects, timestamp FROM object_memory ORDER BY id DESC LIMIT 20"
    ).fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "object": r[1],
            "location": r[2],
            "nearby": r[3],
            "timestamp": r[4]
        }
        for r in rows
    ]


# Standalone runner for Person 4 unit testing
if __name__ == "__main__":
    print("[Person 4 - Contextual Memory Standalone Test]")
    init_db()
    res = recall_object("keys")
    print("Recall Keys:", res["spoken_response"])
    print("All Memories:", list_all_memories())

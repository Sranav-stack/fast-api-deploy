import os
import ezdxf
import json
import math
import sys
import chromadb
from chromadb.config import Settings
from embedding.bgme import OpenRouterBGEEmbeddingFunction
from dotenv import load_dotenv
load_dotenv


# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB CREDENTIALS — rotate these keys, they were shared publicly
# ─────────────────────────────────────────────────────────────────────────────

CHROMA_API_KEY = "ck-318j9yobzStwgKbGBsBd5mR22KoL6rZQqv2DWJmeQc68"
CHROMA_TENANT  = "99526d4b-48cf-4b20-896b-0947aa36d4ab"
CHROMA_DB      = "ingestion"          # collection name inside your DB


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — tune layer names to match your actual DXF
# ─────────────────────────────────────────────────────────────────────────────

ROOM_LABEL_LAYERS   = {"ROOM_NAMES", "ROOMS", "TEXT", "A-AREA-IDEN", "LABELS"}
ADJACENCY_THRESHOLD = 50.0   # units — adjust to your DXF scale

AREA_KEYWORDS = {
    "drilling": "Drilling Area",
    "rig base": "Rig Base",
    "control":  "Control Room",
    "pump":     "Pump Room",
    "storage":  "Storage Area",
    "office":   "Office Area",
    "medical":  "Medical Bay",
    "generator":"Generator Room",
    "mine":     "Mine Area",
    "shaft":    "Shaft Area",
    "workshop": "Workshop",
}

DIRECTION_MAP = [
    (-22.5,   22.5,  "east"),
    ( 22.5,   67.5,  "northeast"),
    ( 67.5,  112.5,  "north"),
    (112.5,  157.5,  "northwest"),
    (157.5,  180.0,  "west"),
    (-180.0, -157.5, "west"),
    (-157.5, -112.5, "southwest"),
    (-112.5,  -67.5, "south"),
    (-67.5,   -22.5, "southeast"),
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — PARSE DXF
# ─────────────────────────────────────────────────────────────────────────────

def classify_area(label: str) -> str:
    lower = label.lower()
    for keyword, area in AREA_KEYWORDS.items():
        if keyword in lower:
            return area
    return "General Area"


def parse_dxf(filepath: str) -> list[dict]:
    """
    Parse a DXF file and return a list of room records:
        {
            "label":       "Room 304 Drilling Area",
            "area":        "Drilling Area",
            "x":           100.0,
            "y":           40.0,
            "description": ""     <- filled by enrich_descriptions()
        }
    """
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()

    raw_texts = []
    for entity in msp:
        if entity.dxftype() not in ("TEXT", "MTEXT"):
            continue
        try:
            layer = entity.dxf.layer.upper()
            if entity.dxftype() == "TEXT":
                text = entity.dxf.text.strip()
                x, y = entity.dxf.insert.x, entity.dxf.insert.y
            else:
                text = entity.plain_mtext().strip()
                x, y = entity.dxf.insert.x, entity.dxf.insert.y

            if not text:
                continue

            on_label_layer  = layer in ROOM_LABEL_LAYERS
            looks_like_room = any(kw in text.lower() for kw in
                                  ["room", "area", "bay", "lab", "office",
                                   "shaft", "pump", "store", "drill",
                                   "control", "workshop"])
            if on_label_layer or looks_like_room:
                raw_texts.append({"text": text, "x": x, "y": y})

        except Exception:
            continue

    # De-duplicate entries < 2 units apart (repeated annotations)
    seen, unique = [], []
    for t in raw_texts:
        if not any(math.dist((t["x"], t["y"]), (s["x"], s["y"])) < 2.0 for s in seen):
            unique.append(t)
            seen.append(t)

    rooms = []
    for t in unique:
        label = t["text"]
        area  = classify_area(label)
        rooms.append({
            "label":       label,
            "area":        area,
            "x":           round(t["x"], 2),
            "y":           round(t["y"], 2),
            "description": "",
        })

    print(f"[parse] Extracted {len(rooms)} rooms from '{filepath}'")
    return rooms


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ENRICH DESCRIPTIONS WITH SPATIAL CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

def _cardinal(dx: float, dy: float) -> str:
    angle = math.degrees(math.atan2(dy, dx))
    for lo, hi, label in DIRECTION_MAP:
        if lo <= angle <= hi:
            return label
    return "nearby"


def enrich_descriptions(rooms: list[dict],
                        threshold: float = ADJACENCY_THRESHOLD) -> list[dict]:
    """
    Build a natural language description for each room that includes
    its zone and which rooms are adjacent + in which direction.
    This makes embeddings spatially aware at query time.
    """
    for room in rooms:
        neighbours = []
        for other in rooms:
            if other["label"] == room["label"]:
                continue
            dist = math.dist((room["x"], room["y"]), (other["x"], other["y"]))
            if dist <= threshold:
                direction = _cardinal(other["x"] - room["x"], other["y"] - room["y"])
                neighbours.append(f"{other['label']} ({direction})")

        adjacency = (
            "Adjacent rooms: " + ", ".join(neighbours) + "."
            if neighbours else
            "No rooms detected nearby."
        )
        room["description"] = (
            f"{room['label']} is located in the {room['area']} zone. {adjacency}"
        )

    print(f"[enrich] Enriched descriptions for {len(rooms)} rooms")
    return rooms


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EMBED
# ─────────────────────────────────────────────────────────────────────────────

def embed_rooms(rooms: list[dict]) -> list[dict]:
    """
    Embed each room's description using OpenRouterBGEEmbeddingFunction.
    Adds an 'embedding' key (list[float]) to each room dict.
    """
    embedding_fn = OpenRouterBGEEmbeddingFunction()
    descriptions = [r["description"] for r in rooms]

    embeddings = embedding_fn(descriptions)   # list[list[float]]

    for room, vec in zip(rooms, embeddings):
        room["embedding"] = vec

    print(f"[embed] Embedded {len(rooms)} rooms")
    return rooms


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PUSH TO CHROMADB
# ─────────────────────────────────────────────────────────────────────────────

def push_to_chroma(rooms: list[dict], collection_name: str = CHROMA_DB) -> None:
    """
    Push embedded rooms to ChromaDB cloud.

    Each room is stored with:
      - id        : stable unique ID derived from the room label
      - embedding : the BGE-M3 vector
      - document  : the enriched natural language description
      - metadata  : label, area, x, y  (filterable fields in queries)
    """
    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENET"),   # or CHROMA_TENANT if you fix typo
        database=os.getenv("CHROMA_DB"),
    )

    # Get or create collection.
    # embedding_function=None because we supply pre-computed vectors ourselves.
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},   # cosine distance suits BGE-M3
    )

    ids        = []
    embeddings = []
    documents  = []
    metadatas  = []

    for room in rooms:
        # Stable, collision-safe ID from the label
        room_id = room["label"].lower().replace(" ", "_").replace("/", "_")

        ids.append(room_id)
        embeddings.append(room["embedding"])
        documents.append(room["description"])
        metadatas.append({
            "label": room["label"],
            "area":  room["area"],
            "x":     room["x"],
            "y":     room["y"],
        })

    # Upsert so re-running the script doesn't create duplicates
    collection.upsert(
        ids        = ids,
        embeddings = embeddings,
        documents  = documents,
        metadatas  = metadatas,
    )

    print(f"[chroma] Upserted {len(ids)} rooms into collection '{collection_name}'")
    print(f"[chroma] Collection now has {collection.count()} total documents")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO DXF GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def create_demo_dxf(filepath: str = "demo_rig.dxf") -> str:
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    doc.layers.new("ROOMS", dxfattribs={"color": 2})

    room_data = [
        ("Main Room Rig Base",       0,   0),
        ("Control Room Rig Base",    0,  40),
        ("Storage Room Rig Base",   40,   0),
        ("Room 304 Drilling Area", 100,  40),
        ("Room 305 Drilling Area", 100,   0),
        ("Pump Room Drilling Area",  60,  40),
        ("Office 101",               0,  80),
        ("Medical Bay",             40,  80),
        ("Generator Room",         100,  80),
    ]
    for label, x, y in room_data:
        msp.add_text(label, dxfattribs={"layer": "ROOMS", "insert": (x, y), "height": 2.5})

    doc.saveas(filepath)
    print(f"[demo] Synthetic DXF saved to '{filepath}'")
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dxf_path = sys.argv[1] if len(sys.argv) > 1 else create_demo_dxf()

    # 1. Parse
    rooms = parse_dxf(dxf_path)

    # 2. Enrich descriptions with spatial context
    rooms = enrich_descriptions(rooms)

    # 3. Embed
    rooms = embed_rooms(rooms)

    # 4. Push to ChromaDB
    push_to_chroma(rooms)

    # Preview first room (without the full embedding vector)
    print("\n[result] Sample stored document (first room):")
    preview = {k: v for k, v in rooms[0].items() if k != "embedding"}
    preview["embedding"] = f"[{rooms[0]['embedding'][0]:.4f}, ...] (dim={len(rooms[0]['embedding'])})"
    print(json.dumps(preview, indent=2))
from pathlib import Path

# -------------------------
# Neo4j
# -------------------------

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "stockholm"


# -------------------------
# Zones
# -------------------------
"""
"Residential"
2
"Industrial"
3
"Civic"
4
"Agricultural"
5
"Commercial"
6
"Office"
7
"Utility"
8
"Green"

"""
ZONE_TYPES = [
    "residential",
    "civic",
    "industrial",
    "green",
    "offices",
    "utility",
    "commercial",
    "agricultural"
]

ZONE_TO_ID = {
    zone: i
    for i, zone in enumerate(ZONE_TYPES)
}

ID_TO_ZONE = {
    i: zone
    for zone, i in ZONE_TO_ID.items()
}

NUM_ZONES = len(ZONE_TYPES)


# -------------------------
# Model
# -------------------------

HIDDEN_DIM = 128
NUM_GNN_LAYERS = 3
DROPOUT = 0.15

EMBEDDING_DIM = HIDDEN_DIM


# -------------------------
# Training
# -------------------------

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EPOCHS = 100

ROOT_LOSS_WEIGHT = 1.0
EXPAND_LOSS_WEIGHT = 1.0
ZONE_LOSS_WEIGHT = 1.0
STOP_LOSS_WEIGHT = 0.5

DEVICE = "cuda"


# -------------------------
# Data
# -------------------------

DATA_DIR = Path("data")
MODEL_DIR = Path("checkpoints")

NORMALIZATION_FILE = DATA_DIR / "normalization.json"
MODEL_FILE = MODEL_DIR / "city_generator.pt"

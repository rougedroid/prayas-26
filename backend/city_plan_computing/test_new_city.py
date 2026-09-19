import torch
import numpy as np

from config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    ZONE_TYPES,
    ZONE_TO_ID,
    HIDDEN_DIM,
    DROPOUT,
)

from neo4j_interface.loader import Neo4jLoader

from graph.features import (
    extract_static_matrix,
    fit_normalization,
    normalize_matrix,
)

from graph.pyg_graph import (
    build_pyg_graph,
)

from models.generator import (
    CityGenerator,
)

from generation.generator import (
    CityGeneratorEngine,
)


# ============================================================
# ONLY THING YOU SHOULD CHANGE
# ============================================================

TEST_DATABASE = "amsterdam"


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/model1"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATABASE
# ============================================================

def load_cells(database):

    with Neo4jLoader(
        NEO4J_URI,
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
        database,
    ) as loader:

        cells = loader.load_cells()

    if not cells:
        raise RuntimeError(
            f"No Cell nodes found in "
            f"Neo4j database '{database}'."
        )

    return cells


def load_city():

    print("=" * 70)
    print("LOADING TEST CITY")
    print("=" * 70)

    print(f"Database : {TEST_DATABASE}")
    print(f"Device   : {DEVICE}")
    print()

    cells = load_cells(TEST_DATABASE)

    print(f"Loaded {len(cells)} cells")

    return cells


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint_file():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        return checkpoint

    # Raw state_dict (old format)
    return {
        "model_state_dict": checkpoint,
    }


def resolve_normalization(checkpoint, test_cells):
    """
    The test city must be normalized with the TRAINING
    statistics, never with its own.

    1. Checkpoint has them        -> use them.
    2. Checkpoint has none (older training runs never saved
       them)                      -> refit from the TRAINING
                                     database (NEO4J_DATABASE),
                                     which reproduces them.
    """

    print()
    print("=" * 70)
    print("NORMALIZATION")
    print("=" * 70)

    normalization = checkpoint.get(
        "normalization",
        None,
    )

    if normalization is not None:

        print("Using normalization stored in checkpoint.")

        return normalization

    print(
        "Checkpoint has no normalization statistics."
    )

    print(
        f"Refitting from TRAINING database "
        f"'{NEO4J_DATABASE}'."
    )

    if TEST_DATABASE == NEO4J_DATABASE:
        training_cells = test_cells
    else:
        training_cells = load_cells(NEO4J_DATABASE)

    return fit_normalization(
        extract_static_matrix(
            training_cells
        )
    )


# ============================================================
# BUILD FEATURES + GRAPH
# ============================================================

def prepare_city(cells, normalization):

    print()
    print("=" * 70)
    print("PREPARING TEST CITY")
    print("=" * 70)

    static_raw = extract_static_matrix(
        cells
    )

    static_normalized = normalize_matrix(
        static_raw,
        normalization,
    )

    static_features = torch.tensor(
        static_normalized,
        dtype=torch.float32,
    )

    print(
        f"Static features: "
        f"{static_features.shape[1]}"
    )

    graph = build_pyg_graph(
        cells,
        normalization,
        ZONE_TO_ID,
    )

    print(f"Nodes: {graph.num_nodes}")
    print(f"Edges: {graph.edge_index.shape[1]}")
    print(f"Graph features: {graph.x.shape[1]}")

    return (
        graph,
        static_features,
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(static_dim, checkpoint):

    print()
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    num_zones = len(ZONE_TYPES)

    # Must match gen_main.py:
    #   static + assigned(1) + zone one-hot + neighbor stats(3)
    input_dim = (
        static_dim
        + 1
        + num_zones
        + 3
    )

    print(f"Input dim : {input_dim}")
    print(f"Num zones : {num_zones}")
    print(f"Checkpoint: {MODEL_PATH}")

    model = CityGenerator(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_zones=num_zones,
        dropout=DROPOUT,
    )

    model = model.to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded checkpoint epoch: "
        f"{checkpoint.get('epoch', '?')}"
    )

    return model


# ============================================================
# GENERATE
# ============================================================

def generate_city(
    model,
    graph,
    static_features,
):

    print()
    print("=" * 70)
    print("GENERATING TEST CITY")
    print("=" * 70)

    engine = CityGeneratorEngine(
        model=model,
        graph=graph,
        static_features=static_features,
        num_zones=len(ZONE_TYPES),
        device=DEVICE,
    )

    generated_zones, generation_log = (
        engine.generate()
    )

    print()
    print("Generation complete.")

    print(
        f"Assigned cells: "
        f"{(generated_zones >= 0).sum().item()}"
        f"/{len(generated_zones)}"
    )

    sizes = []

    for entry in generation_log:

        if entry["action"] == "START_ZONE":
            sizes.append(1)

        elif (
            entry["action"] == "EXPAND"
            and sizes
        ):
            sizes[-1] += 1

    if sizes:

        print(
            f"Zones generated: {len(sizes)} | "
            f"largest: {max(sizes)} | "
            f"median size: "
            f"{int(np.median(sizes))}"
        )

    return (
        generated_zones,
        generation_log,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    cells,
    generated_zones,
):

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    generated = (
        generated_zones
        .detach()
        .cpu()
        .numpy()
    )

    actual = []

    for cell in cells:

        zone_name = cell["type"].strip()

        if zone_name not in ZONE_TO_ID:

            print(
                f"WARNING: Unknown zone type: "
                f"{cell['type']}"
            )

            actual.append(-1)

        else:

            actual.append(
                ZONE_TO_ID[zone_name]
            )

    actual = np.array(
        actual,
        dtype=int,
    )

    valid = actual >= 0

    if valid.sum() == 0:

        print(
            "No valid ground-truth "
            "zone labels."
        )

        return

    accuracy = (
        generated[valid]
        == actual[valid]
    ).mean()

    print()
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Valid cells      : {valid.sum()}")
    print()

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print("ZONE DISTRIBUTION")
    print("-" * 70)

    for zone_id, zone_name in enumerate(ZONE_TYPES):

        actual_count = (actual == zone_id).sum()
        generated_count = (generated == zone_id).sum()

        print(
            f"{zone_name:20s} "
            f"actual={actual_count:5d} "
            f"generated={generated_count:5d}"
        )

    # --------------------------------------------------------
    # Per-zone accuracy
    # --------------------------------------------------------

    print()
    print("PER-ZONE ACCURACY")
    print("-" * 70)

    for zone_id, zone_name in enumerate(ZONE_TYPES):

        mask = actual == zone_id

        count = mask.sum()

        if count > 0:

            zone_accuracy = (
                generated[mask] == zone_id
            ).mean()

            print(
                f"{zone_name:20s} "
                f"count={count:5d} "
                f"accuracy={zone_accuracy:.4f}"
            )

        else:

            print(
                f"{zone_name:20s} "
                f"count={0:5d} "
                f"accuracy=N/A"
            )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print()
    print("CONFUSION MATRIX")
    print("-" * 70)

    num_zones = len(ZONE_TYPES)

    confusion = np.zeros(
        (num_zones, num_zones),
        dtype=int,
    )

    for a, p in zip(
        actual[valid],
        generated[valid],
    ):

        if (
            0 <= a < num_zones
            and 0 <= p < num_zones
        ):

            confusion[a, p] += 1

    header = [
        name[:12]
        for name in ZONE_TYPES
    ]

    print(
        "actual \\ pred | "
        + " ".join(
            f"{name:>12}"
            for name in header
        )
    )

    for i in range(num_zones):

        print(
            f"{ZONE_TYPES[i][:12]:>12} | "
            + " ".join(
                f"{confusion[i, j]:12d}"
                for j in range(num_zones)
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    cells = load_city()

    checkpoint = load_checkpoint_file()

    normalization = resolve_normalization(
        checkpoint,
        cells,
    )

    (
        graph,
        static_features,
    ) = prepare_city(
        cells,
        normalization,
    )

    model = build_model(
        static_features.shape[1],
        checkpoint,
    )

    (
        generated_zones,
        generation_log,
    ) = generate_city(
        model=model,
        graph=graph,
        static_features=static_features,
    )

    evaluate(
        cells=cells,
        generated_zones=generated_zones,
    )


if __name__ == "__main__":
    main()
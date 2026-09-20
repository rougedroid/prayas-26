import torch
from config import (
    CITY_ID,
    DROPOUT,
    HIDDEN_DIM,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    # Keep this derived from the shared feature list as grid attributes evolve.
    # A density-observed flag distinguishes an unavailable building layer from 0% coverage.
    # The model is trained from scratch, so input width is not persisted between runs.
    ZONE_TO_ID,
    ZONE_TYPES,
)
from generation.generator import (
    CityGeneratorEngine,
)
from graph.features import (
    CONTINUOUS_FEATURES,
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
from neo4j_interface.loader import Neo4jLoader
from training.build_sequences import (
    build_city_sequence,
)
from training.dataset import (
    create_training_states,
)
from training.train import (
    train_model,
)


def main():

    print("Connecting to Neo4j...")

    with Neo4jLoader(
        NEO4J_URI,
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
        NEO4J_DATABASE,
    ) as loader:
        cells = loader.load_cells(CITY_ID)

    print(f"Loaded {len(cells)} cells for {CITY_ID}")

    # -------------------------
    # Normalization
    # -------------------------

    static_raw = extract_static_matrix(cells)

    normalization = fit_normalization(static_raw)

    static_normalized = normalize_matrix(
        static_raw,
        normalization,
    )

    static_features = torch.tensor(
        static_normalized,
        dtype=torch.float32,
    )

    # -------------------------
    # PyG graph
    # -------------------------

    graph = build_pyg_graph(
        cells,
        normalization,
        ZONE_TO_ID,
    )

    print(f"Nodes: {graph.num_nodes}")

    print(f"Edges: {graph.edge_index.shape[1]}")

    print(f"Static features: {graph.x.shape[1]}")

    # -------------------------
    # Training sequence
    # -------------------------

    zone_assignments = [ZONE_TO_ID[cell["type"]] for cell in cells]

    sequence = build_city_sequence(
        zone_assignments,
        graph.neighbor_lists,
        len(ZONE_TYPES),
    )

    print(f"Training actions: {len(sequence)}")

    training_states = create_training_states(
        sequence,
        graph.num_nodes,
    )

    print(f"Training states: {len(training_states)}")

    # -------------------------
    # Model
    # -------------------------

    # Static features:
    # Continuous features are defined with the Neo4j-to-model schema in graph.features.
    #
    # Dynamic:
    # assigned              = 1
    # zone one-hot          = num_zones
    # neighbor assigned     = 1
    # neighbor same-zone    = 1
    # neighbor unassigned   = 1

    input_dim = 6 + 1 + len(ZONE_TYPES) + 3

    model = CityGenerator(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_zones=len(ZONE_TYPES),
        dropout=DROPOUT,
    )

    print(model)

    # -------------------------
    # Train
    # -------------------------

    model = train_model(
        model=model,
        graph=graph,
        training_states=training_states,
        static_features=static_features,
        num_zones=len(ZONE_TYPES),
    )

    # -------------------------
    # Generate
    # -------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    engine = CityGeneratorEngine(
        model=model,
        graph=graph,
        static_features=static_features,
        num_zones=len(ZONE_TYPES),
        device=device,
    )

    generated_zones, log = engine.generate()

    print("Generation complete.")

    print(f"Assigned cells: {(generated_zones >= 0).sum().item()}")


if __name__ == "__main__":
    main()

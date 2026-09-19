import torch

from config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    ZONE_TYPES,
    ZONE_TO_ID,
    HIDDEN_DIM,
    DROPOUT,
    MODEL_FILE
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

from training.build_sequences import (
    build_city_sequence,
)

from training.dataset import (
    create_training_states,
)

from models.generator import (
    CityGenerator,
)

from training.train import (
    train_model,
)

from generation.generator import (
    CityGeneratorEngine,
)


def main():

    print("Connecting to Neo4j...")

    with Neo4jLoader(
        NEO4J_URI,
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
        NEO4J_DATABASE,
    ) as loader:

        cells = loader.load_cells()

    print(
        f"Loaded {len(cells)} cells"
    )

    # -------------------------
    # Normalization
    # -------------------------

    static_raw = extract_static_matrix(
        cells
    )

    normalization = fit_normalization(
        static_raw
    )

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

    print(
        f"Nodes: {graph.num_nodes}"
    )

    print(
        f"Edges: {graph.edge_index.shape[1]}"
    )

    print(
        f"Static features: "
        f"{graph.x.shape[1]}"
    )

    # -------------------------
    # Training sequence
    # -------------------------

    zone_assignments = [
        ZONE_TO_ID[cell["type"]]
        for cell in cells
    ]

    sequence = build_city_sequence(
        zone_assignments,
        graph.neighbor_lists,
        len(ZONE_TYPES),
    )

    print(
        f"Training actions: "
        f"{len(sequence)}"
    )

    training_states = (
        create_training_states(
            sequence,
            graph.num_nodes,
        )
    )

    print(
        f"Training states: "
        f"{len(training_states)}"
    )

    # -------------------------
    # Model
    # -------------------------

    # Static features:
    # 4 continuous features.
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
        checkpoint_path = "models/model1"
    )

    # -------------------------
    # Generate
    # -------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    engine = CityGeneratorEngine(
        model=model,
        graph=graph,
        static_features=static_features,
        num_zones=len(ZONE_TYPES),
        device=device,
    )

    generated_zones, log = (
        engine.generate()
    )

    print(
        "Generation complete."
    )

    print(
        f"Assigned cells: "
        f"{(generated_zones >= 0).sum().item()}"
    )


if __name__ == "__main__":
    main()

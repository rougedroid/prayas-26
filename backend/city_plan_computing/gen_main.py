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
    print_generation_metrics,
)

from generation.generator import (
    CityGeneratorEngine,
)


CHECKPOINT_PATH = "models/model2"


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

    # Input = static features (whatever the pipeline produced)
    #       + dynamic features:
    #           assigned              = 1
    #           zone one-hot          = num_zones
    #           neighbor assigned     = 1
    #           neighbor same-zone    = 1
    #           neighbor unassigned   = 1
    #
    # Taken from static_features.shape[1] so it can never
    # drift out of sync with the feature extractor.

    static_dim = static_features.shape[1]

    input_dim = static_dim + 1 + len(ZONE_TYPES) + 3

    print(
        f"Input dim: {input_dim} "
        f"(static {static_dim} + dynamic "
        f"{1 + len(ZONE_TYPES) + 3})"
    )

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

    # normalization is saved into the checkpoint so
    # test_newcity.py can reuse the TRAINING statistics.

    model = train_model(
        model=model,
        graph=graph,
        training_states=training_states,
        static_features=static_features,
        num_zones=len(ZONE_TYPES),
        checkpoint_path=CHECKPOINT_PATH,
        normalization=normalization,
    )

    # -------------------------
    # Generate
    # -------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

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

    # -------------------------
    # Real generation accuracy
    # -------------------------

    actual_zones = torch.tensor(
        zone_assignments,
        dtype=torch.long,
    )

    print_generation_metrics(
        generated_zones,
        actual_zones,
        ZONE_TYPES,
    )

    zone_sizes = []

    for entry in log:

        if entry["action"] == "START_ZONE":
            zone_sizes.append(1)

        elif (
            entry["action"] == "EXPAND"
            and zone_sizes
        ):
            zone_sizes[-1] += 1

    if zone_sizes:

        print(
            f"Zones generated: {len(zone_sizes)} | "
            f"largest: {max(zone_sizes)} | "
            f"mean size: "
            f"{sum(zone_sizes) / len(zone_sizes):.1f}"
        )


if __name__ == "__main__":
    main()
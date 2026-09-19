import json
import numpy as np

CONTINUOUS_FEATURES = [
    "population",
    "green_cover",
    "elevation",
    "distance_to_boundary",
    "x",
    "y",
]


def build_zone_mapping(zone_types):
    return {
        zone: i
        for i, zone in enumerate(zone_types)
    }


def encode_zone(zone, zone_to_id):
    result = np.zeros(
        len(zone_to_id),
        dtype=np.float32,
    )

    if zone in zone_to_id:
        result[zone_to_id[zone]] = 1.0

    return result


def extract_static_matrix(cells):
    matrix = np.array([
        [
            float(cell["population"] or 0.0),
            float(cell["green_cover"] or 0.0),
            float(cell["elevation"] or 0.0),
            float(cell["distance_to_boundary"] or 0.0),
            float(cell["x"] or 0.0),
            float(cell["y"] or 0.0),
        ]
        for cell in cells
    ], dtype=np.float32)

    return matrix

def fit_normalization(matrix):
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)

    # Prevent division by zero for constant features.
    std[std < 1e-8] = 1.0

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def normalize_matrix(matrix, statistics):
    mean = np.asarray(
        statistics["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        statistics["std"],
        dtype=np.float32,
    )

    return (matrix - mean) / std


def save_normalization(statistics, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w") as f:
        json.dump(statistics, f, indent=2)


def load_normalization(path):
    with open(path, "r") as f:
        return json.load(f)


def build_dynamic_features(
    assigned_zones,
    neighbor_lists,
    num_zones,
):
    """
    assigned_zones[i]:

        -1  = unassigned
        0+  = zone ID

    Returns:

        [assigned,
         zone_one_hot...,
         neighbor_assigned_fraction,
         neighbor_same_zone_fraction,
         neighbor_unassigned_fraction]
    """

    num_nodes = len(assigned_zones)

    result = np.zeros(
        (
            num_nodes,
            1 + num_zones + 3,
        ),
        dtype=np.float32,
    )

    for i in range(num_nodes):

        zone = assigned_zones[i]

        # Assigned flag.
        if zone >= 0:
            result[i, 0] = 1.0

            result[
                i,
                1 + zone,
            ] = 1.0

        neighbors = neighbor_lists[i]

        if not neighbors:
            continue

        assigned_count = 0
        same_zone_count = 0
        unassigned_count = 0

        for neighbor in neighbors:

            neighbor_zone = assigned_zones[neighbor]

            if neighbor_zone == -1:
                unassigned_count += 1

            else:
                assigned_count += 1

                if zone >= 0 and neighbor_zone == zone:
                    same_zone_count += 1

        degree = len(neighbors)

        result[
            i,
            1 + num_zones,
        ] = assigned_count / degree

        result[
            i,
            2 + num_zones,
        ] = same_zone_count / degree

        result[
            i,
            3 + num_zones,
        ] = unassigned_count / degree

    return result

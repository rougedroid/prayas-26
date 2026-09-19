"""
Masked-zone GNN: learn what zone a cell SHOULD have given its
surroundings, then use it to audit existing cities.

Idea
----
Hide the zone label of some cells, show the GNN everything else
(static features + neighbours' zones), and train it to predict the
hidden zones. A model trained on well-planned cities learns "what
usually goes where". At audit time every cell is hidden in turn; where
the model confidently disagrees with the actual zone, the cell is
flagged and the model's zone is the suggestion.
"""

import csv
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

from config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    ZONE_TYPES,
    ZONE_TO_ID,
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


NUM_ZONES = len(ZONE_TYPES)


# ============================================================
# MODEL
# ============================================================

class CellZoner(nn.Module):
    """
    3-layer GraphSAGE encoder + classification head.
    Output: logits over zone types for every node.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_zones,
        static_dim,
        dropout=0.3,
    ):
        super().__init__()

        self.num_zones = num_zones

        # Where the neighbour-zone histogram sits inside the input
        # (see build_input): static | onehot | visible | HIST | ...
        self.hist_lo = static_dim + num_zones + 1
        self.hist_hi = self.hist_lo + num_zones

        self.gnn1 = SAGEConv(input_dim, hidden_dim)
        self.gnn2 = SAGEConv(hidden_dim, hidden_dim)
        self.gnn3 = SAGEConv(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_zones),
        )

        # The GNN branch starts as a no-op ...
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        # ... and this skip path starts as "predict the most common
        # neighbour zone" (a strong baseline). Training then only has
        # to learn where the surroundings are NOT the whole story.
        self.skip = nn.Linear(num_zones, num_zones)

        with torch.no_grad():
            self.skip.weight.copy_(
                4.0 * torch.eye(num_zones)
            )
            self.skip.bias.zero_()

    def forward(self, x, edge_index):

        hist = x[:, self.hist_lo:self.hist_hi]

        h = self.dropout(
            F.relu(self.norm1(self.gnn1(x, edge_index)))
        )

        h = self.dropout(
            F.relu(self.norm2(self.gnn2(h, edge_index)))
        )

        h = F.relu(
            self.norm3(self.gnn3(h, edge_index))
        )

        return self.head(h) + self.skip(hist)


def model_input_dim(static_dim):
    # static + own one-hot(Z) + visible flag(1)
    # + neighbour histogram(Z) + unknown-neighbour fraction(1)
    # + degree(1)
    return static_dim + 2 * NUM_ZONES + 3


# ============================================================
# DATA
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
            f"No Cell nodes found in database '{database}'."
        )

    return cells


def cell_labels(cells):
    """
    Zone id per cell; -1 for unknown zone types (these are never
    used as targets or as context).
    """

    labels = []
    unknown = Counter()

    for cell in cells:

        name = str(cell["type"]).strip()

        if name in ZONE_TO_ID:
            labels.append(ZONE_TO_ID[name])
        else:
            labels.append(-1)
            unknown[name] += 1

    if unknown:
        print(
            f"WARNING: unknown zone types ignored: "
            f"{dict(unknown)}"
        )

    return np.array(labels, dtype=np.int64)


def fit_multi_normalization(cells_list):
    """
    One normalization fitted over ALL training cities together.
    """

    raws = [
        np.asarray(
            extract_static_matrix(cells),
            dtype=np.float32,
        )
        for cells in cells_list
    ]

    return fit_normalization(
        np.vstack(raws)
    )


def cell_identifier(cell, index):

    for key in (
        "id",
        "cell_id",
        "cellId",
        "uid",
        "uuid",
        "name",
    ):
        try:
            value = cell[key]
        except Exception:
            continue

        if value is not None:
            return value

    return index


class CityData:
    """
    One city as tensors on `device`.
    """

    def __init__(
        self,
        name,
        cells,
        normalization,
        device,
        static_mode="global",
    ):
        """
        static_mode:
          "global" - normalise with the training-set statistics
          "city"   - z-score each column within this city only
                     (transfers better between cities of different
                     scale)
          "none"   - ignore static features; use only the zones of
                     surrounding cells
        """

        self.name = name
        self.cells = cells
        self.device = device
        self.static_mode = static_mode

        static_raw = np.asarray(
            extract_static_matrix(cells),
            dtype=np.float32,
        )

        if static_mode == "global":

            static_np = np.asarray(
                normalize_matrix(
                    static_raw,
                    normalization,
                ),
                dtype=np.float32,
            )

        elif static_mode == "city":

            mean = static_raw.mean(axis=0, keepdims=True)
            std = static_raw.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0

            static_np = np.clip(
                (static_raw - mean) / std,
                -5.0,
                5.0,
            ).astype(np.float32)

        elif static_mode == "none":

            static_np = np.zeros_like(static_raw)

        else:
            raise ValueError(
                f"Unknown static_mode '{static_mode}'"
            )

        self.static = torch.tensor(
            static_np,
            dtype=torch.float32,
            device=device,
        )

        graph = build_pyg_graph(
            cells,
            normalization,
            ZONE_TO_ID,
        )

        if graph.num_nodes != self.static.shape[0]:
            raise RuntimeError(
                f"{name}: graph has {graph.num_nodes} nodes but "
                f"{self.static.shape[0]} static rows."
            )

        # Symmetrise, drop duplicates and self loops.
        ei = graph.edge_index.long()
        ei = torch.cat([ei, ei.flip(0)], dim=1)
        ei = torch.unique(ei, dim=1)
        ei = ei[:, ei[0] != ei[1]]

        self.edge_index = ei.to(device)

        self.num_nodes = self.static.shape[0]

        self.labels_cpu = torch.tensor(
            cell_labels(cells),
            dtype=torch.long,
        )

        self.labels = self.labels_cpu.to(device)

        self.known_idx = (
            self.labels >= 0
        ).nonzero(as_tuple=True)[0]

        self._neighbors = None

    def neighbor_lists(self):

        if self._neighbors is None:

            nb = [[] for _ in range(self.num_nodes)]

            src, dst = self.edge_index.cpu().tolist()

            for s, d in zip(src, dst):
                nb[d].append(s)

            self._neighbors = nb

        return self._neighbors


# ============================================================
# FEATURES (labels of hidden cells are NEVER visible)
# ============================================================

def neighbor_histogram(
    labels,
    hidden,
    edge_index,
):

    n = labels.shape[0]
    device = labels.device

    visible = (labels >= 0) & (~hidden)

    onehot = torch.zeros(
        n,
        NUM_ZONES,
        device=device,
    )

    idx = visible.nonzero(as_tuple=True)[0]
    onehot[idx, labels[idx]] = 1.0

    src, dst = edge_index

    counts = torch.zeros(
        n,
        NUM_ZONES,
        device=device,
    ).index_add_(0, dst, onehot[src])

    deg = torch.zeros(
        n,
        device=device,
    ).index_add_(
        0,
        dst,
        torch.ones(src.shape[0], device=device),
    )

    return visible, onehot, counts, deg


def build_input(
    static,
    labels,
    hidden,
    edge_index,
):
    """
    Node features with the zones of `hidden` cells removed.

    labels : (N,) long, -1 = unknown
    hidden : (N,) bool, True = label hidden from the model
    """

    visible, onehot, counts, deg = neighbor_histogram(
        labels,
        hidden,
        edge_index,
    )

    deg_c = deg.clamp(min=1.0).unsqueeze(1)

    hist = counts / deg_c

    known_frac = counts.sum(
        dim=1,
        keepdim=True,
    ) / deg_c

    unknown_frac = 1.0 - known_frac

    deg_feat = deg.unsqueeze(1) / 8.0

    return torch.cat(
        [
            static,
            onehot,
            visible.float().unsqueeze(1),
            hist,
            unknown_frac,
            deg_feat,
        ],
        dim=1,
    )


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict_probs(
    model,
    city,
    folds=8,
    rounds=3,
    seed=0,
):
    """
    For every known cell: hide it (together with ~1/folds of the
    other cells), predict its zone, and average over `rounds`
    random partitions.

    Returns (N, num_zones) CPU tensor of probabilities.
    """

    model.eval()

    gen = torch.Generator().manual_seed(seed)

    n = city.num_nodes

    known = city.known_idx.cpu()

    acc = torch.zeros(n, NUM_ZONES)
    cnt = torch.zeros(n)

    for _ in range(rounds):

        perm = known[
            torch.randperm(
                len(known),
                generator=gen,
            )
        ]

        for chunk in torch.tensor_split(perm, folds):

            if len(chunk) == 0:
                continue

            hidden = torch.zeros(
                n,
                dtype=torch.bool,
            )

            hidden[chunk] = True

            x = build_input(
                city.static,
                city.labels,
                hidden.to(city.device),
                city.edge_index,
            )

            probs = F.softmax(
                model(x, city.edge_index),
                dim=-1,
            ).cpu()

            acc[chunk] += probs[chunk]
            cnt[chunk] += 1

    return acc / cnt.clamp(min=1).unsqueeze(1)


def flag_mask(
    probs,
    labels,
    margin,
    min_conf,
):
    """
    A cell is flagged when the model's top zone differs from the
    actual zone AND is clearly more likely than it
    (top - actual >= margin) AND is itself confident (>= min_conf).
    """

    known = labels >= 0

    pred = probs.argmax(dim=1)

    p_pred = probs.max(dim=1).values

    p_act = probs.gather(
        1,
        labels.clamp(min=0).unsqueeze(1),
    ).squeeze(1)

    return (
        known
        & (pred != labels)
        & ((p_pred - p_act) >= margin)
        & (p_pred >= min_conf)
    )


def evaluate_city(
    model,
    city,
    folds=8,
    rounds=2,
    margin=0.35,
    min_conf=0.40,
):

    probs = predict_probs(
        model,
        city,
        folds=folds,
        rounds=rounds,
    )

    labels = city.labels_cpu

    known = labels >= 0

    pred = probs.argmax(dim=1)

    accuracy = (
        pred[known] == labels[known]
    ).float().mean().item()

    recalls = []

    for z in range(NUM_ZONES):

        m = known & (labels == z)

        if m.sum() > 0:
            recalls.append(
                (pred[m] == z).float().mean().item()
            )

    balanced = float(np.mean(recalls)) if recalls else 0.0

    counts = torch.bincount(
        labels[known],
        minlength=NUM_ZONES,
    )

    majority = (
        counts.max().item()
        / max(known.sum().item(), 1)
    )

    # Baseline: most common zone among the cell's neighbours.
    _, _, nb_counts, _ = neighbor_histogram(
        city.labels,
        torch.zeros_like(city.labels, dtype=torch.bool),
        city.edge_index,
    )

    nb_counts = nb_counts.cpu()

    has_nb = known & (nb_counts.sum(dim=1) > 0)

    nb_acc = (
        nb_counts.argmax(dim=1)[has_nb]
        == labels[has_nb]
    ).float().mean().item()

    flagged = flag_mask(
        probs,
        labels,
        margin,
        min_conf,
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "majority_baseline": majority,
        "neighbor_baseline": nb_acc,
        "flag_rate": (
            flagged.sum().item()
            / max(known.sum().item(), 1)
        ),
        "cells": int(known.sum().item()),
    }


# ============================================================
# AUDIT
# ============================================================

def audit_city(
    city,
    probs,
    margin=0.35,
    min_conf=0.40,
):
    """
    Returns a list of flagged cells, worst first.
    """

    labels = city.labels_cpu

    flagged = flag_mask(
        probs,
        labels,
        margin,
        min_conf,
    )

    pred = probs.argmax(dim=1)

    nbrs = city.neighbor_lists()

    findings = []

    for i in flagged.nonzero(as_tuple=True)[0].tolist():

        actual = int(labels[i])
        suggested = int(pred[i])

        p_actual = float(probs[i, actual])
        p_suggested = float(probs[i, suggested])

        around = Counter(
            int(labels[j])
            for j in nbrs[i]
            if int(labels[j]) >= 0
        )

        around_text = ", ".join(
            f"{ZONE_TYPES[z]}x{n}"
            for z, n in around.most_common()
        )

        findings.append(
            {
                "index": i,
                "cell_id": cell_identifier(
                    city.cells[i],
                    i,
                ),
                "actual_zone": ZONE_TYPES[actual],
                "suggested_zone": ZONE_TYPES[suggested],
                "suggested_confidence": round(p_suggested, 4),
                "actual_probability": round(p_actual, 4),
                "gap": round(p_suggested - p_actual, 4),
                "neighbours": around_text,
            }
        )

    findings.sort(
        key=lambda f: f["gap"],
        reverse=True,
    )

    return findings


def write_findings_csv(path, findings):

    fields = [
        "index",
        "cell_id",
        "actual_zone",
        "suggested_zone",
        "suggested_confidence",
        "actual_probability",
        "gap",
        "neighbours",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(findings)


# ============================================================
# CHECKPOINTS
# ============================================================

def save_zoner_checkpoint(
    path,
    model,
    epoch,
    normalization,
    meta,
    metrics=None,
):
    from pathlib import Path

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "normalization": normalization,
            "meta": meta,
            "metrics": metrics,
        },
        path,
    )


def load_zoner_checkpoint(path, device):

    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    meta = ckpt["meta"]

    if list(meta["zone_types"]) != list(ZONE_TYPES):
        raise RuntimeError(
            "ZONE_TYPES in config.py differ from the ones this "
            "checkpoint was trained with."
        )

    model = CellZoner(
        input_dim=meta["input_dim"],
        hidden_dim=meta["hidden_dim"],
        num_zones=meta["num_zones"],
        static_dim=meta["static_dim"],
        dropout=meta["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    return model, ckpt
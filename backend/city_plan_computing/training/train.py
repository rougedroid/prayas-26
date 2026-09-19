import torch
import torch.nn.functional as F

from tqdm import tqdm
from collections import Counter
from pathlib import Path

from graph.features import build_dynamic_features


# ============================================================
# DYNAMIC GRAPH FEATURES
# ============================================================

def build_dynamic_features_tensor(
    assigned_zones,
    neighbor_lists,
    num_zones,
    device,
):
    dynamic = build_dynamic_features(
        assigned_zones.cpu().numpy(),
        neighbor_lists,
        num_zones,
    )

    return torch.tensor(
        dynamic,
        dtype=torch.float32,
        device=device,
    )


def get_frontier(
    assigned_zones,
    zone_id,
    neighbor_lists,
):
    """
    Return all currently-unassigned cells adjacent to
    the currently active zone.
    """

    frontier = set()

    for node in range(len(assigned_zones)):

        if assigned_zones[node] != zone_id:
            continue

        for neighbor in neighbor_lists[node]:

            if assigned_zones[neighbor] == -1:
                frontier.add(neighbor)

    return sorted(frontier)


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    path,
    normalization=None,
):
    """
    Save everything required to continue training later.
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),
    }

    if normalization is not None:
        checkpoint["normalization"] = normalization

    torch.save(
        checkpoint,
        path,
    )

    print(
        f"Checkpoint saved -> {path}"
    )


def checkpoint_exists(path):
    return (
        path is not None
        and Path(path).exists()
    )


def ask_checkpoint_mode(path):
    """
    Ask the user whether to start a new model or
    continue from an existing checkpoint.

    Returns:
        "new"
        "load"
    """

    path = Path(path)

    print()
    print("=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)

    if path.exists():

        print()
        print(
            f"Existing model found:\n"
            f"  {path}"
        )

        print()
        print("Choose:")
        print("  [N] Start a NEW model")
        print("  [L] LOAD the existing model and continue training")

        while True:

            choice = input(
                "\nYour choice [N/L]: "
            ).strip().lower()

            if choice in ("n", "new"):
                return "new"

            if choice in ("l", "load"):
                return "load"

            print(
                "Please enter N or L."
            )

    else:

        print()
        print(
            "No existing checkpoint found."
        )

        print(
            "Starting a NEW model."
        )

        return "new"


def load_checkpoint(
    model,
    optimizer,
    path,
    device,
):
    """
    Load model + optimizer state.

    Returns:
        starting_epoch
        normalization
    """

    path = Path(path)

    print()
    print(
        f"Loading checkpoint:\n"
        f"  {path}"
    )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    start_epoch = checkpoint.get(
        "epoch",
        0,
    )

    normalization = checkpoint.get(
        "normalization",
        None,
    )

    print(
        f"Loaded checkpoint from epoch "
        f"{start_epoch}"
    )

    if normalization is not None:
        print(
            "Normalization statistics loaded "
            "from checkpoint."
        )
    else:
        print(
            "WARNING: checkpoint contains no "
            "normalization statistics."
        )

    return (
        start_epoch,
        normalization,
    )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    graph,
    training_states,
    static_features,
    num_zones,

    epochs=100,

    learning_rate=1e-3,
    weight_decay=1e-4,

    root_loss_weight=1.0,
    expand_loss_weight=1.0,
    zone_loss_weight=1.0,

    checkpoint_path=None,
    normalization=None,

    ask_model_choice=True,
):
    """
    Train the city generator.

    If a checkpoint exists, the user is asked whether to:

        N -> start a completely new model
        L -> load the existing model and continue training

    The checkpoint stores:

        model weights
        optimizer state
        epoch
        normalization statistics
    """

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    static_features = static_features.to(
        device
    )

    edge_index = graph.edge_index.to(
        device
    )

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    start_epoch = 0

    model_mode = "new"

    if (
        ask_model_choice
        and checkpoint_path is not None
    ):
        model_mode = ask_checkpoint_mode(
            checkpoint_path
        )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # --------------------------------------------------------
    # LOAD EXISTING MODEL
    # --------------------------------------------------------

    checkpoint_normalization = None

    if (
        model_mode == "load"
        and checkpoint_path is not None
    ):

        (
            start_epoch,
            checkpoint_normalization,
        ) = load_checkpoint(
            model=model,
            optimizer=optimizer,
            path=checkpoint_path,
            device=device,
        )

        # If the checkpoint contains normalization
        # statistics, prefer those for future saving.
        if checkpoint_normalization is not None:
            normalization = (
                checkpoint_normalization
            )

    # --------------------------------------------------------
    # TRAINING INFORMATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CITY GENERATOR TRAINING")
    print("=" * 70)

    print(
        f"Device:              {device}"
    )

    print(
        f"Model mode:          "
        f"{'CONTINUE' if model_mode == 'load' else 'NEW'}"
    )

    print(
        f"Starting epoch:      {start_epoch}"
    )

    print(
        f"Training epochs:     {epochs}"
    )

    print(
        f"Training states:     "
        f"{len(training_states)}"
    )

    print(
        f"Learning rate:       "
        f"{learning_rate}"
    )

    print(
        f"Weight decay:        "
        f"{weight_decay}"
    )

    if checkpoint_path is not None:
        print(
            f"Checkpoint:          "
            f"{checkpoint_path}"
        )

    print("=" * 70)
    print()

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    history = []

    # --------------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        start_epoch + epochs,
    ):

        model.train()

        total_loss = 0.0

        root_loss_total = 0.0
        expand_loss_total = 0.0
        zone_loss_total = 0.0

        root_count = 0
        expand_count = 0
        zone_count = 0

        progress = tqdm(
            training_states,
            desc=(
                f"Epoch "
                f"{epoch + 1:03d}/"
                f"{start_epoch + epochs:03d}"
            ),
            leave=False,
        )

        # ----------------------------------------------------
        # TRAINING STATES
        # ----------------------------------------------------

        for state in progress:

            optimizer.zero_grad(
                set_to_none=True
            )

            assigned = (
                state.assigned_zones
            )

            # -----------------------------------------------
            # BUILD CURRENT GRAPH FEATURES
            # -----------------------------------------------

            dynamic = (
                build_dynamic_features_tensor(
                    assigned_zones=assigned,
                    neighbor_lists=graph.neighbor_lists,
                    num_zones=num_zones,
                    device=device,
                )
            )

            x = torch.cat(
                [
                    static_features,
                    dynamic,
                ],
                dim=1,
            )

            # -----------------------------------------------
            # FORWARD PASS
            # -----------------------------------------------

            outputs = model(
                x,
                edge_index,
            )

            h = outputs[
                "embeddings"
            ]

            loss = None

            # -----------------------------------------------
            # START_ZONE
            # -----------------------------------------------

            if state.root_target is not None:

                unassigned = torch.where(
                    assigned == -1
                )[0].to(device)

                if len(unassigned) > 0:

                    # ---------------------------------------
                    # ROOT LOSS
                    # ---------------------------------------

                    root_logits = (
                        outputs[
                            "root_logits"
                        ][unassigned]
                    )

                    target_position = (
                        unassigned
                        == state.root_target
                    ).nonzero(
                        as_tuple=False
                    )

                    if len(target_position) > 0:

                        target_position = (
                            target_position[
                                0
                            ].item()
                        )

                        root_target = torch.tensor(
                            [target_position],
                            dtype=torch.long,
                            device=device,
                        )

                        root_loss = (
                            F.cross_entropy(
                                root_logits.unsqueeze(0),
                                root_target,
                            )
                        )

                        # -----------------------------------
                        # ZONE LOSS
                        # -----------------------------------

                        zone_logits = (
                            outputs[
                                "zone_logits"
                            ][
                                state.root_target
                            ].unsqueeze(0)
                        )

                        zone_target = torch.tensor(
                            [
                                state.zone_target
                            ],
                            dtype=torch.long,
                            device=device,
                        )

                        zone_loss = (
                            F.cross_entropy(
                                zone_logits,
                                zone_target,
                            )
                        )

                        if loss is None:
                            loss = torch.tensor(
                                0.0,
                                device=device,
                            )

                        loss = (
                            loss
                            + root_loss_weight
                            * root_loss
                            + zone_loss_weight
                            * zone_loss
                        )

                        root_loss_total += (
                            root_loss.item()
                        )

                        zone_loss_total += (
                            zone_loss.item()
                        )

                        root_count += 1
                        zone_count += 1

            # -----------------------------------------------
            # EXPANSION
            # -----------------------------------------------

            if (
                state.expand_target
                is not None
            ):

                if (
                    state.active_zone
                    is not None
                ):

                    frontier = get_frontier(
                        assigned_zones=assigned,
                        zone_id=state.active_zone,
                        neighbor_lists=graph.neighbor_lists,
                    )

                    if len(frontier) > 0:

                        frontier_tensor = (
                            torch.tensor(
                                frontier,
                                dtype=torch.long,
                                device=device,
                            )
                        )

                        expand_logits = (
                            model.expansion_scores(
                                h,
                                frontier_tensor,
                                state.active_zone,
                            )
                        )

                        target_position = (
                            frontier_tensor
                            == state.expand_target
                        ).nonzero(
                            as_tuple=False
                        )

                        if len(target_position) > 0:

                            target_position = (
                                target_position[
                                    0
                                ].item()
                            )

                            expand_target = (
                                torch.tensor(
                                    [target_position],
                                    dtype=torch.long,
                                    device=device,
                                )
                            )

                            expand_loss = (
                                F.cross_entropy(
                                    expand_logits.unsqueeze(0),
                                    expand_target,
                                )
                            )

                            if loss is None:
                                loss = torch.tensor(
                                    0.0,
                                    device=device,
                                )

                            loss = (
                                loss
                                + expand_loss_weight
                                * expand_loss
                            )

                            expand_loss_total += (
                                expand_loss.item()
                            )

                            expand_count += 1

            # -----------------------------------------------
            # NO LOSS?
            # -----------------------------------------------

            if loss is None:
                continue

            # -----------------------------------------------
            # BACKPROP
            # -----------------------------------------------

            loss.backward()

            # Prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            # -----------------------------------------------
            # STATISTICS
            # -----------------------------------------------

            loss_value = (
                loss.detach().item()
            )

            total_loss += loss_value

            progress.set_postfix(
                loss=f"{loss_value:.4f}"
            )

        # ----------------------------------------------------
        # AVERAGES
        # ----------------------------------------------------

        num_states = max(
            len(training_states),
            1,
        )

        avg_total = (
            total_loss / num_states
        )

        avg_root = (
            root_loss_total
            / max(root_count, 1)
        )

        avg_expand = (
            expand_loss_total
            / max(expand_count, 1)
        )

        avg_zone = (
            zone_loss_total
            / max(zone_count, 1)
        )

        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": avg_total,
            "root": avg_root,
            "expand": avg_expand,
            "zone": avg_zone,
        }

        history.append(
            epoch_metrics
        )

        # ----------------------------------------------------
        # PRINT EPOCH RESULTS
        # ----------------------------------------------------

        print(
            f"Epoch {epoch + 1:03d} | "
            f"loss={avg_total:.4f} | "
            f"root={avg_root:.4f} | "
            f"expand={avg_expand:.4f} | "
            f"zone={avg_zone:.4f}"
        )

        # ----------------------------------------------------
        # SAVE AFTER EVERY EPOCH
        # ----------------------------------------------------

        if checkpoint_path is not None:

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                path=checkpoint_path,
                normalization=normalization,
            )

    # ========================================================
    # FINAL TRAINING REPORT
    # ========================================================

    if len(history) > 0:

        first = history[0]
        last = history[-1]

        print()
        print("=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

        print(
            f"Initial loss:       "
            f"{first['loss']:.6f}"
        )

        print(
            f"Final loss:         "
            f"{last['loss']:.6f}"
        )

        if first["loss"] > 0:

            reduction = (
                1.0
                - last["loss"]
                / first["loss"]
            ) * 100

            print(
                f"Loss reduction:     "
                f"{reduction:.2f}%"
            )

        print()

        print(
            f"Final root loss:    "
            f"{last['root']:.6f}"
        )

        print(
            f"Final expand loss:  "
            f"{last['expand']:.6f}"
        )

        print(
            f"Final zone loss:    "
            f"{last['zone']:.6f}"
        )

        print()

        print(
            f"Root states:        "
            f"{root_count}"
        )

        print(
            f"Expansion states:   "
            f"{expand_count}"
        )

        print(
            f"Zone states:        "
            f"{zone_count}"
        )

        print("=" * 70)
        print()

    return model


# ============================================================
# GENERATION METRICS
# ============================================================

def print_generation_metrics(
    generated_zones,
    actual_zones,
    zone_types,
):
    """
    Compare generated zoning against the known
    zoning labels in the source city.
    """

    generated = (
        generated_zones
        .detach()
        .cpu()
    )

    actual = (
        actual_zones
        .detach()
        .cpu()
    )

    total_cells = len(actual)

    assigned_mask = (
        generated >= 0
    )

    assigned_cells = (
        assigned_mask.sum().item()
    )

    unassigned_cells = (
        total_cells
        - assigned_cells
    )

    coverage = (
        assigned_cells
        / total_cells
        * 100
        if total_cells > 0
        else 0.0
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 70)
    print("GENERATION RESULTS")
    print("=" * 70)

    print(
        f"Total cells:          "
        f"{total_cells}"
    )

    print(
        f"Assigned cells:       "
        f"{assigned_cells}"
    )

    print(
        f"Unassigned cells:     "
        f"{unassigned_cells}"
    )

    print(
        f"Coverage:             "
        f"{coverage:.2f}%"
    )

    # ========================================================
    # GENERATED DISTRIBUTION
    # ========================================================

    print()
    print("GENERATED ZONES")
    print("-" * 50)

    if assigned_cells > 0:

        generated_counts = Counter(
            generated[
                assigned_mask
            ].tolist()
        )

    else:
        generated_counts = Counter()

    for zone_id, zone_name in enumerate(
        zone_types
    ):

        count = generated_counts.get(
            zone_id,
            0,
        )

        percentage = (
            count
            / assigned_cells
            * 100
            if assigned_cells > 0
            else 0.0
        )

        print(
            f"{zone_name:<20}"
            f"{count:>8} "
            f"({percentage:>6.2f}%)"
        )

    # ========================================================
    # ACTUAL DISTRIBUTION
    # ========================================================

    print()
    print("ACTUAL ZONES")
    print("-" * 50)

    actual_counts = Counter(
        actual.tolist()
    )

    for zone_id, zone_name in enumerate(
        zone_types
    ):

        count = actual_counts.get(
            zone_id,
            0,
        )

        percentage = (
            count
            / total_cells
            * 100
            if total_cells > 0
            else 0.0
        )

        print(
            f"{zone_name:<20}"
            f"{count:>8} "
            f"({percentage:>6.2f}%)"
        )

    # ========================================================
    # ACCURACY
    # ========================================================

    print()
    print("COMPARISON")
    print("-" * 50)

    if assigned_cells > 0:

        correct = (
            generated[assigned_mask]
            == actual[assigned_mask]
        ).sum().item()

        accuracy = (
            correct
            / assigned_cells
            * 100
        )

        print(
            f"Accuracy on assigned: "
            f"{accuracy:.2f}%"
        )

    else:

        print(
            "Accuracy on assigned: "
            "N/A"
        )

    if assigned_cells == total_cells:

        overall_accuracy = (
            generated == actual
        ).float().mean().item() * 100

        print(
            f"Overall accuracy:     "
            f"{overall_accuracy:.2f}%"
        )

    else:

        print(
            "Overall accuracy:     "
            "N/A (incomplete generation)"
        )

    print("=" * 70)
    print()
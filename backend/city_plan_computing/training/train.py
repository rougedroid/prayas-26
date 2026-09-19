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
    (Identical to CityGeneratorEngine._frontier.)
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
# STOP LABELS
# ============================================================

def build_stop_labels(training_states):
    """
    The generator asks the stop head "should this zone stop?"
    right AFTER every expansion. This builds the matching
    training labels.

    For every state that contains an expansion:
        label 0 -> the next state is another expansion of the
                   same zone (zone keeps growing)
        label 1 -> anything else (new root, or end of sequence)

    If a state already carries a `stop_target` attribute,
    that is used instead.

    Returns:
        dict {state_index: 0 or 1}
    """

    labels = {}
    n = len(training_states)

    for i, state in enumerate(training_states):

        if state.expand_target is None:
            continue

        if state.active_zone is None:
            continue

        explicit = getattr(
            state,
            "stop_target",
            None,
        )

        if explicit is not None:
            labels[i] = int(explicit)
            continue

        if i + 1 >= n:
            labels[i] = 1
            continue

        nxt = training_states[i + 1]

        continues = (
            nxt.expand_target is not None
            and nxt.root_target is None
            and nxt.active_zone == state.active_zone
        )

        labels[i] = 0 if continues else 1

    return labels


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

    # weights_only=False: the checkpoint contains the
    # normalization statistics (numpy objects), which the
    # default weights_only=True loader rejects on newer torch.
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
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
    stop_loss_weight=1.0,

    checkpoint_path=None,
    normalization=None,

    ask_model_choice=True,
    shuffle=True,
):
    """
    Train the city generator.

    Heads trained:
        root    - which unassigned cell starts the next zone
        zone    - which zone type that root gets
        expand  - which frontier cell the zone grows into
        stop    - whether the zone stops after an expansion

    The stop head MUST be trained: the generation engine calls it
    after every expansion.

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

        if checkpoint_normalization is not None:
            normalization = (
                checkpoint_normalization
            )

    # --------------------------------------------------------
    # STOP LABELS
    # --------------------------------------------------------

    stop_labels = build_stop_labels(
        training_states
    )

    n_stop_pos = sum(
        1 for v in stop_labels.values() if v == 1
    )

    n_stop_neg = len(stop_labels) - n_stop_pos

    n_root_states = sum(
        1
        for s in training_states
        if s.root_target is not None
    )

    train_stop = (
        n_stop_pos > 0
        and n_stop_neg > 0
    )

    if train_stop:

        # Stops are rare compared with continues; without
        # re-weighting the head would learn "never stop", and
        # the first zone would flood the whole graph.
        pos_weight_value = min(
            max(n_stop_neg / n_stop_pos, 1.0),
            100.0,
        )

    else:
        pos_weight_value = 1.0

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    # --------------------------------------------------------
    # TRAINING INFORMATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CITY GENERATOR TRAINING")
    print("=" * 70)

    print(f"Device:              {device}")

    print(
        f"Model mode:          "
        f"{'CONTINUE' if model_mode == 'load' else 'NEW'}"
    )

    print(f"Starting epoch:      {start_epoch}")
    print(f"Training epochs:     {epochs}")

    print(
        f"Training states:     "
        f"{len(training_states)}"
    )

    print(f"Learning rate:       {learning_rate}")
    print(f"Weight decay:        {weight_decay}")
    print(f"Shuffle states:      {shuffle}")

    print(
        f"Zone starts:         {n_root_states}"
    )

    print(
        f"Stop labels:         "
        f"{n_stop_pos} stop / {n_stop_neg} continue "
        f"(pos_weight={pos_weight_value:.2f})"
    )

    if not train_stop:
        print(
            "WARNING: could not build both stop and "
            "continue labels. Stop head will NOT be "
            "trained. Check build_stop_labels()."
        )

    if checkpoint_path is not None:
        print(f"Checkpoint:          {checkpoint_path}")

    print("=" * 70)
    print()

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    history = []

    num_states_total = len(training_states)

    root_count = 0
    expand_count = 0
    zone_count = 0
    stop_count = 0

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
        stop_loss_total = 0.0

        root_count = 0
        expand_count = 0
        zone_count = 0
        stop_count = 0

        root_correct = 0
        expand_correct = 0
        zone_correct = 0

        stop_pos_total = 0
        stop_pos_correct = 0
        stop_neg_total = 0
        stop_neg_correct = 0

        if shuffle:
            order = torch.randperm(
                num_states_total
            ).tolist()
        else:
            order = list(
                range(num_states_total)
            )

        progress = tqdm(
            order,
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

        for idx in progress:

            state = training_states[idx]

            optimizer.zero_grad(
                set_to_none=True
            )

            assigned = state.assigned_zones

            # -----------------------------------------------
            # BUILD CURRENT GRAPH FEATURES
            # -----------------------------------------------

            dynamic = build_dynamic_features_tensor(
                assigned_zones=assigned,
                neighbor_lists=graph.neighbor_lists,
                num_zones=num_zones,
                device=device,
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

            h = outputs["embeddings"]

            loss = None

            # -----------------------------------------------
            # START_ZONE (root + zone type)
            # -----------------------------------------------

            if state.root_target is not None:

                unassigned = torch.where(
                    assigned == -1
                )[0].to(device)

                if len(unassigned) > 0:

                    root_logits = (
                        outputs["root_logits"][unassigned]
                    )

                    target_position = (
                        unassigned == state.root_target
                    ).nonzero(
                        as_tuple=False
                    )

                    if len(target_position) > 0:

                        target_position = (
                            target_position[0].item()
                        )

                        root_target = torch.tensor(
                            [target_position],
                            dtype=torch.long,
                            device=device,
                        )

                        root_loss = F.cross_entropy(
                            root_logits.unsqueeze(0),
                            root_target,
                        )

                        zone_logits = (
                            outputs["zone_logits"][
                                state.root_target
                            ].unsqueeze(0)
                        )

                        zone_target = torch.tensor(
                            [state.zone_target],
                            dtype=torch.long,
                            device=device,
                        )

                        zone_loss = F.cross_entropy(
                            zone_logits,
                            zone_target,
                        )

                        if loss is None:
                            loss = torch.tensor(
                                0.0,
                                device=device,
                            )

                        loss = (
                            loss
                            + root_loss_weight * root_loss
                            + zone_loss_weight * zone_loss
                        )

                        root_loss_total += root_loss.item()
                        zone_loss_total += zone_loss.item()

                        root_count += 1
                        zone_count += 1

                        if (
                            torch.argmax(root_logits).item()
                            == target_position
                        ):
                            root_correct += 1

                        if (
                            torch.argmax(zone_logits).item()
                            == int(state.zone_target)
                        ):
                            zone_correct += 1

            # -----------------------------------------------
            # EXPANSION (+ STOP)
            # -----------------------------------------------

            if (
                state.expand_target is not None
                and state.active_zone is not None
            ):

                frontier = get_frontier(
                    assigned_zones=assigned,
                    zone_id=state.active_zone,
                    neighbor_lists=graph.neighbor_lists,
                )

                if len(frontier) > 0:

                    frontier_tensor = torch.tensor(
                        frontier,
                        dtype=torch.long,
                        device=device,
                    )

                    expand_logits = model.expansion_scores(
                        h,
                        frontier_tensor,
                        state.active_zone,
                    )

                    target_position = (
                        frontier_tensor
                        == state.expand_target
                    ).nonzero(
                        as_tuple=False
                    )

                    if len(target_position) > 0:

                        target_position = (
                            target_position[0].item()
                        )

                        expand_target = torch.tensor(
                            [target_position],
                            dtype=torch.long,
                            device=device,
                        )

                        expand_loss = F.cross_entropy(
                            expand_logits.unsqueeze(0),
                            expand_target,
                        )

                        if loss is None:
                            loss = torch.tensor(
                                0.0,
                                device=device,
                            )

                        loss = (
                            loss
                            + expand_loss_weight * expand_loss
                        )

                        expand_loss_total += expand_loss.item()
                        expand_count += 1

                        if (
                            torch.argmax(expand_logits).item()
                            == target_position
                        ):
                            expand_correct += 1

                        # ---------------------------------
                        # STOP HEAD
                        #
                        # Mirrors the generator exactly:
                        #   h          = embeddings of the state
                        #                BEFORE the expansion
                        #   zone_nodes = all cells of the zone
                        #                AFTER the expansion
                        # ---------------------------------

                        if train_stop and idx in stop_labels:

                            zone_nodes = torch.where(
                                assigned == state.active_zone
                            )[0].to(device)

                            new_cell = torch.tensor(
                                [int(state.expand_target)],
                                dtype=torch.long,
                                device=device,
                            )

                            zone_nodes = torch.unique(
                                torch.cat(
                                    [
                                        zone_nodes,
                                        new_cell,
                                    ]
                                )
                            )

                            stop_logit = model.stop_score(
                                h,
                                zone_nodes,
                            ).reshape(1)

                            stop_label_value = stop_labels[idx]

                            stop_label = torch.tensor(
                                [float(stop_label_value)],
                                dtype=torch.float32,
                                device=device,
                            )

                            stop_loss = (
                                F.binary_cross_entropy_with_logits(
                                    stop_logit,
                                    stop_label,
                                    pos_weight=pos_weight,
                                )
                            )

                            loss = (
                                loss
                                + stop_loss_weight * stop_loss
                            )

                            stop_loss_total += stop_loss.item()
                            stop_count += 1

                            predicted_stop = (
                                stop_logit.item() > 0.0
                            )

                            if stop_label_value == 1:
                                stop_pos_total += 1
                                if predicted_stop:
                                    stop_pos_correct += 1
                            else:
                                stop_neg_total += 1
                                if not predicted_stop:
                                    stop_neg_correct += 1

            # -----------------------------------------------
            # NO LOSS?
            # -----------------------------------------------

            if loss is None:
                continue

            # -----------------------------------------------
            # BACKPROP
            # -----------------------------------------------

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            loss_value = loss.detach().item()

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

        avg_total = total_loss / num_states

        avg_root = root_loss_total / max(root_count, 1)
        avg_expand = expand_loss_total / max(expand_count, 1)
        avg_zone = zone_loss_total / max(zone_count, 1)
        avg_stop = stop_loss_total / max(stop_count, 1)

        root_acc = root_correct / max(root_count, 1)
        expand_acc = expand_correct / max(expand_count, 1)
        zone_acc = zone_correct / max(zone_count, 1)

        stop_recall = (
            stop_pos_correct / max(stop_pos_total, 1)
        )

        stop_specificity = (
            stop_neg_correct / max(stop_neg_total, 1)
        )

        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": avg_total,
            "root": avg_root,
            "expand": avg_expand,
            "zone": avg_zone,
            "stop": avg_stop,
            "root_acc": root_acc,
            "expand_acc": expand_acc,
            "zone_acc": zone_acc,
            "stop_recall": stop_recall,
            "stop_specificity": stop_specificity,
        }

        history.append(epoch_metrics)

        # ----------------------------------------------------
        # PRINT EPOCH RESULTS
        # ----------------------------------------------------

        print(
            f"Epoch {epoch + 1:03d} | "
            f"loss={avg_total:.4f} | "
            f"root={avg_root:.4f} | "
            f"expand={avg_expand:.4f} | "
            f"zone={avg_zone:.4f} | "
            f"stop={avg_stop:.4f}"
        )

        print(
            f"          teacher-forced acc: "
            f"root={root_acc:.3f} "
            f"expand={expand_acc:.3f} "
            f"zone={zone_acc:.3f} "
            f"| stop recall={stop_recall:.3f} "
            f"specificity={stop_specificity:.3f}"
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

        print(f"Initial loss:       {first['loss']:.6f}")
        print(f"Final loss:         {last['loss']:.6f}")

        if first["loss"] > 0:

            reduction = (
                1.0
                - last["loss"]
                / first["loss"]
            ) * 100

            print(f"Loss reduction:     {reduction:.2f}%")

        print()

        print(f"Final root loss:    {last['root']:.6f}")
        print(f"Final expand loss:  {last['expand']:.6f}")
        print(f"Final zone loss:    {last['zone']:.6f}")
        print(f"Final stop loss:    {last['stop']:.6f}")

        print()

        print(
            "NOTE: accuracies above are TEACHER-FORCED "
            "(each step sees the true partial city). "
            "They are not generation accuracy."
        )

        print()

        print(f"Root states:        {root_count}")
        print(f"Expansion states:   {expand_count}")
        print(f"Zone states:        {zone_count}")
        print(f"Stop states:        {stop_count}")

        print("=" * 70)
        print()

    model.eval()

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

    assigned_mask = generated >= 0

    assigned_cells = assigned_mask.sum().item()

    unassigned_cells = total_cells - assigned_cells

    coverage = (
        assigned_cells / total_cells * 100
        if total_cells > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("GENERATION RESULTS")
    print("=" * 70)

    print(f"Total cells:          {total_cells}")
    print(f"Assigned cells:       {assigned_cells}")
    print(f"Unassigned cells:     {unassigned_cells}")
    print(f"Coverage:             {coverage:.2f}%")

    # ========================================================
    # GENERATED DISTRIBUTION
    # ========================================================

    print()
    print("GENERATED ZONES")
    print("-" * 50)

    if assigned_cells > 0:
        generated_counts = Counter(
            generated[assigned_mask].tolist()
        )
    else:
        generated_counts = Counter()

    for zone_id, zone_name in enumerate(zone_types):

        count = generated_counts.get(zone_id, 0)

        percentage = (
            count / assigned_cells * 100
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

    actual_counts = Counter(actual.tolist())

    for zone_id, zone_name in enumerate(zone_types):

        count = actual_counts.get(zone_id, 0)

        percentage = (
            count / total_cells * 100
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

        accuracy = correct / assigned_cells * 100

        print(f"Accuracy on assigned: {accuracy:.2f}%")

    else:

        print("Accuracy on assigned: N/A")

    if assigned_cells == total_cells:

        overall_accuracy = (
            generated == actual
        ).float().mean().item() * 100

        print(f"Overall accuracy:     {overall_accuracy:.2f}%")

    else:

        print(
            "Overall accuracy:     "
            "N/A (incomplete generation)"
        )

    print("=" * 70)
    print()
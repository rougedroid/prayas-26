import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    ZONE_TYPES,
)

from zoner import (
    NUM_ZONES,
    CellZoner,
    CityData,
    load_cells,
    fit_multi_normalization,
    model_input_dim,
    build_input,
    evaluate_city,
    save_zoner_checkpoint,
)


# ============================================================
# SETTINGS
# ============================================================

# Well-planned cities to LEARN from.
TRAIN_DATABASES = ["tokyo", "amsterdam","stockholm", "london"]

# Held-out well-planned cities: used to choose the best variant and
# best epoch. Keep the city you finally audit OUT of both lists.
# (If it is Amsterdam, move it out of VAL before the final audit run
# is trusted, or add another city here.)
VAL_DATABASES = ["barcelona"]

CHECKPOINT_PATH = "models/zoner1"

# Static-feature handling to compare. The best one on VAL is kept.
#   context = ignore static features, use only surrounding zones
#   city    = static features z-scored inside each city
#   global  = static features normalised with training statistics
VARIANTS = ["context", "city", "global"]

STATIC_MODE_IF_NO_VAL = "city"

EPOCHS = 200
EPOCHS_IF_NO_VAL = 60
PASSES_PER_CITY = 4

ZONER_HIDDEN = 64
ZONER_DROPOUT = 0.3

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

MASK_RANGE = (0.10, 0.30)     # fraction of cells hidden per pass
STATIC_NOISE = 0.10
STATIC_BLOCK_DROP = 0.5       # chance the whole static block is zeroed
LABEL_SMOOTHING = 0.0
CLASS_WEIGHT_POWER = 0.0      # 0 = none. Weighting hurt accuracy before.

EVAL_EVERY = 5
PATIENCE_EVALS = 100

SEED = 0


# ============================================================
# ONE VARIANT
# ============================================================

def val_metrics(model, val_cities):

    results = [
        evaluate_city(model, city, rounds=2)
        for city in val_cities
    ]

    keys = (
        "accuracy",
        "balanced_accuracy",
        "neighbor_baseline",
        "majority_baseline",
        "flag_rate",
    )

    return {
        k: float(np.mean([r[k] for r in results]))
        for k in keys
    }


def run_variant(
    name,
    static_mode,
    train_raw,
    val_raw,
    normalization,
    device,
):

    print()
    print("#" * 70)
    print(f"VARIANT '{name}'  (static_mode={static_mode})")
    print("#" * 70)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_cities = [
        CityData(db, cells, normalization, device, static_mode)
        for db, cells in train_raw.items()
    ]

    val_cities = [
        CityData(db, cells, normalization, device, static_mode)
        for db, cells in val_raw.items()
    ]

    has_val = len(val_cities) > 0

    static_dim = train_cities[0].static.shape[1]

    # ---- optional mild class weights ----

    counts = torch.zeros(NUM_ZONES)

    for city in train_cities:

        labels = city.labels_cpu[city.labels_cpu >= 0]

        counts += torch.bincount(
            labels,
            minlength=NUM_ZONES,
        ).float()

    weights = torch.ones(NUM_ZONES)

    if CLASS_WEIGHT_POWER > 0:

        present = counts > 0

        weights = torch.zeros(NUM_ZONES)
        weights[present] = counts[present] ** (-CLASS_WEIGHT_POWER)
        weights[present] /= weights[present].mean()

    weights = weights.to(device)

    # ---- model ----

    input_dim = model_input_dim(static_dim)

    model = CellZoner(
        input_dim=input_dim,
        hidden_dim=ZONER_HIDDEN,
        num_zones=NUM_ZONES,
        static_dim=static_dim,
        dropout=ZONER_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    meta = {
        "input_dim": input_dim,
        "static_dim": static_dim,
        "hidden_dim": ZONER_HIDDEN,
        "num_zones": NUM_ZONES,
        "dropout": ZONER_DROPOUT,
        "zone_types": list(ZONE_TYPES),
        "train_databases": list(train_raw.keys()),
        "static_mode": static_mode,
    }

    path = f"{CHECKPOINT_PATH}_{name}"

    best = {
        "score": -1.0,
        "epoch": -1,
        "metrics": None,
    }

    epochs = EPOCHS if has_val else EPOCHS_IF_NO_VAL

    def evaluate_and_save(epoch, train_line):

        m = val_metrics(model, val_cities)

        print(
            f"{train_line} | VAL acc={m['accuracy']:.3f} "
            f"balanced={m['balanced_accuracy']:.3f} "
            f"(neighbour-majority {m['neighbor_baseline']:.3f}, "
            f"always-majority {m['majority_baseline']:.3f}) "
            f"flagged={m['flag_rate']:.1%}"
        )

        if m["accuracy"] > best["score"]:

            best["score"] = m["accuracy"]
            best["epoch"] = epoch
            best["metrics"] = m

            save_zoner_checkpoint(
                path,
                model,
                epoch,
                normalization,
                meta,
                metrics=m,
            )

            return True

        return False

    # ---- epoch 0 = untrained model = pure neighbour-majority ----

    if has_val:
        evaluate_and_save(0, "Epoch 000 (untrained)")

    evals_without_gain = 0

    for epoch in range(1, epochs + 1):

        model.train()

        loss_sum = 0.0
        loss_n = 0
        correct = 0
        total = 0

        for city in train_cities:

            n = city.num_nodes

            for _ in range(PASSES_PER_CITY):

                ratio = random.uniform(*MASK_RANGE)

                k = max(
                    1,
                    int(round(ratio * len(city.known_idx))),
                )

                chosen = city.known_idx[
                    torch.randperm(
                        len(city.known_idx),
                        device=device,
                    )[:k]
                ]

                hidden = torch.zeros(
                    n,
                    dtype=torch.bool,
                    device=device,
                )

                hidden[chosen] = True

                static = city.static

                if static_mode != "none":

                    if random.random() < STATIC_BLOCK_DROP:
                        static = torch.zeros_like(static)
                    else:
                        static = static + (
                            STATIC_NOISE
                            * torch.randn_like(static)
                        )

                x = build_input(
                    static,
                    city.labels,
                    hidden,
                    city.edge_index,
                )

                logits = model(x, city.edge_index)

                loss = F.cross_entropy(
                    logits[chosen],
                    city.labels[chosen],
                    weight=weights,
                    label_smoothing=LABEL_SMOOTHING,
                )

                optimizer.zero_grad(set_to_none=True)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                loss_sum += loss.item()
                loss_n += 1

                correct += (
                    logits[chosen].argmax(dim=1)
                    == city.labels[chosen]
                ).sum().item()

                total += len(chosen)

        if (
            epoch % EVAL_EVERY != 0
            and epoch != epochs
        ):
            continue

        train_line = (
            f"Epoch {epoch:03d} | "
            f"loss={loss_sum / max(loss_n, 1):.4f} | "
            f"train acc={correct / max(total, 1):.3f}"
        )

        if has_val:

            improved = evaluate_and_save(epoch, train_line)

            if improved:
                evals_without_gain = 0
            else:
                evals_without_gain += 1

                if evals_without_gain >= PATIENCE_EVALS:
                    print(
                        f"Early stop: no gain for "
                        f"{PATIENCE_EVALS} evaluations."
                    )
                    break

        else:

            print(train_line)

    if not has_val:

        save_zoner_checkpoint(
            path,
            model,
            epochs,
            normalization,
            meta,
        )

        best["epoch"] = epochs

    return {
        "name": name,
        "static_mode": static_mode,
        "path": path,
        **best,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    overlap = set(TRAIN_DATABASES) & set(VAL_DATABASES)

    if overlap:
        raise ValueError(
            f"Databases in both TRAIN and VAL: {overlap}"
        )

    print("=" * 70)
    print("MASKED-ZONE TRAINING")
    print("=" * 70)
    print(f"Device : {device}")
    print(f"Train  : {TRAIN_DATABASES}")
    print(f"Val    : {VAL_DATABASES if VAL_DATABASES else 'none'}")

    train_raw = {
        db: load_cells(db) for db in TRAIN_DATABASES
    }

    val_raw = {
        db: load_cells(db) for db in VAL_DATABASES
    }

    normalization = fit_multi_normalization(
        list(train_raw.values())
    )

    has_val = len(val_raw) > 0

    if has_val:
        variants = list(VARIANTS)
    else:
        variants = ["single"]

        print()
        print(
            "No validation cities: training one model with "
            f"static_mode='{STATIC_MODE_IF_NO_VAL}'. "
            "Put a held-out city in VAL_DATABASES to compare "
            "variants and get honest scores."
        )

    mode_of = {
        "context": "none",
        "city": "city",
        "global": "global",
        "single": STATIC_MODE_IF_NO_VAL,
    }

    results = []

    for name in variants:

        results.append(
            run_variant(
                name,
                mode_of[name],
                train_raw,
                val_raw,
                normalization,
                device,
            )
        )

    # --------------------------------------------------------
    # SUMMARY + KEEP BEST
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if has_val:

        for r in results:

            m = r["metrics"]

            print(
                f"{r['name']:<10} best epoch {r['epoch']:>3} | "
                f"val acc={m['accuracy']:.3f} "
                f"balanced={m['balanced_accuracy']:.3f} "
                f"flagged={m['flag_rate']:.1%}"
            )

        print(
            f"(neighbour-majority baseline on val: "
            f"{results[0]['metrics']['neighbor_baseline']:.3f})"
        )

        winner = max(results, key=lambda r: r["score"])

    else:

        winner = results[0]

    shutil.copyfile(
        winner["path"],
        CHECKPOINT_PATH,
    )

    print()
    print(
        f"Winner: '{winner['name']}' "
        f"(static_mode={winner['static_mode']}) "
        f"-> copied to {CHECKPOINT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
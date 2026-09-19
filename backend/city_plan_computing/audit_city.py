import sys
from collections import Counter

import torch

from zoner import (
    load_cells,
    load_zoner_checkpoint,
    CityData,
    predict_probs,
    audit_city,
    write_findings_csv,
)


# ============================================================
# SETTINGS
# ============================================================

AUDIT_DATABASE = "amsterdam"      # or: python audit_city.py <database>

CHECKPOINT_PATH = "models/zoner1"

FOLDS = 8          # each cell is hidden together with ~1/FOLDS of the city
ROUNDS = 5         # random repeats, averaged (more = steadier)

# A cell is flagged when the model's best zone differs from the actual
# zone, is at least MARGIN more probable, and has at least MIN_CONF.
# Lower = flag more cells, higher = flag only the clearest problems.
MARGIN = 0.35
MIN_CONF = 0.40

TOP_N = 40


def main():

    database = (
        sys.argv[1]
        if len(sys.argv) > 1
        else AUDIT_DATABASE
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, ckpt = load_zoner_checkpoint(
        CHECKPOINT_PATH,
        device,
    )

    trained_on = ckpt["meta"].get("train_databases", [])

    print("=" * 70)
    print("ZONING AUDIT")
    print("=" * 70)
    print(f"Database   : {database}")
    print(f"Checkpoint : {CHECKPOINT_PATH} (epoch {ckpt.get('epoch')})")
    print(f"Trained on : {trained_on}")

    if database in trained_on:
        print(
            "WARNING: this city was in the training set. The model has "
            "partly memorised it, so it will under-report bad cells. "
            "Audit a city it has not seen."
        )

    normalization = ckpt["normalization"]

    cells = load_cells(database)

    static_mode = ckpt["meta"].get("static_mode", "global")

    print(f"Static mode: {static_mode}")

    city = CityData(
        database,
        cells,
        normalization,
        device,
        static_mode=static_mode,
    )

    probs = predict_probs(
        model,
        city,
        folds=FOLDS,
        rounds=ROUNDS,
    )

    findings = audit_city(
        city,
        probs,
        margin=MARGIN,
        min_conf=MIN_CONF,
    )

    known = len(city.known_idx)

    print()
    print(
        f"Flagged {len(findings)} of {known} cells "
        f"({len(findings) / max(known, 1):.1%})"
    )

    if not findings:
        print("Nothing flagged at these thresholds.")
        return

    print()
    print("MOST LIKELY MISZONED CELLS")
    print("-" * 70)

    for rank, f in enumerate(findings[:TOP_N], start=1):

        print(
            f"{rank:>3}. cell {f['cell_id']}: "
            f"{f['actual_zone']} -> {f['suggested_zone']} "
            f"(model {f['suggested_confidence']:.0%} vs "
            f"{f['actual_probability']:.0%} for current)  "
            f"neighbours: {f['neighbours'] or 'none'}"
        )

    print()
    print("SUGGESTED CHANGES")
    print("-" * 70)

    transitions = Counter(
        (f["actual_zone"], f["suggested_zone"])
        for f in findings
    )

    for (a, s), n in transitions.most_common():
        print(f"{a:<20} -> {s:<20} {n:>5}")

    out_path = f"audit_{database}.csv"

    write_findings_csv(out_path, findings)

    print()
    print(f"Full list written to {out_path}")


if __name__ == "__main__":
    main()
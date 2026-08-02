#!/usr/bin/env python3
"""Generate TuRBO-1-style trust-region Bayesian optimisation proposals.

Requires PyTorch, GPyTorch and BoTorch. This utility fits a Gaussian process to
completed Stage-B evaluations and optimises q-expected improvement inside a
box centred on the best eligible point. It generates proposals only; the Slurm
array evaluates them with the C++ simulator.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

PARAMETERS = [
    "market_maker_interval_ms",
    "market_maker_min_spread_ticks",
    "momentum_rate_per_second",
    "momentum_threshold_ticks",
    "informed_rate_per_second",
    "informed_signal_precision",
    "institutional_rate_per_second",
    "institutional_participation_cap",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--length", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    try:
        import torch
        from botorch.acquisition import qExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as error:
        raise SystemExit(
            "TuRBO refinement requires torch, gpytorch and botorch. "
            "Install them in a dedicated Python environment."
        ) from error

    rows: list[dict[str, str]] = []
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    def usable(row: dict[str, str]) -> bool:
        if row.get("structurally_valid") not in (None, "", "1"):
            return False
        if row.get("terminated_early") == "1":
            return False
        try:
            return math.isfinite(float(row.get("distance", "nan")))
        except (TypeError, ValueError):
            return False

    rows = [row for row in rows if usable(row)]
    if len(rows) < 10:
        raise SystemExit("At least ten completed evaluations are required")

    torch.manual_seed(args.seed)
    x = torch.tensor(
        [[float(row[f"u_{name}"]) for name in PARAMETERS] for row in rows],
        dtype=torch.double,
    )
    # BoTorch maximises; use negative distance.
    y = torch.tensor(
        [[-float(row["distance"])] for row in rows], dtype=torch.double
    )

    model = SingleTaskGP(
        x,
        y,
        input_transform=Normalize(d=len(PARAMETERS)),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    best_index = int(torch.argmax(y).item())
    centre = x[best_index]
    half = max(0.01, min(1.0, args.length)) / 2.0
    lower = torch.clamp(centre - half, 0.0, 1.0)
    upper = torch.clamp(centre + half, 0.0, 1.0)
    bounds = torch.stack([lower, upper])

    acquisition = qExpectedImprovement(model=model, best_f=float(y.max()))
    candidates, _ = optimize_acqf(
        acquisition,
        bounds=bounds,
        q=args.n,
        num_restarts=10,
        raw_samples=512,
        sequential=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", *[f"u_{name}" for name in PARAMETERS]])
        for index, candidate in enumerate(candidates.detach().cpu().tolist()):
            writer.writerow([index, *[f"{value:.17g}" for value in candidate]])


if __name__ == "__main__":
    main()

"""Compare hierarchical clustering and PCA followed by K-means on development data."""

import argparse
import json
from os import environ
from pathlib import Path

environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from baskets import build_panel, load_inputs
from evaluate import (
    HOLDOUT_START, TARGET, VALIDATION_YEARS, WITH_PERSISTENCE,
    block_interval, daily_metrics, fit_ridge, predict_ridge, split_at,
)

PCA_COMPONENTS = 5
KMEANS_STARTS = 5
KMEANS_SEED = 42
RIDGE_PENALTY = 100  # Selected for the HC persistence model in Experiment 01.


def seed_centers(points, clusters, rng):
    centers = np.empty((clusters, points.shape[1]))
    centers[0] = points[rng.integers(len(points))]
    closest = np.sum((points - centers[0]) ** 2, axis=1)
    for index in range(1, clusters):
        total = closest.sum()
        probabilities = closest / total if total > 0 else None
        centers[index] = points[rng.choice(len(points), p=probabilities)]
        closest = np.minimum(closest, np.sum((points - centers[index]) ** 2, axis=1))
    return centers


def kmeans_labels(points, clusters):
    rng = np.random.default_rng(KMEANS_SEED)
    best_labels = None
    best_inertia = np.inf

    for _ in range(KMEANS_STARTS):
        centers = seed_centers(points, clusters, rng)
        labels = np.full(len(points), -1, dtype=int)
        for _ in range(100):
            # The squared norm of each point is constant across candidate centers.
            distances = np.sum(centers ** 2, axis=1) - 2 * (points @ centers.T)
            assigned = distances.argmin(axis=1)
            if np.array_equal(assigned, labels):
                break
            labels = assigned
            counts = np.bincount(labels, minlength=clusters)[:, None]
            totals = np.zeros_like(centers)
            np.add.at(totals, labels, points)
            np.divide(totals, counts, out=centers, where=counts > 0)

        distances = np.sum(centers ** 2, axis=1) - 2 * (points @ centers.T)
        labels = distances.argmin(axis=1)
        inertia = np.sum((points - centers[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    return best_labels


def pca_kmeans_labels(normalized, clusters):
    # Each row is a stock and each column is a date in the trailing return window.
    points = normalized.T - normalized.mean(axis=1)
    _, directions = np.linalg.eigh(points.T @ points)
    scores = points @ directions[:, -PCA_COMPONENTS:]
    return kmeans_labels(scores, clusters)


def score_panel(panel, coverage):
    fields = [*WITH_PERSISTENCE, TARGET]
    usable = panel.loc[np.isfinite(panel[fields]).all(axis=1) & panel["label_end"].lt(HOLDOUT_START)]
    daily = []
    for year in VALIDATION_YEARS:
        train, test = split_at(usable, pd.Timestamp(year, 1, 1), pd.Timestamp(year + 2, 1, 1))
        model = fit_ridge(train, WITH_PERSISTENCE, RIDGE_PENALTY)
        scored = daily_metrics(test, predict_ridge(model, test, WITH_PERSISTENCE))
        mean_return = test.groupby("date")[TARGET].mean()
        scored["selection_gain"] = scored["top_two_excess"] - mean_return
        daily.append(scored)
    largest = panel.groupby("date")["size"].max() / coverage.set_index("date")["eligible_stocks"]
    return pd.concat(daily).assign(largest_share=largest)


def summarize(daily, calendar):
    result = {"paired_dates": len(daily)}
    for method in ("hc", "pca"):
        result[method] = {
            "rank_ic": float(daily[f"rank_ic_{method}"].mean()),
            "median_baskets": float(daily[f"baskets_{method}"].median()),
            "median_largest_share": float(daily[f"largest_share_{method}"].median()),
            "mean_top_two_excess": float(daily[f"top_two_excess_{method}"].mean()),
            "mean_selection_gain": float(daily[f"selection_gain_{method}"].mean()),
        }
    gap = daily["rank_ic_pca"] - daily["rank_ic_hc"]
    result["delta_rank_ic"] = float(gap.mean())
    result["delta_rank_ic_ci95"] = block_interval(gap, calendar)
    result["pca_rank_ic_ci95"] = block_interval(daily["rank_ic_pca"], calendar)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True, type=Path)
    parser.add_argument("--volume", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/basket_comparison"))
    args = parser.parse_args()
    prices, volumes = load_inputs(args.prices, args.volume)
    cutoff = HOLDOUT_START - pd.Timedelta(days=1)

    hc_panel, hc_coverage = build_panel(prices, volumes, through=cutoff)
    hc = score_panel(hc_panel, hc_coverage)
    print(f"Scored HC on {len(hc):,} validation sessions.", flush=True)
    pca_panel, pca_coverage = build_panel(prices, volumes, through=cutoff, cluster_labels=pca_kmeans_labels)
    pca = score_panel(pca_panel, pca_coverage)
    pd.testing.assert_frame_equal(
        hc_coverage[["date", "eligible_stocks"]], pca_coverage[["date", "eligible_stocks"]],
    )

    paired = hc.join(pca, how="inner", lsuffix="_hc", rsuffix="_pca")
    calendar = pd.DatetimeIndex(hc_coverage["date"])
    calendar = calendar[calendar.year >= VALIDATION_YEARS[0]]
    blocks = []
    for year in VALIDATION_YEARS:
        block = paired[(paired.index.year >= year) & (paired.index.year < year + 2)]
        if block.empty:
            raise ValueError(f"No common scoring dates in the fold starting in {year}.")
        sessions = calendar[(calendar.year >= year) & (calendar.year < year + 2)]
        blocks.append({"start_year": year, **summarize(block, sessions)})

    summary = {
        "pca_components": PCA_COMPONENTS,
        "kmeans_starts": KMEANS_STARTS,
        "kmeans_seed": KMEANS_SEED,
        "ridge_penalty": RIDGE_PENALTY,
        "scored_sessions_hc": len(hc),
        "scored_sessions_pca": len(pca),
        "common_sessions": summarize(paired, calendar),
        "validation_blocks": blocks,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output / "paired_daily.csv", index_label="date")
    pca_coverage.to_csv(args.output / "pca_coverage.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Scored PCA + K-means on {len(pca):,} sessions and compared {len(paired):,} paired sessions.")


if __name__ == "__main__":
    main()

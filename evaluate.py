"""Tune both Ridge models and compare their dated predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = ["relative_strength", "breadth", "unusual_volume"]
WITH_PERSISTENCE = [*BASE, "persistence"]
TARGET = "excess_return_5d"
LAMBDAS = (0.01, 0.1, 1, 10, 100)
VALIDATION_YEARS = (2015, 2017, 2019, 2021)
HOLDOUT_START = pd.Timestamp("2023-01-01")


def fit_ridge(train, features, penalty):
    x = train[features].to_numpy()
    y = train[TARGET].to_numpy()
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    x = (x - center) / scale
    intercept = y.mean()
    # The research objective uses mean squared error, so lambda is unscaled here.
    gram = x.T @ x / len(x) + penalty * np.eye(len(features))
    coefficients = np.linalg.solve(gram, x.T @ (y - intercept) / len(x))
    return center, scale, intercept, coefficients


def predict_ridge(model, test, features):
    center, scale, intercept, coefficients = model
    return intercept + (test[features].to_numpy() - center) / scale @ coefficients


def split_at(panel, start, end=None):
    train = panel.loc[(panel["date"] < start) & (panel["label_end"] < start)]
    test = panel.loc[panel["date"] >= start]
    if end is not None:
        test = test.loc[(test["date"] < end) & (test["label_end"] < end)]
    if train.empty or test.empty:
        raise ValueError(f"No eligible training or evaluation rows for the split at {start.date()}.")
    return train, test


def daily_metrics(test, forecast):
    frame = test[["date", "basket_id", TARGET]].assign(forecast=forecast)
    grouped = frame.groupby("date", sort=True)
    ranks = grouped[["forecast", TARGET]].rank(method="average")
    centered = ranks - ranks.groupby(frame["date"]).transform("mean")
    forecast_rank = centered["forecast"]
    target_rank = centered[TARGET]
    date = frame["date"]
    covariance = (forecast_rank * target_rank).groupby(date).sum()
    denominator = np.sqrt(
        forecast_rank.pow(2).groupby(date).sum() * target_rank.pow(2).groupby(date).sum()
    )
    ic = covariance.div(denominator.where(denominator > 0))
    count = grouped.size()
    top_two = (
        frame.sort_values(["date", "forecast", "basket_id"], ascending=[True, False, True])
        .groupby("date", sort=True).head(2).groupby("date", sort=True)[TARGET].mean()
    )
    return pd.DataFrame({"rank_ic": ic, "top_two_excess": top_two, "baskets": count}).loc[
        count.ge(3) & ic.notna()
    ]


def validation_metrics(folds, features, penalty):
    return [
        daily_metrics(test, predict_ridge(fit_ridge(train, features, penalty), test, features))
        for train, test in folds
    ]


def tune(folds, features):
    scores = {}
    by_penalty = {}
    for penalty in LAMBDAS:
        by_penalty[penalty] = validation_metrics(folds, features, penalty)
        if all(not frame.empty for frame in by_penalty[penalty]):
            scores[penalty] = float(np.mean([
                frame["rank_ic"].mean() for frame in by_penalty[penalty]
            ]))
    if not scores:
        raise ValueError("No lambda produced defined Rank IC in every validation fold.")
    # Prefer the smaller penalty when scores are exactly tied.
    chosen = max(scores, key=scores.get)
    return chosen, scores, by_penalty[chosen]


def pair(base, added):
    joined = base.join(added, how="inner", lsuffix="_base", rsuffix="_persistence")
    if not joined["baskets_base"].equals(joined["baskets_persistence"]):
        raise ValueError("Both specifications must evaluate the same stock baskets.")
    joined["delta_rank_ic"] = joined["rank_ic_persistence"] - joined["rank_ic_base"]
    joined["delta_top_two_excess"] = (
        joined["top_two_excess_persistence"] - joined["top_two_excess_base"]
    )
    return joined


def block_interval(daily, calendar, block=20, repetitions=2_000):
    # Reindex first so a block spans trading sessions even when its Rank IC is undefined.
    sample = daily.reindex(calendar).to_numpy()
    if np.isfinite(sample).sum() < block:
        return None
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(sample) - block + 1, size=(repetitions, int(np.ceil(len(sample) / block))))
    positions = starts[:, :, None] + np.arange(block)
    values = sample[positions.reshape(repetitions, -1)[:, :len(sample)]]
    counts = np.isfinite(values).sum(axis=1)
    means = np.nansum(values[counts > 0], axis=1) / counts[counts > 0]
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize(daily, calendar):
    if daily.empty:
        return {"paired_dates": 0, "delta_rank_ic": None, "delta_rank_ic_ci95": None,
                "delta_top_two_excess": None}
    return {
        "paired_dates": len(daily),
        "delta_rank_ic": float(daily["delta_rank_ic"].mean()),
        "delta_rank_ic_ci95": block_interval(daily["delta_rank_ic"], calendar),
        "delta_top_two_excess": float(daily["delta_top_two_excess"].mean()),
    }


def evaluate(panel_path: Path, coverage_path: Path, output: Path, include_holdout=False):
    panel = pd.read_csv(panel_path, parse_dates=["date", "label_end"])
    coverage = pd.read_csv(coverage_path, parse_dates=["date"])
    fields = [*WITH_PERSISTENCE, TARGET]
    panel = panel.loc[np.isfinite(panel[fields]).all(axis=1) & panel["label_end"].notna()]
    if panel.duplicated(["date", "basket_id"]).any():
        raise ValueError("Each basket must appear at most once per session.")
    development = panel.loc[panel["label_end"] < HOLDOUT_START]
    folds = [
        split_at(development, pd.Timestamp(year, 1, 1), pd.Timestamp(year + 2, 1, 1))
        for year in VALIDATION_YEARS
    ]
    base_lambda, base_scores, base_validation = tune(folds, BASE)
    added_lambda, added_scores, added_validation = tune(folds, WITH_PERSISTENCE)
    summary = {
        "lambda_base": base_lambda, "lambda_persistence": added_lambda,
        "validation_scores_base": base_scores, "validation_scores_persistence": added_scores,
        "validation_blocks": [],
    }
    calendar = pd.DatetimeIndex(coverage["date"].drop_duplicates().sort_values())
    validation_days = []
    for year, base, added in zip(VALIDATION_YEARS, base_validation, added_validation, strict=True):
        daily = pair(base, added)
        sessions = calendar[(calendar.year >= year) & (calendar.year < year + 2)]
        summary["validation_blocks"].append({"start_year": year, **summarize(daily, sessions)})
        validation_days.append(daily.assign(start_year=year))

    output.mkdir(parents=True, exist_ok=True)
    pd.concat(validation_days).to_csv(output / "validation_daily.csv", index_label="date")
    if include_holdout:
        train, test = split_at(panel, HOLDOUT_START)
        base = daily_metrics(test, predict_ridge(fit_ridge(train, BASE, base_lambda), test, BASE))
        added = daily_metrics(test, predict_ridge(
            fit_ridge(train, WITH_PERSISTENCE, added_lambda), test, WITH_PERSISTENCE
        ))
        daily = pair(base, added)
        end = test["date"].max()
        summary["holdout"] = summarize(daily, calendar[(calendar >= HOLDOUT_START) & (calendar <= end)])
        positive_blocks = sum((fold["delta_rank_ic"] or 0) > 0 for fold in summary["validation_blocks"])
        interval = summary["holdout"]["delta_rank_ic_ci95"]
        if interval is None or interval[0] <= 0 <= interval[1]:
            decision = "inconclusive"
        elif interval[0] > 0 and summary["holdout"]["delta_rank_ic"] > 0 and positive_blocks > len(folds) / 2:
            decision = "supports_retaining_persistence"
        else:
            decision = "retention_criterion_not_met"
        summary["holdout"]["decision"] = decision
        daily.to_csv(output / "holdout_daily.csv", index_label="date")
    filename = "holdout_summary.json" if include_holdout else "validation_summary.json"
    (output / filename).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

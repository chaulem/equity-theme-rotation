"""Test a 21-session momentum acceleration signal in the weight-decayed MLP."""

import argparse
import json
from os import environ
from pathlib import Path

environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from baskets import read_wide_csv
from evaluate import HOLDOUT_START, TARGET, VALIDATION_YEARS, WITH_PERSISTENCE, block_interval, daily_metrics, split_at

SEEDS = (42, 43, 44)
WEIGHT_DECAY = 0.1
LEARNING_RATE = 0.001
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 10
HIDDEN_UNITS = 8
WINDOW = 21
RECENT = 7
ACCELERATION = "momentum_acceleration"


def acceleration_signal(panel, prices):
    prices = prices.where(np.isfinite(prices) & prices.gt(0))
    returns = prices.pct_change(fill_method=None).to_numpy()
    ticker_index = {ticker: i for i, ticker in enumerate(prices.columns)}
    spy = ticker_index["SPY"]
    sessions = prices.index.get_indexer(panel["date"])
    if (sessions < WINDOW).any():
        raise ValueError("Each basket needs a matching price date with 21 prior daily returns.")

    scores = []
    for end, members in zip(sessions, panel["members"], strict=True):
        columns = [ticker_index[ticker] for ticker in members.split("|")]
        window = returns[end - WINDOW + 1:end + 1]
        excess = np.log1p(window[:, columns].mean(axis=1)) - np.log1p(window[:, spy])
        scores.append(excess[-RECENT:].mean() - excess[:-RECENT].mean())
    if not np.isfinite(scores).all():
        raise ValueError("The 21-session signal could not be computed for all baskets.")
    return scores


def forward(x, weights):
    w1, b1, w2, b2 = weights
    hidden = np.maximum(0, x @ w1 + b1)
    return hidden, hidden @ w2 + b2


def gradient(x, y, weights):
    hidden, prediction = forward(x, weights)
    output = 2 * (prediction - y) / len(y)
    hidden_gradient = output[:, None] * weights[2] * (hidden > 0)
    return x.T @ hidden_gradient, hidden_gradient.sum(axis=0), hidden.T @ output, output.sum()


def fit_mlp(train, features, seed, epochs, validation=None):
    x = train[features].to_numpy()
    y = train[TARGET].to_numpy()
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    x_std[x_std == 0] = 1
    y_mean, y_std = y.mean(), y.std() or 1.0
    x, y = (x - x_mean) / x_std, (y - y_mean) / y_std
    if validation is not None:
        x_valid = (validation[features].to_numpy() - x_mean) / x_std
        y_valid = (validation[TARGET].to_numpy() - y_mean) / y_std

    rng = np.random.default_rng(seed)
    bound = 1 / np.sqrt(HIDDEN_UNITS)
    weights = [
        rng.uniform(-.5, .5, (x.shape[1], HIDDEN_UNITS)), rng.uniform(-.5, .5, HIDDEN_UNITS),
        rng.uniform(-bound, bound, HIDDEN_UNITS), np.array(0.),
    ]
    first = [np.zeros_like(weight) for weight in weights]
    second = [np.zeros_like(weight) for weight in weights]
    best_epoch, best_error, step = epochs, np.inf, 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            step += 1
            for index, grad in enumerate(gradient(x[batch], y[batch], weights)):
                first[index] = .9 * first[index] + .1 * grad
                second[index] = .999 * second[index] + .001 * grad ** 2
                mean = first[index] / (1 - .9 ** step)
                variance = second[index] / (1 - .999 ** step)
                # Decoupled decay applies to the two weight matrices and leaves biases free.
                if index in (0, 2):
                    weights[index] *= 1 - LEARNING_RATE * WEIGHT_DECAY
                weights[index] -= LEARNING_RATE * mean / (np.sqrt(variance) + 1e-8)
        if validation is not None:
            error = np.mean((forward(x_valid, weights)[1] - y_valid) ** 2)
            if error < best_error:
                best_epoch, best_error = epoch, error
            if epoch - best_epoch >= PATIENCE:
                break
    return (weights, x_mean, x_std, y_mean, y_std), best_epoch


def score_panel(panel):
    alternatives = {"strength_10": WITH_PERSISTENCE, "acceleration_7_14": [ACCELERATION, *WITH_PERSISTENCE[1:]]}
    blocks = []
    for year in VALIDATION_YEARS:
        train, test = split_at(panel, pd.Timestamp(year, 1, 1), pd.Timestamp(year + 2, 1, 1))
        cv_start = train["date"].drop_duplicates().sort_values().iloc[-252]
        inner_train, inner_valid = split_at(train, cv_start)
        scores = {}
        for name, features in alternatives.items():
            seeds = []
            for seed in SEEDS:
                _, epoch = fit_mlp(inner_train, features, seed, MAX_EPOCHS, validation=inner_valid)
                model, _ = fit_mlp(train, features, seed, epoch)
                weights, x_mean, x_std, y_mean, y_std = model
                x = (test[features].to_numpy() - x_mean) / x_std
                forecast = forward(x, weights)[1] * y_std + y_mean
                seeds.append(daily_metrics(test, forecast))
            per_seed = pd.concat(seeds, keys=SEEDS, axis=1, join="inner")
            scores[name] = per_seed.T.groupby(level=1, sort=False).mean().T

        daily = pd.concat(scores, axis=1, join="inner")
        daily[("meta", "candidate_mean")] = test.groupby("date")[TARGET].mean()
        daily[("meta", "fold")] = year
        blocks.append(daily)
    return pd.concat(blocks)


def summarize(daily, calendar):
    summary = {"sessions": len(daily)}
    for model in ("strength_10", "acceleration_7_14"):
        rank = daily[model]["rank_ic"]
        edge = 1e4 * (daily[model]["top_two_excess"] - daily[("meta", "candidate_mean")])
        summary[model] = {"rank_ic": float(rank.mean()), "top_two_gain_bps": float(edge.mean())}
    rank = daily["acceleration_7_14"]["rank_ic"] - daily["strength_10"]["rank_ic"]
    top_two = 1e4 * (daily["acceleration_7_14"]["top_two_excess"] - daily["strength_10"]["top_two_excess"])
    summary["delta_rank_ic_ci95"] = block_interval(rank, calendar)
    summary["delta_top_two_bps_ci95"] = block_interval(top_two, calendar)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", required=True, type=Path)
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/momentum_acceleration"))
    args = parser.parse_args()

    panel = pd.read_csv(args.input_dir / "basket_panel.csv", parse_dates=["date", "label_end"])
    coverage = pd.read_csv(args.input_dir / "coverage.csv", parse_dates=["date"])
    panel = panel.loc[panel["label_end"].lt(HOLDOUT_START)].dropna(subset=[*WITH_PERSISTENCE, TARGET])
    prices = read_wide_csv(args.prices).loc[:panel["date"].max()]
    panel[ACCELERATION] = acceleration_signal(panel, prices)
    calendar = pd.DatetimeIndex(coverage.loc[
        (coverage["date"].dt.year >= VALIDATION_YEARS[0]) & coverage["date"].lt(HOLDOUT_START), "date",
    ])
    daily = score_panel(panel)
    summary = {
        "weight_decay": WEIGHT_DECAY,
        "seeds": SEEDS,
        "validation": summarize(daily, calendar),
        "folds": {
            int(year): summarize(block, calendar[(calendar.year >= year) & (calendar.year < year + 2)])
            for year, block in daily.groupby(("meta", "fold"))
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output / "daily.csv", index_label="date")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

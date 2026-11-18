"""Build dated stock baskets, features, and outcomes for Experiment 01."""

from pathlib import Path

import numpy as np
import pandas as pd

LOOKBACK = 60
FEATURE_WINDOW = 10
HORIZON = 5
CLUSTERS = 10
MIN_MEMBERS = 3


def read_wide_csv(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path, index_col="date", parse_dates=["date"])
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError(f"Invalid session dates in {path.name}")
    if panel.index.hasnans or not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError(f"Dates must be valid, unique, and increasing in {path.name}")
    return panel.astype(float)


def load_inputs(price_path: Path, volume_path: Path):
    prices = read_wide_csv(price_path)
    volumes = read_wide_csv(volume_path)
    if not prices.index.equals(volumes.index):
        raise ValueError("Price and volume inputs must share the same ordered session calendar.")
    if "SPY" not in prices:
        raise ValueError("The price input must contain SPY.")
    tickers = prices.columns.drop("SPY").sort_values()
    if not tickers.isin(volumes.columns).all():
        raise ValueError("The volume input is missing one or more stock columns.")
    prices = prices[["SPY", *tickers]].where(lambda x: np.isfinite(x) & x.gt(0))
    volumes = volumes[tickers].where(lambda x: np.isfinite(x) & x.ge(0))
    return prices, volumes


def average_linkage_labels(distances: np.ndarray, clusters: int) -> np.ndarray:
    """Cut an average-linkage tree using a nearest-neighbor chain."""
    size = len(distances)
    if not 1 <= clusters <= size:
        raise ValueError("The cluster count must be between one and the number of stocks.")

    distances = distances.copy()
    np.fill_diagonal(distances, np.inf)
    weights = np.ones(size, dtype=int)
    chain = []
    merges = []
    for _ in range(size - 1):
        if not chain:
            chain.append(int(np.flatnonzero(weights)[0]))
        while True:
            stock = chain[-1]
            neighbor = int(distances[stock].argmin())
            if len(chain) > 1:
                previous = chain[-2]
                if distances[stock, previous] <= distances[stock, neighbor]:
                    neighbor = previous
                if neighbor == previous:
                    del chain[-2:]
                    break
            chain.append(neighbor)

        # Keep the lower index as the cluster's representative.
        left, right = sorted((stock, neighbor))
        merges.append((distances[left, right], left, right))
        total = weights[left] + weights[right]
        weights[right] = 0
        remaining = np.flatnonzero(weights)
        remaining = remaining[remaining != left]
        distance = (
            weights[left] * distances[left, remaining]
            + (total - weights[left]) * distances[right, remaining]
        ) / total
        distances[left, remaining] = distance
        distances[remaining, left] = distance
        distances[right, :] = np.inf
        distances[:, right] = np.inf
        weights[left] = total

    # Independent merges can be discovered out of order by the chain.
    merges.sort(key=lambda merge: merge[0])
    parents = np.arange(size)

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for _, left, right in merges[:size - clusters]:
        parents[root(right)] = root(left)
    return np.unique([root(i) for i in range(size)], return_inverse=True)[1]


def build_panel(prices, volumes, start=None, through=None):
    """Build point-in-time features and separate five-session labels."""
    if through is not None:
        prices = prices.loc[:through]
        volumes = volumes.loc[:through]
    dates = prices.index
    returns = prices.pct_change(fill_method=None).to_numpy()
    adjusted_close = prices.to_numpy()
    tickers = prices.columns.to_numpy()[1:]
    volume_baseline = volumes.shift(1).rolling(FEATURE_WINDOW).median()
    volume_ratio = volumes.div(volume_baseline.where(volume_baseline.gt(0))).to_numpy()

    records = []
    coverage = []
    for end in range(LOOKBACK, len(dates)):
        date = dates[end]
        if start is not None and date < start:
            continue
        history = returns[end - LOOKBACK + 1:end + 1, 1:]
        window = slice(end - FEATURE_WINDOW + 1, end + 1)
        spy_history = returns[window, 0]
        ratios = volume_ratio[window]
        eligible = np.isfinite(history).all(axis=0) & np.isfinite(ratios).all(axis=0)
        columns = np.flatnonzero(eligible)
        centered = history[:, columns] - history[:, columns].mean(axis=0)
        norm = np.linalg.norm(centered, axis=0)
        columns = columns[norm > 0]
        daily_coverage = {
            "date": date, "eligible_stocks": len(columns), "retained_baskets": 0,
            "small_baskets": 0, "missing_outcomes": 0, "truncated_outcomes": 0,
            "scorable_baskets": 0,
        }
        coverage.append(daily_coverage)
        if len(columns) < CLUSTERS or not np.isfinite(spy_history).all():
            continue

        normalized = centered[:, norm > 0] / norm[norm > 0]
        distances = np.clip(1 - (normalized.T @ normalized), 0, 2)
        groups = average_linkage_labels(distances, CLUSTERS)
        last = end + HORIZON
        for group in range(CLUSTERS):
            members = columns[groups == group]
            if len(members) < MIN_MEMBERS:
                daily_coverage["small_baskets"] += 1
                continue
            daily_coverage["retained_baskets"] += 1

            member_return = returns[window, 1:][:, members]
            basket_return = member_return.mean(axis=1)
            strength = np.prod(1 + basket_return) - np.prod(1 + spy_history)
            outcome = np.nan
            label_end = pd.NaT
            if last >= len(dates):
                daily_coverage["truncated_outcomes"] += 1
            else:
                label_end = dates[last]
                prices_at_exit = adjusted_close[last, np.r_[0, members + 1]]
                if not np.isfinite(prices_at_exit).all():
                    daily_coverage["missing_outcomes"] += 1
                else:
                    gains = prices_at_exit / adjusted_close[end, np.r_[0, members + 1]] - 1
                    outcome = gains[1:].mean() - gains[0]
                    daily_coverage["scorable_baskets"] += 1

            records.append({
                "date": date, "basket_id": group, "size": len(members),
                "members": "|".join(tickers[members]), "relative_strength": strength,
                "breadth": (member_return > 0).mean(),
                "unusual_volume": np.median(ratios[:, members]),
                "persistence": (basket_return > spy_history).mean(),
                "label_end": label_end, "excess_return_5d": outcome,
            })
    if not records:
        raise ValueError("No baskets were formed. Check the dates and source coverage.")
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(coverage)

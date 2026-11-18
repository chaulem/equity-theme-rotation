# Forecasting Equity Theme Rotation with Data-Driven Stock Baskets

This project studies how to form equity baskets from historical co-movement and forecast their relative performance. It evaluates out-of-sample forecasts before considering execution and trading costs.

The [research pipeline and work board](PIPELINE.md) tracks feature decisions, experiment results, and open tasks before holdout evaluation.

## Experiment 01 — Momentum persistence

Does the fraction of recent sessions in which a basket outperforms SPY improve its five-session return ranking? Both Ridge models use the same baskets, relative strength, breadth, and unusual volume. The second model adds the fraction of the last 10 sessions in which the basket outperformed SPY.

- Rebuild average-linkage hierarchical clusters at each close from 60 sessions of returns. Cut the tree into 10 clusters and keep clusters with at least three members.
- Measure each outcome using the members known on the forecast date. Begin equally weighted at that close, hold shares for five sessions, and subtract the matched SPY return.
- Fit and tune on expanding walk-forward folds from 2015–2022. Purge outcomes that extend into the next split. Keep 2023 onward for the final holdout.
- Compare paired daily Rank IC and top-two realized excess returns. Resample paired differences in 20-session blocks for the Rank IC confidence interval.

The adjusted-close target measures forecast quality. The project will test next-session execution and costs separately. The Stage 2.7 dataset uses a current constituent list, which limits inference about the historical S&P 500 universe.

## Run the experiment

Use Python 3.10 or later. Supply the BAB Stage 2.7 price and volume CSVs locally. Each CSV has a `date` column and one column per ticker. Prices are adjusted closes and must include SPY. The files must share the same ordered trading calendar. Raw inputs and generated results are kept out of Git.

```sh
python -m pip install -r requirements.txt
python experiment01.py build --prices data/stage27_prices.csv --volume data/stage27_volume.csv --through 2022-12-31
python experiment01.py evaluate
```

`basket_panel.csv` records membership, features, and outcomes. `coverage.csv` records stock eligibility, small clusters, and missing or unavailable labels. The validation run writes daily paired scores and a summary.

After freezing the specification, rebuild without `--through` and run `python experiment01.py evaluate --holdout` to evaluate the final period.

## Basket formation comparison

```sh
python compare_baskets.py --prices data/stage27_prices.csv --volume data/stage27_volume.csv
```

PCA keeps five unwhitened components of the same normalized 60-session return histories used by hierarchical clustering. K-means uses five seeded starts and requests ten clusters. Both methods keep baskets with at least three names and refit Ridge with the persistence features and the same penalty (`100`). The comparison reports basket sizes, daily Rank IC, and realized top-two returns on common validation dates through 2022. The top-two selection gain uses that method's equally weighted candidate baskets as its benchmark.

## Later experiments

Compare basket formation with fixed industry groups. Test options-volume features and reversal filters in separate comparisons. Study ETF execution only after evaluating the forecasts.

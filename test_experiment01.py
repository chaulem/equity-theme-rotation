import unittest

import numpy as np
import pandas as pd

from baskets import average_linkage_labels, build_panel
from evaluate import split_at


class ExperimentTests(unittest.TestCase):
    def test_cut_orders_independent_merges_by_distance(self):
        # The first chain reaches A-B, but C-D must merge first at a three-cluster cut.
        distances = np.array([[0, .3, .8, .8], [.3, 0, .8, .8],
                              [.8, .8, 0, .1], [.8, .8, .1, 0]])
        labels = average_linkage_labels(distances, 3)
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[1])

    def test_future_prices_change_labels_without_changing_features(self):
        rng = np.random.default_rng(3)
        dates = pd.bdate_range("2020-01-01", periods=70)
        members = [f"STOCK{i:02d}" for i in range(30)]
        group_returns = rng.normal(0, .01, (len(dates), 10))
        stock_returns = np.repeat(group_returns, 3, axis=1)
        spy_returns = rng.normal(0, .01, (len(dates), 1))
        prices = pd.DataFrame(
            100 * np.cumprod(1 + np.hstack([spy_returns, stock_returns]), axis=0),
            index=dates, columns=["SPY", *members],
        )
        volumes = pd.DataFrame(5_000, index=dates, columns=members)
        as_of = dates[60]
        before, _ = build_panel(prices, volumes, start=as_of)
        prices.loc[dates[61]:, members] *= 1.03
        after, _ = build_panel(prices, volumes, start=as_of)
        before = before.loc[before["date"] == as_of]
        after = after.loc[after["date"] == as_of]
        self.assertEqual(len(before), 10)
        pd.testing.assert_frame_equal(
            before.drop(columns="excess_return_5d"), after.drop(columns="excess_return_5d")
        )
        self.assertTrue(after["excess_return_5d"].gt(before["excess_return_5d"]).all())

    def test_fold_excludes_label_ending_on_first_validation_date(self):
        dates = pd.bdate_range("2022-01-03", periods=12)
        frame = pd.DataFrame({"date": dates[:7], "label_end": dates[5:]})
        train, test = split_at(frame, dates[6])
        self.assertEqual(train["date"].tolist(), [dates[0]])
        self.assertEqual(test["date"].tolist(), [dates[6]])


if __name__ == "__main__":
    unittest.main()

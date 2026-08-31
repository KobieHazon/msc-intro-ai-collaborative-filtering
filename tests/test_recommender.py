import numpy as np
import pandas as pd
import pytest

from collaborative_filtering import Recommender


@pytest.fixture
def ratings():
    return pd.DataFrame(
        [
            [5.0, 4.0, np.nan, 1.0],
            [4.0, np.nan, 2.0, 1.0],
            [1.0, 2.0, 5.0, np.nan],
        ],
        index=["user-a", "user-b", "user-c"],
        columns=["item-1", "item-2", "item-3", "item-4"],
    )


@pytest.mark.parametrize("strategy", ["user", "item"])
def test_fit_produces_predictions_for_each_strategy(ratings, strategy):
    recommender = Recommender(strategy).fit(ratings)

    assert recommender.pred.shape == ratings.shape
    assert list(recommender.pred.index) == list(ratings.index)
    assert list(recommender.pred.columns) == list(ratings.columns)


def test_recommendations_exclude_items_already_rated(ratings):
    recommender = Recommender("user").fit(ratings)

    recommendations = set(recommender.recommend_items("user-a", k=3))

    assert recommendations == {"item-3"}


def test_unknown_user_returns_none(ratings):
    recommender = Recommender().fit(ratings)

    assert recommender.recommend_items("missing") is None


def test_invalid_strategy_is_rejected():
    with pytest.raises(ValueError, match="strategy"):
        Recommender("invalid")

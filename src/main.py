from pathlib import Path

import pandas as pd

from collaborative_filtering import Recommender
from data import print_data, watch_data_info
from evaluation import RMSE, precision_at_k, recall_at_k

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


if __name__ == "__main__":
    dataset = pd.read_csv(DATA_DIR / "train.csv")
    test_set = pd.read_csv(DATA_DIR / "test.csv")

    # PART 1 - DATA
    watch_data_info(dataset)
    print_data(dataset)

    user_item_matrix_raw = pd.pivot_table(
        dataset, index="UserId", columns="ProductId", values="Rating", aggfunc="sum"
    )
    print(user_item_matrix_raw.head())

    # PATR 2 - COLLABORATING FILLTERING RECOMMENDATION SYSTEM
    recommender_user = Recommender().fit(user_item_matrix_raw)
    recommender_item = Recommender("item").fit(user_item_matrix_raw)

    print(recommender_user.recommend_items("AQWF3BBBDL4QJ"))
    print(recommender_item.recommend_items("A3EO0WA7R3LVBQ"))

    # PART 3 - EVALUATION
    RMSE(test_set, recommender_user)
    RMSE(test_set, recommender_item)
    precision_at_k(test_set, recommender_user, 20)
    recall_at_k(test_set, recommender_user, 20)

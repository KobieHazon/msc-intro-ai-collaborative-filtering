"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Author: Kobie Hazon

"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


class Recommender:
    def __init__(self, strategy="user"):
        if strategy not in {"user", "item"}:
            raise ValueError("strategy must be 'user' or 'item'")
        self.strategy = strategy
        self.similarity = np.nan
        self.user_item_matrix = None
        self.pred = None

    def fit(self, matrix):
        """
        fits training data into recommendation system.
        it uses the collaborative filtering algorithm that is user or item based.
        """
        self.user_item_matrix = matrix
        if self.strategy == "user":
            # User - User based collaborative filtering
            start_time = time.time()
            mean_user_ratings = self.user_item_matrix.mean(axis=1).to_numpy().reshape(-1, 1)
            normalized_user_ratings = self.get_normalized_user_item_matrix(mean_user_ratings)
            user_similarity = self._cosine_similarity(normalized_user_ratings)
            pred = (
                mean_user_ratings
                + user_similarity.dot(normalized_user_ratings)
                / np.array([np.abs(user_similarity).sum(axis=1)]).T
            )
            self.pred = pd.DataFrame(
                pred.round(2),
                index=self.user_item_matrix.index.values,
                columns=self.user_item_matrix.columns.values,
            )
            # self.pred should contain your prediction metrix.

            time_taken = time.time() - start_time
            print("User Model in {} seconds".format(time_taken))
            return self

        else:
            # Item - Item based collaborative filtering
            start_time = time.time()
            mean_user_ratings = self.user_item_matrix.mean(axis=1).to_numpy().reshape(-1, 1)
            normalized_user_ratings = self.get_normalized_user_item_matrix(mean_user_ratings)
            item_similarity = self._cosine_similarity(normalized_user_ratings.T)
            pred = mean_user_ratings + normalized_user_ratings.dot(item_similarity) / np.array(
                [np.abs(item_similarity).sum(axis=1)]
            )
            self.pred = pd.DataFrame(
                pred.round(2),
                index=self.user_item_matrix.index.values,
                columns=self.user_item_matrix.columns.values,
            )
            # self.pred should contain your prediction metrix.

            time_taken = time.time() - start_time
            print("Item Model in {} seconds".format(time_taken))
            return self

    @staticmethod
    def _cosine_similarity(matrix):
        # Some Accelerate-backed NumPy builds emit spurious matmul warnings for
        # this large calculation even when every result is finite.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            similarity = 1 - pairwise_distances(matrix, metric="cosine")
        if not np.isfinite(similarity).all():
            raise FloatingPointError("cosine similarity produced non-finite values")
        return similarity

    def recommend_items(self, user_id, k=5):
        """
        recommends k items for user <user_id> based on the collaborative filtering algorithm chosen.
        the training data should be fitted beforehand using the fit method.
        """

        if self.user_item_matrix is None:
            raise Exception("can't recommend items before recommender is fitted with data")
        if user_id not in self.user_item_matrix.index:
            return None

        user_predicted_ratings = self.pred.loc[user_id].copy()
        user_matrix_row = self.user_item_matrix.loc[user_id]
        user_predicted_ratings[~np.isnan(user_matrix_row)] = 0
        user_predicted_ratings_unrated = user_predicted_ratings[np.isnan(user_matrix_row)]

        sorted_user_predicted_ratings = user_predicted_ratings_unrated.sort_values(
            ascending=False, kind="stable"
        )
        user_recommended_items = sorted_user_predicted_ratings[:k]

        return user_recommended_items.index.values

    def get_normalized_user_item_matrix(self, mean_user_ratings):
        np_user_item_matrix = self.user_item_matrix.to_numpy()
        user_ratings_diff = np_user_item_matrix - mean_user_ratings + 0.001
        user_ratings_diff[np.isnan(user_ratings_diff)] = 0
        return user_ratings_diff

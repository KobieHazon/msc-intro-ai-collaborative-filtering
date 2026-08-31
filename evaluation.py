"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Author: Kobie Hazon

"""
from statistics import mean
from typing import Set

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import mean_squared_error


def get_rmse_benchmark(user_item_matrix: DataFrame, user_id: str) -> float:
    """
    returns the benchmark recommendation model for the RMSE evaluation.
    the benchmark essentially returns the mean ratings for user <user_id> as his predicted rating.
    """
    mean_user_ratings = user_item_matrix.mean(axis=1)
    return mean_user_ratings[user_id]


def RMSE(test_set, cf):
    """
    returns rmse evaluation of collaborative-filtering model cf using test_set.
    """
    true_ratings = test_set.sort_values(by=['UserId', 'ProductId'])['Rating']
    # sort the test set to recover the rating order after getting the predictions ratings
    test_user_item_mask_matrix = pd.pivot_table(test_set, index='UserId',
                                                columns='ProductId', values='Rating', aggfunc=lambda _: 1)
    # create mask matrix for the user-item pairs according to the test_set ratings

    test_predictions = test_user_item_mask_matrix.multiply(cf.pred, fill_value=0)
    # multiply to get matrix with prediction ratings only on test_set pairs
    test_predictions_set = test_predictions.reset_index().\
        melt(id_vars='UserId', value_name='Rating').\
        query('Rating != 0').\
        sort_values(by=['UserId', 'ProductId'])
    # return the matrix with test_set predictions to the dataset format
    prediction_ratings = test_predictions_set['Rating']
    recommender_mse = mean_squared_error(true_ratings, prediction_ratings)
    # calculating the appropriate prediction_ratings is faster than a loop like I did for the benchmark :)
    recommender_rmse = np.sqrt(recommender_mse)
    print(f"RMSE for cf-recommender ({cf.strategy} based): {round(recommender_rmse, 5)}")

    user_mean_ratings = cf.user_item_matrix.mean(axis=1)  # calculate rating for each user
    benchmark_ratings = [user_mean_ratings[test_case[1]['UserId']] for test_case in test_set.iterrows()]
    # get the benchmark ratings, the mean for each user in the same order as the test_set
    benchmark_mse = mean_squared_error(test_set['Rating'], benchmark_ratings)
    benchmark_rmse = np.sqrt(benchmark_mse)
    print(f"RMSE for benchmark: {round(benchmark_rmse, 5)}")


def get_relevant_items(user_ratings: Series, threshold: float = 3) -> Set[str]:
    """
    returns relevant items for recommendation system recommendations.
    according to the exercise, it is items that the user rated more than threshold.
    """
    relevant_items = user_ratings.loc[lambda rating: rating >= threshold]
    return set(relevant_items.index)


def get_precision_recall_benchmark(user_item_matrix: DataFrame, k: int) -> Set[str]:
    """
    returns the benchmark model for the precision and recall evaluation.
    the benchmark uses the mean ratings for each product to always the k top rated products.
    """
    product_mean_rating = user_item_matrix.mean(axis=0)
    sorted_product_mean_rating = product_mean_rating.sort_values(ascending=False, kind='stable')
    benchmark_product_recommended_items = set(sorted_product_mean_rating.index.values[:k])
    return benchmark_product_recommended_items


def calculate_precision(recommender_items: Set[str], relevant_items: Set[str], k: int) -> float:
    """
    returns precision of recommender.
    receives the recommended items and the relevant items, and the number of items we recommend.
    """
    relevant_items_recommended = relevant_items.intersection(recommender_items)
    precision_value = len(relevant_items_recommended) / k
    return precision_value


def precision_at_k(test_set, cf, k):
    """
    returns the precision value of cf model on test_set with k items to recommend.
    """
    benchmark_user_precisions = []
    recommender_user_precisions = []
    test_user_item_matrix = pd.pivot_table(test_set, index='UserId',
                                           columns='ProductId', values='Rating', aggfunc=np.sum)
    training_user_item_matrix = cf.user_item_matrix
    benchmark_items = get_precision_recall_benchmark(training_user_item_matrix, k)

    for user_id, test_user_ratings in test_user_item_matrix.iterrows():
        test_user_relevant_items = get_relevant_items(test_user_ratings)
        if len(test_user_relevant_items) == 0:
            continue
        user_recommended_items = set(cf.recommend_items(user_id, k))
        user_precision = calculate_precision(user_recommended_items, test_user_relevant_items, k)
        recommender_user_precisions.append(user_precision)

        benchmark_precision = calculate_precision(benchmark_items, test_user_relevant_items, k)
        benchmark_user_precisions.append(benchmark_precision)

    mean_recommender_user_precisions = mean(recommender_user_precisions)
    print(f"precision@{k} for cf-recommender ({cf.strategy} based): {round(mean_recommender_user_precisions, 5)}")
    mean_benchmark_user_precisions = mean(benchmark_user_precisions)
    print(f"precision@{k} for benchmark: {round(mean_benchmark_user_precisions, 5)}")


def recall_at_k(test_set, cf, k):
    """
    returns the precision value of cf model on test_set with k items to recommend.
    """
    benchmark_user_recalls = []
    recommender_user_recalls = []
    test_user_item_matrix = pd.pivot_table(test_set, index='UserId',
                                           columns='ProductId', values='Rating', aggfunc=np.sum)
    training_user_item_matrix = cf.user_item_matrix
    benchmark_items = get_precision_recall_benchmark(training_user_item_matrix, k)

    for user_id, test_user_ratings in test_user_item_matrix.iterrows():
        test_user_relevant_items = get_relevant_items(test_user_ratings)
        if len(test_user_relevant_items) == 0:
            continue
        recommender_user_recommended_items = set(cf.recommend_items(user_id, k))
        recommender_user_relevant_items_recommended = test_user_relevant_items.intersection(
            recommender_user_recommended_items)
        recommender_user_recall = len(recommender_user_relevant_items_recommended) / len(test_user_relevant_items)
        recommender_user_recalls.append(recommender_user_recall)  # TODO: export loop to another function

        benchmark_user_relevant_items_recommended = test_user_relevant_items.intersection(benchmark_items)
        benchmark_user_precision = len(benchmark_user_relevant_items_recommended) / len(test_user_relevant_items)
        benchmark_user_recalls.append(benchmark_user_precision)

    mean_recommender_recalls = mean(recommender_user_recalls)
    print(f"recall@{k} for cf-recommender ({cf.strategy} based): {round(mean_recommender_recalls, 5)}")
    mean_benchmark_recalls = mean(benchmark_user_recalls)
    print(f"recall@{k} for benchmark: {round(mean_benchmark_recalls, 5)}")

"""
Introduction to Artificial Intelligence, 89570, Bar Ilan University, ISRAEL

Author: Kobie Hazon

"""
from collections import Counter


def watch_data_info(data):
    # This function returns the first 5 rows for the object based on position.
    # It is useful for quickly testing if your object has the right type of data in it.
    print(data.head())

    # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
    print(data.info())

    # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
    print(data.describe(include='all').transpose())


def print_data(data):
    """
    prints statistics about a dataset.
    The statistics are: number of users, numbers of products, number of rankings,
    minimum/maximum number of ratings given to a product and minimum/maximum number of products ratings by user.
    """
    user_counter = Counter(data["UserId"])
    product_counter = Counter(data["ProductId"])
    print(f"number of users are :  {len(user_counter)}")
    print(f"number of products ranked are : {len(product_counter)}")
    print(f"number of ranking are: {len(data)}")
    print(f"minimum number of ratings given to a product : {product_counter.most_common()[-1][1]}")
    print(f"maximum number of ratings given to a product : {product_counter.most_common(n=1)[0][1]}")
    print(f"minimum number of products ratings by user : {user_counter.most_common()[-1][1]}")
    print(f"maximum number of products ratings by user : {user_counter.most_common(n=1)[0][1]}")

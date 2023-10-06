from typing import List

import pandas as pd


def calc_exponential_features(
    df: pd.DataFrame,
    features_list: List,
    include_all_numeric: bool = False,
    number_of_powers: int = 3,
):
    """
    calculates exponential features for a given list of numeric features
    :df: dataframe
    :features_list: list of features to calculate exponential features
    :include_all_numeric: boolean, if True calculates exponential
                          features for all numeric columns in df
    :number_of_powers: how many powers we want to take
    returns inplace calculated features with postfix _power_{i}
    """

    final_features_list = df.select_dtypes(include='number')

    if not include_all_numeric:
        final_features_list = features_list

    for power in list(range(2, number_of_powers + 1)):
        for feature in final_features_list:
            df[f'{feature}_power_{power}'] = df[feature] ** power

    return df

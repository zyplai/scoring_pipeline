from typing import Tuple

import pandas as pd
from zypl_macro.library import DataGetter

from configs.config import settings
from utils.basic_utils import snake_case


def get_macro_data(country: str):

    getter_instance = DataGetter()
    getter_instance.auth(settings.ZYPL_MACRO.auth_token)

    daily = getter_instance.get_data(country=country, frequency='Daily')
    monthly = getter_instance.get_data(country=country, frequency='Monthly')
    quarterly = getter_instance.get_data(country=country, frequency='Quarterly')

    return daily, monthly, quarterly


def calc_macro_features(
    data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    num_of_lags: int = 3,
    window: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    calc moving averages, lags, returns etc.
    expand monthly and quarterly dates to join by days all
    Also, .shift(1) must be done here so we just merge by date
    """
    pass


def prepare_macro_features(
    country: str, num_of_lags: int = 3, window: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    macro_data = get_macro_data(settings.FEATURES_PARAMS.partners_country)

    # calc features
    # macro_features = calc_macro_features(
    #     macro_data,
    #     settings.FEATURES_PARAMS.num_of_lags,
    #     settings.FEATURES_PARAMS.window,
    # )

    daily, monthly, quarterly = macro_data
    daily = snake_case(daily)
    monthly = snake_case(monthly)
    quarterly = snake_case(quarterly)

    return daily, monthly, quarterly

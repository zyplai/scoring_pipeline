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
    daily, monthly, quarterly = data

    # Calculate moving averages
    daily['rolling_mean'] = daily['stock_market'].rolling(window=window).mean()

    # Calculate lags
    for lag in range(1, num_of_lags + 1):
        daily[f'lag_{lag}'] = daily['stock_market'].shift(lag)

    # Calculate returns
    daily['returns'] = daily['stock_market'].pct_change()
    
    # Expand monthly and quarterly dates
    expanded_monthly = monthly.reindex(daily.index, method='ffill')
    expanded_quarterly = quarterly.reindex(daily.index, method='ffill')

    daily = daily.dropna()
    daily = daily.drop(columns=['stock_market'], errors='ignore')

    return daily.copy(), expanded_monthly.copy(), expanded_quarterly.copy()



def prepare_macro_features(
    country: str, num_of_lags: int = 3, window: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    macro_data = get_macro_data(settings.FEATURES_PARAMS.partners_country)

    # calc features #
    macro_features = calc_macro_features(
        macro_data,
        settings.FEATURES_PARAMS.num_of_lags,
        settings.FEATURES_PARAMS.window,
    )

    daily, monthly, quarterly = macro_data

    print('#'*50)
    print( daily.info() )
    print('#'*50)
    print( monthly.info() )
    print('#'*50)
    print( quarterly.info() )
    print('#'*50)

    daily = snake_case(daily)
    monthly = snake_case(monthly)
    quarterly = snake_case(quarterly)

    return daily, monthly, quarterly

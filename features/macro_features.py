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
    """
    daily, monthly, quarterly = data

    # # Calculate moving averages
    # daily['rolling_mean'] = daily['stock_market'].rolling(window=window).mean()

    # # Calculate lags
    # for lag in range(1, num_of_lags + 1):
    #     daily[f'lag_{lag}'] = daily['stock_market'].shift(lag)

    # # Calculate returns
    # daily['returns'] = daily['stock_market'].pct_change()

    return daily.copy(), monthly.copy(), quarterly.copy()



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

    daily.drop('Country',axis=1,inplace=True)
    monthly.drop('Country',axis=1,inplace=True)
    quarterly.drop('Country',axis=1,inplace=True)

    monthly['Date'] = monthly['Date'].dt.to_period('M').dt.to_timestamp()
    quarterly['Date'] = quarterly['Date'].dt.to_period('M').dt.to_timestamp()

    print(daily.info())
    print(monthly.info())
    print( quarterly.info() )

    daily = snake_case(daily)
    monthly = snake_case(monthly)
    quarterly = snake_case(quarterly)

    return daily, monthly, quarterly
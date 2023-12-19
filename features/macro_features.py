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
    num_of_lags: int = 0,
    window: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    calc moving averages, lags, returns etc.
    """
    daily, monthly, quarterly = data

    #########################################################################
    macro_cols = daily.drop(['Date'],axis=1).columns

    # Calculate moving averages
    for col in macro_cols:
        col_name = 'rolling_mean_'+col
        daily[col_name] = daily[col].rolling(window=window).mean()
        
    # Calculate lags
    for col in macro_cols :
        for lag in range(1, num_of_lags + 1):
            col_name = col+'_lag_'+str(lag)
            daily[col_name] = daily[col].shift(lag)
            
    # Calculate returns #
    for col in macro_cols :
        col_name = col + '_pct_change'
        daily[col_name] = daily[col].pct_change()

    #########################################################################
    macro_cols = monthly.drop(['Date'],axis=1).columns

    # Calculate moving averages
    for col in macro_cols:
        col_name = 'rolling_mean_'+col
        monthly[col_name] = monthly[col].rolling(window=window).mean()
        
    # Calculate lags
    for col in macro_cols :
        for lag in range(1, num_of_lags + 1):
            col_name = col+'_lag_'+str(lag)
            monthly[col_name] = monthly[col].shift(lag)
            
    # Calculate returns #
    for col in macro_cols :
        col_name = col + '_pct_change'
        monthly[col_name] = monthly[col].pct_change()
    #########################################################################
    macro_cols = quarterly.drop(['Date'],axis=1).columns
    
    # Calculate moving averages
    for col in macro_cols:
        col_name = 'rolling_mean_'+col
        quarterly[col_name] = quarterly[col].rolling(window=window).mean()
        
    # Calculate lags
    for col in macro_cols :
        for lag in range(1, num_of_lags + 1):
            col_name = col+'_lag_'+str(lag)
            quarterly[col_name] = quarterly[col].shift(lag)
            
    # Calculate returns #
    for col in macro_cols :
        col_name = col + '_pct_change'
        quarterly[col_name] = quarterly[col].pct_change()
    #########################################################################

    return daily.copy(), monthly.copy(), quarterly.copy()



def prepare_macro_features(
    country: str, num_of_lags: int = 3, window: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    macro_data = get_macro_data(settings.FEATURES_PARAMS.partners_country)

    daily, monthly, quarterly = macro_data

    daily.drop('Country',axis=1,inplace=True)
    monthly.drop('Country',axis=1,inplace=True)
    quarterly.drop('Country',axis=1,inplace=True)

    # calc features #
    macro_features = calc_macro_features(
        macro_data,
        settings.FEATURES_PARAMS.num_of_lags,
        settings.FEATURES_PARAMS.window,
    )

    monthly['Date'] = monthly['Date'].dt.to_period('M').dt.to_timestamp()
    quarterly['Date'] = quarterly['Date'].dt.to_period('M').dt.to_timestamp()

    daily = snake_case(daily)
    monthly = snake_case(monthly)
    quarterly = snake_case(quarterly)

    return daily, monthly, quarterly
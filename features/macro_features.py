from typing import Tuple

from zypl_macro.library import DataGetter

from configs.config import settings


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
):
    """
    calc moving averages, lags, returns etc.
    """
    pass

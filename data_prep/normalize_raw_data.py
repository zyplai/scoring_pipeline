import pandas as pd

from configs.config import settings


def map_col_names(df: pd.DataFrame):
    df.rename(columns=settings.COLNAMES_MAPPER.mapper, inplace=True)


def define_target(
    df: pd.DataFrame,
    number_of_days: int = 90,
    cumulative_delays: str = 'cumulative_days_of_late_payments',
):
    df['target'] = (df[cumulative_delays] > number_of_days).astype(int)

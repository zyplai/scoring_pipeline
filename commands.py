import logging
import warnings
from datetime import datetime

import fire
import pandas as pd

from configs.config import settings
from data_prep.normalize_raw_data import define_target, map_col_names
from data_prep.sample_prep import prepare_main_sample
from features.cat_features import TargetMeanEncoder
from features.macro_features import prepare_macro_features
from features.sfa import SFA
from models.train import fit, predict
from utils.basic_utils import read_file

warnings.filterwarnings("ignore")


def preprocess_raw_sample():
    logging.info('--- Reading train sample and prepraing for training...')
    train_sample = read_file(settings.TRAIN_SAMPLE_PROPS.train_sample_path)
    map_col_names(train_sample)
    define_target(
        train_sample, cumulative_delays=settings.TRAIN_SAMPLE_PROPS.cumulative_days
    )

    return train_sample


def features_processing(df: pd.DataFrame, target_encoder: bool):

    if target_encoder:
        mean_encoder = TargetMeanEncoder()
        mean_encoder.fit(df[df['is_train'] == 1])
        df = mean_encoder.transform(df)

    return df


def enrich_with_features(df: pd.DataFrame):
    daily, monthly, quarterly = prepare_macro_features(
        country=settings.FEATURES_PARAMS.country,
        num_of_lags=settings.FEATURES_PARAMS.num_of_lags,
        window=settings.FEATURES_PARAMS.window,
    )

    df.rename(columns={f'{settings.FEATURES_PARAMS.date_col}': 'date'}, inplace=True)
    final_df = pd.merge(df, daily, on='date', how='left')
    final_df = pd.merge(final_df, monthly, on='date', how='left')
    final_df = pd.merge(final_df, quarterly, on='date', how='left')

    return final_df


def run_scoring_pipe():
    sample = preprocess_raw_sample()
    clean_sample = prepare_main_sample(
        df=sample, test_size=settings.TRAIN_SAMPLE_PROPS.test_size
    )

    clean_sample = features_processing(clean_sample, target_encoder=True)

    run_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    trained_model = fit(clean_sample, run_time)
    predictions = predict(clean_sample, trained_model)


def run_sfa():
    sample = preprocess_raw_sample()
    clean_sample = prepare_main_sample(
        df=sample, test_size=settings.TRAIN_SAMPLE_PROPS.test_size
    )
    clean_sample = features_processing(clean_sample, target_encoder=True)
    sfa = SFA(clean_sample)
    sfa.get_sfa_results()


if __name__ == '__main__':
    fire.Fire({
        "run_scoring": run_scoring_pipe,
        "run_sfa": run_sfa
    })

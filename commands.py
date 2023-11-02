import logging
import warnings

import fire
import pandas as pd

from configs.config import settings
from data_prep.normalize_raw_data import define_target, map_col_names
from data_prep.sample_prep import prepare_main_sample
from features.macro_features import prepare_macro_features
from models.train import fit, predict
from validation.adversarial_val import perform_adv_val
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
    # sample = preprocess_raw_sample()
    # clean_sample = prepare_main_sample(
    #     df=sample, test_size=settings.TRAIN_SAMPLE_PROPS.test_size
    # )
    # fit(clean_sample)
    # predict(clean_sample)
    
    # train_data = read_file(settings.TRAIN_SAMPLE_PROPS.train_sample_path)
    # blind_data = read_file(settings.BLIND_SAMPLE_PROPS.blind_path)
    
    train_data = pd.read_excel('C:\\Users\\masrur\\code\\finca-guatemala\\data\\train_data.xlsx')
    blind_data = pd.read_excel('C:\\Users\\masrur\\code\\finca-guatemala\\data\\blind_data.xlsx')
                                      
    adv_auc = perform_adv_val(train_data, blind_data)
    
    print(adv_auc)


if __name__ == '__main__':
    fire.Fire({"run_scoring": run_scoring_pipe})

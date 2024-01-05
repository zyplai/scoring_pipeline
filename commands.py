import logging
import pprint
import warnings
from datetime import date, datetime

import fire
import pandas as pd

from configs.config import settings
from data_prep.normalize_raw_data import define_target, map_col_names
from data_prep.sample_prep import prepare_val_sample
from features.cat_features import TargetMeanEncoder
from features.macro_features import prepare_macro_features
from features.sfa import SFA
from models.train import fit, predict
from validation.sfa_report import create_sfa_report
from utils.basic_utils import read_file
from validation.model_report import create_report_fpdf
from validation.ks_test import compare_datasets


warnings.filterwarnings("ignore")


def preprocess_raw_sample():
    logging.info('--- Reading train sample and prepraing for training...')
    train_sample = read_file(settings.TRAIN_SAMPLE_PROPS.train_sample_path)
    map_col_names(train_sample)
    define_target(
        df=train_sample,
        cumulative_delays=settings.TRAIN_SAMPLE_PROPS.cumulative_days,
        number_of_days=settings.TRAIN_SAMPLE_PROPS.target_days,
    )

    return train_sample


def features_processing(
    df: pd.DataFrame, run_time: datetime, target_encoder: bool
) -> pd.DataFrame:
    if target_encoder:
        mean_encoder = TargetMeanEncoder()
        mean_encoder.fit(df[df['is_train'] == 1], run_time)
        df = mean_encoder.transform(df)

        num_cols = [col for col in settings.SET_FEATURES.features_list if col not in settings.SET_FEATURES.cat_feature_list] # get numeric columns
        tme_cols = [cat_col + '_tme' for cat_col in settings.SET_FEATURES.cat_feature_list] # create list of tme columns
        settings.SET_FEATURES.features_list_tme = num_cols + tme_cols # todo:

    return df


def enrich_with_features(df: pd.DataFrame, enabled):
    if enabled:
        daily, monthly, quarterly = prepare_macro_features(
            country=settings.FEATURES_PARAMS.partners_country,
            num_of_lags=settings.FEATURES_PARAMS.num_of_lags,
            window=settings.FEATURES_PARAMS.window,
        )

        df.rename(columns={f'{settings.FEATURES_PARAMS.date_col}': 'date'}, inplace=True)
        final_df = pd.merge(df, daily, on='date', how='left')
        final_df = pd.merge(final_df, monthly, on='date', how='left')
        final_df = pd.merge(final_df, quarterly, on='date', how='left')
        return final_df
    else:
        return df


def run_scoring_pipe():
    sample = preprocess_raw_sample()
    clean_sample = prepare_val_sample(
        df=sample, test_size=settings.TRAIN_SAMPLE_PROPS.test_size
    )
        
    run_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    clean_sample = features_processing(
        clean_sample, run_time, target_encoder=settings.TARGET_MEAN_ENCODE.target_encode
    )
    
    clean_sample = enrich_with_features(clean_sample, enabled=settings.MACRO.enrichment)

    trained_model = fit(clean_sample, run_time)
    predictions = predict(clean_sample, trained_model, run_time)

    create_report_fpdf(predictions, trained_model, run_time)


def run_sfa():
    sample = preprocess_raw_sample()
    clean_sample = prepare_val_sample(
        df=sample, test_size=settings.TRAIN_SAMPLE_PROPS.test_size
    )

    run_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    clean_sample = features_processing(clean_sample, run_time, target_encoder=True)

    sfa = SFA(clean_sample)
    sfa_res = sfa.get_sfa_results(run_time)
    corr_path = sfa.spearman_corr(run_time)
    create_sfa_report(sfa_res, corr_path, run_time)


if __name__ == '__main__':
    fire.Fire({"run_scoring": run_scoring_pipe, "run_sfa": run_sfa})


# sfa, report versions
# folder naming
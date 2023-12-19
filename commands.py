import logging
import pprint
import warnings
from datetime import date, datetime

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
from validation.model_report import create_report_fpdf

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

    return df


def enrich_with_features(df: pd.DataFrame):
    daily, monthly, quarterly = prepare_macro_features(
        country=settings.FEATURES_PARAMS.partners_country,
        num_of_lags=settings.FEATURES_PARAMS.num_of_lags,
        window=settings.FEATURES_PARAMS.window,
    )
    
    df.rename(columns={f'{settings.FEATURES_PARAMS.date_col}': 'date'}, inplace=True)

    final_df = pd.merge(df, daily, on='date', how='left')

    final_df = pd.merge(final_df, monthly.drop('date',axis=1),
        left_on=[final_df['date'].dt.year, final_df['date'].dt.month],
        right_on=[monthly['date'].dt.year, monthly['date'].dt.month],
        how='left').drop(columns=['key_0','key_1'])

    final_df = pd.merge(final_df, quarterly.drop('date',axis=1),
        left_on=[final_df['date'].dt.year, final_df['date'].dt.month],
        right_on=[quarterly['date'].dt.year, quarterly['date'].dt.month],
        how='left').drop(columns=['key_0','key_1'])

    return final_df


def custom_proc(df: pd.DataFrame):
    df2 = df[(df['status_of_loan'] == 'Active') & (df['target'] == 1) | (df['status_of_loan'] != 'Active')]
    
    df2['collateral'] = df2['collateral'].replace({'No': 0, 'Yes': 1}).astype('int32')
    df2['gender'] = df2['gender'].replace({'Female': 0, 'Male': 1}).astype('int32')
    
    df2['AECB Point In Time Score'].replace({
        'NA value': None,
        'NH value': None,
        'NR value': None
    }, inplace=True)
    
    df2['AECB Point In Time Score'] = df2['AECB Point In Time Score'].astype(float)
    
    return df2
    


def run_scoring_pipe():

    sample = preprocess_raw_sample()

    data = custom_proc(sample)
    
    clean_sample = prepare_main_sample(
        df=data, test_size=settings.TRAIN_SAMPLE_PROPS.test_size
    )

    run_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    clean_sample = features_processing(
        clean_sample, run_time, target_encoder=settings.TARGET_MEAN_ENCODE.target_encode
    )

    clean_sample = enrich_with_features(clean_sample)

    print(clean_sample[settings.SET_FEATURES.features_list].head(5))
    print(clean_sample[settings.SET_FEATURES.features_list].info())
    print(clean_sample[settings.SET_FEATURES.features_list].isna().sum())

    trained_model = fit(clean_sample, run_time)
    predictions = predict(clean_sample, trained_model)

    create_report_fpdf(predictions, trained_model, run_time)


def run_sfa():
    sample = preprocess_raw_sample()
    clean_sample = prepare_main_sample(
        df=sample, test_size=settings.TRAIN_SAMPLE_PROPS.test_size
    )

    run_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    clean_sample = features_processing(clean_sample, run_time, target_encoder=True)

    sfa = SFA(clean_sample)
    sfa_results = sfa.get_sfa_results(run_time)
    sfa_corr = sfa.spearman_corr(run_time)
    create_sfa_report(sfa_results, sfa_corr, run_time)



if __name__ == '__main__':
    fire.Fire({"run_scoring": run_scoring_pipe, "run_sfa": run_sfa})
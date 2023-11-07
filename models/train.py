import logging
import os
from datetime import datetime

import catboost as cb
import pandas as pd
from sklearn.metrics import roc_auc_score

from configs.config import settings
from data_prep.normalize_raw_data import map_col_names
from .model_validator import create_validator
from utils.basic_utils import (
    save_pickle,
    load_pickle,
    read_file,
    save_toml
)


def fit(df: pd.DataFrame) -> None:
    """
    Fit and train a CatBoost classifier model.

    Parameters
    ----------
    df : pd.DataFrame
        main_sample df with defined factors and is_train bool.
    params : dict
        params to set.
    output_dir : str
        directory to store results.
    features_list : list
        feature names to use.
    cat_feature_list : list
        list of categorical features.
    type_ : str
        baseline / final.

    Returns
    -------
    None

    This function splits the input dataframe into train and test, initializes
    a CatBoostClassifier and fits it on the train data while evaluating on
    the test data then save the model artifact.
    """

    # split into train and test
    X_train = df.loc[df['is_train'] == 1].reset_index(drop=True)[
        settings.SET_FEATURES.features_list
    ]
    y_train = df.loc[df['is_train'] == 1, ['target']].reset_index(drop=True)
    X_test = df.loc[df['is_train'] == 0].reset_index(drop=True)[
        settings.SET_FEATURES.features_list
    ]
    y_test = df.loc[df['is_train'] == 0, ['target']].reset_index(drop=True)

    # init model and fit
    logging.info('------- Fitting the model...')
    cbm = cb.CatBoostClassifier(**settings.SET_FEATURES.model_params,
                                verbose=False)

    model = cbm.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        cat_features=settings.SET_FEATURES.cat_feature_list,
    )

    logging.info('------- Model trained...')

    # create a timestamp for the current run
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # create the directory for the current run
    run_dir = os.path.join(
                os.getcwd(),
                settings.SET_FEATURES.output_dir,
                f'run_{current_datetime}'
            )
    model_artifact_dir = f'{run_dir}/model_artifact'
    try:
        model_path = f'{model_artifact_dir}/{settings.SET_FEATURES.type_}.pkl'
        create_validator(run_dir,
                         df[settings.SET_FEATURES.features_list],
                         'target',
                         'variable_validator')
        save_toml(run_dir)
        # save model in pickle file
        save_pickle(model, model_path)
        logging.info('------- Model saved...')
    except OSError:
        os.makedirs(run_dir)
        os.makedirs(model_artifact_dir)
        model_path = f'{model_artifact_dir}/{settings.SET_FEATURES.type_}.pkl'
        create_validator(run_dir,
                         df[settings.SET_FEATURES.features_list],
                         'target',
                         'variable_validator')
        save_toml(run_dir)
        # save model in pickle file
        save_pickle(model, model_path)
        logging.info('------- Model saved...')


def predict(df: pd.DataFrame, inference: bool = False) -> pd.DataFrame:
    """
    Make predictions on the input data using the trained model.

    Parameters:
    df (pd.DataFrame): Dataframe containing features and labels
    inference (bool, optional): Whether running in inference mode on blind data.
                                Default is False.

    Returns:
    pd.DataFrame: Dataframe with predictions added as a new column

    This function takes a dataframe to make predictions by loaded model.

    If running in inference mode, it reads a separate blind sample file, makes
    predictions on that and returns it.

    Otherwise, it makes predictions on the whole data then print AUC by train and test
    set and returns a dataframe with actuals and predictions.
    """ # noqa

    try:
        model = load_pickle(settings.MODEL_PATH.baseline_model)
        logging.info('---Model loaded...')

        if inference:

            logging.info('---Reading blind sample and prepraing for prediction...')
            blind_data = read_file(settings.BLIND_SAMPLE_PROPS.blind_path)
            map_col_names(blind_data)

            blind_data['predictions'] = model.predict_proba(
                blind_data[settings.SET_FEATURES.features_list])[:, 1]

            return blind_data

        else:

            df['predictions'] = model.predict_proba(
                df[settings.SET_FEATURES.features_list])[:, 1]

            auc_train = roc_auc_score(
                df[df['is_train'] == 1]['target'],
                df[df['is_train'] == 1]['predictions'])

            auc_test = roc_auc_score(
                df[df['is_train'] == 0]['target'],
                df[df['is_train'] == 0]['predictions'])

            print("Train AUC: ", auc_train, '\nTest AUC', auc_test)

            return df

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")

import logging
import os
from functools import partial
from datetime import datetime

import numpy as np
import catboost as cb
import pandas as pd

from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score, KFold

from configs.config import settings
from data_prep.normalize_raw_data import map_col_names
from utils.basic_utils import read_file, save_pickle, save_toml

from models.model_validator import create_validator

import optuna
from optuna.samplers import TPESampler
from catboost.utils import eval_metric


def objective( trial, X, y, X_train, y_train ):
    params = {}
    list_params = ['bootstrap_type', 'boosting_type' ]

    for param in settings.TUNING.tuning_params :
        if param in list_params :
            params[param] = trial.suggest_categorical(param, settings.TUNING.tuning_params[param] )
        else :
            params[ param ] = trial.suggest_float(param, settings.TUNING.tuning_params[param][0],  
                                                    settings.TUNING.tuning_params[param][1]),

    model = cb.CatBoostClassifier(**params, random_seed=settings.TUNING.tuning_params['random_seed'])
    model.fit(X_train, y_train, verbose=0)

    kf = KFold(n_splits=5, shuffle=True, random_state=settings.TUNING.tuning_params['random_seed'])
    roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    scores = cross_val_score(model, X, y, scoring=roc_auc_scorer, cv=kf, verbose=0)

    return np.mean(scores)


def optimize_hyperparameters( X,y, X_train, y_train ) :
    study = optuna.create_study()
    partial_objective = partial(objective, X=X, y=y, X_train=X_train, y_train=y_train)
    study.optimize(partial_objective, n_trials=50)
    best_params = study.best_params

    return best_params


def fit(df: pd.DataFrame, run_time) -> cb.CatBoostClassifier:
    """
    This function splits the input dataframe into train and test, initializes
    a CatBoostClassifier and fits it on the train data while evaluating on the test data then save the model artifact.

    Args:
        df (pd.DataFrame): main_sample df with defined factors and is_train bool.

    Returns:
        object: The trained CatBoost modeol
    """
    if settings.TARGET_MEAN_ENCODE.target_encode :
        feature_list = settings.SET_FEATURES.features_list_tme
        cat_feature_list = []
    else :
        feature_list = settings.SET_FEATURES.features_list
        cat_feature_list = settings.SET_FEATURES.cat_feature_list

    # split into train and test
    X_train = df.loc[df['is_train'] == 1].reset_index(drop=True)[feature_list]
    y_train = df.loc[df['is_train'] == 1, ['target']].reset_index(drop=True)
    
    X_test = df.loc[df['is_train'] == 0].reset_index(drop=True)[feature_list]
    y_test = df.loc[df['is_train'] == 0, ['target']].reset_index(drop=True)

    X = pd.concat([X_train,X_test])
    y = pd.concat([y_train,y_test])
    
    # init model and fit
    logging.info('------- Fitting the model...')

    if settings.TUNING.use_tuning == False :
        cbm = cb.CatBoostClassifier(**settings.SET_FEATURES.model_params, verbose=False)
        cbm.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=cat_feature_list)

    else :
        best_params = optimize_hyperparameters( X,y, X_train, y_train )
        print(best_params)

        cbm = cb.CatBoostClassifier(
            **best_params,
            random_seed = settings.SET_FEATURES.model_params['random_seed']
        )

        cbm.fit(X_train, y_train,
            eval_set=(X_test,y_test),
            cat_features=cat_feature_list,
            verbose=False
        )

    logging.info('------- Model trained...')
    # create a timestamp for the current run

    # create the directory for the current run
    run_dir = os.path.join(
        os.getcwd(), settings.SET_FEATURES.output_dir, f'run_{run_time}'
    )
    model_artifact_dir = f'{run_dir}/model_artifact'
    try:
        model_path = f'{model_artifact_dir}/{settings.SET_FEATURES.type_}.pkl'
        create_validator(
            run_dir,
            df[settings.SET_FEATURES.features_list],
            'target',
            'variable_validator',
        )
        save_toml(run_dir)
        # save model in pickle file
        save_pickle(cbm, model_path)
        logging.info('------- Model saved...')
    except OSError:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(model_artifact_dir)
        model_path = f'{model_artifact_dir}/{settings.SET_FEATURES.type_}.pkl'
        create_validator(
            run_dir,
            df[settings.SET_FEATURES.features_list],
            'target',
            'variable_validator',
        )
        save_toml(run_dir)
        # save model in pickle file
        save_pickle(cbm, model_path)
        logging.info('------- Model saved...')

    return cbm


def predict(
    df: pd.DataFrame,
    model: cb.CatBoostClassifier,
    run_time: datetime,
    inference: bool = False
) -> pd.DataFrame:
    """
    Make predictions on the input data using the trained model.

    Parameters:
    df (pd.DataFrame): Dataframe containing features and labels
    model (object): Trained CatBoost model
    inference (bool, optional): Whether running in inference mode on blind data.
                                Default is False.

    Returns:
    pd.DataFrame: Dataframe with predictions added as a new column

    This function takes a dataframe to make predictions by loaded model.

    If running in inference mode, it reads a separate blind sample file, makes
    predictions on that and returns it.

    Otherwise, it makes predictions on the whole data then print AUC by train and test
    set and returns a dataframe with actuals and predictions.
    """

    if inference:
        logging.info('---Reading blind sample and prepraing for prediction...')
        blind_data = read_file(settings.BLIND_SAMPLE_PROPS.blind_path)
        map_col_names(blind_data)

        blind_data['predictions'] = model.predict_proba(
            blind_data[settings.SET_FEATURES.features_list]
        )[:, 1]

        return blind_data

    else:
        if settings.TARGET_MEAN_ENCODE.target_encode:
            df['predictions'] = model.predict_proba(df[settings.SET_FEATURES.features_list_tme])[:, 1]
        else:
            df['predictions'] = model.predict_proba(df[settings.SET_FEATURES.features_list])[:, 1]

        auc_train = roc_auc_score(
            df[df['is_train'] == 1]['target'], df[df['is_train'] == 1]['predictions']
        )

        auc_test = roc_auc_score(
            df[df['is_train'] == 0]['target'], df[df['is_train'] == 0]['predictions']
        )

        # mlflow_track(model, df, auc_train, auc_test, run_time)

        print("Train AUC: ", auc_train, '\nTest AUC', auc_test)

        return df

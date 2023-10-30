import logging
import os
from datetime import datetime

import catboost as cb
import pandas as pd

from configs.config import settings
from data_prep.normalize_raw_data import map_col_names
from utils.basic_utils import (
    save_pickle,
    read_file
)


def fit(df: pd.DataFrame) -> cb.CatBoostClassifier:
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
    CatBoostClassifier: Trained CatBoost model

    This function splits the input dataframe into train and test, initializes
    a CatBoostClassifier and fits it on the train data while evaluating on
    the test data. It returns the trained CatBoost model.
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

    return model


def predict(df: pd.DataFrame,
            model: str,
            inference: bool = False) -> pd.DataFrame:
    """
    Make predictions on the input data using the trained model.

    Parameters:
    df (pd.DataFrame): Dataframe containing features and labels
    model (CatBoostClassifier): Trained CatBoost model
    inference (bool, optional): Whether running in inference mode on blind data.
                                Default is False.

    Returns:
    pd.DataFrame: Dataframe with predictions added as a new column

    This function takes a trained CatBoost model and input data to make predictions.

    If running in inference mode, it reads a separate blind sample file, makes
    predictions on that and returns it.

    Otherwise, it splits the input dataframe into train and test, makes predictions
    on the test set and returns a dataframe with actuals and predictions. It also
    saves the model as a pickle file.
    """ # noqa

    # evaluate results
    if inference:

        logging.info('---Reading blind sample and prepraing for prediction...')
        blind_sample = read_file(settings.BLIND_SAMPLE_PROPS.blind_sample_path)
        map_col_names(blind_sample)
        blind_sample = blind_sample[settings.SET_FEATURES.features_list]
        blind_preds = model.predict_proba(blind_sample)[:, 1]

        blind_sample['predictions'] = blind_preds

        return blind_sample

    else:
        X_test = df.loc[df['is_train'] == 0].reset_index(drop=True)[
            settings.SET_FEATURES.features_list
        ]
        y_test = df.loc[df['is_train'] == 0, ['target']].reset_index(drop=True)
        y_test_preds = model.predict_proba(X_test)[:, 1]

        X_test['target'] = y_test
        X_test['predictions'] = y_test_preds

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

            # save model in pickle file
            save_pickle(model, model_path)
        except OSError:
            os.makedirs(run_dir)
            os.makedirs(model_artifact_dir)
            model_path = f'{model_artifact_dir}/{settings.SET_FEATURES.type_}.pkl'

            # save model in pickle file
            save_pickle(model, model_path)

        return X_test

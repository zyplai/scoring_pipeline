import logging
import os
from datetime import datetime

import catboost as cb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from configs.config import settings
from data_prep.normalize_raw_data import map_col_names
from utils.basic_utils import (
    gini,
    save_pickle,
    auc_roc,
    read_file
)


def fit(df: pd.DataFrame) -> str:
    """
    function to fit and evaluate lightgbm for baseline

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
    str: Path to the saved model artifact.

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


def predict(df: pd.DataFrame, model: str, inference: bool = False):

    # evaluate results
    if inference:

        logging.info('---Reading blind sample and prepraing for prediction...')
        blind_sample = read_file(settings.BLIND_SAMPLE_PROPS.blind_sample_path)
        map_col_names(blind_sample)
        blind_sample = blind_sample[settings.SET_FEATURES.features_list]
        blind_preds = model.predict_proba(blind_sample)[:, 1] # noqa

    else:
        X_train = df.loc[df['is_train'] == 1].reset_index(drop=True)[
            settings.SET_FEATURES.features_list
        ]
        y_train = df.loc[df['is_train'] == 1,
                         ['target']].reset_index(drop=True)
        X_test = df.loc[df['is_train'] == 0].reset_index(drop=True)[
            settings.SET_FEATURES.features_list
        ]
        y_test = df.loc[df['is_train'] == 0, ['target']].reset_index(drop=True)
        y_train_preds = model.predict_proba(X_train)[:, 1]
        y_test_preds = model.predict_proba(X_test)[:, 1]

        # create a timestamp for the current run
        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        try:
            # create the directory for the current run
            run_dir = os.path.join(
                    os.getcwd(),
                    settings.SET_FEATURES.output_dir,
                    f'run_{current_datetime}'
                )
            model_artifact_dir = f'{run_dir}/model_artifact'
            model_path = f'{model_artifact_dir}/{settings.SET_FEATURES.type_}.pkl'

            # save model in pickle file
            save_pickle(model, model_path)
        except OSError:
            os.makedirs(run_dir)
            os.makedirs(model_artifact_dir)
            model_path = f'{model_artifact_dir}/{settings.SET_FEATURES.type_}.pkl'

            # save model in pickle file
            save_pickle(model, model_path)

        # calc gini on train and test
        gini_results = {
            'train_gini': gini(y_train, y_train_preds),
            'test_gini': gini(y_test, y_test_preds),
            'train_auc': auc_roc(y_train, y_train_preds),
            'test_auc': auc_roc(y_test, y_test_preds),
        }

        # save gini to txt file
        gini_path = f'{run_dir}/gini.txt'
        with open(gini_path, 'w') as f:
            f.write(str(gini_results))

        # get factor importance
        feature_importance = pd.DataFrame(
            sorted(zip(model.feature_importances_, X_train.columns)),
            columns=['Value', 'Feature Name'],
        )
        feature_importance = feature_importance.loc[
            feature_importance['Value'] > 0
        ].reset_index(drop=True)

        # save feature importance plot
        feature_importance_path = f'{run_dir}/feature_importance.png'
        plt.figure(figsize=(15, 10))
        sns.barplot(
            x='Value',
            y='Feature Name',
            data=feature_importance.sort_values(by='Value', ascending=False),
        )
        plt.title(f'{settings.SET_FEATURES.type_} model feature importance')
        plt.tight_layout()
        plt.savefig(feature_importance_path)

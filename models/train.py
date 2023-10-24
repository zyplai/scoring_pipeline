import logging
import os
import datetime

import catboost as cb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from configs.config import settings
from utils.basic_utils import gini, save_pickle, auc_roc


def fit_predict_catboost(df: pd.DataFrame):
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
    results tuple

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

    # create a timestamp for the current run
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # create the directory for the current run
    run_dir = os.path.join(
        os.getcwd(),
        settings.SET_FEATURES.output_dir,
        f'run_{current_datetime}'
    )

    os.makedirs(run_dir, exist_ok=True)

    # create the model artifact directory
    model_artifact_dir = f'{run_dir}/model_artifact'
    os.makedirs(model_artifact_dir, exist_ok=True)

    # save model in pickle file
    model_path = f'{model_artifact_dir}/{settings.SET_FEATURES.type_}.pkl'
    save_pickle(model, model_path)

    # evaluate results
    y_train_preds = model.predict_proba(X_train)[:, 1]
    y_test_preds = model.predict_proba(X_test)[:, 1]

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

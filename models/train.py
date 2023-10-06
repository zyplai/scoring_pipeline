import logging

import catboost as cb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from configs.config import settings
from utils.basic_utils import gini, save_pickle


def fit_predict_catboost(
    df: pd.DataFrame,
    output_dir: str,
    features_list: list,
    cat_feature_list: list,
    type_: str,
    train_drop_cols: list,
    model_path: str,
):
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
    results dict

    """

    # split into train and test
    X_train = df.loc[df['is_train'] == 1].reset_index(drop=True)[features_list]
    y_train = df.loc[df['is_train'] == 1, ['target']].reset_index(drop=True)
    X_test = df.loc[df['is_train'] == 0].reset_index(drop=True)[features_list]
    y_test = df.loc[df['is_train'] == 0, ['target']].reset_index(drop=True)

    # init model and fit
    cbm = cb.LGBMClassifier(**settings.SET_FEATURES.model_params)

    model = cbm.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        categorical_feature=cat_feature_list,
    )

    # save model in pickle file
    save_pickle(model, model_path + type_ + '.pkl')

    # evaluate results
    y_train_preds = model.predict_proba(X_train)[:, 1]
    y_test_preds = model.predict_proba(X_test)[:, 1]

    # calc gini on train and test
    gini_results = {
        'train_gini': gini(y_train, y_train_preds),
        'test_gini': gini(y_test, y_test_preds),
    }

    # save gini to txt file
    with open(output_dir + '_gini.txt', 'w') as f:
        f.write(str(gini_results))

    # get factor importance
    feature_importance = pd.DataFrame(
        sorted(zip(model.feature_importances_, X_train.columns)),
        columns=['Value', 'Feature Name'],
    )
    feature_importance = feature_importance.loc[
        feature_importance['Value'] > 0
    ].reset_index(drop=True)

    plt.figure(figsize=(15, 10))
    sns.barplot(
        x='Value',
        y='Feature Name',
        data=feature_importance.sort_values(by='Value', ascending=False),
    )
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig(output_dir + 'lgbm_importances_{}.png'.format(type))

    return gini_results, feature_importance

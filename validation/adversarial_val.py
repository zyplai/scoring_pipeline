from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from configs.config import settings
from utils.basic_utils import read_file


def train_adv_val_model(data: pd.DataFrame, target: str) -> float:
    """
    Train adversarial model

    Args:
        data (pd.DataFrame): data to use in adversarial validation
        target (str): classification target

    Returns:
        float: AUC of the adversarial validation model
    """

    clf = CatBoostClassifier(iterations=200, random_state=1)

    if settings.TARGET_MEAN_ENCODE.target_encode:
        X = data[settings.SET_FEATURES.features_list_tme].copy()
        y = data[target].copy()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=0
        )

        clf.fit(X_train, y_train, cat_features=[], verbose=False)
    else:
        X = data[settings.SET_FEATURES.features_list].copy()
        y = data[target].copy()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=0
        )
        clf.fit(
            X_train,
            y_train,
            cat_features=settings.SET_FEATURES.cat_feature_list,
            verbose=False,
        )
        
    features = data[settings.SET_FEATURES.features_list].columns
        
    feature_importance = clf.feature_importances_
    sorted_idx = np.argsort(feature_importance)

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
    plt.savefig('data/adv_val_feature_importance.png', bbox_inches='tight')
    plt.close()

    preds = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)

    return round(float(auc), 4)


def perform_adv_val(train_set: pd.DataFrame, inference: bool = True) -> float:
    """
    Perform adversarial validation on train or blind set

    Args:
        train_set (pd.DataFrame): Dataset used for training the model
        inference (bool, optional): Set True if adversarial validation is to be done on Blind set. Defaults to False for train/test split validation.

    Returns:
        float: AUC of the model
    """
    if inference:
        blind_set = read_file(settings.BLIND_SAMPLE_PROPS.blind_path)
        
        cat_columns = settings.SET_FEATURES.cat_feature_list
        blind_set[cat_columns] = blind_set[cat_columns].fillna('N/A') 

        target = 'is_blind'
        train_set[target] = 0
        blind_set[target] = 1

        data_all = pd.concat([train_set, blind_set], ignore_index=True)

        blind_auc = train_adv_val_model(data_all, target)
        return blind_auc

    else:
        target = 'is_train'
        train_test_auc = train_adv_val_model(train_set, target)
        return train_test_auc

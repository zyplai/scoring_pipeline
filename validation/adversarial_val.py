from catboost import CatBoostClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from configs.config import settings


def perform_adv_val(train_set: pd.DataFrame, blind_set: pd.DataFrame) -> float:
    """
    Perform adversarial validation on train and blind set

    Args:
        train_set (pd.DataFrame): Dataset used for training the model
        blind_set (pd.DataFrame): Blind set for model validation

    Returns:
        float: AUC of the model
    """
    
    target = 'set'
    train_set[target] = 0
    blind_set[target] = 1
    
    data_all = pd.concat([train_set, blind_set])
    data_all = data_all.reset_index(drop=True)
    
    data_all[settings.SET_FEATURES.cat_feature_list] = data_all[settings.SET_FEATURES.cat_feature_list].fillna('N/A')
        
    X = data_all[settings.SET_FEATURES.features_list].copy()
    y = data_all[target].copy()
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=0) 
            
    clf = CatBoostClassifier(
        iterations=200,
        random_state=1
    )

    clf.fit(
        X_train, y_train, 
        cat_features=settings.SET_FEATURES.cat_feature_list, 
        verbose=False
    )
    
    preds = clf.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, preds)
    
    return round(float(auc), 4)
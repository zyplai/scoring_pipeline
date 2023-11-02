from catboost import CatBoostClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from configs.config import settings


def perform_adv_val(train_set: pd.DataFrame, blind_set: pd.DataFrame):
    target = 'set'
    train_set[target] = 0
    blind_set[target] = 1
    
    data_all = pd.concat([train_set, blind_set])
    data_all.index = range(len(data_all)) # type: ignore
    
    features_list = [
        'Type of product', 'Branch', 'Location', 'Gender', 'Family status', 'Type of client (old / new)', 'Education level', 'Employment sphere', 
        'Presence of housing', 'Co-borrower', 'Collateral', 'Loan amount', 'Duration at disbursement (months)', 'Interest rate', 'Age', 'Monthly income', 
        'Quantity of prior loans', 'Quantity of dependents'
    ]

    cat_feature_list = [
        'Type of product', 'Branch', 'Location', 'Gender', 'Family status', 'Type of client (old / new)', 'Education level', 'Employment sphere', 
        'Presence of housing', 'Co-borrower', 'Collateral'
    ]
    
    data_all[cat_feature_list] = data_all[cat_feature_list].fillna('N/A')
    
    X = data_all[features_list].copy()
    y = data_all[target].copy()
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=0) 
            
    clf = CatBoostClassifier(
        iterations=500,
        random_state=1
    )

    clf.fit(
        X_train, y_train, 
        cat_features=cat_feature_list, 
        verbose=False
    )
    
    preds = clf.predict_proba(X_val)[:, 1]
    
    auc = roc_auc_score(y_val, preds)
    
    return auc
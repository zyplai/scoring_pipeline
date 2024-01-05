import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from configs.config import settings


def prepare_val_sample(df: pd.DataFrame, test_size: int = 0.15, random_state: int = 42):
    if settings.VALIDATION.out_of_time:
        df.sort_values(settings.FEATURES_PARAMS.date_col, inplace=True)
            
        split_index = int(1-test_size * len(df)) - 1
        
        train = df.iloc[:split_index, :].copy()
        test = df.iloc[split_index:, :].copy()

        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        
        train['is_train'] = 1
        test['is_train'] = 0
    else:
        train, test = train_test_split(
            df, test_size=test_size, shuffle=True, random_state=random_state
        )
        # concat back with label is_train
        train['is_train'] = 1
        test['is_train'] = 0
    
    # concat to one df
    df = pd.concat([train, test], ignore_index=True)

    cat_columns = settings.SET_FEATURES.cat_feature_list
    df[cat_columns] = df[cat_columns].fillna('N/A') 
    
    return df

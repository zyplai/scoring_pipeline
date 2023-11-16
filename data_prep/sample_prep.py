import logging

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_main_sample(df: pd.DataFrame, test_size: int = 0.3, random_state: int = 42):

    X, y = df.drop('target', axis=1), df[['target']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )
    logging.info(
        '\t --- Train and test default rates are {} and {} respectively'.format(
            y_train.mean()[0], y_test.mean()[0]
        )
    )

    # concat back with label is_train
    X_train['is_train'] = 1
    X_test['is_train'] = 0

    # concat X and y
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    # concat to one df
    df = pd.concat([train, test], ignore_index=True)

    cat_columns = list(df.select_dtypes(object))
    df[cat_columns] = df[cat_columns].fillna('N/A')

    return df

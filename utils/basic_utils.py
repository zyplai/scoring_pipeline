import json
import pickle

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

"""
    Reading and saving files functions
"""


def read_file(
    path: str,
    sheetname: str = None,
    dtypes: list = None,
    skiprows: int = None,
 ):
    """
    function to read files in most common formats
    :path: relative path to file
    :root_path: root path of the local machine
    :dtypes: specify types for columns in case of excel files
    :return: df
    """

    # excel formats
    if any(file_type in path for file_type in ['xlsx', 'xls', 'xlsb']):
        output = pd.read_excel(
            path, sheet_name=sheetname, dtype=dtypes, skiprows=skiprows
        )

    # hdf files
    elif any(file_type in path for file_type in ['hd5', 'h5']):
        output = pd.read_hdf(path, key='df')

    # parquet
    elif any(file_type in path for file_type in ['parquet']):
        output = pd.read_parquet(path, engine='pyarrow')

    else:
        raise ValueError(
            'Wrong data type is specified {}. Please, double check your file extension'.format(
                path
            )
        )

    return output


def save_file(file: object, path: str, format: str = 'fixed'):
    """
    function to save files in most common formats
    :path: relative path to file
    :root_path: root path of the local machine
    :dtypes: specify types for columns in case of excel files
    :return: df
    """

    # excel formats
    if any(file_type in path for file_type in ['xlsx', 'xls', 'xlsb']):
        output = file.to_excel(path)

    # hdf files
    elif any(file_type in path for file_type in ['hd5', 'h5']):
        output = file.to_hdf(path, key='df', format=format)

    # parquet
    elif any(file_type in path for file_type in ['parquet']):
        output = file.to_parquet(path, engine='pyarrow')

    else:
        raise ValueError(
            'Wrong data type is specified {}. Please, double check your file extension'.format(
                path
            )
        )

    return output


def save_json(file, file_path, encoding='utf-8', ensure_ascii=False):
    """function to save file as json"""

    with open(file_path, "w", encoding=encoding) as jsonfile:
        json.dump(file, jsonfile, ensure_ascii=ensure_ascii)


def read_json(file_path, root_path='C:/Users/Shuhratjon.Khalilbek/', encoding='utf-8'):
    """function to read file as json"""

    with open(root_path + file_path, encoding=encoding) as f:
        json_file = json.load(f)

    return json_file


def save_pickle(file, path):
    """
    function to save files to pickle (mostly to store ML models)
    """
    with open(path, 'wb') as f:
        pickle.dump(file, f)

    print('File has been saved to {}'.format(root_path + path))


def load_pickle(file, path):
    """
    function to save files to pickle (mostly to store ML models)
    """
    with open(path, 'rb') as f:
        pickle_file = pickle.load(f)

    return pickle_file


def gini(y_true, y_score):

    # get roc auc value
    rocauc_score = roc_auc_score(y_true, y_score)

    # calc gini
    gini_val = 2 * rocauc_score - 1

    return gini_val


def fit_predict_lgb(
    df: pd.DataFrame,
    params: dict,
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
    X_train = (
        df.loc[df['is_train'] == 1]
        .drop(train_drop_cols, axis=1)
        .reset_index(drop=True)[features_list]
    )
    y_train = df.loc[df['is_train'] == 1, ['default_flag']].reset_index(drop=True)
    X_test = (
        df.loc[df['is_train'] == 0]
        .drop(train_drop_cols, axis=1)
        .reset_index(drop=True)[features_list]
    )
    y_test = df.loc[df['is_train'] == 0, ['default_flag']].reset_index(drop=True)

    # init model and fit
    lgb_model = lgb.LGBMClassifier(**params)

    model = lgb_model.fit(
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


def fit_predict_logit(
    df,
    logit_params,
    output_dir,
    final_features_list,
    cat_feature_list,
    train_drop_cols,
    model_path,
):
    """
    function to run fit predict for logit and save results
    """
    # split into train and test
    X_train = (
        df.loc[df['is_train'] == 1]
        .drop(train_drop_cols, axis=1)
        .reset_index(drop=True)[final_features_list]
    )
    y_train = df.loc[df['is_train'] == 1, ['default_flag']].reset_index(drop=True)
    X_test = (
        df.loc[df['is_train'] == 0]
        .drop(train_drop_cols, axis=1)
        .reset_index(drop=True)[final_features_list]
    )
    y_test = df.loc[df['is_train'] == 0, ['default_flag']].reset_index(drop=True)

    logit = LogisticRegression(**logit_params, solver='liblinear')
    logit_model = logit.fit(X_train, y_train)

    # save model in pickle file
    save_pickle(logit_model, model_path + '.pkl')

    # evaluate results
    y_train_preds = logit_model.predict_proba(X_train)[:, 1]
    y_test_preds = logit_model.predict_proba(X_test)[:, 1]

    # calc gini on train and test
    gini_results = {
        'train_gini': gini(y_train, y_train_preds),
        'test_gini': gini(y_test, y_test_preds),
    }

    # save gini to txt file
    with open(output_dir + '_gini.txt', 'w') as f:
        f.write(str(gini_results))


def fit_predict_rf(
    df: pd.DataFrame,
    rf_params: dict,
    output_dir: str,
    features_list: list,
    cat_feature_list: list,
    train_drop_cols: list,
    model_path: str,
):
    """
    function to fit and evaluate lightgbm for baseline

    Parameters
    ----------
    df : pd.DataFrame
        main_sample df with defined factors and is_train bool.
    rf_params : dict
        params to set.
    output_dir : str
        directory to store results.
    features_list : list
        feature names to use.
    cat_feature_list : list
        list of categorical features.
    type : str
        baseline / final.

    Returns
    -------
    results dict

    """
    # split into train and test
    X_train = (
        df.loc[df['is_train'] == 1]
        .drop(train_drop_cols, axis=1)
        .reset_index(drop=True)[features_list]
    )
    y_train = df.loc[df['is_train'] == 1, ['default_flag']].reset_index(drop=True)
    X_test = (
        df.loc[df['is_train'] == 0]
        .drop(train_drop_cols, axis=1)
        .reset_index(drop=True)[features_list]
    )
    y_test = df.loc[df['is_train'] == 0, ['default_flag']].reset_index(drop=True)

    # fit
    rf = RandomForestClassifier(**rf_params, random_state=42)
    rf.fit(X_train, y_train)

    # save model in pickle file
    save_pickle(rf, model_path + '.pkl')

    # evaluate results
    y_train_preds = rf.predict_proba(X_train)[:, 1]
    y_test_preds = rf.predict_proba(X_test)[:, 1]

    # calc gini on train and test
    gini_results = {
        'train_gini': gini(y_train, y_train_preds),
        'test_gini': gini(y_test, y_test_preds),
    }

    # save gini to txt file
    with open(output_dir + '_gini.txt', 'w') as f:
        f.write(str(gini_results))

    # feature importnace based on feature permutation
    feature_names = list(X_train.columns)
    importance = permutation_importance(
        rf, X_test, y_test, n_repeats=10, random_state=43, n_jobs=3
    )
    importance_df = pd.Series(importance.importances_mean, index=feature_names)
    importance_df.sort_values(ascending=False, inplace=True)

    # plot and save
    fig, ax = plt.subplots()
    importance_df.plot.bar(yerr=importance.importances_std, ax=ax)
    ax.set_ylabel('Mean accuracy decrease')
    plt.tight_layout()
    plt.savefig(output_dir + 'rf_importances.png')


"""
    END
"""

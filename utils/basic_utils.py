import json
import os
import pickle
import shutil

import pandas as pd
from sklearn.metrics import roc_auc_score


def read_file(path: str):
    """
    function to read files in most common formats
    :path: relative path to file
    :root_path: root path of the local machine
    :dtypes: specify types for columns in case of excel files
    :return: df
    """

    # excel formats
    if any(file_type in path for file_type in ['xlsx', 'xls', 'xlsb']):
        output = pd.read_excel(path)

    # csv files
    elif any(file_type in path for file_type in ['csv']):
        output = pd.read_csv(path)

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


def save_toml(dir_path):
    """
    Copy TOML configuration files from a 'configs' directory
    to a specified directory.

    Args:
        dir_path (str): The destination directory where TOML files will be copied.

    This function scans a 'configs' directory located in the current working directory
    and copies all TOML files that have filenames starting with the specified prefixes
    ('m' or 'd') to the specified 'dir_path'.

    Example:
    If the 'configs' directory contains TOML files like 'myconfig.toml' and 'devconfig.toml',
    and 'dir_path' is set to '/path/to/destination', this function will copy these files to
    '/path/to/destination' as 'myconfig.toml' and 'devconfig.toml', respectively.
    """  # noqa

    parent_dir = os.getcwd()
    config_dir = os.path.join(parent_dir, 'configs')
    prefixes = ['m', 'd']

    for i in os.listdir(config_dir):
        if i.endswith('.toml') and i.startswith(tuple(prefixes)):
            shutil.copy(os.path.join(config_dir, i), os.path.join(dir_path, i))


def save_txt(dir_path, config_content):
    with open(f'{dir_path}/config.txt', 'w') as file:
        file.write(config_content)


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

    print('File has been saved to {}'.format(path))


def load_pickle(path, file=None):
    """
    function to save files to pickle (mostly to store ML models)
    """
    with open(path, 'rb') as f:
        pickle_file = pickle.load(f)

    return pickle_file


def gini(y_true, y_score):
    rocauc_score = roc_auc_score(y_true, y_score)
    gini_val = 2 * rocauc_score - 1
    return gini_val


def snake_case(df: pd.DataFrame):
    df.columns = [x.lower().replace(' ', '_') for x in df.columns]
    return df

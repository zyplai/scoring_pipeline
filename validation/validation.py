# -*- coding: utf-8 -*-
"""
@author: shuhratjon.khalilbekov@ru.ey.com
"""

# path to main sampl
"""
    0. Configuration
"""
# path to all data
main_sample_path = (
    '/Desktop/workflow/PD/data/2. clean data/samples/train_test_sample.hd5'
)
fin_factors_path = '/Desktop/workflow/PD/data/3. factors/fin_factors.hd5'
qual_factors_path = '/Desktop/workflow/PD/data/3. factors/qual_factors.hd5'

# results dir
output_dir = 'C:/Users/Shuhratjon.Khalilbek/Desktop/workflow/PD/data/4.1. validation/'

# useful colname
psi_cols = ['Q2_1', 'Q1', 'Q3_5', 'Q4', 'scoring_year']
main_sample_cols = ['tax_code', 'scoring_date', 'fin_year', 'default_flag', 'is_train']
train_drop_cols = ['default_flag', 'tax_code', 'scoring_date', 'fin_year', 'is_train']
cat_feature_list = [
    'Q1',
    'Q1_1',
    'Q1_2',
    'Q1_3',
    'Q1_4',
    'Q1_5',
    'Q1_6',
    'Q1_7',
    'Q1_8',
    'Q1_9',
    'Q1_10',
    'Q1_11',
    'Q1_12',
    'Q2',
    'Q2_1',
    'Q2_2',
    'Q2_3',
    'Q2_4',
]

"""
    1. Modules and functions
"""
# settings for convenient work
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from utilities.basic_utils import *
from utilities.psi import *

warnings.filterwarnings('ignore')

"""
    2. Main
"""
if __name__ == '__main__':

    # prepare data
    df = construct_main_df(
        main_sample_path,
        fin_factors_path,
        qual_factors_path,
        main_sample_cols,
        cat_feature_list,
    )
    df['scoring_year'] = df['scoring_date'].dt.year

    # split into train / test
    train = df.loc[df['is_train'] == 1].reset_index(drop=True)
    test = df.loc[df['is_train'] == 0].reset_index(drop=True)

    # calc psi
    psi_results = get_all_psi(train, test, psi_cols, output_dir)

"""
    END
"""

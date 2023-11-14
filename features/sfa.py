# -*- coding: utf-8 -*-
"""
@author: shuhratjon.khalilbekov@ru.ey.com
"""
import os
from datetime import datetime
from itertools import product

import catboost as cb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from tqdm import tqdm

from configs.config import settings
from utils.basic_utils import gini


class SFA:
    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            main_sample + feature df merged.
        feature_list : str
            list of features to perform SFA.
        lgb_params : dict
            params to use for LightGBM.

        Returns
        -------
        df with SFA results

        """
        self.df = df
        self.feature_list = settings.SET_FEATURES.features_list
        self.params = settings.SFA_PARAMS.model_params
        self.cat_features = settings.SET_FEATURES.cat_feature_list

    def __run_sfa(self, feature_name):
        """
        method to run Single Factor Analysis (SFA) for a given factor
        :feature_name: feature to fit on
        """
        # get train set only
        X_train = self.df.loc[self.df['is_train'] == 1,
                              [feature_name]].reset_index(drop=True)
        y_train = self.df.loc[self.df['is_train'] == 1,
                              ['target']].reset_index(drop=True)
        y_train = y_train.astype('int')

        # init model and fit for each factor
        cb_model = cb.CatBoostClassifier(**self.params, verbose=False)

        if self.df[feature_name].dtype == object:
            model = cb_model.fit(X_train,
                                 y_train,
                                 cat_features=[feature_name])
        else:
            model = cb_model.fit(X_train,
                                 y_train,
                                 cat_features=[])
        # evaluate performance
        y_train_preds = model.predict_proba(X_train)[:, 1]
        factor_train_gini = gini(y_train, y_train_preds)

        # return results
        results = {'factor': [feature_name],
                   'gini': [round(factor_train_gini, 2)]}

        return results

    def get_sfa_results(self):
        """
        method to run sfa on all data and store results
        """
        # init empty df to store results
        output = pd.DataFrame()

        for f in tqdm(self.feature_list):
            # run sfa
            sfa_result = self.__run_sfa(f)

            # concat result to main df
            output = pd.concat([output, pd.DataFrame(sfa_result)],
                               ignore_index=True)

        # sort values for convenience
        output = output.sort_values('gini',
                                    ascending=False).reset_index(drop=True)

        current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # create the directory for the current run
        sfa_dir = os.path.join(
                    os.getcwd(),
                    settings.SET_FEATURES.output_dir,
                    f'sfa_result_{current_datetime}'
            )
        try:
            output.to_csv(f'{sfa_dir}/sfa_result.csv')

        except OSError:
            os.makedirs(sfa_dir)
            output.to_csv(f'{sfa_dir}/sfa_result.csv')

        return output

    def cramers_v(self, x, y):
        """
        metho to run correlation between categorical factos
        source: https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

        Parameters
        ----------
        x : np.array
            factor 1 series.
        y : np.array
            factor 2 series.

        Returns
        -------
        float
            value between 0 and 1.

        """
        # get cf matrix
        confusion_matrix = pd.crosstab(x, y)

        # get chisquared test result
        chi2 = chi2_contingency(confusion_matrix)[0]

        # normalize
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = np.max([0, phi2 - ((k - 1) * (r - 1)) / (n - 1)])
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)

        # corr result
        corr_result = np.sqrt(phi2corr / np.min([(kcorr - 1), (rcorr - 1)]))

        return corr_result

    def get_categories_corr(self):
        """
        method to get correaltions between categorical features

        Parameters
        ----------
        df : pd.DataFrame
            DESCRIPTION.
        qual_features_list : list
            DESCRIPTION.

        Returns
        -------
        df - matrix with correlations as values

        """
        # get combinations of factors
        combos = list(product(self.feature_list, self.feature_list))

        # init dict to store results
        output = pd.DataFrame(combos, columns=['factor1', 'factor2'])
        output['corr'] = None

        for x, y in tqdm(combos):
            # get corr
            cor = self.cramers_v(self.df[x], self.df[y])

            # save result
            output.loc[
                (output['factor1'] == x) & (output['factor2'] == y), 'corr'
            ] = cor

        # make matrix
        output['corr'] = output['corr'].astype(float)
        output_pivot = pd.pivot_table(
            output, columns='factor1', index='factor2', values='corr'
        ).reset_index()

        return output_pivot

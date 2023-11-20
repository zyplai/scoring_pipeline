from typing import List

import numpy as np
import pandas as pd

from configs.config import settings


class WoeEncoder:
    def __init__(
        self, df: pd.DataFrame, features: List, target_name: str = 'default_flag'
    ):
        self.df = df
        self.features = features
        self.target_name = target_name
        self.woe_dict = {f: None for f in features}

    def woe(self):
        """
        function to calculate woe
        """
        woes = []

        for feature_name in self.features:

            # get count, number of goods and bads
            woe_df = (
                self.df.groupby(feature_name)
                .agg({self.target_name: ['count', 'sum']})
                .reset_index()
            )
            woe_df.columns = [feature_name, 'count', 'goods']
            woe_df['bads'] = woe_df['count'] - woe_df['goods']

            # calc shares
            woe_df['goods_share'] = woe_df['goods'] / woe_df['goods'].sum()
            woe_df['bads_share'] = woe_df['bads'] / woe_df['bads'].sum()
            woe_df['shares_diff'] = woe_df['goods_share'] - woe_df['bads_share']

            # calc woe and iv
            woe_df['woe'] = np.log(woe_df['bads_share'] / woe_df['goods_share'])
            woe_df['woe'] = woe_df['woe'].replace([np.inf, -np.inf], np.nan).fillna(0)
            woe_df['iv'] = woe_df['shares_diff'] * woe_df['woe']

            # calc DR for general info
            woe_df['dr'] = woe_df['goods'] / woe_df['count']
            woe_df = woe_df.sort_values('woe', ascending=False)
            woe_df['feature_name'] = feature_name

            woe_df.rename(columns={feature_name: 'value'}, inplace=True)
            woes.append(woe_df)
            self.woe_dict[feature_name] = dict(zip(woe_df['value'], woe_df['woe']))

        # concat results for info
        output = pd.concat(woes, ignore_index=True)

        return output

    def transform_woe(self):
        """
        function to transform qualitative factor values to woe inplace
        """
        for k in self.woe_dict:

            self.df[k + '_woe'] = self.df[k].map(self.woe_dict[k]).astype(float)

        # check for nans
        if self.df[[k + '_woe' for k in self.woe_dict]].isnull().sum().sum() > 0:
            raise ValueError('There NaN in woe transformation!')

    def get_woe(self):
        """
        function to get woe and transform features
        """
        # get woe
        woe_df = self.woe()

        # transform
        final_df = self.transform_woe()

        return final_df, self.woe_dict, woe_df


class CatEncoder:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def encode_category(self, cat_feature_name: str, numeric_feature_name: str):

        mapper_output = (
            self.df.groupby(cat_feature_name)[numeric_feature_name].mean().to_dict()
        )
        assert (
            np.sum([v for k, v in mapper_output.items()]) > 0
        ), f'All mean encoders are zero!'
        return mapper_output

    def fit_transform_encoder(self, cat_feature_name: str, numeric_feature_name: str):

        cat_feature_encoder = self.encode_category(
            cat_feature_name, numeric_feature_name
        )
        self.df[f'{cat_feature_name}_encoded'] = self.df[cat_feature_name].map(
            cat_feature_encoder
        )


class TargetMeanEncoder:
    def __init__(self):
        self.feature_list = settings.SET_FEATURES.features_list
        self.cat_features = settings.SET_FEATURES.cat_feature_list
        self.mapping = {}

    def fit(self, train: pd.DataFrame) -> None:
        for col in self.cat_features:
            stats = train['target'].groupby(train[col]).agg(['mean'])
            self.mapping[col] = stats

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.cat_features:

            stats = self.mapping[col]

            df[col] = df[col].map(stats['mean'])

        return df

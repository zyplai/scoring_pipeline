import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calc_psi(population: pd.DataFrame, sample: pd.DataFrame, feature_name):
    """
    function to calculate Population Stability Index

    Parameters
    ----------
    population : pd.DataFrame
        Population sample (full dataset, main_sample.hd5).
    sample : pd.DataFrame
        train/test sample.
    feature_name : list
        feature name to calc PSI.

    Returns
    -------
    dataframe with calculated PSI on given feature name (categorical)

    """
    # define dict wit df
    data = {'population': population.copy(), 'sample': sample.copy()}
    shares = {'population': None, 'sample': None}

    # for each df cacl shares by feature name and calc psi
    for k in data.keys():
        print('calculating shares on {}'.format(k))

        # get shares
        shares[k] = (
            pd.DataFrame(
                data[k].groupby(feature_name)[feature_name].count() / len(data[k]) * 100
            )
            .rename(columns={feature_name: k})
            .reset_index()
        )

    # aggregate results in one df, then calc psi by formula
    psi_df = pd.merge(shares['population'], shares['sample'], on=feature_name)

    # psi
    psi_df['psi'] = (psi_df['population'] - psi_df['sample']) * np.log(
        psi_df['population'] / psi_df['sample']
    )

    return psi_df


def get_all_psi(
    population: pd.DataFrame, sample: pd.DataFrame, feature_names, output_dir
):
    """
    function to get results of PSI for a given range of features

    Parameters
    ----------
    population : pd.DataFrame
        DESCRIPTION.
    sample : pd.DataFrame
        DESCRIPTION.
    feature_names : TYPE
        DESCRIPTION.

    Returns
    -------
    dict with results

    """
    # init dict to store results
    output_dict = {f: None for f in feature_names}

    # init calculation
    for f in feature_names:
        print('\t calculation PSI for {}'.format(f))

        # calc PSI
        tmp = calc_psi(population, sample, f)

        # plot groupped bars and save
        tmp.set_index(f).drop('psi', axis=1).plot.bar(color=['#0069D1', '#000080'])

        # save result
        output_dict[f] = tmp.copy()

    return output_dict

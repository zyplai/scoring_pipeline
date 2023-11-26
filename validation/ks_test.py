import scipy.stats as stats


def compare_datasets(train_data, blind_data, alpha=0.05):
    """
    Compare two datasets using the Kolmogorov-Smirnov test to determine
    if they come from the same distribution.

    Args:
        train_data (pandas.DataFrame): The first dataset (train data).
        blind_data (pandas.DataFrame): The second dataset (blind data).
        alpha (float): The significance level for the hypothesis test.
        Default is 0.05.

    Returns:
        dict: A dictionary containing the column names, p-values, and
        similarity status for each numeric column.
    """

    train_numeric_cols = train_data.select_dtypes(include='number').columns
    blind_numeric_cols = blind_data.select_dtypes(include='number').columns

    train_data = train_data[train_numeric_cols].dropna()
    blind_data = blind_data[blind_numeric_cols].dropna()

    result = {}

    for column in train_data.columns:
        _, p_value = stats.ks_2samp(train_data[column], blind_data[column])

        if p_value > alpha:
            similarity = "Likely from the same distribution"
        else:
            similarity = "Likely from different distribution"

        result[column] = {"p_value": p_value, "similarity": similarity}

    return result

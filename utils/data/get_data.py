"""
load and split data set
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def get_dataset(data_path: str, index_col: str = None) -> pd.DataFrame:
    """
    Load dataset
    :param data_path: path for data
    :param index_col: name of index_column
    :return: pd.DataFrame
    """
    return pd.read_csv(data_path, index_col=index_col)


def split_data_set(data: pd.DataFrame,
                   rand: int,
                   test_size: float,
                   target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset for train and test
    :param data: your dataset
    :param rand: random_state value
    :param test_size: value of test size
    :param target_col: target column for stratify
    :return: train and test data
    """
    train, test = train_test_split(data, test_size=test_size, random_state=rand, stratify=data[target_col])

    return train, test

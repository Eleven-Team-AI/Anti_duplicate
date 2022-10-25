"""
Preparing and clean data
"""
import pandas as pd
import string
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords


def clear_string(text: str, stop_words: set) -> str:
    """
    Function for cleaning text
    :param text: input text
    :param stop_words: nltk stop words
    :return: cleaned text
    """
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text


def preparing_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preparing dataset
    :param data: your dataset
    :return: pd.DataFrame
    """
    # get stop words
    stop = set(stopwords.words('english'))

    # change types
    data.is_duplicate = data.is_duplicate.astype('float32')
    data[data.select_dtypes('object').columns] = data[data.select_dtypes('object').columns].astype(str)

    for i in data.select_dtypes('object').columns:
        data[i] = data.apply(lambda x: clear_string(x[i], stop_words=stop), axis=1)

    data.dropna(inplace=True)

    return data


def get_data_loader(data: pd.DataFrame, target_col: str, batch_size: int) -> DataLoader:
    """
    Create data loader for input model
    :param data: dataset
    :param target_col: target column
    :param batch_size: batch_size
    :return:
    """
    samples = []

    for row in range(data.shape[0]):
        label = data.iloc[row][target_col]
        inp_example = InputExample(texts=[data.iloc[row][0], data.iloc[row][1]], label=label)
        samples.append(inp_example)

    data_loader = DataLoader(samples, shuffle=True, batch_size=batch_size)

    return data_loader

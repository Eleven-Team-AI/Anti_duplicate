"""
Creating embeddings of bert
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Union


def get_embeddings(model: SentenceTransformer,
                   data: Union[str, pd.DataFrame],
                   text_col: str = None) -> np.array:
    """
    Get embedding function
    :param model: your trained model
    :param data: dataset
    :param text_col: columns with text
    :return: np.array
    """
    if type(data) != str:
        data = data[text_col].to_list()
    # get embeddings
    embeddings = model.encode(data, convert_to_tensor=True)

    return embeddings

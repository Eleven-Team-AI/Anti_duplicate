"""
Model prediction
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import util


def find_nearest_idx(query_embedding: np.array, embeddings: np.array, k: int) -> list:
    """
    Finding indices of the best texts
    :param query_embedding: query embeddings
    :param embeddings: np.array vectors to find nearest with query
    :param k: number of the nearest embeddings to find
    """

    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_k_ind = torch.topk(cos_scores, k).indices

    return [int(i) for i in top_k_ind]


def show_result(indexes: list, data: pd.DataFrame) -> None:
    """
    Show result of prediction
    :param data: dataset
    :param indexes: list of the nearest indexes
    :return: None
    """
    print('We found several similar posts:\n')
    for i in indexes:
        print(f'Record number: {i}, content {data.iloc[i][0]}')

"""
Training model
"""
from sentence_transformers import SentenceTransformer, losses
import torch
from typing import Callable
from torch.utils.data import DataLoader
import joblib


def training_saving_bert(model_name: str, dataloader: DataLoader, n_epoch: int) -> Callable:
    """
    Training model function
    :param model_name: name of model
    :param dataloader: dataloader
    :param n_epoch: number of epochs
    :return: model
    """
    torch.cuda.empty_cache()
    model = SentenceTransformer(model_name)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(dataloader, train_loss)], epochs=n_epoch, warmup_steps=100)

    return model

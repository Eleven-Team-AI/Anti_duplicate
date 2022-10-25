"""
Pipeline of data preparation and model training
"""
from .data.get_data import get_dataset
from .data.processing_data import preparing_data, get_data_loader
from .models.get_bert_embeddings import get_embeddings
from .models.train_model import training_saving_bert
from .models.predict_model import *
import yaml
import joblib


def pipline_training(config_path: str) -> None:
    """
    Full cycle of data acquisition, preprocessing and model training
    :param config_path: path to config file
    :return: None
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # get config
    data_config = config['data']
    train_config = config['train']

    # get data
    train_data = get_dataset(data_config['raw_data_path'], index_col=data_config['index_col'])

    # preprocessing
    train_data = preparing_data(train_data)

    # get data loader
    data_loader = get_data_loader(train_data,
                                  data_config['target_col'],
                                  batch_size=train_config['batch_size'])

    # train
    model = training_saving_bert(train_config['model_name'],
                                 dataloader=data_loader,
                                 n_epoch=train_config['n_epoch'])

    joblib.dump(model, train_config['model_path'])

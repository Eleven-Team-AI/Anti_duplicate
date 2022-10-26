"""
Pipeline for data preparation, model training and prediction
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
    joblib.dump(train_data, data_config['processed_data_path'])


def pipeline_create_embeddings(config_path: str) -> None:
    """
    Pipeline for create embeddings
    :param config_path: path to config file
    :return: None
    """

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # get config
    data_config = config['data']
    train_config = config['train']

    # get data
    df = get_dataset(data_config['processed_data_path'], index_col=data_config['index_col'])

    # get_model
    model = joblib.load(train_config['model_path'])

    texts = df[df.select_dtypes(object).columns].to_list()

    # get embeddings
    embedding = get_embeddings(model, texts, data_config['text_col'])

    joblib.dump(embedding, train_config['embeddings_path'])


def pipeline_prediction(config_path: str) -> None:
    """
    Prediction pipeline
    :param config_path: path to config file
    :return: None
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # get config
    data_config = config['data']
    train_config = config['train']

    query = input()

    # load staff
    embeddings = joblib.load(train_config['embeddings_path'])
    model = joblib.load(train_config['model_path'])
    df = joblib.load(data_config['processed_data_path'])

    # get query embedding
    query_emb = get_embeddings(model, query)

    indexes = find_nearest_idx(query_emb, embeddings)

    show_result(indexes, df)

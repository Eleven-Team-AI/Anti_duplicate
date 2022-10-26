Duplicate search
==============================
## Project description
Service for search duplicate in companies names. Service use BERT tokenizer and search by cosinus distance.
Our model expect that model use embindings 
## Dataset description
We use [Educaton Dataset](https://drive.google.com/file/d/1e9bdr7wcQX_YBudQcsKj-sMoIGxQOlK4/view?usp=sharing) for search duplicates.

## Start project
```bash
git clone https://github.com/Eleven-Team-AI/Anti_duplicate # clone
cd Anti_duplicate
pip install -r requirements.txt  # install
```
For start example:
1. Download [model](https://drive.google.com/file/d/1NN8536lIojlTqyoS_0XaJ6aZ-mUdH_RM/view) in path [/models](https://github.com/Eleven-Team-AI/Anti_duplicate/tree/main/models)
2. Download [prepared embeddings](https://drive.google.com/file/d/1EO_UrArhIwpcY32HA2ZQ5E5clyxFwiUW/view) in path [/data/processed](https://github.com/Eleven-Team-AI/Anti_duplicate/tree/main/data/processed) 
3. Unzip downloaded embiddings

Existing parametr for start
```bash
python3 -m src --help
```
```bash
--preprocess - parametr for create bert embedings(requier GPU)
--train - parametr for train model
--predict - parametr for predict group
```
Preprocess
```bash
python3 -m src --preprocess config.yaml
```
Train
```bash
python3 -m src --train config.yaml
```
Predict
```bash
python3 -m src --predict config.yaml #start predict example
```
Expecting result
```bash
Enter company name: jx nippon oil
We found several similar posts:
Record number: 18895, content jx nippon oil gas exploration technical service
```
```


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │      ├── BERT_MAIN.ipynb       <- Notebook with EDA and experimnets with BERT
    │      ├── EDA_&_clean_data.ipynb     <- Notebook with EDA and clean data and clasification
    │      ├── Samples.ipynb     <- Notebook clean data
    │      ├── TF_IDF experiment.ipynb     <- Notebook with experiments with tf idf vector tokenizer and log reg for detect pairs
    │      ├── Word2vec.ipynb    <- Notebook with word2vec model experiments
    │      ├── doc2vec&clustering.ipynb   <- Notebook with doc2vec model experiments and clustering
    │      └── pos_encoding&clustering.ipynb <- Notebook with GPT tokenizer experiments and clustering 
    │   
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │   
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── __main__.py    <- Main file for start module with params
    │   ├── pipelines.py   <- Main file with pipline functions
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── get_data.py
    │   │   ├── processing_data.py
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   ├── get_bert_embeddings.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── config.yaml        <- config file for training and preprocessing

## Experiments

## Metrics


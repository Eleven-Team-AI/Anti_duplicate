Duplicate search
==============================
## Project description
Service for search duplicate in companies names. Service use BERT tokenizer and search by cosinus distance.

## Dataset description
We use [Educaton Dataset] (https://drive.google.com/file/d/1e9bdr7wcQX_YBudQcsKj-sMoIGxQOlK4/view?usp=sharing) for search duplicates.

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
    │      ├── "EDA_&_find_duplicate_ipynb".ipynb       <- Notebook with EDA and experimnets with BERT
    │      ├── EDA_&_clean_data.ipynb     <- Notebook with EDA and clean data
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
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Experiments

## Metrics


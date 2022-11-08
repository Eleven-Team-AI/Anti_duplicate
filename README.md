Duplicate search
==============================
## Project description
Service for search duplicate in companies names. Service use BERT tokenizer and search by cosinus distance.
IMPORTANT!
Our model expect that in database for release exist only unique records for companies and one record match one company.
## Dataset description
We use [Educaton Dataset](https://drive.google.com/file/d/1e9bdr7wcQX_YBudQcsKj-sMoIGxQOlK4/view?usp=sharing) for search duplicates.

## Start project
```bash
python3 -m venv env
source env/bin/activate
```
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
python3 -m utils --help
```
```bash
--preprocess - parametr for create bert embedings(requier GPU)
--train - parametr for train model
--predict - parametr for predict group
```
Preprocess
```bash
python3 -m utils --preprocess config.yaml
```
Train
```bash
python3 -m utils --train config.yaml
```
Predict
```bash
python3 -m utils --predict config.yaml #start predict example
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
    │   └── processed      <- The final, canonical data sets for modeling.
    │       ├── cleaned.csv
    │       ├── test.csv
    │       └── train.csv
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
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── pipline        <- Main pipline
    │   │   └── pipelines.py
    │   └──  utils
    │       ├── data
    │       │   ├── get_data.py
    │       │   ├── processing_data.py
    │       │   └── make_dataset.py
    │       ├── models         <- Scripts to train models and then use trained models to make
    │       │   │                 predictions
    │       │   ├── predict_model.py
    │       │   ├── get_bert_embeddings.py
    │       │   └── train_model.py
    │       └── weights <- weights for models
    │
    └── config.yaml        <- config file for training and preprocessing
------------
```
## Experiments
### Model Selection

First of all, we started with the analysis and cleaning of the data, the result can be seen in `notebooks/EDA_&_clean_data.ipynb`. 
Our first idea was to solve this problem as a classification problem. We classified pairs of names in the table as duplicates and non-duplicates.
#### Sentence BERT
To create embeddings, we tried a pre-trained BERT and then re-trained it on our data.
then we searched for close embeddings using cosine distance over a predefined trash hold.
BERT model name you can see in `config.yaml` at `model_name`. Experiments and results you can see in 
`notebooks/EDA_&_clean_data.ipynb`. 
#### TF-IDF
We trained Logistic Regression on embeddings built with TF-IDF.
Experiments and results you can see in `notebooks/TF_IDF experiment.ipynb`.

### Search for dublicates

We used the classification problem to measure the metrics and choose a model for finding duplicates.
The search for duplicates is the search for top K instances that are close in cosine distance.
We chose Finetuned BERT, experiments and work examples you can see in `notebooks/BERT_MAIN.ipynb`.



## Metrics
| Embeddings    |Algorithm                   |F1            |ROC AUC       |Pres/Recall   |
| ------------- | ------------- |----------------------------|------------- |-------------|
| Pretrain BERT | closest in cosine distance |0.160         |0.846         |0.123/0.231   |
| Finetuned BERT|closest in cosine distance  |0.910         | -            |0.966/0.861   |
| TF-IDF        |Logistic Regression         |0.617         | 0.920        |0.573/0.920   |


## Speed
The measurements were made on Intel(R) Xeon(R) CPU @ 2.20GHz.

<img width="630" alt="Снимок экрана 2022-10-26 в 14 51 52" src="https://user-images.githubusercontent.com/99802770/198031037-c7d8ec02-396a-4891-b9a4-ac7cbcdeaeed.png">



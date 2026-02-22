import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')

# Label Mappings
FAKE_LABELS = ['false', 'pants-fire', 'barely-true']
TRUE_LABELS = ['true', 'mostly-true', 'half-true']

def load_data():
    """
    Reads train, valid, and test TSV files.
    Returns:
        train_df, valid_df, test_df
    """
    columns = [
        'id', 'label', 'statement', 'subjects', 'speaker', 'speaker_job', 'state', 'party',
        'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 
        'context'
    ]
    
    train = pd.read_csv(os.path.join(DATA_PATH, 'train.tsv'), sep='\t', header=None, names=columns)
    valid = pd.read_csv(os.path.join(DATA_PATH, 'valid.tsv'), sep='\t', header=None, names=columns)
    test = pd.read_csv(os.path.join(DATA_PATH, 'test.tsv'), sep='\t', header=None, names=columns)
    
    # Fill NaN for text columns
    text_cols = ['statement', 'subjects', 'speaker', 'speaker_job', 'state', 'party', 'context']
    for col in text_cols:
        train[col] = train[col].fillna('unknown').astype(str)
        valid[col] = valid[col].fillna('unknown').astype(str)
        test[col] = test[col].fillna('unknown').astype(str)

    # Fill NaN for count columns with 0
    count_cols = ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']
    for col in count_cols:
        train[col] = train[col].fillna(0)
        valid[col] = valid[col].fillna(0)
        test[col] = test[col].fillna(0)
    
    # Final dtype check
    for col in text_cols:
        train[col] = train[col].astype(str)
        valid[col] = valid[col].astype(str)
        test[col] = test[col].astype(str)
        
    for col in count_cols:
        train[col] = train[col].astype(float)
        valid[col] = valid[col].astype(float)
        test[col] = test[col].astype(float)

    return train, valid, test

def get_binary_labels(y):
    """
    Maps 6-class labels to binary (0: Fake, 1: True).
    """
    return y.apply(lambda x: 0 if x in FAKE_LABELS else 1)

def get_feature_pipeline():
    """
    Returns a ColumnTransformer for metadata preprocessing.
    Categorical: OneHotEncoder (handle_unknown='ignore')
    Numerical: StandardScaler
    """
    categorical_features = ['subjects', 'speaker', 'speaker_job', 'state', 'party', 'context']
    numerical_features = ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']

    # We already handled NaNs in load_data, so we can simplify
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' 
    )

    return preprocessor

def get_tfidf_vectorizer(corpus, max_features=5000):
    """
    Fits and returns a TF-IDF vectorizer on the given corpus.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    vectorizer.fit(corpus)
    return vectorizer

def combine_features(X_text_tfidf, X_metadata_processed):
    """
    Combines TF-IDF features and processed metadata using hstack.
    """
    return hstack([X_text_tfidf, X_metadata_processed])

######################## Libraries ########################
### Python Modules
import os
import warnings

### External Modules
# Cloud Libraries
import boto3

# Data Science Libraries
import pandas as pd

# Machine Learning Libraries
import torch

# Developer Tools
from dotenv import load_dotenv

### Internal Modules
from tools import retry_with_exponential_backoff, timeit



######################## Config ########################
# File Paramaters
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "True" # Necessary for handling parrallel tokens
load_dotenv(dotenv_path='.env.public')

# S3
s3 = boto3.resource('s3')

CONTENT_BUCKET = os.getenv('CONTENT_BUCKET')
CONTENT_KEY = os.getenv('CONTENT_KEY')

EMBEDDING_BUCKET = os.getenv('EMBEDDING_BUCKET')
EMBEDDING_KEY = os.getenv('EMBEDDING_KEY')



######################## Functions: Loading in Data  ########################

def get_data(bucket:str, key:str):
    '''
    Function to get byte data from S3 bucket
    Args:
        - bucket: str: name of the bucket
        - key: str: key of the file in the bucket
    Returns:
        - data: bytes: data from the file'''
    obj = s3.Object(bucket, key)
    data = obj.get()['Body'].read()
    return data


def get_json(bucket:str, key:str) -> pd.DataFrame:
    '''
    Function to get json data from S3 bucket
    Args:
        - bucket: str: name of the bucket
        - key: str: key of the file in the bucket
    Returns:
        - df: pd.Dataframe: data from the file in a dataframe format'''
    
    data = get_data(bucket, key)
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    df = pd.read_json(data)
    return df


def download_file(bucket:str, key:str, local_path:str):
    '''
    Function to download a file from S3 bucket
    Args:
        - bucket: str: name of the bucket
        - key: str: key of the file in the bucket
        - local_path: str: path to save the file
    Returns:
        - None'''
    
    s3.Bucket(bucket).download_file(key, local_path)

### Loading in the data from S3
def loading_content() -> pd.DataFrame:
    '''
    Function to load in the content data from S3
    Args:
        - None
    Returns:
        - df: pd.: dataframe of the content data
    '''
    df = get_json(CONTENT_BUCKET, CONTENT_KEY)
    return df


def loading_embeddings(local_path:str='embeddings.pth') -> dict:
    '''
    Function to load in the embeddings from S3
    Args:
        - local_path: str: relative path to save the embeddings
    Returns:
        - dictionary:
            tensor: BERT Embeddings
            article_id: tensor of article_ids
            unix_time: tensor of unix time when the article was created.
    '''
    download_file(EMBEDDING_BUCKET, EMBEDDING_KEY, local_path)
    return torch.load(local_path)
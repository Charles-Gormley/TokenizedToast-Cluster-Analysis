######################## Libraries ########################
### Python Modules
import pickle
import logging
import os

### Internal Modules 
from load_data import download_file
from tools import timeit

### External Modules
# Data Science Libraries
import numpy as np
import pandas as pd

# Developer Tools 
from dotenv import load_dotenv


######################## Config ########################
load_dotenv(dotenv_path='.env.public')

CLICKBAIT_BUCKET = os.getenv('CLICKBAIT_BUCKET')
VECTORIZOR_KEY = os.getenv('VECTORIZOR_KEY')
CLICKBAIT_MODEL_KEY = os.getenv('CLICKBAIT_MODEL_KEY')


######################## Class ########################
class ClickBaitFilterInference:
    def __init__(self):
        self.wv = None
        self.model = None

    def load_google_word2vec(self, local_path:str='wordvec-google-news-300.pkl'):
        '''Function to load in the google word2vec model
        Args:
            - local_path: str: local path to the model
        Returns:
            - None
        '''
        # Check if the local path exists at location if it does not, download from s3.
        if not os.path.exists(local_path):
            logging.info(f"Downloading {VECTORIZOR_KEY} from {CLICKBAIT_BUCKET} This is about 3GB & may take a while...")
            download_file(CLICKBAIT_BUCKET, VECTORIZOR_KEY, local_path)
        
        self.wv = pickle.load(open(local_path, 'rb'))

    def load_clickbait_model(self, local_path:str='clickbait-model.pkl'):
        '''Function to obtain the clickbait model from the s3 bucket
        Args:
            - model_name: str: name of the model to obtain
        Returns:
            - Model Object'''
        download_file(CLICKBAIT_BUCKET, CLICKBAIT_MODEL_KEY, local_path)

        with open(local_path, 'rb') as file:
            self.model = pickle.load(file)

    def load_models(self):
        '''Function to load in the google word2vec model and the clickbait model
        Args:
            - None
        Returns:
            - None'''
        if self.wv is None:
            self.load_google_word2vec()
        if self.model is None:
            self.load_clickbait_model()

    def text_to_vector(self, text:str):
        '''Function to convert text to a vector. This is going to be .apply(ied) to all pandas rows in column 'title'. 
        Args:
            - text: str: text to convert to a vector
        Returns:
            - vector: np.array: vector representation of the text'''
        words = text.split() 
        
        # Initialize an empty list to store vectors
        word_vectors = []

        for word in words:
            if word in self.wv.key_to_index:
                vector_representation = self.wv[word]
                word_vectors.append(vector_representation)
        
        # If we have at least one word vector, return the mean vector
        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            # Return a zero vector if none of the words were in the model's vocabulary
            return np.zeros(300)
    
    def clickbait_predict(self, df:pd.DataFrame, column_name:str='title') -> pd.Series:
        ''' This is the main Function of this class.
        Function to predict the clickbait score for each article in the dataframe
        Args:
            - df: pd.DataFrame: dataframe of articles
        Returns:
            - df: pd.DataFrame: dataframe with the clickbait score
                clickbait score is 1 if the article is clickbait and 0 if it is not'''
        self.load_models()

        # Apply the text_to_vector function to the content column
        vectors = df[column_name].apply(self.text_to_vector)

        X = pd.DataFrame(vectors.tolist(), index=df.index)
        predictions = self.model.predict(X)

        return predictions

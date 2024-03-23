######################## Libraries ########################
### Python Modules
import logging 
import sys
import argparse
import pickle
import json

### Internal Modules
from load_data import loading_content, loading_embeddings
from clickbait_filter import ClickBaitFilterInference
from sentiment_ner import SentimentNER
from partition import cluster_partition

### External Modules
# Data Science Libraries
import numpy as np
import pandas as pd

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# Clustering Libraries
from sklearn.metrics.pairwise import cosine_similarity

# Machine Learning Libraries
import torch


######################## Config ########################
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

######################## Arguments ########################
parser = argparse.ArgumentParser()
parser.add_argument('--testing', action='store_true', help='Run the script in testing mode')
parser.add_argument('--partitioning', action='store_true', help='Run the script in partitioning mode')

args = parser.parse_args()
testing = args.testing
partitioning = args.partitioning

######################## Loading in Data ########################
logging.info('Loading in Data...')

logging.info("Loading Content")
df = loading_content()

logging.info("Loading Embeddings")
embeddings = loading_embeddings()

######################## Cleaning Data ########################
logging.info('Cleaning Data...')

logging.info('Removing Duplicates')
df = df.drop_duplicates(subset='title')

logging.info("Applying Clickbait Filter")
cbf = ClickBaitFilterInference()
df["clickbait"] = cbf.clickbait_predict(df, 'title')
df = df[df["clickbait"] == 0]

logging.info(f"Removing Article IDs that exist in df but not in embeddings: {len(df)}")
embeddings_article_ids = list(np.array(embeddings['articleID']))
df = df[df["articleID"].isin(embeddings_article_ids)]
logging.info(f"Removed Article IDs that exist in df but not in embeddings: {len(df)}")

logging.info("Removing/Keeeping kept articleIDs from Embeddings")
article_ids = set(df['articleID'])
article_indices = [i for i, article_id in enumerate(embeddings['articleID']) if int(article_id) in article_ids]
for key in embeddings.keys():
    embeddings[key] = torch.tensor(embeddings[key])[article_indices]

# Remove Duplicate Article IDs from embeddings
logging.info("Removing Duplicate Article IDs from Embeddings")
non_clickbait_article_ids = df['articleID'].tolist()
non_clickbait_article_ids_set = set(non_clickbait_article_ids)
article_indices = []

# Debug: Track potentially duplicated IDs
debug_duplicate_ids_found = []

# Collect indices of non-clickbait articles
for i, article_id in enumerate(embeddings['articleID']):
    article_id = int(article_id)  # Ensure article_id is an integer
    if article_id in non_clickbait_article_ids_set:
        if article_id in debug_duplicate_ids_found:
            pass
        else:
            debug_duplicate_ids_found.append(article_id)
            article_indices.append(i)

for key in embeddings.keys():
    embeddings[key] = torch.tensor(embeddings[key][article_indices])



# Assert that df_final and embeddings now have the same length for 'articleID'
assert len(df) == len(embeddings['articleID']), "Mismatch in length after filtering and matching."

# Asset that embeddings and df are the same length 
assert len(df) == len(embeddings['articleID'])
######################## Article-Level Features ########################
logging.info("Applying Article-Level Features")

logging.info("Applying Article Length")
df['article_length'] = df['content'].apply(lambda x: len(x.split()))

logging.info("Applying Vader Sentiment Analysis")
sid = SentimentIntensityAnalyzer()
df["vader_sentiment"] = df["content"].apply(lambda x: sid.polarity_scores(x))

logging.info("Applying TextBlob Sentiment Analysis")
textblob_series = df["content"].apply(lambda x: TextBlob(x).sentiment)
df["subjectivity"] = textblob_series.apply(lambda x: x.subjectivity)
df["polarity"] = textblob_series.apply(lambda x: x.polarity)

######################## Entity-Level Features ########################
logging.info("Applying Entity-Level Features")

logging.info("Initializing SentimentNER")
sentiment_ner = SentimentNER()
content_annotations = sentiment_ner.annotate_series(df['content'])

logging.info("Extracting Unique Entity Words")
processed_content_annotations = sentiment_ner.process_annotations(content_annotations)

content_entities = sentiment_ner.extract_entity_words_with_duplicates(processed_content_annotations)

######################## TFIDF-Vectorization ########################
logging.info("Applying TFIDF-Vectorization")

logging.info("Initializing TFIDF Vectorizer")
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

logging.info("Fitting TFIDF Vectorizer & Similarity Matrix  with Content")
tfidf_matrix_content = tfidf_vectorizer.fit_transform(df['content'])
similarity_matrix_content = cosine_similarity(tfidf_matrix_content)
content_weight = 0.4882

logging.info("Fitting TFIDF Vectorizer & Similarity Matrix with Title")
tfidf_matrix_title = tfidf_vectorizer.fit_transform(df['title'])
similarity_matrix_title = cosine_similarity(tfidf_matrix_title)
title_weight = 0.32988

logging.info("Fitting TFIDF Vectorizer & Similarity Matrix with unique_content_entities ")
tfidf_matrix_content_entities = tfidf_vectorizer.fit_transform(content_entities)
similarity_matrix_content_entities = cosine_similarity(tfidf_matrix_content_entities)

logging.info("Fitting similarity Matrix with BERT Embeddings")
similarity_matrix_embeddings = cosine_similarity(embeddings['tensor'])
embeddings_weight = 0.1818

logging.info("Saving Data")
df.to_csv('df.csv')
torch.save(embeddings, 'embeddings.pt')

######################## Saving Data ########################
with open('content_annotations.pkl', 'wb') as f:
    pickle.dump(content_annotations, f)

# Save the 3 similarity matrices
with open('similarity_matrix_title.pkl', 'wb') as f:
    pickle.dump(similarity_matrix_title, f)

with open('similarity_matrix_content.pkl', 'wb') as f:
    pickle.dump(similarity_matrix_content, f)

with open('similarity_matrix_embeddings.pkl', 'wb') as f:
    pickle.dump(similarity_matrix_embeddings, f)

with open('similarity_data_content_entities.pkl', 'wb') as f:
    pickle.dump(similarity_matrix_content_entities, f)

sentiment_ner = SentimentNER()

######################## Loading in Data ########################
logging.info('Loading in Data...')
df = pd.read_csv('df.csv')
embeddings = torch.load('embeddings.pt')

with open('content_annotations.pkl', 'rb') as f:
    content_annotations = pickle.load(f)

with open('similarity_matrix_title.pkl', 'rb') as f:
    similarity_matrix_title = pickle.load(f)
title_weight = 0.32988

with open('similarity_matrix_content.pkl', 'rb') as f:
    similarity_matrix_content = pickle.load(f)
content_weight = 0.4882

with open('similarity_matrix_embeddings.pkl', 'rb') as f:
    similarity_matrix_embeddings = pickle.load(f)
embeddings_weight = 0.1818

with open('similarity_data_content_entities.pkl', 'rb') as f:
    similarity_matrix_content_entities = pickle.load(f)

######################## Partition Data ########################
logging.info("Combining Similarity Matrices")
similarity_matrix = content_weight * similarity_matrix_content + \
                    title_weight * similarity_matrix_title + \
                    embeddings_weight * similarity_matrix_embeddings
resolution = 0.2123
z_score = 17.71774


logging.info("Clustering Data")
partition_data = cluster_partition(z_score, resolution, similarity_matrix, similarity_matrix_content_entities, df, embeddings)

######################## Opinion Extraction ########################
logging.info("Extracting Opinions")

# Ensure each cluster has suitable ner similarity
clusters_objects = sentiment_ner.extract_opinions(partition_data, df)
with open('clusters_objects.json', 'w') as f:
    json.dump(clusters_objects, f)

######################## Libraries ########################
### Python Modules
import logging
import os
import json
from collections import defaultdict
import re

### Internal Modules
from tools import timeit, retry_with_exponential_backoff

### External Modules
# Data Science Libraries
import pandas as pd
import numpy as np

# NLP Libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Machine Learning Libraries
import torch
from torch.nn import DataParallel
from sklearn.cluster import MeanShift, estimate_bandwidth
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel

# Developer Tools
from dotenv import load_dotenv


######################## Config ########################
load_dotenv(dotenv_path='.env.public')
NER_MODEL_HF = os.getenv('NER_MODEL_HF')

with open('hf-ner-labels.json', 'r') as file:
    labels = json.load(file)


######################## Class ########################
class SentimentNER:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
        self.sid = SentimentIntensityAnalyzer()

        self.load_ner_model()
        self.load_pipeline()

    
    def load_ner_model(self):
        """Loads the NER model and tokenizer from the Hugging Face model hub."""
        self.tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_HF)
        self.model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_HF)
        self.model.to(self.device)

    def load_pipeline(self):
        """Initializes the NER pipeline with the loaded model and tokenizer."""
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=0)
        self.model = DataParallel(self.model)

    
    def annotate_series(self, series: pd.Series, batch_size: int = 64):
        """Annotates a pandas Series with NER tags using the loaded pipeline."""
        annotations = self.pipeline(series.tolist(), batch_size=batch_size)
        return annotations

    
    def process_annotations(self, annotations, min_threshold: float = 0.85):
        """Processes the annotations to extract complete entities."""
        complete_entities = []
        for content_annotation in annotations:
            content_entities, current_entity = [], None
            for annotation in content_annotation:
                entity_type = annotation['entity']
                if entity_type.startswith('B-'):
                    current_entity = self._finalize_entity(content_entities, current_entity, min_threshold)
                    current_entity = self._start_new_entity(annotation)
                elif entity_type.startswith('I-'):
                    current_entity = self._update_current_entity(current_entity, annotation)
                elif entity_type == 'O':
                    current_entity = self._finalize_entity(content_entities, current_entity, min_threshold)
            self._finalize_entity(content_entities, current_entity, min_threshold)
            complete_entities.append(content_entities)
        self._calculate_and_save_entity_metrics(complete_entities)
        return complete_entities

    # Utility methods are marked as private with a leading underscore
    def _add_to_entity_list(self, entity_list: list, entity, min_threshold: float = 0.85):
        """Adds an entity to the list if it meets the minimum score threshold."""
        if entity is not None and entity["score"] > min_threshold:
            entity_list.append(entity)

    def _start_new_entity(self, annotation: dict):
        """Starts a new entity based on the annotation."""
        base_entity_type = annotation['entity'][2:]  # Remove the 'B-' prefix
        return {
            'type': base_entity_type,
            'word': annotation['word'].replace('##', ''),
            'score': annotation['score'],
            "start": annotation['start'],
            "end": annotation['end']
        }

    def _update_current_entity(self, current_entity: dict, annotation: dict):
        """Updates the current entity with information from the annotation."""
        if current_entity is not None:
            current_entity['word'] += annotation['word'].replace('##', '')
            current_entity['score'] *= annotation['score']
            current_entity['end'] = annotation['end']
        return current_entity

    def _finalize_entity(self, content_entities, current_entity: dict, min_threshold: float):
        """Finalizes the current entity and resets for the next one."""
        self._add_to_entity_list(content_entities, current_entity, min_threshold)
        return None  # Reset current_entity

    def _calculate_and_save_entity_metrics(self, complete_entities):
        """Calculates and logs entity metrics from the annotations."""
        entity_counts = [len(entities) for entities in complete_entities]
        logging.debug(f"Average Entity Count: {np.mean(entity_counts)}")
        logging.debug(f"Median Entity Count: {np.median(entity_counts)}")
        logging.debug(f"Max Entity Count: {np.max(entity_counts)}")
        logging.debug(f"Min Entity Count: {np.min(entity_counts)}")

    
    def extract_unique_entity_words(self, complete_entities):
        """
        Extracts and deduplicates entity words from the processed annotations.
        
        Args:
            complete_entities (List[List[Dict]]): A list of lists, where each inner list contains dictionaries
            of entity information for each content piece.
        
        Returns:
            List[List[str]]: A list of lists, where each inner list contains unique entity words for each content piece.
        """
        all_unique_entity_words = []

        for content_entities in complete_entities:
            unique_words = set()

            for entity in content_entities:
                # Add the 'word' from each entity dictionary to the set, ensuring uniqueness
                unique_words.add(entity['word'])

            # Convert the set of unique words for this content piece into a list and append to the master list
            all_unique_entity_words.append(list(unique_words))

        return all_unique_entity_words
    
    
    def extract_entity_words_with_duplicates(self, complete_entities):
        """
        Extracts entity words from the processed annotations, allowing duplicates within each content piece.

        Args:
            complete_entities (List[List[Dict]]): A list of lists, where each inner list contains dictionaries
            of entity information for each content piece.

        Returns:
            List[List[str]]: A list of lists, where each inner list contains entity words for each content piece,
            including duplicates.
        """
        all_entity_words_with_duplicates = []

        for content_entities in complete_entities:
            entity_words = []

            for entity in content_entities:
                # Add the 'word' from each entity dictionary to the list, allowing duplicates
                entity_words.append(entity['word'])

            # Append the list of entity words for this content piece to the master list
            all_entity_words_with_duplicates.append(' '.join(entity_words))

        return all_entity_words_with_duplicates
    

    def calculate_sentiments_for_unique_entities(self, complete_entities):
        """
        Incrementally calculates and averages sentiment scores for entities within the same content piece.
        """
        updated_entities_with_averaged_sentiments = []

        for entities in complete_entities:
            # Store the running averages and counts
            entity_info = defaultdict(lambda: {'polarity': 0, 'subjectivity': 0, 'compound': 0, 'count': 0})

            for entity in entities:
                entity_text = entity['word']
                
                # Calculate current sentiment scores
                vader_scores = self.sid.polarity_scores(entity_text)
                textblob_scores = TextBlob(entity_text).sentiment
                
                # Update running averages
                info = entity_info[entity_text]
                cur_count = info['count'] + 1
                info['polarity'] = (info['polarity'] * info['count'] + textblob_scores.polarity) / cur_count
                info['subjectivity'] = (info['subjectivity'] * info['count'] + textblob_scores.subjectivity) / cur_count
                info['compound'] = (info['compound'] * info['count'] + vader_scores['compound']) / cur_count
                info['count'] = cur_count

            # Prepare the final list of entities with averaged sentiment scores
            averaged_entities_list = [
                {'word': text, 'polarity': info['polarity'], 'subjectivity': info['subjectivity'], 'compound': info['compound'], 'article_id': entities[0]['article_id']}
                for text, info in entity_info.items()
            ]

            updated_entities_with_averaged_sentiments.append(averaged_entities_list)

        return updated_entities_with_averaged_sentiments
    
    def calculate_ner_similarity_score(self, entities_i, entities_j):
        """
        Calculates the similarity score between two lists of entities.

        Args:
            entities_i (List[Dict]): A list of entity dictionaries.
            entities_j (List[Dict]): A list of entity dictionaries.

        Returns:
            float: The similarity score between the two entity lists.
        """
        # Extract the unique entity words from each list
        
        unique_words_i = set(entities_i)
        unique_words_j = set(entities_j)

        # Calculate the Jaccard similarity between the two sets of unique words
        intersection = len(unique_words_i.intersection(unique_words_j))
        if (intersection != 0):
            return True
        return False
        
    def evaluate_inner_cluster_ner_similarity(self, unique_entity_list:list[str], ner_similarity_threshold:float=0.2,):
        """
        Evaluates the similarity between article's unique entities within a cluster.

        Args:
            processed_entities (List[List[Dict]]): A list of lists, where each inner list contains dictionaries
            of entity information for each content piece.

        Returns:
            float: The similarity score between entities in the cluster.
        """
        similarity_scores = []

        # Compare each object with every other object in the cluster
        for i, entities_i in enumerate(unique_entity_list):
            for j, entities_j in enumerate(unique_entity_list):
                if i != j:
                    # Calculate similarity between the two objects
                    similar = self.calculate_ner_similarity_score(entities_i, entities_j)
                    if not similar:
                        return False
        
        return True

    def find_common_entities(self, list_of_entity_lists):
        """
        Takes a list of lists, where each inner list contains unique entities, and returns a set
        of entities that exist in all of the original lists.

        Args:
            list_of_entity_lists (List[List[str]]): A list of lists, where each inner list contains unique entities.

        Returns:
            Set[str]: A set of entities that are common across all provided entity lists.
        """
        # Convert the first list to a set to start with
        if not list_of_entity_lists:
            return set()  # Return an empty set if the input is empty
        
        common_entities = set(list_of_entity_lists[0])

        # Iterate over the rest of the lists and update the common_entities set
        for entity_list in list_of_entity_lists[1:]:
            common_entities.intersection_update(entity_list)

        return common_entities
    
    def retrieve_sentences_surrounding_entity(self, content:str, entity_start:int, entity_end:int, window_size:int = 1) -> list:
        '''
        This function retrieves the sentences surrounding the entity within a window size, 
        taking into account potential spacing issues around the entity.

        Args:
            content (str): The content of the article
            entity_start (int): The start index of the entity within the content
            entity_end (int): The end index of the entity within the content
            window_size (int): The size of the window to retrieve sentences around the entity

        Returns:
            list: A list of sentences surrounding the entity
        '''
        # Split the content into sentences using a regex that considers periods, exclamation points, and question marks.
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Initialize variables to keep track of character indices and the target sentence's index
        running_index = 0
        entity_sentence_indices = []

        for i, sentence in enumerate(sentences):
            sentence_end_index = running_index + len(sentence)
            
            # Check if the entity overlaps with the current sentence
            if entity_start < sentence_end_index and entity_end > running_index:
                entity_sentence_indices.append(i)
            
            # Update the running index for the start of the next sentence, including skipped spaces
            running_index = sentence_end_index + 1  # Adjust for the space after the sentence
        
        if not entity_sentence_indices:
            return []  # Entity not found within the sentences

        # Determine the range of sentences to include, based on the first and last occurrences of the entity
        start_index = max(0, min(entity_sentence_indices) - window_size)
        end_index = min(len(sentences), max(entity_sentence_indices) + window_size + 1)
        
        # Retrieve the surrounding sentences based on the calculated range
        surrounding_sentences = sentences[start_index:end_index]
        
        return surrounding_sentences
                
    def calculate_sentiments_for_common_cluster_entities(self, common_entities, processed_entities, original_indices:list[int], df:pd.DataFrame):
        '''
        Function to calculate the sentiment scores for the common entities in the cluster
        
        Args:
            - common_entities: set: set of common entities in the cluster
            - processed_entities: list: list of processed entities in the cluster
        Returns:
            - list: list of entities with sentiment scores, polarity, and subjectivity scores, along with article_id
        '''
        updated_entities_with_averaged_sentiments = []

        for i, entities in enumerate(processed_entities):
            # Store the running averages and counts
            entity_info = defaultdict(lambda: {'polarity': 0, 'subjectivity': 0, 'compound': 0, 'count': 0})

            content = str(df[df['index'] == original_indices[i]]['content'])
            article_id = df[df['index'] == original_indices[i]]['articleID']
            title = df[df['index'] == original_indices[i]]['title']

            for entity in entities:
                if entity['word'] not in common_entities:
                    continue

                entity_text = entity['word']
                retrieved_sentences = self.retrieve_sentences_surrounding_entity(content, entity['start'], entity['end'])
                surrounding_sentences_str = " ".join(retrieved_sentences)
                print(surrounding_sentences_str)
                print("surrounding sentence ^^")

                # Apply the sentiment scores to the surrounding sentences
                vader_scores = self.sid.polarity_scores(surrounding_sentences_str)

                textblob_scores = TextBlob(surrounding_sentences_str).sentiment
                
                # Update running averages
                info = entity_info[entity_text]
                cur_count = info['count'] + 1
                info['polarity'] = (info['polarity'] * info['count'] + textblob_scores.polarity) / cur_count
                info['subjectivity'] = (info['subjectivity'] * info['count'] + textblob_scores.subjectivity) / cur_count
                info['compound'] = (info['compound'] * info['count'] + vader_scores['compound']) / cur_count
                info['count'] = cur_count

            # Prepare the final list of entities with averaged sentiment scores
            averaged_entities_list = [
                {'word': text, 'polarity': info['polarity'], 'subjectivity': info['subjectivity'], 'compound': info['compound'], 'article_id': article_id}
                for text, info in entity_info.items()
            ]

            updated_entities_with_averaged_sentiments.append(averaged_entities_list)

        return updated_entities_with_averaged_sentiments

    def get_clusters_with_sentiments(self, clusters, annotations, df:pd.DataFrame, initial_ner_threshold=0.85, max_iterations=3, ner_similarity_threshold=0.1):
        """
        Processes annotations for each cluster, adjusting thresholds and pruning outliers based on similarity.

        Args:
            clusters (List[List[Dict]]): A list of clusters, each containing content annotations.
            initial_threshold (float): Initial threshold for processing annotations.
            max_iterations (int): Maximum iterations for threshold adjustment.
            improvement_threshold (float): Minimum improvement required to continue iterations.
        """

        deleted_clusters = 0
        processed_clusters = 0
        clusters_with_sentiments = []

        for cluster in clusters:
            if len(cluster) < 2:
                continue  # Skip clusters smaller than the minimum size
            
            original_indices = [article_metadata['original_index'] for article_metadata  in cluster]
            processed_clusters += 1
            content_annotations = [annotations[original_index] for original_index in original_indices]

            current_ner_threshold = initial_ner_threshold
            iteration = 0
            while iteration < max_iterations:
                # Process (or re-process) annotations for the entire cluster
                processed_entities = self.process_annotations(content_annotations, current_ner_threshold)
                unique_entity_list = self.extract_unique_entity_words(processed_entities)
                
                # Evaluate similarity between entities in the cluster
                similarity_score_check = self.evaluate_inner_cluster_ner_similarity(unique_entity_list, ner_similarity_threshold)

                if not similarity_score_check:
                    # Adjust the NER threshold and re-process annotations
                    current_ner_threshold -= 0.2 * iteration
                    iteration += 1

                else:
                    # If this is the case then we should extract sentiments from the cluster and save a new type of cluster object:
                    #AxExS where A is the article, E is the entity and S is the sentiment. There will be one of these matrices for each cluster.
                    cluster_common_entities = self.find_common_entities(unique_entity_list)
                    cluster_sentiments = self.calculate_sentiments_for_common_cluster_entities(cluster_common_entities, processed_entities, original_indices, df)
                    clusters_with_sentiments.append(cluster_sentiments)

                    break  # Similarity is acceptable, no need for further iterations

                if iteration == max_iterations: # If we reach the maximum iterations we are just going to delete the cluster
                    deleted_clusters += 1

        logging.debug(f"Deleted {deleted_clusters} clusters due to low NER similarity.")
        logging.debug(f"Processed Clusters: {processed_clusters}")
        logging.debug(f"Total Clusters: {len(clusters)}")
        return clusters_with_sentiments
    
    def extract_opinions(self, clusters:list, df:pd.DataFrame) -> list[dict]:


        cluster_objects = []
        for i, cluster in enumerate(clusters):
            if len(cluster) < 2:
                continue 
            
            article_indices = [article_metadata['articleID'] for article_metadata in cluster]
            cluster_df = df[df['articleID'].isin(article_indices)][["articleID", "vader_sentiment", "subjectivity", "polarity", "title"]]
            
            compound_values = []
            for i in range(len(cluster_df['vader_sentiment'])):
                compound_values.append(eval((cluster_df['vader_sentiment'].iloc[i]))["compound"])
            cluster_df['vader_compound'] = compound_values

            features = cluster_df[['vader_compound', 'subjectivity', 'polarity']].values
            
            try: 
                bandwidth = estimate_bandwidth(features, quantile=0.1, n_samples=500)
                if bandwidth == 0.0:
                    # Set a default bandwidth or handle the case appropriately
                    bandwidth = 1  # or set a specific float value

                mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                mean_shift.fit(features)
                labels = mean_shift.labels_
            except:
                labels = [0] * len(features)

            cluster_df['cluster_label'] = labels

            cluster_object = {"cluster_id": i}

            for label, group in cluster_df.groupby('cluster_label'):
                # Each article's data is stored as a dictionary in the list for the corresponding cluster label
                cluster_object[f"Opinion-{label}"] = group.apply(lambda row: {"articleID": row['articleID'],
                                                                "vader_sentiment": row['vader_sentiment'],
                                                                "subjectivity": row['subjectivity'],
                                                                "polarity": row['polarity'],
                                                                    "title": row['title'],}, axis=1).tolist()
            
            cluster_objects.append(cluster_object)
        return cluster_objects

                
                
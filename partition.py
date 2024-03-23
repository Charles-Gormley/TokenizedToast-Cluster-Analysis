
import json
import networkx as nx
import numpy as np
import logging
import time
import pandas as pd
from community import community_louvain

# TODO: This should be a class in the future where we can easily swap in and out different clustering, similarity weights, matrices, etc.

def cluster_partition(z_score, resolution, similarity_matrix, similarity_matrix_content_entities, df, embeddings, ):
    logging.info("Initializing Graph")
    G = nx.Graph()
    logging.info("Adding Nodes to Graph")
    for i in range(len(embeddings['articleID'])):
        article_id = int(embeddings['articleID'][i])
        try: 
            article_data = df[df['articleID'] == article_id].iloc[0]
            G.add_node(article_id, **article_data.to_dict()) # Attaching metadata to nodes
        except:
            pass 
    
    logging.info("Adding Edges to Graph")
    for i in range(similarity_matrix.shape[0]): # TODO: Optimize this code
        mean = np.mean(similarity_matrix[i])
        std = np.std(similarity_matrix[i])

        max_thresh = .999
        min_thresh = mean + (z_score*std)

        # Division by zero check.
        if (max_thresh < min_thresh):
            min_thresh = max_thresh - .1

        min_thresh_ner_tfidf = .10
                
        
        for j in range(i + 1, similarity_matrix.shape[1]): # Starting at i + 1 to avoid redundant calculations
            
            if similarity_matrix[i, j] > min_thresh and similarity_matrix[i, j]< max_thresh and similarity_matrix_content_entities[i, j] > min_thresh_ner_tfidf :
                cur_article_id = int(embeddings['articleID'][i])
                target_article_id = int(embeddings['articleID'][j])

                normalized_weight = (max_thresh-similarity_matrix[i, j])/(max_thresh-min_thresh)

                G.add_edge(cur_article_id, target_article_id, weight=normalized_weight) # TODO: Adjust the strength of the edge or the distance of the edge based on the simialiarity


    # partition data with resolution 0.97
    logging.info("Clustering Data")
    partition = community_louvain.best_partition(G, resolution=resolution)

    # Partition
    # Saving Partition
    
    df['index'] = df.index
    df_with_index = df.rename(columns={'index': 'original_index'})

    # Saving Partition
    logging.info("Saving Partition")
    partition_df = pd.DataFrame(partition.items(), columns=['articleID', 'partition'])

    # Merge while ensuring the original index is included.
    merge_df = pd.merge(partition_df, df_with_index, on='articleID')

    partition_list = []

    for partition_id in merge_df['partition'].unique():
        # Ensure to grab the original index along with the title for each partition.
        subset = merge_df[merge_df['partition'] == partition_id][["original_index", "articleID"]].to_dict('records')

        
        partition_list.append(subset)
        
    # Save the partition list to json
    cur_time = str(round(time.time()))
    with open(f'partition-{cur_time}.json', 'w') as f:
        json.dump(partition_list, f)
    
    return partition_list
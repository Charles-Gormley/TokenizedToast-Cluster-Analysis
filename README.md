# TokenizedToast-Cluster-Analysis
This toolkit is designed to automate the process of analyzing and partitioning textual content, particularly focusing on articles or segments for podcast preparation. It leverages a combination of natural language processing (NLP), sentiment analysis, and clustering algorithms to filter, analyze, and organize content based on various linguistic and semantic features.

## Key Features
Clickbait Filtering: Automatically identifies and filters out clickbait content to ensure the quality and relevance of the articles being processed.
Sentiment Analysis: Utilizes both VADER Sentiment and TextBlob for comprehensive sentiment analysis, providing insights into the emotional tone of the content.
Named Entity Recognition (NER): Extracts entities and their related sentiments to highlight significant topics and opinions within the content.
TFIDF Vectorization and Cosine Similarity: Analyzes textual content to identify thematic similarities between different articles or segments, supporting the clustering process.
Clustering and Partitioning: Employs clustering algorithms to organize content into coherent groups, facilitating targeted content recommendation and segmentation.
##Installation
Ensure you have Python 3.x installed along with the following packages: numpy, pandas, nltk, sklearn, torch, vaderSentiment, and textblob. Most dependencies can be installed via pip install -r requirements.txt

## Workflow
The toolkit's workflow comprises loading data, cleaning and filtering, applying NLP techniques for feature extraction, vectorizing content for similarity analysis, and finally clustering and partitioning the content based on similarity and sentiment features.

## Future Directions
Continuous Content Recommendations: Aiming to evolve from batch processing to a more dynamic, real-time recommendation system, enhancing content relevancy and personalization.
Enhanced NER and Sentiment Analysis: Integrating more sophisticated NER models and sentiment analysis techniques for deeper content insights.
Scalability and Performance Improvements: Optimizing algorithms and data processing steps to handle larger datasets more efficiently.

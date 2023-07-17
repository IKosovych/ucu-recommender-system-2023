import sys
sys.path.append('../ucu-recommender-system-2023/')

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from data_uploader.uploader import DataUploader

class ContentBasedFiltering:
    """
    A Content-Based Filtering algorithm
    """

    def __init__(self):
        data_uploader = DataUploader()
        self.movies, _, _ = data_uploader.get_user_item_data_surprise()

    def fit(self):
        # Use TF-IDF to convert the genres into vectors
        tfidf = TfidfVectorizer(stop_words='english')
        self.movies['genres'] = self.movies['genres'].fillna('')
        tfidf_matrix = tfidf.fit_transform(self.movies['genres'])

        # Compute the cosine similarity matrix
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Create a reverse mapping of movie titles and DataFrame indices
        self.indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()

    def get_recommendations(self, title):
        # Get the index of the movie that matches the title
        idx = self.indices[title]

        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return self.movies['title'].iloc[movie_indices]

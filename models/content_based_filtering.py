import sys

sys.path.append('../ucu-recommender-system-2023/')

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split


class ContentBasedFiltering:
    """
    A content-based filtering algorithm based on movie genres
    """

    def __init__(self, data_uploader, train_size=0.75):
        self.movies, self.ratings = data_uploader.df_movies, data_uploader.df_ratings

        self.movies['genres'] = self.movies['genres'].fillna('')
        self.indices = pd.Series(self.movies.index, index=self.movies['movieId']).drop_duplicates()

        # Split ratings data into training and testing sets
        self.train_set, self.test_set = train_test_split(self.ratings, test_size=1 - train_size, random_state=42,
                                                         stratify=self.ratings['userId'])

    def fit(self, save=True, file_name=None):
        # Use TF-IDF to convert the genres into vectors
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies['genres'])

        # Compute the cosine similarity matrix
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        if save:
            with open(file_name, 'wb') as handle:
                pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, file_name):
        with open(file_name, 'rb') as handle:
            tmp_dict = pickle.load(handle)
            self.__dict__.update(tmp_dict)

    def predict(self, user_id):
        # Get the movies that the user has watched
        watched_movies = self.train_set[self.train_set['userId'] == user_id]['movieId'].tolist()

        # Get the pairwise similarity scores of all movies with each movie the user has watched
        sim_scores = [list(enumerate(self.cosine_sim[self.indices[movie]])) for movie in watched_movies]

        # Flatten the list and sort it based on the similarity scores
        sim_scores = sorted([score for sublist in sim_scores for score in sublist], key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return self.movies['title'].iloc[movie_indices]

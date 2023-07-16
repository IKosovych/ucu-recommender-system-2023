import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedFiltering:
    """
    A content-based filtering algorithm based on movie genres
    """

    def __init__(self):
        self.movies = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/ml-latest-small/movies.csv')
        self.movies['genres'] = self.movies['genres'].fillna('')
        self.indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()

    def fit(self):
        # Use TF-IDF to convert the genres into vectors
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies['genres'])

        # Compute the cosine similarity matrix
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

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

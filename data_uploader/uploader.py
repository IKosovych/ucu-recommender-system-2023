import sys

sys.path.append('../ucu-recommender-system-2023/')

import pandas as pd
import surprise
from surprise import Dataset, Reader


class DataUploader:
    def __init__(self):
        self.df_movies = pd.read_csv(
            'data/ml-1m/movies.dat',
            sep="::",
            names=['movieId', 'title', 'genres'],
            encoding='latin-1',
            engine='python',
        )
        self.df_ratings = pd.read_csv(
            'data/ml-1m/ratings.dat',
            sep="::",
            names=['userId', 'movieId', 'rating', 'timestamp'],
            engine='python',
        )

        ratings_dict = {'itemID': list(self.df_ratings.movieId),
                        'userID': list(self.df_ratings.userId),
                        'rating': list(self.df_ratings.rating)}
        self.df = pd.DataFrame(ratings_dict)

        reader = Reader(rating_scale=(0.5, 5.0))
        self.dataset = Dataset.load_from_df(self.df[['userID', 'itemID', 'rating']], reader)

        self.train_set = None
        self.eval_set = None

    def split_train_eval_sets(self, test_size=0.25):
        self.train_set, self.eval_set = surprise.model_selection.train_test_split(self.dataset,
                                                                                  test_size=test_size)

import sys
sys.path.append('../ucu-recommender-system-2023/')

import pandas as pd
import surprise
from surprise import Dataset, Reader


class DataUploader:
    def __init__(self):
        self.df_movies = pd.read_csv('data/ml-latest-small/movies.csv')
        self.df_ratings = pd.read_csv('data/ml-latest-small/ratings.csv')

        ratings_dict = {'itemID': list(self.df_ratings.movieId),
                        'userID': list(self.df_ratings.userId),
                        'rating': list(self.df_ratings.rating)}
        self.df = pd.DataFrame(ratings_dict)
        reader = Reader(rating_scale=(0.5, 5.0))
        self.data = Dataset.load_from_df(self.df[['userID', 'itemID', 'rating']], reader)

    def get_user_item_data_surprise(self, test_size=0.25):
        train_set, test_set = surprise.model_selection.train_test_split(self.data, test_size=test_size)
        return self.df_movies, train_set, test_set  


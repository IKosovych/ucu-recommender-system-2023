import sys
sys.path.append('../ucu-recommender-system-2023/')

import numpy as np
import pandas as pd
from data_uploader.uploader import DataUploader

class BaselineModel:
    """
    A baseline recommendation model that recommends movies 
    based on their popularity and average ratings
    """
    def __init__(self):
        data_uploader = DataUploader()
        _, self.train_set, self.test_set = data_uploader.get_user_item_data_surprise()

    def fit(self):
        self.train_set['count'] = 1
        df_ratings_aggr = self.train_set.groupby('movieId')[['rating', 'count']].sum()
        df_ratings_aggr['rating_avg'] = round(df_ratings_aggr['rating'] / df_ratings_aggr['count'],1)
        df_ratings_aggr['rank'] = np.log(df_ratings_aggr['rating']) * df_ratings_aggr['rating_avg']

        self.df_ranked_films = pd.merge(self.train_set, df_ratings_aggr[['rank', 'rating_avg']], on='movieId', how='outer') \
            .sort_values(by=['rank'], ascending=False) \
            .fillna(0)

    def predict_on_testset(self, user_id):
        filtered_films = self.df_ranked_films.merge(self.train_set[self.train_set['userId'] == user_id]['movieId'], 
                                                     on='movieId', 
                                                     how='left', 
                                                     indicator=True)

        filtered_films = filtered_films[filtered_films['_merge'] == 'left_only']
        return filtered_films['movieId'].head(10)

import sys
sys.path.append('../ucu-recommender-system-2023/')

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import surprise
from surprise import Dataset, Reader

class ABtestDistributor:
    def __init__(self, data_uploader):
        # Use the DataUploader instance
        self.df_movies = data_uploader.df_movies
        self.df_ratings = data_uploader.df_ratings
        self.data = data_uploader.data  # Assuming data_uploader has a 'data' attribute

        self.user_list = pd.DataFrame()
        self.model_a = ''
        self.model_b = ''

    def split_users(self, model_a, model_b, split=0.3, n_clusters=2):
        # create base list of users
        self.model_a = model_a
        self.model_b = model_b
        self.user_list['userId'] = self.df_ratings['userId'].unique()
        self.user_list['model'] = model_a

        feature_list = self.calculate_user_features()
        self.cluster_users(feature_list, n_clusters)

        for n in range(n_clusters):
            ids = self.user_list[self.user_list['user_type'] == n]['userId']
            num_elements = int(split * len(ids))
            
             # Randomly select 30% of elements from the array
            selected_elements = np.random.choice(ids, size=num_elements, replace=False)

            self.user_list.loc[self.user_list['userId'].isin(selected_elements), 'model'] = model_b

    def user_exists(self, user_id):
        return not self.user_list[self.user_list['userId'] == user_id].empty

    def get_model_name(self, user_id):
        return self.user_list[self.user_list['userId'] == user_id].model.values[0]


    def cluster_users(self, features, n_clusters=2):
        X = self.user_list[features].to_numpy(dtype='int')

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        self.user_list['user_type'] = pd.Series(kmeans.labels_)

    def calculate_user_features(self):
        self.user_list = self.user_list.set_index('userId')
        self.user_list['total_rating_count'] = self.df_ratings.groupby('userId')['rating'].count()
        self.user_list['total_rating_sum'] = self.df_ratings.groupby('userId')['rating'].sum()
        self.user_list = self.user_list.reset_index()

        return ['total_rating_count', 'total_rating_sum']

    def log_interaction(self, user_id, movie_id, action):
        # Logs for evaluation, action could be 'watch', 'click', 'rate', etc.
        # For simplicity, we just append logs to a CSV.
        with open('interaction_log.csv', 'a') as file:
            file.write(f"{user_id},{movie_id},{action}\n")

    def evaluate_models(self):
        # Here, we read from the interaction_log and perform your evaluation
        # For demonstration purposes, let's just count interactions.
        try:
            interactions = pd.read_csv('interaction_log.csv', names=['userId', 'movieId', 'action'])
            model_a_interactions = interactions[interactions['userId'].isin(self.user_list[self.user_list['model'] == self.model_a]['userId'])]
            model_b_interactions = interactions[interactions['userId'].isin(self.user_list[self.user_list['model'] == self.model_b]['userId'])]
            
            model_a_count = len(model_a_interactions)
            model_b_count = len(model_b_interactions)
            
            return {
                self.model_a: model_a_count,
                self.model_b: model_b_count
            }
        except FileNotFoundError:
            print("Interaction log not found!")
            return {}

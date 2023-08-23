import sys

sys.path.append('../ucu-recommender-system-2023/')

import pickle
import numpy as np
import pandas as pd


class BaselineModel:
    """
    A baseline recommendation model that recommends movies
    based on their popularity and average ratings
    """

    def __init__(self, data_uploader):
        self.df_movies = data_uploader.df_movies

        if data_uploader.train_set:
            self.train_set = self.trainset_to_dataframe(data_uploader.train_set)
        else:
            data_uploader.split_train_eval_sets()
            self.train_set = self.trainset_to_dataframe(data_uploader.train_set)

    def trainset_to_dataframe(self, trainset):
        user_ids = []
        item_ids = []
        ratings = []
        for uid, iid, rating in trainset.all_ratings():
            user_ids.append(trainset.to_raw_uid(uid))
            item_ids.append(trainset.to_raw_iid(iid))
            ratings.append(rating)

        df = pd.DataFrame({'userId': user_ids, 'movieId': item_ids, 'rating': ratings})
        return df

    def fit(self, save=True, file_name=None):
        self.train_set['count'] = 1
        df_ratings_aggr = self.train_set.groupby('movieId')[['rating', 'count']].sum()
        df_ratings_aggr['rating_avg'] = round(df_ratings_aggr['rating'] / df_ratings_aggr['count'], 1)
        df_ratings_aggr['rank'] = np.log(df_ratings_aggr['rating']) * df_ratings_aggr['rating_avg']

        self.df_ranked_films = pd.merge(self.train_set, df_ratings_aggr[['rank', 'rating_avg']], on='movieId',
                                        how='outer') \
            .sort_values(by=['rank'], ascending=False) \
            .fillna(0)

        if save:
            with open(file_name, 'wb') as handle:
                pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, file_name):
        with open(file_name, 'rb') as handle:
            tmp_dict = pickle.load(handle)
            self.__dict__.update(tmp_dict)

    def predict(self, user_id):
        filtered_films = self.df_ranked_films.merge(self.train_set[self.train_set['userId'] == user_id]['movieId'],
                                                    on='movieId',
                                                    how='left',
                                                    indicator=True)

        filtered_films = filtered_films[filtered_films['_merge'] == 'left_only']
        # drop duplicates and get top 10 recommendations
        top_10_recs = filtered_films.drop_duplicates(subset=['movieId']).head(10)
        top_10_recs = top_10_recs.merge(self.df_movies, on='movieId')
        return top_10_recs[['movieId', 'title', 'rank']]

    def evaluate_ndcg(self, k=10):
        users = self.train_set['userId'].unique()
        average_ndcg = []

        for user in users:
            actual_movies = self.train_set[self.train_set['userId'] == user]['movieId'].tolist()
            recommended_movies = self.predict(user)['movieId'].tolist()

            # Binarize actual and predicted movies
            lb = LabelBinarizer()
            lb.fit(list(self.df_movies['movieId']))
            actual_binary = lb.transform(actual_movies)[:k]
            recommended_binary = lb.transform(recommended_movies)

            # Compute NDCG for this user
            user_ndcg = ndcg_score(actual_binary, recommended_binary, k=k)
            average_ndcg.append(user_ndcg)

        return np.mean(average_ndcg)

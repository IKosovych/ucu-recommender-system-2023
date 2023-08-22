import sys
sys.path.append('../ucu-recommender-system-2023/')

import pandas as pd
import numpy as np
from surprise import dump


class BaseClass:
    def __init__(self, data_uploader):
        self.df = data_uploader.df
        self.train_set, self.eval_set = data_uploader.train_set, data_uploader.eval_set
        self.df_movies = data_uploader.df_movies

    def fit(self):
        self.model = None

    def load_model(self, file_name):
        _, self.model = dump.load(file_name)

    def save_model(self, file_name):
        if self.model:
            dump.dump(file_name, algo=self.model)
        else:
            print('Fit the model before saving!')

    def predict(self):
        return self.model.test(self.eval_set)

    def get_recommendations(self, user_id: int, k: int):
        pred_rating_user = []
        all_items_ids = sorted(np.unique(self.df.itemID))
        for item_id in all_items_ids:
            pred_rating_user.append(self.model.predict(uid=user_id, iid=item_id)[3])

        df_pred = pd.DataFrame(index=all_items_ids, columns=['true', 'pred'])
        df_init = self.df.pivot(index='itemID', columns='userID', values='rating')
        df_init.fillna(0, inplace=True)
        df_pred['true'] = df_init.loc[:, user_id]
        df_pred['pred'] = pred_rating_user
        df_pred = df_pred.sort_values(by=['pred'], ascending=False)
        df_pred = df_pred[df_pred['true'] == 0]

        top_recommendations = self.df_movies[
            self.df_movies['movieId'].isin(df_pred.head(k).index)]['title']

        return top_recommendations

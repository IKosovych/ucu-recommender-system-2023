import sys
sys.path.append('../ucu-recommender-system-2023/')

import pandas as pd
import numpy as np

from data_uploader.uploader import DataUploader


class BaseClass:
    def __init__(self):
        self.data_uploader = DataUploader()
        self.df, self.train_set, self.test_set = self.data_uploader.get_user_item_data_surprise()

    def fit(self):
        self.model = None

    def predict_on_testset(self):
        return self.model.test(self.test_set)

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

        top_recommendations = self.data_uploader.df_movies[
            self.data_uploader.df_movies['movieId'].isin(df_pred.head(k).index)]['title']

        return top_recommendations

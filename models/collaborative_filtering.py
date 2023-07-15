import sys
sys.path.append('../ucu-recommender-system-2023/')

from surprise.prediction_algorithms.knns import KNNBasic, KNNWithMeans, KNNWithZScore

from data_uploader.uploader import DataUploader


class CollaborativeFiltering:
    """
    A user-user & item-item Collaborative filtering algorithm

    With 3 KNN modifications:
    basic - a basic collaborative filtering algorithm;
    means - -||- taking into account the mean ratings of each user;
    z-score - taking into account the z-score normalization of each user.

    And 3 similarity measures:
    cosine, msd (Mean Squared Difference) and pearson
    """

    def __init__(self):
        data_uploader = DataUploader()
        _, self.train_set, self.test_set = data_uploader.get_user_item_data_surprise()

    def fit(self, n_neighbours: int, knn_modification="basic", user_based=True, sim_measure='cosine'):
        sim_options = {
            "name": sim_measure,
            "user_based": user_based,  # compute similarities between users or items
        }

        if knn_modification == 'basic':
            knn = KNNBasic(k=n_neighbours, sim_options=sim_options)
        elif knn_modification == 'means':
            knn = KNNWithMeans(k=n_neighbours, sim_options=sim_options)
        elif knn_modification == 'z-score':
            knn = KNNWithZScore(k=n_neighbours, sim_options=sim_options)

        knn.fit(self.train_set)
        return knn

    def predict_on_testset(self, model):
        return model.test(self.test_set)

import pandas as pd
from surprise import accuracy, dump
from surprise.model_selection import cross_validate

from models.collaborative_filtering import CollaborativeFiltering
from models.matrix_factorization import MatrixFactorization
from data_uploader.uploader import DataUploader
from models.content_based_filtering import ContentBasedFiltering
from models.baseline_model import BaselineModel

### FIT & EVALUATE MODELS
data_uploader = DataUploader()
data_uploader.split_train_eval_sets()

bm = BaselineModel(data_uploader)
bm.fit(save=True, file_name='dumped_models/baseline.pickle')
bm_ndcg_score = bm.evaluate_ndcg()
print(f"Average NDCG for Baseline Model: {bm_ndcg_score:.4f}")

cbf = ContentBasedFiltering(data_uploader)
cbf.fit(save=True, file_name='dumped_models/content_based.pickle')
average_ndcg_from_cdf = cbf.evaluate_ndcg()
print(f'Average NDCG for ContentBasedFiltering model: {average_ndcg_from_cdf}')

# Collaborative Filtering
df = pd.DataFrame(columns=['KNN_modification', 'type', 'similarity_measure',
                           'eval-RMSE', 'eval-MSE', 'eval-MAE'])

l = ['basic'] * 3 + ['means'] * 3 + ['z-score'] * 3
df['KNN_modification'] = l * 2
df['type'] = ['user_based'] * 9 + ['item_based'] * 9
df['similarity_measure'] = ['cosine', 'msd', 'pearson'] * 6

rmses = []
mses = []
maes = []

cf = CollaborativeFiltering(data_uploader)
for user_based in [True, False]:
    for knn_modification in ['basic', 'means', 'z-score']:
        for sim_measure in ['cosine', 'msd', 'pearson']:
            cf.fit(knn_modification=knn_modification,
                   user_based=user_based,
                   sim_measure=sim_measure,
                   save=True,
                   file_name=f'dumped_models/cf_{knn_modification}_{str(user_based)}_{sim_measure}.pickle')

            predictions = cf.predict()
            rmses.append(accuracy.rmse(predictions))
            mses.append(accuracy.mse(predictions))
            maes.append(accuracy.mae(predictions))

df.to_csv('cf_eval_metrics.csv', index=False)

# Matrix Factorization
df = pd.DataFrame(columns=['factorization',
                           'eval-RMSE', 'eval-MSE', 'eval-MAE'])
df['factorization'] = ['SVD', 'PMF', 'NMF']

rmses = []
mses = []
maes = []

mf = MatrixFactorization(data_uploader)

for factorization in ['svd', 'pmf', 'nmf']:
    mf.fit(factorization=factorization,
           save=True,
           file_name=f'dumped_models/mf_{factorization}.pickle')

    predictions = mf.predict()
    rmses.append(accuracy.rmse(predictions))
    mses.append(accuracy.mse(predictions))
    maes.append(accuracy.mae(predictions))

df.to_csv('mf_eval_metrics.csv', index=False)

### CROSS-VALIDATION
print(cross_validate(mf.model, data_uploader.dataset,
                     measures=["RMSE", "MAE"],
                     cv=5,
                     verbose=True))

### INFERENCE

data_uploader = DataUploader()

# Baseline or Content-based Filtering
model_name = 'dumped_models/baseline.pickle'
b = BaselineModel(data_uploader)
b.load_model(model_name)

print(b.predict(user_id=1))

# Collaborative Filtering or Matrix factorization
model_name = 'dumped_models/mf_svd.pickle'
mf = MatrixFactorization(data_uploader)
mf.load_model(model_name)

print(mf.get_recommendations(user_id=4728, k=10))

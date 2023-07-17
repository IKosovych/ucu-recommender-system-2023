from models.collaborative_filtering import CollaborativeFiltering
from models.matrix_factorization import MatrixFactorization
from data_uploader.uploader import DataUploader

from surprise import accuracy
from surprise.model_selection import cross_validate

cf = CollaborativeFiltering()
cf.fit()
# Get top-10 recommendations
print(cf.get_recommendations(user_id=1, k=10))

mf = MatrixFactorization()
mf.fit()
# Get top-10 recommendations
print(mf.get_recommendations(user_id=1, k=10))

# Metrics (there are all metrics from this lib)
predictions = cf.predict_on_testset()
print(accuracy.rmse(predictions))
print(accuracy.mse(predictions))
print(accuracy.mae(predictions))
print(accuracy.fcp(predictions))

# Cross-validatiion
du = DataUploader()
print(cross_validate(mf.model, du.data, measures=["RMSE", "MAE"], cv=5, verbose=True))

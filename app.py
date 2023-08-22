import pandas as pd
import streamlit as st

from surprise import accuracy
from surprise.model_selection import cross_validate

from models.collaborative_filtering import CollaborativeFiltering
from models.matrix_factorization import MatrixFactorization
from models.content_based_filtering import ContentBasedFiltering
from models.baseline_model import BaselineModel
from data_uploader.uploader import DataUploader
from ab_distributor import ABtestDistributor


st.title('Evaluation')
st.header("Collaborative Filtering")
cf = CollaborativeFiltering()
cf.fit()
# Get top-10 recommendations
st.subheader("Get top-10 recommendations")
st.dataframe(cf.get_recommendations(user_id=1, k=10))

st.header("Matrix Factorization")
mf = MatrixFactorization()
mf.fit()

# Get top-10 recommendations
st.subheader("Get top-10 recommendations")
st.dataframe(mf.get_recommendations(user_id=1, k=10))

# Metrics (there are all metrics from this lib)
st.subheader("Metrics (there are all metrics from this lib)")
predictions = cf.predict_on_testset()
st.write(f"RMSE - {accuracy.rmse(predictions)}")
st.write(f"MSE - {accuracy.mse(predictions)}")
st.write(f"MAE - {accuracy.mae(predictions)}")
st.write(f"FCP - {accuracy.fcp(predictions)}")

# Cross-validatiion
du = DataUploader()
st.subheader("Cross-validatiion")
st.write(cross_validate(mf.model, du.data, measures=["RMSE", "MAE"], cv=5, verbose=True))

st.header("Content-Based Filtering")

cbf = ContentBasedFiltering()
cbf.fit()

number = st.number_input('Insert a user number: ', min_value=1, max_value=100, value=5, step=1)

recommendations = cbf.predict_on_testset(number)
st.subheader("Recommendations for User {}:".format(number))
st.dataframe(recommendations)

st.header("Baseline Model")
bm = BaselineModel()
bm.fit()

recommendations = bm.predict_on_testset(number)
st.subheader("Recommendations for User {}:".format(number))
st.dataframe(recommendations)

st.header("AB Test User Stliting")

# Get the distributor's model choice for a specific user
distributor = ABtestDistributor(data_uploader=du)
distributor.split_users(model_a='ContentBasedFiltering', model_b='BaselineModel')

number_distributor = st.number_input('Insert a user ID for AB test split:', min_value=1, max_value=10000, value=4, step=1)

model_name = distributor.get_model_name(number_distributor)
if model_name == 'ContentBasedFiltering':
    cbf_distributor = ContentBasedFiltering()
    cbf_distributor.fit()
    recommendations_distributor = cbf_distributor.predict_on_testset(number_distributor)
elif model_name == 'BaselineModel':
    bm_distributor = BaselineModel()
    bm_distributor.fit()
    recommendations_distributor = bm_distributor.predict_on_testset(number_distributor)

st.subheader("Recommendations for User {} (Distributor's Choice - {}):".format(number_distributor, model_name))
st.dataframe(recommendations_distributor)

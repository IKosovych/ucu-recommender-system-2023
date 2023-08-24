import pandas as pd
import numpy as np
import streamlit as st

from surprise.model_selection import cross_validate

from models.collaborative_filtering import CollaborativeFiltering
from models.matrix_factorization import MatrixFactorization
from models.content_based_filtering import ContentBasedFiltering
from models.baseline_model import BaselineModel
from data_uploader.uploader import DataUploader
from ab_distributor import ABtestDistributor

data_uploader = DataUploader()

st.title('Evaluation')
st.header("Collaborative Filtering")
model_name = 'dumped_models/best/cf_basic_True_msd.pickle'
cf = CollaborativeFiltering(data_uploader)
cf.load_model(model_name)

# Get top-10 recommendations
number = st.number_input('Insert a user number: ', min_value=1, max_value=6000, value=5, step=1)

st.subheader(f"Top-10 recommendations for the User {number}")
st.dataframe(cf.get_recommendations(user_id=number, k=10))

st.header("Matrix Factorization")
model_name = 'dumped_models/best/mf_pmf.pickle'
mf = MatrixFactorization(data_uploader)
mf.load_model(model_name)

# Get top-10 recommendations
number = st.number_input('Insert a user number: ', min_value=1, max_value=6000, value=5, step=1)

st.subheader(f"Top-10 recommendations for the User {number}")
st.dataframe(mf.get_recommendations(user_id=number, k=10))

# Metrics
st.subheader("Best Metrics for Collaborative Filtering on the evaluation set")
metrics_df = pd.read_csv('cf_eval_metrics.csv')

st.write(f"RMSE - {np.round(metrics_df.loc[1]['eval-RMSE'], 2)}")
st.write(f"MSE - {np.round(metrics_df.loc[1]['eval-MSE'], 2)}")
st.write(f"MAE - {np.round(metrics_df.loc[1]['eval-MAE'], 2)}")

st.subheader("Best Metrics for Matrix Factorization on the evaluation set")
metrics_df = pd.read_csv('mf_eval_metrics.csv')

st.write(f"RMSE - {np.round(metrics_df.loc[1]['eval-RMSE'], 2)}")
st.write(f"MSE - {np.round(metrics_df.loc[1]['eval-MSE'], 2)}")
st.write(f"MAE - {np.round(metrics_df.loc[1]['eval-MAE'], 2)}")

# Cross-validatiion
st.subheader("Cross-validation for Matrix Factorization")
st.write(cross_validate(mf.model, data_uploader.dataset,
                        measures=["RMSE", "MAE"], cv=5, verbose=True))


st.header("Content-Based Filtering")
model_name = 'dumped_models/best/content_based.pickle'
cb = ContentBasedFiltering(data_uploader=data_uploader)
cb.load_model(model_name)

number = st.number_input('Insert a user number: ', min_value=1, max_value=100, value=5, step=1)

recommendations = cb.predict(user_id=number)
st.subheader("Recommendations for User {}:".format(number))
st.dataframe(recommendations)

st.header("Baseline Model")
model_name = 'dumped_models/best/baseline.pickle'
bm = BaselineModel(data_uploader)
bm.load_model(model_name)

recommendations = bm.predict(number)
st.subheader("Recommendations for User {}:".format(number))
st.dataframe(recommendations)

st.header("AB Test User Stliting")

# Get the distributor's model choice for a specific user
distributor = ABtestDistributor(data_uploader=data_uploader)
distributor.split_users(model_a='ContentBasedFiltering', model_b='BaselineModel')

number_distributor = st.number_input('Insert a user ID for AB test split:', min_value=1, max_value=10000, value=4, step=1)

model_name = distributor.get_model_name(number_distributor)
if model_name == 'ContentBasedFiltering':
    model_name = 'dumped_models/best/content_based.pickle'
    cbf_distributor = ContentBasedFiltering(data_uploader=data_uploader)
    cbf_distributor.load_model(model_name)
    recommendations_distributor = cbf_distributor.predict(number_distributor)
elif model_name == 'BaselineModel':
    model_name = 'dumped_models/best/baseline.pickle'
    bm_distributor = BaselineModel(data_uploader=data_uploader)
    bm_distributor.load_model(model_name)
    recommendations_distributor = bm_distributor.predict(number_distributor)

st.subheader("Recommendations for User {} (Distributor's Choice - {}):".format(number_distributor, model_name))
st.dataframe(recommendations_distributor)

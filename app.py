import pandas as pd
import streamlit as st

from surprise import accuracy
from surprise.model_selection import cross_validate

from models.collaborative_filtering import CollaborativeFiltering
from models.matrix_factorization import MatrixFactorization
from models.content_based_filtering import ContentBasedFiltering
from models.baseline_model import BaselineModel
from data_uploader.uploader import DataUploader


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
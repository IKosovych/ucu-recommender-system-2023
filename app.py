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
mf = MatrixFactorization()
mf.fit()

# Get top-10 recommendations
st.subheader("Get top-10 recommendations")
st.dataframe(mf.get_recommendations(user_id=1, k=10))

# Metrics (there are all metrics from this lib)
st.subheader("Metrics (there are all metrics from this lib)")
predictions = cf.predict_on_testset()
st.write(accuracy.rmse(predictions))
st.write(accuracy.mse(predictions))
st.write(accuracy.mae(predictions))
st.write(accuracy.fcp(predictions))

# Cross-validatiion
du = DataUploader()
st.subheader("Cross-validatiion")
st.write(cross_validate(mf.model, du.data, measures=["RMSE", "MAE"], cv=5, verbose=True))

st.header("Content-Based Filtering")

cbf = ContentBasedFiltering()
#cbf.fit()

#recommendations = cbf.predict_on_testset(2)

#st.write("Recommendations for User {}:".format(2))
#for idx, title in enumerate(recommendations):
#    print("{}: {}".format(idx+1, title))

#data = [{'Index': idx+1, 'Title': title} for idx, title in enumerate(recommendations)]

# Create a DataFrame from the list of dictionaries
#df = pd.DataFrame(data)
#st.dataframe(df)
#average_ndcg = cbf.evaluate()
#print(f'Average NDCG: {average_ndcg}')
bm = BaselineModel()
bm.fit()
print(bm.predict_on_testset(1))
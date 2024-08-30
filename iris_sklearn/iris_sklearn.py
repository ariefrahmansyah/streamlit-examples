import pickle

import streamlit as st

model = pickle.load(open("iris_sklearn/model.pkl", "rb"))
target_names = ["setosa", "versicolor", "virginica"]

st.title("Iris Classifier 🌺")

sepal_length = st.number_input("Sepal length", key="sepal_length")
sepal_width = st.number_input("Sepal width", key="sepal_width")
petal_length = st.number_input("Petal length", key="petal_length")
petal_width = st.number_input("Petal width", key="petal_width")

if st.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.write(f"Prediction: {prediction[0]}")
    st.write(f"Predicted target name: {target_names[prediction[0]]}")

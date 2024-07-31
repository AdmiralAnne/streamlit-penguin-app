import streamlit as st
import pandas as pd
# import numpy as np


# giving a title to the application
st.title('🐧 Penguin Species Prediction App 🐧')

st.info('This is a simple Machine Learning app that predicts species of a penguin based on key features.')
# use st.write if you want to write normal texts instead of having the blue background

# create a dataframe that stores the penguin data csv file -> already cleaned
df = pd.read_csv("penguins_cleaned.csv")
df
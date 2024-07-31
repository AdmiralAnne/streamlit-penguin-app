import streamlit as st
import pandas as pd
# import numpy as np


# giving a title to the application
st.title('🐧 Penguin Species Prediction App 🐧')

st.info('This is a simple Machine Learning app that predicts species of a penguin based on key features.')
# use st.write if you want to write normal texts instead of having the blue background

# add an expander -> Like a toggle button that expands upon clicking
with st.expander('**Data**'):
    # add a title for this dataframe that's being displayed
    st.write('Raw Data csv')
    # create a dataframe that stores the penguin data csv file -> already cleaned
    df = pd.read_csv("penguins_cleaned.csv")
    df

    # divide the data into x and y -> features vs Label
    # lets make x first -> which is the dataset without the species feature
    st.write(**feature_set**)
    # create a variable x that stores the featurset
    X = df.drop['species',axis=1]
    X
    # y will be just the species feature column i.e our label

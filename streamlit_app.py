import streamlit as st
import pandas as pd
# import numpy as np


# giving a title to the application
st.title('🐧Penguin Species Prediction App')

st.info('This is a simple Machine Learning app that predicts species of a penguin based on key features.')
# use st.write if you want to write normal texts instead of having the blue background

# add an expander -> Like a toggle button that expands upon clicking
with st.expander('**Data**'):
    # add a title for this dataframe that's being displayed
    st.write('**Raw Data csv**')
    # create a dataframe that stores the penguin data csv file -> already cleaned
    df = pd.read_csv("penguins_cleaned.csv")
    df
    st.divider()
    
    # divide the data into x and y -> features vs Label
    # lets make x first -> which is the dataset without the species feature
    st.write('**feature_set**')
    # create a variable x that stores the featurset
    X = df.drop('species', axis=1)
    X
    st.divider()
    
    # y will be just the species feature column i.e our label
    st.write('**Labels (what we wanna predict)**')
    # create a variable y that stores the featurset
    Y = df['species'] # alternately you can use df.species
    # always remember to print y
    Y

with st.expander('**Data Visualization**'):
    st.write('body mass vs bill length')
    # "bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"
    st.scatter_chart(data=df, x="bill_length_mm", y="body_mass_g", x_label="bill_length", y_label="body_mass", color='species', size=None, width=None, height=None, use_container_width=True)

# Input Features
# st.sidebar is as the name suggests -> used to create a sidebar that can store our widgets
# species	island	bill_length_mm	bill_depth_mm	flipper_length_mm	body_mass_g	sex
with st.sidebar:
    st.header('Input Features')
    # create a var that stores the select box called island that will help us select a specific Island
    island = st.selectbox('island', ('Torgersen','Biscoe','Dream'))
    # another one for gender / sex
    sex = st.selectbox('sex',('male','female'))
    # create a slider for the features
    bill_length_mm = st.slider("Select bill_Length", min_value=30.0, max_value=60.0, value=45.0, step=1.0, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    bill_depth_mm = st.slider("Select bill_depth", min_value=13.0, max_value=25.0, value=18.0, step=1.0, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    flipper_length_mm = st.slider("Select flipper_length", min_value=172.0, max_value=231.0, value=200.0, step=1.0, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    body_mass_g = st.slider("Select body_mass", min_value=2700.0, max_value=6200.0, value=4200.0, step=1.0, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

# encoding data
# create df for input features -> whatever we have selected on the sliders and boxes
# name the new df as "inpput_data" -> First Create a dictionary called data with all the key:value pairs
# follow the format column_name:value and so on
data = {'island':island,
        'bill_length_mm':bill_length_mm,
        'bill_depth_mm':bill_depth_mm,
        'flipper_length_mm':flipper_length_mm,
        'body_mass_g':body_mass_g,
        'sex':sex,
        }

input_df = pd.DataFrame(data, index=[0])
# create a new data frame where we concatinate the old df (X-> features) and the new one(selected inputs)
full_df = pd.concat([input_df,X], axis=0)
# add these two to expander later

# Encode X
# encoding since some data is String -> convert categorical to numberic using - dummies
encode = ['island', 'sex']
# add some encoded data as new columns
full_df = pd.get_dummies(full_df, prefix=encode)
# but keep in mind, we only want the input row -> so, we'll select and print only the first row
input_row = full_df[:1] # select all columns, but select only 1st row

with st.expander('**Input Data**'):
    st.write('**Selected values from input features**')
    input_df
    st.divider()
    st.write('**Features and Selected values**')
    full_df
    st.divider()
    st.write('**Encoded input values df**')
    input_row

# Encode Y
target_mapper = {
    'Adelie':0,
    'Gentoo':1,
    'Chinstrap':2,
}
# convet the Species to numeric format now
# since our target is the "species" feature -> we will call this encoder fn "target_encoder"
def target_encoder(val):
    # here we will use the target mapper
    return target_mapper[val]

# lets do the conversion here on a new dataframe called.. Y_encoded
Y_encoded = Y.apply(target_encoder)

# another expander called data prep with the encoded values
with st.expander('**Data Preparation**'):
    st.write('**Encoded input values df**')
    input_row
    st.write('**Encoded Y / Target values**')
    Y_encoded

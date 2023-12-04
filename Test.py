#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='NPR Speaker Identification using Text Data',
    layout='wide')

#---------------------------------#
# Model building
def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    # Data splitting
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)

    rf = RandomForestRegressor(n_estimators=parameter_n_estimators)
    rf.fit(X_train, Y_train)

    st.subheader('2. Model Performance')

#---------------------------------#
st.write("""
# Speaker Prediction
In this implementation, the *RandomForestRegressor()* function is used in this app for build a regression model using the **Random Forest** algorithm.
Try adjusting the hyperparameters!
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('2.1. Learning Parameters'):
    parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
    

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    
    build_model(df)


    build_model(df)


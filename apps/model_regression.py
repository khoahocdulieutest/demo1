#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
#get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline


from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

DATA_FILE = None

def app(state):
    df = st.session_state['df_OnehotEncoder']
    data = st.session_state['data']

    # In[31]:


    df_ohe = pd.get_dummies(data=df, columns=["region"])
    df_ohe.head()


    # In[32]:


    st.write(df_ohe.columns)


    # In[33]:


    # Choose TotalVolumebc it has high corr with '4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags'
    X = df_ohe.drop(['Date', "AveragePrice",'type', '4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags'], axis=1)
    y = df['AveragePrice']


    # In[34]:


    X.head()


    # Buoc 4&5: Modeling & Evaluation/Analyze & Report
    st.write("## Buoc 4&5: Modeling & Evaluation/Analyze & Report")
    # In[35]:


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


    # In[36]:


    # Have many range values => Scaler (and large samples ~ 18k) => StandardScaler
    st.write('### Have many range values => Scaler (and large samples ~ 18k) => StandardScaler')

    # In[37]:

    st.write("#### LinearRegression")
    pipe_LR = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
    pipe_LR.fit(X_train, y_train)
    y_pred_LR = pipe_LR.predict(X_test)
    st.write(f'r2_score = {r2_score(y_test, y_pred_LR)}')


    # In[38]:


    mae_LR = mean_absolute_error(y_test, y_pred_LR)
    st.write(f"mean_absolute_error= {mae_LR}")


    # In[39]:

    st.write("#### RandomForestRegressor")
    pipe_RF = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor())])
    pipe_RF.fit(X_train, y_train)
    y_pred_RF = pipe_LR.predict(X_test)
    r2_score_RF = r2_score(y_test, y_pred_LR)
    st.write(f'r2_score = {r2_score_RF}')

    # In[40]:


    mae_RF = mean_absolute_error(y_test, y_pred_RF)
    mae_RF
    st.write(f"mean_absolute_error= {mae_RF}")

    # In[41]:

    st.write("#### XGBRegressor")
    pipe_XGB = Pipeline([('scaler', StandardScaler()), ('xgb', XGBRegressor())])
    pipe_XGB.fit(X_train, y_train)
    y_pred_XGB = pipe_XGB.predict(X_test)
    r2_score_XGB = r2_score(y_test, y_pred_XGB)
    st.write(f'r2_score = {r2_score_XGB}')

    # In[42]:


    mae_XGB = mean_absolute_error(y_test, y_pred_XGB)
    mae_XGB
    st.write(f"mean_absolute_error= {mae_XGB}")

    # Select RandomForestRegressor bc it has highest r^2 and lowest MAE
    st.markdown("##### Select RandomForestRegressor bc it has highest $r^2$ and lowest $MAE$")
    # In[43]:


    df43 = pd.DataFrame(pipe_RF['rf'].feature_importances_,
                index=X_train.columns,
                columns=['feature_importances']).sort_values(by=['feature_importances'],
                                                           ascending=False)

    st.dataframe(df43)
    # In[44]:


    pre_df = pd.DataFrame(pipe_RF['rf'].feature_importances_,              index=X_train.columns,              columns=['feature_importances']).sort_values(by=['feature_importances'],                                                            ascending=False)

    st.dataframe(pre_df)
    # Request 2: Organic Avocado Average Price Prediction for the future in California ARIMA & PROPHET - Time Series Algorrithm (0.5)

    # In[45]:

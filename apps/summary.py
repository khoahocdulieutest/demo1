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
    # In[2]:
    #st.set_page_config(page_title="Topic 1", page_icon="📈")

    st.markdown("# Summary about project")
    st.sidebar.header("Summary about project")
    if 'data' in st.session_state:
        #st.write(st.session_state['data'])
        print('session_state', st.session_state['data'])
        #data = pd.read_csv("avocado.csv")
        #data = pd.read_csv(st.session_state['data'])
        data = st.session_state['data']
    else:
        return




    # In[3]:


    data.info()


    # In[4]:


    data.head()


    # In[5]:


    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]


    # In[6]:


    data.head(2)


    # Request 1: Organic Avocados' Price Prediction
    # Lineear Regression, Random Forest, XGB Regressor...- Regression Algorithm

    # Buoc 3: Data preparation

    # In[7]:


    df = data.copy(deep=True)


    # First EDA =>Check data

    # In[8]:


    #pp.ProfileReport(df)


    # From Pandas Profilling Report we see that:
    #     + No missing cells
    #     + No duplicate rows

    # In[9]:


    # we can check again
    df.isnull().any()


    # In[10]:


    df.isna().any()


    # In[11]:


    df.duplicated().any()


    # In[12]:


    df.shape


    # Wheter the Categorical Features('type'/'region') affected the 'AveragePrice'?

    # Whheter the Categorical Features('type') affected the 'AveragePrice'?

    # In[13]:


    #AveragePrice Distribution/ Boxplot of 2 types of Avocados
    fig, ax = plt.subplots(figsize=(10,5))
    sns.displot(df, x="AveragePrice",hue="type", stat="density")
    #plt.show()
    st.pyplot(fig)

    # In[14]:

    fig, ax = plt.subplots(figsize=(10,5))
    sns.displot(df, x="AveragePrice",hue="type", stat="probability")
    #plt.show()
    st.pyplot(fig)


    # In[15]:

    fig, ax = plt.subplots(figsize=(10,5))
    sns.displot(df, x="AveragePrice",hue="type", multiple="dodge")
    #plt.show()
    st.pyplot(fig)

    # In[16]:

    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df, x="type", y="AveragePrice")
    #plt.show()
    st.pyplot(fig)


    # Organic Avocado is more expansive than conventional avocado.
    # "AveragePrice" was affected by "type"

    # Whether the "region" affected 'AveragePrice' ?

    # In[17]:


    # type == "organic"
    fig, ax = plt.subplots(figsize=(20,8))
    sns.boxplot(data = df[df['type']=='organic'],
               x="region", y="AveragePrice", ax=ax)
    plt.xticks(rotation=90)
    #plt.show()
    st.pyplot(fig)


    # Some region have high price
    # Some region have low price

    # In[18]:


    # type == 'conventional'
    fig, ax = plt.subplots(figsize=(20,8))
    sns.boxplot(data = df[df['type']=='conventional'],
               x="region", y="AveragePrice", ax=ax)
    plt.xticks(rotation=90)
    #plt.show()
    st.pyplot(fig)


    # Some region have high price
    # Some region have low price

    # Whether the Continuous Features affected the "AveragePrice"

    # In[19]:


    #Correlation
    corr = df.corr()
    corr


    # In[20]:


    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
    #plt.show()
    st.pyplot(fig)


    # Corr between AveragePrice with other factor is very low
    # other factor don't affect to AveragePrice

    # Feature Engineering
    # + About Date, we can now the season in USA from https://seasonsyear.com/USA
    #     + US'Spring  months are March, April and May (3,4,5)
    #     + US'summer  months are june, July and August (6,7,8)
    #     + US'autumn  months are Sep, Oct and Nov (9,10,11)
    #     + US'winter  months are Dec, jan and Feb (12,1,2)

    # In[21]:


    def convert_moth(month):
        if month == 3 or month == 4 or month == 5:
            return 0
        elif month == 6 or month == 7 or month == 8:
            return 1
        if month == 9 or month == 10 or month == 11:
            return 2
        else:
            return 3


    # In[22]:


    df['Date'] = pd.to_datetime(df["Date"])


    # In[23]:


    df['Month'] = pd.DatetimeIndex(df["Date"]).month


    # In[24]:


    df['Season'] = df["Month"].apply(lambda x: convert_moth(x))


    # In[25]:


    df.info()


    # In[26]:


    df.head()


    # Whether "Season" affect AveragePrice ?
    st.markdown('### Whether "Season" affect AveragePrice ?')
    # In[27]:


    #type = conventional
    fig, ax = plt.subplots(figsize=(20,6))
    sns.boxplot(data = df[df['type']=='conventional'],
               x="Season", y="AveragePrice", ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)


    # In[28]:


    #type = Organic
    st.markdown('####Organic')
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(data = df[df['type']=='organic'],
               x="Season", y="AveragePrice", ax=ax)
    plt.xticks(rotation=90)
    #plt.show()
    st.pyplot(fig)


    # Yes, AveragePrice was affected by "Season" (both in 'Organic' type and "conventional" type)
    st.markdown('#### Yes, AveragePrice was affected by "Season" (both in $Organic$ type and $conventional$ type)')
    # In[29]:


    # Label Encoder and OnehotEncoder for 'type' and 'region'
    st.markdown("#### Label Encoder and OnehotEncoder for $type$ and $region$")
    le = LabelEncoder()
    df['type_new'] = le.fit_transform(df['type'])


    # In[30]:


    st.write(df.head())

    st.session_state['df_OnehotEncoder'] = df
    # In[31]:





import streamlit as st
import time
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

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)
#
# progress_bar = st.sidebar.progress(0)
# status_text = st.sidebar.empty()
# last_rows = np.random.randn(1, 1)
# chart = st.line_chart(last_rows)
#
# for i in range(1, 101):
#     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#     status_text.text("%i%% Complete" % i)
#     chart.add_rows(new_rows)
#     progress_bar.progress(i)
#     last_rows = new_rows
#     time.sleep(0.05)
#
# progress_bar.empty()
#
# # Streamlit widgets automatically run the script from top to bottom. Since
# # this button is not connected to any other logic, it just causes a plain
# # rerun.
# st.button("Re-run")

data = pd.read_csv(r"D:\DAC\HAnh\demo1\avocado.csv")
st.write(data.head())
#st.write(data.info())
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
df = data.copy(deep=True)
#report = pp.ProfileReport(df)
st.write(df.head())

#13
fig, ax = plt.subplots(figsize=(10,5))
sns.displot(df, x="AveragePrice", hue="type", stat="density")
st.pyplot(fig)


#16
fig, ax = plt.subplots(figsize=(20,20))
sns.boxplot(data=df, x="type", y="AveragePrice")
st.pyplot(fig)

corr = df.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corr, vmin=-1, vmax=1, annot=True)

st.pyplot(fig)

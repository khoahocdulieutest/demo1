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


# In[2]:
st.set_page_config(page_title="Topic 1", page_icon="📈")

st.markdown("# Topic 1")
st.sidebar.header("Topic 1")


data =pd.read_csv("avocado.csv")


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


pp.ProfileReport(df)


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

# In[27]:


#type = conventional
fig, ax = plt.subplots(figsize=(20,6))
sns.boxplot(data = df[df['type']=='conventional'],
           x="Season", y="AveragePrice", ax=ax)
plt.xticks(rotation=90)
plt.show()


# In[28]:


#type = Organic
fig, ax = plt.subplots(figsize=(10,6))
sns.boxplot(data = df[df['type']=='organic'],
           x="Season", y="AveragePrice", ax=ax)
plt.xticks(rotation=90)
plt.show()


# Yes, AveragePrice was affected by "Season" (both in 'Organic' type and "conventional" type)

# In[29]:


# Label Encoder and OnehotEncoder for 'type' and 'region'
le = LabelEncoder()
df['type_new'] = le.fit_transform(df['type'])


# In[30]:


df.head()


# In[31]:


df_ohe = pd.get_dummies(data=df, columns=["region"])
df_ohe.head()


# In[32]:


df_ohe.columns


# In[33]:


# Choose TotalVolumebc it has high corr with '4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags'
X = df_ohe.drop(['Date', "AveragePrice",'type', '4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags'], axis=1)
y = df['AveragePrice']


# In[34]:


X.head()


# Buoc 4&5: Modeling & Evaluation/Analyze & Report

# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[36]:


# Have many range values => Scaler (and large samples ~ 18k) => StandardScaler


# In[37]:


pipe_LR = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
pipe_LR.fit(X_train, y_train)
y_pred_LR = pipe_LR.predict(X_test)
r2_score(y_test, y_pred_LR)


# In[38]:


mae_LR = mean_absolute_error(y_test, y_pred_LR)
mae_LR


# In[39]:


pipe_RF = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestRegressor())])
pipe_RF.fit(X_train, y_train)
y_pred_RF = pipe_LR.predict(X_test)
r2_score(y_test, y_pred_LR)


# In[40]:


mae_RF = mean_absolute_error(y_test, y_pred_RF)
mae_RF


# In[41]:


pipe_XGB = Pipeline([('scaler', StandardScaler()), ('xgb', XGBRegressor())])
pipe_XGB.fit(X_train, y_train)
y_pred_XGB = pipe_XGB.predict(X_test)
r2_score(y_test, y_pred_XGB)


# In[42]:


mae_XGB = mean_absolute_error(y_test, y_pred_XGB)
mae_XGB


# Select RandomForestRegressor bc it has highest r^2 and lowest MAE

# In[43]:


pd.DataFrame(pipe_RF['rf'].feature_importances_,
            index=X_train.columns,
            columns=['feature_importances']).sort_values(by=['feature_importances'],
                                                       ascending=False)


# In[44]:


pd.DataFrame(pipe_RF['rf'].feature_importances_,              index=X_train.columns,              columns=['feature_importances']).sort_values(by=['feature_importances'],                                                            ascending=False)


# Request 2: Organic Avocado Average Price Prediction for the future in California ARIMA & PROPHET - Time Series Algorrithm (0.5) 

# In[45]:


# Make new dataframe from orginal dataframe: data
df_ca = data[data['region'] == 'California']
df_ca['Date'] = df_ca['Date'].str[:-3]
df_ca = df_ca[data['type'] == 'organic']


# In[46]:


df_ca.shape


# In[47]:


agg = {'AveragePrice': 'mean'}
df_ca_gr = df_ca.groupby(df_ca['Date']).aggregate(agg).reset_index()
df_ca_gr.head()


# In[48]:


df_ts = pd.DataFrame()
df_ts['ds'] = pd.to_datetime(df_ca_gr['Date'])
df_ts['y'] = df_ca_gr['AveragePrice']
df_ts.head()


# In[49]:


df_ts.tail()


# In[50]:


df_ts.shape


# In[51]:


# Mean of Organic Avocado AveragePrice in California
df_ts['y'].mean()


# Use df_ts1 for ARIMA, df_ts for Prophet

# In[52]:


df_ts1 = df_ts.copy(deep=False)


# In[53]:


df_ts1.index = pd.to_datetime(df_ts1.ds)


# In[54]:


df_ts1.index


# In[55]:


df_ts1.head()


# In[56]:


df_ts1 = df_ts1.drop(['ds'], axis=1)


# In[57]:


plt.figure(figsize=(15,8))
plt.plot(df_ts1)
plt.title("Avocados' AveragePrice in California")
plt.show()


# In[58]:


decompose_result = seasonal_decompose(df_ts1, model='Mmultiplicative')
decompose_result


# In[59]:


plt.figure(figsize=(15,4))
decompose_result.plot()
plt.show()


# In[60]:


plt.figure(figsize=(15,4))
plt.plot(decompose_result.trend)
plt.show()


# In[61]:


plt.figure(figsize=(15,4))
plt.plot(decompose_result.seasonal)
plt.show()


# In[62]:


plt.figure(figsize=(15,4))
plt.plot(decompose_result.resid)
plt.show()


# With the above result, we can clearly see the seasonal component of the data, and also see that trenf is nonlinear. Residual ranges from 0.85 => 1.15

# Cuoc 4&5: Modeling & Evaluation/Analyze and Report

# Arima

# In[63]:


stepwise_model = auto_arima(df_ts1, start_p=2, start_q=2,
                           max_p=3, max_q=3, m=12,
                           start_P=1, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)


# In[64]:


print(stepwise_model.aic())


# In[65]:


df_ts1.shape


# In[66]:


train = df_ts1.loc['2015-01-01':'2017-06-01']
test = df_ts1.loc['2017-06-01':]


# In[67]:


len(test)


# In[68]:


#Fit model
stepwise_model.fit(train)


# In[69]:


future_forecast = stepwise_model.predict(n_predict=len(test))


# In[70]:


future_forecast


# In[71]:


rmse = sqrt(mean_squared_error(test,future_forecast))
rmse


# In[72]:


mae = mean_absolute_error(test, future_forecast)
mae


# In[73]:


future_forecast = pd.DataFrame(future_forecast, index = test.index, columns=['Prediction'])


# In[74]:


#Visualize the result
plt.figure(figsize=(12,6))
plt.plot(test, label='AveragePrice')
plt.plot(future_forecast, label='AveragePrice Prediction')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()


# In[75]:


plt.figure(figsize=(15,8))
plt.plot(df_ts1, label='AveragePrice All')
plt.plot(future_forecast, label='Prediction')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()


# In[76]:


plt.plot(test, label='AveragePrice')
plt.plot(future_forecast, label='Prediction')
plt.plot(future_forecast-0.5*rmse, label='Prediction')


# Because the increase and decrease is not uniform , it is difficult to find and adaptive value
# (Vi su tang giam khong dong deu nen kho tim duoc gia tri thich nghi)

# Prediction for the next 12 months

# In[77]:


future_forecast_12 = stepwise_model.predict(n_periods=len(test)+12)
future_forecast_12


# In[78]:


plt.figure(figsize=(8,5))
plt.plot(future_forecast_12[len(test):], color='red', label='Prediction')
plt.xticks(rotation='vertical')
plt.title('Prediction next 12 months')
plt.legend()
plt.show()


# In[79]:


future_forecast_12[len(test):]


# In[80]:


months = pd.date_range('2018-04-01', '2019-03-01', freq='MS').strftime("%Y-%m-%d").tolist()


# In[81]:


new_prdict = pd.DataFrame({
    'ds' : months,
    'y': future_forecast_12[len(test):]
})
new_prdict


# + Because of the small amount of data(just over 3 years), the prediction of organic avocado's Average Price in California isn't accurate,
# mae ~ 0.32 (about~20% of the mean AveragePrice),which is quite high compared to the AveragePrice of ~ 1.68)
# + Try another prediction algorithm: Prophet (of Facebook)

# In[82]:


# Prophet


# In[83]:


# create test dataset, remove last 10 months
train = df_ts.drop(df_ts.index[-10:])
train.tail()


# In[84]:


test = df_ts.drop(df_ts.index[0:-10])
test


# In[85]:


len(test)


# # Build model

# In[86]:


model = Prophet(yearly_seasonality=True,                daily_seasonality=False, weekly_seasonality=False)


# In[87]:


model.fit(train)


# In[88]:


# 10 month in test and 12 month to predict new values
months = pd.date_range('2017-06-01','2019-03-01',
                    freq='MS').strftime("%Y-%m-%d").tolist()
future = pd.DataFrame(months)
future.columns = ['ds']
future['ds'] = pd.to_datetime(future['ds'])


# In[89]:


#Use the model to make a forecast
forecast = model.predict(future)


# In[90]:


forecast[["ds",'yhat']].head(10)


# In[91]:


#calculate MAE/RWSE between expected and predicted values for december
y_test = test['y'].values
y_pred = forecast['yhat'].values[:10]
mae_p = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' %mae_p)


# + This result shows that prophet's rmse and mae are better than ARIMA Although the amount of data is small (just over 3 years), it is acceptable to predict the organic avocado AveragePrice in California, mae = 0.16(about 10% of AveragePrice), compared to the AveragePrice ~1.68.
# + We can see that Prophet (Facebooks) algorihm give better results

# # Choose Prophet for predicting avocado prices in California in near future

# In[92]:


y_test


# In[93]:


y_pred


# In[94]:


y_test_value = pd.DataFrame(y_test, index = pd.to_datetime(test['ds']), columns=['Actual'])
y_pred_value = pd.DataFrame(y_pred, index = pd.to_datetime(test['ds']), columns=['Prediction'])


# In[95]:


y_test_value


# In[96]:


y_pred_value


# In[97]:


#Visualize thre result
plt.figure(figsize=(12,6))
plt.plot(y_test_value, label='AveragePrice')
plt.plot(y_pred_value, label='AveragePrice')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()


# In[98]:


fig = model.plot(forecast)
fig.show()
a = add_changepoints_to_plot(fig.gca(), model, forecast)


# In[99]:


fig1 =model.plot_components(forecast)
fig1.show()


# Prediction for next 12 months

# In[100]:


forecast[['ds','yhat']].tail(12)


# Long-term prediction for the next 5 years ==> Consider whether to expand Cultivation/production, and trading

# In[101]:


m = Prophet(yearly_seasonality=True,                daily_seasonality=False, weekly_seasonality=False)
m.fit(df_ts)
future = m.make_future_dataframe(periods=12*5, freq="M") # next 5 years


# In[102]:


forecast = m.predict(future)


# In[103]:


forecast[['ds','yhat', 'yhat_lower', 'yhat_upper','trend', 'trend_lower', 'trend_upper' ]].tail(12)


# In[104]:


forecast.shape


# In[105]:


fig = m.plot(forecast)
fig.show()
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# In[106]:


fig1 =m.plot_components(forecast)
fig1.show()


# In[107]:


plt.figure(figsize=(12,6))
plt.plot(df_ts['y'], label='AveragePrice')
plt.plot(forecast['yhat'], label='AveragePrice with next 60 months prediction',
        color='red')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()


# Base on the above results , we can see that it is possile to expand the cultivation/production and trading of organic avocados in California

# Request 3: conventional Avocado Average Price Prediction for the future in California (0.3)

# In[108]:


# Make new dataframe from orginal dataframe: data
df_ca = data[data['region'] == 'California']
df_ca['Date'] = df_ca['Date'].str[:-3]
df_ca = df_ca[data['type'] == 'conventional']


# In[109]:


df_ca.shape


# In[110]:


agg = {'AveragePrice': 'mean'}
df_ca_gr = df_ca.groupby(df_ca['Date']).aggregate(agg).reset_index()
df_ca_gr.head()


# In[111]:


df_ts = pd.DataFrame()
df_ts['ds'] = pd.to_datetime(df_ca_gr['Date'])
df_ts['y'] = df_ca_gr['AveragePrice']
df_ts.head()


# In[112]:


df_ts.tail()


# In[113]:


df_ts.shape


# In[114]:


# Mean of conventinal Avocado AveragePrice in California
df_ts['y'].mean()


# In[115]:


df_ca.shape

agg = {'AveragePrice': 'mean'}
df_ca_gr = df_ca.groupby(df_ca['Date']).aggregate(agg).reset_index()
df_ca_gr.head()

df_ts = pd.DataFrame()
df_ts['ds'] = pd.to_datetime(df_ca_gr['Date'])
df_ts['y'] = df_ca_gr['AveragePrice']
df_ts.head()

df_ts.tail()

df_ts.shape

# Mean of Organic Avocado AveragePrice in California
df_ts['y'].mean()


# In[116]:


# Prophet


# In[117]:


# create test dataset, remove last 10 months
train = df_ts.drop(df_ts.index[-10:])
train.tail()


# In[118]:


test = df_ts.drop(df_ts.index[0:-10])
test


# In[119]:


len(test)


# # Build model

# In[120]:


model = Prophet(yearly_seasonality=True,                daily_seasonality=False, weekly_seasonality=False)


# In[121]:


model.fit(train)


# In[122]:


# 10 month in test and 12 month to predict new values
months = pd.date_range('2017-06-01','2019-03-01',
                    freq='MS').strftime("%Y-%m-%d").tolist()
future = pd.DataFrame(months)
future.columns = ['ds']
future['ds'] = pd.to_datetime(future['ds'])


# In[123]:


#Use the model to make a forecast
forecast = model.predict(future)


# In[124]:


forecast[["ds",'yhat']].head(10)


# In[125]:


#calculate MAE/RWSE between expected and predicted values for december
y_test = test['y'].values
y_pred = forecast['yhat'].values[:10]
mae_p = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' %mae_p)


# + it is acceptable to predict the conventinal avocado AveragePrice in California, mae = 0.20(about 20% of AveragePrice), compared to the AveragePrice ~1.10.

# # Choose Prophet for predicting avocado prices in California in near future

# In[126]:


y_test


# In[127]:


y_pred


# In[128]:


y_test_value = pd.DataFrame(y_test, index = pd.to_datetime(test['ds']), columns=['Actual'])
y_pred_value = pd.DataFrame(y_pred, index = pd.to_datetime(test['ds']), columns=['Prediction'])


# In[129]:


y_test_value


# In[130]:


y_pred_value


# In[131]:


#Visualize thre result
plt.figure(figsize=(12,6))
plt.plot(y_test_value, label='AveragePrice')
plt.plot(y_pred_value, label='AveragePrice')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()


# In[132]:


fig = model.plot(forecast)
fig.show()
a = add_changepoints_to_plot(fig.gca(), model, forecast)


# In[133]:


fig1 =model.plot_components(forecast)
fig1.show()


# Prediction for next 12 months

# In[134]:


forecast[['ds','yhat']].tail(12)


# Long-term prediction for the next 5 years ==> Consider whether to expand Cultivation/production, and trading

# In[135]:


m = Prophet(yearly_seasonality=True,                daily_seasonality=False, weekly_seasonality=False)
m.fit(df_ts)
future = m.make_future_dataframe(periods=12*5, freq="M") # next 5 years


# In[136]:


forecast = m.predict(future)


# In[137]:


forecast[['ds','yhat', 'yhat_lower', 'yhat_upper','trend', 'trend_lower', 'trend_upper' ]].tail(12)


# In[138]:


forecast.shape


# In[139]:


fig = m.plot(forecast)
fig.show()
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# In[140]:


fig1 =m.plot_components(forecast)
fig1.show()


# In[141]:


plt.figure(figsize=(12,6))
plt.plot(df_ts['y'], label='AveragePrice')
plt.plot(forecast['yhat'], label='AveragePrice with next 60 months prediction',
        color='red')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()


# + Base on the above results , we can see that it is possile to expand the cultivation/production and trading of organic avocados in California
# + Even conventional avocado is more potential because trend raise ratio (2017 -2023(predict)) of conventinal is higherthan organic

# Request 4: Coose 1 Reginal have most potential to expand Avocado(organic and conventional) bussiness
# Reason ?
# prove it ?

# In[142]:


#Potential = Quantity x Price 
data['Volume x Price'] = data['Total Volume'] * data['AveragePrice']
data.head(5)


# In[143]:


mask = data['type']=='organic'
g = sns.factorplot('Volume x Price','region',data=data[mask],
                   hue='year',
                   size=13,
                   aspect=0.8,
                   palette='magma',
                   join=False,
              )


# In[144]:


mask = data['type']=='conventional'
g = sns.factorplot('Volume x Price','region',data=data[mask],
                   hue='year',
                   size=13,
                   aspect=0.8,
                   palette='magma',
                   join=False,
                  )


# In[145]:


#mask = data['type']=='conventional'
g = sns.factorplot('Volume x Price','region',data=data[mask],
                   hue='year',
                   size=13,
                   aspect=0.8,
                   palette='magma',
                   join=False,
              )


# Choose West to expand business Avocado because:
# + Potential = Price x Volume
# + West have highest point(except TotalUS but we don't choose totalUS because it is whole country)
# + And we should focus to conventional because:
# + In conventional, West have 1st point
# + In organic, West have 2nd point

# we analyze West-Organic

# In[146]:


# Make new dataframe from orginal dataframe: data
df_we = data[data['region'] == 'West']
df_we['Date'] = df_we['Date'].str[:-3]
df_we = df_we[data['type'] == 'organic']


# In[147]:


df_we.shape


# In[148]:


agg = {'AveragePrice': 'mean'}
df_we_gr = df_we.groupby(df_we['Date']).aggregate(agg).reset_index()
df_we_gr.head()


# In[149]:


df_ts = pd.DataFrame()
df_ts['ds'] = pd.to_datetime(df_we_gr['Date'])
df_ts['y'] = df_we_gr['AveragePrice']
df_ts.head()


# In[150]:


df_ts.tail()


# In[151]:


df_ts.shape


# In[152]:


# Mean of Organic Avocado AveragePrice in West
df_ts['y'].mean()


# Use df_ts1 for ARIMA, df_ts for Prophet

# In[153]:


df_ts1 = df_ts.copy(deep=False)


# In[154]:


df_ts1.index = pd.to_datetime(df_ts1.ds)


# In[155]:


df_ts1.index


# In[156]:


df_ts1.head()


# In[157]:


df_ts1 = df_ts1.drop(['ds'], axis=1)


# In[158]:


plt.figure(figsize=(15,8))
plt.plot(df_ts1)
plt.title("Avocados' AveragePrice in West")
plt.show()


# In[159]:


decompose_result = seasonal_decompose(df_ts1, model='Mmultiplicative')
decompose_result


# In[160]:


plt.figure(figsize=(15,4))
decompose_result.plot()
plt.show()


# In[161]:


plt.figure(figsize=(15,4))
plt.plot(decompose_result.trend)
plt.show()


# In[162]:


plt.figure(figsize=(15,4))
plt.plot(decompose_result.seasonal)
plt.show()


# In[163]:


plt.figure(figsize=(15,4))
plt.plot(decompose_result.resid)
plt.show()


# With the above result, we can clearly see the seasonal component of the data, and also see that trenf is nonlinear. Residual ranges from -0.2 => 0.3

# Cuoc 4&5: Modeling & Evaluation/Analyze and Report

# Arima

# In[164]:


stepwise_model = auto_arima(df_ts1, start_p=2, start_q=2,
                           max_p=3, max_q=3, m=12,
                           start_P=1, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)


# In[165]:


print(stepwise_model.aic())


# In[166]:


df_ts1.shape


# In[167]:


train = df_ts1.loc['2015-01-01':'2017-06-01']
test = df_ts1.loc['2017-06-01':]


# In[168]:


len(test)


# In[169]:


#Fit model
stepwise_model.fit(train)


# In[170]:


future_forecast = stepwise_model.predict(n_predict=len(test))


# In[171]:


future_forecast


# In[172]:


rmse = sqrt(mean_squared_error(test,future_forecast))
rmse


# In[173]:


mae = mean_absolute_error(test, future_forecast)
mae


# In[174]:


future_forecast = pd.DataFrame(future_forecast, index = test.index, columns=['Prediction'])


# In[175]:


#Visualize the result
plt.figure(figsize=(12,6))
plt.plot(test, label='AveragePrice')
plt.plot(future_forecast, label='AveragePrice Prediction')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()


# In[176]:


plt.figure(figsize=(15,8))
plt.plot(df_ts1, label='AveragePrice All')
plt.plot(future_forecast, label='Prediction')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()


# In[177]:


plt.plot(test, label='AveragePrice')
plt.plot(future_forecast, label='Prediction')
plt.plot(future_forecast-0.5*rmse, label='Prediction')


# Because the increase and decrease is not uniform , it is difficult to find and adaptive value
# (Vi su tang giam khong dong deu nen kho tim duoc gia tri thich nghi)

# Prediction for the next 12 months

# In[178]:


future_forecast_12 = stepwise_model.predict(n_periods=len(test)+12)
future_forecast_12


# In[179]:


plt.figure(figsize=(8,5))
plt.plot(future_forecast_12[len(test):], color='red', label='Prediction')
plt.xticks(rotation='vertical')
plt.title('Prediction next 12 months')
plt.legend()
plt.show()


# In[180]:


future_forecast_12[len(test):]


# In[181]:


months = pd.date_range('2018-04-01', '2019-03-01', freq='MS').strftime("%Y-%m-%d").tolist()


# In[182]:


new_prdict = pd.DataFrame({
    'ds' : months,
    'y': future_forecast_12[len(test):]
})
new_prdict


# + Because of the small amount of data(just over 3 years), the prediction of organic avocado's Average Price in West isn't accurate,
# + Try another prediction algorithm: Prophet (of Facebook)

# In[183]:


# Prophet


# In[184]:


# create test dataset, remove last 10 months
train = df_ts.drop(df_ts.index[-10:])
train.tail()


# In[185]:


test = df_ts.drop(df_ts.index[0:-10])
test


# In[186]:


len(test)


# # Build model

# In[187]:


model = Prophet(yearly_seasonality=True,                daily_seasonality=False, weekly_seasonality=False)


# In[188]:


model.fit(train)


# In[189]:


# 10 month in test and 12 month to predict new values
months = pd.date_range('2017-06-01','2019-03-01',
                    freq='MS').strftime("%Y-%m-%d").tolist()
future = pd.DataFrame(months)
future.columns = ['ds']
future['ds'] = pd.to_datetime(future['ds'])


# In[190]:


#Use the model to make a forecast
forecast = model.predict(future)


# In[191]:


forecast[["ds",'yhat']].head(10)


# In[192]:


#calculate MAE/RWSE between expected and predicted values for december
y_test = test['y'].values
y_pred = forecast['yhat'].values[:10]
mae_p = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' %mae_p)


# + This result shows that prophet's rmse and mae are little higher than ARIMA (0.419 vs 0.374) but prophet have more tool visualize (better chart)
# + We can see that Prophet (Facebooks) algorihm give nearly same as results with ARIMA
# + we choose Prophet algorihm

# # Choose Prophet for predicting avocado prices in West in near future

# In[193]:


y_test


# In[194]:


y_pred


# In[195]:


y_test_value = pd.DataFrame(y_test, index = pd.to_datetime(test['ds']), columns=['Actual'])
y_pred_value = pd.DataFrame(y_pred, index = pd.to_datetime(test['ds']), columns=['Prediction'])


# In[196]:


y_test_value


# In[197]:


y_pred_value


# In[198]:


#Visualize thre result
plt.figure(figsize=(12,6))
plt.plot(y_test_value, label='AveragePrice')
plt.plot(y_pred_value, label='AveragePrice')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()


# In[199]:


fig = model.plot(forecast)
fig.show()
a = add_changepoints_to_plot(fig.gca(), model, forecast)


# In[200]:


fig1 =model.plot_components(forecast)
fig1.show()


# Prediction for next 12 months

# In[201]:


forecast[['ds','yhat']].tail(12)


# Long-term prediction for the next 5 years ==> Consider whether to expand Cultivation/production, and trading

# In[202]:


m = Prophet(yearly_seasonality=True,                daily_seasonality=False, weekly_seasonality=False)
m.fit(df_ts)
future = m.make_future_dataframe(periods=12*5, freq="M") # next 5 years


# In[203]:


forecast = m.predict(future)


# In[204]:


forecast[['ds','yhat', 'yhat_lower', 'yhat_upper','trend', 'trend_lower', 'trend_upper' ]].tail(12)


# In[205]:


forecast.shape


# In[206]:


fig = m.plot(forecast)
fig.show()
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# In[207]:


fig1 =m.plot_components(forecast)
fig1.show()


# In[208]:


plt.figure(figsize=(12,6))
plt.plot(df_ts['y'], label='AveragePrice')
plt.plot(forecast['yhat'], label='AveragePrice with next 60 months prediction',
        color='red')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()


# Base on the above results , we can see that it is possile to expand the cultivation/production and trading of organic avocados in West

# In[209]:


# Make new dataframe from orginal dataframe: data
df_we = data[data['region'] == 'West']
df_we['Date'] = df_we['Date'].str[:-3]
df_we = df_we[data['type'] == 'conventional']


# In[210]:


df_we.shape


# In[211]:


agg = {'AveragePrice': 'mean'}
df_we_gr = df_we.groupby(df_we['Date']).aggregate(agg).reset_index()
df_we_gr.head()


# In[212]:


df_ts = pd.DataFrame()
df_ts['ds'] = pd.to_datetime(df_ca_gr['Date'])
df_ts['y'] = df_we_gr['AveragePrice']
df_ts.head()


# In[213]:


df_ts.tail()


# In[214]:


df_ts.shape


# In[215]:


# Mean of conventinal Avocado AveragePrice in West
df_ts['y'].mean()


# In[216]:


df_we.shape

agg = {'AveragePrice': 'mean'}
df_we_gr = df_we.groupby(df_we['Date']).aggregate(agg).reset_index()
df_we_gr.head()

df_ts = pd.DataFrame()
df_ts['ds'] = pd.to_datetime(df_we_gr['Date'])
df_ts['y'] = df_we_gr['AveragePrice']
df_ts.head()

df_ts.tail()

df_ts.shape

# Mean of Conventinal Avocado AveragePrice in West
df_ts['y'].mean()


# In[217]:


# Prophet


# In[218]:


# create test dataset, remove last 10 months
train = df_ts.drop(df_ts.index[-10:])
train.tail()


# In[219]:


test = df_ts.drop(df_ts.index[0:-10])
test


# In[220]:


len(test)


# # Build model

# In[221]:


model = Prophet(yearly_seasonality=True,                daily_seasonality=False, weekly_seasonality=False)


# In[222]:


model.fit(train)


# In[223]:


# 10 month in test and 12 month to predict new values
months = pd.date_range('2017-06-01','2019-03-01',
                    freq='MS').strftime("%Y-%m-%d").tolist()
future = pd.DataFrame(months)
future.columns = ['ds']
future['ds'] = pd.to_datetime(future['ds'])


# In[224]:


#Use the model to make a forecast
forecast = model.predict(future)


# In[225]:


forecast[["ds",'yhat']].head(10)


# In[226]:


#calculate MAE/RWSE between expected and predicted values for december
y_test = test['y'].values
y_pred = forecast['yhat'].values[:10]
mae_p = mean_absolute_error(y_test, y_pred)
print('MAE: %.3f' %mae_p)


# + it is acceptable to predict the conventinal avocado AveragePrice in California, mae = 0.141(about 14% of AveragePrice), compared to the AveragePrice ~1.05.

# # Choose Prophet for predicting avocado prices in West in near future

# In[227]:


y_test


# In[228]:


y_pred


# In[229]:


y_test_value = pd.DataFrame(y_test, index = pd.to_datetime(test['ds']), columns=['Actual'])
y_pred_value = pd.DataFrame(y_pred, index = pd.to_datetime(test['ds']), columns=['Prediction'])


# In[230]:


y_test_value


# In[231]:


y_pred_value


# In[232]:


#Visualize thre result
plt.figure(figsize=(12,6))
plt.plot(y_test_value, label='AveragePrice')
plt.plot(y_pred_value, label='AveragePrice')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()


# In[233]:


fig = model.plot(forecast)
fig.show()
a = add_changepoints_to_plot(fig.gca(), model, forecast)


# In[234]:


fig1 =model.plot_components(forecast)
fig1.show()


# Prediction for next 12 months

# In[235]:


forecast[['ds','yhat']].tail(12)


# Long-term prediction for the next 5 years ==> Consider whether to expand Cultivation/production, and trading

# In[236]:


m = Prophet(yearly_seasonality=True,                daily_seasonality=False, weekly_seasonality=False)
m.fit(df_ts)
future = m.make_future_dataframe(periods=12*5, freq="M") # next 5 years


# In[237]:


forecast = m.predict(future)


# In[238]:


forecast[['ds','yhat', 'yhat_lower', 'yhat_upper','trend', 'trend_lower', 'trend_upper' ]].tail(12)


# In[239]:


forecast.shape


# In[240]:


fig = m.plot(forecast)
fig.show()
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# In[241]:


fig1 =m.plot_components(forecast)
fig1.show()


# In[242]:


plt.figure(figsize=(12,6))
plt.plot(df_ts['y'], label='AveragePrice')
plt.plot(forecast['yhat'], label='AveragePrice with next 60 months prediction',
        color='red')
plt.xticks(rotation='vertical')
plt.legend()
plt.show()


# + Base on the above results , we can see that it is possile to expand the cultivation/production and trading of organic avocados in West
# + Even conventional avocado is more potential because trend raise ratio (2017 -2023(predict)) of conventinal is higher than organic

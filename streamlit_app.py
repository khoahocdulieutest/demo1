import streamlit as st
from multi_app import MultiApp
from apps import home, topic1_project, summary, model_regression, model_timeseries
STATE = dict()
app = MultiApp(STATE)

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Summary", summary.app)
app.add_app("Predict by Regression", model_regression.app)
app.add_app("Predict by Time series", model_timeseries.app)
# The main app
app.run()
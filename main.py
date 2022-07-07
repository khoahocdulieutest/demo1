import streamlit as st
from multi_app import MultiApp
from apps import home, topic1_project  # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Topic 1", topic1_project.app)
# app.add_app("Data Stats", data_stats.app)
# app.add_app("Huấn luyện", visual_loss.app)
# app.add_app("Kết quả", visual_app.app)
# The main app
app.run()
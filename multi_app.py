"""Frameworks for running multiple Streamlit applications as a single app.
"""
import streamlit as st
import pandas as pd

class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """
    def __init__(self, state):
        self.apps = []
        self.state = state

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.radio(
            'Go To',
            self.apps,
            format_func=lambda app: app['title'])



        data_file = st.sidebar.file_uploader("Upload CSV",type=["csv"])
        if data_file is not None:
            file_details = {"filename": data_file.name, "filetype": data_file.type,
                            "filesize": data_file.size}

            st.write(file_details)
            df = pd.read_csv(data_file)
            #st.dataframe(df)
            #self.state['data'] = data_file.name
            st.write(data_file.name)
            st.session_state['data'] = df

        option_model = st.sidebar.selectbox(
            'Choose the model?',
            ('LinearRegression', 'RandomForestRegressor', 'XGBRegressor'))

        st.write('You selected:', option_model)

        app['function'](self.state)
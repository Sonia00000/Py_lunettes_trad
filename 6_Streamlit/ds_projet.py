# -*- coding: utf-8 -*-

#Frameworks for running multiple Streamlit applications as a single app.

import streamlit as st

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
    def __init__(self):
        self.apps = []

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
            "Aller à l'onglet :",
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()
        

st.sidebar.title("Py Translator Glasses")
st.sidebar.markdown("## Menu")
        
# Activation du multi-pages
app = MultiApp()

import ds_projet_intro
import ds_projet_dataset    
import ds_projet_metho
import ds_projet_model
import ds_projet_demo
import ds_projet_conclusion

app.add_app("Introduction", ds_projet_intro.app)
app.add_app("Analyse des datasets", ds_projet_dataset.app)
app.add_app("Méthodologie", ds_projet_metho.app)
app.add_app("Modélisation", ds_projet_model.app)
app.add_app("Application - Démo", ds_projet_demo.app)
app.add_app("Conclusion et Perspectives", ds_projet_conclusion.app)

app.run()

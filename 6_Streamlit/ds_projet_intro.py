
import streamlit as st

def app():
    st.title("Introduction")
    
    st.markdown("## Contexte")
    st.markdown("Projet réalisé dans le cadre d'une certification de Data Science entre juillet et octobre 2021.") 
    
    st.markdown("## Concepteurs")
    st.markdown("MHEDHBI Sonia, MOREAU Coralie, SAADAOUI Mounir.")
    
    st.markdown("## Description du projet")
    st.markdown("Les lunettes connectées sont un système pré-existant de transcription de phrases parlées en langue anglaise vers une version écrite et diffusée sur le verre des lunettes. Notre projet consiste à réaliser un système automatique de traduction de l'anglais vers le français pour implémentation sur ce système existant.")
    
    st.markdown("## Modèles utilisés")
    st.markdown("Modèle statistique, Spatialisation de mots (WordEmbedding), Réseaux de neurones récurrents avec système d'attention (Seq2Seq) avec et sans BeamSearch, Systèmes d'attention seuls (Transformer). Utilisation des scoring BLEU et ROUGE.")
    
    st.markdown("## Langages et outils utilisés")
    st.markdown("Python 3, VSCode, Git, Jupyter, Pandas, Numpy, Tensorflow, Keras.")
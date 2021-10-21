import pandas as pd

import streamlit as st 

from PIL import Image

def app():
    st.title("Jeux de données")
    
    st.write("Deux jeux de données ont été étudiés pour les modèles de traduction phrase par phrase.") 
    
    st.image(Image.open("Images/Tableau_recap.png"))
    st.markdown("## Petit ensemble de phrases")
    
    selec_pt = st.expander("Sélection de phrases au hasard")
    
    phrase_pt=pd.read_csv('Data/small_vocab_fr-eng.csv', sep= ';')
    phrase_pt=phrase_pt.drop(["Français"], axis=1)
    
    selec_pt.write(phrase_pt["English"].sample(5))
    
    wc = st.expander("WordCloud")
    wc.write("**WordCloud des mots en français**")
    wc.image(Image.open("Images/wordcloud_fr.png"))
    
    wc.write("**WordCloud des mots en anglais**")
    wc.image(Image.open("Images/wordcloud_eng.png"))
    
    st.markdown("## Grand ensemble de phrases")
    
    selec_gd = st.expander("Sélection de phrases au hasard")
    
    phrase_gd=pd.read_csv("Data/english.txt", sep = "\n\n", names = ["English"], encoding = 'utf-8', engine = 'python')
    
    selec_gd.write(phrase_gd.sample(5))
    
    comparaison = st.expander("Comparaison du nombre de mots en français et en anglais")
    comparaison.image(Image.open("Images/Comparaison_fr_eng.png"))
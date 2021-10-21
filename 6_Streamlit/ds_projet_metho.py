# -*- coding: utf-8 -*-

import streamlit as st
from PIL import Image

def app():
    header = st.container()
    with header:
        st.title("Méthodologie")
        
    traduction = st.container()
    with traduction:
        st.markdown("## Modèles de traduction")
        st.markdown("### Traduction mot à mot")
        st.markdown("- Système de fenêtre de traduction")
        st.write("Ce système repose sur le principe que pour tout mot anglais, sa traduction se trouve dans une position similaire dans la phrase traduite.")
        fenetre = st.expander("Illustration du système de fenêtre de traduction")
        fenetre.image(Image.open("Images/Trad_fenetre.png"))
        fenetre.write("Cela peut se vérifier sur des phrases très simples mais lorsque les phrases sont plus complexes ou que les langues sont moins proches en termes de syntaxe, cette méthode n'apparait plus efficace.")
        st.markdown("- Word Embedding")
        st.write("Le word embedding permet la repésentation vectorielle des mots en conservant leur signification et leur relation avec d'autres mots.")
        embedding = st.expander("Traduction avec le word embedding")
        embedding.write("*Visualisation par ACP de mots vectorisés*")
        embedding.image(Image.open("Images/Embedding.png"))
        embedding.write("Le principe de la traduction consiste en l’identification d’une correspondance entre l’espace vectoriel des mots sources (anglais) et l’espace vectoriel des mots cibles (français) via une matrice de transformation.")
        st.markdown("")
        st.write("Ainsi, bien que la traduction mot à mot repose sur un modèle naïf à priori rejeté par les conclusions précédentes sur la structure des phrases notamment, elle permet de mieux comprendre les fondements de la méthode de traduction phrase par phrase, notamment l'embedding qui est à la base de tous les modèles qui suivent.")
        st.markdown("### Traduction phrase par phrase")        
        st.markdown("- Modèle Seq2Seq")
        st.markdown("- Transformeur")
        st.write("Ces modèles de traduction phrases par phrases sont développés dans la partie de modélisation.")

        
    projet = st.container()
    with projet:
        st.markdown("## Méthodologie de modélisation")
        st.image(Image.open("Images/Methodologie.png"))
        pre_trait = st.expander("Détails sur l'étape de pre-processing des données")
        pre_trait.write("""
        * Pré-traitement des données :
            * Tokenisation : Chaque phrase est transformée en liste d'entiers.
            * Ajout de tokens spécifiques signalant le début et la fin de chaque phrase
            * Uniformisation de la longueur des phrases : des tokens de padding sont ajoutés pour compléter les phrases trop courtes et les phrases trop longues sont tronquées pour respecter la longueur souhaitée.
        """)
        
    result = st.container()
    with result:
        st.markdown("## Evaluation du modèle")
        st.write("L'évaluation des modèles de traduction phrase par phrase se fait par comparaison des traductions sorties du modèle avec les traductions de référence.")
        st.markdown("- Test de performance intuitif")
        perf = st.expander("Détails sur le test de performance implémenté")
        perf.write("""
            Test de performance : moyenne des deux scores suivants :
            * Ratio de la différence du nombre de mots entre les deux phrases sur le nombre de mots de la phrase de référence
            * Ratio des racines (stemming) des mots communs entre les deux phrases sur le nombre de mots de la phrase de référence
            """)
        st.markdown("- Score BLEU 1-gram")
        bleu = st.expander("Détails sur le score BLEU")
        bleu.write("Le score BLEU (Bilingual Evaluation Understudy Score) est le score de référence pour évaluer les traductions émises par les systèmes de traductions automatiques (phrases candidates) par rapport à des phrases traduites dites de référence.")
        bleu.write("Il permet de calculer un ratio des n-grams communs aux phrases candidates et de référence sur les n-grams présents dans les références (n-grams représentant des groupes de tokens de n tokens) sans distinction d'ordre, et ressort un score compris entre 0 et 1, un score de 1 constituant une correspondance parfaite.")
        st.markdown("- Score ROUGE 1-gram : métrique f1_score")
        rouge = st.expander("Détails sur le score ROUGE")
        rouge.write("Le score ROUGE (Recall-Oriented Understudy for Gisting Evaluation) s'utilise principalement dans les travaux de résumés automatiques de texte mais également pour les traductions automatiques.")
        rouge.write("""
            * le **rappel** (recall) qui se calcule par le ratio du nombre de n-grams communs aux deux phrases sur le nombre de n-grams dans la phrase de référence. Cela permet de juger si la traduction du modèle contient bien l'ensemble des mots de la phrase de référence.
            * la **précision** qui se calcule par le ratio du nombre de n-grams communs aux deux phrases sur le nombre de n-grams dans la traduction du modèle. Cela permet de juger de la pertinence des mots utilisés dans la traduction du modèle.
            * le **f1_score** qui correspond à la moyenne harmonique des deux métriques précédentes. Ce score sera considéré come le score ROUGE du modèle.
            """)
        st.markdown("")
        st.markdown("*Pour plus d'information sur les n-grams utilisés dans les scores BLEU et ROUGE, cliquez ci-dessous*")
        gram = st.expander("Exemple score ROUGE 2-grams")
        gram.write("Les 2-grams représentent des groupes de 2 mots qui s’enchainent dans chaque phrase.")
        gram.image(Image.open("Images/Grams_ex.png"))

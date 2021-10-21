# -*- coding: utf-8 -*-

import streamlit as st
from PIL import Image

def app():

    header = st.container()
    with header:
        st.title("Modélisation")
        
    modele = st.container()
    with modele:
        st.markdown("## Architecture des modèles étudiés")
        st.write("Les modèles de traduction phrase par phrase étudiés sont constitués d'un système encodeur-décodeur qui traite l'intégralité des phrases originales et traduites.")
        
        seq2seq = st.container()
        with seq2seq:
            st.markdown("### Modèle Seq2Seq")
            st.write("Le modèle Seq2Seq prend en entrée une séquence de mots puis renvoit la séquence de mots traduits après un passage dans des couches RNN notamment.")
            st.image(Image.open("Images/Seq2seq_attention.png"))
            with seq2seq.expander("Mécanisme du modèle Seq2Seq"):
                seq2seq_enc, seq2seq_dec = st.columns(2)
                seq2seq_enc.image(Image.open("Images/Encodage.png"))
                seq2seq_dec.image(Image.open("Images/Decodage.png"))
                st.write("Au simple modèle Seq2Seq peuvent être ajoutés des mécanismes d’attention permettant d‘améliorer les performances du modèle basique.")
                st.write("Pour ce faire, ces mécanismes permettent de pallier les difficultés rencontrées par le modèle Seq2Seq sur les phrases plus longues et plus complexes en poussant le modèle à « prêter attention » aux mots les plus importants de la phrase originale pendant la traduction : le modèle fait ainsi des liaisons directes entre la phrase originale et sa traduction (input du décodeur) dans les couches RNN du décodeur.")
                st.write("""
                         Les mécanismes d’attention s’intègrent au niveau du décodeur en créant deux vecteurs consécutifs :
                        * **Un vecteur d’alignement** qui associe à chaque mot de la phrase originale la probabilité (ou score) que le mot traduit corresponde à celui de la phrase originale : on parle alors de poids d’attention.
                        * **Un vecteur de contexte** s’obtient par produit scalaire du vecteur d’alignement et de l’output de l’encodeur. C’est ce vecteur qui sert d’input dans les couches RNN du décodeur.
                         """)       
            
        transformer = st.container()
        with transformer:
            st.markdown("### Transformeur")
            st.write("La méthode de Transformer est une technique plus évoluée pour traiter des données séquentielles, son avantage majeur résidant dans le fait de n’utiliser que des mécanismes d’attention, sans couches récurrentes (trop longues à entraîner). Les mécanismes d’attention sont précieux en traitement du langage car ils permettent aux réseaux de neurones d’identifier les mots les plus importants.")
            st.write(" **Attention is all you need.**")
            st.image(Image.open("Images/Transformer_full.png"))
            
            encoder = st.container()   
            with encoder:
                st.markdown("#### Encodeur")
                enc_meca = st.expander("Mécanisme de l'encodeur")
                enc_meca.write("""
                               Les phrases à traduire sont tout d’abord encodées par une couche d’embedding. Contrairement au modèle Seq2Seq, le transformer traduit l’ensemble des mots en parallèle. Il faut donc rajouter à l’embedding le codage de la position des mots dans la phrase. 
                               
                               La couche d’encodage contient les mécanismes suivants :
                                * L’attention multi-têtes lance plusieurs mécanismes en parallèle s’attachant à des dépendances différentes entre les mots (long/court terme, grammaticale, syntaxique)
                                * Des couches de Dropout et Layer Normalisation vont servir à régulariser le modèle
                                * Des Skip Connections sont rajoutées pour éviter le problème de vanishing gradient à cause des petits gradients de la fonction softmax
                                * Un réseau dense (Feed Forward Neural Network)
                               """)
                encoder_step = st.container()
                with encoder_step.expander("Schéma des structure et étapes de l'encodeur"):
                    enc_graph, enc_desc = st.columns(2)
                    enc_graph.image(Image.open("Images/Transformer_encodeur_step_graph.png"))
                    enc_desc.image(Image.open("Images/Transformer_encodeur_step_desc.png"))
            
            decoder = st.container()
            with decoder:
                st.markdown("#### Décodeur")
                dec_meca = st.expander("Mécanisme du décodeur")
                dec_meca.write("""
                               * L’entrée du Transformer contient à la fois la phrase à traduire et la traduction partielle de celle-ci (mode de fonctionnement Autorégressif). Cette entrée est transformée via une couche d’embedding et le codage de la position des mots de la phrase partiellement traduite.
                               * L’étape d’Auto-Attention permet au Décodeur de compiler les informations en sa possession (phrase partiellement traduite, encodage du mot à traduire) puis d’interagir avec l’Encodeur grâce à un vecteur de requête.
                               * Dans le mécanisme d’attention Encodeur-Décodeur, l’Encodeur fournit des vecteurs avec des clés et valeurs permettant in fine de réaliser un vecteur produit.
                               * Une classification classique (avec couche dense et activation Softmax) est ensuite réalisée afin de déterminer le prochain mot traduit.
                               """)
                decoder_step = st.container()
                with decoder_step.expander("Schéma des structure et étapes du décodeur"):
                    dec_graph, dec_desc = st.columns(2)
                    dec_graph.image(Image.open("Images/Transformer_decodeur_step_graph.png"))
                    dec_desc.image(Image.open("Images/Transformer_decodeur_step_desc.png"))
                    
        beam = st.container()
        with beam:
            st.markdown("### Approche Greedy / Beam-Search")
            st.write("""
                     * Approche Greedy : la phrase traduite est constituée par l'assemblage des traductions les plus probables pour chaque mot source.
                     * Approche Beam-Search : A chaque nouveau mot à traduire, les k traductions les plus probables sont gardées en mémoire. Au mot suivant, les combinaisons des k plus probables traductions de chacune des k traductions en mémoire sont évaluées et seules les k plus probables traductions de ces combinaisons sont conservées. A la fin de la phrase, on peut choisir de ne conserver qu'une à k traductions possibles. 
                     """)
            
    st.markdown("____")
            
    implem = st.container()
    with implem:
        st.markdown("## Modèles implémentés")
        st.image(Image.open("Images/Modelisation.png"))
        
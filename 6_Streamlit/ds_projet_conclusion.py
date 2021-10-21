import streamlit as st

def app():
    st.title('Conclusion et perspectives')
    st.header('Conclusion')
    st.write("Ce projet s'inscrivant dans la continuité du projet Voice Glass a été très intéréssant à réaliser. "
            " Il nous a permis de développer nos connaissances en Deep Learning notamment dans le domaine du traitement automatique des "
            "langues (NLP). De plus, les Voice Glasses étant destinées à un public restreint (sourds et malentendants), élargir"
            " son utilisation à un public plus vaste en implémentant un système de traduction est une idée assez ingénieuse et ambitieuse.")
    
    st.write("Cela nous a également permis de réaliser un projet de data science du début à la fin en suivant les principales étapes"
            " tel que le choix de jeux de données, la data visualisation, la modélisation...")
    st.header("Perspectives")
    st.subheader("Entrainements plus longs et modification de certains paramètres")
    st.write("La première chose que nous aurions voulu faire mais que nous n'avons pu par faute de ressources de calcul et de temps"
            " aurait été de pouvoir réaliser des entrainements de modèles plus aboutis. En effet, l'entrainement et la génération des scores"
            " prenant énormément de temps, nous avons donc été contraints d'aller à l'essentiel en terme d'essais de modèles et de paramètres"
            " d'entraînement. Nous aurions aimé essayer plusieurs combinaisons de paramètres tel que les cellules utilisées pour le RNN  "
            " (GRU et LSTM), les différents mécanismes d'attention (Luong et Bahdanau), le nombre d'époques d'entrainement et les méthodes"
            " de traduction (Greedy et Beam Search). Nous aurions également aimé pouvoir explorer les techninques de post traitement telles"
            " que les modèles de langue.")
    
    st.subheader('Prototypage des lunettes connectées')
    st.write("A l'instar du groupe ayant réalisé le projet Voice Glass, nous aurions aimé prototyper une paire de lunettes connectées capable de traduire en temps réel une phrase prononcée par un interlocuteur "
            " en combinant les deux projets, c'est à dire "
            " la reconnaisance vocale et le système de traduction. ")
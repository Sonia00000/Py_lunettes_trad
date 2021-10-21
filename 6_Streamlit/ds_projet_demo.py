# -*- coding: utf-8 -*-
from numpy.lib.function_base import select
from sklearn import model_selection
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RNN, Lambda, Embedding
import unicodedata
import re
import os


#################################################Prétraitement des données####################################################


def app():
    st.title('Démo')
    
    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    
    
    def clean_sentence(w):
       
        # conversion unicode vers ascii
        w = unicode_to_ascii(w.lower().strip())
    
        # séparation entre un mot et sa ponctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
    
        # remplacement de caractères spéciaux par un espace
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = w.strip()
    
        # ajout des balises <start> et <end> placées respectivement au début et à la fin des phrases
        w = '<start> ' + w + ' <end>'
        
        return w
    
    @st.cache(allow_output_mutation=True)
    def load_transform_data(file_fr, file_en):
        fr = pd.read_csv(file_fr, sep = "\n\n", names = ["Français"], encoding = 'utf-8', engine = 'python')
        en = pd.read_csv(file_en, sep = "\n\n", names = ["English"], encoding = 'utf-8', engine = 'python')
        data = pd.concat([en, fr], axis=1)
        data.English = data.English.apply(lambda x: clean_sentence(x))
        data.Français = data.Français.apply(lambda x: clean_sentence(x))
        return data, pd.concat([en, fr], axis=1)
    
        
    data, data_transf  = load_transform_data('Data/french.txt', 'Data/english.txt')
    
    
    def tokenize(sentences):
        # tokenization des phrases
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(sentences)
    
        # transformation des phrases en séquences
        seq = tokenizer.texts_to_sequences(sentences)
        
        # complète les séquences de sorte à ce qu'ils aient la même longueur
        seq = tf.keras.preprocessing.sequence.pad_sequences(seq,padding='post')
    
        # post : le padding est réalisé à la fin des séquences
        return seq, tokenizer
    
    @st.cache(allow_output_mutation=True)
    def tokenize_data(dataEnglish, dataFrançais):
        input_seq, input_tokenizer = tokenize(dataEnglish) # phrases tokenisées en fonction du vocabulaire anglais  
        target_seq, target_tokenizer = tokenize(dataFrançais)
        return input_seq, input_tokenizer, target_seq, target_tokenizer
    
    input_seq, input_tokenizer, target_seq, target_tokenizer = tokenize_data(data.English, data.Français)
    
    
    # Calcul de la taille des vocabulaires via l'attribut .word_index qui transforme l'index d'un mot en sa chaîne de caractères 
    vocab_size_inp = len(input_tokenizer.word_index)+1 # nombre de mots différents en français 
    vocab_size_targ = len(target_tokenizer.word_index)+1 # nombre de mots différents en anglais 
    
    # Calcul de longeur maximale des séquences 
    max_length_inp, max_length_targ = input_seq.shape[1], target_seq.shape[1] 
    # Création des ensembles d'apprentissage et test
    X_train, X_test, y_train, y_test = train_test_split(input_seq, target_seq,test_size=0.2)
    
    # Paramètres de l'entrainement
    # buffer_size: permet de déterminer combien d'éléments vont être mélangés au fur et à mesure, 
    
    # batch_size: sépare dans l'ordre le dataset en jeux de données de dimension fixe
    batch_size = 64
    buffer_size = int(len(X_train)) 
    BATCH_SIZE = 64
    units = 1024 # nombre d'états cachés (2 puissance 10)
    embedding_dim_beam = 256 # dimension de la matrice d'embedding
    steps_per_epoch = buffer_size//BATCH_SIZE
    
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    example_input_batch, example_target_batch = next(iter(dataset))
    
    def clean_sentence_transf(w):
    
        # séparation entre un mot et sa ponctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        
        return w
    
    @st.cache(allow_output_mutation=True)
    def transform_data(data_transf):
        data_transf.English = data_transf.English.apply(lambda x: clean_sentence_transf(x))
        data_transf.Français = data_transf.Français.apply(lambda x: clean_sentence_transf(x))
    
        corpus_en=[]
        corpus_fr=[]
    
        for sentences in data_transf.English: 
            corpus_en.append(sentences)
    
        for sentences in data_transf.Français: 
            corpus_fr.append(sentences)
    
        return corpus_en, corpus_fr
    
    corpus_en, corpus_fr = transform_data(data_transf)
    
    def tokenize_transf(sentences):
        tokenizer = Tokenizer(num_words = 32000)
        tokenizer.fit_on_texts(sentences)
        seq = tokenizer.texts_to_sequences(sentences)
        return seq, tokenizer
    
    
    @st.cache(allow_output_mutation=True)
    def tokenize_data_transf(corpus_fr, corpus_en):
    
        # Transformation des corpus en une liste de listes de nombres (correspondant chacun à un mot unique)
        tokenized_fr, tokenizer_fr = tokenize_transf(corpus_fr)
        tokenized_en, tokenizer_en = tokenize_transf(corpus_en)
        
        return tokenizer_fr, tokenizer_en,tokenized_fr, tokenized_en 
    
    
    model_list = ['Seq2Seq Greedy', 'Seq2Seq Beam Search Decoder', 'Transformer']
    selected_model = st.selectbox('Veuillez choisir un modèle', model_list)
    st.write("Modèle choisi : ", selected_model)
    
    if selected_model == 'Seq2Seq Greedy':
        st.write("""
                 Evaluation globale du modèle choisi
                 * Score performance : 85%
                 * Score BLEU : 76%
                 * Score ROUGE : 72%
                 """)
        class Encoder(tf.keras.Model):
         
            def __init__(self, vocab_size, embedding_dim, latent_dim):
            
                # vocab_size: taille du vocabulaire dans la langue de départ
                # embedding_dim: dimension de la matrice d'embedding 
                # latent_dim : dimension de l'état caché (units)
                
                super(Encoder, self).__init__() # pour faciliter des questions d'héritage 
                
                self.units = latent_dim
                
                # sélection de l'embedding, fonction qui permet de vectoriser les mots représentés précédemment par leur index 
                self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
                
                # sélection de la cellule de calcul GRU 
                self.gru = tf.keras.layers.GRU(self.units, # nombre d'états cachés - neurones  
                                            return_sequences=True, # retourne une séquence/ bidirectionelle
                                            return_state=True, # permet de conserver la mémoire de l'encodeur et le rend disponible pour le décodeur 
                                            recurrent_initializer='glorot_uniform') # initialisation des poids de la matrice utilisée pour la transformation linéaire des états récurrents par distribution uniforme 
            
            def call(self, x, hidden): # calcul des sorties du modèle 
                x = self.embedding(x)
                output, state = self.gru(x, initial_state = hidden)
                return output, state
            
            def initialize_hidden_state(self, batch_size):
                return tf.zeros((batch_size, self.units)) # définition d'une matrice de dim jeu de données fixé * nombre d'états cachés 
    
        # Paramètres du modèle
    
        latent_dim = 512 # nombre d'états cachés (2 puissance 9)
        embedding_dim = 300 # dimension de la matrice d'embedding
    
    
    
    
        class BahdanauAttention(tf.keras.layers.Layer):
            
            def __init__(self, units):
                super(BahdanauAttention, self).__init__()
                # définition de 3 couches de neuronnes denses 
                self.W1 = tf.keras.layers.Dense(units)# couche de l'état caché 
                self.W2 = tf.keras.layers.Dense(units) # couche de sortie de l'encodeur
                self.V = tf.keras.layers.Dense(1) # couche de sortie du mécanisme d'attention  
    
            def call(self, hidden, enc_output):
                # dimensions de 'hidden' : (batch_size, units)
                # dimensions de 'enc_output' : (batch_size, max_length_inp, units)
                
                hidden_with_time_axis = tf.expand_dims(hidden, 1)
                # dimensions de 'hidden_with_time_axis' : (batch_size, 1, units)
                # ce changement de dimension est nécessaire pour le calcul du score
                
                # Application de la formule score = FC(tanh(FC(EO) + FC(H))
                # avec FC: couche Fully-connected (dense layer) c'est-à-dire W1 et W2
                # EO: sortie de l’Encodeur, encoder_output 
                # H : état caché, hidden_state. 
    
                score = self.V(tf.nn.tanh(self.W1(hidden_with_time_axis) + self.W2(enc_output)))
                # dimensions de 'score' : (batch_size, max_length_inp, 1)
                # dimensions avant d'appliquer 'self.V' : (batch_size, max_length_inp, units)
    
                # calcul des poids d'attention via la fonction softmax 
                attention_weights = tf.nn.softmax(score, axis=1)
                # dimensions de 'attention_weights' : (batch_size, max_length_inp, 1)
                
                # calcul du vecteur contexte par produit scalaire
                context_vector = attention_weights * enc_output
                context_vector = tf.reduce_sum(context_vector, axis=1)
                # dimensions de context_vector après somme (batch_size, units)
    
                return context_vector, attention_weights
    
        class Decoder(tf.keras.Model):
            
            def __init__(self, vocab_size, embedding_dim, latent_dim, attention_layer):
                # même architecture que l'encodeur
                super(Decoder, self).__init__()
                
                self.units = latent_dim
                
                self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
                self.gru = tf.keras.layers.GRU(self.units,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')
                
                # couche dense comprenant le vocabulaire de la langue cible 
                self.fc = tf.keras.layers.Dense(vocab_size)
    
                # prise en compte de la notion d'attention en input du décodeur 
                self.attention = attention_layer(latent_dim)
    
            def call(self, x, hidden, enc_output):
                # dimensions de 'enc_output' : (batch_size, max_length_inp, latent_dim)
                
                context_vector, attention_weights = self.attention(hidden, enc_output)
                # dimensions de 'context_vector' : (batch size, latent_dim)
                # dimensions de 'attention_weights' : (batch_size, max_length_inp, 1)
    
                x = self.embedding(x)
                # dimensions de 'x' : (batch_size, 1, embedding_dim)
    
                x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
                # dimensions de 'x' : (batch_size, 1, embedding_dim + latent_dim)
                
                output, state = self.gru(x)
                
                output = tf.reshape(output, (-1, output.shape[2]))
                # dimensions de 'output' : (batch_size * 1, latent_dim )
    
                output = self.fc(output)
                # dimensions de 'output' : (batch_size, vocab)
    
                return output, state, attention_weights
    
        @st.cache(allow_output_mutation=True)
        def model_creator(vocab_size_inp, embedding_dim,latent_dim, vocab_size_targ ):
            encoder = Encoder(vocab_size_inp, embedding_dim, latent_dim)
            attention_layer = BahdanauAttention(latent_dim)
            decoder = Decoder(vocab_size_targ, embedding_dim, latent_dim, BahdanauAttention)
            return encoder, attention_layer, decoder
    
        encoder, attention_layer, decoder = model_creator(vocab_size_inp, embedding_dim,latent_dim, vocab_size_targ )
    
        hidden = encoder.initialize_hidden_state(batch_size) # état caché
        enc_output, hidden = encoder(example_input_batch, hidden) # sortie
        attention_result, attention_weights = attention_layer(hidden, enc_output)
    
    
        dec_input = tf.random.uniform((batch_size, 1))
        dec_output, _, _ = decoder(dec_input, hidden, enc_output)
        optimizer = tf.keras.optimizers.Adam()
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "cp.ckpt")
    
        #@st.cache(allow_output_mutation=True)
        def restore_model_greedy(checkpoint_dir, optimizer, encoder, decoder):
            checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                        encoder=encoder,
                                        decoder=decoder)
            return checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    
        restore_model_greedy(checkpoint_dir, optimizer, encoder, decoder)
    
        # Définition de la fonction de traduction 
    
        def translate(sentence, is_seq = False):
            
            # Initialisation de la matrice d'attention
            attention_matrix = np.zeros((max_length_targ, max_length_inp))    
            
            # permet de transformer l'entrée sous forme de phrase en séquence (pour de nouvelles phrases non séquencées)
            if not(is_seq):
                # prétraitement de la phrase d'entrée
                sentence = clean_sentence(sentence)
                # transformation de la phrase en séquence
                sentence = input_tokenizer.texts_to_sequences([sentence])
                sentence = tf.keras.preprocessing.sequence.pad_sequences(sentence,
                                                                    maxlen=max_length_inp,
                                                                    padding='post')
                
            # initialisation des variables
            hidden = [tf.zeros((1, latent_dim))]
            enc_out, enc_hidden = encoder(sentence, hidden)
            dec_hidden = enc_hidden
            
            # remplissage du premier élement par l'indice associé à la balise <start>
            dec_input = tf.expand_dims([input_tokenizer.word_index['<start>']], 0)
    
            stop_condition = False # initialisation du status de la traduction -> True correspond à la balise <"end">
            words = [] # initialisation de la phrase de sortie
            t = 0
    
            while not stop_condition:
                
                predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                                dec_hidden,
                                                                enc_out)
    
                # conservation des poids d'attention
                attention_weights = tf.reshape(attention_weights, (-1, ))
                attention_matrix[t] = attention_weights.numpy()
                
                # utilisation du mot ayant la meilleure probabilité pour la prédiction 
                predicted_index = tf.argmax(predictions[0]).numpy()
                
                if predicted_index != 0:
                    word = target_tokenizer.index_word[predicted_index]
                else :
                    word = ''
    
                # retour de la prédiction id dans le modèle 
                dec_input = tf.expand_dims([predicted_index], 0)
                
                # verification des conditions de sortie: 
                if (word == '<end>' or # la balise <end> 
                t >= max_length_targ-1): # longeur maximale atteinte
                    
                    stop_condition = True
                    break
                    
                # ajout du mot à la phrase de sortie    
                words.append(word)
                
                t+=1
    
            return " ".join(words)
        sentence = st.text_input('Veuillez entrer une phrase en anglais')
        if sentence=='':
            st.write('')
        else:
            st.write('Traduction :',translate(sentence))
    
    
    elif selected_model== 'Seq2Seq Beam Search Decoder':
        st.write("""
                 Evaluation globale du modèle choisi
                 * Score performance : 87%
                 * Score BLEU : 78%
                 * Score ROUGE : 80%
                 """)
        k = st.slider('Paramètre K', min_value=1, max_value=10)
        class Encoder_beam(tf.keras.Model):
         
            def __init__(self, vocab_size, embedding_dim_beam, enc_units, batch_sz):
                # vocab_size: taille du vocabulaire dans la langue de départ
                # embedding_dim_beam: dimension de la matrice d'embedding 
                # enc_units : dimension de l'état caché (units)
                #batch_sz = batch size
                super(Encoder_beam, self).__init__()
                self.batch_sz = batch_sz
                self.enc_units = enc_units
                
                # sélection de l'embedding, fonction qui permet de vectoriser les mots représentés précédemment par leur index
                self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim_beam)
            
    
                # sélection de la cellule de calcul LSTM
                self.lstm_layer = tf.keras.layers.LSTM(self.enc_units, # nombre d'états cachés - neurones
                                            return_sequences=True,  # retourne une séquence/ bidirectionelle
                                            return_state=True,  # permet de conserver la mémoire de l'encodeur et le rend disponible pour le décodeur
                                            recurrent_initializer='glorot_uniform')# initialisation des poids de la matrice utilisée pour la transformation linéaire des états récurrents par distribution uniforme
    
            def call(self, x, hidden):
                x = self.embedding(x)
                output, h, c = self.lstm_layer(x, initial_state = hidden)
                return output, h, c
    
    
            def initialize_hidden_state(self):
                return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]
    
    
    
    
    
    
        class Decoder_beam(tf.keras.Model):
            
            def __init__(self, vocab_size, embedding_dim_beam, dec_units, batch_sz, attention_type='luong'):
                super(Decoder_beam, self).__init__()
                self.batch_sz = batch_sz
                self.dec_units = dec_units
                self.attention_type = attention_type
    
                # Embedding Layer
                self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim_beam)
    
                #Final Dense layer on which softmax will be applied
                self.fc = tf.keras.layers.Dense(vocab_size)
    
                # Define the fundamental cell for decoder recurrent structure
                self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)
    
    
    
                # Sampler
                self.sampler = tfa.seq2seq.sampler.TrainingSampler()
    
                # Create attention mechanism with memory = None
                self.attention_mechanism = self.build_attention_mechanism(self.dec_units, 
                                                                        None, self.batch_sz*[max_length_inp], self.attention_type)
    
                # Wrap attention mechanism with the fundamental rnn cell of decoder
                self.rnn_cell = self.build_rnn_cell(batch_sz)
    
                # Define the decoder with respect to fundamental rnn cell
                self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)
    
            
            def build_rnn_cell(self, batch_sz):
                rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, 
                                            self.attention_mechanism, attention_layer_size=self.dec_units)
                return rnn_cell
    
            
            def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='bahdanau'):
                # ------------- #
                # typ: Which sort of attention (Bahdanau, Luong)
                # dec_units: final dimension of attention outputs 
                # memory: encoder hidden states of shape (batch_size, max_length_inp, enc_units)
                # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_inp (for masking purpose)
    
                if(attention_type=='bahdanau'):
                    return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
                else:
                    return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
    
            
            def build_initial_state(self, batch_sz, encoder_state, Dtype):
                decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
                decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
                return decoder_initial_state
    
    
            def call(self, inputs, initial_state):
                x = self.embedding(inputs)
                outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz*[max_length_targ-1])
                return outputs
    
    
    
    
    
    
        @st.cache(allow_output_mutation=True)
        def model_creator_beam( vocab_size_inp, embedding_dim_beam, units, BATCH_SIZE, vocab_size_targ ):
            encoder_beam = Encoder_beam( vocab_size_inp, embedding_dim_beam, units, BATCH_SIZE)
            decoder_beam = Decoder_beam(vocab_size_targ, embedding_dim_beam, units, BATCH_SIZE, 'luong')
            return encoder_beam,  decoder_beam
    
        encoder_beam, decoder_beam = model_creator_beam( vocab_size_inp, embedding_dim_beam, units, BATCH_SIZE, vocab_size_targ )
    
        sample_hidden = encoder_beam.initialize_hidden_state()
        sample_output, sample_h, sample_c = encoder_beam(example_input_batch, sample_hidden)
    
    
        #@st.cache(allow_output_mutation=True)
        def sample_decoder_output_func(decoder_beam, BATCH_SIZE, max_length_targ, sample_output, sample_h, sample_c ):
            sample_x = tf.random.uniform((BATCH_SIZE, max_length_targ))
            decoder_beam.attention_mechanism.setup_memory(sample_output)
            initial_state = decoder_beam.build_initial_state(BATCH_SIZE, [sample_h, sample_c], tf.float32)
            sample_decoder_outputs = decoder_beam(sample_x, initial_state)
            return sample_decoder_outputs
    
        sample_decoder_outputs = sample_decoder_output_func(decoder_beam, BATCH_SIZE, max_length_targ, sample_output, sample_h, sample_c )
    
        optimizer = tf.keras.optimizers.Adam() 
    
        checkpoint_dir_beam = './training_checkpoints_beam'
    
        #@st.cache(allow_output_mutation=True)
        def restore_model(checkpoint_dir_beam, optimizer, encoder, decoder):
            checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                        encoder=encoder,
                                        decoder=decoder)
            return checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir_beam))
    
    
        restore_model(checkpoint_dir_beam, optimizer, encoder_beam, decoder_beam)
        def beam_evaluate_sentence(sentence, beam_width=k):
    
            sentence = sentence.replace('<start>','')
            sentence = sentence.replace('<end>', '')
            sentence = clean_sentence(sentence)
            
    
            inputs = [input_tokenizer.word_index[i] for i in sentence.split(' ')]
            inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                                maxlen=max_length_inp,
                                                                padding='post')
            inputs = tf.convert_to_tensor(inputs)
            inference_batch_size = inputs.shape[0]
            result = ''
    
            enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size,units))]
            enc_out, enc_h, enc_c = encoder_beam(inputs, enc_start_state)
    
            dec_h = enc_h
            dec_c = enc_c
    
            start_tokens = tf.fill([inference_batch_size], target_tokenizer.word_index['<start>'])
            end_token = target_tokenizer.word_index['<end>']
    
            enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
            decoder_beam.attention_mechanism.setup_memory(enc_out)
            #print("beam_with * [batch_size, max_length_inp, rnn_units] :  3 * [1, 16, 1024]] :", enc_out.shape)
    
            # set decoder_inital_state which is an AttentionWrapperState considering beam_width
            hidden_state = tfa.seq2seq.tile_batch([enc_h, enc_c], multiplier=beam_width)
            decoder_initial_state = decoder_beam.rnn_cell.get_initial_state(batch_size=beam_width*inference_batch_size, dtype=tf.float32)
            decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)
    
            # Instantiate BeamSearchDecoder
            decoder_instance = tfa.seq2seq.BeamSearchDecoder(decoder_beam.rnn_cell,beam_width=beam_width, output_layer=decoder_beam.fc)
            decoder_embedding_matrix = decoder_beam.embedding.variables[0]
    
            # The BeamSearchDecoder object's call() function takes care of everything.
            outputs, final_state, sequence_lengths = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
            # outputs is tfa.seq2seq.FinalBeamSearchDecoderOutput object. 
            # The final beam predictions are stored in outputs.predicted_id
            # outputs.beam_search_decoder_output is a tfa.seq2seq.BeamSearchDecoderOutput object which keep tracks of beam_scores and parent_ids while performing a beam decoding step
            # final_state = tfa.seq2seq.BeamSearchDecoderState object.
            # Sequence Length = [inference_batch_size, beam_width] details the maximum length of the beams that are generated
    
    
            # outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width)
            # outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width)
            # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
            final_outputs = tf.transpose(outputs.predicted_ids, perm=(0,2,1))
            beam_scores = tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0,2,1))
    
            return final_outputs.numpy(), beam_scores.numpy()
    
        def beam_translate(sentence):
            result, beam_scores = beam_evaluate_sentence(sentence)
            #print(result.shape, beam_scores.shape)
            for beam, score in zip(result, beam_scores):
                #print(beam.shape, score.shape)
                output = target_tokenizer.sequences_to_texts(beam)
                output = [a[:a.index('<end>')] for a in output]
                beam_score = [a.sum() for a in score]
                #print('Input: %s' % (sentence))
                #for i in range(len(output)):
                    #print('{} Predicted translation: {}  {}'.format(i+1, output[i], beam_score[i]))
            return output
        sentence = st.text_input('Veuillez entrer une phrase en anglais')
        if sentence=='':
            st.write('')
        else:
            st.write('Traduction :',beam_translate(sentence))
    
    else:
        import Transformeur
        st.write("""
                 Evaluation globale du modèle choisi
                 * Score performance : 76%
                 * Score BLEU : 60%
                 * Score ROUGE : 64%
                 """)
        # Appliquer la fonction clean_sentence sur la colonne english
    
        tokenizer_fr, tokenizer_en,tokenized_fr, tokenized_en = tokenize_data_transf(corpus_fr, corpus_en)
    
    
        new_fr = []
        new_en = []
    
        for sentence_fr, sentence_en in zip(tokenized_fr, tokenized_en):
            new_sentence_fr = [32000] + sentence_fr + [32001]
            new_sentence_en = [32000] + sentence_en + [32001]
            new_fr.append(new_sentence_fr)
            new_en.append(new_sentence_en)
    
        tokenized_fr = new_fr
        tokenized_en = new_en
        lengths_fr = []
        lengths_en = []
    
        for sentence_fr, sentence_en in zip(tokenized_fr, tokenized_en):
            lengths_fr.append(len(sentence_fr))
            lengths_en.append(len(sentence_en))
    
        MAX_LENGTH = 60
    
        new_fr = []
        new_en = []
    
        for sentence_fr, sentence_en in zip(tokenized_fr, tokenized_en):
            new_fr.append(sentence_fr + [0] * (MAX_LENGTH - len(sentence_fr)))
            new_en.append(sentence_en + [0] * (MAX_LENGTH - len(sentence_en)))
            
        tokenized_fr = new_fr
        tokenized_en = new_en
    
        # Vérifications
    
        # longueur des phrases 
        for sentence_fr, sentence_en in zip(tokenized_fr, tokenized_en):
            if len(sentence_fr) != MAX_LENGTH or len(sentence_en) != MAX_LENGTH:
                print("Les phrases n'ont pas la bonne longueur.")
                break
    
        # vérification du dernier nombre 
        sample_sentence = tokenized_fr[0]
        if len(sample_sentence) == MAX_LENGTH and sample_sentence[-1] != 0:
            print("Le token de padding utilisé n'est pas le bon.")
    
        X_train_transf, X_test_transf, y_train_transf, y_test_transf = train_test_split(tokenized_en, tokenized_fr,test_size=0.2, random_state=1)
    
        def create_look_ahead_mask(size):
            mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
            return mask  # (size, size)
        def create_padding_mask(seq):
            seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
            # tf.cast fait passer de x (tenseur) à dtype
            # tf.math.equal(seq, 0): permet de vérifier si la sequence seq est égale à 0
    
            # ajouter de deux dimensions avant et après pour obtenir le bon nombre de dimensions (batch_size, 1, 1, seq_len)
            return seq[:, tf.newaxis, tf.newaxis, :]
        
        def create_masks(inp, tar):
            # Encodage du masque de padding de l'entrée du décodeur 
            enc_padding_mask = create_padding_mask(inp)
    
            # Masque de padding utilisé dans le bloc décodeur - second attention 
            # permet depour cacher une partie des sorties de l'encodeur 
            dec_padding_mask = create_padding_mask(inp)
    
            # Masque de padding utilisé dans le bloc de décodeur - première attention  
            # permet de cacher la fin des phrases traduites 
            look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
            dec_target_padding_mask = create_padding_mask(tar)
            combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
            return enc_padding_mask, combined_mask, dec_padding_mask
        
    
        num_layers = 1
        d_model = 128
        dff = 512
        num_heads = 8
    
        input_vocab_size = tokenizer_en.num_words + 2
        target_vocab_size = tokenizer_fr.num_words + 2
        dropout_rate = 0.1
    
    
        transformer = Transformeur.Transformer(num_layers, d_model, num_heads, dff,
                                input_vocab_size, target_vocab_size, 
                                pe_input=input_vocab_size,
                                pe_target=target_vocab_size,
                                rate=dropout_rate)
        
        class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, d_model, warmup_steps=4000):
                super(CustomSchedule, self).__init__()
    
                self.d_model = d_model
                self.d_model = tf.cast(self.d_model, tf.float32)
    
                self.warmup_steps = warmup_steps
    
            def __call__(self, step):
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)
    
                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        
        learning_rate = CustomSchedule(d_model)
    
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                         epsilon=1e-9)
        
        checkpoint_dir = './training_transformer_checkpoints'
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model = transformer)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        def evaluate(inp_sentence):
            start_token = [tokenizer_en.num_words]
            end_token = [tokenizer_en.num_words + 1]
            
            
            # Encodage de la phrase en anglais, en rajoutant les tokens de début 
            # et de fin du vocabulaire anglais
            encoder_input = start_token + tokenizer_en.texts_to_sequences([inp_sentence])[0] + end_token
            
            # Conversion de la liste en tensor de shape (1, seq_len + 2)
            encoder_input = tf.expand_dims(encoder_input, 0)
    
            # initialisation de la traduction partielle avec le token 
            # de début du vocabulaire français
            decoder_input = [tokenizer_fr.num_words]
            
            # Conversion de la liste en tensor de shape (1, 1) 
            # pour pouvoir appliquer le transformer dessus
            output = tf.expand_dims(decoder_input, 0)
    
            for i in range(MAX_LENGTH):
                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                    encoder_input, output)
    
                # predictions est la nouvelle traduction partielle
                predictions = transformer(encoder_input, 
                                                output,
                                                False,
                                                enc_padding_mask,
                                                combined_mask,
                                                dec_padding_mask)
    
                # on récupère les probabilités pour le dernier mot prédit
                # puis l'id du mot le plus probable
                next_word = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
    
                next_word_id = tf.cast(tf.argmax(next_word, axis=-1), tf.int32)
    
                # si c'est le token de fin de phrase, on arrete l'auto-complétion
                if next_word_id == tokenizer_fr.num_words+1:
                    
                    output = tf.squeeze(output, axis=0) # passage à 1 dimension
                    return output
    
                # sinon, on rajoute le mot à la traduction partielle
                output = tf.concat([output, next_word_id], axis=-1)
            
            # si on a atteint la longueur maximale, on retourne 
            # la traduction partielle actuelle
            output = tf.squeeze(output, axis=0) # passage à 1 dimension
    
            return output
        
                                
        def translate_test(inp_sentence):
            result = evaluate(inp_sentence).numpy()
            
            predicted_sentence = tokenizer_fr.sequences_to_texts([[i for i in result
                                    if i < tokenizer_fr.num_words]])[0]
    
            return predicted_sentence
        sentence = st.text_input('Veuillez entrer une phrase en anglais')
        if sentence=='':
            st.write('')
        else:
            st.write('Traduction :',translate_test(sentence))
    
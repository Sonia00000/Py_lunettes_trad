#!/usr/bin/env python
# coding: utf-8

# # Couches d'attention 

# ### Définition de la fonction d'attention par produit scalaire 

# In[1]:


import tensorflow as tf
import numpy as np

def scaled_dot_product_attention(q, k, v, mask):
    """Permet de calculer les poids d'attention via une fonction d'attention
    
    Arguments:
    q: requête de format == (..., seq_len_q, depth)
    v: valeurs possibles de format == (..., seq_len_v, depth_v)
    k: clés de ces valeurs de format == (..., seq_len_k, depth)
    mask: Float tensor

    Le produit scalaire (normalisé) peut être utilisé comme mesure de similarité entre deux vecteurs de la manière suivante:
    - Si le produit scalaire entre la requête et une clé est positif: 
        Les deux vecteurs sont similaires.
    - Si le produit scalaire entre la requête et une clé est négatif: 
        Les deux vecteurs sont très différents.
        
    Retourne:
    output, attention_weights
    """
    
    # Calcul de la dimension des clés 
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    
    # Calcul matriciel de q et de la transposée de k
    dot_products = tf.matmul(q, k, transpose_b=True)
    
    # Division par la racine carrée de dk -> normalisation pour contrer le problème de vanishing gradient
    scaled_dot_products = dot_products / tf.math.sqrt(dk)

    # Application du masque afin d'obtenir des séquences de base de données de même longueur
    if mask is not None:
        # Multiplicaiton du masque par une très petite valeur (considérée comme moins l'infini)
        mask = mask * (-1e10)
        
        # Somme du produit scalaire "scaled_dot_products" et du masque 
        scaled_dot_products = scaled_dot_products + mask
    
    # Application de la fonction "softmax" permettant d'obtenir les poids d'attention
    attention_weights = tf.nn.softmax(scaled_dot_products)
    
    # Calcul du vecteur d'attention par calcul matriciel entre
    # les poids d'attention et les valeurs possibles
    attention_vector = tf.matmul(attention_weights, v)

    return attention_vector, attention_weights


# ### Définition d'une couche d'attention multi-têtes

# In[2]:


class MultiHeadAttention(tf.keras.layers.Layer):
    # création d'une classe multi-tête qui hérite de la classe tf.keras.layers.Layer
    # classe devra être entrainable 
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads # nombre de têtes 
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Scinde la dernière dimension de x en (num_heads, depth).
        Transforme le résultat de telle sorte à ce qu'il soit de la forme
        # (batch_size, num_heads, seq_len, depth)        
        """
        # Restructuration de manière à pouvoir scinder 
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        output = tf.transpose(x, perm=[0, 2, 1, 3])   
        
        return output

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # Application des matrices denses à Q, K et V
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Découpage des vecteurs selon la fonction définie ci-avant 
        q = self.split_heads(q, batch_size) 
        k = self.split_heads(k, batch_size) 
        v = self.split_heads(v, batch_size) 

        # Application du masque précédent 
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # Concaténation des vecteurs d'attention en remodelant le tensor 
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))
        
        # Application de la matrice dense à la sortie
        output = self.dense(concat_attention)

        return output, attention_weights


# ## Encodeur 

# ### Couche d'encodage 

# In[3]:


# Définition de la couche d'encodage comprenant: 

    # - Des couches de Dropout et Layer Normalisation pour régulariser le modèle.
    # - Des Skip Connections pour éviter le problème de vanishing gradient à cause des petits gradients de la fonction softmaxsoftmax.
    # - Un réseau dense (c'est-à-dire un MLP ou Feed Forward Neural Network comme mentionné dans le papier) est ajouté après l'attention multi-têtes.


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = tf.keras.models.Sequential()
        self.ffn.add(tf.keras.layers.Dense(dff, activation='relu'))
        self.ffn.add(tf.keras.layers.Dense(d_model))

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # Step 1 Multihead attentions
        A, _ = self.mha(x, x, x, mask)
        
        # Step 2 Dropout
        A = self.dropout1(A, training=training)
        
        # Step 3 Skip connection 
        x = A + x
        
        # Step 4 Layer normalization 
        x = self.layernorm1(x)
        
        # Step 5 Feed Forward Network 
        O = self.ffn(x)
        
        # Step 6 Dropout
        O = self.dropout2(O)
        
        # Step 7 Skip connection 
        x = O + x
        
        # Step 8 Layer normalization 
        output = self.layernorm2(x)
        
        return output
    
    
# Vérifications

sample_encoder_layer = EncoderLayer(512, 8, 2048)

input_shape = (64, 43, 512)
sample_encoder_layer_output = sample_encoder_layer(
    tf.random.uniform(input_shape), False, None)

print("Input Shape ", input_shape)
print("Output Shape", sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)


# ### Encodage de la position 

# In[4]:


def positional_encoding(position_max, d_model):
    i = np.linspace(0, d_model-1, d_model)
    
    pos_encoding = np.zeros((position_max, d_model))
    
    for pos, row in enumerate(pos_encoding):
        pos_encoding[pos, ::2] = np.sin(pos/(10000 ** (i[::2]/d_model)))
        pos_encoding[pos, 1::2] = np.cos(pos/(10000 ** ((i[1::2]-1)/d_model))) 
    
    return tf.cast(pos_encoding.reshape(1, position_max, d_model), dtype = tf.float32)


# ### Encodeur

# In[5]:


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
        """
        num_layers : Number of Encoding layers in the Encoder.
        
        d_model : Dimensionality of the embedding layer's output.
        
        num_heads : Number of attention heads in the Multi-Head Attention layers.
        
        dff : Number of neurons in the first layer of the Feed Forward Network in the Encoding layers.
        
        input_vocab_size : Size of the vocabulary the embedding layer should train on.
        
        maximum_position_encoding : Maximum number of tokens in a sequence.
        
        rate : Dropout rate in the Encoding layers.      
        
        """
        super(Encoder, self).__init__()

        self.d_model = d_model
        
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)


        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        
        seq_len = tf.shape(x)[1]
        
        # Generate Embedding
        x = self.embedding(x) 

        # Add Positional Encoding
        x += self.pos_encoding[:, :seq_len, :]

        # Apply Dropout
        x = self.dropout(x, training=training)

        # Apply Encoding Layers
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x 


# ## Décodeur 

# In[6]:


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """        
        d_model : Dimensionality of the embedding layer's output.
        
        num_heads : Number of attention heads in the Multi-Head Attention layers.
        
        dff : Number of neurons in the first layer of the Feed Forward Network in the Decoding layers.
        
        rate : Dropout rate in the Decoding layers.      
        """
        
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = tf.keras.models.Sequential()
        self.ffn.add(tf.keras.layers.Dense(dff, activation='relu'))
        self.ffn.add(tf.keras.layers.Dense(d_model))

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
    def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
        """
        x : Input Tensor of the embeddings of the partially translated sentence.
        
        enc_output : Output Tensor of the Decoder.
        
        training : Whether the model is training or not so that Dropout is only applied during training.
        
        look_ahead_mask : Tensor for masking tokens that have not been translated yet.
        
        padding_mask : Tensor masking padding tokens of the Decoder's input.      
        """

        A1, _  = self.mha1(x, x, x, look_ahead_mask)
        A1     = self.dropout1(A1, training=training)
        x      = A1 + x
        x      = self.layernorm1(x)

        A2, _  = self.mha2(enc_output, enc_output, x, padding_mask)
        A2     = self.dropout2(A2, training=training)
        x      = A2 + x
        x      = self.layernorm2(x) 

        O      = self.ffn(x) 
        O      = self.dropout3(O, training=training)
        x      = O + x
        Output = self.layernorm3(x) 

        return Output


# In[7]:


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
        """
        num_layers : Number of Decoding layers in the Decoder.
        
        d_model : Dimensionality of the embedding layer's output.
        
        num_heads : Number of attention heads in the Multi-Head Attention layers.
        
        dff : Number of neurons in the first layer of the Feed Forward Network in the Decoding layers.
        
        target_vocab_size : Size of the vocabulary the embedding layer should train on.
        
        maximum_position_encoding : Maximum number of tokens in a sequence.
        
        rate : Dropout rate in the Decoding layers.      
        """
        
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
        """
        x : Input Tensor of the partially translated sentence.
        
        enc_output : Output Tensor of the Encoder.
        
        training : Whether the model is training or not so that Dropout is only applied during training.
        
        look_ahead_mask : Tensor for masking tokens that have not been translated yet.
        
        padding_mask : Tensor masking padding tokens of the Decoder's input.      
        """
        seq_len = tf.shape(x)[1]
        
        # Generate Embedding
        x = self.embedding(x) 

        # Add Positional Encoding
        x += self.pos_encoding[:, :seq_len, :]

        # Apply Dropout
        x = self.dropout(x, training=training)

        # Apply Decoding Layers
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training,
                                   look_ahead_mask, padding_mask)

        return x 


# ## Transformeur 

# In[8]:


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=0.1):
        """
        num_layers : Number of Encoding/Decoding layers in the Encoder/Decoder.
        d_model    : Dimensionality of the embedding layers' output.
        num_heads  : Number of attention heads in the Multi-Head Attention layers.
        dff        : Number of neurons in the first layer of the Feed Forward Network in the Encoding/Decoding layers.
        rate       : Dropout rate in the Encoding/Decoding layers.    
        
        input_vocab_size  : Size of the vocabulary the Encoder's embedding layer should train on.
        target_vocab_size : Size of the vocabulary the Encoder's embedding layer should train on.
        
        
        pe_input  : Maximum number of tokens in the input sequence of the Encoder.
        pe_target : Maximum number of tokens in the input sequence of the Decoder.
        """       
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size, activation = 'softmax')
    
    def call(self, encoder_input, decoder_input ,training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
        """
        encoder_input : Input Tensor of the original sentence to be given to the Encoder.
        decoder_input : Input Tensor of the partially translated sentence to be given to the Decoder.
        
        training : Whether the model is training or not so that Dropout is only applied during training.
        
        enc_padding_mask : Tensor masking padding tokens of the Encoder's input.
        look_ahead_mask  : Tensor for masking tokens that have not been translated yet.
        dec_padding_mask : Tensor masking padding tokens of the Decoder's input.      
        """

        # Get Encoder Output
        enc_output = self.encoder(encoder_input, training, enc_padding_mask)

        # Get Decoder Output
        dec_output = self.decoder(decoder_input, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        # Perform Classification
        final_output = self.final_layer(dec_output)

        return final_output


# In[9]:


num_layers = 1
d_model    = 128
dff        = 512
num_heads  = 8

input_vocab_size  = 32000
target_vocab_size = 32000
dropout_rate      = 0.1

sample_transformer = Transformer(num_layers, d_model, num_heads, dff,
                                 input_vocab_size, target_vocab_size, 
                                 pe_input = 62, 
                                 pe_target= 62,
                                 rate=dropout_rate)

sample_encoder_input = tf.random.uniform((64, 62))
sample_decoder_input = tf.random.uniform((64, 62))

sample_transformer_output = sample_transformer(sample_encoder_input,
                                               sample_decoder_input,
                                               False,
                                               None, 
                                               None, 
                                               None)

print("Encoder Input Shape:     ", sample_encoder_input.shape)
print("Decoder Input Shape:     ", sample_decoder_input.shape)
print("Transformer Output Shape:", sample_transformer_output.shape)


# In[ ]:





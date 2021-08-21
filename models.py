import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Concatenate, Dropout, GRU
from tensorflow.keras.initializers import Constant

# encoder class
class Encoder(Model):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, emb_matrix, num_layers=1, drop_prob=0):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.embedding = Embedding(vocab_size, embedding_dim, embeddings_initializer=Constant(emb_matrix))
        # multiple number of layers
        self.gru_layers = []
        for i in range(self.num_layers):
            self.gru_layers.append(GRU(self.enc_units, return_state=True, return_sequences=True,
                                       recurrent_initializer='glorot_uniform',
                                       dropout=self.drop_prob))

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru_layers[0](x, initial_state=hidden)
        for i in range(1, self.num_layers - 1):
            output, state = self.gru_layers[i](output)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


# creating Attention model
class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        # query is the hidden state to be passes to the decoder shape = (batchsize, hidden_size)
        # values is the output from the gru , shape = (batchsize, max_len, hidden_size)
        # query with time axis is done to broadcast along time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape = (batchsize, maxlen, 1)
        score = self.V( tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        # self.V converts the shape of scores from (batchsize, maxlen, units) to whatever we have

        # attention weights shape = (batchsize, maxlen, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context vector shape after sum == (batchsize, hiddensize)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# creating decoder class
class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, emb_matrix, num_layers=1, drop_prob=0.1):
        super(Decoder,self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.embedding = Embedding(vocab_size, embedding_dim, embeddings_initializer=Constant(emb_matrix))
        # multiple number of layers
        self.gru_layers = []
        for i in range(self.num_layers):
            self.gru_layers.append(GRU(self.dec_units, return_state = True, return_sequences=True,
                                    recurrent_initializer='glorot_uniform',
                                    dropout = (0 if self.num_layers==1 else self.drop_prob)))
        self.fc = Dense(vocab_size)

        # attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):

        # enc output shape = (batch_sz, maxlen, hiddensize)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after embedding ( batchsize, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation = (batchsize, 1, embedding_dim + hiddensize)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to gru
        output, state = self.gru_layers[0](x, initial_state = hidden)
        for i in range(1, self.num_layers-1):
            output, state = self.gru_layers[i](output)
        # output shape = (batchsize*1, hiddensize)
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape = (batchsize, vocab)
        x = self.fc(output)

        return x, state, attention_weights



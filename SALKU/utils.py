import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

class CosLayer(Layer):
    def __init__(self, **kwargs):
        super(CosLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x, y = inputs
        x = tf.nn.l2_normalize(x, -1)
        y = tf.nn.l2_normalize(y, -1)
        value = tf.reduce_sum(x * y, -1, keepdims=True)
        return value

    def compute_output_shape(self, input_shape):
        return (None, 1)



def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def positional_encoding(pos, d_model):
    def get_angles(position, i):
        # shape=[position_num, d_model]
        return position / np.power(10000., 2. * (i // 2.) / np.float32(d_model))

    angle_rates = get_angles(np.arange(pos)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :])
    # 2i:sin，2i+1:cos
    angle_rates[:, 0::2] = np.sin(angle_rates[:, 0::2])
    angle_rates[:, 1::2] = np.cos(angle_rates[:, 1::2])
    pos_encoding = tf.cast(angle_rates[np.newaxis, ...], dtype=tf.float32)
    return pos_encoding


def scaled_dot_product_attention(q, k, v, mask):
    '''attention(Q, K, V) = softmax(Q * K^T / sqrt(dk)) * V'''
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(q)[-1], tf.float32)
    scaled_attention = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention += mask * -1e9
    attention_weights = tf.nn.softmax(scaled_attention, axis=-1)  # shape=[batch_size, seq_len_q, seq_len_k]
    outputs = tf.matmul(attention_weights, v)  # shape=[batch_size, seq_len_q, depth]
    return outputs, attention_weights


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # x shape=[batch_size, seq_len, d_model]
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask, **kwargs):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # shape=[batch_size, seq_len_q, d_model]
        k = self.wq(k)
        v = self.wq(v)
        q = self.split_heads(q, batch_size)  # shape=[batch_size, num_heads, seq_len_q, depth]
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention shape=[batch_size, num_heads, seq_len_q, depth]
        # attention_weights shape=[batch_size, num_heads, seq_len_q, seq_len_k]
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # shape=[batch_size, seq_len_q, num_heads, depth]
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # shape=[batch_size, seq_len_q, d_model]
        output = self.dense(concat_attention)
        return output, attention_weights

class PositionalWiseFeedForward(Layer):
    def __init__(self, d_model, dff, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        self.w_1 = tf.keras.layers.Dense(dff, activation=tf.nn.relu)
        self.w_2 = tf.keras.layers.Dense(d_model)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, **kwargs):
        x = self.layernorm(inputs)
        residual = x
        x = self.w_1(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = residual + x
        return x

class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionalWiseFeedForward(d_model, dff, dropout_rate)

    def call(self, inputs, training, mask):
        # multi head attention (encoder时Q = K = V)
        x1 = inputs[0]
        x2 = inputs[1]
        att_output, _ = self.mha(x1, x2, x2, mask)
        output = self.ffn(att_output)
        return output

class Encoder(Layer):
    def __init__(self, d_model, num_layers, num_heads, dff,
                 input_vocab_size, maximum_position_encoding, weights, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.emb = tf.keras.layers.Embedding(input_vocab_size, d_model, weights=weights)  # shape=[batch_size, seq_len, d_model]
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)  # shape=[1, max_seq_len, d_model]
        self.encoder_layer = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                              for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense = tf.keras.layers.Dense(d_model)

    def call(self, x, training):
        # inputs shape=[batch_size, seq_len]
        seq_len = tf.shape(x)[1]
        word_embedding = self.emb(x)  # shape=[batch_size, seq_len, d_model]
        word_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        emb = word_embedding + self.pos_encoding[:, :seq_len, :]
        mask = create_padding_mask(x)
        x = self.dropout(emb, training=training)
        for i in range(self.num_layers):
            x = self.encoder_layer[i]([x, x], training, mask)
        output = self.dense(x)
        output = self.layernorm(output + x)
        return output   # shape=[batch_size, seq_len, d_model]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_model)


class JoinEncoder(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(JoinEncoder, self).__init__()
        self.d_model = d_model
        self.layer = EncoderLayer(d_model, num_heads, dff, dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.linear = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        repr_output = self.layer(inputs, mask=None)
        repr_output = self.linear(repr_output)
        repr_output = self.layer_norm(repr_output)
        return repr_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_model)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from base_model import base_model
from utils import Encoder, JoinEncoder
from sklearn.metrics import classification_report

class SALKU_model(base_model):
    def __init__(self, pad_len=100, word_dim=200, epochs=100, batch_size=128):
        self.pad_len = pad_len
        self.accuracy = tf.keras.metrics.Accuracy(name='acc')
        super(SALKU_model, self).__init__(epochs=epochs, batch_size=batch_size, word_dim=word_dim)

    def build_model(self, input_shape):
        input_q1 = Input(shape=input_shape, name='input_q1')
        input_q2 = Input(shape=input_shape, name='input_q2')


        self_attention_encoder = Encoder(d_model=self.word_dim,
                                         num_layers=1,
                                         num_heads=2,
                                         dff=2*self.word_dim,
                                         input_vocab_size=len(self.index2word),
                                         maximum_position_encoding=self.word_dim,
                                         weights = [self.index2vec]
                                         )
        self_attention_q1 = self_attention_encoder(input_q1)
        self_attention_q2 = self_attention_encoder(input_q2)
        join_encoder = JoinEncoder(d_model=self.word_dim,
                                   num_heads=2,
                                   dff=2*self.word_dim)
        join_output_q1 = join_encoder([self_attention_q1, self_attention_q2])
        join_output_q2 = join_encoder([self_attention_q2, self_attention_q1])

        output_q1 = tf.reduce_mean(join_output_q1, axis=-2)
        output_q2 = tf.reduce_mean(join_output_q2, axis=-2)

        concat_layer = Concatenate(axis=-1, name='concat')
        concat_out = concat_layer([output_q1, output_q2])
        hidden_dense = Dense(1024, activation='relu', name='hidden')
        hidden_out = hidden_dense(concat_out)
        dense_layer = Dense(4,activation='softmax', name='classical')
        classical_out = dense_layer(hidden_out)
        model = Model([input_q1,input_q2], classical_out)
        model.summary()
        return model

    def classify(self, inputs):
        x = np.argmax(inputs, axis=-1)
        return x

    def class_to_score(self, input):
        return input

    def my_Accuracy(self, y_true, y_pred):
        pred = tf.argmax(y_pred, axis=-1)
        return self.accuracy(y_true, pred)

    def my_classification_report(self, y_true, y_pred):
        pred = self.classify(y_pred)
        #np.save('SALKU_pred.npy', pred)
        #np.save('SALKU_true.npy', y_true)
        print(classification_report(y_true, pred, labels=[3, 2, 1, 0],
                                    target_names=["isolated", "indirect", "direct", "duplicate"],
                                    digits=3))

    def SALKU_train(self):
        self._train(pad_len=self.pad_len,
                    path='SALKU_reimplementation/SALKU_model/SALKU_model.h5',
                    optimizer=tf.keras.optimizers.Adam(0.0001),
                    loss=tf.losses.sparse_categorical_crossentropy,
                    metrics=[self.my_Accuracy])

    def SALKU_test(self):
        true, pred = self._test(pad_len=self.pad_len,
                                path='SALKU_reimplementation/SALKU_model/SALKU_model.h5',
                                optimizer='adam',
                                loss=tf.losses.sparse_categorical_crossentropy,
                                metrics=[self.my_Accuracy])
        self.my_classification_report(true, pred)
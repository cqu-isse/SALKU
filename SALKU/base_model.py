import os
import pandas as pd
import numpy as np
import tensorflow as tf
from word2vec import word2vec
from data_preprocess import getData

class EarlyStoppingByLossVal(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.001, verbose=0):
        super(EarlyStoppingByLossVal, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            raise Exception("Early stopping requires %s available!" % self.monitor)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


class base_model:
    def __init__(self, epochs, batch_size, word_dim):
        self.epochs = epochs
        self.batch_size = batch_size
        self.word_dim = word_dim
        self.index2word, self.word2index, self.index2vec = self.getWV()

    def getWV(self):
        return word2vec.getWV(self.word_dim)

    def class_to_score(self, input):
        pass

    def divide_train_valid(self, data):
        pass

    def my_classification_report(self, y_true, y_pred, output_dict):
        pass

    def build_model(self, input_shape):
        pass

    def getVector(self, type_str, pad_len):
        assert type_str in ['train', 'test']

        def word_to_index(sentence):
            word_list = str(sentence).strip().split()
            index_list = []
            for word in word_list:
                index_list.append(self.word2index.get(word, 0))
            return ' '.join(str(i) for i in
                            tf.keras.preprocessing.sequence.pad_sequences([index_list], maxlen=pad_len,
                                                                          padding='post', truncating='post', value=0)[
                                0])


        if not os.path.exists('data/w2i_' + str(pad_len) + '_' + type_str + '_data.csv'):
            if not os.path.exists('data/processed_' + type_str + '_data.csv'):
                getData(type_str)
            processed_data = pd.read_csv('data/processed_' + type_str + '_data.csv')
            processed_data['sentence'] = processed_data['sentence'].apply(word_to_index)
            processed_data.to_csv('data/w2i_' + str(pad_len) + '_' +  type_str + '_data.csv', index=False)
            del processed_data

        data = pd.read_csv('data/index_' + type_str + '_data.csv', index_col='id')
        w2i_data = pd.read_csv('data/w2i_' + str(pad_len) + '_' + type_str + '_data.csv', index_col='id')

        def id_to_index(id):
            my_data = w2i_data
            temp_list = str(my_data.loc[id]['sentence']).split()
            return [int(i) for i in temp_list]

        data['q1_id'] = data['q1_id'].apply(id_to_index)
        data['q2_id'] = data['q2_id'].apply(id_to_index)
        data['class'] = data['class'].apply(self.class_to_score)

        data_id = data.index
        data_q1 = np.array(data['q1_id'].values.tolist())
        data_q2 = np.array(data['q2_id'].values.tolist())
        data_class = data['class'].values

        return data_id, data_q1, data_q2, data_class



    def _train(self, pad_len, path, optimizer, loss, metrics, callbacks=None):
        if callbacks is None:
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2,restore_best_weights=True)]
        train_data_id, train_data_q1, train_data_q2, train_data_class = self.getVector('train', pad_len)

        sim_model = self.build_model(train_data_q1.shape[1:])
        sim_model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics)
        sim_model.fit(x=[train_data_q1, train_data_q2], y=train_data_class, validation_split=0.05,
                      epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks)
        sim_model.save_weights(path)

    def _test(self, pad_len, path, optimizer, loss, metrics):
        test_data_id, test_data_q1, test_data_q2, test_data_class = self.getVector('test', pad_len)
        sim_model = self.build_model(test_data_q1.shape[1:])
        if not os.path.exists(path):
            self._train(path, optimizer, loss, metrics, pad_len)
        sim_model.load_weights(path)
        pred = sim_model.predict([test_data_q1, test_data_q2], batch_size=self.batch_size)
        pred = np.squeeze(pred)
        return test_data_class, pred



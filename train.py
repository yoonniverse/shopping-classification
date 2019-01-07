import pickle

import numpy as np
np.random.seed(0)
import pandas as pd

import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K

n_chunk = 108

n_features = 9000
batch_size = int(2**12)
n_epochs = 15

dropout_prob = 0.3


class TrainGenerator(Sequence):
        
    def __init__(self, batch_size):        
        self.n_list = []
        for i in range(0, n_chunk):
            with open('len_chunks/len_chunks-{}.pkl'.format(i), 'rb') as f:
                temp = pickle.load(f)
                self.n_list.append(temp[0])
        self.x_files = ['data/x_train-{}.h5'.format(i) for i in range(0, n_chunk)]
        self.y_files = ['data/y_train-{}.h5'.format(i) for i in range(0, n_chunk)]
        self.batch_size = batch_size
        self.num_csv = 0
        self.my_idx = 0
        
    def __len__(self):
        return int(sum([np.ceil(n/float(self.batch_size)) for n in self.n_list]))
    
    def __getitem__(self, idx):
        if self.my_idx * self.batch_size >= self.n_list[self.num_csv]:
            self.num_csv += 1
            self.my_idx = 0
        batch_x = pd.read_hdf(self.x_files[self.num_csv],
                              start=self.my_idx*self.batch_size, 
                              stop=(self.my_idx+1)*self.batch_size).iloc[:, :n_features].astype(np.float32)
        batch_y = pd.read_hdf(self.y_files[self.num_csv],
                              start=self.my_idx*self.batch_size,
                              stop=(self.my_idx+1)*self.batch_size).iloc[:, :n_features].astype(np.float32)
        self.my_idx += 1
        return batch_x.values, batch_y.values
    
    def on_epoch_end(self):
        self.my_idx = 0
        self.num_csv = 0


def metric(y_true, y_pred):
    s_i = K.flatten(tf.where(tf.equal(tf.reduce_sum(y_true[:, 609:3798], axis=1), 1)))
    d_i = K.flatten(tf.where(tf.equal(tf.reduce_sum(y_true[:, 3798:4201], axis=1), 1)))
    b_pred = K.argmax(y_pred[:, :57])
    m_pred = K.argmax(y_pred[:, 57:609])
    s_pred = K.argmax(tf.gather(y_pred, s_i)[:, 609:3798])
    d_pred = K.argmax(tf.gather(y_pred, d_i)[:, 3798:4201])
    b_true = K.argmax(y_true[:, :57])
    m_true = K.argmax(y_true[:, 57:609])
    s_true = K.argmax(tf.gather(y_true, s_i)[:, 609:3798])
    d_true = K.argmax(tf.gather(y_true, d_i)[:, 3798:4201])
    return (K.mean(tf.cast(tf.equal(b_pred, b_true), tf.float32)) +K.mean(tf.cast(tf.equal(m_pred, m_true), tf.float32))*1.2 +K.mean(tf.cast(tf.equal(s_pred, s_true), tf.float32))*1.3 +K.mean(tf.cast(tf.equal(d_pred, d_true), tf.float32))*1.4)/4


inputs = Input(shape=(n_features,))
x = BatchNormalization()(inputs)
x = Dropout(dropout_prob)(x)

x_price = Lambda(lambda a: a[:,0:1])(x)
x_img = Lambda(lambda a: a[:,1:2049])(x)
x_text = Lambda(lambda a: a[:,2049:])(x)

x_img = Dense(2**12, activation='relu')(x_img)
x_img = BatchNormalization()(x_img)
x_img = Dropout(dropout_prob)(x_img)

x_text = Dense(2**13, activation='relu')(x_text)
x_text = BatchNormalization()(x_text)
x_text = Dropout(dropout_prob)(x_text)

x = Concatenate()([x_price, x_img, x_text])

x = Dense(2**13, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(dropout_prob)(x)

x = Dense(2**13, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(dropout_prob)(x)

b = Dense(57, activation='softmax')(x)
m = Dense(552, activation='softmax')(x)
s = Dense(3189, activation='softmax')(x)
d = Dense(403, activation='softmax')(x)
outputs = Concatenate()([b,m,s,d])

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy', metrics=[metric])

model.summary()

x_val = pd.read_hdf('data/x_val').iloc[:, :n_features].astype(np.float32)
y_val = pd.read_hdf('data/y_val').iloc[:, :n_features].astype(np.float32)

history = model.fit_generator(generator=TrainGenerator(batch_size),
                              epochs=n_epochs,
                              validation_data=(x_val, y_val),
                              shuffle=False)

model.save('model/{}-model.h5'.format(n_features))

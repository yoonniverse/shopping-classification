import numpy as np
import pandas as pd

from tqdm import tqdm

import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras.models import load_model

case = 'dev'
n_chunks = 2

n_features = 9000
model = load_model('model/{}-model.h5'.format(n_features))


def predict(i):
    data = pd.read_hdf('data/{}-{}.h5'.format(case, i)).iloc[:, :n_features+1]
    x = data[[col for col in data.columns if col !='pid']].values
    y = model.predict(x)
    b = np.expand_dims(np.argmax(y[:, :57], axis=-1), axis=-1)+1
    m = np.expand_dims(np.argmax(y[:, 57:609], axis=-1), axis=-1)+1
    s = np.expand_dims(np.argmax(y[:, 609:3798], axis=-1), axis=-1)+2
    d = np.expand_dims(np.argmax(y[:, 3798:4201], axis=-1), axis=-1)+2
    return pd.DataFrame(np.concatenate([b, m, s, d], axis=1), index=data['pid'].str.decode('utf-8'))


prediction = pd.DataFrame()
for k in tqdm(range(n_chunks)):
    prediction = prediction.append(predict(k))

print(prediction.head(10))
print(len(prediction))

prediction.to_csv('predictions/{}-{}.tsv'.format(case, n_features), sep='\t', header=False)

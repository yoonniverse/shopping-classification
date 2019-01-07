import h5py
import pickle
import json
import time

import pandas as pd
import numpy as np

from collections import Counter
import multiprocessing as mp

from konlpy.tag import Okt

# train, dev, test 중에 선택
case = 'dev'

# 메모리가 부족하면 늘리기
n_chunk = 2

n_features = 9000


with open('info.pkl', 'rb') as f:
    relevant_words, median_price = pickle.load(f)

if case != 'train':
    relevant_words = list(pd.read_hdf('data/x_train-0.h5', stop=1).columns[2049:n_features])

with open('cate1.json', encoding='utf-8') as f:
    cate = json.load(f)


def parallelize_dataframe(df, func, num_cores=mp.cpu_count()):
    print('*num cores using for multiprocessing:', num_cores)
    pool = mp.Pool(num_cores)
    df = pd.concat(pool.map(func, np.array_split(df, num_cores)))
    pool.close()
    pool.join()
    return df


def okt(df):
    df['model'] = df['model'].apply(Okt().morphs)
    df['product'] = df['product'].apply(Okt().morphs)
    return df


def count_words(df):
    temp = pd.DataFrame(np.zeros((len(df), len(relevant_words))), index=df.index, columns=relevant_words)
    counter_series = df['text'].apply(Counter)
    for i in df.index:
        temp.loc[i].update(pd.Series(counter_series.loc[i]))
    temp = temp.astype('uint8')
    return pd.concat((df, temp), axis=1)


def one_hot_y(df):
    for i in sorted(cate['b'].values()):
        if i != -1:
            df['bcateid_{}'.format(i)] = (df['bcateid']==i).astype('int8')
    for i in sorted(cate['m'].values()):
        if i != -1:
            df['mcateid_{}'.format(i)] = (df['mcateid']==i).astype('int8')
    for i in sorted(cate['s'].values()):
        if i != -1:
            df['scateid_{}'.format(i)] = (df['scateid']==i).astype('int8')
    for i in sorted(cate['d'].values()):
        if i != -1:
            df['dcateid_{}'.format(i)] = (df['dcateid']==i).astype('int8')
    return df


def make_data(n, data, case):

    # Decode to `utf-8` for text columns
    print('decoding with utf-8')
    now = time.time()
    for key in ['brand', 'maker', 'model', 'product']:
        data[key] = data[key].str.decode('utf-8')
    print(int(time.time()-now))

    # Replace `-1` with `median price` in `price` column
    data.loc[data[data['price']==-1].index, 'price'] = median_price
    data.rename(columns={'price': 'original_price'}, inplace=True)

    #  Split texts from columns `brand` and `maker`
    print('splitting texts')
    now = time.time()
    data['brand'] = data['brand'].str.split('[^가-힣a-zA-Z]')
    data['maker'] = data['maker'].str.split('[^가-힣a-zA-Z]')
    print(int(time.time()-now))

    # Morph texts from columns `model` and `product`
    print('morphing texts')
    now = time.time()
    data = parallelize_dataframe(data, okt)
    print(int(time.time()-now))

    # Join columns `brand`, `maker`, `model`, `product`
    print('joining text columns')
    now = time.time()
    data['text'] = [np.concatenate((data['brand'][i], data['maker'][i], data['model'][i], data['product'][i]), axis=0) for i in range(len(data))]
    print(int(time.time()-now))

    # Count relevant words in column `text`
    print('counting words in text')
    now = time.time()
    data = parallelize_dataframe(data, count_words)
    print(int(time.time()-now))

    # Drop processed columns
    data.drop(['brand', 'maker', 'model', 'product', 'text'], axis=1, inplace=True)

    # Astype `np.float32`
    data['original_price'] = data['original_price'].astype(np.float32)
    print(data.info())

    if case != 'train':
        data.drop(['bcateid', 'mcateid', 'scateid', 'dcateid'], axis=1, inplace=True)
        data.to_hdf('data/{}-{}.h5'.format(case, n), key=case, mode='w')
    else:

        # Divide data into `x` and `y`
        print('dividing data into x and y')
        now = time.time()
        labels = ['bcateid', 'mcateid', 'scateid', 'dcateid']
        features = [e for e in data.columns if e not in labels]
        x = data[features]
        y = data[labels]
        print(int(time.time()-now))

        # One-hot encode labels
        print('one-hot encoding labels')
        now = time.time()
        y = parallelize_dataframe(y, one_hot_y)
        y.drop(labels, axis=1, inplace=True)
        print('x head')
        print(x.head())
        print('y head')
        print(y.head())
        print(int(time.time()-now))

        # Split into `train` and `val`
        print('splitting into train and val')
        now = time.time()
        msk = np.random.rand(len(data)) < 0.98
        x_train = x[msk]
        y_train = y[msk]
        x_val = x[~msk]
        y_val = y[~msk]
        print('len_x_train: ', len(x_train))
        print('len_x_val: ', len(x_val))
        print(int(time.time()-now))

        # Store as hdf5
        print('storing x_train')
        now = time.time()
        x_train.to_hdf('data/x_train-{}.h5'.format(n), key='x_train', mode='w')
        print('time spent storing x_train: {}s'.format(int(time.time()-now)))
        print('storing x_val')
        x_val.to_hdf('data/x_val-{}.h5'.format(n), key='x_val', mode='w')
        print('storing y_train')
        y_train.to_hdf('data/y_train-{}.h5'.format(n), key='y_train', mode='w')
        print('storing y_val')
        y_val.to_hdf('data/y_val-{}.h5'.format(n), key='y_val', mode='w')

        # Store `len_x_train` and `len_x_val` as pickle
        print('storing len_chunks')
        with open('len_chunks/len_chunks-{}.pkl'.format(n), 'wb') as handle:
            pickle.dump((len(x_train), len(x_val)), handle)


if __name__ == '__main__':

    counter = 0

    for k in range(0, 9):

        print('{}th RAW DATA PROCESSING'.format(k))

        # Read specified raw data
        f = h5py.File('raw_data/{}.chunk.0{}'.format(case, k+1, 'r'))[case]
        keys = list(f.keys())
        idxs = np.array_split(np.arange(len(f[keys[0]])), n_chunk)

        for idx in idxs:

            data = pd.DataFrame(columns=keys)
            for key in keys:
                if key == 'img_feat':
                    data = pd.concat((data, pd.DataFrame(f[key][list(idx)], columns=['img_feat_{}'.format(i) for i in range(2048)])), axis=1)
                else:
                    data[key] = f[key][list(idx)]

            data.drop('img_feat', axis=1, inplace=True)

            if case == 'train':
                data.drop('pid', axis=1, inplace=True)
            data.drop('updttm', axis=1, inplace=True)

            global_now = time.time()
            make_data(counter, data, case)
            print('TIME SPENT FOR CHUNK {}: {}s'.format(counter, int(time.time()-global_now)))

            counter += 1

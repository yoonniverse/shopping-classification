import h5py
import pickle
import time

import pandas as pd
import numpy as np

from tqdm import tqdm

from konlpy.tag import Okt

def explore(n):

    # Read columns of specified raw data that we would explore

    f = h5py.File('raw_data/train.chunk.0{}'.format(n), 'r')['train']
    keys = ['brand', 'maker', 'model', 'product', 'price']

    data = pd.DataFrame(columns=keys)

    for key in tqdm(keys):
        data[key] = f[key][:]

    #  Decode text columns to `utf-8`

    for key in tqdm(keys):
        if key != 'price':
            data[key] = data[key].str.decode('utf-8')

    #  Split texts from columns `brand` and `maker`

    data['brand'] = data['brand'].str.split('[^가-힣a-zA-Z]')
    data['maker'] = data['maker'].str.split('[^가-힣a-zA-Z]')

    # Morph texts from columns `model` and `product`

    data['model'] = data['model'].apply(Okt().morphs)
    data['product'] = data['product'].apply(Okt().morphs)

    # Replace `-1` with `np.nan` in `price` column

    data.loc[data[data['price']==-1].index, 'price'] = np.nan

    # Save to pickle format

    median_price = pd.Series(data['price'].median())

    maker_counts = pd.Series(np.concatenate(data['maker'])).value_counts()
    brand_counts = pd.Series(np.concatenate(data['brand'])).value_counts()
    model_counts = pd.Series(np.concatenate(data['model'])).value_counts()
    product_counts = pd.Series(np.concatenate(data['product'])).value_counts()

    total_counts = maker_counts.add(brand_counts, fill_value=0).add(model_counts, fill_value=0).add(product_counts, fill_value=0).sort_values(ascending=False)

    print(n)
    print(total_counts.head())
    print(median_price)

    with open('info/info-{}.pkl'.format(n), 'wb') as handle:
        pickle.dump((total_counts, median_price), handle)


for n in range(1,10):
    now = time.time()
    explore(n)
    print('TIME SPENT FOR ONE CHUNK: {}s'.format(int(time.time()-now)))

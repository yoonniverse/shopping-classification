import pandas as pd

x_val = pd.concat([pd.read_hdf('data/x_val-{}.h5'.format(i)) for i in range(0, 108)])
y_val = pd.concat([pd.read_hdf('data/y_val-{}.h5'.format(i)) for i in range(0, 108)])

x_val.to_hdf('data/x_val', key='x_val', mode='w')
y_val.to_hdf('data/y_val', key='y_val', mode='w')
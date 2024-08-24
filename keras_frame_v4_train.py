from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
import pandas as pd
from tensorflow import keras
import os
import h5py
import numpy as np
from keras.callbacks import EarlyStopping


def pearson_corr(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)


def get_data_slice(factor, ret, date_list, interval):
    data_slice = pd.DataFrame()
    for i in date_list:
        y_slice = pd.DataFrame(ret[i + interval, :])
        X_slice = pd.DataFrame(factor[i, :, :]).T
        data_slice_i = pd.concat([y_slice, X_slice], axis=1)
        data_slice = data_slice.append(data_slice_i)
    data_slice.dropna(axis=0, how='any', inplace=True)

    X = data_slice.iloc[:, 1:]
    y = data_slice.iloc[:, 0]
    return X.values, y.values


if __name__ == '__main__':
    # parameters
    params = {
        'name': '0924test',
        'feature_dim': 5,
        'n_epochs': 50,
        'roll_start': 5000,
        'freq': 40,
        'roll_end': 5730,
        'batch_size': 128,
        'train_size': 100,
        'valid_size': 50,
        'test_size': 1,
        'interval': 20
    }

    if not os.path.exists(params.get('name')):
        os.makedirs(params.get('name'))
    factor = np.array(h5py.File('factorExpr_s.mat')['factorExplr'])
    factor = factor[params.get('roll_start'):params.get('roll_end'), :, :4559]

    ret_tmp = pd.DataFrame(h5py.File('E:/HTSC_INTERN/alpha.mat')['dailyinfo/close_adj']).pct_change(
        params.get('interval'))
    ret_tmp = ret_tmp.iloc[params.get('roll_start'): params.get('roll_end'), ]

    ret = pd.DataFrame()
    for i in range(ret_tmp.shape[0]):
        ret_i = ret_tmp.iloc[i, :]
        ret_i = (ret_i - ret_i.mean()) / (ret_i.std())
        ret = pd.concat([ret, ret_i], axis=1)
    ret = ret.T.values

    date_start = params.get('roll_start') + params.get('train_size') + params.get('interval') * 2 + params.get(
        'valid_size')
    # date_end = params.get('roll_end') - params.get('interval') - params.get('test_size')
    date_end = params.get('roll_end')

    # training
    print('-' * 5 + ' training ' + '-' * 5)
    for t in range(date_start, date_end, params.get('freq')):
        stand_point = t
        print('*' * 10 + str(stand_point) + '*' * 10)
        t = t - params.get('roll_start')

        train_start = t - params.get('interval') * 2 - params.get('valid_size') - params.get('train_size')
        X_train, y_train = get_data_slice(factor, ret, list(range(train_start, train_start + params.get('train_size'))),
                                          params.get('interval'))

        valid_start = t - params.get('valid_size') - params.get('interval')
        X_valid, y_valid = get_data_slice(factor, ret, list(range(valid_start, t - params.get('interval'))),
                                          params.get('interval'))

        model = Sequential()
        model.add(Dense(16, input_dim=params.get('feature_dim'), activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='tanh'))

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        adam = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='mse', optimizer=adam, metrics=[pearson_corr])
        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=params.get('n_epochs'),
                  batch_size=params.get('batch_size'), callbacks=[early_stopping])

        model.save(params.get('name') + '/' + params.get('name') + '_' + str(stand_point) + '.h5')

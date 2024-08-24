import keras
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
import pandas as pd
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
from datetime import datetime as dt


def read_from_mat(file, path):
    ref_df = pd.DataFrame(file[path])
    df = pd.DataFrame(np.zeros(ref_df.shape))
    for i in range(ref_df.shape[0]):
        for j in range(ref_df.shape[1]):
            code = file[file[path][i][j]][:]
            if pd.isnull(code[0][0]):
                df.iloc[i, j] = np.nan
            else:
                df.iloc[i, j] = "".join(chr(c[0]) for c in code)
    return df.T


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


def get_data_slice(factor, ret, date, interval, index, cutpoint):
    if date <= cutpoint:
        y_slice = pd.DataFrame(ret[date + interval, :], index=index)
        X_slice = pd.DataFrame(factor[date, :, :].T, index=index)
        data_slice = pd.concat([y_slice, X_slice], axis=1)
        data_slice.dropna(axis=0, how='any', inplace=True)

        X = data_slice.iloc[:, 1:]
        y = data_slice.iloc[:, 0]
        return X.values, y.values, data_slice.index
    else:
        X_slice = pd.DataFrame(factor[date, :, :].T, index=index)
        X_slice.dropna(axis=0, how='any', inplace=True)
        return X_slice.values, None, X_slice.index


if __name__ == '__main__':
    # parameters
    params = {
        'name': '0923test',
        'feature_dim': 5,
        'n_epochs': 50,
        'roll_start': 5000,
        'freq': 40,
        'roll_end': 5730,
        'batch_size': 128,
        'train_size': 100,
        'valid_size': 50,
        'interval': 20,
        'alpha_mat_path': 'E:/HTSC_INTERN/alpha.mat'
    }

    factor = np.array(h5py.File('factorExpr_s.mat')['factorExplr'])
    factor = factor[params.get('roll_start'):params.get('roll_end'), :, :4559]

    ret_tmp = pd.DataFrame(h5py.File(params.get('alpha_mat_path'))['dailyinfo/close_adj']).pct_change(
        params.get('interval'))
    ret_tmp = ret_tmp.iloc[params.get('roll_start'): params.get('roll_end'), ]
    ret = pd.DataFrame()
    for i in range(ret_tmp.shape[0]):
        ret_i = ret_tmp.iloc[i, :]
        ret_i = (ret_i - ret_i.mean()) / (ret_i.std())
        ret = pd.concat([ret, ret_i], axis=1)
    ret = ret.T.values

    code_index = read_from_mat(h5py.File(params.get('alpha_mat_path')), 'basicinfo/stock_number_wind').iloc[:, 0].values

    date_start = params.get('roll_start') + params.get('train_size') + params.get('interval') * 2 + params.get(
        'valid_size')
    date_end = params.get('roll_end')

    model_list = pd.Series(list(os.walk(params.get('name')))[0][2])
    model_list = model_list.apply(lambda x: int(x[(len(params.get('name')) + 1):][:-3]))
    model_list = model_list.sort_values()

    # testing
    print('-' * 5 + ' testing ' + '-' * 5)
    all_loss = []
    all_corr = []
    pred_mat = pd.DataFrame()
    cutp = date_end - params.get('interval') - params.get('roll_start') - 1
    for t in range(date_start + 1, date_end):
        stand_point = t
        print('*' * 10 + str(stand_point) + '*' * 10)  # matlab index - 1

        test_date = t - params.get('roll_start')
        X_test, y_test, index_slice = get_data_slice(factor, ret, test_date, params.get('interval'), code_index, cutp)

        model_stand_point = model_list[(stand_point - model_list) > 0].iloc[-1]

        model = keras.models.load_model(
            params.get('name') + '/' + params.get('name') + '_' + str(model_stand_point) + '.h5',
            custom_objects={'pearson_corr': pearson_corr})
        print('model: ', model_stand_point)

        pred = model.predict(X_test)
        if test_date <= cutp:
            corr_test = pd.Series(y_test).corr(pd.Series(pred[:, 0]))
            print('corr: ', corr_test)
            all_corr.append(corr_test)

        pred_reindex = pd.Series(np.nan, index=code_index, name=stand_point)
        pred_reindex[index_slice] = pred[:, 0]
        pred_mat = pd.concat([pred_mat, pred_reindex], axis=1)

    print('ic: ', pd.Series(all_corr).mean())
    print('ic_std: ', pd.Series(all_corr).std())
    print('icir: ', pd.Series(all_corr).mean() / pd.Series(all_corr).std())

    plt.title(params.get('name') + '_allcorr')
    plt.plot(pd.Series(all_corr).cumsum())
    plt.show()

    trade_dates = pd.DataFrame(h5py.File(params.get('alpha_mat_path'))['dailyinfo/dates']).iloc[:, 0]
    pred_mat.columns = trade_dates[pred_mat.columns].apply(lambda x: dt.fromordinal(int(x) - 366)).values
    pred_mat.to_csv(params.get('name') + '.csv')

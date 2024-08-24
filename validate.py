import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py


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
        'name': '0923test',
        'feature_dim': 5,
        'n_epochs': 50,
        'roll_start': 3000,
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
    date_end = params.get('roll_end') - params.get('interval') - params.get('test_size')

    factor1_corr = []
    factor2_corr = []
    factor3_corr = []
    factor4_corr = []
    factor5_corr = []
    for t in range(date_start + 1, date_end + 1):
        stand_point = t
        # print('*' * 10 + str(stand_point) + '*' * 10)

        test_start = t - params.get('roll_start')
        X_test, y_test = get_data_slice(factor, ret, list(range(test_start, test_start + params.get('test_size'))),
                                        params.get('interval'))

        factor1_corr.append(pd.Series(X_test[:, 0]).corr(pd.Series(y_test),method='pearson'))
        factor2_corr.append(pd.Series(X_test[:, 1]).corr(pd.Series(y_test),method='pearson'))
        factor3_corr.append(pd.Series(X_test[:, 2]).corr(pd.Series(y_test),method='pearson'))
        factor4_corr.append(pd.Series(X_test[:, 3]).corr(pd.Series(y_test),method='pearson'))
        factor5_corr.append(pd.Series(X_test[:, 4]).corr(pd.Series(y_test),method='pearson'))

    print('*' * 10 + ' results ' + '*' * 10)
    for i in range(5):
        print('factor', i, ' corr: ', pd.Series(globals()['factor'+str(i+1)+'_corr']).mean())

    for i in range(5):
        plt.plot(pd.Series(globals()['factor'+str(i+1)+'_corr']).cumsum(), label='factor ' + str(i))
    plt.legend()
    plt.show()
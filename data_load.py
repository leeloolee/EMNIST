## this is code to transfrom raw data to tf record

import pandas as pd
import numpy as np

def data_loader(DATA_URL):
    data = {key.lower():pd.read_csv(DATA_URL[key]) for key in DATA_URL}
    x_train, y_train = transform_data_to_numpy(data, 'train')
    x_test = transform_data_to_numpy(data, 'test')
    return x_train, y_train, x_test

def check_dataset_correct(data):
    ## data length 확인
    train_length_check = len(data['train'].columns)==787
    test_length_check = (len(data['test'].columns)==786)
    return train_length_check and test_length_check

def transform_data_to_numpy(data, phase ='train'):
    '''
    :param data: pandas dataframe
    :param phase : 'train' or 'test', default =train?
    :return:
    '''
    assert check_dataset_correct(data)
    if phase == 'train':
        x_train = data['train'].drop(['id', 'digit', 'letter'], axis=1).values
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_train = x_train / 255

        y = data['train']['digit']
        y_train = np.zeros((len(y), len(y.unique())))
        for i, digit in enumerate(y):
            y_train[i, digit] = 1
        return x_train, y_train
    elif phase == 'test':
        x_test = data['test'].drop(['id', 'letter'], axis=1).values
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_test = x_test / 255
        return x_test

    else:
        NotImplemented

# TODO : 데이터 전처리
"""
data_preprocess(data):
    NotImplemented
"""

if __name__ == '__main__':
#git
    pass




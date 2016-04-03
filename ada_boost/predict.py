# coding=utf-8
import numpy as np
from boost import ada_classify
from adaboost import ada_boost_train_ds
from data_set import load_data_set as lds


def predict(tr_data_arr, tr_label_arr, test_data_arr, test_label_arr):
    data_arr = np.mat(tr_data_arr)
    f_label_arr = []
    for i in tr_label_arr:
        if i == 1:
            f_label_arr.append(i)
        else:
            f_label_arr.append(-1)
    classifier_arr = ada_boost_train_ds(data_arr, f_label_arr, 30)

    test_label_arr = np.mat([[i] for i in test_label_arr])
    test_label_arr[test_label_arr == 0] = -1

    pred_arr = ada_classify(test_data_arr, classifier_arr)

    diff = test_label_arr - pred_arr
    error = (diff != [0]).sum()

    test_data_len = len(test_data_arr)
    accuracy = (test_data_len - error) * 100.0 / test_data_len

    return test_data_len, error, accuracy

# coding=utf-8
import numpy as np

import log_regres as lr
from data_set import load_data_set as lds


def judge_by_log(x, weis):
    z = np.dot(x, weis)
    res = lr.sigmoid(z)
    if res > 0.5:
        return 1
    return 0


def predict(tr_data_arr, tr_label_arr, pred_data_arr, pred_label_arr):
    weis = lr.stoc_grad_ascent(tr_data_arr, tr_label_arr)

    pred_data_len = len(pred_label_arr)
    wrong = 0
    for idx, data in enumerate(pred_data_arr):
        x = np.array(data)
        res = judge_by_log(x, weis)
        if res != int(pred_label_arr[idx]):
            wrong += 1

    accuracy = (pred_data_len - wrong) * 100.0 / pred_data_len
    return pred_data_len, wrong, accuracy

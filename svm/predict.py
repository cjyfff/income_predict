# coding=utf-8

from data_set import load_data_set as lds
from libsvm.svmutil import (
    svm_train,
    svm_predict,
    svm_problem,
    svm_parameter
)


def data_format(data_arr):
    data_list = []
    for data in data_arr:
        data_dict = {}
        for idx, item in enumerate(data):
            data_dict[idx + 1] = item
        data_list.append(data_dict)
    return data_list


def predict():
    data_arr, label_arr = lds.load('./data_set/adult.data')
    data_arr = data_format(data_arr)
    prob = svm_problem(label_arr, data_arr)
    param = svm_parameter('-c 2048.0 -g 0.001953125')
    svm_model = svm_train(prob, param)

    pred_data_arr, pred_label_arr = lds.load('./data_set/adult.test')
    pred_data_arr = data_format(pred_data_arr)
    pred_data_len = len(pred_label_arr)
    wrong = 0
    for idx, data in enumerate(pred_data_arr):
        p_label, p_acc, p_val = svm_predict(
            [pred_label_arr[idx]], [data], svm_model)
        if int(p_label[0]) != int(pred_label_arr[idx]):
            wrong += 1

    print 'SVM预测错误数目：', wrong
    print 'SVM正确率：', (pred_data_len - wrong) * 1.0 / pred_data_len


def get_cv_data_file():
    """获取进行交叉验证的训练样本"""
    data_arr, label_arr = lds.load('./data_set/adult.data')

    with open('cv_data.smp', 'w+') as fp:
        for d_idx, data in enumerate(data_arr):
            feature_list = []
            for f_idx, feature in enumerate(data):
                feature_list.append('%s:%s' % (f_idx + 1, feature))
            feature_str = ' '.join(feature_list)
            fp.write('%s %s\n' % (label_arr[d_idx], feature_str))

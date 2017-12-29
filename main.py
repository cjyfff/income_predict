#! /usr/bin/env python
# coding=utf-8
import time
import numpy as np
from log_regres import predict as lr_predict
from svm import predict as svm_predict
from ada_boost import predict as ada_predict
from data_set import load_data_set as lds


def main():
    tr_data_arr, tr_label_arr = lds.load('./data_set/adult.data')
    pred_data_arr, pred_label_arr = lds.load('./data_set/adult.test')

    lr_st = time.time()
    lr_pred_data_len, lr_wrong, lr_accuracy = lr_predict.predict(
        tr_data_arr, tr_label_arr, pred_data_arr, pred_label_arr)
    lr_et = time.time()

    svm_st = time.time()
    svm_pred_data_len, svm_wrong, svm_accuracy = svm_predict.predict(
        tr_data_arr, tr_label_arr, pred_data_arr, pred_label_arr)
    svm_et = time.time()

    ada_st = time.time()
    ada_pred_data_len, ada_wrong, ada_accuracy = ada_predict.predict(
        tr_data_arr, tr_label_arr, pred_data_arr, pred_label_arr)
    ada_et = time.time()

    print '测试样本总数：', lr_pred_data_len
    print 'LR预测错误数：', lr_wrong
    print 'LR预测准确率：%s' % lr_accuracy, '%'
    print 'LR训练模型以及预测共耗时：%s秒' % (lr_et - lr_st)
    print '---------------------'
    print 'SVM预测错误数：', svm_wrong
    print 'SVM预测准确率：%s' % svm_accuracy, '%'
    print 'SVM训练模型以及预测共耗时：%s秒' % (svm_et - svm_st)
    print '---------------------'
    print 'AdaBoost预测错误数：', ada_wrong
    print 'AdaBoost预测准确率：%s' % ada_accuracy, '%'
    print 'AdaBoost训练模型以及预测共耗时：%s秒' % (ada_et - ada_st)


def test_adaboost_roc():
    """计算AdaBoost的ROC以及AUC"""
    from ada_boost.adaboost import ada_boost_train_ds, plot_roc
    tr_data_arr, tr_label_arr = lds.load('./data_set/adult.data')
    data_arr = np.mat(tr_data_arr)
    f_label_arr = []
    for i in tr_label_arr:
        if i == 1:
            f_label_arr.append(i)
        else:
            f_label_arr.append(-1)
    classifier_arr, agg_class_est = ada_boost_train_ds(data_arr, f_label_arr, 30)
    plot_roc(agg_class_est.T, tr_label_arr)


if __name__ == '__main__':
    main()
    # test_adaboost_roc()

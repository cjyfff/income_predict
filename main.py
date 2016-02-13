#! /usr/bin/env python
# coding=utf-8
import time
from log_regres import predict as lr_predict
from svm import predict as svm_predict


def main():
    lr_st = time.time()
    lr_pred_data_len, lr_wrong, lr_accuracy = lr_predict.predict()
    lr_et = time.time()

    svm_st = time.time()
    svm_pred_data_len, svm_wrong, svm_accuracy = svm_predict.predict()
    svm_et = time.time()

    print '测试样本总数：', lr_pred_data_len
    print 'LR预测错误数：', lr_wrong
    print 'LR预测准确率：%s' % lr_accuracy
    print 'LR训练模型以及预测共耗时：%s秒' % (lr_et - lr_st)
    print '---------------------'
    print 'SVM预测错误数：', svm_wrong
    print 'SVM预测准确率：%s' % svm_accuracy
    print 'SVM训练模型以及预测共耗时：%s秒' % (svm_et - svm_st)


if __name__ == '__main__':
    main()

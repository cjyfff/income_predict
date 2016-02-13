#! /usr/bin/env python
# coding=utf-8
from log_regres import predict as lr_predict
from svm import predict as svm_predict


def main():
    lr_predict.predict()

    svm_predict.predict()
    # svm_predict.get_cv_data_file()

if __name__ == '__main__':
    main()

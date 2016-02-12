#! /usr/bin/env python
# coding=utf-8
from data_set import load_data_set as lds
from log_regres import log_regres as lr


def main():
    data_arr, label_mat = lds.load('./data_set/adult.data')
    w = lr.grad_ascent(data_arr, label_mat)
    print('weights:', w)


if __name__ == '__main__':
    main()

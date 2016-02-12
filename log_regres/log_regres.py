#! /usr/bin/env python
# coding=utf-8
import random
import numpy as np

from data_set import load_data_set as lds


def sigmoid(in_x):
    return 1.0 / (1 + np.exp(-in_x))


def grad_ascent(data_mat_in, class_labels):
    """梯度上升"""
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).T
    m, n = data_matrix.shape
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))
    for k in xrange(max_cycles):
        h = sigmoid(np.dot(data_matrix, weights))
        error = (label_mat - h)
        """
        data_matrix.T * error就是目标函数
        每一次迭代得出的weights都会使得目标函数增长
        """
        weights += alpha * np.dot(data_matrix.T, error)
    return weights


def stoc_grad_ascent(data_matrix, class_labels, num_iter=150):
    """随机梯度上升"""
    data_matrix = np.array(data_matrix)
    class_labels = np.array(class_labels)
    m, n = data_matrix.shape
    """假如weights是作为参数传入的话，即可实现线上学习"""
    weights = np.ones(n)
    for j in xrange(num_iter):
        data_index = range(m)
        for i in xrange(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights += alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])
    return weights


if __name__ == '__main__':
    data_arr, label_mat = lds.load('./data_set/adult.data')
    w = grad_ascent(data_arr, label_mat)
    print('weights:', w)

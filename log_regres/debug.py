# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import log_regres as lr


def load_data_set():
    data_mat = []
    label_mat = []
    fr = open('test_set.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def plot_best_fit(wei):
    weights = wei
    data_mat, label_mat = load_data_set()
    data_arr = np.array(data_mat)
    m = data_arr.shape[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in xrange(m):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def test():
    data_arr, label_mat = load_data_set()
    print 'data_arr: ', data_arr
    print 'label_mat: ', label_mat
    w = lr.grad_ascent(data_arr, label_mat)
    # w = stoc_grad_ascent(data_arr, label_mat)
    print('weights: ', w)
    plot_best_fit(w)


if __name__ == '__main__':
    test()

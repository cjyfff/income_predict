# coding=utf-8
import pandas as pd
import load_data_set


def analyze():
    data_mat, label_mat = load_data_set.load('./adult.data')
    data = pd.DataFrame(data_mat, columns=['constant', 'age', 'workclass', 'education',
                                           'education-num', 'marital-status', 'occupation',
                                           'relationship', 'race', 'sex', 'capital-gain',
                                           'capital-loss', 'hours-per-week', 'native-country'])
    label = pd.Series(label_mat)
    print('最大年龄： %s' % max(data['age']))
    print('最小年龄： %s' % min(data['age']))
    print('正样本数目： %s' % len(label[label == 1]))
    print('负样本数目： %s' % len(label[label == 0]))


if __name__ == '__main__':
    analyze()

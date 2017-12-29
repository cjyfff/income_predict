# coding=utf-8
import sys
import traceback
import pandas as pd
import load_data_set as lds


def load_data(data_file):
    data = []
    label = []

    work_class_emu = lds.WorkClassEmu()
    education_emu = lds.EducationEmu()
    marital_status_emu = lds.MaritalStatusEmu()
    occupation_emu = lds.OccupationEmu()
    relationship_emu = lds.RelationshipEmu()
    race_emu = lds.RaceEmu()
    sex_emu = lds.SexEmu()
    native_country_emu = lds.NativeCountryEmu()

    with open(data_file, 'r') as fp:
        line = fp.readline()
        while line:
            line = line.split(',')
            if len(line) == 1:
                line = fp.readline()
                continue
            try:
                """
                数据集的fnlwgt暂时不知道含义，
                而且该值比较大，不做归一化的话会影响模型的拟合，
                故暂时忽略该项。
                """
                data_line = [1.0,
                             float(line[0]),
                             work_class_emu.run(line[1]),
                             education_emu.run(line[3]),
                             float(line[4]),
                             marital_status_emu.run(line[5]),
                             occupation_emu.run(line[6]),
                             relationship_emu.run(line[7]),
                             race_emu.run(line[8]),
                             sex_emu.run(line[9]),
                             float(line[10]),
                             float(line[11]),
                             float(line[12]),
                             native_country_emu.run(line[13])
                             ]
                data.append(data_line)

                label_data = line[14].strip()
                if label_data in ['>50K', '>50K.']:
                    label.append(1)
                elif label_data in ['<=50K', '<=50K.']:
                    label.append(0)
                else:
                    raise ValueError('Invalid label: %s' % label_data)

            except Exception as err:
                print 'error happen:', err
                traceback.print_exc()
                sys.exit(1)
            line = fp.readline()
    return data, label


def analyze():
    data_mat, label_mat = load_data('./adult.data')
    data = pd.DataFrame(data_mat, columns=['constant', 'age', 'workclass', 'education',
                                           'education-num', 'marital-status', 'occupation',
                                           'relationship', 'race', 'sex', 'capital-gain',
                                           'capital-loss', 'hours-per-week', 'native-country'])
    label = pd.Series(label_mat)
    print('最大年龄： %s' % max(data['age']))
    print('最小年龄： %s' % min(data['age']))
    print('最长受教育时间： %s' % max(data['education-num']))
    print('最短受教育时间： %s' % min(data['education-num']))
    print('最长一周工作时间： %s' % max(data['hours-per-week']))
    print('最短一周工作时间： %s' % min(data['hours-per-week']))
    print('正样本数目： %s' % len(label[label == 1]))
    print('负样本数目： %s' % len(label[label == 0]))


if __name__ == '__main__':
    analyze()

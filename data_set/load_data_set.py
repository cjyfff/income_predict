# coding=utf-8
import sys
import traceback


class DataEnu(object):
    def normalization_data(self, data):
        min_value = min(self.enumerate_dict.values())
        max_value = max(self.enumerate_dict.values())
        return (self.enumerate_dict[data] + 0.1 - min_value) * 1.0 / max_value

    def run(self, data):
        data = data.strip()
        if data in self.enumerate_dict:
            return self.normalization_data(data)
        return 0.001


class WorkClassEnu(DataEnu):
    enumerate_dict = {'Private': 1,
                      'Self-emp-not-inc': 2,
                      'Self-emp-inc': 3,
                      'Federal-gov': 4,
                      'Local-gov': 5,
                      'State-gov': 6,
                      'Without-pay': 7,
                      'Never-worked': 8}


class EducationEnu(DataEnu):
    enumerate_dict = {'Bachelors': 1,
                      'Some-college': 2,
                      '11th': 3,
                      'HS-grad': 4,
                      'Prof-school': 5,
                      'Assoc-acdm': 6,
                      'Assoc-voc': 7,
                      '9th': 8,
                      '7th-8th': 9,
                      '12th': 10,
                      'Masters': 11,
                      '1st-4th': 12,
                      '10th': 13,
                      'Doctorate': 14,
                      '5th-6th': 15,
                      'Preschool': 16}


class MaritalStatusEnu(DataEnu):
    enumerate_dict = {'Married-civ-spouse': 1,
                      'Divorced': 2,
                      'Never-married': 3,
                      'Separated': 4,
                      'Widowed': 5,
                      'Married-spouse-absent': 6,
                      'Married-AF-spouse': 7}


def load(data_file):
    data = []

    work_class_enu = WorkClassEnu()
    education_enu = EducationEnu()
    marital_status_enu = MaritalStatusEnu()

    with open(data_file, 'r') as fp:
        line = fp.readline()
        while line:
            line = line.split(',')
            if len(line) < 3:
                break
            try:
                """
                数据集的第三项fnlwgt暂时不知道含义，
                而且该值比较大，不做归一化的话会影响模型的拟合，
                故暂时忽略该项。
                """
                data_line = [line[0],
                             work_class_enu.run(line[1]),
                             education_enu.run(line[3]),
                             line[4],
                             marital_status_enu.run(line[5])
                             ]
                data.append(data_line)
            except Exception as err:
                print 'error happen:', err
                traceback.print_exc()
                sys.exit(1)
            line = fp.readline()
    print data


if __name__ == '__main__':
    load('./adult.data')

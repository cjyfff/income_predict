# coding=utf-8
import sys
import traceback
import numpy as np


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


class OccupationEnu(DataEnu):
    enumerate_dict = {'Tech-support': 1,
                      'Craft-repair': 2,
                      'Other-service': 3,
                      'Sales': 4,
                      'Exec-managerial': 5,
                      'Prof-specialty': 6,
                      'Handlers-cleaners': 7,
                      'Machine-op-inspct': 8,
                      'Adm-clerical': 9,
                      'Farming-fishing': 10,
                      'Transport-moving': 11,
                      'Priv-house-serv': 12,
                      'Protective-serv': 13,
                      'Armed-Forces': 14}


class RelationshipEnu(DataEnu):
    enumerate_dict = {'Wife': 1,
                      'Own-child': 2,
                      'Husband': 3,
                      'Not-in-family': 4,
                      'Other-relative': 5,
                      'Unmarried': 6}


class RaceEnu(DataEnu):
    enumerate_dict = {'White': 1,
                      'Asian-Pac-Islander': 2,
                      'Amer-Indian-Eskimo': 3,
                      'Other': 4,
                      'Black': 5}


class SexEnu(DataEnu):
    enumerate_dict = {'Female': 1,
                      'Male': 2}


class NativeCountryEnu(DataEnu):
    enumerate_dict = {'United-States': 1,
                      'Cambodia': 2,
                      'England': 3,
                      'Puerto-Rico': 4,
                      'Canada': 5,
                      'Germany': 6,
                      'Outlying-US(Guam-USVI-etc)': 7,
                      'India': 8,
                      'Japan': 9,
                      'Greece': 10,
                      'South': 11,
                      'China': 12,
                      'Cuba': 13,
                      'Iran': 14,
                      'Honduras': 15,
                      'Philippines': 16,
                      'Italy': 17,
                      'Poland': 18,
                      'Jamaica': 19,
                      'Vietnam': 20,
                      'Mexico': 21,
                      'Portugal': 22,
                      'Ireland': 23,
                      'France': 24,
                      'Dominican-Republic': 25,
                      'Laos': 26,
                      'Ecuador': 27,
                      'Taiwan': 28,
                      'Haiti': 29,
                      'Columbia': 30,
                      'Hungary': 31,
                      'Guatemala': 32,
                      'Nicaragua': 33,
                      'Scotland': 34,
                      'Thailand': 35,
                      'Yugoslavia': 36,
                      'El-Salvador': 37,
                      'Trinadad&Tobago': 38,
                      'Peru': 39,
                      'Hong': 40,
                      'Holand-Netherlands': 41}


def get_capital_gain_and_capital_loss_range(data_file):
    """
    capital_gain和capital_loss的数值都比较大，
    因此需要获得这两项数据的最大最小值，以便进行归一化
    """
    with open(data_file, 'r') as fp:
        capital_gain = []
        capital_loss = []
        line = fp.readline()
        while line:
            try:
                line = line.split(',')
                if len(line) == 1:
                    line = fp.readline()
                    continue
                capital_gain.append(line[10].strip())
                capital_loss.append(line[11].strip())
            except Exception as err:
                print 'error happen:', err
                traceback.print_exc()
                sys.exit(1)
            line = fp.readline()
    res = [[float(max(capital_gain)), float(min(capital_gain))],
           [float(max(capital_loss)), float(min(capital_loss))]]
    return res


def discretized_age(age):
    """对age进行离散化"""
    if age < 20:
        return 0.1
    elif age < 40:
        return 0.2
    elif age < 60:
        return 0.3
    return 0.5


def load(data_file):
    data = []
    label = []

    capital_data = get_capital_gain_and_capital_loss_range(data_file)
    capital_gain_max, capital_gain_min = capital_data[0]
    capital_loss_max, capital_loss_min = capital_data[1]

    work_class_enu = WorkClassEnu()
    education_enu = EducationEnu()
    marital_status_enu = MaritalStatusEnu()
    occupation_enu = OccupationEnu()
    relationship_enu = RelationshipEnu()
    race_enu = RaceEnu()
    sex_enu = SexEnu()
    native_country_enu = NativeCountryEnu()

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
                             discretized_age(float(line[0])),
                             work_class_enu.run(line[1]),
                             education_enu.run(line[3]),
                             float(line[4]),
                             marital_status_enu.run(line[5]),
                             occupation_enu.run(line[6]),
                             relationship_enu.run(line[7]),
                             race_enu.run(line[8]),
                             sex_enu.run(line[9]),
                             (float(line[10]) + 0.1 - capital_gain_min) / capital_gain_max,
                             (float(line[11]) + 0.1 - capital_loss_min) / capital_loss_max,
                             float(line[12]),
                             native_country_enu.run(line[13])
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


if __name__ == '__main__':
    data_mat, label_mat = load('./adult.data')
    print np.mat(data_mat)

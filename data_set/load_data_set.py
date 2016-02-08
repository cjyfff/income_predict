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


def load(data_file):
    data = []

    work_class_enu = WorkClassEnu()
    education_enu = EducationEnu()
    marital_status_enu = MaritalStatusEnu()
    occupation_enu = OccupationEnu()
    relationship_enu = RelationshipEnu()
    race_enu = RaceEnu()
    sex_enu = SexEnu()

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
                data_line = [line[0],
                             work_class_enu.run(line[1]),
                             education_enu.run(line[3]),
                             line[4],
                             marital_status_enu.run(line[5]),
                             occupation_enu.run(line[6]),
                             relationship_enu.run(line[7]),
                             race_enu.run(line[8]),
                             sex_enu.run(line[9]),
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

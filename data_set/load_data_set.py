# coding = utf-8


class DataEnu(object):
    def __init__(self, data):
        self.data = data.strip()

    def normalization_data(self, data):
        min_value = min(self.enumerate_dict.values())
        max_value = max(self.enumerate_dict.values())
        return (self.enumerate_dict[data] + 0.1 - min_value) * 1.0 / max_value

    def run(self):
        if self.data in self.enumerate_dict:
            return self.normalization_data(self.data)
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


def load(file):
    data = []
    with open(file, 'r') as fp:
        line = fp.readline()
        while line:
            line = line.split(',')
            try:
                data_line = [line[0],
                             WorkClassEnu(line[1]).run()
                             ]
                data.append(data_line)
            except Exception as err:
                print 'error happen:', err
            line = fp.readline()
    print data


if __name__ == '__main__':
    load('./adult.data')

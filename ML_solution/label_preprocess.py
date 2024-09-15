import os

class Preprocess:
    def __init__(self):
        self.result = {}

    def merge(self, segments):
        """
        :param segments: list of overlapping intervals with same label, segment[i][0] - start, segment[i][1] - end
        :return: list of intervals with no overlaps
        """
        merged = []
        segments.sort(key=lambda x: x[0])

        for segment in segments:
            if not merged or segment[0] > merged[-1][1]:
                merged.append(segment)
            else:
                merged[-1][1] = segment[1]

        return merged

    def labeling(self, intervals: list[list[float], int]):
        """
        :param intervals: list of intervals where intervals[i][0] - start intervals[i][1] - end intervals[i][2] - label
        :return: labels - dictionary, keys - labels, values - intervals
        """
        labels = {}

        for interval in intervals:
            if interval[-1] not in labels:
                labels[interval[-1]] = [interval[0]]
            else:
                labels[interval[-1]].append(interval[0])

        # print(labels)
        for key in labels:
            labels[key] = self.merge(labels[key])

        return labels

    def apply(self):
        """whole directory check"""
        for root, dirs, files in os.walk("/Users/konstantin/Desktop/TUEV_data/edf/train", topdown=True):
            for name in files:
                if name.endswith('.rec'):
                    file_path = os.path.join(root, name)

                    dct = {}
                    """
                    node[0] - chanel
                    node[1] - start
                    node[2] - end
                    node[3] - label
                    """
                    with open(file_path, 'r') as file:
                        for node in file.readlines():
                            node = node[:-1]

                            node = node.split(',')

                            if int(node[0]) not in dct:
                                dct[int(node[0])] = [[[float(node[1]), float(node[2])], int(node[3])]]
                            else:
                                dct[int(node[0])].append([[float(node[1]), float(node[2])], int(node[3])])

                    for key in dct:
                        dct[key] = self.labeling(dct[key])

                    self.result[name[:-4]] = dct
        return self.result

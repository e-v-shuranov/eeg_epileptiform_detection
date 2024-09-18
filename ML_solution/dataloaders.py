
import torch
import random
import csv
import numpy as np

from model_cnn_lstm_baseline import *

class TEST_TUEV(torch.utils.data.Dataset):
    def __init__(self, path): #, tuh_filtered_stat_vals):
        super(TEST_TUEV, self).__init__()
        #read csv
        # self.data = pd.read_csv(path, sep=',')
        file = open(path, "r")
        # self.data = list(csv.reader(file, delimiter=","))[0]
        reader = csv.DictReader(file)
        self.data = list(reader)

        file.close()
        print("len(self.data):",len(self.data))

    def __len__(self):
        return len(self.data)
        # self.data.shape[1]

    def __getitem__(self, idx):
        fname = self.data[idx]["path"]
        data = np.load(fname, allow_pickle=True)
        isnan = np.isnan(data)
        if sum(sum(isnan))>0:
            print("data:", data)

        sel_row = random.randint(0, data.shape[0]-1)
        label = int(self.data[idx]["label"])
        label_one_hot =torch.zeros(num_classes+1)
        label_one_hot[label] = 1
        return {'data': torch.from_numpy(np.array(data[sel_row])).float(),
                'label': torch.from_numpy(np.array(label_one_hot)).long()}
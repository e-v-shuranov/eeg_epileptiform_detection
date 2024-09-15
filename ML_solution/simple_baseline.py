import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import csv

device = "cuda:1" if torch.cuda.is_available() else "cpu"

num_classes = 6
class simple_EEGNet(nn.Module):
    def __init__(self):
        super(simple_EEGNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.rnn = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, num_classes+1)

    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))
        x = x.permute(0,2,1)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

def my_get_dummies(array):
    columns = ['1', '2', '3', '4', '5', '6', 'None']
    tmp_df = pd.DataFrame(np.zeros((len(array), 7), float),
                          columns = columns)

    for index in range(len(array)):
        tmp_df.iloc[index] = [1 if idx in array[index] else 0 for idx in columns]
        if index > 0 and index % 30000 == 0:
            break

    return tmp_df


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
        data = np.load(fname, allow_pickle=True).item()
        label = self.data[idx]["label"]
        return {'data': torch.from_numpy(np.array(data)).float(),
                'label': torch.from_numpy(np.array(label)).float()}

class TEST_TUEV_old(torch.utils.data.Dataset):
    def __init__(self, path): #, tuh_filtered_stat_vals):
        super(TEST_TUEV, self).__init__()
        #read csv
        df = pd.read_csv(path, sep=',')
        df = df.rename(columns={'FFT-image': 'FFT_image'})
        tmp_df = pd.DataFrame(np.zeros((len(df.FFT_image_signal_0), 96), float),
                              columns=[f'FFT_image_signal_{index}' for index in range(96)])
        for index in range(len(df.FFT_image_signal_0)):
            tmp_df.iloc[index] = np.array(df.FFT_image[index][1:-1].split(), float)
            if index%10000 == 0:
                print("progress:", index,len(df.FFT_image))
            if index> 0 and index%30000 == 0:
                break
        # df.drop('FFT_image', axis=1)
        self.data = pd.concat([df['channel-number'], df['window_start'], tmp_df], axis=1)
        self.label = my_get_dummies(df['label'])


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]

        label = self.label.iloc[idx]
        return {'data': torch.from_numpy(np.array(data)).float(),
                'label': torch.from_numpy(np.array(label)).float()}

def main():
    # Example usage
    # loss = nn.BCEWithLogitsLoss()
    # input = torch.randn(3, requires_grad=True)
    # target = torch.empty(3).random_(2)
    #
    # output = loss(input, target)
    # output.backward()
    # print(input)

    model = simple_EEGNet().to(device)
    # input_data = torch.randn(16, 1, 2500)  # batch_size, channels, seq_length
    # output = model(input_data)
    #
    # print(output.size())
    loss_function = nn.BCEWithLogitsLoss()
    batch_sz = 2

    # val_dataset  = TEST_TUEV('/home/eshuranov/projects/eeg_stud/EEG_validation.csv')
    train_dataset  = TEST_TUEV('/home/eshuranov/projects/eeg_epileptiform_detection/EEG2Rep/Dataset/out_tuev/row_list.csv')


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=1,
                                               drop_last=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_sz, shuffle=True, num_workers=1,
    #                                            drop_last=True)
    steps = 0
    for epoch in range(20):
        sum_loss = 0
        for batch in train_loader:
            output = model(batch['data'].to(device))

            loss = loss_function(output, batch['label'].to(device))
            sum_loss += loss.item()
            loss.backward()
        steps += 1
        print("Epoch: ",epoch,"sum_loss: ", loss)
        if steps != 0 and steps % 5 == 0:
            try:
                with torch.no_grad():
                    loss = 0
                    for batch in train_loader:
                        output = model(batch)
                        loss += torch.nn.CrossEntropyLoss()(output, batch['label'])
                    print('Val Loss: {}\t'.format(loss))
            except:
                raise




if __name__ == '__main__':
    print("main started")
    main()
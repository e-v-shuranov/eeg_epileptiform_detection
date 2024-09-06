import torch
import torch.nn as nn
import numpy as np
import pandas as pd


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
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0,2,1)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x


class TEST_TUEV(torch.utils.data.Dataset):
    def __init__(self, path): #, tuh_filtered_stat_vals):
        super(TEST_TUEV, self).__init__()
        #read csv
        df = pd.read_csv(path, sep=',')
        df = df.rename(columns={'FFT-image': 'FFT_image'})
        tmp_df = pd.DataFrame(np.zeros((len(df.FFT_image), 96), float),
                              columns=[f'FFT_image_signal_{index}' for index in range(96)])
        for index in range(len(df.FFT_image)):
            tmp_df.iloc[index] = np.array(df.FFT_image[index][1:-1].split(), float)
            if index%10000 == 0:
                print("progress:", index,len(df.FFT_image))
            if index> 0 and index%100000 == 0:
                break
        df.drop('FFT_image', axis=1)
        self.data = pd.concat([df['channel_number', df['window_start'], tmp_df]], axis=1)
        self.label = df['label']


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = self.data[idx]

        label = self.label[idx]
        return {'data': data,
                'label': label}



def main():
    # Example usage
    model = simple_EEGNet()
    # input_data = torch.randn(16, 1, 2500)  # batch_size, channels, seq_length
    # output = model(input_data)
    #
    # print(output.size())

    batch_sz = 16

    train_dataset,val_dataset  = TEST_TUEV('/home/eshuranov/projects/eeg_stud/Dataset/TUEV_csv/data.csv')
    # val_dataset = TEST_TUEV('/Dataset/TUEV_csv/data.csv',is_train = False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=1,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_sz, shuffle=True, num_workers=1,
                                               drop_last=True)
    steps = 0
    for epoch in range(20):
        for batch in train_loader:
            output = model(batch)

            loss = torch.nn.CrossEntropyLoss()(output, batch['label'])
            loss.backward()
        steps += 1
        if steps != 0 and steps % 5 == 0:
            try:
                with torch.no_grad():
                    loss = 0
                    for batch in val_loader:
                        output = model(batch)
                        loss += torch.nn.CrossEntropyLoss()(output, batch['label'])
                    print('Val Loss: {}\t'.format(loss))
            except:
                raise


if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import csv
import random



device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

def set_random_state(random_state:int=0):
    """Initialize random generators.

    Parameters
    ==========
    random_state : int = 0
        Determines random number generation for centroid initialization.
        Use an int to make the randomness deterministic.
    """
    torch.manual_seed(random_state)
    random.seed(random_state)
    np.random.seed(random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.cuda.manual_seed(random_state)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    random_state = 42
    set_random_state(random_state)

    model = simple_EEGNet().to(device)
    # input_data = torch.randn(16, 1, 2500)  # batch_size, channels, seq_length

    loss_function = nn.BCEWithLogitsLoss()
    batch_sz = 16

    val_dataset  = TEST_TUEV('/home/eshuranov/projects/eeg_epileptiform_detection/EEG2Rep/Dataset/out_tuev_eval/row_list.csv')
    train_dataset  = TEST_TUEV('/home/eshuranov/projects/eeg_epileptiform_detection/EEG2Rep/Dataset/out_tuev/row_list.csv')


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=1,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_sz, shuffle=True, num_workers=1,
                                               drop_last=True)
    lr_d = 1e-3     #1e-6   #
    optim = torch.optim.AdamW(model.parameters(), lr=lr_d) #, weight_decay=1)



    steps = 0
    for epoch in range(2000):
        sum_loss = 0
        ii = 0
        for batch in train_loader:
            if batch['data'].numpy().min() == float('-inf') or batch['data'].numpy().max() == float('inf'):
                continue
            optim.zero_grad()
            output = model(batch['data'].to(device))

            loss = loss_function(output, batch['label'].float().to(device))

            sum_loss += loss.item()
            # print("tmp Epoch: ", epoch,"ii: ", ii, "loss: ", loss.item(), "sum_loss: ", sum_loss)
            ii +=1
            # if epoch>-1 and ii>1105:
            #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            loss.backward()
            optim.step()
            # scheduler.step()

        steps += 1
        print("Epoch: ",epoch,"sum_loss: ", sum_loss)
        if steps != 0 and steps % 10 == 0:
            try:
                with torch.no_grad():
                    sum_vloss = 0
                    for batch in train_loader:
                        if batch['data'].numpy().min() == float('-inf') or batch['data'].numpy().max() == float('inf'):
                            continue
                        output = model(batch['data'].to(device))
                        loss = loss_function(output, batch['label'].float().to(device))
                        sum_vloss += loss.item()
                    print('Val Loss: {}\t'.format(sum_loss))
                    Path = '/home/eshuranov/projects/eeg_epileptiform_detection' + str(epoch) + '_step' + str(steps) + '_of_' + str(int(len(train_loader))) + '_loss_' + str(sum_loss) + '_vloss_' + str(sum_vloss)+ '.pt'
                    torch.save(model.state_dict(), Path)
            except:
                raise



if __name__ == '__main__':
    print("main started")

    main()
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import csv
import random



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
        data = np.load(fname, allow_pickle=True)
        isnan = np.isnan(data)
        if sum(sum(isnan))>0:
            print("data:", data)

        sel_row = random.randint(0, data.shape[0]-1)
        label = int(self.data[idx]["label"])
        label_one_hot =torch.zeros(num_classes+1)
        # if label > 6:
        #     print(label)
        label_one_hot[label] = 1
        return {'data': torch.from_numpy(np.array(data[sel_row])).float(),
                'label': torch.from_numpy(np.array(label_one_hot)).long()}

def main2():
    import time
    from sklearn.metrics import accuracy_score
    start_time = time.time()

    n_epoch = 50

    train_dataset = TEST_TUEV('/home/eshuranov/projects/eeg_epileptiform_detection/EEG2Rep/Dataset/out_tuev/row_list.csv')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1,
                                               drop_last=True)

                                               # set_random_state(random_state)

    criterion = nn.CrossEntropyLoss()
    network = simple_EEGNet()

    # можно попробовать другой optimizer, тоже считается улучшением
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001)

    train_loss = []
    val_loss = []

    train_acc = []
    val_acc = []

    for e in range(n_epoch):
        print(f'epoch #{e + 1}')

        # train
        loss_list = []
        outputs = []
        targets = []
        for i_batch, sample_batched in enumerate(train_loader):

            x = sample_batched['data']
            y = sample_batched['label'].float()
            optimizer.zero_grad()

            output = network(x)
            outputs.append(output.argmax(axis=1))

            target = y.argmax( dim=1)
            targets.append(target)

            loss = criterion(output, target)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

        y_true = torch.hstack(targets).numpy()
        y_pred = torch.hstack(outputs).numpy()
        acc = accuracy_score(y_true, y_pred)

        train_loss.append(np.mean(loss_list))
        train_acc.append(acc)

        print(f'[train] mean loss: {train_loss[-1]}')
        print(f'[train] accuracy:  {acc}')

        loss_list = []
        outputs = []
        targets = []
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(train_loader):
                x = sample_batched['data']
                y = sample_batched['label'].float()
                # optimizer.zero_grad()

                output = network(x)
                outputs.append(output.argmax(axis=1))

                target = y.argmax(axis=1)
                targets.append(target)

                loss = criterion(output, target.long())
                loss_list.append(loss.item())
                # loss.backward()
                # optimizer.step()

            y_true = torch.hstack(targets).numpy()
            y_pred = torch.hstack(outputs).numpy()
            acc = accuracy_score(y_true, y_pred)

            val_loss.append(np.mean(loss_list))
            val_acc.append(acc)

            print(f'[val] mean loss:   {val_loss[-1]}')
            print(f'[val] accuracy:    {acc}', end="\n\n")

    print(f"Execution time: {(time.time() - start_time):.2f} seconds")

def main():

    model = simple_EEGNet().to(device)
    # input_data = torch.randn(16, 1, 2500)  # batch_size, channels, seq_length

    loss_function = nn.BCEWithLogitsLoss()
    batch_sz = 16

    # val_dataset  = TEST_TUEV('/home/eshuranov/projects/eeg_stud/EEG_validation.csv')
    train_dataset  = TEST_TUEV('/home/eshuranov/projects/eeg_epileptiform_detection/EEG2Rep/Dataset/out_tuev/row_list.csv')


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=1,
                                               drop_last=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_sz, shuffle=True, num_workers=1,
    #                                            drop_last=True)
    lr_d = 1e-6     #1e-6   #
    optim = torch.optim.AdamW(model.parameters(), lr=lr_d) #, weight_decay=1)

    steps = 0
    for epoch in range(20):
        sum_loss = 0
        for batch in train_loader:
            optim.zero_grad()
            output = model(batch['data'].to(device))

            loss = loss_function(output, batch['label'].float().to(device))

            sum_loss += loss.item()
            print("tmp Epoch: ", epoch, "sum_loss: ", sum_loss)
            loss.backward()
            optim.step()
            # scheduler.step()

        steps += 1
        print("Epoch: ",epoch,"sum_loss: ", sum_loss)
        if steps != 0 and steps % 5 == 0:
            try:
                with torch.no_grad():
                    sum_loss = 0
                    for batch in train_loader:
                        output = model(batch['data'].to(device))
                        loss = loss_function(output, batch['label'].float().to(device))
                        sum_loss += loss.item()
                    print('Val Loss: {}\t'.format(sum_loss))
            except:
                raise




if __name__ == '__main__':
    print("main started")
    main()
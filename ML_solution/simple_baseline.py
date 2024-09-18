# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
#
# import random
#
# from model_cnn_lstm_baseline import *
from dataloaders import *



device = "cuda:0" if torch.cuda.is_available() else "cpu"


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
                    for batch in val_loader:
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
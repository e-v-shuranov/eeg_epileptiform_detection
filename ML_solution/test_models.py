# from model_cnn_lstm_baseline import *
# import torch
# import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
from dataloaders import *

# import dataloaders

def short_check_results(model, test_loader,device):
    model.eval()
    preds = []
    reals = []
    loss_function = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        sum_vloss = 0
        for batch in test_loader:
            if batch['data'].numpy().min() == float('-inf') or batch['data'].numpy().max() == float('inf'):
                continue
            output = model(batch['data'].to(device))
            loss = loss_function(output, batch['label'].float().to(device))
            sum_vloss += loss.item()
            y_pred = output.argmax(dim=1)
            # y_pred = test_dataset.argmax(-1))[2][0] + +precision_recall_fscore_support(reals, preds.argmax(-1))[2][1]) / 2
            reals.extend(batch['label'].argmax(dim=1))
            preds.extend(y_pred)

        reals = np.array([i.tolist() for i in reals])
        preds = np.array([i.tolist() for i in preds])
        acc = np.sum(preds == reals) / preds.shape[0]  # Не сбалансированная!!
        all_scores = precision_recall_fscore_support(preds, reals)
        balensed_acc = balanced_accuracy_score(reals, preds)
        print('balensed_acc:', balensed_acc, 'acc:', acc, 'f1 precision, recall, fscore ,amount:', all_scores,'Val Loss: {}\t'.format(sum_vloss))
    return balensed_acc

def metrics_calcilation(model_path,device):
    batch_sz = 16
    val_dataset  = TEST_TUEV('/home/eshuranov/projects/eeg_epileptiform_detection/EEG2Rep/Dataset/out_tuev_eval/row_list.csv')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_sz, shuffle=True, num_workers=1,
                                             drop_last=True)
    train_dataset  = TEST_TUEV('/home/eshuranov/projects/eeg_epileptiform_detection/EEG2Rep/Dataset/out_tuev/row_list.csv')


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=1,
                                               drop_last=True)
    model = simple_EEGNet().to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    return short_check_results(model, train_loader,device)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = '/home/eshuranov/projects/eeg_epileptiform_detection1719_step1720_of_1216_loss_196.19630297645926.pt'
    res = metrics_calcilation(model_path, device)
    print("res: ",res)


if __name__ == '__main__':
    print("run test_models")
    main()
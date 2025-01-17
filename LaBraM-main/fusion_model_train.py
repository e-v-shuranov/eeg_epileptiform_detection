import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import utils
import os
from torch.optim.lr_scheduler import StepLR

class Tfusion_clf(nn.Module):
    def __init__(self, input_size=12, hidden_size1=16, hidden_size2=16, num_classes=6):
        super(Tfusion_clf, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        # self.fc3 = nn.Linear(hidden_size1, num_classes)

        # В данной простой сети в качестве функций активации используем ReLU

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # на выходе logits (для CrossEntropyLoss)
        return x


def train(train_continue = False, model_path=None, model_storage_path=None, binary_tests=True):
    model = Tfusion_clf()
    if train_continue:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay = 0.0005)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1, verbose=True)
    scheduler =torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=4e-3, step_size_up=20, step_size_down=100, cycle_momentum=False)
    batch_size = 96
    num_epochs = 500

    metrics = ["accuracy", "balanced_accuracy", "cohen_kappa"]

    # path_for_emb_storage = "/media/public/Datasets/TUEV/tuev/edf/emb_for_fusion_half_banana/train_emb xgb_modal level 4 on train only.pkl"
    # path_for_test_emb_storage = "/media/public/Datasets/TUEV/tuev/edf/emb_for_fusion_half_banana/test_emb xgb_modal level 4 on train only.pkl"
    #
    #
    path_for_emb_storage = "/media/public/Datasets/TUEV/tuev/edf/emb_for_fusion_half_banana/train_emb xgb_modal on train only.pkl"
    path_for_test_emb_storage = "/media/public/Datasets/TUEV/tuev/edf/emb_for_fusion_half_banana/test_emb xgb_modal on train only.pkl"

    with open(path_for_emb_storage, 'rb') as handle:
        emb_for_store_train = pickle.load(handle)

    with open(path_for_test_emb_storage, 'rb') as handle:
        emb_for_store_test = pickle.load(handle)

    train_x = torch.tensor(emb_for_store_train[0], dtype=torch.float32)
    train_y = torch.tensor(emb_for_store_train[1], dtype=torch.long)

    test_x = torch.tensor(emb_for_store_test[0], dtype=torch.float32)
    test_y = torch.tensor(emb_for_store_test[1], dtype=torch.long)

    with torch.no_grad():
        outputs = model(test_x)
        predicted = torch.argmax(outputs, dim=1)
        accuracy = (predicted == test_y).float().mean().item()

        outputs = model(train_x)
        predicted = torch.argmax(outputs, dim=1)
        train_accuracy = (predicted == train_y).float().mean().item()

        print(f"---Epoch [{0}/{num_epochs}], Loss: {0:.4f}, Accuracy: {accuracy * 100:.2f}% train_accuracy: {train_accuracy * 100:.2f}% ")


    # ---dataloder ---
    N = len(emb_for_store_train[0])
    balanced_accuracy_train = []
    balanced_accuracy_test = []

    for epoch in range(num_epochs):
        permutation = torch.randperm(N)

        for i in range(0, N, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = train_x[indices], train_y[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                outputs = model(test_x)
                predicted = torch.argmax(outputs, dim=1)
                accuracy = (predicted == test_y).float().mean().item()
                multiclas_ret_test = utils.get_metrics(predicted.numpy(), test_y.numpy(), metrics, True)

                binary_predicted = (predicted< 3).float()
                binary_target = (test_y < 3).float()
                binary_accuracy = (binary_predicted == binary_target).float().mean().item()
                ret = utils.get_metrics(binary_predicted.numpy(), binary_target.numpy(), metrics,True)


                outputs = model(train_x)
                predicted = torch.argmax(outputs, dim=1)
                train_accuracy = (predicted == train_y).float().mean().item()
                multiclas_ret_train = utils.get_metrics(predicted.numpy(), train_y.numpy(), metrics, True)

                binary_predicted = (predicted< 3).float()
                binary_target = (train_y < 3).float()
                train_binary_accuracy = (binary_predicted == binary_target).float().mean().item()
                train_ret = utils.get_metrics(binary_predicted.numpy(), binary_target.numpy(), metrics,True)
                if binary_tests:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], ", ret, "accuracy:", accuracy, "binary_accuracy:",
                          binary_accuracy,"train:", train_ret,"accuracy:", train_accuracy,"binary_accuracy:", train_binary_accuracy)
                    balanced_accuracy_train.append(train_ret['balanced_accuracy'])
                    balanced_accuracy_test.append(ret['balanced_accuracy'])
                else:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], ", multiclas_ret_test ,"train:", multiclas_ret_train)
                    balanced_accuracy_train.append(multiclas_ret_train['balanced_accuracy'])
                    balanced_accuracy_test.append(multiclas_ret_test['balanced_accuracy'])

                if model_storage_path is not None:
                    path = os.path.join(model_storage_path, 'checkpoint'+ str(epoch) +'.pth')
                    torch.save(model.state_dict(), path)
                # print("train:", train_ret,"accuracy:", train_accuracy,"binary_accuracy:", train_binary_accuracy)

                # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}% train_accuracy: {train_accuracy * 100:.2f}% ")

    get_amplitude(balanced_accuracy_test)
    get_amplitude(balanced_accuracy_train)

import numpy as np
import scipy
import matplotlib.pyplot as plt
def get_amplitude(x):
    plt.plot(x, color="blue")
    plt.show()
    print("max x:", max(x))
    return






def main():
    train_continue=False
    model_path=None
    model_storage_path="/media/public/ckpts/fussion"
    fusion_clf = train(model_storage_path=model_storage_path, binary_tests = False)
    with open("fusion_model.pkl", "wb") as file:
        pickle.dump(fusion_clf, file)

if __name__ == '__main__':
    main()
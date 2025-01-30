import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import utils
import os
from torch.optim.lr_scheduler import StepLR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight


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

def get_data():

    #     xgb_model_wav4.pkl
    # path_for_emb_storage = "/media/public/Datasets/TUEV/tuev/edf/emb_for_fusion_half_banana/train_emb xgb_modal level 4 on train only.pkl"
    # path_for_test_emb_storage = "/media/public/Datasets/TUEV/tuev/edf/emb_for_fusion_half_banana/test_emb xgb_modal level 4 on train only.pkl"
    #     xgb_model_train_only_check2.pkl
    # path_for_emb_storage = "/media/public/Datasets/TUEV/tuev/edf/emb_for_fusion_half_banana/train_emb xgb_modal on train only.pkl"
    # path_for_test_emb_storage = "/media/public/Datasets/TUEV/tuev/edf/emb_for_fusion_half_banana/test_emb xgb_modal on train only.pkl"
    #
    # path_for_emb_storage = "/media/public/Datasets/TUEV/tuev/edf/emb_for_fusion_half_banana/copy_labram_emb_train.pkl"
    # path_for_test_emb_storage = "/media/public/Datasets/TUEV/tuev/edf/emb_for_fusion_half_banana/copy_labram_emb_test.pkl"

    # xgb_model_4_level.pkl
    path_for_emb_storage = "/media/public/Datasets/TUEV/tuev/edf/emb_for_fusion_half_banana/train_emb xgb_modal level 4 from Konstantin.pkl"
    path_for_test_emb_storage = "/media/public/Datasets/TUEV/tuev/edf/emb_for_fusion_half_banana/test_emb xgb_modal level 4 from Konstantin.pkl"


    with open(path_for_emb_storage, 'rb') as handle:
        emb_for_store_train = pickle.load(handle)

    with open(path_for_test_emb_storage, 'rb') as handle:
        emb_for_store_test = pickle.load(handle)

    train_x = torch.tensor(emb_for_store_train[0], dtype=torch.float32)
    train_y = torch.tensor(emb_for_store_train[1], dtype=torch.long)

    test_x = torch.tensor(emb_for_store_test[0], dtype=torch.float32)
    test_y = torch.tensor(emb_for_store_test[1], dtype=torch.long)

    return train_x, train_y, test_x, test_y

def metrics_evaluation(model,test_x,test_y,train_x,train_y, epoch, num_epochs, current_lr, torchmodel = True):
    metrics = ["accuracy", "balanced_accuracy", "cohen_kappa"]
    outputs = model(test_x)
    if torchmodel:
        predicted = torch.argmax(outputs, dim=1)
        binary_predicted = (predicted < 3).float().numpy()
        predicted=predicted.numpy()
    else:
        predicted = outputs
        binary_predicted = (predicted < 3).astype(float)

    binary_target = (test_y < 3).float()
    multiclas_ret_test = utils.get_metrics(predicted, test_y.numpy(), metrics, True)
    ret = utils.get_metrics(binary_predicted, binary_target.numpy(), metrics, True)

    outputs = model(train_x)
    if torchmodel:
        predicted = torch.argmax(outputs, dim=1)
        binary_predicted = (predicted < 3).float().numpy()
        predicted = predicted.numpy()
    else:
        predicted = outputs
        binary_predicted = (predicted < 3).astype(float)
    binary_target = (train_y < 3).float()

    multiclas_ret_train = utils.get_metrics(predicted, train_y.numpy(), metrics, True)
    train_ret = utils.get_metrics(binary_predicted, binary_target.numpy(), metrics, True)

    # print(f"Epoch [{epoch + 1}/{num_epochs}], ", multiclas_ret_test, "train:", multiclas_ret_train, "bin test:", ret, "bin train:", train_ret)
    print(f"Epoch [{epoch + 1}/{num_epochs}], balanced_accuracy test:", multiclas_ret_test['balanced_accuracy'],
          "train:", multiclas_ret_train['balanced_accuracy'], "bin test:", ret['balanced_accuracy'], "bin train:",
          train_ret['balanced_accuracy'], "lr: ", current_lr)
    return multiclas_ret_test,ret,multiclas_ret_train,train_ret


def train(train_continue = False, model_path=None, model_storage_path=None):
    model = Tfusion_clf()
    if train_continue:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()


    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay = 0.000001)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1, verbose=True)
    scheduler =torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=4e-2, step_size_up=20, step_size_down=100, cycle_momentum=False)
    batch_size = 96
    num_epochs = 500

    train_x, train_y, test_x, test_y = get_data()

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_y.numpy()), y=train_y.numpy())
    class_weights = torch.Tensor(class_weights)
    # new_weights = torch.Tensor([22.4793,  1.7637,  2.2963, 12.6218,  1.1795,  0.2484])
    # new_weights = torch.Tensor([22.4793,  1.0,  1.5, 12.6218,  1.0,  0.02484])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # data_loader = DataLoader(list(zip(train_x, train_y)), batch_size=32, shuffle=True)

    with torch.no_grad():
        outputs = model(test_x)
        predicted = torch.argmax(outputs, dim=1)
        accuracy = (predicted == test_y).float().mean().item()

        outputs = model(train_x)
        predicted = torch.argmax(outputs, dim=1)
        train_accuracy = (predicted == train_y).float().mean().item()

        print(f"---Epoch [{0}/{num_epochs}], Loss: {0:.4f}, Accuracy: {accuracy * 100:.2f}% train_accuracy: {train_accuracy * 100:.2f}% ")


    # ---dataloder ---
    N = len(train_x)
    balanced_accuracy_train = []
    balanced_accuracy_test = []
    balanced_accuracy_train_binary = []
    balanced_accuracy_test_binary = []
    lr = []
    max_balanced_accuracy_test = 0

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
            current_lr = scheduler.get_last_lr()
        scheduler.step()

        if (epoch + 1) % 1 == 0:

            with torch.no_grad():
                multiclas_ret_test,ret,multiclas_ret_train,train_ret = metrics_evaluation(model, test_x, test_y, train_x, train_y, epoch, num_epochs, current_lr)

                balanced_accuracy_train_binary.append(train_ret['balanced_accuracy'])
                balanced_accuracy_test_binary.append(ret['balanced_accuracy'])
                balanced_accuracy_train.append(multiclas_ret_train['balanced_accuracy'])
                balanced_accuracy_test.append(multiclas_ret_test['balanced_accuracy'])
                lr.append(current_lr)
                if max_balanced_accuracy_test<multiclas_ret_test['balanced_accuracy']:
                    max_balanced_accuracy_test=multiclas_ret_test['balanced_accuracy']
                    max_acc_model = os.path.join(model_storage_path, 'checkpoint'+ str(epoch) +'.pth')

                if model_storage_path is not None:
                    path = os.path.join(model_storage_path, 'checkpoint'+ str(epoch) +'.pth')
                    torch.save(model.state_dict(), path)

    get_amplitude("multi test ",balanced_accuracy_test)
    get_amplitude("multi train ",balanced_accuracy_train)
    get_amplitude("bin test ",balanced_accuracy_test_binary)
    get_amplitude("bin train ",balanced_accuracy_train_binary)
    get_amplitude("lr",lr)
    if model_storage_path is not None:
        print("model with max acc: ", max_acc_model)
        model.load_state_dict(torch.load(max_acc_model, weights_only=True))
        model.eval()
        print("Test cm:")
        outputs = model(test_x)
        cm = confusion_matrix(test_y, torch.argmax(outputs, dim=1))
        print(cm)
        print("Train cm:")
        outputs = model(train_x)
        cm = confusion_matrix(train_y, torch.argmax(outputs, dim=1))
        print(cm)



import numpy as np
import scipy
import matplotlib.pyplot as plt
def get_amplitude(label,x):
    plt.plot(x, color="blue")
    plt.show()
    print(label," max", max(x))
    return




def xg_train():
    train_x, train_y, test_x, test_y = get_data()
    train_x[:,6:12]=0
    test_x[:,6:12]=0
    # train_x[:,10]=0
    # test_x[:,10]=0

    xgb_clf = GradientBoostingClassifier()
    xgb_clf.fit(train_x, train_y)

    # y_pred = xgb_clf.predict(test_x)

    with open("xgb_model_fussion.pkl", "wb") as file:
        pickle.dump(xgb_clf, file)

    multiclas_ret_test, ret, multiclas_ret_train, train_ret = metrics_evaluation(xgb_clf.predict, test_x, test_y, train_x,
                                                                                 train_y,0,0,0,torchmodel=False)

def main():
    # xg_train()
    #
    train_continue=False
    model_path=None
    model_storage_path="/media/public/ckpts/fussion"
    fusion_clf = train(model_storage_path=model_storage_path)
    with open("fusion_model.pkl", "wb") as file:
        pickle.dump(fusion_clf, file)

if __name__ == '__main__':
    main()
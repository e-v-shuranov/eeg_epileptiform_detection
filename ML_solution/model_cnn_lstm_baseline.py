import torch.nn as nn


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
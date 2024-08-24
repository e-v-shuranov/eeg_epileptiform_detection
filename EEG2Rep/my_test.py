import torch


class test_model(torch.nn.Module):
    def __init__(self, hidden_size=128, chnls=len(10)):
        super(test_model, self).__init__()

        self.hidden_size = hidden_size
        self.chnls = chnls
        self.output_size = hidden_size * chnls
        self.NClasters = 10
        self.test = torch.nn.Linear(self.hidden_size, 1)


        self.mask_embedding = torch.nn.Parameter(torch.normal(0, self.output_size ** (-0.5), size=(self.output_size,)),
                                                 requires_grad=True)

        self.negtive_classification = torch.nn.Sequential(
            torch.nn.Linear(self.output_size, 1),
        )

        self.upconvolution = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(in_channels=self.output_size, out_channels=512, kernel_size=11, stride=1,
                                     padding=5),
            torch.nn.ReLU6(),
            torch.nn.ConvTranspose1d(in_channels=512, out_channels=chnls * 10, kernel_size=9,  stride=1,
                                     padding=4),
            torch.nn.ReLU6(),
            torch.nn.ConvTranspose1d(in_channels=chnls * 10, out_channels=512, kernel_size=9,  stride=1,
                                     padding=4),
            torch.nn.ReLU6(),
            torch.nn.ConvTranspose1d(in_channels=512, out_channels=self.output_size, kernel_size=7,  stride=1,
                                     padding=3),
            torch.nn.ReLU6(),
            torch.nn.LayerNorm(self.output_size)
        )

        self.eeg_classification = torch.nn.Sequential(
            torch.nn.Linear(self.output_size, 2),
        )

        self.HuBert_classification = torch.nn.Sequential(
            torch.nn.Linear(self.output_size, self.NClasters),
        )


    def forward(self, input):
        out = self.test(input)
        return out



if __name__ == '__main__':
    data_x = ran

    model = test_model()

    for

import torch.nn as nn
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


class BayesianLSTM(PyroModule):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BayesianLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = PyroModule[nn.LSTM](input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm.weight_ih_l0 = PyroSample(dist.Normal(0., 1.).expand([4 * hidden_dim, input_dim]).to_event(2))
        self.lstm.weight_hh_l0 = PyroSample(dist.Normal(0., 1.).expand([4 * hidden_dim, hidden_dim]).to_event(2))
        self.lstm.bias_ih_l0 = PyroSample(dist.Normal(0., 1.).expand([4 * hidden_dim]).to_event(1))
        self.lstm.bias_hh_l0 = PyroSample(dist.Normal(0., 1.).expand([4 * hidden_dim]).to_event(1))
        self.fc_std = PyroModule[nn.Linear](hidden_dim, output_dim)
        self.fc_std.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, hidden_dim]).to_event(2))
        self.fc_std.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))

        self.c0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_dim))
        self.h0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_dim))
        self.fc = PyroModule[nn.Linear](hidden_dim, output_dim)
        self.fc.weight = PyroSample(dist.Normal(0., 1.).expand([output_dim, hidden_dim]).to_event(2))
        self.fc.bias = PyroSample(dist.Normal(0., 1.).expand([output_dim]).to_event(1))

    def forward(self, x, labels=None, h0=None, c0=None):
        batch_size = x.size(0)  # Get the batch size from the input data and convert it to an integer

        h0 = self.h0.repeat(1, batch_size, 1)
        c0 = self.c0.repeat(1, batch_size, 1)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        output = self.fc(out[:, -1, :])

        # Assuming a fixed standard deviation for the output distribution
        output_std = torch.tensor(0.1)

        obs = pyro.sample("obs", dist.Normal(output, output_std), obs=labels)
        return output


input_dim = 1
hidden_dim = 48
num_layers = 2
output_dim = 1
model = BayesianLSTM(input_dim, hidden_dim, num_layers, output_dim)


def getModel():
    return model

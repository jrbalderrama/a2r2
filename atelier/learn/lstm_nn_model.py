import torch
from torch.nn import LSTM, Linear, Module


class LSTMModel(Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout):
        super().__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = LSTM(
            input_dim,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=dropout,
        )

        # Fully connected layer
        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, x):
        # initializing hidden state for first input with zeros
        h0 = torch.zeros(
            self.layer_dim,
            x.size(0),
            self.hidden_dim,
        ).requires_grad_()

        # initializing cell state for first input with zeros
        c0 = torch.zeros(
            self.layer_dim,
            x.size(0),
            self.hidden_dim,
        ).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        # (squeezing is equivalent to: `out = out[:, -1, :]`)
        out = torch.squeeze(out)

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

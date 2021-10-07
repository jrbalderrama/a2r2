import numpy as np
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


# Helper to train the NN model
class RunnerHelper:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, X, y):

        # set model to train mode
        self.model.train()

        # make predictions
        ŷ = self.model(X)

        # compute loss
        loss = self.loss_fn(ŷ, y)

        # compute gradients
        loss.backward()

        # update parameters
        self.optimizer.step()

        # reset to zero gradients
        self.optimizer.zero_grad()

        # returns loss
        return loss.item()

    def val_step(self, X, y):

        # set model to eval mode
        self.model.eval()

        # make prediction
        ŷ = self.model(X)

        # compute loss
        loss = self.loss_fn(ŷ, y)

        # return loss
        return loss.item()

    def train(self, train_loader, val_loader, n_epochs):
        for epoch in range(1, n_epochs + 1):
            batch_train_losses = []
            for x_train, y_train in train_loader:
                x_train = torch.unsqueeze(x_train, 1)
                train_loss = self.train_step(x_train, y_train)
                batch_train_losses.append(train_loss)

            training_loss = np.mean(batch_train_losses)
            self.train_losses.append(training_loss)
            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = torch.unsqueeze(x_val, 1)
                    val_loss = self.val_step(x_val, y_val)
                    batch_val_losses.append(val_loss)

                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 20 == 0):
                print(
                    f"[{epoch:3d}/{n_epochs}] "
                    f"Training loss: {training_loss:.4f}"
                    f"\t Validation loss: {validation_loss:.4f}"
                )

    def evaluate(self, test_loader):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = torch.unsqueeze(x_test, 1)
                self.model.eval()
                ŷ = self.model(x_test)
                predictions.append(ŷ.detach().numpy())
                values.append(y_test.detach().numpy())

        return predictions, values

    def get_losses(self):
        return self.train_losses, self.val_losses

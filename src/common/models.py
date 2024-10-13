################################################################################
#
#                                   Torch Models
#
################################################################################
# Rather than native PyTorch, we'll use PyTorch Lightning to define our models.
# This is a high-level library that abstracts away a lot of the boilerplate.
# It also makes it easier to run experiments and log metrics.

# Libraries
import torch
import lightning as L

################################################################################
# Name classification task


class RNNClassifier(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        output_size: int,
        epochs: int,
        data_length: int,
        unit_type: str = "gru",
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.unit_type = unit_type.lower()
        self.epochs = epochs
        self.data_length = data_length
        self.num_layers = num_layers
        self.dropout = dropout

        # Layers
        self.embedding = torch.nn.Embedding(input_size, embedding_size)

        if self.unit_type == "gru":
            self.rnn = torch.nn.GRU(
                embedding_size,
                hidden_size,
                batch_first=True,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        elif self.unit_type == "lstm":
            self.rnn = torch.nn.LSTM(
                embedding_size,
                hidden_size,
                batch_first=True,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        elif self.unit_type == "rnn":
            self.rnn = torch.nn.RNN(
                embedding_size,
                hidden_size,
                batch_first=True,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )

        # Fully connected layer. This takes us from the hidden size to the
        # output size. Note that we could just set the hidden size to the output
        # size, but if our vocabulary is large, the number of parameters will
        # explode.
        self.fc = torch.nn.Linear(
            hidden_size,
            output_size,
        )
        
        # Activation. Use log_softmax to get log probabilities.
        # The last dimension is our output dimension.
        self.activation = torch.nn.LogSoftmax(dim=-1)

        # If we're using negative log-likelihood loss, we need to use log
        # softmax as the final layer of the model.
        self.criterion = torch.nn.NLLLoss()

        return None

    def forward(self, x, hidden=None):
        """
        Forward pass through the model.
        """

        x = self.embedding(x)

        # Note that the LSTM is special. It returns output and a tuple of hidden
        # and cell states. We only need the hidden state.
        if self.unit_type == "lstm":
            x, (hidden, cell) = self.rnn(x, hidden)
        else:
            x, hidden = self.rnn(x, hidden)

        x = self.fc(x)
        x = self.activation(x)

        return x, hidden

    def _model_step(self, batch, batch_idx):
        """
        Generic step for training, validation, and testing.
        """
        # Pull out inputs and targets from the batch.
        x, y, name, country = batch

        # Forward pass through RNN.
        output, hidden = self.forward(x)

        # Note that our output is size (batch, seq_len, num_classes)
        # Our prediction is the output corresponding to the last
        # sequence within a batch.
        pred = output[:, -1, :]

        # This is now shape (batch, num_classes). The second dimension
        # got squeezed automatically.

        # Note that y is shape (batch, 1). We need to squeeze it to
        # (batch,) to use NLLLoss.
        target = y.squeeze()

        loss = self.criterion(pred, target)
        accuracy = (pred.argmax(1) == target).float().mean()

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._model_step(batch, batch_idx)

        # Metrics to log during training.
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
            }
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Runs automatically at the end of each epoch.
        """

        loss, accuracy = self._model_step(batch, batch_idx)
        self.log_dict(
            {
                "validation_loss": loss,
                "validation_accuracy": accuracy,
            }
        )

        return None

    def test_step(self, batch, batch_idx):
        loss, accuracy = self._model_step(batch, batch_idx)
        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": accuracy,
            }
        )

        return None

    def configure_optimizers(self):
        """
        We'll use the Adam optimizer with a one-cycle learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            steps_per_epoch=self.data_length,
            epochs=self.epochs,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


################################################################################

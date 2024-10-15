################################################################################
#
#                                   Torch Models
#
################################################################################
# Rather than native PyTorch, we'll use PyTorch Lightning to define our models.
# This is a high-level library that abstracts away a little bit of the
# boilerplate, and  makes it easier to run experiments and log metrics.

# This file contains classes for:
# - A simple RNN classifier for name classification.
# - Encoder-Decoder model for machine translation.


# Libraries
import torch
import lightning as L
from typing import Callable

from common.language import Language

from common.utils import get_logger

logger = get_logger(__name__)

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


class EncoderDecoder(L.LightningModule):
    def __init__(
        self,
        source_language: Language,
        target_language: Language,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        output_size: int,
        epochs: int,
        data_length: int,
        max_output_length: int = 20,
        scheduler: str = "onecycle",
    ):
        super().__init__()

        # Parameters
        self.source_language = source_language
        self.target_language = target_language
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.epochs = epochs
        self.data_length = data_length
        self.max_output_length = max_output_length
        self.scheduler = scheduler

        self.criterion = torch.nn.NLLLoss()

        assert self.scheduler in ["onecycle", "reduceonplateau"]
        assert isinstance(self.source_language, Language)
        assert isinstance(self.target_language, Language)

        # Embedding
        self.input_embedding = torch.nn.Embedding(
            self.input_size, self.embedding_size
        )
        self.output_embedding = torch.nn.Embedding(
            self.output_size, self.embedding_size
        )

        # Encoder
        self.encoder = torch.nn.GRU(
            self.embedding_size,
            self.hidden_size,
            batch_first=True,
        )

        # Not that we apply a linear transformation to the context vector
        # This is probably not necessary if the hidden size of the encoder
        # and decoder are the same. But maybe the tanh activation helps?
        self.context_transform = torch.nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )

        self.tanh = torch.nn.Tanh()

        # Decoder
        # The input to the decoder will be a concat of the
        self.decoder = torch.nn.GRU(
            self.embedding_size + self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        # Dense layer to convert decoder output to vocab size
        self.dense1 = torch.nn.Linear(self.hidden_size, self.embedding_size)

        self.dense2 = torch.nn.Linear(self.embedding_size, self.output_size)

        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def encoder_step(self, x):
        """
        Push inputs through encoder.
        """

        # Shapes are:
        # x: (batch_size, seq_len)
        # decoder_input: (batch_size, seq_len-1)
        # decoder_target: (batch_size, seq_len-1)

        # Construct embeddings
        x = self.input_embedding(x)
        # x: (batch_size, seq_len, embedding_size)

        # Run the encoder
        encoder_output, context_vector = self.encoder(x)
        # output: (batch_size, seq_len, hidden_size)
        # context_vector: (1, batch_size, hidden_size)

        # Run the context vector through a linear layer
        context_vector = self.context_transform(context_vector)
        # context_vector: (1, batch_size, hidden_size)

        # Apply tanh activation
        context_vector = self.tanh(context_vector)

        return context_vector

    def decoder_step(self, decoder_input, context_vector, decoder_state=None):
        """
        Push inputs through decoder.
        """

        # Embed target outputs
        decoder_input = self.output_embedding(decoder_input)
        # decoder_input: (batch_size, seq_len-1, embedding_size)

        # Currently the context vector is (1, batch_size, hidden_size)
        # decoder_input is (batch_size, seq_len-1, embedding_size)
        # Rather than construct a special RNN module that can handle two inputs,
        # we're simply going to concatenate the context vector to the decoder input.

        # Permute the dimensions of context vector to be conformable with decoder_input
        context_vector = context_vector.permute(1, 0, 2)
        # context_vector: (batch_size, 1, hidden_size)

        # Make copies of the context vector along the sequence length demiension
        context_vector = context_vector.repeat(1, decoder_input.shape[1], 1)
        # context_vector: (batch_size, seq_len-1, hidden_size)

        decoder_input = torch.cat([decoder_input, context_vector], dim=2)
        # decoder_input: (batch_size, seq_len-1, embedding_size + hidden_size)

        # Now run the decoder_input through the decoder
        decoder_output, decoder_state = self.decoder(
            decoder_input, decoder_state
        )
        # decoder_output: (batch_size, seq_len-1, hidden_size)

        # decoder_output is (batch_size, seq_len-1, hidden_size)
        # Add a dense layer to convert it to the decoder_output vocab size
        decoder_output = self.dense1(decoder_output)
        # decoder_output: (batch_size, seq_len-1, embedding_size)

        decoder_output = self.dense2(decoder_output)
        # decoder_output: (batch_size, seq_len-1, output_size)

        # Now log softmax the decoder_output along last dimension
        decoder_output = self.log_softmax(decoder_output)
        # output: (batch_size, seq_len-1, output_size)

        return decoder_output, decoder_state

    def _model_step(self, batch):
        """
        Generic step for training, validation, and testing.
        """

        x, y, name, country = batch

        # First thing we need to do is transform our target tensor. We need two
        # versions of it:
        # - One that has the <EOS> token removed; this is the input to the
        #   decoder.
        # - One that has the <SOS> token removed; this is the teacher-forced
        #   target for the decoder.
        decoder_input = y[:, :-1]
        decoder_target = y[:, 1:]

        # Get context vector
        context_vector = self.encoder_step(x)

        # Get decoder output. Remember the initial hidden state for the decoder
        # is the context vector.
        decoder_output, _ = self.decoder_step(
            decoder_input, context_vector, context_vector
        )
        # decoder_output: (batch_size, seq_len-1, output_size)

        # To use our NLL loss, we need to reshape the output and target
        # The outputs need to be (N, class_size) and the targets (N)
        # So flatten the batch and sequence dimensions.
        decoder_output = decoder_output.reshape(-1, self.output_size)
        decoder_target = decoder_target.reshape(-1)

        # output: (batch_size * seq_len-1, output_size)
        # decoder_target: (batch_size * seq_len-1, )

        loss = self.criterion(decoder_output, decoder_target)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._model_step(batch)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._model_step(batch)

        self.log("validation_loss", loss)

        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        if self.scheduler == "onecycle":
            # One cycle learning rate scheduler
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=0.01,
                steps_per_epoch=self.data_length,
                epochs=self.epochs,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }

        if self.scheduler == "reduceonplateau":
            # Reduce on plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                verbose=True,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "validation_loss",
            }

        return None

    def inference(self, name: str):
        """
        This is where our tensors are not precomputed, but we need to run the
        model on a single example.

        We have to tokenize the input (according to the provided language
        model), run the encoder to generate the context vector; and then run the
        decoder iteratively to generate the output, starting by feeding it the
        <SOS> token. Then we need to detokenize the output to get the final
        string.
        """
        assert isinstance(name, str)

        # Tokenize the input
        input_tokens = self.source_language.tokenizer(name)

        # Add <SOS> and <EOS> tokens
        input_tokens = ["<SOS>"] + input_tokens + ["<EOS>"]

        input_indices = self.source_language.token_to_index(input_tokens)

        x = torch.tensor(input_indices).unsqueeze(0).long()
        # x should have shape (1, seq_len)
        assert x.shape[0] == 1

        # Get context vector
        context_vector = self.encoder_step(x)

        # Now we need to run the decoder
        # We'll start with a start token and no state
        decoder_input = torch.tensor(
            [self.target_language.token_to_index("<SOS>")]
        ).unsqueeze(0)
        decoder_state = context_vector
        # decoder_input: (1, 1) and it's a long

        # We'll keep track of the output
        output = []

        reached_eos = False
        for i in range(self.max_output_length):
            # Run the decoder
            decoder_output, decoder_state = self.decoder_step(
                decoder_input, context_vector, decoder_state
            )
            # decoder_output: (1, 1, output_size)

            # Get the most likely token
            token = torch.argmax(decoder_output, dim=-1)
            # token: (1, 1)

            # Append to the output
            output.append(token.squeeze(0).item())

            # If we've reached the end of the sentence, break
            if token.item() == self.target_language.token_to_index("<EOS>"):
                reached_eos = True
                break

            # Update the decoder input
            decoder_input = token

        if not reached_eos:
            logger.info("Failed to reach EOS token")

        # Convert the output to a string
        output = [self.target_language.index_to_token(x) for x in output]
        # If the last token is <EOS> remove it
        if output[-1] == "<EOS>":
            output = output[:-1]

        # Detokenize the output using provided function
        output = self.target_language.detokenizer(output)

        return output

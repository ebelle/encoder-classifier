import torch
import torch.nn as nn
import torch.nn.functional as F

# Classifier
class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        enc_hid_dim,
        num_layers,
        enc_dropout,
        bidirectional,
        pad_idx,
    ):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.enc_embedding = nn.Embedding(
            input_dim, emb_dim, padding_idx=pad_idx, sparse=True
        )

        #  LSTM
        self.rnn = nn.LSTM(
            emb_dim,
            enc_hid_dim,
            num_layers=num_layers,
            dropout=enc_dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(enc_dropout)

    def forward(self, x, input_lengths):
        # Convert input_sequence to embeddings
        x = self.dropout(self.enc_embedding(x))
        # Pack the sequence of embeddings
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths)

        # Run packed embeddings through the RNN, and then unpack the sequences
        x = self.rnn(x)[0]

        x = nn.utils.rnn.pad_packed_sequence(x)[0]
        # outputs = [src len, batch size, hid dim * num directions]

        # The ouput of a RNN has shape (seq_len, batch, hidden_size * num_directions)
        # Because the Encoder is bidirectional, combine the results from the forward
        # and reverse
        if self.bidirectional:
            # outputs = [src len, batch size, hid dim]
            x = x[:, :, : self.enc_hid_dim] + x[:, :, self.enc_hid_dim :]

        # hidden = [n layers * num directions, batch size, hid dim]
        return x


class Decoder(nn.Module):
    def __init__(
        self, enc_hid_dim, dec_hid_dim, output_dim, dec_dropout, bidirectional, pad_idx,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dec_dropout)

        self.hidden_layer = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.final_out = nn.Linear(dec_hid_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, encoder_outputs):

        x = self.hidden_layer(encoder_outputs)
        x = self.dropout(x)

        x = self.final_out(x)
        x = self.softmax(x)
        return x


class Classifier(nn.Module):
    def __init__(
        self,
        new_state_dict,
        input_dim,
        emb_dim,
        enc_hid_dim,
        dec_hid_dim,
        output_dim,
        num_layers,
        enc_dropout,
        dec_dropout,
        bidirectional,
        pad_idx,
        freeze_encoder=False,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim,
            emb_dim,
            enc_hid_dim,
            num_layers,
            enc_dropout,
            bidirectional,
            pad_idx,
        )
        # load data from pre-trained encoder
        self.encoder.load_state_dict(new_state_dict)
        # optionally freeze encoder
        if freeze_encoder == True:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.decoder = Decoder(
            enc_hid_dim, dec_hid_dim, output_dim, dec_dropout, bidirectional, pad_idx
        )

    def forward(self, src, src_len):

        # this feeds the encoder output directly into the decoder
        # and returns the decoder output (predictions)
        return self.decoder(self.encoder(src, src_len))

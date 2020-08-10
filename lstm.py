import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(
        self, input_dim, emb_dim, hid_dim, num_layers, dropout, bidirectional, pad_idx
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.enc_embedding = nn.Embedding(
            input_dim, emb_dim, padding_idx=pad_idx, sparse=True
        )

        #  LSTM
        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_lengths):
        # Convert input_sequence to embeddings
        x = self.dropout(self.enc_embedding(x))
        # Pack the sequence of embeddings
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths)

        # Run packed embeddings through the RNN, and then unpack the sequences
        x, (hidden, cell) = self.rnn(x)

        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        # outputs = [src len, batch size, hid dim * num directions]

        # The ouput of a RNN has shape (seq_len, batch, hidden_size * num_directions)
        # Because the Encoder is bidirectional, combine the results from the forward
        # and reverse
        if self.bidirectional:
            # outputs = [src len, batch size, hid dim]
            x = x[:, :, : self.hid_dim] + x[:, :, self.hid_dim :]

        # hidden = [n layers * num directions, batch size, hid dim]
        return x, hidden, cell


class Attention(nn.Module):
    def __init__(self, hid_dim, dropout, bidirectional):
        super().__init__()

        self.hid_dim = hid_dim
        self.bidirectional = bidirectional
        self.fc = nn.Linear(hid_dim * 2 if self.bidirectional else hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def dot_score(self, hidden_state, encoder_states):
        return torch.sum(hidden_state * encoder_states, dim=2)

    def forward(self, hidden, encoder_outputs, mask):

        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            hidden = hidden.unsqueeze(0)
        hidden = torch.tanh(self.dropout(self.fc(hidden)))

        attn = self.dot_score(hidden, encoder_outputs)
        del (hidden, encoder_outputs)
        # Transpose max_length and batch_size dimensions
        attn.t_()
        # Apply mask so network does not attend <pad> tokens
        attn = attn.masked_fill(mask == 0, -1e10)
        # Softmax over attention scores
        attn = F.softmax(attn, dim=1)
        return attn.unsqueeze(1)


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        emb_dim,
        hid_dim,
        num_layers,
        dropout,
        bidirectional,
        attention,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.hid_dim = hid_dim

        self.dec_embedding = nn.Embedding(output_dim, emb_dim, sparse=True)

        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, word_input, hidden, cell, encoder_outputs, mask):

        # word_input = [batch size]
        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # mask = [batch size, src len]
        word_input = word_input.unsqueeze(0)
        # word_input = [1, batch size]
        word_input = self.dropout(self.dec_embedding(word_input))
        # word_embedded = [1, batch size, emb dim]
        # Run embedded input word and hidden through RNN
        word_input, (hidden, cell) = self.rnn(word_input, (hidden, cell))
        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn = self.attention(hidden, encoder_outputs, mask)
        attn = attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        if self.bidirectional == True:
            word_input = (
                word_input[:, :, : self.hid_dim] + word_input[:, :, self.hid_dim :]
            )

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        word_input = word_input.squeeze(0)  # S=1 x B x N -> B x N
        attn = attn.squeeze(1)  # B x S=1 x N -> B x N
        word_input = F.log_softmax(
            self.fc_out(torch.cat((word_input, attn), dim=1)), dim=1
        )
        # Return final output, hidden state
        return word_input, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        hid_dim,
        output_dim,
        num_layers,
        dropout,
        bidirectional,
        src_pad_idx,
        device,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim, emb_dim, hid_dim, num_layers, dropout, bidirectional, src_pad_idx
        )
        self.attention = Attention(hid_dim, dropout, bidirectional)
        self.decoder = Decoder(
            output_dim,
            emb_dim,
            hid_dim,
            num_layers,
            dropout,
            bidirectional,
            self.attention,
        )
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio):

        # src = [src len, batch size]
        # src_len = [batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden, cell = self.encoder(src, src_len)

        # first input to the decoder is the <sos> tokens
        inputs = trg[0, :]
        mask = self.create_mask(src)
        # mask = [batch size, src len]
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            # receive output tensor (predictions) and new hidden state
            output, hidden, cell = self.decoder(
                inputs, hidden, cell, encoder_outputs, mask
            )
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            inputs = trg[t] if teacher_force else top1
        return outputs

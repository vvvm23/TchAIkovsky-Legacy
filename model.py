import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.cuda.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerModel(nn.Module):
    def __init__(self, nb_out, nb_in, nb_heads, nb_hidden, nb_layers, dropout=0.3, device=torch.cuda.device('cuda' if torch.cuda.is_available else 'cpu')):
        super(TransformerModel, self).__init__()
        
        self.device = device
        self.nb_in = nb_in
        self.src_mask = None

        self.position_encoder = PositionalEncoding(nb_in, dropout=dropout)
        self.embedding = nn.Embedding(nb_out, nb_in)

        encoder_layers = nn.TransformerEncoderLayer(nb_in, nb_heads, nb_hidden, dropout)
        self.trans_encoder = nn.TransformerEncoder(encoder_layers, nb_layers)

        self.decoder = nn.Linear(nb_in, nb_out)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    def _generate_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, mask=True):
        if mask:
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                # mask = self._generate_mask(len(x)).to(self.device)
                mask = self._generate_mask(len(x)).to(self.device)
                self.src_mask = mask
        else:
            self.src_mask = None
        
        x = self.embedding(x) * math.sqrt(self.nb_in)
        x = self.position_encoder(x)
        x = self.trans_encoder(x, self.src_mask)
        out = self.decoder(x)

        return F.log_softmax(out, dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, nb_in, dropout=0.1, max_length=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, nb_in)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, nb_in, 2).float() * (-math.log(10000.0) / nb_in))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, nb_in, nb_emd, nb_heads, nb_hidden, nb_layers, dropout=0.1, device=torch.cuda.device('cuda:0' if torch.cuda.is_available else 'cpu')):
        super(TransformerModel, self).__init__()
        
        self.device = device
        self.nb_in = nb_in
        self.tgt_mask = None

        self.emd = nn.Embedding(nb_in, nb_emd)
        torch.nn.init.kaiming_normal_(self.emd.weight)

        self.pos = PositionalEncoding(nb_emd, dropout=dropout)

        self.transformer = nn.Transformer(nb_emd, nb_heads, nb_layers, nb_layers, nb_hidden, dropout)
        for p in self.transformer.parameters():
            torch.nn.init.kaiming_normal_(p.data)
            break

        self.linear = nn.Linear(nb_emd, nb_in)
        torch.nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, src, tgt):
        if self.tgt_mask == None: 
            self.tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(self.device)

        src_pad_mask = torch.zeros_like(src).type(torch.BoolTensor).to(self.device)
        src_pad_mask[src == 0] = True

        tgt_pad_mask = torch.zeros_like(tgt).type(torch.BoolTensor).to(self.device)
        tgt_pad_mask[tgt == 0] = True

        src = self.emd(src) * torch.sqrt(torch.tensor(self.nb_in).float())
        tgt = self.emd(tgt) * torch.sqrt(torch.tensor(self.nb_in).float())

        src = self.pos(src)
        tgt = self.pos(tgt)

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        out = self.transformer(src, tgt, tgt_mask=self.tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = out.permute(1, 0, 2)
        out = self.linear(out)
        # out = F.log_softmax(out, dim=-1)
        return out

class TransformerEncoderModel(nn.Module):
    def __init__(self, nb_out, nb_in, nb_heads, nb_hidden, nb_layers, dropout=0.1, device=torch.cuda.device('cuda:0' if torch.cuda.is_available else 'cpu')):
        super(TransformerEncoderModel, self).__init__()
        
        self.device = device
        self.nb_in = nb_in
        self.src_mask = None

        self.position_encoder = PositionalEncoding(nb_in, dropout=dropout)
        self.embedding = nn.Embedding(nb_out, nb_in)

        encoder_layers = nn.TransformerEncoderLayer(nb_in, nb_heads, nb_hidden, dropout)
        self.trans_encoder = nn.TransformerEncoder(encoder_layers, nb_layers)

        self.decoder = nn.Linear(nb_in, nb_out)

    def _generate_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def forward(self, x, mask=True):
        if mask:
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                mask = self._generate_mask(x.shape[1])
                self.src_mask = mask
        else:
            self.src_mask = None
        
        # x = self.embedding(x) * math.sqrt(self.nb_in)
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.nb_in).float())
        x = self.position_encoder(x)
        x = x.transpose(0, 1)
        x = self.trans_encoder(x, self.src_mask)
        x = x.transpose(1, 0)

        out = self.decoder(x)
    
        out = F.log_softmax(out, dim=-1)

        return out

class PositionalEncoding(nn.Module):
    def __init__(self, nb_in, dropout=0.1, max_length=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, nb_in)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, nb_in, 2).float() * (-math.log(10000.0) / nb_in))
        div_term = torch.exp(torch.arange(0, nb_in, 2).float() * (-torch.log(torch.tensor(10000.0)) / nb_in))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = TransformerModel(200, 512, 8, 1024, 6, device=device)
    print(x)

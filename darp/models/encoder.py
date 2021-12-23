import torch.nn as nn
from .transformer_block import TransformerBlock
import torch

from utils import  quinconx


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        typ,
        encoder_bn
    ):

        super(Encoder, self).__init__()
        self.typ = typ
        self.embed_size = embed_size
        self.device = device
        # self.word_embedding = nn.Embedding(50000, embed_size)
        self.word_embedding = nn.Embedding(10000, embed_size)
        # self.position_embedding1 = nn.Embedding(src_vocab_size, embed_size//2)
        # self.position_embedding2 = nn.Embedding(src_vocab_size, embed_size//2)
        if self.typ in [11, 12]:
            self.position_embedding = nn.Linear(self.embed_size, self.embed_size).to(self.device)
        else :
            self.position_embedding = nn.Embedding(src_vocab_size, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    batch_norm=encoder_bn
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, positions=None, times=None, layers_out=1, out_process=1):
        N, seq_length, _ = x.shape
        if positions is None:
            positions = torch.tensor([0 for i in range(seq_length-1)] + [1]).expand(N, seq_length).to(self.device)

        if self.typ in [9, 10] :
            postim = quinconx([positions.to(self.device), times.to(self.device)], d=2)
        elif self.typ in [11, 12]:
            postim = self.position_embedding(torch.cat([positions.to(self.device), times.to(self.device)], dim=-1))
        elif self.typ in [1,2,3,4,5,6,7,8, 13, 14, 15, 16, 17, 18, 19, 26]:
            postim = positions.to(self.device) + times.to(self.device)

        out = self.dropout(
            (x.to(self.device) + postim)
        )
        #self.word_embedding(

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        saving_layers = []
        for i, layer in enumerate(self.layers):
            out = layer(out, out, out, mask)
            if len(self.layers) - 1 - i < layers_out :
                saving_layers.append(out)

        if layers_out == 1:
            return out
        elif out_process==1:
            return torch.cat(saving_layers, dim=-1)
        elif out_process==2:
            return torch.sum(torch.stack(saving_layers), dim=0)

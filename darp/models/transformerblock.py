import torch.nn as nn
import torch
from encoder import Encoder

class TransformerBlock(nn.Module):
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

        super(TransformerBlock, self).__init__()
        self.typ = typ
        self.embed_size = embed_size
        self.device = device
        # TODO check num_embedding, set randomly to 10000
        self.word_embedding = nn.Embedding(10000, embed_size)

        # thibaut version
        # self.position_embedding = nn.Embedding(src_vocab_size, embed_size)
        # version from tutorial
        self.position_embedding = nn.Embedding(max_length, embed_size)
        # arrange all the modules for encoder
        self.layers = nn.ModuleList(
            [
                Encoder(
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

    # forward method for encoder:
    # perform some data processing of the input, add input embedding and positional embedding,
    # pass through layers of encoder block
    def forward(self, x, mask, positions=None, times=None, layers_out=1, out_process=1):
        # todo wonder why we need more embedding also here, if it is already done in Trans18 class
        N, seq_length, _ = x.shape
        if positions is None:
            positions = torch.tensor([0 for i in range(seq_length - 1)] + [1]).expand(N, seq_length).to(self.device)

        # add position and time embedding
        postim = positions.to(self.device) + times.to(self.device)
        # x is environment embedding
        out = self.dropout(
            (x.to(self.device) + postim)
        )

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        saving_layers = []
        for i, layer in enumerate(self.layers):
            out = layer(out, out, out, mask)
            if len(self.layers) - 1 - i < layers_out:
                saving_layers.append(out)

        if layers_out == 1:
            return out
        elif out_process == 1:
            return torch.cat(saving_layers, dim=-1)
        elif out_process == 2:
            return torch.sum(torch.stack(saving_layers), dim=0)

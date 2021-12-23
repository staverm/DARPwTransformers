import torch.nn as nn
from .attention import SelfAttention


class Encoder(nn.Module):
    """
    Encoder block for transformer.
    """
    def __init__(self, embed_size, heads, dropout, forward_expansion, batch_norm=False):
        super(Encoder, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.norm1 = nn.BatchNorm1d(embed_size)
            self.norm2 = nn.BatchNorm1d(embed_size)
        else:
            self.norm1 = nn.LayerNorm(embed_size)
            self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # self-attention
        attention = self.attention(value, key, query, mask)

        # add and normalize
        x = (attention + query)
        if self.batch_norm:
            x = x.permute(1, 2, 0).contiguous()
        x = self.dropout(self.norm1(x))
        if self.batch_norm:
            x = x.permute(2, 0, 1).contiguous()

        # feed forward
        forward = self.feed_forward(x)

        # add and normalize
        x = forward + x
        if self.batch_norm:
            x = x.permute(1, 2, 0).contiguous()
        out = self.dropout(self.norm2(x))
        if self.batch_norm:
            out = out.permute(2, 0, 1).contiguous()
        return out

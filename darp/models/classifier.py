import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    Classifier built for darp problem. Consists of only Linear layer.
    """
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            classifier_type,
            dropout,
            max_length,
            typ,
            decoder_bn
    ):

        super(Classifier, self).__init__()
        self.typ = typ
        self.embed_size = embed_size
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.device = device
        self.classifier_type = classifier_type
        self.classifier_expansion = 4
        self.max_length = max_length

        self.fc_out = nn.Linear(self.embed_size * max_length, self.trg_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc_out(x.flatten(start_dim=1))  # OUT: bsz, trg_sz
        return out.squeeze(1)

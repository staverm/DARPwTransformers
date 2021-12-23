import torch.nn as nn
from .transformer_block import TransformerBlock
import torch


class Classifier(nn.Module):
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

        if self.classifier_type in [2]:
            self.classifier_expansion = 8
        elif self.classifier_type in [3] :
            self.fc_out_out = nn.Linear(max_length, 1)
        elif self.classifier_type in [4] :
            self.fc_out_out = nn.Linear(self.embed_size, self.trg_vocab_size)
        elif self.classifier_type in [5] :
            self.fc_out = nn.Linear(self.embed_size, self.trg_vocab_size)
            self.fc_out_out = nn.Linear(max_length, 1)
        elif self.classifier_type in [6, 7] :
            self.fc_out = nn.Linear(self.embed_size, self.trg_vocab_size)
        elif self.classifier_type in [8, 12] :
            self.fc_out = nn.Linear(self.embed_size * max_length, self.trg_vocab_size)
        elif self.classifier_type in [9]:
            self.trans1 = TransformerBlock(
                self.embed_size,
                heads,
                dropout=dropout,
                forward_expansion=self.classifier_expansion,
                batch_norm=decoder_bn
            )
            self.fc_out = nn.Linear(self.embed_size * max_length, self.trg_vocab_size)
        elif self.classifier_type in [10]:
            self.fc_out = nn.Linear(self.embed_size * 4, self.trg_vocab_size)
            self.fc_out_out = nn.Linear(max_length, 1)
        elif self.classifier_type in [11] :
            self.fc_out = nn.Linear(self.embed_size * 4 * max_length, self.trg_vocab_size)


        self.fc1 = nn.Linear(self.embed_size, self.classifier_expansion * self.embed_size )
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(self.embed_size * self.classifier_expansion, trg_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, trg=None, mask=None):
        N, seq_length, emb = x.shape

        if self.classifier_type in [4]:
            xx = self.fc_out_out(x)
            output = xx[:,0,:]
            return output

        elif self.classifier_type in [5, 10]:
            xx = self.fc_out(x)
            xx = self.ReLU(xx.permute(0, 2, 1))
            xx = self.fc_out_out(xx)
            output = xx.squeeze(-1)
            return output

        elif self.classifier_type in [6]:
            out = self.fc_out(x)    # IN: bsz, seq_l, emb_size
            current_driver_indice = trg
            out = torch.gather(out, 1, current_driver_indice.unsqueeze(-1).expand(-1, -1, out.shape[-1]))
            return out.squeeze(1)
        elif self.classifier_type in [7]:
            out = self.fc_out(x)    # out: bsz, seq_l, trg_sz
            current_driver_indice = trg +  2*(self.trg_vocab_size-1)
            out = torch.gather(out, 1, current_driver_indice.unsqueeze(-1).expand(-1, -1, out.shape[-1]))
            return out.squeeze(1)
        elif self.classifier_type in [8, 11, 12]:
             # IN: bsz, seq_l, embedding_sz
             # flatten: bsz, seq_l * embedding_sz
            out = self.fc_out(x.flatten(start_dim=1)) #OUT: bsz, trg_sz
            return out.squeeze(1)
        elif self.classifier_type in [9]:
            x = self.trans1(x, x, x, mask)
            out = self.fc_out(x.flatten(start_dim=1)) #OUT: bsz, trg_sz
            return out.squeeze(1)


        intermediate = self.ReLU(self.fc1(x))
        output = self.fc2(self.dropout(intermediate))
        if self.classifier_type in [3]:
            xx = self.ReLU(output.permute(0, 2, 1))
            xx = self.fc_out_out(xx)
            output = xx.squeeze(-1)
        elif self.classifier_type in [1, 2]:
            output = output[:,0,:]

        return output
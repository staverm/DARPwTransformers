import math
import torch
import torch.nn as nn
from .encoder import Encoder
from .classifier import Classifier

from utils import get_device, plotting, quinconx
from .classifier import Classifier
from .encoder import Encoder


class Trans18(nn.Module):
    """
    Transformer as described in the report. As input takes position, time and environment embedding vectors. Uses
    <code>num_layers</code> to determine the number of encoder blocks. The end classifier corresponds to classical
    neural network.
    """
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx=0,  # 0
            trg_pad_idx=0,  # 0
            embed_size=128,  # 128
            num_layers=6,  # 6
            forward_expansion=4,  # 4
            heads=8,  # 8
            dropout=0,  # 0
            extremas=None,
            device="",
            max_length=100,  # 100
            typ=None,
            max_time=2000,
            classifier_type=1,
            encoder_bn=False,
            decoder_bn=False
    ):

        super(Trans18, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoders = nn.ModuleList(
            Encoder(
                embed_size,
                heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
                batch_norm=encoder_bn
            )
            for _ in range(num_layers)
        )

        self.classifier = Classifier(
            src_vocab_size,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            self.device,
            classifier_type,
            dropout,
            max_length,
            typ,
            decoder_bn
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.src_vocab_size = src_vocab_size
        self.max_length = max_length
        self.max_time = max_time
        self.classifier_type = classifier_type

        # Boxing stuff
        self.extremas = extremas
        self.siderange = int(math.sqrt(self.src_vocab_size))

        self.embed_size = embed_size
        mapping_size = embed_size // 2
        scale = 10
        self.B_gauss = torch.normal(0, 1, size=(mapping_size, 2)) * scale

        self.pe = self.generate_positional_encoding(self.embed_size, max_len=1000)

        self.trg_vocab_size = trg_vocab_size
        self.typ = typ

        self.pos_embedding = nn.Embedding(src_vocab_size, embed_size)

        # ENVIRONMENT EMBEDDING
        self.ind_embedding1 = nn.Embedding(100, self.embed_size)
        self.ind_embedding2 = nn.Embedding(100, self.embed_size // 2)
        self.ind_embedding3 = nn.Embedding(100, self.embed_size // 4)
        self.ind_embedding33 = nn.Embedding(100, self.embed_size // 4)
        self.ind_embedding4 = nn.Linear(4 + 3, self.embed_size)  # 4 pour info et 3 pour le trunk

        # POSITION EMBEDDING
        # Driver, pickup, dropoff
        self.input_emb1 = nn.Linear(2, self.embed_size).to(self.device)
        self.input_emb2 = nn.Linear(2, self.embed_size).to(self.device)
        self.input_emb3 = nn.Linear(2, self.embed_size).to(self.device)

        # TIME EMBEDDING
        self.time_embedding1 = nn.Embedding(self.max_time * 2 + 1, self.embed_size)
        self.time_embedding2 = nn.Embedding(self.max_time * 2 + 1, self.embed_size // 2)


    def summary(self):
        txt = '***** Model Summary *****\n'
        txt += ' - This model is a Transformer with an input of the decoder that loops the old outputs \n'
        txt += '\t the default sizes are he ones proposed by the original paper.\n'
        txt += ''
        print(txt)

    def make_src_mask(self, src):
        # Little change towards original: src got embedding dimention in additon
        src_mask = (src[:, :, 0] != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg, positions, times):
        environment = self.env_encoding(src)
        if not positions is None:
            positions = self.positional_encoding(positions)
        if not times is None:
            times = self.times_encoding(times)

        src_mask = self.make_src_mask(environment)

        src = environment.to(self.device)
        dropout = nn.Dropout(self.dropout)
        src = dropout(src)
        for i, encoder in enumerate(self.encoders):
            src = encoder(src, src, src, src_mask)  # (value, key, query, mask)


        if self.classifier_type in [6, 7]:
            classification = self.classifier(src, trg=trg)
        elif self.classifier_type in [9]:
            classification = self.classifier(src, mask=src_mask)
        else:
            classification = self.classifier(src)

        return classification

    def generate_positional_encoding(self, d_model, max_len):
        """
        From xbresson
        Create standard transformer PEs.
        Inputs :
          d_model is a scalar correspoding to the hidden dimension
          max_len is the maximum length of the sequence
        Output :
          pe of size (max_len, d_model), where d_model=dim_emb, max_len=1000
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def fourier_feature(self, coordonates):
        # coordonates = torch.stack(coordonates).permute(1, 0)
        pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2 * 2)
        x = (pi * coordonates).double()
        transB = torch.transpose(self.B_gauss, 0, 1).double()
        if x.shape[1] == 4:
            transB = torch.cat([transB, transB])
        x_proj = x.matmul(transB)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)


    def quinconx(self, l):
        nb = len(l)
        if nb == 2:
            a, b = l
            return torch.cat([a.unsqueeze(-1), b.unsqueeze(-1)], dim=-1).flatten(start_dim=1)
        elif nb == 3:
            a, b, c = l
            q1 = torch.cat([b.unsqueeze(-1), c.unsqueeze(-1)], dim=-1).flatten(start_dim=1)
            return torch.cat([a.unsqueeze(-1), q1.unsqueeze(-1)], dim=-1).flatten(start_dim=1)

    def env_encoding(self, src):
        w, ts, ds = src

        # World
        world_emb = [self.ind_embedding1(w[0].long().to(self.device))]

        # Targets
        targets_emb = []
        for target in ts:
            # bij_id = 2 + target[0] + (target[0] - 1)*10 + (target[1]+2)
            # stack the embedding of 2 data points
            em1 = self.ind_embedding2((target[0] + self.trg_vocab_size + target[1] + 2).long().to(self.device))
            em2 = self.ind_embedding3((target[1] + 2 + target[2] * 10).long().to(self.device))
            em3 = self.ind_embedding3((target[0] + target[2] * 10).long().to(self.device))
            di1 = self.ind_embedding33((target[3]).long().to(self.device))
            di2 = self.ind_embedding33((target[4]).long().to(self.device))

            targets_emb.append(self.quinconx([em1, em2, di1]))
            targets_emb.append(self.quinconx([em1, em3, di2]))

        drivers_emb = [self.ind_embedding4(torch.stack(driver, dim=-1).double().to(self.device)) for driver in ds]

        final_emb = torch.stack(world_emb + targets_emb + drivers_emb)

        return final_emb.permute(1, 0, 2)

    def positional_encoding(self, position):
        # bsz = position[0][0].shape[-2]
        depot_position = position[0]
        targets_pickups = [pos[:, 0:2] for pos in position[1]]
        targets_dropoff = [pos[:, 2:] for pos in position[1]]
        drivers = [pos for pos in position[2]]

        # World
        d1 = [torch.stack([self.input_emb1(depot_position.double().to(self.device))])]

        # Targets
        for pick, doff in zip(targets_pickups, targets_dropoff):
            d2 = torch.stack([self.input_emb2(pick.double().to(self.device))])
            d25 = torch.stack([self.input_emb3(doff.double().to(self.device))])
            d1.append(d2)
            d1.append(d25)

        # Drivers
        for driver in drivers:
            d3 = torch.stack([self.input_emb1(driver.double().to(self.device))])
            d1.append(d3)
        d1 = torch.cat(d1)

        return d1.permute(1, 0, 2)

    def times_encoding(self, times):
        bsz = times[0].shape[-1]
        current_time = times[0]
        targets_4d = times[1]
        drivers_1d = times[2]

        # World
        d1 = [self.time_embedding1(current_time.long().to(self.device)).unsqueeze(0)]

        # Targets
        for target in targets_4d:
            em1 = self.time_embedding2((target[:, 0] - current_time + self.max_time).to(self.device).long())
            em2 = self.time_embedding2((target[:, 1] - current_time + self.max_time).to(self.device).long())
            d2 = torch.stack([self.quinconx([em1, em2])])
            em3 = self.time_embedding2((target[:, 2] - current_time + self.max_time).to(self.device).long())
            em4 = self.time_embedding2((target[:, 3] - current_time + self.max_time).to(self.device).long())
            d22 = torch.stack([self.quinconx([em3, em4])])

            d1.append(d2)
            d1.append(d22)

        # Drivers
        for driver in drivers_1d:
            d3 = torch.stack([self.time_embedding1((driver).long().to(self.device))])
            d1.append(d3)

        d1 = torch.cat(d1)

        return d1.permute(1, 0, 2)

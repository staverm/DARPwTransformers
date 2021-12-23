import math
import torch
import torch.nn as nn
from .encoder import Encoder
from .classifier import Classifier

from utils import get_device


class Trans18(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx=0,         #0
        trg_pad_idx=0,          #0
        embed_size=128,         #128
        num_layers=6,           #6
        forward_expansion=4,    #4
        heads=8,                #8
        dropout=0,              #0
        extremas=None,
        device="",
        max_length=100,         #100
        typ=None,
        max_time=2000,
        classifier_type=1,
        encoder_bn=False,
        decoder_bn=False
    ):

        super(Trans18, self).__init__()
        self.device = device #get_device()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            self.device,
            forward_expansion,
            dropout,
            max_length,
            typ,
            encoder_bn
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
        # self.boxh = abs(self.extremas[2] - self.extremas[0]) / self.siderange
        # self.boxw = abs(self.extremas[3] - self.extremas[1]) / self.siderange

        self.embed_size = embed_size
        mapping_size = embed_size // 2
        scale = 10
        self.B_gauss = torch.normal(0, 1, size=(mapping_size, 2)) * scale

        self.pe = self.generate_positional_encoding(self.embed_size, max_len=1000)

        self.trg_vocab_size = trg_vocab_size
        self.typ = typ

        self.pos_embedding = nn.Embedding(src_vocab_size, embed_size)

        # ENVIRONMENT EMBEDDING
        if self.typ in [1, 2, 5]:
            self.ind_embedding = nn.Embedding(100, self.embed_size)
        elif self.typ in [6, 8, 9, 10, 11, 12, 13]:
            self.ind_embedding1 = nn.Embedding(100, self.embed_size)
            self.ind_embedding2 = nn.Embedding(100, self.embed_size // 2)
            self.ind_embedding3 = nn.Embedding(100, self.embed_size // 2)
        elif self.typ in [14, 15] :
            self.ind_embedding1 = nn.Embedding(100, self.embed_size)
            self.ind_embedding2 = nn.Embedding(100, self.embed_size // 2)
            self.ind_embedding3 = nn.Embedding(100, self.embed_size // 2)
            self.ind_embedding4 = nn.Linear(4 + 3, self.embed_size)         # 4 pour info et 3 pour le trunk
        elif self.typ in [16, 17, 18, 19, 26] :
            self.ind_embedding1 = nn.Embedding(100, self.embed_size)
            self.ind_embedding2 = nn.Embedding(100, self.embed_size // 2)
            self.ind_embedding3 = nn.Embedding(100, self.embed_size // 4)
            self.ind_embedding33 = nn.Embedding(100, self.embed_size // 4)
            self.ind_embedding4 = nn.Linear(4 + 3, self.embed_size)         # 4 pour info et 3 pour le trunk
        elif self.typ in [7]:
            self.ind_embedding1 = nn.Embedding(100, 8)
            self.ind_embedding11 = nn.Linear(8, self.embed_size)
            self.ind_embedding2 = nn.Embedding(100, 8)
            self.ind_embedding3 = nn.Embedding(100, 8)
            self.ind_embedding22 = nn.Linear(16, self.embed_size)
        elif self.typ in [3]:
            self.ind_embedding = nn.Linear(2, self.embed_size)
        elif self.typ in [4]:
            # driver, #target pickup, #target dropoff
            self.ind_embedding1 = nn.Linear(2, self.embed_size).to(self.device)
            self.ind_embedding2 = nn.Linear(2, self.embed_size).to(self.device)
            self.ind_embedding3 = nn.Linear(2, self.embed_size).to(self.device)

        # POSITION EMBEDDING
        if self.typ in [1, 3]:
            self.input_emb = nn.Linear(2, self.embed_size)
        elif self.typ in [2]:
            self.input_emb = nn.Linear(self.embed_size, self.embed_size).to(self.device)
        elif self.typ in [4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 26] :
            # Driver, pickup, dropoff
            self.input_emb1 = nn.Linear(2, self.embed_size).to(self.device)
            self.input_emb2 = nn.Linear(2, self.embed_size).to(self.device)
            self.input_emb3 = nn.Linear(2, self.embed_size).to(self.device)
        elif self.typ in [9, 10, 11, 12] :
            # Driver, pickup, dropoff
            self.input_emb1 = nn.Linear(2, self.embed_size//2).to(self.device)
            self.input_emb2 = nn.Linear(2, self.embed_size//2).to(self.device)
            self.input_emb3 = nn.Linear(2, self.embed_size//2).to(self.device)

        # TIME EMBEDDING
        if self.typ in [1, 2, 3, 4, 5, 6, 7]:
            self.time_embedding1 = nn.Embedding(self.max_time, self.embed_size)
            self.time_embedding2 = nn.Embedding(self.max_time, self.embed_size // 2)
        elif self.typ in [13, 14, 15, 16, 17, 18, 19, 26]:
            self.time_embedding1 = nn.Embedding(self.max_time * 2 + 1, self.embed_size)
            self.time_embedding2 = nn.Embedding(self.max_time * 2 + 1, self.embed_size // 2)
        elif self.typ in [8]:
            self.time_embedding1 = nn.Linear(2, self.embed_size).to(self.device)
            self.time_embedding2 = nn.Linear(2, self.embed_size).to(self.device)
            self.time_embedding3 = nn.Linear(2, self.embed_size).to(self.device)
        elif self.typ in [10, 11]:
            self.time_embedding1 = nn.Linear(2, self.embed_size // 2).to(self.device)
            self.time_embedding2 = nn.Linear(2, self.embed_size // 2).to(self.device)
            self.time_embedding3 = nn.Linear(2, self.embed_size // 2).to(self.device)
        elif self.typ in [9, 12]:
            self.time_embedding1 = nn.Embedding(self.max_time, self.embed_size // 2)
            self.time_embedding2 = nn.Embedding(self.max_time, self.embed_size // 4)


    def summary(self):
        txt = '***** Model Summary *****\n'
        txt += ' - This model is a Transformer with an input of the decoder that loops the old outputs \n'
        txt += '\t the default sizes are he ones proposed by the original paper.\n'
        txt += ''
        print(txt)

    def make_src_mask(self, src):
        # Little change towards original: src got embedding dimention in additon
        src_mask = (src[:,:,0] != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        if self.typ in [17, 18, 19]:
            N, trg_len, _ = trg[1].shape
        else :
            N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg, positions, times):
        # Encoode positoin and env
        nb_targets = len(positions[1])
        nb_drivers = len(positions[2])
        src = self.env_encoding(src)
        if not positions is None :
            positions = self.positional_encoding(positions)
        if not times is None :
            times = self.times_encoding(times)

        src_mask = self.make_src_mask(src)

        encodded_layers_out = 1
        out_process=1
        if self.classifier_type in [10, 11, 12]:
            encodded_layers_out = 4
        if self.classifier_type in [12]:
            out_process = 2

        enc_src = self.encoder(src, src_mask, positions=positions, times=times, layers_out=encodded_layers_out, out_process=out_process)#[:, :nb_targets])

        if self.classifier_type in [6, 7]:
            classification = self.classifier(enc_src, trg=trg)
        elif self.classifier_type in [9] :
            classification = self.classifier(enc_src, mask=src_mask)
        else :
            classification = self.classifier(enc_src)

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
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
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

    # def bidim2int(self, coordonate):
    #     h = abs(coordonate[:,0] - self.extremas[0]) / self.boxh
    #     w = abs(coordonate[:,1] - self.extremas[1]) / self.boxw
    #     return h.add(w * self.siderange).long()

    def quinconx(self, l):
        nb = len(l)
        if nb==2:
            a, b = l
            return torch.cat([a.unsqueeze(-1), b.unsqueeze(-1)], dim=-1).flatten(start_dim=1)
        elif nb==3:
            a, b, c = l
            q1 = torch.cat([b.unsqueeze(-1), c.unsqueeze(-1)], dim=-1).flatten(start_dim=1)
            return torch.cat([a.unsqueeze(-1), q1.unsqueeze(-1)], dim=-1).flatten(start_dim=1)




    def env_encoding(self, src):
        w, ts, ds = src
        embeddig_size = self.embed_size
        bsz = w[0].shape[-1]

        # World
        if self.typ in [1, 2, 5]:
            world_emb = [self.ind_embedding(w[0].long().to(self.device))]
        elif self.typ in [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26]:
            world_emb = [self.ind_embedding1(w[0].long().to(self.device))]
        elif self.typ in [7]:
            world_emb = [self.ind_embedding11(self.ind_embedding1(w[0].long().to(self.device)))]
        elif self.typ in [3]:
            world_emb = [self.ind_embedding(torch.stack([w[0], w[0]], dim=-1).double().to(self.device))]
        elif self.typ in [4]:
            world_emb = [self.ind_embedding1(torch.stack([w[0], w[0]], dim=-1).double().to(self.device))]
        else :
            raise "Nah"

        # Targets
        targets_emb = []
        for target in ts :
            # bij_id = 2 + target[0] + (target[0] - 1)*10 + (target[1]+2)
            if self.typ in [1, 2]:
                targets_emb.append(self.ind_embedding((target[0] + self.trg_vocab_size + target[1]+2).long().to(self.device)))
                targets_emb.append(self.ind_embedding((target[0]).long().to(self.device)))
            elif self.typ in [5]:
                # embed with primitive bijection
                targets_emb.append(self.ind_embedding((target[0] + self.trg_vocab_size + target[1]+2).long().to(self.device)))
                targets_emb.append(self.ind_embedding((target[0] + self.trg_vocab_size + target[1]+2).long().to(self.device)))
            elif self.typ in [6, 8, 9, 10, 11, 12, 13, 14]:
                # stack the embedding of 2 data points
                em1 = self.ind_embedding2((target[0] + self.trg_vocab_size + target[1]+2).long().to(self.device))
                em2 = self.ind_embedding3((target[1]+2).long().to(self.device))
                em3 = self.ind_embedding3((target[0]).long().to(self.device))

                targets_emb.append(self.quinconx([em1, em2]))
                targets_emb.append(self.quinconx([em1, em3]))

            elif self.typ in [15]:
                # stack the embedding of 2 data points
                em1 = self.ind_embedding2((target[0] + self.trg_vocab_size + target[1]+2).long().to(self.device))
                em2 = self.ind_embedding3((target[1]+2 + target[2]*10).long().to(self.device))
                em3 = self.ind_embedding3((target[0] + target[2]*10).long().to(self.device))

                targets_emb.append(self.quinconx([em1, em2]))
                targets_emb.append(self.quinconx([em1, em3]))

            elif self.typ in [16, 17, 18, 19, 26]:
                # stack the embedding of 2 data points
                em1 = self.ind_embedding2((target[0] + self.trg_vocab_size + target[1]+2).long().to(self.device))
                em2 = self.ind_embedding3((target[1]+2 + target[2]*10).long().to(self.device))
                em3 = self.ind_embedding3((target[0] + target[2]*10).long().to(self.device))
                di1 = self.ind_embedding33((target[3]).long().to(self.device))
                di2 = self.ind_embedding33((target[4]).long().to(self.device))

                targets_emb.append(self.quinconx([em1, em2, di1]))
                targets_emb.append(self.quinconx([em1, em3, di2]))

            elif self.typ in [7]:
                # stack the embedding of 2 data points
                em1 = self.ind_embedding2((target[0] + self.trg_vocab_size + target[1]+2).long().to(self.device))
                em2 = self.ind_embedding3((target[1]+2).long().to(self.device))
                em3 = self.ind_embedding3((target[0]).long().to(self.device))
                em = self.ind_embedding22(self.quinconx([em1, em2])).double().to(self.device)
                targets_emb.append(em)
                em = self.ind_embedding22(self.quinconx([em1, em3])).double().to(self.device)
                targets_emb.append(em)
            elif self.typ in [3]:
                targets_emb.append(self.ind_embedding(torch.stack([target[0], target[1]+2], dim=-1).double().to(self.device)))
                targets_emb.append(self.ind_embedding(torch.stack([target[0], target[0]], dim=-1).double().to(self.device)))
            elif self.typ in [4]:
                targets_emb.append(self.ind_embedding2(torch.stack([target[0], target[1]+2], dim=-1).double().to(self.device)))
                targets_emb.append(self.ind_embedding3(torch.stack([target[0], target[1]+2], dim=-1).double().to(self.device)))
            else :
                raise "Nah"

        # Drivers
        if self.typ in [1, 2, 5]:
            drivers_emb = [self.ind_embedding(driver[0].long().to(self.device)) for driver in ds]
        elif self.typ in [6, 8, 9, 10, 11, 12, 13]:
            drivers_emb = [self.ind_embedding1(driver[0].long().to(self.device)) for driver in ds]

        elif self.typ in [14, 15, 16, 17, 18, 19, 26]:
            # em1 = [self.ind_embedding1(driver[0].long().to(self.device)) for driver in ds]

            drivers_emb = [self.ind_embedding4(torch.stack(driver, dim=-1).double().to(self.device)) for driver in ds]
            # drivers_emb = [self.quinconx(em1, em2)]

        elif self.typ in [7]:
            drivers_emb = [self.ind_embedding11(self.ind_embedding1(driver[0].long().to(self.device))) for driver in ds]
        elif self.typ in [3]:
            drivers_emb = [self.ind_embedding(torch.stack([driver[0], driver[0]], dim=-1).double().to(self.device)) for driver in ds]
        elif self.typ in [4]:
            drivers_emb = [self.ind_embedding1(torch.stack([driver[0], driver[0]], dim=-1).double().to(self.device)) for driver in ds]
        else :
            raise "Nah"


        final_emb = torch.stack(world_emb + targets_emb + drivers_emb)

        return final_emb.permute(1, 0, 2)




    def positional_encoding(self, position):
        # bsz = position[0][0].shape[-2]
        depot_position = position[0]
        targets_pickups = [pos[:, 0:2] for pos in position[1]]
        targets_dropoff = [pos[:, 2:] for pos in position[1]]
        drivers = [pos for pos in position[2]]

        # World
        if self.typ in [1, 3]:
            d1 = [torch.stack([self.input_emb(depot_position.to(self.device))])]
        elif self.typ in [2]:
            d1 = [torch.stack([self.input_emb(self.fourier_feature(depot_position).to(self.device))])]
        elif self.typ in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26]:
            d1 = [torch.stack([self.input_emb1(depot_position.double().to(self.device))])]
        else :
            raise "Na"

        # Targets
        for pick, doff in zip(targets_pickups, targets_dropoff) :
            if self.typ in [1, 3]:
                d2 = torch.stack([self.input_emb(pick.to(self.device))])
                d25 = torch.stack([self.input_emb(doff.to(self.device))])
            elif self.typ in [2]:
                d2 = torch.stack([self.input_emb(self.fourier_feature(pick).to(self.device))])
                d25 = torch.stack([self.input_emb(self.fourier_feature(doff).to(self.device))])
            elif self.typ in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26]:
                d2 = torch.stack([self.input_emb2(pick.double().to(self.device))])
                d25 = torch.stack([self.input_emb3(doff.double().to(self.device))])
            else :
                raise "Nah"
            d1.append(d2)
            d1.append(d25)

        # Drivers
        for driver in drivers :
            if self.typ in [1, 3]:
                d3 = torch.stack([self.input_emb(driver.to(self.device))])
            elif self.typ in [2]:
                d3 = torch.stack([self.input_emb(self.fourier_feature(driver).to(self.device))])
            elif self.typ in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26]:
                d3 = torch.stack([self.input_emb1(driver.double().to(self.device))])
            else :
                raise "Nah"
            d1.append(d3)
        d1 = torch.cat(d1)

        return d1.permute(1, 0, 2)


    def times_encoding(self, times):
        bsz = times[0].shape[-1]
        current_time = times[0]
        targets_4d = times[1]
        drivers_1d = times[2]

        # World
        if self.typ in [1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 15, 16, 17,  18, 19, 26]:
            d1 = [self.time_embedding1(current_time.long().to(self.device)).unsqueeze(0)]
        elif self.typ in [8, 10, 11] :
            d1 = [self.time_embedding1(torch.stack([current_time, current_time], dim=-1).double().to(self.device))]

        # Targets
        for target in targets_4d :
            if self.typ in [1, 2, 3, 4, 5, 6, 7, 9, 12]:
                em1 = self.time_embedding2(target[:, 0].to(self.device).long())
                em2 = self.time_embedding2(target[:, 1].to(self.device).long())
                d2 = torch.stack([self.quinconx([em1, em2])])
                em3 = self.time_embedding2(target[:, 2].to(self.device).long())
                em4 = self.time_embedding2(target[:, 3].to(self.device).long())
                d22 = torch.stack([self.quinconx([em3, em4])])

            elif self.typ in [13, 14, 15, 16, 17, 18, 19, 26]:
                em1 = self.time_embedding2((target[:, 0] - current_time + self.max_time).to(self.device).long())
                em2 = self.time_embedding2((target[:, 1] - current_time + self.max_time).to(self.device).long())
                d2 = torch.stack([self.quinconx([em1, em2])])
                em3 = self.time_embedding2((target[:, 2] - current_time + self.max_time).to(self.device).long())
                em4 = self.time_embedding2((target[:, 3] - current_time + self.max_time).to(self.device).long())
                d22 = torch.stack([self.quinconx([em3, em4])])

            elif self.typ in [8, 10, 11]:
                d2 = self.time_embedding2(torch.stack([target[:, 0], target[:, 1]], dim=-1).double().to(self.device))
                d22 = self.time_embedding2(torch.stack([target[:, 2], target[:, 3]], dim=-1).double().to(self.device))
            else :
                raise "Nah"
            d1.append(d2)
            d1.append(d22)

        # Drivers
        for driver in drivers_1d :
            if self.typ in [1, 2, 3, 4, 5, 6, 7, 9, 12]:
                d3 = torch.stack([self.time_embedding1(driver.long().to(self.device))])
            elif self.typ in [13, 14, 15, 16, 17, 18, 19, 26]:
                d3 = torch.stack([self.time_embedding1((driver).long().to(self.device))])
            elif self.typ in [8, 10, 11]:
                d3 = self.time_embedding3(torch.stack([driver, current_time], dim=-1).double().to(self.device))
            else :
                raise "Nah"
            d1.append(d3)

        if self.typ in [1,2,3,4,5,6,7, 9, 12, 13, 14, 15, 16, 17, 18, 19, 26]:
            d1 = torch.cat(d1)
        elif self.typ in [8, 10, 11]:
            d1 = torch.stack(d1)

        return d1.permute(1, 0, 2)



def example_usage():
    device = get_device()
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 5, 6, 4, 3, 9, 5, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(
        device
    )
    trg = torch.tensor([[0], [0]]).to(device)

    positions = [
                torch.tensor([[0.1,0.1],[0.1,0.1]]),
                torch.tensor([[[0.2, 0.2]], [[0.2, 0.2]]]),
                torch.tensor([[[0.3, 0.3, 0.4, 0.4], [1.1, 1.1, 1.2, 1.2]], [[0.3, 0.3, 0.4, 0.4], [1.1, 1.1, 1.2, 1.2]]])
                ]


    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Trans18(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1], positions=positions, times=None)
    print(out)
    print(out.argmax(-1))
    print(out.shape)
 

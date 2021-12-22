from torch.utils.data import Dataset
import torch
import numpy as np

class MemoryDataset(Dataset):
    """ Customed Dataset class for our Instances data
    """
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        """ Returns a couple (image, neares 1hot)"""
        instance = self.data[idx]
        return instance


class SupervisionDataset(Dataset):
    """ Customed Dataset class for our Instances data
    """
    def __init__(self, data_list, typ=None, augment=None):
        # super(SupervisionDataset, self).__init__(data_list)
        self.data = data_list
        self.augment = augment
        self.typ = typ


    def __len__(self):
        return len(self.data)


    def pos_augmentation(self, positions, rand_vect, dilate=None):
        ''' Transposing all popsition information by a 2D vectors in [-1,1]²
            In addition the 2D points are rotated by a random value [-pi, pi]
        '''
        depot, targets, drivers = positions

        get_rot = lambda theta: torch.tensor([[torch.cos(theta*np.pi), -torch.sin(theta*np.pi)],
                                [torch.sin(theta*np.pi), torch.cos(theta*np.pi)]]).double()

        rand_vect = rand_vect * 2 - 1
        rot = get_rot(torch.tensor(rand_vect[2]))
        trsp = rand_vect[:2]
        mirror = rand_vect[3]

        comp_aug2d = lambda pos: (torch.matmul(pos, rot) + trsp) * dilate if torch.is_tensor(pos) else (torch.matmul(torch.from_numpy(pos), rot) + trsp) * dilate
        comp_aug4d = lambda pos: torch.cat([torch.matmul(pos[:2], rot) + trsp,
                                            torch.matmul(pos[2:], rot) + trsp]) * dilate if torch.is_tensor(pos) else torch.cat([torch.matmul(torch.from_numpy(pos[:2]), rot) + trsp,
                                                                                                                                 torch.matmul(torch.from_numpy(pos[2:]), rot) + trsp]) * dilate

        depot = comp_aug2d(depot)
        targets = list(map(comp_aug4d, targets))
        drivers = list(map(comp_aug2d, drivers))

        return [depot, targets, drivers]


    def time_augmentation(self, time, rand_vect, dilate=None):
        ''' Shifting all time information by a random amount
            This transposition value is in [0, 10]
            half of the time it's rounded to an integer
        '''
        current, targets, drivers = time
        rand1, rand2 = rand_vect

        # rand1, rand2 = torch.rand(2)
        trsp = rand1 * 10
        if rand2 > 0.5:
            trsp = torch.round(trsp)

        aug4d = lambda tem: (tem + trsp) * dilate if torch.is_tensor(tem) else (torch.from_numpy(tem) + trsp) * dilate
        aug1d = lambda tem: (tem + trsp) * dilate

        current = aug1d(current)
        targets = list(map(aug4d, targets))
        drivers = list(map(aug1d, drivers))

        return [current, targets, drivers]

    def distance_dilatation(self, targets, dilate):
        dilat_distance = lambda t: [t[0], t[1], t[2], t[3]*dilate, t[4]*dilate]
        targets = list(map(dilat_distance, targets))
        return targets



    def __getitem__(self, idx):
        """ simple idx """
        if hasattr(self, 'augment') and self.augment is not None and self.augment:
            obs, sup = self.data[idx]
            world, targets, drivers, positions, time_constraints = obs
            rand_vect = torch.rand(2+4+1).double()     # 2 for time (1round, 1shift) + 4 for pos (2transp vect, 1rot, 1miror)  + 1dilation
            dilate = 1 + (rand_vect[-1] * 2 - 1)/10
            positions = self.pos_augmentation(positions, rand_vect[2:-1], dilate)
            time_constraints = self.time_augmentation(time_constraints, rand_vect[:2], dilate)
            targets = self.distance_dilatation(targets, dilate)
            return [world, targets, drivers, positions, time_constraints], sup

        else :
            return self.data[idx]


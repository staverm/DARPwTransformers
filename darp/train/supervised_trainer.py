import time
import json
import numpy as np
import math
import copy
from icecream import ic

from moviepy.editor import *
from matplotlib.image import imsave
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from .evaluations import *

from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, ConcatDataset
from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch.optim as optim

from models import *
from utils import *
from environments import DarEnv, DarPixelEnv, DarSeqEnv
from utils import get_device, objdict, SupervisionDataset
from strategies import NNStrategy #, NNStrategyV2
from generator import DataFileGenerator #, RFGenerator
from .save_examples import *
import sys

torch.autograd.set_detect_anomaly(True)


class SupervisedTrainer():
    def __init__(self, flags, sacred=None):
        ''' Inintialisation of the trainner:
                Entends to load all the correct set up, ready to train
        '''

        # Incorporate arguments to the object parameters
        for key in flags:
            setattr(self, key, flags[key])

        # Create saving experient dir
        self.path_name = self.rootdir + '/data/rl_experiments/' + self.alias + time.strftime("%d-%H-%M") + '_typ' + str(self.typ)
        ic(' ** Saving train path: ', self.path_name)
        if not os.path.exists(self.path_name):
            os.makedirs(self.path_name, exist_ok=True)
        else :
            ic(' Already such a path.. adding random seed')
            self.path_name = self.path_name + '#' + str(torch.randint(0, 10000, [1]).item())
            os.makedirs(self.path_name, exist_ok=True)

        # Save parameters
        with open(self.path_name + '/parameters.json', 'w') as f:
            json.dump(vars(self), f)

        self.sacred = sacred

        self.device = get_device()

        #### RL elements

        self.encoder_bn = False
        self.decoder_bn = False
        self.classifier_type = 1

        reward_function = globals()[self.reward_function]()
        if self.typ in [33]:
            # 1 layer with all infformation concatenated
            self.classifier_type = 8
            self.encoder_bn=False
            self.decoder_bn=False
            self.rep_type = '16'
            self.model='Trans18'
            self.emb_typ = 26
        else :
            raise "Find your own typ men"

        self.env = DarSeqEnv(size=self.image_size,
                          target_population=self.nb_target,
                          driver_population=self.nb_drivers,
                          reward_function=reward_function,
                          rep_type=self.rep_type,
                          max_step=self.max_step,
                          test_env=False,
                          timeless=self.timeless,
                          dataset=self.dataset,
                          verbose=self.verbose)

        self.eval_env = DarSeqEnv(size=self.image_size,
                                  target_population=self.nb_target,
                                  driver_population=self.nb_drivers,
                                  reward_function=reward_function,
                                  rep_type=self.rep_type,
                                  max_step=self.max_step,
                                  timeless=self.timeless,
                                  test_env=True,
                                  dataset=self.dataset)

        # DATASET EVAL
        self.dataset_instance_folder = '../data/cordeau/'
        self.inst_name = 'a' + str(self.nb_drivers) + '-' + str(self.nb_target)
        data_instance = self.dataset_instance_folder + self.inst_name + '.txt'
        reward_function = globals()[self.reward_function]()
        self.dataset_env = DarSeqEnv(size=self.image_size, target_population=self.nb_target, driver_population=self.nb_drivers,
                                  rep_type=self.rep_type, reward_function=reward_function, test_env=True, dataset=data_instance)
        # Get the best known solution from .txt instanaces
        with open(self.dataset_instance_folder + 'gschwind_results.txt') as f:
            for line in f :
                inst, bks = line.split(' ')
                if inst == self.inst_name :
                    self.dataset_env.best_cost = float(bks)
                    break;
            f.close()

        self.best_eval_metric = [0, 1000, 300, 300] # accuracy + loss + dataset GAP + online GAP
        self.train_rounds = 40
        self.partial_data_state=0
        self.nb_target = self.env.target_population
        self.nb_drivers = self.env.driver_population
        self.image_size = self.env.size
        self.max_time=int(self.env.time_end)

        if self.pretrain:
            self.supervision_function = 'nn'

        if self.supervision_function == 'nn':
            self.supervision = NNStrategy(reward_function=self.reward_function,
                                      env=self.env)
        else :
            raise ValueError('Could not find the supervision function demanded: '+ self.supervision_function)

        # Model Choice
    
        if self.model=='Trans18':
            self.model = globals()[self.model](src_vocab_size=50000,
                                                 trg_vocab_size=self.vocab_size + 1,
                                                 max_length=self.nb_target*2 + self.nb_drivers + 1,
                                                 src_pad_idx=-1,
                                                 trg_pad_idx=-1,
                                                 embed_size=self.embed_size,
                                                 dropout=self.dropout,
                                                 extremas=self.env.extremas,
                                                 device=self.device,
                                                 num_layers=self.num_layers,
                                                 heads=self.heads,
                                                 forward_expansion=self.forward_expansion,
                                                 typ=self.emb_typ,
                                                 max_time=int(self.env.time_end),
                                                 classifier_type=self.classifier_type,
                                                 encoder_bn=self.encoder_bn,
                                                 decoder_bn=self.decoder_bn).to(self.device).double()
        else :
            raise "self.model in PPOTrainer is not found"

        # loss
        if self.criterion == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif self.criterion == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        else :
            raise "Not found criterion"

        if self.pretrain :
            self.dist_criterion = nn.L1Loss()
            self.class_criterion = nn.CrossEntropyLoss()

        # optimizer
        if self.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.95)
        else :
            raise "Not found optimizer"

        # Scheduler
        if self.scheduler == 'plateau' :
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=self.patience, factor=self.gamma)
        elif self.scheduler == 'step':
            self.scheduler = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=self.gamma)

        # Checkpoint
        if self.checkpoint_dir :
            ic(' -- -- -- -- -- Loading  -- -- -- -- -- --')
            if self.typ >= 40 :
                self.model.load_state_dict(torch.load(self.rootdir + '/data/rl_experiments/' + self.checkpoint_dir).state_dict(), strict=False)
            else:
                self.model.load_state_dict(torch.load(self.rootdir + '/data/rl_experiments/' + self.checkpoint_dir).state_dict())
            ic(' -- The model weights has been loaded ! --')
            ic(' -----------------------------------------')

        if self.rl < 10000:
            self.baseline_model = copy.deepcopy(self.model)

        # number of elements passed throgh the model for each epoch
        self.testing_size = self.batch_size * (10000 // self.batch_size)    #About 10k
        self.training_size = self.batch_size * (100000 // self.batch_size)   #About 100k

        self.current_epoch = 0


        ic(' *// What is this train about //* ')
        for item in vars(self):
            if item == "model":
                vars(self)[item].summary()
            else :
                ic(item, ':', vars(self)[item])


    def generate_supervision_data(self):
        ic('\t ** Generation Started **')
        number_batch = self.data_size // self.batch_size
        size = number_batch * self.batch_size
        if self.datadir:
            data_directory = self.datadir
        else :
            data_directory = self.rootdir + '/data/supervision_data/'

        data = []
        if self.dataset:
            self.eval_episodes = 1
            data_type = self.dataset.split('/')[-1].split('.')[0]
            saving_name = data_directory + data_type + '_s{s}_tless{tt}_fun{sf}_typ{ty}/'.format(s=str(self.data_size),
                                                                                                                tt=str(self.timeless),
                                                                                                                sf=str(self.supervision_function),
                                                                                                                ty=str(self.rep_type))
        else :
            saving_name = data_directory + 's{s}_t{t}_d{d}_i{i}_tless{tt}_fun{sf}_typ{ty}/'.format(s=str(self.data_size),
                                                                                                              t=str(self.nb_target),
                                                                                                              d=str(self.nb_drivers),
                                                                                                              i=str(self.image_size),
                                                                                                              tt=str(self.timeless),
                                                                                                              sf=str(self.supervision_function),
                                                                                                              ty=str(self.rep_type))

        action_counter = np.zeros(self.vocab_size + 1)
        self.data_part = 0

        def load_dataset():
            return [saving_name + file for file in os.listdir(saving_name)]
            # files_names = os.listdir(saving_name)
            # datasets = []
            # for file in files_names:
            #     ic('Datafile folder:', saving_name)
            #     ic(file)
            #     datasets.append(torch.load(saving_name + file))
            # return ConcatDataset(datasets)

        def partial_name(size):
            name = saving_name + '/dataset_elementN' + str(self.data_part) + '_size' + str(size) + '.pt'
            self.data_part += 1
            return name

        if os.path.isdir(saving_name) :
            ic('This data is already out there !')
            dataset = load_dataset()
            return dataset
        else :
            os.makedirs(saving_name)

        done = True
        last_save_size = 0
        sub_data = []
        sub_action_counter = np.zeros(self.vocab_size + 1)
        observation = self.env.reset()

        # Generate a Memory batch
        for element in range(size):

            if done :
                if self.env.is_fit_solution():
                    data = data + sub_data
                    action_counter = action_counter + sub_action_counter
                elif self.pretrain:
                    # Might wana subsample thiiiis
                    data = data + sub_data
                    action_counter = action_counter + sub_action_counter
                else :
                    ic('/!\ Found a non feasable solution. It is not saved')

                if sys.getsizeof(data) > 100000: #200k bytes.
                    last_save_size += len(data)
                    train_data = SupervisionDataset(data, augment=self.augmentation, typ=self.typ)
                    name = partial_name(len(data))
                    torch.save(train_data, name)
                    data = []
                    ic('Saving data status')

                observation = self.env.reset()
                sub_data = []
                sub_action_counter = np.zeros(self.vocab_size + 1)


            supervised_action = self.supervision.action_choice()
            supervised_action = torch.tensor([supervised_action]).type(torch.LongTensor).to(self.device)

            sub_data.append([observation, supervised_action])
            observation, reward, done, info = self.env.step(supervised_action)

            sub_action_counter[supervised_action-1] += 1

            if element % 1000 == 0:
                ic('Generating data... [{i}/{ii}] memorry:{m}'.format(i=last_save_size + len(data), ii=self.data_size, m=sys.getsizeof(data)))

        train_data = SupervisionDataset(data, augment=self.augmentation, typ=self.typ)
        name = partial_name(len(data))
        torch.save(train_data, name)

        ic('Done Generating !')
        self.criterion.weight = torch.from_numpy(action_counter).to(self.device)
        data = load_dataset()
        return data

    def pretrain_log(self, name, time_distance, pick_distance, drop_distance, correct_loaded, correct_available, total):
        if self.sacred is not None:
            self.sacred.get_logger().report_scalar(title=name,
                series='Time distance', value=time_distance/total, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title=name,
                series='Pick distance', value=pick_distance/total, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title=name,
                series='Drop distance', value=drop_distance/total, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title=name,
                series='Acc /% Loaded', value=100*correct_loaded/total, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title=name,
                series='Acc /% available', value=100*correct_available/total, iteration=self.current_epoch)

    def pretrain_loss(self, output, prior_kwlg):
        out_time, out_pick, out_drop, out_loaded, out_available = output
        d_stuff, t_stuff = prior_kwlg
        d_stuff = torch.stack([torch.stack(elmt) for elmt in d_stuff]).to(self.device).permute(2, 0, 1)
        t_stuff = torch.stack([torch.stack(elmt) for elmt in t_stuff]).to(self.device).permute(2, 0, 1)
        prior_time, prior_loaded = d_stuff[:,:,0], d_stuff[:,:,1]
        prior_available, prior_pick, prior_drop = t_stuff[:,:,0], t_stuff[:,:,1], t_stuff[:,:,2]

        # The divisions are a ruff way to normalise the losses values
        time_distance = self.dist_criterion(out_time, prior_time)
        l1 = time_distance / self.max_time
        pick_distance = self.dist_criterion(out_pick, prior_pick)
        l2 = pick_distance / (2*self.image_size)
        drop_distance = self.dist_criterion(out_drop, prior_drop)
        l3 = drop_distance / (2*self.image_size)
        correct_loaded = 0
        l4 = 0
        for i in range(out_loaded.shape[1]):
            l4 += self.class_criterion(out_loaded[:,i], prior_loaded[:,i].long())
            correct_loaded += np.sum((out_loaded[:,i].argmax(-1) == prior_loaded[:,i].long()).cpu().numpy())
        l4 = l4 / (out_loaded.shape[1]*out_loaded.shape[-1])
        correct_loaded = correct_loaded / out_loaded.shape[1]

        correct_available = 0
        l5 = 0
        for i in range(out_available.shape[1]):
            l5 += self.class_criterion(out_available[:,i], prior_available[:,i].long())
            correct_available += np.sum((out_available[:,i].argmax(-1) == prior_available[:,i].long()).cpu().numpy())
        l5 = l5 / (out_available.shape[1]*out_available.shape[-1])
        correct_available = correct_available / out_available.shape[1]

        final_loss = l1 + l2 + l3 + l4 +l5

        # ic(l1, l2, l3, l4, l5)
        return final_loss, time_distance, pick_distance, drop_distance, correct_loaded, correct_available


    def train(self, dataloader):
        max_test_accuracy = 0
        running_loss = 0
        total = 0
        correct = nearest_accuracy = pointing_accuracy = 0
        mean_time_distance, mean_pick_distance, mean_drop_distance, mean_correct_loaded, mean_correct_available = 0, 0, 0, 0, 0

        self.model.train()
        for i, data in enumerate(dataloader):
            # set the parameter gradients to zero
            self.optimizer.zero_grad()

            observation, supervised_action = data

            if self.typ >= 40 :
                world, targets, drivers, positions, time_constraints, prior_kwlg = observation
            else :
                world, targets, drivers, positions, time_constraints = observation
            info_block = [world, targets, drivers]
            # Current player as trg elmt
            if self.typ in [17, 18, 19]:
                target_tensor = world
            else :
                target_tensor = world[1].unsqueeze(-1).type(torch.LongTensor).to(self.device)
            model_action = self.model(info_block,
                                      target_tensor,
                                      positions=positions,
                                      times=time_constraints)

            supervised_action = supervised_action.to(self.device)

            if self.pretrain :
                loss, time_distance, pick_distance, drop_distance, correct_loaded, correct_available = self.pretrain_loss(model_action, prior_kwlg)
                total += supervised_action.size(0)
                running_loss += loss.item()
                mean_time_distance += time_distance
                mean_pick_distance += pick_distance
                mean_drop_distance += drop_distance
                mean_correct_loaded += correct_loaded
                mean_correct_available += correct_available
                correct += (correct_available + correct_loaded)/2
            else :
                loss = self.criterion(model_action.squeeze(1), supervised_action.squeeze(-1))
                total += supervised_action.size(0)
                correct += np.sum((model_action.squeeze(1).argmax(-1) == supervised_action.squeeze(-1)).cpu().numpy())
                running_loss += loss.item()

            loss.backward()
            # update the gradients
            self.optimizer.step()

            if i == self.train_rounds:
                break

        acc = 100 * correct/total
        ic('-> Réussite: ', acc, '%')
        ic('-> Loss:', 100*running_loss/total)
        self.scheduler.step(running_loss)
        if self.pretrain :
            self.pretrain_log('Pretrain train',
                              mean_time_distance,
                              mean_pick_distance,
                              mean_drop_distance,
                              mean_correct_loaded,
                              mean_correct_available,
                              total)

        if self.sacred is not None:
            self.sacred.get_logger().report_scalar(title='Train stats',
                series='train loss', value=100*running_loss/total, iteration=self.current_epoch)
            self.sacred.get_logger().report_scalar(title='Train stats',
                series='Train accuracy', value=acc, iteration=self.current_epoch)


    def load_partial_data(self):
        if len(self.dataset_names) > 10:
            part_size = len(self.dataset_names) // 20
            files_names = self.dataset_names[self.partial_data_state*part_size:(self.partial_data_state+1)*part_size - 1]
            ic(len(files_names))
        else :
            files_names = self.dataset_names
        datasets = []
        for file in files_names:
            ic('Datafile folder:', file)
            datasets.append(torch.load(file))
        self.partial_data_state += 1
        self.partial_data_state = self.partial_data_state % 10
        return ConcatDataset(datasets)

    def updating_data(self):
        if self.current_epoch == 0 :
            # First time (generate) and get files names of the data
            if self.supervision_function == 'rf':
                self.dataset_names = self.supervision.generate_dataset()
            elif self.pretrain : #FIXME no differance
                self.dataset_names = self.generate_supervision_data() 
            else :
                self.dataset_names = self.generate_supervision_data()

        ic(len(self.dataset_names))
        # Load a 10th of the data
        dataset = self.load_partial_data()

        # Take care of the loaded dataset part to
        action_counter = np.zeros(self.vocab_size + 1)
        if True :
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(0.1 * dataset_size))
            train_indices, val_indices = indices[split:], indices[:split]
            if self.shuffle :
                np.random.shuffle(train_indices)
                np.random.shuffle(val_indices)
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)
            supervision_data = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                       sampler=train_sampler)
            validation_data = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                        sampler=valid_sampler)
        else :
            for data in dataset:
                o, a = data
                action_counter[a] += 1
            self.criterion.weight = torch.from_numpy(action_counter).to(self.device)
            supervision_data = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
            validation_data = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle) #TODO it was empty

        return supervision_data,  validation_data

    def run(self):
        """
            Just the main training loop
            (Eventually generate data)
            Train and evaluate
        """
        round_counter = 1e6 // (self.train_rounds -1)
        ic('\t ** Learning START ! **')
        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train
            # Every million visits update the dataset
            if round_counter * self.batch_size * self.train_rounds > self.data_size // 20:
                if 'supervision_data' in locals():
                    del supervision_data
                    del validation_data
                ic(epoch)
                supervision_data, validation_data = self.updating_data() #FIXME always 
                round_counter = 0
            self.train(supervision_data)
            # self.rl_train()

            # Evaluate
            if True:
                self.best_eval_metric = offline_evaluation(validation_data,
                self.model,
                self.sacred,
                self.emb_typ,
                self.typ,
                self.device,
                self.best_eval_metric,
                self.current_epoch,
                self.path_name,
                self.checkpoint_type,
                self.criterion,
                saving=True)
            elif self.rl <= epoch:
                ic(self.rl)
                self.best_eval_metric = dataset_evaluation(self.model,
                self.sacred,
                self.emb_typ,
                self.typ,
                self.device,
                self.best_eval_metric,
                self.current_epoch,
                self.path_name,
                self.checkpoint_type,
                self.inst_name,
                self.dataset_env)
            else :
                self.online_evaluation()
                if self.dataset:
                    self.best_eval_metric = online_evaluation( 
                        self.model, 
                        self.sacred, 
                        self.emb_typ, 
                        self.typ, 
                        self.best_eval_metric, 
                        self.current_epoch, 
                        self.path_name,
                        self.checkpoint_type,
                        self.eval_env,
                        self.eval_episodes,
                        self.supervision,
                        self.example_format,
                        self.criterion,
                        self.device,
                        full_test=False)

            round_counter +=1

        ic('\t ** Learning DONE ! **')





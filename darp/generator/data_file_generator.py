#!/usr/bin/env python3

import os

from environments import DarSeqEnv
# from dialRL.utils import get_device, trans25_coord2int
# from dialRL.strategies import NNStrategy


class DataFileGenerator():
    '''
    Generator from an environment object to a data file .txt element

    Number of vehicles, Number of requests, Maximum route duration, Vehicle capacity, Maximum ride time
      3 48 480 6 90
     Depot info
      0   -1.044    2.000  0  0    0 1440
      <id> <x> <y> <service time> <demand> <TW start> <TW end>
      1   -2.973    6.414 10  1    0 1440
      ....

      Changement to A&B dataset cordeau 2006 !
    '''

    def __init__(self, env=None,  out_dir=None, data_size=None):
        if env is None :
            self.params = {
                'size': 10,
                'target_population': 5,
                'driver_population': 2,
                'reward_function': 'ConstantReward',
                'time_end': 1400,
                'max_step': 2000,
                'timeless' : False,
                'dataset': '../data/cordeau/a2-16.txt', #'/home/tibo/Documents/Prog/EPFL/own/data/instances/cordeau2006/a2-16.txt',
                'test_env': False,
                'out_dir': '../generator_out/',
                'data_size': 1
            }
            self.data_size = self.params['data_size']
            self.out_dir = self.params['out_dir']
            self.env = DarSeqEnv(size=self.params['size'],
                            target_population=self.params['target_population'],
                            driver_population=self.params['driver_population'],
                            reward_function=self.params['reward_function'],
                            time_end=self.params['time_end'],
                            max_step=self.params['max_step'],
                            timeless=self.params['timeless'],
                            dataset=self.params['dataset'],
                            test_env=self.params['test_env'])
        else :
            self.env = env
            self.out_dir = out_dir
            self.data_size = data_size


    def generate_file(self, tmp_name=None):
        file_names = []
        os.makedirs(self.out_dir, exist_ok=True)
        for n in range(self.data_size):

            observation = self.env.reset()
            text_lists = []

            text_lists.append([self.env.driver_population, self.env.target_population, self.env.time_limit, self.env.drivers[0].max_capacity, self.env.max_ride_time])
            text_lists.append([0, self.env.depot_position[0], self.env.depot_position[1], 0, 0, 0, int(self.env.time_end)])
            for target in self.env.targets:
                text_lists.append([target.identity, target.pickup[0], target.pickup[1], target.service_time, 1, int(target.start_fork[0]), int(target.start_fork[1])])
            for target in self.env.targets:
                text_lists.append([target.identity + len(self.env.targets), target.dropoff[0], target.dropoff[1], target.service_time, -1, int(target.end_fork[0]), int(target.end_fork[1])])
            text_lists.append([len(self.env.targets)*2 + 1, self.env.depot_position[0], self.env.depot_position[1], 0, 0, 0, int(self.env.time_end)])
            self.env.close()

            # Join the text info into 1 file data

            text_lists2 = []
            for text in text_lists :
                text_lists2.append('\t'.join(map(str, map(lambda x: round(x, 3), text))))
            text_lists2.append('')
            final_string = '\n'.join(text_lists2)

            if tmp_name is None:
                file_name = str(n)
            else :
                file_name = tmp_name

            name = self.out_dir + '/' + file_name + '.txt'
            file_names.append(name)
            with open(name, 'w') as write_file:
                write_file.write(final_string)
            write_file.close()

        with open(self.out_dir + '/INDEX.txt', 'w') as write_file:
            for n in range(self.data_size):
                write_file.write(file_name + '\n')

        write_file.close()
        return file_names

if __name__ == '__main__':
    gen = DataFileGenerator()
    gen.generate_file()

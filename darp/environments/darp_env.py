import numpy as np
import gym
from gym import spaces

from utils import instance2world, indice2image_coordonates, distance
from .darp_instance import DarPInstance

class DarEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, size, target_population, driver_population, time_end=1400, max_step=10):
        super(DarEnv, self).__init__()

        self.size = size
        self.max_step = max_step
        self.time_end = time_end
        self.target_population = target_population
        self.driver_population = driver_population
        self.action_space = spaces.Discrete(target_population + 1)
        #spaces.Discrete(size**2)
        self.observation_space = spaces.Box(low=-self.size,
                high=self.size,
                shape=(self.size+1, self.size+1),
                dtype=np.int16)
        self.max_reward = int(1.5 * self.size)
        self.reward_range = (- self.max_reward, self.max_reward)
        self.current_episode = 0
        self.success_episode = []
        self.cumulative_reward = 0
        self.last_aim = None
        self.last_cell = None



    def representation(self):
        world = np.zeros((self.size, self.size))

        for i, driver in enumerate(self.instance.drivers):
            world[driver.position[0]][driver.position[1]] = i + 1

        for j, target in enumerate(self.instance.targets):
            world[target.pickup[0]][target.pickup[1]] = j + self.driver_population + 1
            world[target.dropoff[0]][target.dropoff[1]] = - j - self.driver_population - 1

        return world


    def reset(self):
        self.instance = DarPInstance(size=self.size,
                                        population=self.target_population,
                                        drivers=self.driver_population,
                                        time_end=self.time_end,
                                        verbose=False)

        self.instance.random_generation()
        # print('* Reset - Instance image : ', self.instance.image)
        self.targets = self.instance.targets.copy()
        self.drivers = self.instance.drivers.copy()

        self.current_player = 1
        # distance is -1 if wrong aiming. 0 if there is no start of game yet and x if aimed corectly
        self.distance = 0
        self.current_step = 0
        self.cumulative_reward = 0
        self.world = self.representation()
        self.last_aim = None
        self.last_cell = None
        self.short_log = ''
        return self._next_observation()


    def del_target(self, position):
        filter_fun = lambda x : x[0] == position[0] and x[1] == position[1]
        indices = [i for i in range(len(self.targets)) if filter_fun(self.targets[i])]
        for indice in indices:
            del self.targets[indice]


    def targets_to_go(self):
        count = 0
        for target in self.targets:
            if target.state == 0 or target.state == -1:
                count += 1
        return count


    def _next_observation(self):

        # max_coordonate = max(self.depot_position)
        # for t in targets :
        #     x, y = np.abs(t.pickup)
        #     xx, yy = np.abs(t.dropoff)
        #     max_coordonate = max(x, y, xx, yy)



        obs = self.world
        addition_info = np.zeros(self.size)
        addition_info[0] = self.current_player
        for i,t in enumerate(self.drivers[self.current_player - 1].loaded):
            if 1+i < self.size :
                addition_info[1+i] = t.identity
        obs = np.append(obs, np.array([addition_info]), axis=0)
        addition_info2 =  np.expand_dims(np.zeros(self.size+1), axis=0)
        obs = np.append(obs, addition_info2.T, axis=1)
        return obs


    def _take_action(self, action):
        """ Action: destination point as an indice of the map vactor. (Ex: 1548 over 2500)
        """

        current_pos = np.where(self.world == self.current_player)
        # indice = np.argmax(action)
        next_pose = indice2image_coordonates(action, self.size)
        # next_pose = utils.indice2image_coordonates(action, self.size)
        self.last_aim = next_pose
        self.last_cell = self.world[next_pose]

        #In case we aim an empty box
        if self.world[next_pose] == 0 :
            self.distance = -1
            self.short_log = 'Aimed empty'

        elif int(self.world[next_pose]) in range(1,self.driver_population + 1) :
            self.distance = -2
            self.short_log = 'Aimed a driver'

        else :
            target_indice = int(abs(self.world[next_pose])) - self.driver_population - 1
            aimed_target = self.targets[target_indice]

            # In case we need to pick it up
            if self.world[next_pose] > 0 and aimed_target.state == -1 :
                result = self.drivers[self.current_player - 1].load(aimed_target)
                # Managed to load the target
                if result :
                    aimed_target.state = 0
                    self.world[next_pose] = self.current_player
                    self.world[current_pos] = 0

                    self.drivers[self.current_player - 1].position = next_pose
                    self.distance = distance(next_pose, current_pos)
                    self.short_log = 'Aimed right, and pick up !'

                # couldnt load target in driver
                else :
                    self.distance = -3
                    self.short_log = 'Aimed right BUT driver full'

            # In case of a drop off
            elif self.world[next_pose] < 0 and aimed_target.state == 0 :
                result = self.drivers[self.current_player - 1].unload(aimed_target)
                if result :
                    aimed_target.state = 1
                    self.world[next_pose] = self.current_player
                    self.world[current_pos] = 0
                    self.drivers[self.current_player - 1].position = next_pose
                    self.distance = distance(next_pose, current_pos)

                    self.short_log = 'Aimed right, and droping off !'

                else :
                    self.distance = -4
                    self.short_log = 'Aimed right BUT driver doesnt contain that target'

            elif self.world[next_pose] < 0 and aimed_target.state == -1 :
                self.short_log = 'Aiming Dropoff point without having loaded on driver'
                self.distance = -5
            else :
                self.short_log = 'Other wrong doing(should be somthing like aiming dropoff)'
                self.distance = -6


    def reward(self, distance):
        return self.max_reward - distance #Linear distance
        #int(1.5 * self.size * (1 / distance))


    def step(self, action):
        # Curently the action is a heatmap ..
        # NOO now it's an indice in the list of action n*n

        self._take_action(action)
        self.current_step += 1

        if self.distance < 0:
            reward = -1 #-int(self.max_reward//2)
            done = False
        elif self.distance > 0:
            reward = self.reward(self.distance)
            done = False
            # update drivers turn
            self.current_player = ((self.current_player + 1 - 1) % (self.driver_population) ) + 1
        # End of simulation
        elif self.distance == 0 :
            print("What ?")


        self.cumulative_reward += reward

        if self.targets_to_go() == 0 :
            done = True
        if self.current_step >= self.max_step:
            done = True

        if done:
            self.current_episode += 1
            # print('End of episode, total reward :', self.cumulative_reward)

        obs = self._next_observation()

        # Last element is info (dict)
        return obs, reward, done, {}


    def render(self):
        print('\n--------------------- [Step', self.current_step, ']')
        print('World: ')
        print(self.world)

        if self.distance < 0 :
            print(f'Player {self.current_player} go lost ....')
        else :
            print(f'Player {self.current_player} aimed right')

        print('present player: ', self.current_player)
        print('Last aimed to : ', self.last_aim)
        print('Cell of last aime : ', self.last_cell)
        print('Targets to go: ', self.targets_to_go())
        print('Cumulative reward : ', self.cumulative_reward)
        print('Additional  information : ', self.short_log)
        print('---------------------\n')


if __name__ == '__main__':
    env = DarEnv(size=4, target_population=5, driver_population=1, time_end=1400, max_step=20)
    cumulative_reward = 0
    observation = env.reset()
    for t in range(20):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        print('Cumulative reward : ', cumulative_reward, ' (', reward, ')')
        if done:
            print("\n ** \t Episode finished after {} timesteps".format(t+1))
            env.render()
            break
    env.close()
 

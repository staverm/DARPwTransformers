#!/usr/bin/env python3

class BaseReward():
    def __init__(self):
        pass

    def compute(self, distance, done):
        return 1

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "(Reward function D->X)"



class ConstantReward(BaseReward):
    def __init__(self):
        """ This function returns a constant reward of -1, 0 or 1
        according to the planed distance that the choice leads to
        """
        super().__init__()

    def compute(self, distance, done, env):
        if distance < 0 :
            return -1
        elif distance == 0:
            return 0
        else :
            return 1


class ProportionalReward(BaseReward):
    def __init__(self):
        """
        This function returns a reward of -1, 0 or (MD - distance)
        according to the planed distance that the choice leads to
        """
        super().__init__()

    def compute(self, distance, done, env):
        if distance < 0 :
            return -1
        elif distance == 0:
            return 0
        else :
            return env.max_reward - distance


class NoNegativeProportionalReward(BaseReward):
    def __init__(self):
        """
        This function returns a reward of 0 or (MD - distance)
        according to the planed distance that the choice leads to
        """
        super().__init__()

    def compute(self, distance, done, env):
        if distance <= 0 :
            return 0
        else :
            return env.max_reward - distance


class ProportionalEndReward(BaseReward):
    def __init__(self):
        """
        This function returns a reward of (MD - total_distance) at the end, if finished
        according to the planed distance that the choice leads to
        """
        super().__init__()

    def compute(self, distance, done, env):
        if done and env.is_fit_solution() :
            return (env.max_step * env.max_reward) - env.total_distance
        elif done :
            return - (env.max_step * env.max_reward)
        else :
            return 0

class ProportionalEndDistance(BaseReward):
    def __init__(self):
        """
        This function returns a reward of (MD - total_distance) at the end, if finished
        according to the planed distance that the choice leads to
        """
        super().__init__()

    def compute(self, distance, done, env):
        if done and env.is_fit_solution() :
            return env.total_distance
        elif done :
            return (env.size * env.target_population * 4)
        else :
            return 0


class NoNegativeProportionalEndReward(BaseReward):
    def __init__(self):
        """
        This function returns a reward of (MD - total_distance) at the end, if finished
        according to the planed distance that the choice leads to
        """
        super().__init__()

    def compute(self, distance, done, env):
        if done and env.is_fit_solution() :
            return (env.max_step * env.max_reward) - env.total_distance
        else :
            return 0

class NoNegativeEndReward(BaseReward):
    def __init__(self):
        """
        This function returns a reward of (MD - total_distance) at the end, if finished
        according to the planed distance that the choice leads to
        """
        super().__init__()

    def compute(self, distance, done, env):
        if done and env.is_fit_solution() :
            return 1
        else :
            return 0

class EndReward(BaseReward):
    def __init__(self):
        """
        This function returns a reward of (MD - total_distance) at the end, if finished
        according to the planed distance that the choice leads to
        """
        super().__init__()

    def compute(self, distance, done, env):
        if done and env.is_fit_solution() :
            return 1
        elif done :
            return -1
        else :
            return 0

class HybridProportionalReward(BaseReward):
    def __init__(self):
        """
        This function returns a reward of (MD - total_distance) at the end, if finished
        according to the planed distance that the choice leads to
        """
        super().__init__()

    def compute(self, distance, done, env):
        if done and env.is_fit_solution() :
            return (env.max_step * env.max_reward) - env.total_distance
        elif done :
            return - (env.max_step * env.max_reward)
        else :
            if distance < 0 :
                return -1
            elif distance == 0:
                return 0
            else :
                return env.max_reward - distance

class ConstantDistributionReward(BaseReward):
    def __init__(self):
        """
        Rwd proportional to distributed state
        """
        super().__init__()
        self.distribution = None

    def compute(self, distance, done, env):
        new_distribution = env.targets_states()
        rwd = 0
        if self.distribution is None or not(self.distribution == new_distribution) :
            for t, count in enumerate(new_distribution):
                rwd += t * count
            self.distribution = new_distribution
        return rwd

class JustDistance(BaseReward):
    def __init__(self):
        """
        Reward = Distance lors du calcul
        """
        super().__init__()

    def compute(self, distance, done, env):
        if distance < 0 :
            return 100
        elif distance == 0 :
            return 10
        else :
            return distance

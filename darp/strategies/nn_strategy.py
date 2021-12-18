import numpy as np

from .base_strategy import BaseStrategy
from utils import distance


class NNStrategy(BaseStrategy):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def action_choice(self, observation=None):
        player_nb = self.env.current_player
        player = self.env.drivers[player_nb - 1]
        choice = (None, np.inf)
        for target in self.env.targets :

            # Potential pickup
            if target.state == -2 :
                if player.can_load(target, self.env.time_step) :
                    d = distance(player.position, target.pickup)
                    if choice[1] > d:
                        choice = (target, d)

            # Potential Dropoff
            elif target.state == 0 :
                if player.can_unload(target, self.env.time_step) :
                    d = distance(player.position, target.dropoff)
                    if choice[1] > d:
                        choice = (target, d)

        if choice[0] is None :
            return 0
        return choice[0].identity


if __name__ == '__main__':
    strat = NNStrategy(size=4,
                        target_population=24,
                        driver_population=3,
                        reward_function='ConstantReward',
                        time_end=1400,
                        max_step=5000,
                        timeless=False,
                        dataset='../data/cordeau/a2-16.txt',
                        test_env=True)

    strat.run()
 

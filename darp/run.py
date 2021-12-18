 
from strategies import NNStrategy
from utils import ConstantReward

if __name__ == '__main__':

    rwd_fun = ConstantReward()
    strat = NNStrategy(size=4,
                        target_population=24,
                        driver_population=3,
                        reward_function=rwd_fun,
                        time_end=1400,
                        max_step=5000,
                        timeless=False,
                        dataset='../data/cordeau/a2-16.txt',
                        test_env=True)

    strat.run()

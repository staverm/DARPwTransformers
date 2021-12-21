import numpy as np
import os
# from matplotlib.image import imsave

# from moviepy.editor import *
import drawSvg as draw

from environments import DarSeqEnv

class BaseStrategy():
    def __init__(self,
                 size=4,
                 target_population=2,
                 driver_population=2,
                 reward_function='',
                 time_end=1400,
                 max_step=5000,
                 timeless=False,
                 env=None,
                 dataset='./data/instances/cordeau2003/tabu1.txt',
                 test_env=False,
                 recording=False):

        self.data = dataset
        self.rwd_fun = reward_function
        # self.rwd_fun =  globals()[reward_function]()
        self.recording = recording
        if env is None:
            self.env = DarSeqEnv(size=size,
                            target_population=target_population,
                            driver_population=driver_population,
                            reward_function=self.rwd_fun,
                            time_end=time_end,
                            max_step=max_step,
                            timeless=timeless,
                            dataset=self.data,
                            test_env=test_env)
        else :
            self.env = env
        self.max_step = max_step

    def norm_image(self, image, type=None, scale=10):
        image = np.kron(image, np.ones((scale, scale)))
        if type=='rgb':
            ret = np.empty((image.shape[0], image.shape[0], 3), dtype=np.uint8)
            ret[:, :, 0] = image.copy()
            ret[:, :, 1] = image.copy()
            ret[:, :, 2] = image.copy()
            image = ret.copy()
        return (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)


    # def save_video(self, dir, noms):
    #     # Save the imges as video
    #     video_name = dir + '/Strat_res.mp4'
    #     clips = [ImageClip(m).set_duration(0.2)
    #           for m in noms]

    #     concat_clip = concatenate_videoclips(clips, method="compose")
    #     concat_clip.write_videofile(video_name, fps=24, verbose=None, logger=None)
    #     del concat_clip
    #     del clips


    # def save_svg_video(self, dir, svgs):
    #     video_name = dir + '/Strat_res.mp4'

    #     with draw.animate_video(video_name) as anim:
    #         # Add each frame to the animation
    #         for s in svgs:
    #             anim.draw_frame(s)


    # def save_image(self, dir, t, image):
    #     image = self.norm_image(image, scale=1)
    #     save_name = dir + '/' + \
    #            str(t) + \
    #            'b.png'
    #     imsave(save_name, image)
    #     return save_name

    def save_svg(self, dir, t, svg):
        save_name = dir + '/' + \
               str(t) + \
               'b.svg'
        svg.saveSvg(save_name)
        return svg


    def run(self):
        print("RUN NN STRAT")
        # Image saving dir
        dir = './plots/rl_experiments/strategies/' + \
               self.__class__.__name__
        os.makedirs(dir, exist_ok=True)
        cumulative_reward = 0
        observation = self.env.reset()
        svgs = [self.save_svg(dir, 0, self.env.get_svg_representation())]
        last_time = -1
        self.env.render()
        for t in range(self.max_step):
            action = self.action_choice(observation)
            observation, reward, done, info = self.env.step(action)
            cumulative_reward += reward
            print('Cumulative reward : ', cumulative_reward, ' (', reward, ')')
            self.env.render()

            # If time updataed, save image
            if self.env.time_step > last_time :
                # last_time = self.env.time_step
                svgs.append(self.save_svg(dir, t+1, self.env.get_svg_representation()))

            if done:
                print("\n ** \t Episode finished after {} time steps".format(t+1))
                break

        # self.save_video(dir, images)
        # self.save_svg_video(dir, svgs)
        self.env.close()


    def action_choice(self):
        pass
 

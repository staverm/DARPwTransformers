from moviepy.editor import *
from matplotlib.image import imsave
import os
import drawSvg as draw
import numpy as np


def save_example(self, observations, rewards, number, time_step):
        noms = []
        dir = self.path_name + '/example/' + str(time_step) + '/ex_number' + str(number)
        if dir is not None:
            os.makedirs(dir, exist_ok=True)

            for i, obs in enumerate(observations):
                save_name = dir + '/' + str(i) + '_r=' + str(rewards[i]) + '.png' 
                image = obs
                imsave(save_name, image)
                noms.append(save_name)

        # Save the imges as video
        video_name = dir + 'r=' + str(np.sum(rewards)) + '.mp4'
        clips = [ImageClip(m).set_duration(0.2)
              for m in noms]

        concat_clip = concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile(video_name, fps=24, verbose=None, logger=None)

        if self.sacred :
            self.sacred.get_logger().report_media('video', 'Res_' + str(number) + '_Rwd=' + str(np.sum(rewards)),
                                                  iteration=time_step,
                                                  local_path=video_name)
        del concat_clip
        del clips


def save_svg_example(self, observations, rewards, number, time_step):
    dir = self.path_name + '/example/' + str(time_step) + '/ex_number' + str(number)
    video_name = dir + '/Strat_res.mp4'
    if dir is not None:
        os.makedirs(dir, exist_ok=True)

    with draw.animate_video(video_name, align_right=True, align_bottom=True) as anim:
        # Add each frame to the animation
        for i, s in enumerate(observations):
            anim.draw_frame(s)
            if i==len(observations)-1 or i==0:
                for i in range(5):
                    anim.draw_frame(s)

    if self.sacred :
        self.sacred.get_logger().report_media('Gif', 'Res_' + str(number) + '_Rwd=' + str(np.sum(rewards)),
                                                iteration=time_step,
                                                local_path=video_name)

import os
from python_eulerian_video_magnification.magnifycolor import MagnifyColor
from python_eulerian_video_magnification.metadata import MetaData
from python_eulerian_video_magnification.mode import Mode
import cv2
import numpy as np

'''
    Magnification video
'''
def generate_mag_video(vid_path, mag_path):
    if not os.path.exists(mag_path):
        os.makedirs(mag_path)

    vidlist = os.listdir(vid_path)
    vidlist.sort()
    for vidname in vidlist:
        print("{} - {} ... ".format(video_dir_name, vidname), end='', flush=True)
        vidpath = vid_path + vidname

        save_vid_path = mag_path# + vidname

        MagnifyColor(MetaData(file_name=vidpath, low=0.833, high=2, levels=1,
                    amplification=10, output_folder=save_vid_path, mode=Mode.COLOR, suffix='color')).do_magnify()
        print("Done")

if __name__=="__main__":

    #video_dir_name = "Real/"
    video_dir_name = "Synthesized/"

    data_root_dir = "D:/Datasets/CelebDF v2/3_Align_Video/"
    datadir = data_root_dir + video_dir_name

    new_vid_root_dir = "D:/Datasets/CelebDF v2/4_Magnify_Video/"
    newviddir = new_vid_root_dir + video_dir_name
    if not os.path.exists(newviddir):
        os.makedirs(newviddir)

    generate_mag_video(datadir, newviddir)

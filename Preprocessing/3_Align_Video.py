import os
import cv2
import numpy as np

'''
    Generate video with aligned face. These videos will be used in motion magnificaiton.
'''
def generate_align_video(video_frame_path, video_store_path):
    if not os.path.exists(video_store_path):
        os.makedirs(video_store_path)

    vidlist = os.listdir(video_frame_path)
    vidlist.sort()
    for vidname in vidlist:
        print("{} - {} ... ".format(video_frame_path, vidname), end='', flush=True)
        vidpath = video_frame_path + vidname +'/'

        save_vid_path = video_store_path + vidname + '.mp4'

        os.system("ffmpeg -i \"{}%04d.jpg\" \"{}\"".format(vidpath, save_vid_path))
        print("Done")


if __name__=="__main__":

    video_dir_name = "Real/"
    #video_dir_name = "Synthesized/"

    data_root_dir = "D:/Datasets/CelebDF v2/2_Resize/"
    datadir = data_root_dir + video_dir_name

    new_vid_root_dir = "D:/Datasets/CelebDF v2/3_Align_Video/"
    newviddir = new_vid_root_dir + video_dir_name
    if not os.path.exists(newviddir):
        os.makedirs(newviddir)

    generate_align_video(datadir, newviddir)

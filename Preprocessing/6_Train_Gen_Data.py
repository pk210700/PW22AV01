import numpy as np
from PK.classifiers import *
from PK.pipeline import *
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

import os
import cv2

classifier = Meso4()
classifier.load('D:/Sem VI - Textbooks & Materials/CAPSTONE PROJECT/DeepRhythm_Code/Meso4_DF.h5')

real_video_dir = "D:/Datasets/CelebDF v2/4_Magnify_Video/Real_Frame/"
real_vid_list = os.listdir(real_video_dir)
real_vid_list.sort()
real_vid_dict = {x:real_video_dir + x +'/' for x in real_vid_list}

fake_video_dir = "D:/Datasets/CelebDF v2/4_Magnify_Video/Synthesized_Frame/"
fake_vid_list = os.listdir(fake_video_dir)
fake_vid_list.sort()
fake_vid_dict = {x:fake_video_dir + x +'/' for x in fake_vid_list}

video_dict = real_vid_dict
video_dict.update(fake_vid_dict)
data_name_list = real_vid_list + fake_vid_list

fake_mit_dir = "D:/Datasets/CelebDF v2/5_MMST_Map/Synthesized/"
fake_mit_list = os.listdir(fake_mit_dir)
fake_mit_list.sort()
fake_mit_dict = {x:fake_mit_dir + x for x in fake_mit_list}

real_mit_dir = "D:/Datasets/CelebDF v2/5_MMST_Map/Real/"
real_mit_list = os.listdir(real_mit_dir)
real_mit_list.sort()
real_mit_dict = {x:real_mit_dir + x for x in real_mit_list}

mit_dict = fake_mit_dict
mit_dict.update(real_mit_dict)

save_path = "D:/Datasets/CelebDF v2/6_Train_Data v2/"
if not os.path.exists(save_path):
    os.mkdir(save_path)

data_Meso_set = []
data_mit_set = []
data_y_set = []
data_name_set = []

for vid_name in data_name_list:
    print("video {}:".format(vid_name), end='', flush=True)
    if vid_name+'.npy' in mit_dict:
        vid_path = video_dict[vid_name]
        img_list = os.listdir(vid_path)
        if img_list==[]:
            continue
        img_list.sort()
        data_Meso = np.ones(300) * 0.5
        try:
            for img_name in img_list:
                if int(img_name[:-4]) >= 300:
                    continue
                img_path = vid_path + img_name
                img = cv2.resize(cv2.imread(img_path), (256, 256))
                pred = classifier.predict(np.array([img]))
                data_Meso[int(img_name[:-4])] = pred
        except Exception:
            continue
        data_Meso_set.append(data_Meso)
        data_mit = np.load(mit_dict[vid_name+'.npy'])
        #data_mit = np.swapaxes(data_mit,0,1)
        data_mit_set.append(data_mit)
        if vid_name.count('_')!=1:
            data_y_set.append(1)
            print(1)
        else:
            data_y_set.append(0)
            print(0)
        data_name_set.append(vid_name)

    print("  ", np.shape(data_Meso_set))
    print("  ", np.shape(data_mit_set))
    print("  ", np.shape(data_y_set))
    print("  ", np.shape(data_name_set))

print("SAVING .... ")
np.save(save_path+"Meso.npy", data_Meso_set)
np.save(save_path+"mit.npy", data_mit_set)
np.save(save_path+"y.npy", data_y_set)
np.save(save_path+"name.npy", data_name_set)

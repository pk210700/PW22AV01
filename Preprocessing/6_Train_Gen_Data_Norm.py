import numpy as np
# from PK.classifiers import *
# from PK.pipeline import *
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

import os
import cv2
#
# classifier = Meso4()
# classifier.load('D:/Sem VI - Textbooks & Materials/CAPSTONE PROJECT/DeepRhythm_Code/Meso4_DF.h5')

real_f_t_video_dir = "E:/CAPSTONE/CAPSTONE/synth_pre/Normalised/Real/Normalized/" # f(t) real frame dir
real_f_t_vid_list = os.listdir(real_f_t_video_dir)
real_f_t_vid_list.sort()
real_f_t_vid_dict = {x:real_f_t_video_dir + x +'/' for x in real_f_t_vid_list}

real_cons_video_dir = "E:/CAPSTONE/CAPSTONE/synth_pre/Normalised/Real/Consecutive/" # consecutive real frame dir
real_cons_vid_list = os.listdir(real_cons_video_dir)
real_cons_vid_list.sort()
real_cons_vid_dict = {x:real_cons_video_dir + x +'/' for x in real_cons_vid_list}

synth_f_t_video_dir = "E:/CAPSTONE/CAPSTONE/synth_pre/Normalised/Synthesized/Normalized/" # f(t) real frame dir
synth_f_t_vid_list = os.listdir(synth_f_t_video_dir)
synth_f_t_vid_list.sort()
synth_f_t_vid_dict = {x:synth_f_t_video_dir + x +'/' for x in synth_f_t_vid_list}

synth_cons_video_dir = "E:/CAPSTONE/CAPSTONE/synth_pre/Normalised/Synthesized/Consecutive/" # consecutive real frame dir
synth_cons_vid_list = os.listdir(synth_cons_video_dir)
synth_cons_vid_list.sort()
synth_cons_vid_dict = {x:synth_cons_video_dir + x +'/' for x in synth_cons_vid_list}

f_t_video_dict = dict()
f_t_video_dict.update(real_f_t_vid_dict)
f_t_video_dict.update(synth_f_t_vid_dict)

cons_video_dict = dict()
cons_video_dict.update(real_cons_vid_dict)
cons_video_dict.update(synth_cons_vid_dict)

# fake_video_dir = "D:/Datasets/CelebDF v2/4_Magnify_Video/Synthesized_Frame/"
# fake_vid_list = os.listdir(fake_video_dir)
# fake_vid_list.sort()
# fake_vid_dict = {x:fake_video_dir + x +'/' for x in fake_vid_list}

# video_dict = real_vid_dict
# video_dict.update(fake_vid_dict)
# data_name_list = real_vid_list + fake_vid_list

# fake_mit_dir = "D:/Datasets/CelebDF v2/5_MMST_Map/Synthesized/"
# fake_mit_list = os.listdir(fake_mit_dir)
# fake_mit_list.sort()
# fake_mit_dict = {x:fake_mit_dir + x for x in fake_mit_list}

# real_mit_dir = "D:/Datasets/CelebDF v2/5_MMST_Map/Real/"
# real_mit_list = os.listdir(real_mit_dir)
# real_mit_list.sort()
# real_mit_dict = {x:real_mit_dir + x for x in real_mit_list}

# mit_dict = fake_mit_dict
# mit_dict.update(real_mit_dict)

save_path = "D:/Datasets/CelebDF v2/6_Train_Data_Norm/"
if not os.path.exists(save_path):
    os.mkdir(save_path)

# data_Meso_set = []
# data_mit_set = []
# data_y_set = []
# data_name_set = []

data_f_t_set = []
data_cons_set = []
data_set = []
data_y_set = []
data_name_set = []
data_name_list = list(cons_video_dict.keys())
skipped=[]
uneven=[]
for vid_name in data_name_list:
    print("video {}:".format(vid_name), end='', flush=True)
    # if vid_name+'.npy' in mit_dict:
    cons_vid_path = cons_video_dict[vid_name]
    f_t_vid_path = f_t_video_dict[vid_name]
    cons_img_list = os.listdir(cons_vid_path)
    f_t_img_list = os.listdir(f_t_vid_path)
    # if cons_img_list==[] or f_t_img_list==[]:
    #     skipped.append((cons_vid_path,f_t_vid_path))
    #     continue
    # if(len(cons_img_list)!=len(f_t_img_list) and len(cons_img_list)<300):
    #     uneven.append((vid_name,len(cons_img_list)!=len(f_t_img_list),len(cons_img_list)<300,len(f_t_img_list)<300))
    cons_img_list.sort()
    f_t_img_list.sort()
    # print(cons_img_list)
    # print(f_t_img_list)
    #data_Meso = np.ones(300) * 0.5
    data_f_t=[]
    data_cons=[]
    try:
        for img_f_t,img_cons in zip(f_t_img_list,cons_img_list):
            if int(img_f_t[:-4]) >= 300 or int(img_cons[:-4]) >= 300:
                continue
            img_f_t_path = f_t_vid_path + img_f_t
            img_cons_path = cons_vid_path + img_cons
            img_f_t = cv2.imread(img_f_t_path)
            img_cons = cv2.imread(img_cons_path)
            #pred = classifier.predict(np.array([img]))
            #data_Meso[int(img_name[:-4])] = pred
            data_cons.append(img_cons)
            data_f_t.append(img_f_t)
    except Exception:
        print('Exception occured!')
        continue
    # print(data_cons)
    # print(data_f_t)
    data_cons_set.append(data_cons)
    data_f_t_set.append(data_f_t)
    data_set.append((data_cons,data_f_t))
    # data_Meso_set.append(data_Meso)
    # data_mit = np.load(mit_dict[vid_name+'.npy'])
    # #data_mit = np.swapaxes(data_mit,0,1)
    # data_mit_set.append(data_mit)
    if vid_name.count('_')!=1:
        data_y_set.append(1)
        print(1)
    else:
        data_y_set.append(0)
        print(0)
    data_name_set.append(vid_name)
    print("  ", np.shape(data_cons_set))
    print("  ", np.shape(data_f_t_set))
    print("  ", np.shape(data_set))
    #print("  ", np.shape(data_f_t_set))
    print("  ", np.shape(data_y_set))
    print("  ", np.shape(data_name_set))

# [0][1] -> f_t
# [0][0] -> cons
    
print("SAVING .... ")
np.save(save_path+"f_t.npy", data_f_t_set)
np.save(save_path+"cons.npy", data_cons_set)
np.save(save_path+"data.npy", data_set)
np.save(save_path+"y.npy", data_y_set)
np.save(save_path+"name.npy", data_name_set)

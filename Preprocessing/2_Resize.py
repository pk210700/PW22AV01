import os
import cv2
import numpy as np

'''
    Resize the aligned face so they can be formed into a video, which will
    be used into motion magnification.
'''
def resize_frame(align_path, resize_path):
    if not os.path.exists(resize_path):
        os.makedirs(align_path)

    vidlist = os.listdir(align_path)
    vidlist.sort()
    for vidname in vidlist:
        print("{} - {} ... ".format(align_path, vidname), end='', flush=True)
        vidpath = align_path + vidname + '/'

        save_vid_path = resize_path + vidname + '/'
        os.mkdir(save_vid_path)

        imglist = os.listdir(vidpath)
        imglist.sort()
        f_h, f_w = None, None
        for imgname in imglist:
            if imgname.endswith('.npy'):
                continue
            if imgname.endswith('_face.jpg'):
                continue

            imgpath = vidpath + imgname

            img = cv2.imread(imgpath)
            if f_h is None:
                f_h, f_w = img.shape[:2]
                f_h = 300
                f_w = 300

            img = cv2.resize(img, (f_w, f_h))

            save_img_path = save_vid_path + imgname
            cv2.imwrite(save_img_path, img)

        idx = None
        imglist = os.listdir(save_vid_path)
        imglist.sort()
        imglist.append("0300.jpg")
        for imgname in imglist:
            if int(imgname[:4]) >= 300:
                break
            if idx is None and int(imgname[:4])!=0:
                for i in range(0, int(imgname[:4])):
                    ori_img = save_vid_path + imgname
                    new_img = save_vid_path + ("%04d.jpg"%i)
                    o=ori_img.replace('/','\\')
                    n=new_img.replace('/','\\')
                    os.system("copy \"" + o + "\" \"" + n+"\"")
            if idx is not None and idx < int(imgname[:4]):
                for i in range(idx, int(imgname[:4])):
                    ori_img = save_vid_path + ("%04d.jpg"%(idx-1))
                    new_img = save_vid_path + ("%04d.jpg"%i)
                    o=ori_img.replace('/','\\')
                    n=new_img.replace('/','\\')
                    os.system("copy \"" + o + "\" \"" + n+"\"")
            idx = int(imgname[:4]) + 1
        print("Done")


if __name__=="__main__":

    video_dir_name = "Real/"
    #video_dir_name = "Synthesized/"

    data_root_dir = "D:/Datasets/CelebDF v2/1_Align/"
    datadir = data_root_dir + video_dir_name

    resize_root_dir = "D:/Datasets/CelebDF v2/2_Resize/"
    resizedir = resize_root_dir + video_dir_name
    if not os.path.exists(resizedir):
        os.makedirs(resizedir)

    resize_frame(datadir, resizedir)

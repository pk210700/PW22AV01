import numpy as np
import dlib
import cv2
import os
import time


def calculate_ROI(img, block_h, block_w):
    height, width, channels = np.shape(img)
    blk_h, blk_w = int(height/block_h), int(width/block_w)
    roi_seg = np.zeros((block_h*block_w, channels))
    for bh in range(block_h):
        for bw in range(block_w):
            roi_seg[bh*block_w+bw] = [np.average(img[blk_h*bh:blk_h*(bh+1),blk_w*bw:blk_w*(bw+1),i]) for i in range(3)]
    return roi_seg

def get_frame_seg(img, ROI_h, ROI_w):
    ROI_seg = calculate_ROI(img, ROI_h, ROI_w)
    return ROI_seg

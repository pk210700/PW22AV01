# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 07:06:45 2021

@author: PK
"""

'''import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

layer = MyDenseLayer(10)

_ = layer(tf.zeros([10, 5])) # Calling the layer `.builds` it.

print([var.name for var in layer.trainable_variables])'''

import os
import cv2
#path="D:/Datasets/CelebDF v2/4_Magnify_Video/Real/"
path="D:/Datasets/CelebDF v2/4_Magnify_Video/Synthesized/"
d=os.listdir(path)
print(d)
'''for i in d:
    s=i
    #s=s[:s.index('_',(s.index('_',s.index('_')+1)+1))]+".mp4" # Synthesized
    s=s[:s.index('_',s.index('_')+1)]+".mp4" # Real
    #os.system("cd "+path)
    os.rename(path+i,path+s)
    #print(s)'''

#path1="D:/Datasets/CelebDF v2/4_Magnify_Video/Real_Frame/"
path1="D:/Datasets/CelebDF v2/4_Magnify_Video/Synthesized_Frame/"
for i in d:
    cam = cv2.VideoCapture(path+i)
    s=i[:i.index('.')+1]
    os.makedirs(path1+s)
    currentframe = 0
  
    while(True):
        ret,frame = cam.read()
  
        if ret:
            name = path1+s+'/%3d'%currentframe + '.jpg'
            print ('Creating...' + name)
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
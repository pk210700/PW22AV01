# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:18:49 2021

@author: bhatn
"""



import numpy as np
import os
import cv2




image_path = 'C:/Users/bhatn/OneDrive/Desktop/synth_pre/Motion_Mag/Motion_Magnified/Real/'
image_name_video = []
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for f in [f[:f.index('.')] for f in os.listdir(image_path)]:
    print(f)
    '''if not(".mp4" in f): #OULU
        continue'''

    file= image_path+'/'+ f + '.mp4'
    print(file)
    cap = cv2.VideoCapture(file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    nFrames = cap.get(7)
    max_frames = int(nFrames)
    partial_path = 'C:/Users/bhatn/OneDrive/Desktop/synth_pre/Normalised/Synthesized'+'/'+f
    if not(os.path.exists(partial_path)) :
        os.mkdir(partial_path);
    partial_path2 ='C:/Users/bhatn/OneDrive/Desktop/synth_pre/Normalised/Real'+'/'+f
    if not(os.path.exists(partial_path2)) :
        os.mkdir(partial_path2);

    L = 300
    C_R=np.empty((L,L,max_frames))
    C_G=np.empty((L,L,max_frames))
    C_B=np.empty((L,L,max_frames))

    D_R=np.empty((L,L,max_frames))
    D_G=np.empty((L,L,max_frames))
    D_B=np.empty((L,L,max_frames))

    D_R2=np.empty((L,L,max_frames))
    D_G2=np.empty((L,L,max_frames))
    D_B2=np.empty((L,L,max_frames))

    mean_R = np.empty((L,L))
    mean_G = np.empty((L,L))
    mean_B = np.empty((L,L))

    deviation_R = np.empty((L,L))
    deviation_G = np.empty((L,L))
    deviation_B = np.empty((L,L))

    image = np.empty((L,L,3))

    mean_CR = np.empty((L,L))
    mean_CG = np.empty((L,L))
    mean_CB = np.empty((L,L))

    deviation_CR = np.empty((L,L))
    deviation_CG = np.empty((L,L))
    deviation_CB = np.empty((L,L))
    ka            = 1


    while(cap.isOpened() and ka< max_frames):
        ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        '''faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        #rectangle around the faces
        for (x, y, w, h) in faces:
            # face = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y + h, x:x + w]


        face = cv2.resize(face, (L,L), interpolation = cv2.INTER_AREA)'''
        face=frame
        # cv2.imshow('img', face)
        # cv2.waitKey()
        C_R[:,:,ka] = face[:,:,0]
        C_G[:,:,ka] = face[:,:,1]
        C_B[:,:,ka] = face[:,:,2]


        if ka > 1:
            D_R[:,:,ka-1] = ( C_R[:,:,ka] - C_R[:,:,ka-1] ) / ( C_R[:,:,ka] + C_R[:,:,ka-1]);
            D_G[:,:,ka-1] = ( C_G[:,:,ka] - C_G[:,:,ka-1] ) / ( C_G[:,:,ka] + C_G[:,:,ka-1]);
            D_B[:,:,ka-1] = ( C_B[:,:,ka] - C_B[:,:,ka-1] ) / ( C_B[:,:,ka] + C_B[:,:,ka-1]);
        ka = ka+1



    for i in range(0,L):
        for j in range(0,L):
            mean_R[i,j]=np.mean(D_R[i,j,:])
            mean_G[i,j]=np.mean(D_G[i,j,:])
            mean_B[i,j]=np.mean(D_B[i,j,:])
            deviation_R[i,j]=np.std(D_R[i,j,:])
            deviation_G[i,j]=np.std(D_G[i,j,:])
            deviation_B[i,j]=np.std(D_B[i,j,:])

    for i in range(0,L):
        for j in range(0,L):
            mean_CR[i,j]=np.mean(C_R[i,j,:])
            mean_CG[i,j]=np.mean(C_G[i,j,:])
            mean_CB[i,j]=np.mean(C_B[i,j,:])
            deviation_CR[i,j]=np.std(C_R[i,j,:])
            deviation_CG[i,j]=np.std(C_G[i,j,:])
            deviation_CB[i,j]=np.std(C_B[i,j,:])

    for k in range(0,max_frames):
        D_R2[:,:,k] = (C_R[:,:,k] - mean_CR)/(deviation_CR+000.1)
        D_G2[:,:,k] = (C_G[:,:,k] - mean_CG)/(deviation_CG+000.1)
        D_B2[:,:,k] = (C_B[:,:,k] - mean_CB)/(deviation_CB+000.1)



    for k in range(0,max_frames):

        image[:,:,0] = D_R2[:,:,k]
        image[:,:,1] = D_G2[:,:,k]
        image[:,:,2] = D_B2[:,:,k]

        image= np.uint8(image)

        nombre_salvar= os.path.join(partial_path2,str(k)+'.png')
        cv2.imwrite(nombre_salvar, image)


    for k in range(0,max_frames):

        D_R[:,:,k] = (D_R[:,:,k] - mean_R)/(deviation_R+000.1)
        D_G[:,:,k] = (D_G[:,:,k] - mean_G)/(deviation_G+000.1)
        D_B[:,:,k] = (D_B[:,:,k] - mean_B)/(deviation_B+000.1)

    for k in range(0,max_frames):

        image[:,:,0] = D_R[:,:,k]
        image[:,:,1] = D_G[:,:,k]
        image[:,:,2] = D_B[:,:,k]

        image= np.uint8(image)

        nombre_salvar= os.path.join(partial_path,str(k)+'.png')
        cv2.imwrite(nombre_salvar, image)


    cap.release()
    cv2.destroyAllWindows()
print("Exiting...")

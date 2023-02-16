
import cv2
import os
import tifffile
import numpy as np


X = []
Y_MG = []
Y_EA = []
img_number = -1
path = 'data/'
address_list = []
for root, dirs, files in os.walk(path, topdown=False):
    if "PAT 930" in root:
        # print(root)  # to show progress and debug
        if any(["final" in f for f in files]):  # This is to make sure that all data is segmented by EA and MG.
            # As a result of running the code for the first time, it was stated that 066, 068 band 087 don't have maks
            for name in files:
                imgAddress = os.path.join(root, name)
                if "initial combined" in name:
                    img = tifffile.imread(imgAddress)
                    X.append(img)
                elif not any(["initial combined" in f for f in files]) and "initial" in name:
                    img = tifffile.imread(imgAddress)
                    X.append(img)
                # Extracting masks

                if "MG" in name:
                    img = tifffile.imread(imgAddress)
                    Y_MG.append(img)
                elif "EA" in name:
                    img = tifffile.imread(imgAddress)
                    Y_EA.append(img)
        address_list.append(root)
        # print('------------\n')
# Now list X contains all the training pictures with tiff format, next we will save 3D tiff files as 2D images to training set
# Y_MG and Y_EA represent the masks created by two summer students

#------------------------------ Training Set -------------------------------
img_number = 1
X_train=[]
y_train=[]
for sample in range(len(X)-9):   #the last 9 sets are for testing.
    input_tiff_img = X[sample]
    for i in range(input_tiff_img.shape[0]):
        resized_X = cv2.resize(input_tiff_img[i], (222, 200))
        resized_Y_MG = cv2.resize(Y_MG[sample][i], (222, 200))
        resized_Y_EA = cv2.resize(Y_EA[sample][i], (222, 200))
        mask=np.logical_or(resized_Y_MG, resized_Y_EA).astype(np.uint8)  # We will train based on greatest common area of two masks
        X_train.append(resized_X/255)
        y_train.append(mask)
        # cv2.imwrite("data/preprocessedImages/TrainingSet/X_train/X_train"+str(img_number)+".jpg", resized_X)
        # cv2.imwrite("data/preprocessedImages/TrainingSet/y_train/y_train"+str(img_number)+".jpg", mask*255)  # *255 is to change ones to white pixels
        img_number += 1
X_train=np.array(X_train)
y_train=np.array(y_train)
#------------------------------ Test Set -------------------------------
img_number = 1
X_test=[]
y_test=[]
for sample in range(len(X)-9, len(X)):
    input_tiff_img = X[sample]
    for i in range(input_tiff_img.shape[0]):
        resized_X = cv2.resize(input_tiff_img[i], (222, 200))
        resized_Y_MG = cv2.resize(Y_MG[sample][i], (222, 200))
        resized_Y_EA = cv2.resize(Y_EA[sample][i], (222, 200))
        mask=np.logical_or(resized_Y_MG, resized_Y_EA).astype(np.uint8)
        X_test.append(resized_X/255)
        y_test.append(mask)
        # cv2.imwrite("data/preprocessedImages/TestSet/X_test/X_test"+str(img_number)+".jpg", resized_X)
        # cv2.imwrite("data/preprocessedImages/TestSet/y_test/y_test"+str(img_number)+".jpg", mask*255)
        img_number += 1
X_test=np.array(X_test)
y_test=np.array(y_test)

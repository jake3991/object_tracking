import numpy as np
# import the necessary packages
import argparse
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math



def read_image_filenames(directory):
    names=sorted(glob.glob(directory+"*.jpg"))
    return names

def read(name,rot):
    
    #read the image
    image = cv2.imread(name,0)

    #convert to size
    image = cv2.resize(image,(1280,960))

    #rotate
    rows,cols = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)
    image = cv2.warpAffine(image,M,(cols,rows))

    return image


def segment_image_2(channel,thresh_min,thresh_max):

    binary = np.zeros_like(channel)
    binary[(channel >= thresh_min) & (channel <= thresh_max)] = 255

    seg = cv2.bitwise_not(binary)
    
    return seg

filenames=read_image_filenames("/home/avengers/Desktop/centroid_locator/ros bags/bag_1/")

for name in filenames:
    img = read(name,0)

    seg = segment_image_2(img,80,115)

    cv2.imshow("image",seg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()